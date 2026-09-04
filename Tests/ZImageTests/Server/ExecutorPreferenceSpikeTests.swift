import XCTest
import Dispatch

/// 0.B-0 gating spike (FDD-ui-api-parity.md §4.1, comfybox#300).
///
/// Answers the four exit criteria before 0.B-1 is built on top of them:
///   (i)   does `Task(executorPreference:)` compile under swift-tools 5.9 with an
///         `#available(macOS 15, *)` guard?
///   (ii)  does the preference demonstrably carry into a *nonisolated async*
///         callee (the shape of the render call chain)?
///   (iii) does it survive an INNER unstructured `Task {}` boundary, or does the
///         inner task fall back to the global cooperative pool?
///   (iv)  does `DispatchQueue` conform to `TaskExecutor` in this SDK, or is a
///         small custom conformance required?
///
/// This is throwaway scaffolding per the FDD (0.B-0 "hours, first, throwaway
/// code allowed") kept as a permanent regression test because the answers are
/// load-bearing for 0.B-1's design and cheap to keep green.
final class ExecutorPreferenceSpikeTests: XCTestCase {

  /// A custom `TaskExecutor` backed by a dedicated serial `DispatchQueue`,
  /// used here to prove re-attachment (iii) with a distinguishable label.
  /// NOTE: (iv) below found that plain `DispatchQueue` conforms to
  /// `TaskExecutor` directly in this SDK, so this custom type is not required
  /// purely for conformance — it exists in this spike so each test can hand
  /// out a uniquely-labeled executor and assert on the resulting queue label.
  /// 0.B-1's actual `RenderTaskExecutor` still needs a custom type, but for
  /// width-2 concurrency control, not for `TaskExecutor` conformance.
  @available(macOS 15.0, *)
  final class SpikeTaskExecutor: TaskExecutor, @unchecked Sendable {
    let queue: DispatchQueue
    private let labelValue: String

    init(label: String) {
      self.labelValue = label
      self.queue = DispatchQueue(label: label)
    }

    func enqueue(_ job: consuming ExecutorJob) {
      let unowned = UnownedJob(job)
      let executor = self.asUnownedTaskExecutor()
      queue.async {
        unowned.runSynchronously(on: executor)
      }
    }

    var label: String { labelValue }
  }

  /// Reads the label of whichever `DispatchQueue` the CURRENT thread is
  /// executing on. Empty string if none (e.g. a raw cooperative-pool thread
  /// with no dispatch queue label attached).
  private func currentQueueLabel() -> String {
    String(cString: __dispatch_queue_get_label(nil))
  }

  /// The shape under test: a `nonisolated async` function, exactly like
  /// `ZImagePipeline.generateFromRequest` in the render call chain — no
  /// actor isolation, so per SE-0338 it would normally run on the caller's
  /// executor (or the global concurrent executor for a plain unstructured
  /// `Task {}`).
  nonisolated func nonisolatedAsyncCallee(tag: String) async -> String {
    // A cooperative yield, same as the render path awaiting into MLX —
    // if the preference didn't carry, resumption would land back on the
    // global pool rather than the preferred queue.
    await Task.yield()
    return "\(tag)::\(currentQueueLabel())"
  }

  // MARK: - (i) + (ii) — compiles under the guard, and carries into the nonisolated callee

  func test_executorPreferenceCarriesIntoNonisolatedAsyncCallee() async throws {
    guard #available(macOS 15.0, *) else {
      throw XCTSkip("SE-0417 task executor preference requires macOS 15+; this host predates it.")
    }

    let executor = SpikeTaskExecutor(label: "spike.render.outer")

    let result = await Task(executorPreference: executor) {
      await self.nonisolatedAsyncCallee(tag: "outer")
    }.value

    print("[SPIKE ii] outer nonisolated callee ran on: \(result)")
    XCTAssertTrue(
      result.hasSuffix(executor.label),
      "executorPreference did not carry into the nonisolated async callee — got \(result)")
  }

  // MARK: - (iii) — inner unstructured Task {} boundary

  func test_preferenceDoesNotSurviveInnerUnstructuredTaskWithoutReattachment() async throws {
    guard #available(macOS 15.0, *) else {
      throw XCTSkip("SE-0417 task executor preference requires macOS 15+; this host predates it.")
    }

    let executor = SpikeTaskExecutor(label: "spike.render.inner-noreattach")

    let result = await Task(executorPreference: executor) { () async -> String in
      // Mirrors WarmServer.swift's `renderTask = Task { await self.runGenerate(...) }`
      // pattern at :7030/:7037: an INNER unstructured Task spawned from inside
      // a task that already has an executor preference, with NO preference
      // re-attached at the inner spawn.
      let inner = Task {
        await self.nonisolatedAsyncCallee(tag: "inner-no-reattach")
      }
      return await inner.value
    }.value

    print("[SPIKE iii-a] inner Task WITHOUT re-attached preference ran on: \(result)")
    // Falsifiable claim: preference does NOT propagate across an unstructured
    // Task{} boundary. If this assertion ever starts failing (i.e. the suffix
    // DOES match), SE-0417 propagation semantics changed and 0.B-1's
    // "re-attach at all three sites" design should be revisited — the
    // re-attachment would become redundant, not wrong.
    XCTAssertFalse(
      result.hasSuffix(executor.label),
      "unexpected: preference survived an inner unstructured Task{} boundary without re-attachment (\(result)) — re-check whether 0.B-1 needs re-attachment at :7030/:7037 at all")
  }

  func test_preferenceSurvivesInnerUnstructuredTaskWithReattachment() async throws {
    guard #available(macOS 15.0, *) else {
      throw XCTSkip("SE-0417 task executor preference requires macOS 15+; this host predates it.")
    }

    let executor = SpikeTaskExecutor(label: "spike.render.inner-reattach")

    let result = await Task(executorPreference: executor) { () async -> String in
      // Same shape, but re-attach the SAME preference at the inner spawn —
      // this is exactly what 0.B-1 does at :7030 and :7037.
      let inner = Task(executorPreference: executor) {
        await self.nonisolatedAsyncCallee(tag: "inner-reattached")
      }
      return await inner.value
    }.value

    print("[SPIKE iii-b] inner Task WITH re-attached preference ran on: \(result)")
    XCTAssertTrue(
      result.hasSuffix(executor.label),
      "re-attaching executorPreference at the inner Task{} boundary did not carry — got \(result)")
  }

  // MARK: - (iv) — DispatchQueue's TaskExecutor conformance

  /// SURPRISE (corrected from the FDD's working assumption): plain
  /// `DispatchQueue` conforms to `TaskExecutor` DIRECTLY in this SDK (Swift
  /// 6.4 / macOS 27 Dispatch overlay) — no custom `enqueue(_:)` conformance
  /// is required to pass a queue as `executorPreference:`. Verified both by
  /// runtime cast here and by a standalone `let te: any TaskExecutor = q`
  /// static-type check compiling clean.
  ///
  /// This does NOT eliminate the need for a small custom `RenderTaskExecutor`
  /// wrapper in 0.B-1, though — it changes *why* one is needed. A single
  /// `DispatchQueue` is either serial (effective width 1) or `.concurrent`
  /// (unbounded width); neither gives the "exactly 2" concurrency §3.1.3
  /// wants for #1479 image/video coexistence. The custom wrapper's job is
  /// width control (e.g. a semaphore-gated concurrent queue, or two serial
  /// queues round-robined), not `TaskExecutor` conformance.
  func test_dispatchQueueConformsToTaskExecutorDirectly() throws {
    guard #available(macOS 15.0, *) else {
      throw XCTSkip("TaskExecutor requires macOS 15+; this host predates it.")
    }
    let queue = DispatchQueue(label: "spike.plain-dispatch-queue")
    XCTAssertTrue(
      queue is any TaskExecutor,
      "DispatchQueue no longer conforms to TaskExecutor directly — RenderTaskExecutor would need its own enqueue(_:) conformance again, revisit 0.B-1")
  }
}
