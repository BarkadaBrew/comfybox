// RenderTaskExecutor.swift — SE-0417 task executor preference for the render
// task tree (0.B-1, FDD-ui-api-parity.md §3.1.3, comfybox#300).
//
// 0.A (WarmServerCoordinator's dedicated DispatchSerialQueue executor,
// shipped fd08ba9) pins the coordinator ACTOR's isolated code off the shared
// cooperative pool. It was necessary but not sufficient: per SE-0338, a
// nonisolated async function (like ZImagePipeline.generateFromRequest) runs
// on the global concurrent executor, not the caller's — so the moment the
// coordinator awaits into the pipeline, execution hops back onto the
// cooperative pool and blocks there on MLX's internal mutex for the whole
// render, starving every unrelated `await` in the process (measured:
// 2964/2972 samples in __psynch_cvwait during a render, pre-0.A; post-0.A,
// the same exhaustion recurred on the user-initiated-qos cooperative tier).
//
// RenderTaskExecutor gives the render task tree its own home so none of that
// nonisolated async code ever touches the cooperative pool. Width is 2 (not
// 1) so an in-flight image render and a parked/handing-off video render can
// coexist under #1479's preemption handoff without contending for the same
// single worker.
//
// 0.B-0 spike findings this design rests on (ExecutorPreferenceSpikeTests.swift):
//   - `Task(executorPreference:)` compiles clean under swift-tools 5.9 with
//     `#available(macOS 15.0, *)` guards at call sites, PROVIDED any type
//     conforming to `TaskExecutor` (this one) is itself annotated
//     `@available(macOS 15.0, *)` — a call-site guard alone is not enough for
//     the conformance declaration itself.
//   - The preference DOES carry into nonisolated async callees (confirmed).
//   - The preference does NOT survive an inner unstructured `Task {}` boundary
//     without re-attachment — it falls back to the global cooperative pool.
//     This is why every spawn site in the render tree re-attaches explicitly
//     rather than relying on inheritance.
//   - Plain `DispatchQueue` conforms to `TaskExecutor` directly in this SDK
//     (Swift 6.4 / macOS 27) — the custom type below exists for width-2
//     concurrency control (a single DispatchQueue is either serial [width 1]
//     or unboundedly concurrent), not for the protocol conformance itself.

import Dispatch
import Foundation

/// A `TaskExecutor` that runs at most `width` job segments concurrently, on a
/// dedicated dispatch queue away from the Swift cooperative pool.
///
/// Available macOS 15+ (SE-0417); the package's platform floor is macOS 14
/// (`Package.swift:6`), so every construction and use site is guarded with
/// `#available(macOS 15.0, *)` and falls back to today's unstructured-`Task{}`
/// behavior on older hosts.
@available(macOS 15.0, *)
final class RenderTaskExecutor: TaskExecutor, @unchecked Sendable {
  /// An image render and a parked/handing-off video render (#1479 preemption)
  /// can coexist; anything beyond that queues behind the semaphore rather
  /// than spawning unbounded concurrent GPU work.
  static let width = 2

  private let queue: DispatchQueue
  private let semaphore: DispatchSemaphore

  init(label: String = "com.comfybox.render-task-executor") {
    self.queue = DispatchQueue(label: label, attributes: .concurrent)
    self.semaphore = DispatchSemaphore(value: Self.width)
  }

  func enqueue(_ job: consuming ExecutorJob) {
    let unownedJob = UnownedJob(job)
    let unownedExecutor = self.asUnownedTaskExecutor()
    let semaphore = self.semaphore
    queue.async {
      semaphore.wait()
      defer { semaphore.signal() }
      unownedJob.runSynchronously(on: unownedExecutor)
    }
  }
}

/// Process-wide toggle for 0.B-1, independent of 0.A and the (future) 0.B-2
/// control-plane flag. Unset or "1" = on (today's target state); "0" =
/// exactly pre-0.B-1 behavior — plain unstructured `Task {}` at the three
/// render spawn sites, cooperative pool included. Read once at first use, not
/// cached across the process's lifetime is intentional here: this is a
/// startup-time rollback knob (§4.1 Rollback), not something toggled live.
enum RenderTaskExecutorFlag {
  static var isEnabled: Bool {
    ProcessInfo.processInfo.environment["COMFYBOX_RENDER_TASK_EXECUTOR"] != "0"
  }
}
