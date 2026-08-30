// RouteTaskExecutorTests.swift — 0.B-1 rework (v2.3, FDD-ui-api-parity.md
// §3.1.3/§4.1, comfybox#300).
//
// v2.2's 0.B-1 moved the RENDER off the cooperative pool with
// `RenderTaskExecutor` and crashed LTX in production (native MLX mutex
// EINVAL from cross-thread eval migration under `.concurrent` GCD queues).
// v2.3 deletes that design and inverts it: the async-internals ROUTE
// handlers (`/v1/enhance`, `/v1/civitai/search`, `/v1/civitai/harvest`) move
// off the pool instead, via `RouteTaskExecutor`; the render is untouched.
//
// This suite covers exactly the delta the rework needs to prove:
//   1. `RouteTaskExecutor` carries an SE-0417 preference into a nonisolated
//      async callee (reusing ExecutorPreferenceSpikeTests' idiom, but against
//      the real production type rather than the spike's throwaway one).
//   2. `RouteTaskExecutorFlag` — same env var name as the deleted render
//      flag (`COMFYBOX_RENDER_TASK_EXECUTOR`), repurposed — correctly gates
//      on/off, which is what makes `respondOnRouteExecutor`'s "0" path a
//      plain passthrough (that private method lives in WarmServer.swift and
//      isn't independently unit-testable, but its entire behavior reduces to
//      this flag plus the SE-0417 mechanism proven in (1), both covered
//      here).
//   3. A source-scan assertion that the render call chain
//      (`WarmServerCoordinator`) contains zero `executorPreference` /
//      `RenderTaskExecutor` / `spawnRenderTask` references, so a future
//      change can't silently reintroduce 0.B-1's original (crash-inducing)
//      render-side attachment without this test catching it — CI cannot
//      exercise MLX directly (§4.1 "Tests"), so this static check is the
//      only thing standing between a regression and a repeat of the LTX
//      crash.
//
// The full integration AC (routes answer <2s under a live render, with a
// zero-crash LTX+Krea2+audio soak) is explicitly NOT a CI-testable claim
// (FDD §4.1) and is covered by the separate soak cycle, not here.

import Dispatch
import Foundation
import XCTest

@testable import ZImage

final class RouteTaskExecutorTests: XCTestCase {

  // MARK: - Test isolation for the env-var flag

  private var originalFlagValue: String?

  override func setUp() {
    super.setUp()
    originalFlagValue = ProcessInfo.processInfo.environment["COMFYBOX_RENDER_TASK_EXECUTOR"]
  }

  override func tearDown() {
    if let original = originalFlagValue {
      setenv("COMFYBOX_RENDER_TASK_EXECUTOR", original, 1)
    } else {
      unsetenv("COMFYBOX_RENDER_TASK_EXECUTOR")
    }
    super.tearDown()
  }

  // MARK: - 1. Attachment: preference carries into the route handler's callee chain

  /// Mirrors `ExecutorPreferenceSpikeTests.test_executorPreferenceCarriesIntoNonisolatedAsyncCallee`,
  /// but against the real `RouteTaskExecutor` (0.B-0's spike proved the
  /// generic SE-0417 mechanism; this proves the production type wired up for
  /// routes actually exhibits it too).
  private func currentQueueLabel() -> String {
    String(cString: __dispatch_queue_get_label(nil))
  }

  /// Same shape as the route handlers this executor is for
  /// (`enhancePromptResponse`, `civitaiSearchRoute`, `civitaiHarvestRoute`):
  /// a `nonisolated async` function with an internal suspension point.
  nonisolated func nonisolatedAsyncCallee(tag: String) async -> String {
    await Task.yield()
    return "\(tag)::\(currentQueueLabel())"
  }

  func test_routeTaskExecutorCarriesPreferenceIntoNonisolatedAsyncCallee() async throws {
    guard #available(macOS 15.0, *) else {
      throw XCTSkip("SE-0417 task executor preference requires macOS 15+; this host predates it.")
    }

    let executor = RouteTaskExecutor(label: "test.route-task-executor.attachment")

    let result = await withTaskExecutorPreference(executor) {
      await self.nonisolatedAsyncCallee(tag: "route")
    }

    XCTAssertTrue(
      result.hasSuffix("test.route-task-executor.attachment"),
      "RouteTaskExecutor preference did not carry into the nonisolated async callee — got \(result)")
  }

  // MARK: - 2. Flag: gates the route-executor preference on/off

  func test_routeTaskExecutorFlagDefaultsOnWhenUnset() {
    unsetenv("COMFYBOX_RENDER_TASK_EXECUTOR")
    XCTAssertTrue(RouteTaskExecutorFlag.isEnabled, "unset COMFYBOX_RENDER_TASK_EXECUTOR should default the route executor ON")
  }

  /// "Flag-off passthrough": `COMFYBOX_RENDER_TASK_EXECUTOR=0` is the
  /// rollback knob (FDD §4.1 Rollback) that reverts the three route arms to
  /// running inline, exactly the pre-0.B-1 (and pre-0.B-1-rework) behavior.
  /// `respondOnRouteExecutor` (private, WarmServer.swift) reduces to this
  /// flag check plus a no-op passthrough when it reads false — this is the
  /// unit-testable half of that guarantee.
  func test_routeTaskExecutorFlagOffWithExplicitZero() {
    setenv("COMFYBOX_RENDER_TASK_EXECUTOR", "0", 1)
    XCTAssertFalse(RouteTaskExecutorFlag.isEnabled, "COMFYBOX_RENDER_TASK_EXECUTOR=0 must disable the route executor preference")
  }

  // MARK: - 3. Source scan: the render call chain stays executor-free

  /// Static guardrail for the exact failure this rework fixes: 0.B-1's v2.2
  /// design attached `executorPreference` to the RENDER task tree and
  /// crashed LTX in production. The render path has no flag of its own
  /// (FDD §4.1 Rollback — "the render path has no flag because it is not
  /// modified"), so the only thing preventing a silent reintroduction is
  /// this never lands in `WarmServerCoordinator` again. CI cannot exercise
  /// MLX (§4.1 "Tests"), so this scan — not a runtime assertion — is the
  /// regression gate.
  func test_renderCallChainContainsNoExecutorPreference() throws {
    let testFileURL = URL(fileURLWithPath: #filePath)
    let repoRoot = testFileURL
      .deletingLastPathComponent()  // RouteTaskExecutorTests.swift -> Server/
      .deletingLastPathComponent()  // Server/ -> ZImageTests/
      .deletingLastPathComponent()  // ZImageTests/ -> Tests/
      .deletingLastPathComponent()  // Tests/ -> repo root
    let warmServerURL = repoRoot
      .appendingPathComponent("Sources/ZImage/Server/WarmServer.swift")

    let source = try String(contentsOf: warmServerURL, encoding: .utf8)

    guard let coordinatorStart = source.range(of: "private actor WarmServerCoordinator {") else {
      XCTFail("could not locate `private actor WarmServerCoordinator {` in WarmServer.swift — has it moved or been renamed?")
      return
    }
    guard let coordinatorEnd = source.range(
      of: "struct HTTPRequest {",
      range: coordinatorStart.upperBound..<source.endIndex
    ) else {
      XCTFail("could not locate the `struct HTTPRequest {` anchor after WarmServerCoordinator — update this test's boundary if the file was reorganized")
      return
    }

    let renderCallChain = source[coordinatorStart.upperBound..<coordinatorEnd.lowerBound]

    XCTAssertFalse(
      renderCallChain.contains("executorPreference"),
      "WarmServerCoordinator (the render call chain) must not attach a task executor preference — this is exactly the v2.2 design that crashed LTX in production (native MLX mutex EINVAL from cross-thread eval migration). Route handlers get the executor preference in WarmServer.respond(to:) instead.")
    XCTAssertFalse(
      renderCallChain.contains("RenderTaskExecutor"),
      "RenderTaskExecutor was deleted (0.B-1 v2.3 rework) — it must not reappear in the render call chain.")
    XCTAssertFalse(
      renderCallChain.contains("spawnRenderTask"),
      "spawnRenderTask (the v2.2 render-executor spawn helper) was deleted — the three render spawn sites must be plain `Task {}`.")
  }

  /// The FDD's MLX-free precondition (§3.1.3 "Verified safe to move"),
  /// encoded: the route handlers this executor lifts (`/v1/enhance`,
  /// `/v1/civitai/search`, `/v1/civitai/harvest`) are only safe on a
  /// `.concurrent` executor because their callee graph never touches MLX —
  /// thread migration is harmless to network/disk/actor I/O but corrupts
  /// MLX's thread-affine native eval state (the v2.2 LTX crash). If any of
  /// these files ever gains `import MLX`, that safety argument is void and
  /// the route in question must come OFF the executor (or the MLX work must
  /// move out of the handler's callee graph).
  func test_routeExecutorCalleeGraphIsMLXFree() throws {
    let testFileURL = URL(fileURLWithPath: #filePath)
    let repoRoot = testFileURL
      .deletingLastPathComponent()  // RouteTaskExecutorTests.swift -> Server/
      .deletingLastPathComponent()  // Server/ -> ZImageTests/
      .deletingLastPathComponent()  // ZImageTests/ -> Tests/
      .deletingLastPathComponent()  // Tests/ -> repo root

    // The handlers' callee graph, per the FDD's trace @ 30e2757:
    // - /v1/enhance -> PromptOptimizer.optimize (HTTP to ollama/LM Studio)
    // - /v1/civitai/search -> CivitAIClient.searchModels (HTTPS)
    // - /v1/civitai/harvest -> CivitAIHarvestRunner.run (paged HTTP + repo
    //   upsert; the runner and PromptRepositoryStore both live on disk/network)
    let calleeGraphFiles = [
      "Sources/ZImage/Telegram/PromptOptimizer.swift",
      "Sources/ZImage/CivitAI/CivitAIClient.swift",
      "Sources/ZImage/Server/CivitAIConduitRoutes.swift",
      "Sources/ZImage/Server/PromptRepositoryStore.swift",
        "Sources/ZImage/Server/CharacterStore.swift",
        "Sources/ZImage/Server/RenderTraceStore.swift",
        "Sources/ZImage/Server/ContentModeStore.swift",
        "Sources/ZImage/CivitAI/CivitAISecrets.swift",
        "Sources/ZImage/Server/ComfyBoxServerConfig.swift",
    ]

    for relativePath in calleeGraphFiles {
      let url = repoRoot.appendingPathComponent(relativePath)
      let source = try String(contentsOf: url, encoding: .utf8)
      for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        XCTAssertFalse(
          trimmed == "import MLX" || trimmed.hasPrefix("import MLX ")
            || trimmed.hasPrefix("import MLXNN") || trimmed.hasPrefix("import MLXRandom")
            || trimmed.hasPrefix("import MLXFast"),
          "\(relativePath) imports MLX — the route-executor safety argument (§3.1.3: victims are MLX-free, so thread migration is harmless) no longer holds; take the affected route off RouteTaskExecutor or move the MLX work out of its callee graph.")
      }
    }
  }
}
