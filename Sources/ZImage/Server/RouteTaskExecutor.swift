// RouteTaskExecutor.swift — SE-0417 task executor preference for the
// async-internals ROUTE handlers (0.B-1, v2.3 rework, FDD-ui-api-parity.md
// §3.1.3/§4.1, comfybox#300).
//
// v2.2's 0.B-1 moved the render off the cooperative pool (RenderTaskExecutor,
// deleted here) and crashed LTX in production: a `.concurrent` GCD queue does
// not preserve OS-thread identity across a task's suspension points, so the
// render's MLX eval calls migrated across OS threads at every `await` —
// violating MLX's documented single-thread eval invariant
// (`Source/MLX/State.swift:22-23`, `Device.swift:154-155`) and surfacing as
// `pthread_mutex_lock` -> EINVAL ("mutex lock failed: Invalid argument") at
// `mlx-c/transforms.cpp:73`. Reverted via `COMFYBOX_RENDER_TASK_EXECUTOR=0`,
// 52 crash-restart cycles in ~1h.
//
// v2.3 inverts the design: move the *victims* off the pool instead of the
// render. 0.B-2 (`f134d64`, PR#321) already made the sync-servable control
// set answer in ~1ms during a render without touching the pool at all, so the
// only routes still starving are the async-internals set: `/v1/enhance`,
// `/v1/civitai/search`, `/v1/civitai/harvest`. Those handlers only ever
// `await` on network/disk/actor I/O (`PromptOptimizer.optimize`,
// `CivitAIClient.searchModels`, `CivitAIHarvestRunner.run`) — verified MLX-free
// at `@ 30e2757` (§3.1.3) — so thread migration is harmless to them, unlike
// the render. The render is NOT moved: it stays on the cooperative pool
// exactly as it is today (see the plain `Task {}` spawn sites reverted
// alongside this file), so MLX only ever sees the same-thread synchronous
// eval stretches it always has.
//
// Unlike `RenderTaskExecutor`, this executor does not need a width gate:
// these handlers are I/O-bound, not GPU-bound, so unbounded concurrency is
// fine and lets independent harvests/enhances proceed in parallel. A plain
// `.concurrent` `DispatchQueue` conforms to `TaskExecutor` directly in this
// SDK (0.B-0 spike, ExecutorPreferenceSpikeTests.swift, finding iv), so no
// custom `enqueue(_:)` conformance is required — this type exists only to
// give the route-executor preference a distinguishable, documented home.

import Dispatch
import Foundation

/// A `TaskExecutor` that runs route-handler job segments on a dedicated
/// concurrent dispatch queue, away from the Swift cooperative pool.
///
/// Available macOS 15+ (SE-0417); the package's platform floor is macOS 14
/// (`Package.swift:6`), so every construction and use site is guarded with
/// `#available(macOS 15.0, *)` and falls back to today's unstructured-`Task{}`
/// behavior (on the cooperative pool) on older hosts.
@available(macOS 15.0, *)
final class RouteTaskExecutor: TaskExecutor, @unchecked Sendable {
  private let queue: DispatchQueue

  init(label: String = "com.comfybox.route-task-executor") {
    self.queue = DispatchQueue(label: label, attributes: .concurrent)
  }

  func enqueue(_ job: consuming ExecutorJob) {
    let unownedJob = UnownedJob(job)
    let unownedExecutor = self.asUnownedTaskExecutor()
    queue.async {
      unownedJob.runSynchronously(on: unownedExecutor)
    }
  }
}

/// Process-wide toggle for 0.B-1 (v2.3 rework). Same env var name as the
/// original (now-deleted) render-executor flag — kept unchanged to avoid a
/// plist/ops change (FDD §4.1 "The flag, repurposed") — but its meaning is now
/// the ROUTE-executor preference for the async-internals handlers, not a
/// render-side attachment. Unset or "1" = on (routes run on
/// `RouteTaskExecutor`); "0" = today's all-on-the-pool behavior for those
/// three routes (the render was never moved and has no flag of its own — its
/// spawn sites in `WarmServerCoordinator` are plain `Task {}` unconditionally,
/// reverted alongside this file).
enum RouteTaskExecutorFlag {
  static var isEnabled: Bool {
    ProcessInfo.processInfo.environment["COMFYBOX_RENDER_TASK_EXECUTOR"] != "0"
  }
}
