import Foundation

// comfybox#308: `/health.render_count` and `last_render_duration_ms` stayed 0
// through an entire HQ two-pass soak even though the watchdog confirmed jobs
// completing and the queue advancing. Root cause: the six image-family
// `run*Generate` methods on `WarmServerCoordinator` each hand-increment
// `successfulRenderCount`/`failedRenderCount`/`lastRenderDurationMs` inline
// on their own success/catch paths, but the LOCAL VIDEO completion path (the
// `.localVideo` case in the coordinator's process loop) never did — it only
// resumes the caller's continuation. Every video render, HQ two-pass
// included, was invisible to `/health`.
//
// Extracted as a pure step (not a private-inline `+= 1`) so the increment
// logic is unit-testable without spinning up the actor/pipeline.

/// One render's terminal outcome, as far as the health counters care.
public enum RenderCompletionEvent: Equatable, Sendable {
  case succeeded(durationMs: Int)
  case failed
}

/// The subset of `/health`'s render-accounting fields this task fixes.
/// Mirrors the existing image-path semantics exactly: success bumps
/// `successCount` and stamps `lastDurationMs`; failure bumps `failedCount`
/// and leaves `lastDurationMs` alone (a failed render has no duration worth
/// reporting as "last render").
public struct RenderHealthCounters: Equatable, Sendable {
  public var successCount: Int
  public var failedCount: Int
  public var lastDurationMs: Int?

  public init(successCount: Int = 0, failedCount: Int = 0, lastDurationMs: Int? = nil) {
    self.successCount = successCount
    self.failedCount = failedCount
    self.lastDurationMs = lastDurationMs
  }

  public mutating func apply(_ event: RenderCompletionEvent) {
    switch event {
    case .succeeded(let durationMs):
      successCount += 1
      lastDurationMs = durationMs
    case .failed:
      failedCount += 1
    }
  }
}
