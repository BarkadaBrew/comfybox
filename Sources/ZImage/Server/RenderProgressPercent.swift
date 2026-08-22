// RenderProgressPercent.swift — the one mapping from "steps walked" to
// `/health.progress_percent` (WP-E10).
//
// The Z-Image path publishes `Int(GenerationProgress.fractionCompleted * 100)`
// from its denoising callback. The Krea 2 arm has a `(step, total)` callback
// that only wrote a log line, so `/health.progress_percent` sat at 0 for the
// whole of a ~2-minute Raw render and the desktop progress bar never moved.
// Both families now go through this function, on one scale.

import Foundation

enum RenderProgressPercent {

  /// `step` of `total` as a 0…100 integer, truncated exactly as the Z-Image
  /// path's `Int(fraction * 100)` truncates. Degenerate inputs (a zero total,
  /// a step past the end) are clamped rather than trapped: a progress bar is
  /// never worth failing a render for.
  static func of(step: Int, total: Int) -> Int {
    guard total > 0, step > 0 else { return 0 }
    guard step < total else { return 100 }
    return Int((Double(step) / Double(total)) * 100)
  }
}
