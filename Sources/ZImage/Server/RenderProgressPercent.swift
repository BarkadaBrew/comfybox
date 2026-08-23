// RenderProgressPercent.swift — the one mapping from "steps walked" to
// `/health.progress_percent` (WP-E10).
//
// The Krea 2 arm has a `(step, total)` callback that only wrote a log line, so
// `/health.progress_percent` sat at 0 for the whole of a ~2-minute Raw render
// and the desktop progress bar never moved. It now publishes through here.
//
// This is the Krea 2 path's mapping ONLY. The Z-Image path still computes its
// own `Int(fractionCompleted * 100)` inside the pipeline
// (`ZImagePipeline.swift`, `PipelineSnapshot.swift`), where the fraction is
// already known; the two agree by construction (same truncation, same 0…100
// scale) rather than by sharing this function. Folding that path in here is a
// worthwhile tidy-up, and is not this work package.

import Foundation

enum RenderProgressPercent {

  /// `step` of `total` as a 0…100 integer, truncated exactly the way the
  /// Z-Image path's `Int(fraction * 100)` truncates, so the two report on one
  /// scale. Degenerate inputs (a zero total,
  /// a step past the end) are clamped rather than trapped: a progress bar is
  /// never worth failing a render for.
  static func of(step: Int, total: Int) -> Int {
    guard total > 0, step > 0 else { return 0 }
    guard step < total else { return 100 }
    return Int((Double(step) / Double(total)) * 100)
  }
}
