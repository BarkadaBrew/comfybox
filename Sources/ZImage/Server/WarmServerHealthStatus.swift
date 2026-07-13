// WarmServerHealthStatus.swift — pure /health status derivation.
//
// Extracted so the status logic is unit-testable without a live server. The
// /health payload is otherwise assembled from a lock-based snapshot (so the
// endpoint never blocks behind an in-flight render on the coordinator actor,
// #217); this is the one branchy bit of that assembly.

import Foundation

public enum WarmServerHealthStatus {
  /// Derive the `status` string for /health.
  /// - `shutting_down` wins over everything.
  /// - `render_stale` when a render has been active longer than the threshold
  ///   (likely deadlocked — see #141).
  /// - `ok` otherwise.
  public static func derive(shuttingDown: Bool, activeRenderAgeMs: Int?, staleThresholdMs: Int) -> String {
    if shuttingDown { return "shutting_down" }
    if let age = activeRenderAgeMs, age > staleThresholdMs { return "render_stale" }
    return "ok"
  }
}
