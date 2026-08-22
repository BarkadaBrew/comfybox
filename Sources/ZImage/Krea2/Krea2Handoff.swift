// Krea2Handoff.swift — The mandatory base-handoff log line (FDD D17, AC-59a;
// WP-E10 "E9b", Addendum A.2).
//
// Two `.krea2` pool entries (~22.5 GB each) cannot co-reside under the 40 GB
// budget, so every base switch that touches the family is a ~67 s evict-and-
// reload. The line names OUTGOING and INCOMING spec/variant so a slow A/B is
// attributable rather than mysterious — and it fires ONLY on an actual swap:
// a no-op re-activation of the resident base, a cold start with nothing to
// hand off from, or a switch between two non-krea2 bases emits nothing.

import Foundation

public enum Krea2Handoff {

  /// One side of a handoff: the pool spec and a variant descriptor. For the
  /// krea2 family the descriptor is the PHYSICAL variant read off the loaded
  /// pipeline (`"raw"` / `"turbo"`), or `"unknown"` when the coordinator holds
  /// none — never an assumed `turbo`. For other families it is the family name.
  public struct Side: Equatable, Sendable {
    public let spec: String
    public let variant: String
    public let isKrea2: Bool

    public init(spec: String, variant: String, isKrea2: Bool? = nil) {
      self.spec = spec
      self.variant = variant
      self.isKrea2 = isKrea2 ?? (variant == "raw" || variant == "turbo" || variant == "unknown")
    }

    init(spec: String, family: WarmModelFamily, krea2Variant: Krea2Variant?) {
      self.spec = spec
      self.isKrea2 = family == .krea2
      self.variant = family == .krea2 ? (krea2Variant?.rawValue ?? "unknown") : family.rawValue
    }

    var descriptor: String { "\(spec)/\(variant)" }
  }

  /// The log line for a base handoff, or nil when there is no handoff to log:
  /// no resident base (`outgoing == nil`), the same base re-activated, or a
  /// swap in which neither side is krea2.
  public static func logLine(outgoing: Side?, incoming: Side, loadTimeMs: Int) -> String? {
    guard let outgoing else { return nil }
    guard outgoing != incoming else { return nil }
    guard outgoing.isKrea2 || incoming.isKrea2 else { return nil }
    return "krea2 handoff: \(outgoing.descriptor) → \(incoming.descriptor) (loadTimeMs=\(loadTimeMs))"
  }
}
