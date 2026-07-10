// BatchSeedSweep.swift — Pure seed-walk logic for batch generation
// (Studio Packs FR-4 / #202: "fixed seed sweeps seed+1, seed+2…"; extended
// with an explicit direction control per the "Seed Walk" section).
//
// Extracted out of GenerationView.submitGeneration() so it's directly
// unit-testable without a view/engine harness.

import Foundation

/// How a batch's seed changes from one iteration to the next when a fixed
/// base seed is set. `.up` is the original/default sweep direction.
public enum SeedWalkDirection: String, CaseIterable, Sendable, Identifiable, Codable {
  case up, down, random
  public var id: String { rawValue }

  public var label: String {
    switch self {
    case .up: return "Up"
    case .down: return "Down"
    case .random: return "Random"
    }
  }
}

public enum BatchSeedSweep {
  /// The seed to use for batch item `index` (0-based) given a `baseSeed`.
  /// `baseSeed == 0` is passed through unchanged — the caller (server/engine)
  /// resolves 0 to a fresh random seed independently on each request, so a
  /// batch of "random" requests already varies without any client-side RNG.
  ///
  /// With a non-zero base:
  /// - `.up` sweeps `base, base+1, base+2, …`
  /// - `.down` sweeps `base, base-1, base-2, …`, clamped at 1 (never 0 —
  ///   that value has its own "let the server randomize" meaning elsewhere).
  /// - `.random` ignores the base entirely and picks a fresh seed every
  ///   iteration, including the first.
  public static func seed(baseSeed: UInt64, index: Int, direction: SeedWalkDirection = .up) -> UInt64 {
    guard direction != .random else {
      return UInt64.random(in: 1..<UInt64(UInt32.max))
    }
    guard baseSeed > 0 else { return 0 }
    let step = UInt64(index)
    return direction == .up ? baseSeed + step : (baseSeed > step ? baseSeed - step : 1)
  }
}
