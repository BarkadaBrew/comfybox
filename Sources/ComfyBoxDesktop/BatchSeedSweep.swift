// BatchSeedSweep.swift — Pure seed-sweep logic for batch generation
// (Studio Packs FR-4 / #202: "fixed seed sweeps seed+1, seed+2…").
//
// Extracted out of GenerationView.submitGeneration() so it's directly
// unit-testable without a view/engine harness.

import Foundation

public enum BatchSeedSweep {
  /// The seed to use for batch item `index` (0-based) given a `baseSeed`.
  /// `baseSeed == 0` is passed through unchanged — the caller (server/engine)
  /// resolves 0 to a fresh random seed independently on each request, so a
  /// batch of "random" requests already varies without any client-side RNG.
  /// A non-zero base sweeps deterministically: `base, base+1, base+2, …`.
  public static func seed(baseSeed: UInt64, index: Int) -> UInt64 {
    guard baseSeed > 0 else { return 0 }
    return baseSeed + UInt64(index)
  }
}
