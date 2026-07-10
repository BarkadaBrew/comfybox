// BatchSeedSweepTests.swift — Batch seed-sweep logic (Studio Packs FR-4 / #202)

import Testing
@testable import ComfyBoxDesktop

@Suite("BatchSeedSweep")
struct BatchSeedSweepTests {
  @Test("fixed base seed sweeps base, base+1, base+2…")
  func fixedSeedSweeps() {
    #expect(BatchSeedSweep.seed(baseSeed: 100, index: 0) == 100)
    #expect(BatchSeedSweep.seed(baseSeed: 100, index: 1) == 101)
    #expect(BatchSeedSweep.seed(baseSeed: 100, index: 2) == 102)
  }

  @Test("zero base (random) is passed through unchanged at every index")
  func randomBasePassesThrough() {
    #expect(BatchSeedSweep.seed(baseSeed: 0, index: 0) == 0)
    #expect(BatchSeedSweep.seed(baseSeed: 0, index: 1) == 0)
    #expect(BatchSeedSweep.seed(baseSeed: 0, index: 5) == 0)
  }

  @Test("sweep never wraps or overflows for reasonable batch sizes")
  func noOverflowForReasonableBatch() {
    let base: UInt64 = UInt64.max - 3
    #expect(BatchSeedSweep.seed(baseSeed: base, index: 0) == base)
    #expect(BatchSeedSweep.seed(baseSeed: base, index: 1) == base + 1)
  }
}
