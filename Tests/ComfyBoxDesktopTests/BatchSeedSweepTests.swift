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

  @Test("explicit .up direction matches the default")
  func explicitUpMatchesDefault() {
    #expect(BatchSeedSweep.seed(baseSeed: 100, index: 2, direction: .up) == 102)
  }

  @Test(".down direction sweeps base, base-1, base-2…")
  func downDirectionSweepsDown() {
    #expect(BatchSeedSweep.seed(baseSeed: 100, index: 0, direction: .down) == 100)
    #expect(BatchSeedSweep.seed(baseSeed: 100, index: 1, direction: .down) == 99)
    #expect(BatchSeedSweep.seed(baseSeed: 100, index: 2, direction: .down) == 98)
  }

  @Test(".down direction clamps at 1, never reaching 0's special random meaning")
  func downDirectionClampsAtOne() {
    #expect(BatchSeedSweep.seed(baseSeed: 3, index: 5, direction: .down) == 1)
    #expect(BatchSeedSweep.seed(baseSeed: 3, index: 3, direction: .down) == 1)
  }

  @Test(".down direction with a zero base still passes through unchanged")
  func downDirectionZeroBasePassesThrough() {
    #expect(BatchSeedSweep.seed(baseSeed: 0, index: 3, direction: .down) == 0)
  }

  @Test(".random direction ignores the base and picks a fresh seed every time")
  func randomDirectionIgnoresBase() {
    let values = (0..<20).map { BatchSeedSweep.seed(baseSeed: 100, index: $0, direction: .random) }
    #expect(values.allSatisfy { $0 > 0 })
    // Overwhelmingly unlikely to all collide if it's genuinely randomizing.
    #expect(Set(values).count > 1)
  }

  @Test("SeedWalkDirection labels are human-readable")
  func directionLabels() {
    #expect(SeedWalkDirection.up.label == "Up")
    #expect(SeedWalkDirection.down.label == "Down")
    #expect(SeedWalkDirection.random.label == "Random")
  }
}
