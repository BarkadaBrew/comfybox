import XCTest
import MLX
import MLXRandom
@testable import ZImage

final class LTX2PreemptionTests: XCTestCase {
  func testSignalRaiseClear() {
    let s = PreemptionSignal()
    XCTAssertFalse(s.isRaised)
    s.raise(); XCTAssertTrue(s.isRaised)
    s.clear(); XCTAssertFalse(s.isRaised)
  }

  func testSeededNoiseIsDeterministicPerStep() {
    let a = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 7)
    let b = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 7)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self), "same seed+step must be identical")
    let c = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 42, step: 8)
    XCTAssertFalse(MLX.allClose(a, c).item(Bool.self), "different step must differ")
    let d = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: 43, step: 7)
    XCTAssertFalse(MLX.allClose(a, d).item(Bool.self), "different seed must differ")
  }

  func testSeededNoiseIgnoresGlobalStreamPosition() {
    MLXRandom.seed(1)
    let a = ancestralVideoNoise(shape: [2, 2], seed: 9, step: 3)
    MLXRandom.seed(999)
    _ = MLXRandom.normal([16])   // scramble the global stream
    let b = ancestralVideoNoise(shape: [2, 2], seed: 9, step: 3)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self),
      "seeded draw must be independent of global stream position — this IS the resume guarantee")
  }

  func testUnseededNoiseUsesGlobalStream() {
    MLXRandom.seed(7)
    let a = ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0)
    MLXRandom.seed(7)
    let b = ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0)
    XCTAssertTrue(MLX.allClose(a, b).item(Bool.self),
      "unseeded path must remain the plain global stream (unchanged behaviour)")
  }

  func testNoiseDtypeIsFloat32() {
    XCTAssertEqual(ancestralVideoNoise(shape: [2, 2], seed: 1, step: 0).dtype, .float32)
    XCTAssertEqual(ancestralVideoNoise(shape: [2, 2], seed: nil, step: 0).dtype, .float32)
  }

  /// Regression for the codex-review finding (2026-08-15): the production
  /// chunk scheduler derives each chunk's seed as `request.seed + chunk`
  /// (`LTX2VideoGenerator.swift:911`), so a bare `seed + step` key formula
  /// makes chunk `c` step `i` collide with chunk `c+1` step `i-1` —
  /// `(seed+1) + i == seed + (i+1)`. The step multiplier must make that
  /// class of collision unreachable.
  func testChunkSeedDoesNotAliasStepKey() {
    let s: UInt64 = 42
    let i = 7
    let a = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: s &+ 1, step: i)
    let b = ancestralVideoNoise(shape: [1, 4, 2, 3, 3], seed: s, step: i &+ 1)
    XCTAssertFalse(MLX.allClose(a, b).item(Bool.self),
      "chunk-seed(+1)/step(i) must not collide with chunk-seed(+0)/step(i+1)")
  }
}
