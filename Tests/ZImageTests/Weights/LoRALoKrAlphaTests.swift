import XCTest
import MLX
@testable import ZImage

/// Tests for the LyCORIS LoKr alpha convention: the delta w1 ⊗ w2 is scaled
/// by alpha / dim (dim = smaller w2 dimension), not by alpha directly.
final class LoRALoKrAlphaTests: XCTestCase {

  func testAlphaIsDividedByDim() {
    // dim = min(16, 32) = 16, alpha 8 -> 0.5
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 8.0, w2Shape: [16, 32]), 0.5)
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 8.0, w2Shape: [32, 16]), 0.5)
  }

  func testAlphaEqualToDimYieldsUnitScale() {
    // LyCORIS stores alpha == dim for full-matrix LoKr modules.
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 16.0, w2Shape: [16, 32]), 1.0)
  }

  func testMissingAlphaYieldsUnitScale() {
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: nil, w2Shape: [16, 32]), 1.0)
  }

  func testZeroAlphaYieldsUnitScale() {
    // LyCORIS treats alpha == 0 as unset (scale 1).
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 0.0, w2Shape: [16, 32]), 1.0)
  }

  func testDegenerateShapesYieldUnitScale() {
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 8.0, w2Shape: []), 1.0)
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 8.0, w2Shape: [16]), 1.0)
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 8.0, w2Shape: [0, 16]), 1.0)
  }

  func testAiToolkitSentinelAlphaYieldsUnitScale() {
    // ai-toolkit-trained LoKr adapters (e.g. Krea2-realism-V2, zit_fdpo_v1,
    // zit-sda-v1) write a fixed ~1e10 sentinel into every .alpha tensor
    // regardless of the module's dim, instead of following the alpha == dim
    // convention. Dividing that by a typical dim (hundreds to a few
    // thousand) yields a scale in the millions — this must be clamped to the
    // neutral 1.0, not applied verbatim (it previously reduced generations to
    // pure noise).
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 9_999_221_000.0, w2Shape: [1536, 1536]), 1.0)
    XCTAssertEqual(LoRAApplicator.lokrAlphaScale(alpha: 9_999_221_000.0, w2Shape: [384, 1536]), 1.0)
  }
}
