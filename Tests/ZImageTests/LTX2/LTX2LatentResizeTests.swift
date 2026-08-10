import Foundation
import MLX
import XCTest

@testable import ZImage

/// Bilinear spatial resize for 5D latents [B,C,T,H,W] — the 1.5x refine
/// (Todd 2026-08-07: "1.5 is enough", "render times are already too long").
/// The learned upsampler is fixed 2x; the refine then runs at 2x cost. This
/// resizes the upsampled latent down to the requested scale so the 3-step
/// refine denoise runs on ~56% of the area.
final class LTX2LatentResizeTests: XCTestCase {

  func testIdentityWhenTargetEqualsSource() {
    let x = MLXRandom.normal([1, 4, 2, 8, 6])
    let y = LTX2Conditioning.resizeLatentBilinear(x, height: 8, width: 6)
    XCTAssertEqual(y.shape, x.shape)
    let diff = MLX.abs(y - x).max().item(Float.self)
    XCTAssertLessThan(diff, 1e-5, "same-size resize must be identity")
  }

  func testShapeAndDtypeAtThreeQuarterScale() {
    let x = MLXRandom.normal([1, 128, 3, 16, 12]).asType(.bfloat16)
    let y = LTX2Conditioning.resizeLatentBilinear(x, height: 12, width: 9)
    XCTAssertEqual(y.shape, [1, 128, 3, 12, 9])
    XCTAssertEqual(y.dtype, x.dtype)
  }

  func testConstantFieldIsPreservedExactly() {
    // Bilinear of a constant is the constant — catches weight normalization bugs.
    let x = MLXArray.ones([1, 2, 1, 10, 10]) * 3.5
    let y = LTX2Conditioning.resizeLatentBilinear(x, height: 7, width: 7)
    let maxDev = MLX.abs(y - 3.5).max().item(Float.self)
    XCTAssertLessThan(maxDev, 1e-4)
  }

  func testLinearRampIsPreserved() {
    // Bilinear reproduces linear signals up to boundary handling: a horizontal
    // ramp downsampled must stay monotonic with matching endpoints.
    let w = 16
    let ramp = (0..<w).map { Float($0) }
    var data = [Float]()
    for _ in 0..<8 { data += ramp }                    // H=8 rows of the ramp
    let x = MLXArray(data).reshaped([1, 1, 1, 8, w])
    let y = LTX2Conditioning.resizeLatentBilinear(x, height: 6, width: 12)
    let row = y[0, 0, 0, 0].asArray(Float.self)
    for i in 1..<row.count {
      XCTAssertGreaterThan(row[i], row[i-1], "ramp must stay monotonic")
    }
    XCTAssertEqual(row.first!, 0, accuracy: 0.75)
    XCTAssertEqual(row.last!, Float(w - 1), accuracy: 0.75)
  }
}
