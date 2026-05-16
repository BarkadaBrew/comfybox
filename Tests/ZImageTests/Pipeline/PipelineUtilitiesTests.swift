import XCTest
import MLX
@testable import ZImage

final class PipelineUtilitiesTests: XCTestCase {
  func testZImagePackedImageSeqLenUsesLatentPatchTokens() {
    XCTAssertEqual(
      PipelineUtilities.zImagePackedImageSeqLen(latentHeight: 128, latentWidth: 128),
      4096
    )
    XCTAssertEqual(
      PipelineUtilities.zImagePackedImageSeqLen(latentHeight: 64, latentWidth: 64),
      1024
    )
  }

  func testZImagePackedImageSeqLenClampsTinyDimensions() {
    XCTAssertEqual(
      PipelineUtilities.zImagePackedImageSeqLen(latentHeight: 1, latentWidth: 1),
      1
    )
  }

  func testDecodeLatentsDefaultsToFloat32() {
    let vae = DTypeCapturingVAE()
    let latents = MLX.ones([1, 16, 2, 2], dtype: .bfloat16)

    let decoded = PipelineUtilities.decodeLatents(latents, vae: vae, height: 2, width: 2)
    MLX.eval(decoded)

    XCTAssertEqual(vae.lastInputDType, .float32)
  }

  func testDecodeLatentsHonorsExplicitDType() {
    let vae = DTypeCapturingVAE()
    let latents = MLX.ones([1, 16, 2, 2], dtype: .float32)

    let decoded = PipelineUtilities.decodeLatents(
      latents,
      vae: vae,
      height: 2,
      width: 2,
      dtype: .bfloat16
    )
    MLX.eval(decoded)

    XCTAssertEqual(vae.lastInputDType, .bfloat16)
  }
}

private final class DTypeCapturingVAE: VAEImageDecoding {
  var lastInputDType: DType?

  func decode(_ latents: MLXArray, return_dict: Bool) -> (MLXArray, Any) {
    lastInputDType = latents.dtype
    return (MLX.zeros([1, 3, latents.dim(2), latents.dim(3)], dtype: latents.dtype), [:] as [String: Int])
  }
}
