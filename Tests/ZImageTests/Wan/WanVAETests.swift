import XCTest
import MLX
import MLXRandom
import MLXNN
@testable import ZImage

// MARK: - CausalConv3d Tests

final class WanCausalConv3dTests: XCTestCase {

  func testModuleConstruction() {
    let conv = WanCausalConv3d(inChannels: 8, outChannels: 16, kernelSize: 3, padding: 1)
    XCTAssertEqual(conv.paddingT, 1)
    XCTAssertEqual(conv.outChannels, 16)
    XCTAssertEqual(conv.strideT, 1)
  }

  func testKernel1Construction() {
    let conv = WanCausalConv3d(inChannels: 4, outChannels: 8, kernelSize: 1, padding: 0)
    XCTAssertEqual(conv.paddingT, 0)
    XCTAssertEqual(conv.outChannels, 8)
  }

  func testAnisotropicConstruction() {
    let conv = WanCausalConv3d(
      inChannels: 4, outChannels: 8,
      kernelSize: (3, 1, 1), stride: (2, 1, 1), padding: (0, 0, 0)
    )
    XCTAssertEqual(conv.strideT, 2)
    XCTAssertEqual(conv.strideH, 1)
    XCTAssertEqual(conv.paddingT, 0)
  }

  func testOutputShapePreserved() throws {
    let conv = WanCausalConv3d(inChannels: 8, outChannels: 16, kernelSize: 3, padding: 1)
    let x = MLXArray.zeros([1, 8, 4, 8, 8], type: Float.self)
    let out = conv(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 16, 4, 8, 8])
  }

  func testKernel1NoPaddingShape() throws {
    let conv = WanCausalConv3d(inChannels: 4, outChannels: 8, kernelSize: 1, padding: 0)
    let x = MLXArray.zeros([1, 4, 2, 4, 4], type: Float.self)
    let out = conv(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 8, 2, 4, 4])
  }

  func testCausalTemporalPaddingShape() throws {
    let conv = WanCausalConv3d(inChannels: 2, outChannels: 2, kernelSize: 3, padding: 1)
    let x = MLXArray.zeros([1, 2, 3, 6, 6], type: Float.self)
    let out = conv(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 2, 3, 6, 6])
  }

  func testTemporalDownsampleStride() throws {
    let conv = WanCausalConv3d(
      inChannels: 4, outChannels: 4,
      kernelSize: (3, 1, 1), stride: (2, 1, 1), padding: (0, 0, 0)
    )
    // kernel=3, stride=2, no padding: output_t = floor((T-3)/2) + 1
    // T=5 -> output_t = floor(2/2) + 1 = 2
    let x = MLXArray.zeros([1, 4, 5, 4, 4], type: Float.self)
    let out = conv(x)
    eval(out)
    XCTAssertEqual(out.shape[2], 2)
    XCTAssertEqual(out.shape[3], 4)
    XCTAssertEqual(out.shape[4], 4)
  }

  func testBatchDimPreservedShape() throws {
    let conv = WanCausalConv3d(inChannels: 4, outChannels: 8, kernelSize: 3, padding: 1)
    let x = MLXArray.zeros([2, 4, 3, 6, 6], type: Float.self)
    let out = conv(x)
    eval(out)
    XCTAssertEqual(out.shape[0], 2)
  }
}

// MARK: - WanVAENorm Tests

final class WanVAENormTests: XCTestCase {

  func testNormVideoConstruction() {
    let norm = WanVAENorm(dim: 16, images: false)
    XCTAssertEqual(norm.scale, 4.0)  // sqrt(16)
    XCTAssertFalse(norm.images)
    XCTAssertEqual(norm.gamma.shape, [16, 1, 1, 1])
  }

  func testNormImageConstruction() {
    let norm = WanVAENorm(dim: 32, images: true)
    XCTAssertTrue(norm.images)
    XCTAssertEqual(norm.gamma.shape, [32, 1, 1])
  }

  func testNormOutputShapeVideo() throws {
    let norm = WanVAENorm(dim: 16, images: false)
    let x = MLXArray.zeros([1, 16, 2, 4, 4], type: Float.self)
    let out = norm(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 16, 2, 4, 4])
  }

  func testNormOutputShapeImages() throws {
    let norm = WanVAENorm(dim: 32, images: true)
    let x = MLXArray.zeros([2, 32, 8, 8], type: Float.self)
    let out = norm(x)
    eval(out)
    XCTAssertEqual(out.shape, [2, 32, 8, 8])
  }

  func testNormL2Behavior() throws {
    // After normalization, each spatial position should have unit L2 norm
    // (before scaling by sqrt(dim) * gamma)
    let norm = WanVAENorm(dim: 4, images: false)
    let x = MLXRandom.normal([1, 4, 1, 2, 2]).asType(.float32)
    let out = norm(x)
    eval(out)
    // The output should be scaled by sqrt(4) * 1.0 = 2.0
    // Check that magnitude is roughly sqrt(dim) = 2.0
    let normSq = MLX.sum(out * out, axis: 1)
    eval(normSq)
    // Each spatial position should have norm^2 = dim (since gamma=1)
    let expected = Float(4.0)  // dim
    let actual = normSq[0, 0, 0, 0].item(Float.self)
    XCTAssertEqual(actual, expected, accuracy: 0.01)
  }
}

// MARK: - ResidualBlock + AttentionBlock Tests

final class WanResidualBlockTests: XCTestCase {

  func testSameChannelsConstruction() {
    let block = WanResidualBlock(inDim: 32, outDim: 32)
    XCTAssertFalse(block.hasShortcut)
  }

  func testDifferentChannelsConstruction() {
    let block = WanResidualBlock(inDim: 16, outDim: 32)
    XCTAssertTrue(block.hasShortcut)
  }

  func testSameChannelsShape() throws {
    let block = WanResidualBlock(inDim: 32, outDim: 32)
    let x = MLXArray.zeros([1, 32, 2, 4, 4], type: Float.self)
    let out = block(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 32, 2, 4, 4])
  }

  func testDifferentChannelsShape() throws {
    let block = WanResidualBlock(inDim: 16, outDim: 32)
    let x = MLXArray.zeros([1, 16, 2, 4, 4], type: Float.self)
    let out = block(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 32, 2, 4, 4])
  }

  func testWan21Channels() throws {
    // Test with actual Wan 2.1 channel dims: 96->192
    let block = WanResidualBlock(inDim: 96, outDim: 192)
    let x = MLXArray.zeros([1, 96, 1, 4, 4], type: Float.self)
    let out = block(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 192, 1, 4, 4])
  }
}

final class WanAttentionBlockTests: XCTestCase {

  func testConstruction() {
    let block = WanAttentionBlock(dim: 32)
    XCTAssertEqual(block.dim, 32)
  }

  func testOutputShapePreserved() throws {
    let block = WanAttentionBlock(dim: 32)
    let x = MLXArray.zeros([1, 32, 2, 4, 4], type: Float.self)
    let out = block(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 32, 2, 4, 4])
  }

  func testWan21ChannelDim() throws {
    // Wan 2.1 uses dim=384 for attention in the middle block
    let block = WanAttentionBlock(dim: 384)
    let x = MLXArray.zeros([1, 384, 1, 2, 2], type: Float.self)
    let out = block(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 384, 1, 2, 2])
  }
}

// MARK: - Resample Tests

final class WanResampleTests: XCTestCase {

  func testDownsample2dShape() throws {
    let ds = WanResample(dim: 96, mode: .downsample2d)
    let x = MLXArray.zeros([1, 96, 2, 8, 8], type: Float.self)
    let out = ds(x)
    eval(out)
    // Spatial /2, temporal preserved, channels preserved
    XCTAssertEqual(out.shape, [1, 96, 2, 4, 4])
  }

  func testDownsample3dShape() throws {
    let ds = WanResample(dim: 192, mode: .downsample3d)
    // time_conv: kernel=3, stride=2, pad=0 -> output_t = floor((T-3)/2) + 1
    // T=5 -> output_t = floor(2/2) + 1 = 2
    let x = MLXArray.zeros([1, 192, 5, 8, 8], type: Float.self)
    let out = ds(x)
    eval(out)
    // Spatial /2, temporal -> 2, channels preserved
    XCTAssertEqual(out.shape, [1, 192, 2, 4, 4])
  }

  func testUpsample2dShape() throws {
    let us = WanResample(dim: 192, mode: .upsample2d)
    let x = MLXArray.zeros([1, 192, 2, 4, 4], type: Float.self)
    let out = us(x)
    eval(out)
    // Spatial *2, temporal preserved, channels halved
    XCTAssertEqual(out.shape, [1, 96, 2, 8, 8])
  }

  func testUpsample3dShape() throws {
    let us = WanResample(dim: 384, mode: .upsample3d)
    let x = MLXArray.zeros([1, 384, 2, 4, 4], type: Float.self)
    let out = us(x)
    eval(out)
    // Spatial *2, temporal *2, channels halved
    XCTAssertEqual(out.shape, [1, 192, 4, 8, 8])
  }
}

// MARK: - Middle Block Tests

final class WanMiddleLayersTests: XCTestCase {

  func testPreservesShape() throws {
    let mid = WanMiddleLayers(dim: 384)
    let x = MLXArray.zeros([1, 384, 1, 2, 2], type: Float.self)
    let out = mid(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 384, 1, 2, 2])
  }
}

// MARK: - Encoder, Decoder, VAE Tests

final class WanEncoder3dTests: XCTestCase {

  func testConstruction() {
    let _ = WanEncoder3d()
  }

  func testEncoderOutputShape() throws {
    // Input: [1, 3, 7, 16, 16]
    // Spatial: 3x downsample /2 -> 16/8 = 2
    // Temporal: 2x downsample (k=3,s=2,p=0): 7 -> 3 -> 1
    // Output channels: z_dim*2 = 32
    let encoder = WanEncoder3d()
    let x = MLXArray.zeros([1, 3, 7, 16, 16], type: Float.self)
    let out = encoder(x)
    eval(out)
    XCTAssertEqual(out.shape[0], 1)
    XCTAssertEqual(out.shape[1], 32) // z_dim * 2 = 16 * 2
    XCTAssertEqual(out.shape[2], 1)  // 7 -> 3 -> 1
    XCTAssertEqual(out.shape[3], 2)  // 16 / 8
    XCTAssertEqual(out.shape[4], 2)  // 16 / 8
  }
}

final class WanDecoder3dTests: XCTestCase {

  func testConstruction() {
    let _ = WanDecoder3d()
  }

  func testDecoderOutputShape() throws {
    // Input: [1, 16, 1, 2, 2]
    // Spatial: 3x upsample *2 -> 2*8 = 16
    // Temporal: 2x upsample *2 -> 1*4 = 4
    // Output channels: 3 (RGB)
    let decoder = WanDecoder3d()
    let x = MLXArray.zeros([1, 16, 1, 2, 2], type: Float.self)
    let out = decoder(x)
    eval(out)
    XCTAssertEqual(out.shape[0], 1)
    XCTAssertEqual(out.shape[1], 3)  // RGB
    XCTAssertEqual(out.shape[2], 4)  // T: 1 -> 2 -> 4
    XCTAssertEqual(out.shape[3], 16) // 2 * 8
    XCTAssertEqual(out.shape[4], 16) // 2 * 8
  }
}

final class WanVAETests: XCTestCase {

  func testConstruction() {
    let _ = WanVAE()
  }

  func testConstants() {
    XCTAssertEqual(WanVAE.zDim, 16)
    XCTAssertEqual(WanVAE.spatialScale, 8)
    XCTAssertEqual(WanVAE.temporalScale, 4)
    XCTAssertEqual(WanVAE.latentMean.count, 16)
    XCTAssertEqual(WanVAE.latentStd.count, 16)
  }

  func testEncodeOutputShape() throws {
    // Input: [1, 3, 7, 16, 16]
    // Encoder: 8x spatial (16->2), temporal 7->3->1
    // conv1: [1, 32, 1, 2, 2]
    // Take first 16 channels: [1, 16, 1, 2, 2]
    let vae = WanVAE()
    let x = MLXArray.zeros([1, 3, 7, 16, 16], type: Float.self)
    let latent = vae.encode(x)
    eval(latent)
    XCTAssertEqual(latent.shape[0], 1)
    XCTAssertEqual(latent.shape[1], 16)
    XCTAssertEqual(latent.shape[2], 1)
    XCTAssertEqual(latent.shape[3], 2)
    XCTAssertEqual(latent.shape[4], 2)
  }

  func testDecodeOutputShape() throws {
    // Input: [1, 16, 1, 2, 2]
    // conv2: [1, 16, 1, 2, 2]
    // Decoder: 8x spatial (2->16), temporal 1->2->4
    let vae = WanVAE()
    let z = MLXArray.zeros([1, 16, 1, 2, 2], type: Float.self)
    let decoded = vae.decode(z)
    eval(decoded)
    XCTAssertEqual(decoded.shape[0], 1)
    XCTAssertEqual(decoded.shape[1], 3)
    XCTAssertEqual(decoded.shape[2], 4)  // T: 1 -> 2 -> 4
    XCTAssertEqual(decoded.shape[3], 16)
    XCTAssertEqual(decoded.shape[4], 16)
  }

  func testRoundTripShapes() throws {
    // Verify encode -> decode produces correct output dimensions
    // T=7 encodes to T=1, which decodes to T=4
    let vae = WanVAE()
    let x = MLXArray.zeros([1, 3, 7, 16, 16], type: Float.self)
    let latent = vae.encode(x)
    eval(latent)
    XCTAssertEqual(latent.shape, [1, 16, 1, 2, 2])
    let decoded = vae.decode(latent)
    eval(decoded)
    XCTAssertEqual(decoded.shape[0], 1)
    XCTAssertEqual(decoded.shape[1], 3)
    XCTAssertEqual(decoded.shape[2], 4)  // temporal upsample 1->2->4
    XCTAssertEqual(decoded.shape[3], 16)
    XCTAssertEqual(decoded.shape[4], 16)
  }
}
