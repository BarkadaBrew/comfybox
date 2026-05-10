import XCTest
import MLX
import MLXRandom
import MLXNN
@testable import ZImage

// MARK: - S1.1: CausalConv3d Tests

final class WanCausalConv3dTests: XCTestCase {

  func testModuleConstruction() {
    let conv = WanCausalConv3d(inChannels: 8, outChannels: 16, kernelSize: 3, padding: 1)
    XCTAssertEqual(conv.padding, 1)
    XCTAssertEqual(conv.kernelSize, 3)
    XCTAssertEqual(conv.stride, 1)
  }

  func testKernel1Construction() {
    let conv = WanCausalConv3d(inChannels: 4, outChannels: 8, kernelSize: 1, padding: 0)
    XCTAssertEqual(conv.padding, 0)
    XCTAssertEqual(conv.kernelSize, 1)
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

  func testChannelExpansionShape() throws {
    let conv = WanCausalConv3d(inChannels: 16, outChannels: 48, kernelSize: 3, padding: 1)
    let x = MLXArray.zeros([1, 16, 2, 4, 4], type: Float.self)
    let out = conv(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 48, 2, 4, 4])
  }

  func testBatchDimPreservedShape() throws {
    let conv = WanCausalConv3d(inChannels: 4, outChannels: 8, kernelSize: 3, padding: 1)
    let x = MLXArray.zeros([2, 4, 3, 6, 6], type: Float.self)
    let out = conv(x)
    eval(out)
    XCTAssertEqual(out.shape[0], 2)
  }
}

// MARK: - S1.2: Normalization Tests

final class WanNormTests: XCTestCase {

  func testRMSNormVideoConstruction() {
    let norm = WanRMSNorm(dim: 16, images: false)
    XCTAssertEqual(norm.scale, 4.0)  // sqrt(16)
    XCTAssertFalse(norm.images)
  }

  func testRMSNormImageConstruction() {
    let norm = WanRMSNorm(dim: 32, images: true)
    XCTAssertTrue(norm.images)
  }

  func testGroupNorm32Construction() {
    let norm = WanGroupNorm32(channels: 128)
    XCTAssertEqual(norm.channels, 128)
  }

  func testRMSNormOutputShapeVideo() throws {
    let norm = WanRMSNorm(dim: 16, images: false)
    let x = MLXArray.zeros([1, 16, 2, 4, 4], type: Float.self)
    let out = norm(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 16, 2, 4, 4])
  }

  func testRMSNormOutputShapeImages() throws {
    let norm = WanRMSNorm(dim: 32, images: true)
    let x = MLXArray.zeros([2, 32, 8, 8], type: Float.self)
    let out = norm(x)
    eval(out)
    XCTAssertEqual(out.shape, [2, 32, 8, 8])
  }

  func testGroupNorm32OutputShape() throws {
    let norm = WanGroupNorm32(channels: 128)
    let x = MLXArray.zeros([1, 128, 2, 4, 4], type: Float.self)
    let out = norm(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 128, 2, 4, 4])
  }
}

// MARK: - S1.3: ResidualBlock + AttentionBlock Tests

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
}

// MARK: - S1.4: DownBlock, UpBlock, MidBlock Tests

final class WanBlockTests: XCTestCase {

  func testMidBlockConstruction() {
    let _ = WanMidBlock(dim: 64)
  }

  func testDownBlockConstruction() {
    let _ = WanDownBlock(
      inDim: 32, outDim: 64, numResBlocks: 2,
      temporalDownsample: false, isLast: false
    )
  }

  func testUpBlockConstruction() {
    let _ = WanResidualUpBlock(
      inDim: 64, outDim: 32, numResBlocks: 2,
      temporalUpsample: false, upFlag: true
    )
  }

  func testMidBlockPreservesShape() throws {
    let mid = WanMidBlock(dim: 64)
    let x = MLXArray.zeros([1, 64, 2, 4, 4], type: Float.self)
    let out = mid(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 64, 2, 4, 4])
  }

  func testDownBlockSpatialDownsample() throws {
    // No temporal downsample — main path and shortcut both do spatial-only
    let down = WanDownBlock(
      inDim: 32, outDim: 64, numResBlocks: 2,
      temporalDownsample: false, isLast: false
    )
    let x = MLXArray.zeros([1, 32, 1, 8, 8], type: Float.self)
    let out = down(x)
    eval(out)
    // Spatial downsample 2x: H/2, W/2, T preserved
    XCTAssertEqual(out.shape, [1, 64, 1, 4, 4])
  }

  func testDownBlockLastNoDownsample() throws {
    let down = WanDownBlock(
      inDim: 64, outDim: 64, numResBlocks: 2,
      temporalDownsample: false, isLast: true
    )
    let x = MLXArray.zeros([1, 64, 1, 4, 4], type: Float.self)
    let out = down(x)
    eval(out)
    XCTAssertEqual(out.shape, [1, 64, 1, 4, 4])
  }

  func testResidualUpBlockSpatialUpsample() throws {
    let up = WanResidualUpBlock(
      inDim: 64, outDim: 32, numResBlocks: 2,
      temporalUpsample: false, upFlag: true
    )
    let x = MLXArray.zeros([1, 64, 1, 4, 4], type: Float.self)
    let out = up(x, firstChunk: true)
    eval(out)
    // Spatial upsample 2x: H*2, W*2
    XCTAssertEqual(out.shape, [1, 32, 1, 8, 8])
  }

  func testResidualUpBlockNoUpsample() throws {
    let up = WanResidualUpBlock(
      inDim: 64, outDim: 32, numResBlocks: 2,
      temporalUpsample: false, upFlag: false
    )
    let x = MLXArray.zeros([1, 64, 1, 4, 4], type: Float.self)
    let out = up(x, firstChunk: true)
    eval(out)
    XCTAssertEqual(out.shape, [1, 32, 1, 4, 4])
  }
}

// MARK: - S1.5: Encoder, Decoder, Patchify, WanVAE Tests

final class WanPatchifyTests: XCTestCase {

  func testPatchifyShape() throws {
    let x = MLXArray.zeros([1, 3, 4, 16, 16], type: Float.self)
    let patched = WanVAE.patchify(x, patchSize: 2)
    eval(patched)
    XCTAssertEqual(patched.shape, [1, 12, 4, 8, 8])
  }

  func testUnpatchifyShape() throws {
    let x = MLXArray.zeros([1, 12, 4, 8, 8], type: Float.self)
    let unpatched = WanVAE.unpatchify(x, patchSize: 2)
    eval(unpatched)
    XCTAssertEqual(unpatched.shape, [1, 3, 4, 16, 16])
  }

  func testPatchifyRoundtrip() throws {
    let x = MLXRandom.normal([1, 3, 2, 4, 4]).asType(.float32)
    let patched = WanVAE.patchify(x, patchSize: 2)
    let restored = WanVAE.unpatchify(patched, patchSize: 2)
    eval(x, restored)
    let diff = MLX.abs(x - restored)
    eval(diff)
    let maxDiff = MLX.max(diff).item(Float.self)
    XCTAssertLessThan(maxDiff, 1e-5)
  }
}

final class WanEncoder3dTests: XCTestCase {

  func testConstruction() {
    let _ = WanEncoder3d()
  }

  func testEncoderOutputShape() throws {
    // Use T=1 (single image), spatial dimensions that survive 3 levels of 2x downsample
    // Input: [1, 12, 1, 16, 16] -> after 3x spatial /2 = [16/8=2, 16/8=2]
    let encoder = WanEncoder3d()
    let x = MLXArray.zeros([1, 12, 1, 16, 16], type: Float.self)
    let out = encoder(x)
    eval(out)
    XCTAssertEqual(out.shape[0], 1)
    XCTAssertEqual(out.shape[1], 96) // z_dim * 2
    XCTAssertEqual(out.shape[2], 1)  // T preserved (no temporal downsample for T=1)
    XCTAssertEqual(out.shape[3], 2)  // 16 / 8
    XCTAssertEqual(out.shape[4], 2)  // 16 / 8
  }
}

final class WanDecoder3dTests: XCTestCase {

  func testConstruction() {
    let _ = WanDecoder3d()
  }

  func testDecoderOutputShape() throws {
    // Decoder: [1, 48, 1, 2, 2] -> 3 levels of 2x upsample = 2*8=16
    // But decoder has 4 blocks with dimMult=[1,2,4,4]:
    //   up_blocks[0]: 1024->1024, no upsample (last block reversed)
    //   up_blocks[1]: 1024->640, upsample 2x: 2->4
    //   up_blocks[2]: 640->320, upsample 2x: 4->8
    //   up_blocks[3]: 320->256, upsample 2x: 8->16
    // So output spatial is 2*8 = 16
    let decoder = WanDecoder3d()
    let x = MLXArray.zeros([1, 48, 1, 2, 2], type: Float.self)
    let out = decoder(x)
    eval(out)
    XCTAssertEqual(out.shape[0], 1)
    XCTAssertEqual(out.shape[1], 12) // out_channels
    XCTAssertEqual(out.shape[2], 1)  // T
    XCTAssertEqual(out.shape[3], 16) // 2 * 8 (3 levels of 2x upsample)
    XCTAssertEqual(out.shape[4], 16) // 2 * 8
  }
}

final class WanVAETests: XCTestCase {

  func testConstruction() {
    let _ = WanVAE()
  }

  func testLatentsMeanCount() {
    XCTAssertEqual(WanVAE.latentsMean.count, 48)
  }

  func testLatentsStdCount() {
    XCTAssertEqual(WanVAE.latentsStd.count, 48)
  }

  func testEncodeOutputShape() throws {
    // Full encode: [1, 3, 1, 32, 32]
    //   patchify(p=2): [1, 12, 1, 16, 16]
    //   encoder 3x spatial/2: [1, 96, 1, 2, 2]
    //   quant_conv: [1, 96, 1, 2, 2]
    //   take first 48 channels: [1, 48, 1, 2, 2]
    let vae = WanVAE()
    let x = MLXArray.zeros([1, 3, 1, 32, 32], type: Float.self)
    let latent = vae.encode(x)
    eval(latent)
    XCTAssertEqual(latent.shape[0], 1)
    XCTAssertEqual(latent.shape[1], 48)
    XCTAssertEqual(latent.shape[2], 1)
    XCTAssertEqual(latent.shape[3], 2)
    XCTAssertEqual(latent.shape[4], 2)
  }

  func testDecodeOutputShape() throws {
    // Full decode: [1, 48, 1, 2, 2]
    //   post_quant_conv: [1, 48, 1, 2, 2]
    //   decoder 3x spatial*2: [1, 12, 1, 16, 16]
    //   unpatchify(p=2): [1, 3, 1, 32, 32]
    let vae = WanVAE()
    let z = MLXArray.zeros([1, 48, 1, 2, 2], type: Float.self)
    let decoded = vae.decode(z)
    eval(decoded)
    XCTAssertEqual(decoded.shape[0], 1)
    XCTAssertEqual(decoded.shape[1], 3)
    XCTAssertEqual(decoded.shape[2], 1)
    XCTAssertEqual(decoded.shape[3], 32)
    XCTAssertEqual(decoded.shape[4], 32)
  }
}
