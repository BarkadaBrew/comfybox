import XCTest
import MLX
@testable import ZImage

final class LTX2QuantizerTests: XCTestCase {

  private let spec = LTX2Quantizer.Spec(bits: 8, groupSize: 64)

  func testPolicyQuantizesBlockProjectionsOnly() {
    // Block attention/FF projections: yes.
    XCTAssertTrue(LTX2Quantizer.shouldQuantize(
      key: "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
      ndim: 2, inDim: 4096, groupSize: 64))
    XCTAssertTrue(LTX2Quantizer.shouldQuantize(
      key: "transformer_blocks.47.ff.net.0.proj.weight",
      ndim: 2, inDim: 4096, groupSize: 64))
    // Outside the blocks (patchify, final proj, caption): no.
    XCTAssertFalse(LTX2Quantizer.shouldQuantize(
      key: "model.diffusion_model.patchify_proj.weight", ndim: 2, inDim: 128, groupSize: 64))
    XCTAssertFalse(LTX2Quantizer.shouldQuantize(
      key: "model.diffusion_model.proj_out.weight", ndim: 2, inDim: 4096, groupSize: 64))
    // Norms and non-2D: no.
    XCTAssertFalse(LTX2Quantizer.shouldQuantize(
      key: "model.diffusion_model.transformer_blocks.0.attn1.norm_q.weight",
      ndim: 1, inDim: 0, groupSize: 64))
    // Audio branch stays bf16.
    XCTAssertFalse(LTX2Quantizer.shouldQuantize(
      key: "model.diffusion_model.transformer_blocks.0.audio_attn.to_q.weight",
      ndim: 2, inDim: 4096, groupSize: 64))
    XCTAssertFalse(LTX2Quantizer.shouldQuantize(
      key: "model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_q.weight",
      ndim: 2, inDim: 4096, groupSize: 64))
    // Indivisible inDim: no.
    XCTAssertFalse(LTX2Quantizer.shouldQuantize(
      key: "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
      ndim: 2, inDim: 100, groupSize: 64))
  }

  func testQuantizeTensorsRoundtrip() {
    let key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
    let dense = MLXRandom.normal([128, 128]).asType(.bfloat16)
    let passthrough = MLXRandom.normal([128]).asType(.bfloat16)
    let weights = [
      key: dense,
      "model.diffusion_model.transformer_blocks.0.attn1.norm_q.weight": passthrough,
      "vae.decoder.conv_in.weight": MLXRandom.normal([4, 4, 3, 3]),
    ]

    let (out, summary) = LTX2Quantizer.quantizeTensors(weights, spec: spec)

    XCTAssertEqual(summary.quantizedCount, 1)
    XCTAssertEqual(summary.passthroughCount, 2)
    // Quantized triplet present, packed uint32 + bf16 siblings.
    let base = String(key.dropLast(".weight".count))
    XCTAssertEqual(out[key]?.dtype, .uint32)
    XCTAssertEqual(out["\(base).scales"]?.dtype, .bfloat16)
    XCTAssertNotNil(out["\(base).biases"])
    // Passthrough untouched.
    XCTAssertEqual(out["vae.decoder.conv_in.weight"]?.dtype, .float32)

    // Dequantized result approximates the dense original.
    guard let (roundtrip, groupSize, bits) = LTX2Quantizer.dequantizeLayer(base: base, weights: out) else {
      return XCTFail("dequantizeLayer failed")
    }
    XCTAssertEqual(groupSize, 64)
    XCTAssertEqual(bits, 8)
    let err = MLX.abs(roundtrip.asType(.float32) - dense.asType(.float32)).max().item(Float.self)
    XCTAssertLessThan(err, 0.05, "int8 roundtrip error should be small")
  }

  func testSanitizePreservesQuantSiblings() {
    // The monolith sanitize path must carry .scales/.biases through with the
    // same renames as .weight so the loader sees a consistent triplet.
    let wq = MLXArray.zeros([128, 32], dtype: .uint32)
    let scales = MLXArray.zeros([128, 2], dtype: .bfloat16)
    let weights = [
      "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight": wq,
      "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.scales": scales,
      "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.scales": scales,
    ]
    let sanitized = LTX2Transformer.sanitizeWeights(weights)
    XCTAssertNotNil(sanitized["transformer_blocks.0.attn1.to_out.weight"])
    XCTAssertNotNil(sanitized["transformer_blocks.0.attn1.to_out.scales"])
    XCTAssertNotNil(sanitized["transformer_blocks.0.ff.proj_in.scales"])
  }
}
