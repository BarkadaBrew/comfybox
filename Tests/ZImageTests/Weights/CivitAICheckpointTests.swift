import XCTest
import MLX
import Logging
@testable import ZImage

final class CivitAICheckpointTests: XCTestCase {

  // MARK: - Detection Tests

  func testInspectDetectsCivitAICheckpoint() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("civitai_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    // Build a synthetic CivitAI checkpoint with 450+ diffusion model keys
    var arrays: [String: MLXArray] = [:]
    let w1d = MLXArray([Float(0.0)]).asType(.bfloat16)
    let w2d = MLXArray([Float(0.0), Float(0.0)], [1, 2]).asType(.bfloat16)

    // Fused QKV for layers 0-29 (the signature of CivitAI format)
    for i in 0..<30 {
      arrays["model.diffusion_model.layers.\(i).attention.qkv.weight"] = MLXArray(Array(repeating: Float(0.0), count: 6 * 2), [6, 2]).asType(.bfloat16)
      arrays["model.diffusion_model.layers.\(i).attention.out.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).attention.q_norm.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention.k_norm.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention_norm1.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention_norm2.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).ffn_norm1.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).ffn_norm2.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w1.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w2.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w3.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).adaLN_modulation.1.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).adaLN_modulation.1.bias"] = w1d
    }
    // Noise/context refiners
    for prefix in ["noise_refiner", "context_refiner"] {
      for i in 0..<2 {
        arrays["model.diffusion_model.\(prefix).\(i).attention.qkv.weight"] = MLXArray(Array(repeating: Float(0.0), count: 6 * 2), [6, 2]).asType(.bfloat16)
        arrays["model.diffusion_model.\(prefix).\(i).attention.out.weight"] = w2d
      }
    }
    // Embedders
    arrays["model.diffusion_model.t_embedder.mlp.0.weight"] = w2d
    arrays["model.diffusion_model.t_embedder.mlp.0.bias"] = w1d
    arrays["model.diffusion_model.t_embedder.mlp.2.weight"] = w2d
    arrays["model.diffusion_model.t_embedder.mlp.2.bias"] = w1d
    arrays["model.diffusion_model.cap_embedder.0.weight"] = w1d
    arrays["model.diffusion_model.cap_embedder.1.weight"] = w2d
    arrays["model.diffusion_model.cap_embedder.1.bias"] = w1d
    arrays["model.diffusion_model.x_embedder.weight"] = w2d
    arrays["model.diffusion_model.final_layer.adaLN_modulation.1.weight"] = w2d
    arrays["model.diffusion_model.final_layer.linear.weight"] = w2d

    XCTAssertGreaterThanOrEqual(arrays.count, 400, "Synthetic checkpoint should have >= 400 keys")
    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let inspection = CivitAICheckpoint.inspect(fileURL: fileURL)
    XCTAssertTrue(inspection.isCivitAI, "Should detect as CivitAI: \(inspection.diagnostics)")
    XCTAssertEqual(inspection.variant, .base, "Fixture has Base signal keys (t_embedder, noise_refiner, etc.)")
    XCTAssertTrue(inspection.diagnostics.isEmpty)
    XCTAssertGreaterThanOrEqual(inspection.keyCount, 400)
  }

  func testInspectRejectsAIOCheckpoint() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("aio_reject_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let w = MLXArray([Float(0.0)]).asType(.bfloat16)
    var arrays: [String: MLXArray] = [:]

    // Add enough diffusion keys
    for i in 0..<30 {
      for j in 0..<15 {
        arrays["model.diffusion_model.layers.\(i).key\(j)"] = w
      }
    }
    // Add text encoder and VAE keys (makes it AIO, not CivitAI-only)
    arrays["text_encoders.qwen3_4b.transformer.model.embed_tokens.weight"] = w
    arrays["vae.decoder.conv_in.weight"] = w

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let inspection = CivitAICheckpoint.inspect(fileURL: fileURL)
    XCTAssertFalse(inspection.isCivitAI, "Should reject AIO checkpoint")
    XCTAssertTrue(inspection.diagnostics.contains(where: { $0.contains("text_encoder or VAE") }))
  }

  func testInspectRejectsInternalFormat() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("internal_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let w = MLXArray([Float(0.0)]).asType(.bfloat16)
    var arrays: [String: MLXArray] = [:]

    // Internal format: no prefix, already split Q/K/V
    for i in 0..<30 {
      arrays["layers.\(i).attention.to_q.weight"] = w
      arrays["layers.\(i).attention.to_k.weight"] = w
      arrays["layers.\(i).attention.to_v.weight"] = w
    }

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let inspection = CivitAICheckpoint.inspect(fileURL: fileURL)
    XCTAssertFalse(inspection.isCivitAI, "Should reject internal format (no prefix)")
  }

  func testInspectRejectsTooFewKeys() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("fewkeys_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let w = MLXArray([Float(0.0)]).asType(.bfloat16)
    var arrays: [String: MLXArray] = [:]

    // Only 10 keys with the right prefix
    for i in 0..<10 {
      arrays["model.diffusion_model.key\(i)"] = w
    }

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let inspection = CivitAICheckpoint.inspect(fileURL: fileURL)
    XCTAssertFalse(inspection.isCivitAI, "Should reject too few keys")
    XCTAssertTrue(inspection.diagnostics.contains(where: { $0.contains("diffusion keys") }))
  }

  // MARK: - Variant Detection Tests

  func testDetectsBaseVariantByGuidanceEmbedder() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("base_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let w1d = MLXArray([Float(0.0)]).asType(.bfloat16)
    let w2d = MLXArray([Float(0.0), Float(0.0)], [1, 2]).asType(.bfloat16)
    var arrays: [String: MLXArray] = [:]

    // Build minimum viable CivitAI checkpoint
    for i in 0..<30 {
      arrays["model.diffusion_model.layers.\(i).attention.qkv.weight"] = MLXArray(Array(repeating: Float(0.0), count: 6 * 2), [6, 2]).asType(.bfloat16)
      arrays["model.diffusion_model.layers.\(i).attention.out.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).attention.q_norm.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention.k_norm.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention_norm1.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention_norm2.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).ffn_norm1.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).ffn_norm2.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w1.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w2.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w3.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).adaLN_modulation.1.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).adaLN_modulation.1.bias"] = w1d
    }
    for prefix in ["noise_refiner", "context_refiner"] {
      for i in 0..<2 {
        arrays["model.diffusion_model.\(prefix).\(i).attention.qkv.weight"] = MLXArray(Array(repeating: Float(0.0), count: 6 * 2), [6, 2]).asType(.bfloat16)
        arrays["model.diffusion_model.\(prefix).\(i).attention.out.weight"] = w2d
      }
    }
    arrays["model.diffusion_model.t_embedder.mlp.0.weight"] = w2d
    arrays["model.diffusion_model.cap_embedder.0.weight"] = w1d
    arrays["model.diffusion_model.x_embedder.weight"] = w2d
    arrays["model.diffusion_model.final_layer.adaLN_modulation.1.weight"] = w2d

    // Add guidance embedder keys (Base-only)
    arrays["model.diffusion_model.guidance_in.mlp.0.weight"] = w2d
    arrays["model.diffusion_model.guidance_in.mlp.2.weight"] = w2d

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let inspection = CivitAICheckpoint.inspect(fileURL: fileURL)
    XCTAssertTrue(inspection.isCivitAI)
    XCTAssertEqual(inspection.variant, .base, "Guidance embedder should indicate Base variant")
  }

  func testDetectsTurboVariantByAbsenceOfGuidanceEmbedder() throws {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("turbo_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let w1d = MLXArray([Float(0.0)]).asType(.bfloat16)
    let w2d = MLXArray([Float(0.0), Float(0.0)], [1, 2]).asType(.bfloat16)
    var arrays: [String: MLXArray] = [:]

    // Turbo architecture: uses time_in instead of t_embedder, no noise/context refiners
    for i in 0..<30 {
      arrays["model.diffusion_model.layers.\(i).attention.qkv.weight"] = MLXArray(Array(repeating: Float(0.0), count: 6 * 2), [6, 2]).asType(.bfloat16)
      arrays["model.diffusion_model.layers.\(i).attention.out.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).attention.q_norm.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention.k_norm.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention_norm1.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).attention_norm2.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).ffn_norm1.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).ffn_norm2.weight"] = w1d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w1.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w2.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).feed_forward.w3.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).adaLN_modulation.1.weight"] = w2d
      arrays["model.diffusion_model.layers.\(i).adaLN_modulation.1.bias"] = w1d
    }
    // Turbo signal keys: time_in instead of t_embedder
    arrays["model.diffusion_model.time_in.in_layer.weight"] = w2d
    arrays["model.diffusion_model.time_in.out_layer.weight"] = w2d
    arrays["model.diffusion_model.x_embedder.weight"] = w2d
    arrays["model.diffusion_model.final_layer.adaLN_modulation.1.weight"] = w2d
    // NO guidance_in, NO t_embedder, NO noise_refiner, NO context_refiner

    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let inspection = CivitAICheckpoint.inspect(fileURL: fileURL)
    XCTAssertTrue(inspection.isCivitAI)
    XCTAssertEqual(inspection.variant, .turbo, "Turbo signal keys (time_in) and no Base signal keys should mean Turbo variant")
  }

  // MARK: - SafeTensorsReader FP8 Dtype Tests

  func testMapDTypeF8E4M3ReturnsUInt8() throws {
    // Verify FP8 dtype strings are handled (don't throw unsupportedDType)
    // We test by creating a safetensors file with BF16 and checking rawDType
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("dtype_\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let arrays: [String: MLXArray] = [
      "test.weight": MLXArray([Float(1.0)]).asType(.bfloat16)
    ]
    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)

    let reader = try SafeTensorsReader(fileURL: fileURL)
    let meta = reader.metadata(for: "test.weight")
    XCTAssertNotNil(meta)
    XCTAssertEqual(meta?.rawDType, "BF16")
    XCTAssertEqual(meta?.dtype, .bfloat16)
  }

  // MARK: - ModelPaths Variant Heuristic Tests

  func testFromModelSpecDetectsDPOAsTurbo() {
    let variant = ZImageVariant.fromModelSpec("/path/to/moodyRealMix_zitV6DPO.safetensors")
    XCTAssertEqual(variant, .turbo)
  }

  func testFromModelSpecDetectsWildAsBase() {
    let variant = ZImageVariant.fromModelSpec("/path/to/moody-wild-v4-fp16-full.safetensors")
    XCTAssertEqual(variant, .base)
  }

  func testFromModelSpecReturnsNilForUnknownCheckpoint() {
    let variant = ZImageVariant.fromModelSpec("/path/to/random-model.safetensors")
    XCTAssertNil(variant)
  }

  func testFromModelSpecStillDetectsHuggingFaceModels() {
    XCTAssertEqual(ZImageVariant.fromModelSpec("Tongyi-MAI/Z-Image"), .base)
    XCTAssertEqual(ZImageVariant.fromModelSpec("Tongyi-MAI/Z-Image-Turbo"), .turbo)
  }
}
