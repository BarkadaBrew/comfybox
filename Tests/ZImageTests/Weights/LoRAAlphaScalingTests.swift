import XCTest
import MLX
@testable import ZImage

/// Tests for kohya/PEFT alpha scaling semantics: effective scale must be
/// userScale * alpha / rank, using per-layer alpha tensors when present.
final class LoRAAlphaScalingTests: XCTestCase {

  private let layerKey = "layers.0.attention.to_q.weight"

  // MARK: - Pure scaling math

  func testEffectiveScaleForLayerUsesPerLayerAlphaAndRank() {
    // down [2, 4] -> layer rank 2; alpha 1 -> scale 0.5
    let loraWeights = LoRAWeights(
      weights: [
        layerKey: (
          down: MLXArray([Float](repeating: 1.0, count: 8), [2, 4]),
          up: MLXArray([Float](repeating: 1.0, count: 8), [4, 2])
        )
      ],
      rank: 2,
      layerAlphas: [layerKey: 1.0]
    )

    XCTAssertEqual(loraWeights.effectiveScale(forLayer: layerKey), 0.5)
    // Key without ".weight" suffix must resolve to the same layer.
    XCTAssertEqual(loraWeights.effectiveScale(forLayer: "layers.0.attention.to_q"), 0.5)
  }

  func testEffectiveScaleForLayerFallsBackToGlobalAlphaWithLayerRank() {
    // Mixed ranks with a single adapter-wide alpha: each layer must use its
    // own rank (alpha 2 / rank 2 = 1.0, alpha 2 / rank 4 = 0.5).
    let key0 = "layers.0.attention.to_q.weight"
    let key1 = "layers.1.attention.to_q.weight"
    let loraWeights = LoRAWeights(
      weights: [
        key0: (
          down: MLXArray([Float](repeating: 1.0, count: 8), [2, 4]),
          up: MLXArray([Float](repeating: 1.0, count: 8), [4, 2])
        ),
        key1: (
          down: MLXArray([Float](repeating: 1.0, count: 16), [4, 4]),
          up: MLXArray([Float](repeating: 1.0, count: 16), [4, 4])
        )
      ],
      rank: 2,
      alpha: 2.0
    )

    XCTAssertEqual(loraWeights.effectiveScale(forLayer: key0), 1.0)
    XCTAssertEqual(loraWeights.effectiveScale(forLayer: key1), 0.5)
  }

  func testEffectiveScaleForLayerDefaultsToOneWithoutAnyAlpha() {
    let loraWeights = LoRAWeights(
      weights: [
        layerKey: (
          down: MLXArray([Float](repeating: 1.0, count: 8), [2, 4]),
          up: MLXArray([Float](repeating: 1.0, count: 8), [4, 2])
        )
      ],
      rank: 2
    )

    XCTAssertEqual(loraWeights.effectiveScale(forLayer: layerKey), 1.0)
  }

  func testEffectiveScaleForUnknownLayerFallsBackToGlobalScale() {
    let loraWeights = LoRAWeights(
      weights: [:],
      rank: 4,
      alpha: 2.0
    )

    XCTAssertEqual(loraWeights.effectiveScale(forLayer: "layers.9.attention.to_q.weight"), 0.5)
  }

  // MARK: - Fused path (merge)

  func testMergeWeightsAppliesPerLayerAlphaScale() {
    // down [1, 2], up [2, 1] -> delta = up @ down = [[3, 6], [4, 8]]
    // alpha 0.5 / rank 1 -> effective scale 0.5
    let baseWeights = [layerKey: MLXArray([Float](repeating: 0.0, count: 4), [2, 2])]
    let loraWeights = LoRAWeights(
      weights: [
        layerKey: (
          down: MLXArray([Float(1.0), 2.0], [1, 2]),
          up: MLXArray([Float(3.0), 4.0], [2, 1])
        )
      ],
      rank: 1,
      layerAlphas: [layerKey: 0.5]
    )

    let merged = LoRAApplicator.mergeWeights(
      baseWeights: baseWeights,
      loraWeights: loraWeights,
      scale: 1.0
    )

    XCTAssertEqual(merged[layerKey]?.asArray(Float.self), [1.5, 3.0, 2.0, 4.0])
  }

  func testMergeWeightsMultipliesUserScaleWithAlphaScale() {
    let baseWeights = [layerKey: MLXArray([Float](repeating: 0.0, count: 4), [2, 2])]
    let loraWeights = LoRAWeights(
      weights: [
        layerKey: (
          down: MLXArray([Float(1.0), 2.0], [1, 2]),
          up: MLXArray([Float(3.0), 4.0], [2, 1])
        )
      ],
      rank: 1,
      layerAlphas: [layerKey: 0.5]
    )

    let merged = LoRAApplicator.mergeWeights(
      baseWeights: baseWeights,
      loraWeights: loraWeights,
      scale: 2.0
    )

    XCTAssertEqual(merged[layerKey]?.asArray(Float.self), [3.0, 6.0, 4.0, 8.0])
  }

  func testMergeThenRemoveRestoresBaseWeights() {
    let baseValues: [Float] = [1.0, 2.0, 3.0, 4.0]
    let baseWeights = [layerKey: MLXArray(baseValues, [2, 2])]
    let loraWeights = LoRAWeights(
      weights: [
        layerKey: (
          down: MLXArray([Float(1.0), 2.0], [1, 2]),
          up: MLXArray([Float(3.0), 4.0], [2, 1])
        )
      ],
      rank: 1,
      layerAlphas: [layerKey: 0.5]
    )

    let merged = LoRAApplicator.mergeWeights(
      baseWeights: baseWeights,
      loraWeights: loraWeights,
      scale: 1.0
    )
    let restored = LoRAApplicator.removeFromWeights(
      mergedWeights: merged,
      loraWeights: loraWeights,
      scale: 1.0
    )

    XCTAssertEqual(restored[layerKey]?.asArray(Float.self), baseValues)
  }

  // MARK: - Dynamic path

  private final class DynamicLoRAStub: DynamicLoRACapable {
    var loraAdapters: [LoRAAdapter] = []
  }

  func testDynamicContributionUsesPerLayerEffectiveScale() {
    // down [2, 4] all ones, up [4, 2] all ones, x all ones:
    // x @ down.T = [4, 4]; @ up.T = [8, 8, 8, 8]; alpha 1 / rank 2 -> * 0.5
    let down = MLXArray([Float](repeating: 1.0, count: 8), [2, 4])
    let up = MLXArray([Float](repeating: 1.0, count: 8), [4, 2])
    let loraWeights = LoRAWeights(
      weights: [layerKey: (down: down, up: up)],
      rank: 2,
      layerAlphas: [layerKey: 1.0]
    )

    // Mirror LoRAApplicator.applyDynamically: the adapter is registered with
    // the per-layer effective scale, then contributes in the forward pass.
    let stub = DynamicLoRAStub()
    stub.addLoRA(down: down, up: up, scale: 1.0 * loraWeights.effectiveScale(forLayer: layerKey))

    let x = MLXArray([Float](repeating: 1.0, count: 4), [1, 4])
    let contribution = stub.computeLoRAContribution(x)

    XCTAssertNotNil(contribution)
    XCTAssertEqual(contribution?.asArray(Float.self), [4.0, 4.0, 4.0, 4.0])
  }

  // MARK: - Loader (synthetic kohya-style file)

  func testLoaderRoutesKohyaAlphaTensorsToPerLayerAlphas() throws {
    let fileURL = try writeLoRAFile(
      arrays: [
        "transformer.transformer_blocks.0.attn.to_q.lora_down.weight":
          MLXArray([Float](repeating: 1.0, count: 32), [4, 8]),
        "transformer.transformer_blocks.0.attn.to_q.lora_up.weight":
          MLXArray([Float](repeating: 1.0, count: 32), [8, 4]),
        "transformer.transformer_blocks.0.attn.to_q.alpha":
          MLXArray(Float(2.0))
      ]
    )
    defer { try? FileManager.default.removeItem(at: fileURL.deletingLastPathComponent()) }

    let loaded = try LoRAWeightLoader.load(from: fileURL)

    XCTAssertEqual(loaded.rank, 4)
    XCTAssertEqual(loaded.layerAlphas["layers.0.attention.to_q.weight"], 2.0)
    // alpha 2 / rank 4 = 0.5
    XCTAssertEqual(loaded.effectiveScale(forLayer: "layers.0.attention.to_q.weight"), 0.5)

    // End-to-end through the fused path: ones-delta entries are 4 (rank sum),
    // scaled by 0.5 -> 2.0 everywhere.
    let baseKey = "layers.0.attention.to_q.weight"
    let base = [baseKey: MLXArray([Float](repeating: 0.0, count: 64), [8, 8])]
    let merged = LoRAApplicator.mergeWeights(baseWeights: base, loraWeights: loaded, scale: 1.0)
    XCTAssertEqual(merged[baseKey]?.asArray(Float.self), [Float](repeating: 2.0, count: 64))
  }

  func testLoaderFallsBackToNetworkAlphaMetadata() throws {
    let fileURL = try writeLoRAFile(
      arrays: [
        "transformer.transformer_blocks.0.attn.to_q.lora_down.weight":
          MLXArray([Float](repeating: 1.0, count: 32), [4, 8]),
        "transformer.transformer_blocks.0.attn.to_q.lora_up.weight":
          MLXArray([Float](repeating: 1.0, count: 32), [8, 4])
      ],
      metadata: ["ss_network_alpha": "2"]
    )
    defer { try? FileManager.default.removeItem(at: fileURL.deletingLastPathComponent()) }

    let loaded = try LoRAWeightLoader.load(from: fileURL)

    XCTAssertEqual(loaded.alpha, 2.0)
    XCTAssertEqual(loaded.effectiveScale, 0.5)
    XCTAssertEqual(loaded.effectiveScale(forLayer: "layers.0.attention.to_q.weight"), 0.5)
  }

  func testLoaderInfersDeterministicRankForMixedRankAdapters() throws {
    let arrays: [String: MLXArray] = [
      "transformer.transformer_blocks.0.attn.to_q.lora_down.weight":
        MLXArray([Float](repeating: 1.0, count: 16), [2, 8]),
      "transformer.transformer_blocks.0.attn.to_q.lora_up.weight":
        MLXArray([Float](repeating: 1.0, count: 16), [8, 2]),
      "transformer.transformer_blocks.0.attn.to_q.alpha":
        MLXArray(Float(1.0)),
      "transformer.transformer_blocks.1.attn.to_q.lora_down.weight":
        MLXArray([Float](repeating: 1.0, count: 32), [4, 8]),
      "transformer.transformer_blocks.1.attn.to_q.lora_up.weight":
        MLXArray([Float](repeating: 1.0, count: 32), [8, 4]),
      "transformer.transformer_blocks.1.attn.to_q.alpha":
        MLXArray(Float(1.0))
    ]

    for _ in 0..<3 {
      let fileURL = try writeLoRAFile(arrays: arrays)
      defer { try? FileManager.default.removeItem(at: fileURL.deletingLastPathComponent()) }

      let loaded = try LoRAWeightLoader.load(from: fileURL)

      // Sorted key order: layers.0 (rank 2) comes first.
      XCTAssertEqual(loaded.rank, 2)
      // Per-layer scaling uses each layer's own alpha and rank.
      XCTAssertEqual(loaded.effectiveScale(forLayer: "layers.0.attention.to_q.weight"), 0.5)
      XCTAssertEqual(loaded.effectiveScale(forLayer: "layers.1.attention.to_q.weight"), 0.25)
    }
  }

  // MARK: - Helpers

  /// Writes the adapter into its own directory so the loader's search for a
  /// sibling adapter_config.json cannot pick up unrelated files.
  private func writeLoRAFile(
    arrays: [String: MLXArray],
    metadata: [String: String] = [:]
  ) throws -> URL {
    let directory = FileManager.default.temporaryDirectory
      .appendingPathComponent("lora_alpha_\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let fileURL = directory.appendingPathComponent("adapter.safetensors")
    try MLX.save(arrays: arrays, metadata: metadata, url: fileURL)
    return fileURL
  }
}
