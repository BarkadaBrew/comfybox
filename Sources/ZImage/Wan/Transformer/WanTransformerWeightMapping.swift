import Foundation
import MLX

/// Weight mapping utilities for the Wan 2.2 I2V-A14B transformer.
///
/// The safetensors weight keys map directly to the Swift module hierarchy.
/// No key remapping is needed -- the module `@ModuleInfo` keys match the
/// safetensors keys exactly.
///
/// ## Key Structure
///
/// 27 keys per block (40 blocks = 1080) + 15 global = 1095 total keys.
///
/// ```
/// blocks.{i}.self_attn.{q,k,v,o}.{weight,bias}     8 keys
/// blocks.{i}.self_attn.norm_{q,k}.weight             2 keys
/// blocks.{i}.cross_attn.{q,k,v,o}.{weight,bias}     8 keys
/// blocks.{i}.cross_attn.norm_{q,k}.weight            2 keys
/// blocks.{i}.ffn.{0,2}.{weight,bias}                4 keys
/// blocks.{i}.norm3.{weight,bias}                     2 keys
/// blocks.{i}.modulation                              1 key
/// Total per block: 27
/// ```
public struct WanTransformerWeightMapping {

  // MARK: - Expected Keys

  /// All expected weight keys for the default I2V-A14B config.
  ///
  /// - Parameter config: Model configuration. Default: `.i2vA14B`.
  /// - Returns: Array of 1095 weight keys.
  public static func expectedKeys(
    config: WanTransformerConfig = .i2vA14B
  ) -> [String] {
    var keys: [String] = []

    // Global keys (15)
    keys.append(contentsOf: [
      "patch_embedding.weight",
      "patch_embedding.bias",
      "text_embedding.0.weight",
      "text_embedding.0.bias",
      "text_embedding.2.weight",
      "text_embedding.2.bias",
      "time_embedding.0.weight",
      "time_embedding.0.bias",
      "time_embedding.2.weight",
      "time_embedding.2.bias",
      "time_projection.1.weight",
      "time_projection.1.bias",
      "head.head.weight",
      "head.head.bias",
      "head.modulation",
    ])

    // Block keys (27 per block)
    for i in 0..<config.numLayers {
      keys.append(contentsOf: [
        // Self-attention (10 keys)
        "blocks.\(i).self_attn.q.weight",
        "blocks.\(i).self_attn.q.bias",
        "blocks.\(i).self_attn.k.weight",
        "blocks.\(i).self_attn.k.bias",
        "blocks.\(i).self_attn.v.weight",
        "blocks.\(i).self_attn.v.bias",
        "blocks.\(i).self_attn.o.weight",
        "blocks.\(i).self_attn.o.bias",
        "blocks.\(i).self_attn.norm_q.weight",
        "blocks.\(i).self_attn.norm_k.weight",
        // Cross-attention (10 keys)
        "blocks.\(i).cross_attn.q.weight",
        "blocks.\(i).cross_attn.q.bias",
        "blocks.\(i).cross_attn.k.weight",
        "blocks.\(i).cross_attn.k.bias",
        "blocks.\(i).cross_attn.v.weight",
        "blocks.\(i).cross_attn.v.bias",
        "blocks.\(i).cross_attn.o.weight",
        "blocks.\(i).cross_attn.o.bias",
        "blocks.\(i).cross_attn.norm_q.weight",
        "blocks.\(i).cross_attn.norm_k.weight",
        // FFN (4 keys)
        "blocks.\(i).ffn.0.weight",
        "blocks.\(i).ffn.0.bias",
        "blocks.\(i).ffn.2.weight",
        "blocks.\(i).ffn.2.bias",
        // Norm3 (2 keys)
        "blocks.\(i).norm3.weight",
        "blocks.\(i).norm3.bias",
        // Modulation (1 key)
        "blocks.\(i).modulation",
      ])
    }

    return keys
  }

  // MARK: - Validation

  /// Validates that all expected keys are present in a weight dictionary.
  ///
  /// - Parameters:
  ///   - weights: Dictionary of loaded weights.
  ///   - config: Model configuration.
  /// - Returns: Tuple of (missing keys, unexpected keys).
  public static func validateKeys(
    _ weights: [String: MLXArray],
    config: WanTransformerConfig = .i2vA14B
  ) -> (missing: [String], unexpected: [String]) {
    let expected = Set(expectedKeys(config: config))
    let actual = Set(weights.keys)

    let missing = expected.subtracting(actual).sorted()
    let unexpected = actual.subtracting(expected).sorted()

    return (missing: missing, unexpected: unexpected)
  }

  // MARK: - Expected Shapes

  /// Expected shapes for all weight keys.
  ///
  /// - Parameter config: Model configuration.
  /// - Returns: Dictionary mapping key names to expected shapes.
  public static func expectedShapes(
    config: WanTransformerConfig = .i2vA14B
  ) -> [String: [Int]] {
    let d = config.dim
    let ff = config.ffnDim
    let td = config.textDim
    let fd = config.freqDim
    let (pT, pH, pW) = config.patchSize
    let outFeatures = config.outDim * pT * pH * pW

    var shapes: [String: [Int]] = [
      // Global
      "patch_embedding.weight": [d, config.inDim, pT, pH, pW],
      "patch_embedding.bias": [d],
      "text_embedding.0.weight": [d, td],
      "text_embedding.0.bias": [d],
      "text_embedding.2.weight": [d, d],
      "text_embedding.2.bias": [d],
      "time_embedding.0.weight": [d, fd],
      "time_embedding.0.bias": [d],
      "time_embedding.2.weight": [d, d],
      "time_embedding.2.bias": [d],
      "time_projection.1.weight": [d * 6, d],
      "time_projection.1.bias": [d * 6],
      "head.head.weight": [outFeatures, d],
      "head.head.bias": [outFeatures],
      "head.modulation": [1, 2, d],
    ]

    // Block shapes
    for i in 0..<config.numLayers {
      // Self-attention
      shapes["blocks.\(i).self_attn.q.weight"] = [d, d]
      shapes["blocks.\(i).self_attn.q.bias"] = [d]
      shapes["blocks.\(i).self_attn.k.weight"] = [d, d]
      shapes["blocks.\(i).self_attn.k.bias"] = [d]
      shapes["blocks.\(i).self_attn.v.weight"] = [d, d]
      shapes["blocks.\(i).self_attn.v.bias"] = [d]
      shapes["blocks.\(i).self_attn.o.weight"] = [d, d]
      shapes["blocks.\(i).self_attn.o.bias"] = [d]
      shapes["blocks.\(i).self_attn.norm_q.weight"] = [d]
      shapes["blocks.\(i).self_attn.norm_k.weight"] = [d]
      // Cross-attention
      shapes["blocks.\(i).cross_attn.q.weight"] = [d, d]
      shapes["blocks.\(i).cross_attn.q.bias"] = [d]
      shapes["blocks.\(i).cross_attn.k.weight"] = [d, d]
      shapes["blocks.\(i).cross_attn.k.bias"] = [d]
      shapes["blocks.\(i).cross_attn.v.weight"] = [d, d]
      shapes["blocks.\(i).cross_attn.v.bias"] = [d]
      shapes["blocks.\(i).cross_attn.o.weight"] = [d, d]
      shapes["blocks.\(i).cross_attn.o.bias"] = [d]
      shapes["blocks.\(i).cross_attn.norm_q.weight"] = [d]
      shapes["blocks.\(i).cross_attn.norm_k.weight"] = [d]
      // FFN
      shapes["blocks.\(i).ffn.0.weight"] = [ff, d]
      shapes["blocks.\(i).ffn.0.bias"] = [ff]
      shapes["blocks.\(i).ffn.2.weight"] = [d, ff]
      shapes["blocks.\(i).ffn.2.bias"] = [d]
      // Norm3
      shapes["blocks.\(i).norm3.weight"] = [d]
      shapes["blocks.\(i).norm3.bias"] = [d]
      // Modulation
      shapes["blocks.\(i).modulation"] = [1, 6, d]
    }

    return shapes
  }
}
