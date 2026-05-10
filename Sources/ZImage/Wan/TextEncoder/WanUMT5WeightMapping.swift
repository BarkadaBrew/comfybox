import Foundation
import MLX

/// Weight mapping utilities for the Wan 2.2 UMT5-XXL text encoder.
///
/// The Wan checkpoint uses its own naming convention that maps directly
/// to the Swift module hierarchy:
///
/// ```
/// Wan .pth Key                                    → Swift Module Path
/// ──────────────────────────────────────────────── ─────────────────────
/// token_embedding.weight                          → token_embedding.weight
/// blocks.{i}.attn.q.weight                        → blocks.{i}.attn.q.weight
/// blocks.{i}.attn.k.weight                        → blocks.{i}.attn.k.weight
/// blocks.{i}.attn.v.weight                        → blocks.{i}.attn.v.weight
/// blocks.{i}.attn.o.weight                        → blocks.{i}.attn.o.weight
/// blocks.{i}.pos_embedding.embedding.weight       → blocks.{i}.pos_embedding.embedding.weight
/// blocks.{i}.norm1.weight                         → blocks.{i}.norm1.weight
/// blocks.{i}.ffn.gate.0.weight                    → blocks.{i}.ffn.gate.0.weight
/// blocks.{i}.ffn.fc1.weight                       → blocks.{i}.ffn.fc1.weight
/// blocks.{i}.ffn.fc2.weight                       → blocks.{i}.ffn.fc2.weight
/// blocks.{i}.norm2.weight                         → blocks.{i}.norm2.weight
/// norm.weight                                     → norm.weight
/// ```
///
/// Because we designed the Swift module hierarchy to match the Wan naming,
/// no key remapping is needed — the safetensors keys are the module paths.
public struct WanUMT5WeightMapping {

  /// All expected weight keys for the default WanUMT5Config.
  ///
  /// Returns 242 keys total: 2 top-level + 10 per layer * 24 layers.
  public static func expectedKeys(config: WanUMT5Config = .wan22I2V) -> [String] {
    var keys: [String] = [
      "token_embedding.weight",
      "norm.weight"
    ]

    for i in 0..<config.numLayers {
      keys.append(contentsOf: [
        "blocks.\(i).attn.q.weight",
        "blocks.\(i).attn.k.weight",
        "blocks.\(i).attn.v.weight",
        "blocks.\(i).attn.o.weight",
        "blocks.\(i).pos_embedding.embedding.weight",
        "blocks.\(i).norm1.weight",
        "blocks.\(i).ffn.gate.0.weight",
        "blocks.\(i).ffn.fc1.weight",
        "blocks.\(i).ffn.fc2.weight",
        "blocks.\(i).norm2.weight",
      ])
    }

    return keys
  }

  /// Validates that all expected keys are present in a weight dictionary.
  ///
  /// - Parameters:
  ///   - weights: Dictionary of loaded weights.
  ///   - config: Model configuration.
  /// - Returns: Tuple of (missing keys, unexpected keys).
  public static func validateKeys(
    _ weights: [String: MLXArray],
    config: WanUMT5Config = .wan22I2V
  ) -> (missing: [String], unexpected: [String]) {
    let expected = Set(expectedKeys(config: config))
    let actual = Set(weights.keys)

    let missing = expected.subtracting(actual).sorted()
    let unexpected = actual.subtracting(expected).sorted()

    return (missing: missing, unexpected: unexpected)
  }

  /// Expected shapes for all weight keys.
  ///
  /// - Parameter config: Model configuration.
  /// - Returns: Dictionary mapping key names to expected shapes.
  public static func expectedShapes(config: WanUMT5Config = .wan22I2V) -> [String: [Int]] {
    let h = config.hiddenSize
    let f = config.ffnHiddenSize
    let v = config.vocabSize
    let nb = config.numBuckets
    let nh = config.numHeads

    var shapes: [String: [Int]] = [
      "token_embedding.weight": [v, h],
      "norm.weight": [h],
    ]

    for i in 0..<config.numLayers {
      shapes["blocks.\(i).attn.q.weight"] = [h, h]
      shapes["blocks.\(i).attn.k.weight"] = [h, h]
      shapes["blocks.\(i).attn.v.weight"] = [h, h]
      shapes["blocks.\(i).attn.o.weight"] = [h, h]
      shapes["blocks.\(i).pos_embedding.embedding.weight"] = [nb, nh]
      shapes["blocks.\(i).norm1.weight"] = [h]
      shapes["blocks.\(i).ffn.gate.0.weight"] = [f, h]
      shapes["blocks.\(i).ffn.fc1.weight"] = [f, h]
      shapes["blocks.\(i).ffn.fc2.weight"] = [h, f]
      shapes["blocks.\(i).norm2.weight"] = [h]
    }

    return shapes
  }
}
