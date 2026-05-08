import Foundation
import MLX

/// Weight key sanitization for Chroma model weights.
///
/// Maps from PyTorch/safetensors naming conventions to Swift module paths.
/// The Chroma weights use `model.diffusion_model.` prefix which is stripped first,
/// then internal key patterns are adjusted for Swift MLXNN module paths.
public enum ChromaWeightMapping {

  /// Sanitize Chroma transformer weight keys for Swift module loading.
  ///
  /// Input keys look like: `model.diffusion_model.double_blocks.0.img_attn.qkv.weight`
  /// Output keys look like: `double_blocks.0.img_attn.qkv.weight`
  ///
  /// Also handles the `.scale` → `.weight` rename for RMSNorm layers.
  public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (key, value) in weights {
      var k = key

      // Strip common prefixes
      if k.hasPrefix("model.diffusion_model.") {
        k = String(k.dropFirst("model.diffusion_model.".count))
      }

      // RMSNorm: .scale -> .weight (MLX convention)
      if k.hasSuffix(".scale") {
        k = String(k.dropLast(6)) + ".weight"
      }

      // MLP layers: img_mlp.0 -> img_mlp.layers.0 (Sequential wrapper)
      for seq in ["img_mlp", "txt_mlp"] {
        if k.contains(".\(seq).") {
          k = k.replacingOccurrences(of: ".\(seq).", with: ".\(seq).layers.")
        }
      }

      // Clean up double .layers.layers.
      while k.contains(".layers.layers.") {
        k = k.replacingOccurrences(of: ".layers.layers.", with: ".layers.")
      }

      result[k] = value
    }
    return result
  }

  /// Load and sanitize weights from a safetensors file.
  ///
  /// - Parameter path: Path to the `.safetensors` file
  /// - Returns: Sanitized weight dictionary
  public static func loadWeights(from path: String) throws -> [String: MLXArray] {
    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: path))
    return sanitize(raw)
  }
}
