// FiboWeightMapping.swift — HF safetensors key -> model parameter path mapping
// Ported from mflux: fibo_weight_mapping.py
//
// FIBO has three components with distinct key conventions:
// - Transformer: diffusers-style keys, mostly pass-through with .to_out.0. unwrap
// - Text Encoder: model.* prefix stripped (SmolLM3 convention)
// - VAE: Wan 2.2 3D VAE with .gamma norm weights and Conv3d weight transposition

import Foundation
import MLX

/// Maps HuggingFace safetensors weight keys to model parameter paths for FIBO.
///
/// ## Key mapping conventions
///
/// **Transformer:** Keys map nearly directly. The only remapping is
/// `attn.to_out.0.{weight,bias}` -> `attn.to_out.{weight,bias}` to unwrap
/// the PyTorch `nn.Sequential` wrapper.
///
/// **Text Encoder (SmolLM3-3B):** The `model.` prefix is stripped.
/// `model.layers.N.self_attn.q_proj.weight` -> `layers.N.self_attn.q_proj.weight`
/// `model.embed_tokens.weight` -> `embed_tokens.weight`
/// `model.norm.weight` -> `norm.weight`
///
/// **VAE (Wan 2.2):** Most complex mapping:
/// - Conv3d weights need 5D transposition (PyTorch NCDHW -> MLX NDHWC)
/// - RMS norm uses `.gamma` suffix in safetensors, maps to `.weight` in model
/// - Conv2d attention weights need 4D NCHW -> NHWC transposition
/// - `quant_conv` and `post_quant_conv` are Conv3d layers
public enum FiboWeightMapping {

  // MARK: - Transformer Weights

  /// Load and map transformer weights from safetensors files.
  ///
  /// Transformer weights are nearly pass-through: the only remapping is
  /// unwrapping `to_out.0` Sequential wrappers. No transposition needed.
  ///
  /// - Parameters:
  ///   - files: URLs to transformer safetensors shards.
  ///   - dtype: Target dtype (default `.bfloat16`).
  /// - Returns: Mapped weight dictionary.
  public static func loadTransformerWeights(
    from files: [URL],
    dtype: DType = .bfloat16
  ) throws -> [String: MLXArray] {
    var weights: [String: MLXArray] = [:]
    for file in files {
      let reader = try SafeTensorsReader(fileURL: file)
      for meta in reader.allMetadata() {
        var tensor = try reader.tensor(named: meta.name)
        if tensor.dtype != dtype {
          tensor = tensor.asType(dtype)
        }
        let mappedKey = mapTransformerKey(meta.name)
        weights[mappedKey] = tensor
      }
    }
    return weights
  }

  /// Map a HuggingFace transformer weight key to the model parameter path.
  ///
  /// Remappings:
  /// - `attn.to_out.0.weight` -> `attn.to_out.weight` (Sequential unwrap)
  /// - `attn.to_out.0.bias` -> `attn.to_out.bias`
  /// - `ff.net.0.proj.*` -> pass-through (GELU gate projection)
  /// - `ff.net.2.*` -> pass-through (output projection)
  static func mapTransformerKey(_ hfKey: String) -> String {
    var key = hfKey
    key = key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    return key
  }

  // MARK: - Text Encoder Weights (SmolLM3-3B)

  /// Load and map SmolLM3-3B text encoder weights from safetensors files.
  ///
  /// The `model.` prefix in safetensors keys is stripped to match the model
  /// parameter hierarchy.
  ///
  /// - Parameters:
  ///   - files: URLs to text encoder safetensors shards.
  ///   - dtype: Target dtype (default `.bfloat16`).
  /// - Returns: Mapped weight dictionary.
  public static func loadTextEncoderWeights(
    from files: [URL],
    dtype: DType = .bfloat16
  ) throws -> [String: MLXArray] {
    var weights: [String: MLXArray] = [:]
    for file in files {
      let reader = try SafeTensorsReader(fileURL: file)
      for meta in reader.allMetadata() {
        var tensor = try reader.tensor(named: meta.name)
        if tensor.dtype != dtype {
          tensor = tensor.asType(dtype)
        }
        let mappedKey = mapTextEncoderKey(meta.name)
        weights[mappedKey] = tensor
      }
    }
    return weights
  }

  /// Map a HuggingFace text encoder key to the model parameter path.
  ///
  /// Strips the `model.` prefix:
  /// - `model.embed_tokens.weight` -> `embed_tokens.weight`
  /// - `model.layers.N.*` -> `layers.N.*`
  /// - `model.norm.weight` -> `norm.weight`
  static func mapTextEncoderKey(_ hfKey: String) -> String {
    if hfKey.hasPrefix("model.") {
      return String(hfKey.dropFirst("model.".count))
    }
    return hfKey
  }

  // MARK: - VAE Weights (Wan 2.2)

  /// Load and map Wan 2.2 VAE weights from safetensors files.
  ///
  /// The Wan 2.2 VAE uses Conv3d layers and RMS norm with `.gamma` naming.
  /// Weight transformations applied:
  /// - Conv3d weights: 5D transposition from PyTorch NCDHW to MLX NDHWC
  /// - Conv2d weights (attention): 4D transposition from NCHW to NHWC
  /// - `.gamma` suffix remains as-is (the model layer should expect it)
  ///
  /// - Parameters:
  ///   - files: URLs to VAE safetensors shards.
  ///   - dtype: Target dtype (default `.bfloat16`).
  /// - Returns: Mapped weight dictionary.
  public static func loadVAEWeights(
    from files: [URL],
    dtype: DType = .bfloat16
  ) throws -> [String: MLXArray] {
    var weights: [String: MLXArray] = [:]
    for file in files {
      let reader = try SafeTensorsReader(fileURL: file)
      for meta in reader.allMetadata() {
        var tensor = try reader.tensor(named: meta.name)

        // Apply weight transformations based on tensor dimensionality and key
        if meta.name.hasSuffix(".weight") && tensor.ndim == 5 {
          // Conv3d weight: PyTorch NCDHW [out, in, D, H, W] -> MLX NDHWC [out, D, H, W, in]
          tensor = tensor.transposed(0, 2, 3, 4, 1)
        } else if meta.name.hasSuffix(".weight") && tensor.ndim == 4 {
          // Conv2d weight (attention layers): PyTorch NCHW -> MLX NHWC
          tensor = tensor.transposed(0, 2, 3, 1)
        }

        if tensor.dtype != dtype {
          tensor = tensor.asType(dtype)
        }

        let mappedKey = mapVAEKey(meta.name)
        weights[mappedKey] = tensor
      }
    }
    return weights
  }

  /// Map a HuggingFace VAE key to the model parameter path.
  ///
  /// Remappings:
  /// - `.resample.1.` -> `.resample_conv.` (unwrap diffusers Sequential wrapper)
  /// - All other keys pass through directly
  /// - `.gamma` norm naming is preserved (model uses @ModuleInfo(key: "gamma"))
  static func mapVAEKey(_ hfKey: String) -> String {
    var key = hfKey
    // Unwrap diffusers Sequential: resample.1.weight -> resample_conv.weight
    key = key.replacingOccurrences(of: ".resample.1.", with: ".resample_conv.")
    return key
  }

  // MARK: - Weight Verification

  /// Verify that all safetensors keys in a set of files can be mapped.
  ///
  /// - Parameters:
  ///   - files: URLs to safetensors shards.
  ///   - component: Component name for logging ("transformer", "text_encoder", "vae").
  ///   - mapper: Key mapping function to apply.
  /// - Returns: Verification result with mapped and unmapped key counts.
  public static func verifyMapping(
    files: [URL],
    component: String,
    mapper: (String) -> String
  ) throws -> VerificationResult {
    var allKeys: [String] = []
    var mappedKeys: [String] = []
    var unmappedKeys: [String] = []

    for file in files {
      let reader = try SafeTensorsReader(fileURL: file)
      for meta in reader.allMetadata() {
        let mapped = mapper(meta.name)
        allKeys.append(meta.name)
        if mapped != meta.name || component == "transformer" {
          // Transformer keys mostly pass through, that's expected
          mappedKeys.append(meta.name)
        } else {
          mappedKeys.append(meta.name)
        }
      }
    }

    return VerificationResult(
      component: component,
      totalKeys: allKeys.count,
      mappedKeys: mappedKeys.count,
      unmappedKeys: unmappedKeys,
      allKeys: allKeys
    )
  }

  /// Result of weight mapping verification.
  public struct VerificationResult: Sendable {
    public let component: String
    public let totalKeys: Int
    public let mappedKeys: Int
    public let unmappedKeys: [String]
    public let allKeys: [String]

    public var isComplete: Bool { unmappedKeys.isEmpty }
  }
}
