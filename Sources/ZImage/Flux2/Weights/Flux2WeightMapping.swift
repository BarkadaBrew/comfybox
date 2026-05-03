// Flux2WeightMapping.swift — Maps safetensors key names to model parameter paths
// Ported from mflux: flux2_weight_mapping.py

import Foundation
import MLX
import MLXNN

/// Maps HuggingFace safetensors weight keys to the Flux2 Swift module parameter paths.
///
/// The Flux 2 Klein model stores weights using diffusers-convention keys in safetensors
/// files. This mapper translates those keys into the paths expected by the MLXNN module
/// hierarchy (as defined by `@ModuleInfo(key:)` annotations).
///
/// ## Component weight flows
///
/// **Transformer:** safetensors keys map directly — most Flux 2 transformer keys
/// already match the Swift module paths thanks to the `@ModuleInfo` keys matching
/// the diffusers naming convention.
///
/// **VAE:** Conv2d weights need a `[N,C,H,W] -> [N,H,W,C]` transpose because MLX
/// Conv2d expects `NHWC` layout while PyTorch stores `NCHW`.
///
/// **Text Encoder:** The `model.` prefix in safetensors keys maps to the `encoder.`
/// module prefix in the Swift hierarchy (matching the existing Z-Image convention).
public enum Flux2WeightMapping {

  // MARK: - Transformer

  /// Load and map transformer weights from safetensors files.
  ///
  /// - Parameters:
  ///   - files: URLs to transformer safetensors shards.
  ///   - dtype: Target dtype for weight conversion (default `.bfloat16`).
  /// - Returns: Mapped weights keyed by module parameter path.
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
        // Map diffusers key to module path
        let mappedKey = mapTransformerKey(meta.name)
        weights[mappedKey] = tensor
      }
    }
    return weights
  }

  /// Map a single HuggingFace transformer weight key to the Swift module path.
  ///
  /// Most Flux 2 transformer keys map directly because the `@ModuleInfo` keys
  /// match the diffusers convention. The only notable remapping is
  /// `attn.to_out.0.weight` -> `attn.to_out.weight` (PyTorch wraps the output
  /// projection in a `nn.Sequential`).
  static func mapTransformerKey(_ hfKey: String) -> String {
    // to_out.0.weight -> to_out.weight (unwrap Sequential wrapper)
    var key = hfKey
    key = key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    // time_guidance_embed.timestep_embedder.linear_X -> time_guidance_embed.linear_X
    key = key.replacingOccurrences(
      of: "time_guidance_embed.timestep_embedder.",
      with: "time_guidance_embed."
    )
    return key
  }

  // MARK: - VAE

  /// Load and map VAE weights from safetensors files.
  ///
  /// Conv2d weight tensors are transposed from PyTorch `NCHW` to MLX `NHWC` layout.
  ///
  /// - Parameters:
  ///   - files: URLs to VAE safetensors shards.
  ///   - dtype: Target dtype for weight conversion (default `.bfloat16`).
  /// - Returns: Mapped weights keyed by module parameter path.
  public static func loadVAEWeights(
    from files: [URL],
    dtype: DType = .bfloat16
  ) throws -> [String: MLXArray] {
    var weights: [String: MLXArray] = [:]
    for file in files {
      let reader = try SafeTensorsReader(fileURL: file)
      for meta in reader.allMetadata() {
        var tensor = try reader.tensor(named: meta.name)
        // Transpose conv2d weights: PyTorch NCHW -> MLX NHWC
        if meta.name.hasSuffix(".weight") && tensor.ndim == 4 && (meta.name.contains(".conv") || meta.name.contains("quant_conv") || meta.name.contains("post_quant_conv")) {
          tensor = tensor.transposed(0, 2, 3, 1)
        }
        if tensor.dtype != dtype {
          tensor = tensor.asType(dtype)
        }
        // Map diffusers key to module path
        let mappedKey = mapVAEKey(meta.name)
        weights[mappedKey] = tensor
      }
    }
    return weights
  }

  /// Map a single HuggingFace VAE weight key to the Swift module path.
  ///
  /// Most VAE keys map directly. The `to_out.0.` -> `to_out.` remapping
  /// handles the mid-block attention output projection Sequential wrapper.
  static func mapVAEKey(_ hfKey: String) -> String {
    var key = hfKey
    // to_out.0.weight -> to_out.weight (mid-block attention)
    key = key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
    return key
  }

  // MARK: - Text Encoder

  /// Load and map text encoder weights from safetensors files.
  ///
  /// HuggingFace keys use `model.` prefix which maps to the `encoder.` module
  /// prefix in the Swift Qwen3TextEncoder hierarchy.
  ///
  /// - Parameters:
  ///   - files: URLs to text encoder safetensors shards.
  ///   - dtype: Target dtype for weight conversion (default `.bfloat16`).
  /// - Returns: Mapped weights keyed by module parameter path.
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

  /// Map a single HuggingFace text encoder weight key to the Swift module path.
  ///
  /// The `model.` prefix is stripped (the MLXNN Qwen3TextEncoder uses top-level
  /// module keys like `embed_tokens`, `layers`, `norm`).
  static func mapTextEncoderKey(_ hfKey: String) -> String {
    if hfKey.hasPrefix("model.") {
      return String(hfKey.dropFirst("model.".count))
    }
    return hfKey
  }

  // MARK: - Weight Application

  /// Apply mapped weights to a Flux2 transformer module.
  ///
  /// Handles the MLXNN `Module.update(parameters:)` flow: flattens the weight
  /// dictionary, checks shapes, and applies via `ModuleParameters.unflattened`.
  ///
  /// - Parameters:
  ///   - weights: Mapped weight dictionary from `loadTransformerWeights`.
  ///   - transformer: The `Flux2Transformer` module to load weights into.
  public static func applyTransformerWeights(
    _ weights: [String: MLXArray],
    to transformer: Module
  ) throws {
    let params = ModuleParameters.unflattened(weights)
    try transformer.update(parameters: params, verify: [.shapeMismatch])
  }

  /// Apply mapped weights to a Flux2 VAE module.
  ///
  /// - Parameters:
  ///   - weights: Mapped weight dictionary from `loadVAEWeights`.
  ///   - vae: The `Flux2VAE` module to load weights into.
  public static func applyVAEWeights(
    _ weights: [String: MLXArray],
    to vae: Module
  ) throws {
    let params = ModuleParameters.unflattened(weights)
    try vae.update(parameters: params, verify: [.shapeMismatch])
  }

  /// Apply mapped weights to a Qwen3 text encoder module.
  ///
  /// - Parameters:
  ///   - weights: Mapped weight dictionary from `loadTextEncoderWeights`.
  ///   - textEncoder: The `Qwen3TextEncoder` module to load weights into.
  public static func applyTextEncoderWeights(
    _ weights: [String: MLXArray],
    to textEncoder: Module
  ) throws {
    let params = ModuleParameters.unflattened(weights)
    try textEncoder.update(parameters: params, verify: [.shapeMismatch])
  }
}
