// Krea2WeightLoader.swift — Loads Krea-2 weights into the Swift modules.
//
// Three components, three transforms:
//  - Transformer (`turbo.safetensors`, 430 tensors): keys already match the
//    Krea2SingleStreamDiT @ModuleInfo tree 1:1 — pass-through, cast to bf16.
//  - Text encoder (`text_encoder/model.safetensors`): strip the `language_model.`
//    prefix (keys then match Qwen3TextEncoder: embed_tokens/layers/norm) and skip
//    the unused `visual.*` vision tower.
//  - VAE (`vae/diffusion_pytorch_model.safetensors`): decoder-only. 3D causal conv
//    kernels [O,I,kT,kH,kW] reduce to their LAST temporal slice for images (see
//    Krea2VAE.swift) then transpose NCHW→NHWC ([O,kH,kW,I]); norm `gamma` tensors
//    flatten to [C]; encoder/quant_conv/time_conv weights are skipped.

import Foundation
import MLX
import MLXNN

public enum Krea2WeightLoaderError: Error, CustomStringConvertible {
  case missingFile(String)
  case unexpectedShape(key: String, shape: [Int])

  public var description: String {
    switch self {
    case .missingFile(let p): return "Krea2 weights: missing file \(p)"
    case .unexpectedShape(let k, let s): return "Krea2 weights: unexpected shape \(s) for \(k)"
    }
  }
}

public enum Krea2WeightLoader {

  // MARK: - Transformer

  /// Load `turbo.safetensors` into the SingleStreamDiT. Keys map 1:1.
  public static func loadTransformer(
    _ transformer: Krea2SingleStreamDiT, from file: URL, dtype: DType = .bfloat16
  ) throws {
    guard FileManager.default.fileExists(atPath: file.path) else {
      throw Krea2WeightLoaderError.missingFile(file.path)
    }
    let reader = try SafeTensorsReader(fileURL: file)
    var weights: [String: MLXArray] = [:]
    for meta in reader.allMetadata() {
      var tensor = try reader.tensor(named: meta.name)
      if tensor.dtype != dtype { tensor = tensor.asType(dtype) }
      weights[Self.mapTransformerKey(meta.name)] = tensor
    }
    try transformer.update(parameters: ModuleParameters.unflattened(weights), verify: [.shapeMismatch])
  }

  /// Remap numeric sequential indices (which MLX-Swift unflattens as array
  /// indices) to the named keys used by the Swift modules.
  static func mapTransformerKey(_ key: String) -> String {
    if key.hasPrefix("tmlp.0.") { return "tmlp.lin0." + key.dropFirst("tmlp.0.".count) }
    if key.hasPrefix("tmlp.2.") { return "tmlp.lin2." + key.dropFirst("tmlp.2.".count) }
    if key.hasPrefix("tproj.1.") { return "tproj.lin1." + key.dropFirst("tproj.1.".count) }
    if key.hasPrefix("txtmlp.0.") { return "txtmlp.norm0." + key.dropFirst("txtmlp.0.".count) }
    if key.hasPrefix("txtmlp.1.") { return "txtmlp.lin1." + key.dropFirst("txtmlp.1.".count) }
    if key.hasPrefix("txtmlp.3.") { return "txtmlp.lin3." + key.dropFirst("txtmlp.3.".count) }
    return key
  }

  // MARK: - Text encoder

  /// Load `text_encoder/model.safetensors` into the (Flux2-shared) Qwen3TextEncoder.
  public static func loadTextEncoder(
    _ encoder: Qwen3TextEncoder, from file: URL, dtype: DType = .bfloat16
  ) throws {
    guard FileManager.default.fileExists(atPath: file.path) else {
      throw Krea2WeightLoaderError.missingFile(file.path)
    }
    let prefix = "language_model."
    let reader = try SafeTensorsReader(fileURL: file)
    var weights: [String: MLXArray] = [:]
    for meta in reader.allMetadata() {
      guard meta.name.hasPrefix(prefix) else { continue }  // skip visual.* tower
      var tensor = try reader.tensor(named: meta.name)
      if tensor.dtype != dtype { tensor = tensor.asType(dtype) }
      weights[String(meta.name.dropFirst(prefix.count))] = tensor
    }
    try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.shapeMismatch])
  }

  // MARK: - VAE (decoder-only)

  /// Load `vae/diffusion_pytorch_model.safetensors` (decoder path only).
  public static func loadVAE(
    _ vae: Krea2VAE, from file: URL, dtype: DType = .float32
  ) throws {
    guard FileManager.default.fileExists(atPath: file.path) else {
      throw Krea2WeightLoaderError.missingFile(file.path)
    }
    let reader = try SafeTensorsReader(fileURL: file)
    var weights: [String: MLXArray] = [:]
    for meta in reader.allMetadata() {
      let key = meta.name
      // Decoder-only: skip the encoder, its quant conv, and unused temporal convs.
      if key.hasPrefix("encoder.") || key.hasPrefix("quant_conv") || key.contains(".time_conv.") {
        continue
      }
      var tensor = try reader.tensor(named: key).asType(dtype)

      if key.hasSuffix(".gamma") {
        // [C,1,1] / [C,1,1,1] -> [C]
        tensor = tensor.reshaped([tensor.dim(0)])
      } else if key.hasSuffix(".weight") && tensor.ndim == 5 {
        // Causal 3D conv [O,I,kT,kH,kW] -> last temporal slice -> NHWC [O,kH,kW,I]
        let kT = tensor.dim(2)
        tensor = tensor[0..., 0..., kT - 1, 0..., 0...].transposed(0, 2, 3, 1)
      } else if key.hasSuffix(".weight") && tensor.ndim == 4 {
        // 2D conv (attention qkv/proj) [O,I,kH,kW] -> [O,kH,kW,I]
        tensor = tensor.transposed(0, 2, 3, 1)
      } else if key.hasSuffix(".weight") && tensor.ndim > 2 {
        throw Krea2WeightLoaderError.unexpectedShape(key: key, shape: tensor.shape)
      }
      weights[key.replacingOccurrences(of: ".resample.1.", with: ".resample.conv.")] = tensor
    }
    try vae.update(parameters: ModuleParameters.unflattened(weights), verify: [.shapeMismatch])
  }
}
