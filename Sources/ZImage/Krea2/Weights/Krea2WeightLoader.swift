// Krea2WeightLoader.swift — Loads Krea-2 weights into the Swift modules.
//
// Three components, three transforms:
//  - Transformer (`turbo.safetensors`, 430 tensors): keys already match the
//    Krea2SingleStreamDiT @ModuleInfo tree 1:1 — pass-through, cast to bf16.
//  - Text encoder (`text_encoder/model.safetensors`): strip the `language_model.`
//    prefix (keys then match Qwen3TextEncoder: embed_tokens/layers/norm) and skip
//    the unused `visual.*` vision tower.
//  - VAE (the model dir's `vae/diffusion_pytorch_model.safetensors`, or any
//    file `payload.vae` names — WP-E9): encoder + decoder. The key layout is
//    sniffed (Qwen-diffusers or Wan-native, Krea2VAEKeyMap) and canonicalised
//    first; then 3D causal conv kernels [O,I,kT,kH,kW] reduce to their LAST
//    temporal slice for images (see Krea2VAE.swift) and transpose NCHW→NHWC
//    ([O,kH,kW,I]); norm `gamma` tensors flatten to [C]; `time_conv` weights
//    are skipped (dead code in the reference forward pass — see
//    Krea2VAEDownsampler's doc comment).

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

  // MARK: - VAE

  /// Load a Krea-2 VAE file (encoder + decoder) into `vae`, in place.
  ///
  /// WP-E9: the file may be the Qwen-Image diffusers VAE (the model dir's
  /// `vae/diffusion_pytorch_model.safetensors`) or the Wan 2.1 FP32 file; the
  /// layout is sniffed from the keys (never the filename) and every Wan key is
  /// canonicalised onto its Krea2VAE path BEFORE the existing 5-D slice /
  /// NHWC transpose / gamma flatten. `layout:` is an optional declaration — a
  /// declared layout that contradicts the keys is `layoutMismatch`.
  ///
  /// Fail-closed and atomic: every key is mapped, transformed and checked
  /// against the module's own parameter set (presence AND shape) before the
  /// first weight is written, so a failure leaves `vae` exactly as it was.
  /// Returns the layout that loaded.
  @discardableResult
  public static func loadVAE(
    _ vae: Krea2VAE, from file: URL, layout: VAELayout? = nil, dtype: DType = .float32
  ) throws -> VAELayout {
    let (weights, used) = try prepareVAEWeights(from: file, layout: layout, dtype: dtype)
    // Preflight against the resident parameter set: an unknown path or a
    // shape mismatch throws here, before `update` mutates anything (MLX's
    // `update` applies progressively — a mid-walk throw would leave a
    // half-swapped decoder).
    let resident = Dictionary(uniqueKeysWithValues: vae.parameters().flattened())
    for (path, tensor) in weights {
      guard let param = resident[path] else {
        throw Krea2VAEKeyMapError.unmappedKey(file: file.path, key: path)
      }
      guard param.shape == tensor.shape else {
        throw Krea2WeightLoaderError.unexpectedShape(key: path, shape: tensor.shape)
      }
    }
    try vae.update(parameters: ModuleParameters.unflattened(weights), verify: [.shapeMismatch])
    return used
  }

  /// Read, canonicalise and transform every tensor of a VAE file into the
  /// Krea2VAE module-path → array dictionary `update` takes. Pure with
  /// respect to any module: nothing is mutated.
  static func prepareVAEWeights(
    from file: URL, layout: VAELayout?, dtype: DType
  ) throws -> (weights: [String: MLXArray], layout: VAELayout) {
    guard FileManager.default.fileExists(atPath: file.path) else {
      throw Krea2WeightLoaderError.missingFile(file.path)
    }
    let reader = try SafeTensorsReader(fileURL: file)
    let detected = try Krea2VAEKeyMap.detectLayout(keys: reader.tensorNames, file: file)
    if let layout, layout != detected {
      throw Krea2VAEKeyMapError.layoutMismatch(file: file.path, requested: layout, detected: detected)
    }
    var weights: [String: MLXArray] = [:]
    for meta in reader.allMetadata() {
      guard let key = Krea2VAEKeyMap.canonicalize(meta.name) else {
        throw Krea2VAEKeyMapError.unmappedKey(file: file.path, key: meta.name)
      }
      // Unused temporal convs (see Krea2VAEDownsampler's doc comment).
      if key.contains(".time_conv.") {
        continue
      }
      var tensor = try reader.tensor(named: meta.name).asType(dtype)

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
      weights[vaeModulePath(forCanonicalKey: key)] = tensor
    }
    return (weights, detected)
  }

  /// Canonical (Qwen-diffusers) key → Krea2VAE module path. The checkpoint
  /// stores the resample conv at a numeric Sequential index, which MLX-Swift
  /// would unflatten as an array index; the module names it `resample.conv`.
  static func vaeModulePath(forCanonicalKey key: String) -> String {
    key.replacingOccurrences(of: ".resample.1.", with: ".resample.conv.")
  }
}
