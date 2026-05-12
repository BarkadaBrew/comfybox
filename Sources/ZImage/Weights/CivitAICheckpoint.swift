// CivitAICheckpoint.swift — Detection and loading for CivitAI transformer-only checkpoints
//
// CivitAI community checkpoints store Z-Image transformer weights in a single
// .safetensors file with a `model.diffusion_model.` prefix. Unlike AIO checkpoints,
// they contain only diffusion model weights (no text encoder, no VAE). Text encoder
// and VAE are loaded from the standard HuggingFace snapshot.
//
// Issue: #139

import Foundation
import MLX
import Logging

public enum CivitAICheckpoint {

  // MARK: - Types

  public struct Inspection: Sendable {
    /// True if this is a CivitAI transformer-only checkpoint.
    public let isCivitAI: Bool
    /// Detected variant (Base vs Turbo), nil if detection failed.
    public let variant: ZImageVariant?
    /// Diagnostic messages for debugging.
    public let diagnostics: [String]
    /// Number of diffusion model keys found.
    public let keyCount: Int
  }

  // MARK: - Detection

  /// Inspect a safetensors file to determine if it is a CivitAI transformer-only checkpoint.
  ///
  /// A CivitAI checkpoint is identified by:
  /// 1. All (or nearly all) keys have the `model.diffusion_model.` prefix
  /// 2. No `text_encoders.*` keys present
  /// 3. No `vae.*` keys present
  /// 4. Contains `model.diffusion_model.layers.0.attention.qkv.weight` (fused QKV)
  ///
  /// This distinguishes it from:
  /// - AIO checkpoints (have text_encoders.* and vae.*)
  /// - HuggingFace format (directory with sharded files, no prefix)
  /// - Transformer-only overrides (no prefix, already in internal format)
  public static func inspect(fileURL: URL) -> Inspection {
    do {
      let reader = try SafeTensorsReader(fileURL: fileURL)
      return inspect(reader: reader)
    } catch {
      return Inspection(isCivitAI: false, variant: nil, diagnostics: ["failed to read: \(error)"], keyCount: 0)
    }
  }

  public static func inspect(reader: SafeTensorsReader) -> Inspection {
    let names = reader.tensorNames
    let nameSet = Set(names)

    // Count keys by prefix
    let diffusionKeys = names.filter { $0.hasPrefix("model.diffusion_model.") }
    let hasTextEncoder = names.contains { $0.hasPrefix("text_encoders.") }
    let hasVAE = names.contains { $0.hasPrefix("vae.") }

    // If it has text encoder or VAE, it is an AIO checkpoint, not CivitAI-transformer-only
    if hasTextEncoder || hasVAE {
      return Inspection(isCivitAI: false, variant: nil,
        diagnostics: ["has text_encoder or VAE keys — use AIO path"], keyCount: diffusionKeys.count)
    }

    // Must have diffusion model keys (Base ~453, Turbo ~201)
    guard diffusionKeys.count >= 150 else {
      return Inspection(isCivitAI: false, variant: nil,
        diagnostics: ["only \(diffusionKeys.count) diffusion keys (expected >= 400)"], keyCount: diffusionKeys.count)
    }

    // Must have fused QKV (the signature of CivitAI format vs already-canonicalized)
    let hasFusedQKV = nameSet.contains("model.diffusion_model.layers.0.attention.qkv.weight")
    guard hasFusedQKV else {
      return Inspection(isCivitAI: false, variant: nil,
        diagnostics: ["no fused QKV keys found"], keyCount: diffusionKeys.count)
    }

    // Detect variant
    let variant = detectVariant(names: nameSet)

    return Inspection(isCivitAI: true, variant: variant, diagnostics: [], keyCount: diffusionKeys.count)
  }

  // MARK: - Variant Detection

  /// Detect whether a CivitAI checkpoint is Base or Turbo.
  ///
  /// Base Z-Image uses `t_embedder` (time embedder) and has `noise_refiner`,
  /// `context_refiner`, `cap_embedder` modules (~453 keys total).
  /// Turbo Z-Image uses `time_in` and has a smaller architecture (~201 keys).
  ///
  /// The raw CivitAI keys retain the `model.diffusion_model.` prefix.
  static func detectVariant(names: Set<String>) -> ZImageVariant {
    // Primary signal: Base has t_embedder and noise_refiner; Turbo does not
    let baseSignalKeys = [
      "model.diffusion_model.t_embedder.mlp.0.weight",
      "model.diffusion_model.noise_refiner.0.attention.qkv.weight",
      "model.diffusion_model.context_refiner.0.attention.qkv.weight",
      "model.diffusion_model.cap_embedder.0.weight",
    ]
    if baseSignalKeys.contains(where: { names.contains($0) }) {
      return .base
    }

    // Secondary signal: Turbo has time_in (not t_embedder)
    let turboSignalKeys = [
      "model.diffusion_model.time_in.in_layer.weight",
      "model.diffusion_model.time_in.out_layer.weight",
    ]
    if turboSignalKeys.contains(where: { names.contains($0) }) {
      return .turbo
    }

    // Tertiary signal: key count — Base has ~453, Turbo has ~201
    if names.count >= 400 {
      return .base
    }

    // Default to Turbo for unrecognized small checkpoints
    return .turbo
  }

  // MARK: - Loading

  /// Load transformer weights from a CivitAI checkpoint.
  ///
  /// Steps:
  /// 1. Read all tensors with `model.diffusion_model.` prefix
  /// 2. Handle FP8/FP16/FP32 -> BF16 conversion
  /// 3. Return raw dict (caller applies canonicalizeTransformerOverride)
  ///
  /// The returned keys still have the `model.diffusion_model.` prefix.
  /// The caller (canonicalizeTransformerOverride) strips it.
  public static func loadTransformerWeights(
    from fileURL: URL,
    dtype: DType = .bfloat16,
    logger: Logger? = nil
  ) throws -> [String: MLXArray] {
    let reader = try SafeTensorsReader(fileURL: fileURL)
    var weights: [String: MLXArray] = [:]
    weights.reserveCapacity(reader.tensorNames.count)

    var fp8Count = 0
    var castCount = 0

    for meta in reader.allMetadata() {
      guard meta.name.hasPrefix("model.diffusion_model.") else { continue }

      var tensor = try reader.tensor(named: meta.name)

      // Handle FP8 (float8_e4m3fn) — SafeTensorsReader maps to .uint8
      if isFP8Tensor(meta) {
        tensor = decodeFP8(tensor)
        fp8Count += 1
      }

      // Cast to target dtype if needed
      if tensor.dtype != dtype {
        tensor = tensor.asType(dtype)
        castCount += 1
      }

      weights[meta.name] = tensor
    }

    logger?.info("CivitAI: loaded \(weights.count) transformer tensors (fp8_decoded=\(fp8Count), dtype_cast=\(castCount))")
    return weights
  }

  // MARK: - FP8 Support

  /// Check if a tensor metadata entry represents FP8 data.
  ///
  /// CivitAI FP8 checkpoints use dtype "F8_E4M3" or "F8_E4M3FN" in the safetensors header.
  /// SafeTensorsReader maps these to .uint8 and preserves the original dtype string in rawDType.
  private static func isFP8Tensor(_ meta: SafeTensorMetadata) -> Bool {
    let raw = meta.rawDType.uppercased()
    return raw == "F8_E4M3" || raw == "F8_E4M3FN" || raw == "F8_E5M2"
  }

  /// Decode FP8 (float8_e4m3fn) tensor to float32, then let caller cast to target dtype.
  ///
  /// FP8 E4M3FN: 1 sign bit, 4 exponent bits, 3 mantissa bits.
  /// Bias = 7, no infinity representation, NaN = 0x7F.
  ///
  /// The decode is done via integer arithmetic on the uint8 raw bytes:
  ///   value = (-1)^sign * 2^(exponent - 7) * (1 + mantissa/8)   [normal]
  ///   value = (-1)^sign * 2^(-6) * (mantissa/8)                  [subnormal, exp=0]
  ///   value = NaN                                                 [exp=15, mantissa=7]
  private static func decodeFP8(_ tensor: MLXArray) -> MLXArray {
    // Promote to int32 for bit manipulation
    let u = tensor.asType(.int32)

    let sign = (u >> 7) & 1
    let exponent = (u >> 3) & 0xF
    let mantissa = u & 0x7

    // Build float32 values
    let signF = MLX.where(sign .== 0, MLXArray(Float(1.0)), MLXArray(Float(-1.0)))

    // Normal: exp > 0 => value = sign * 2^(exp-7) * (1 + mantissa/8)
    // Subnormal: exp == 0 => value = sign * 2^(-6) * (mantissa/8)
    // NaN: exp==15 && mantissa==7 => 0 (treat as zero for practical use)

    let mantissaF = mantissa.asType(.float32) / MLXArray(Float(8.0))
    let expF = exponent.asType(.float32)

    // 2^(exp - 7) for normal, 2^(-6) for subnormal
    let isSubnormal = exponent .== 0
    let isNaN = (exponent .== 15) & (mantissa .== 7)

    let normalPow = MLX.pow(MLXArray(Float(2.0)), expF - MLXArray(Float(7.0)))
    let normalVal = signF * normalPow * (MLXArray(Float(1.0)) + mantissaF)

    let subnormalVal = signF * MLXArray(Float(1.0 / 64.0)) * mantissaF

    var result = MLX.where(isSubnormal, subnormalVal, normalVal)
    result = MLX.where(isNaN, MLXArray(Float(0.0)), result)

    return result
  }
}
