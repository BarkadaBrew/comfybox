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
      let reader = try SafeTensorsReader(fileURL: fileURL, dtypeMapper: mapCivitAIDType)
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
    let variantSignal = detectVariantSignal(names: nameSet)

    // If it has text encoder or VAE, it is an AIO checkpoint, not CivitAI-transformer-only
    if hasTextEncoder || hasVAE {
      return Inspection(isCivitAI: false, variant: nil,
        diagnostics: ["has text_encoder or VAE keys — use AIO path"], keyCount: diffusionKeys.count)
    }

    // Must have diffusion model keys (Base ~453, Turbo ~201)
    guard diffusionKeys.count >= 150 else {
      return Inspection(isCivitAI: false, variant: variantSignal,
        diagnostics: ["only \(diffusionKeys.count) diffusion keys (expected >= 150)"], keyCount: diffusionKeys.count)
    }

    // Must have fused QKV (the signature of CivitAI format vs already-canonicalized)
    let hasFusedQKV = nameSet.contains("model.diffusion_model.layers.0.attention.qkv.weight")
    guard hasFusedQKV else {
      return Inspection(isCivitAI: false, variant: variantSignal,
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
    if hasBaseVariantSignal(names: names) {
      return .base
    }

    // Secondary signal: Turbo has time_in (not t_embedder)
    if hasTurboVariantSignal(names: names) {
      return .turbo
    }

    // Tertiary signal: key count — Base has ~453, Turbo has ~201
    if names.count >= 400 {
      return .base
    }

    // Default to Turbo for unrecognized small checkpoints
    return .turbo
  }

  private static func detectVariantSignal(names: Set<String>) -> ZImageVariant? {
    if hasBaseVariantSignal(names: names) {
      return .base
    }
    if hasTurboVariantSignal(names: names) {
      return .turbo
    }
    return nil
  }

  private static func hasBaseVariantSignal(names: Set<String>) -> Bool {
    containsAnyVariantKey(
      [
        "t_embedder.mlp.0.weight",
        "noise_refiner.0.attention.qkv.weight",
        "context_refiner.0.attention.qkv.weight",
        "cap_embedder.0.weight",
      ],
      in: names
    )
  }

  private static func hasTurboVariantSignal(names: Set<String>) -> Bool {
    containsAnyVariantKey(
      [
        "time_in.in_layer.weight",
        "time_in.out_layer.weight",
      ],
      in: names
    )
  }

  private static func containsAnyVariantKey(_ suffixes: [String], in names: Set<String>) -> Bool {
    let prefixes = ["", "model.diffusion_model.", "diffusion_model.", "model."]
    for suffix in suffixes {
      for prefix in prefixes where names.contains("\(prefix)\(suffix)") {
        return true
      }
    }
    return false
  }

  // MARK: - Loading

  /// Load transformer weights from a CivitAI checkpoint.
  ///
  /// Steps:
  /// 1. Read all tensors with `model.diffusion_model.` prefix
  /// 2. Handle FP8/FP16/FP32 -> target dtype conversion
  /// 3. Return raw dict (caller applies canonicalizeTransformerOverride)
  ///
  /// The returned keys still have the `model.diffusion_model.` prefix.
  /// The caller (canonicalizeTransformerOverride) strips it.
  public static func loadTransformerWeights(
    from fileURL: URL,
    dtype: DType = .bfloat16,
    logger: Logger? = nil
  ) throws -> [String: MLXArray] {
    let reader = try SafeTensorsReader(fileURL: fileURL, dtypeMapper: mapCivitAIDType)
    var weights: [String: MLXArray] = [:]
    weights.reserveCapacity(reader.tensorNames.count)

    var fp8Count = 0
    var castCount = 0

    for meta in reader.allMetadata() {
      guard meta.name.hasPrefix("model.diffusion_model.") else { continue }

      var tensor = try reader.tensor(named: meta.name)

      // Handle FP8 (float8_e4m3fn / float8_e5m2) from CivitAI checkpoints.
      if let format = fp8Format(meta) {
        // Decode through float32 to apply the FP8 exponent/mantissa rules.
        // Casting that result to BF16 is exact for FP8 source values because
        // BF16 has more mantissa bits than either supported FP8 format.
        tensor = decodeFP8(tensor, format: format)
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

  private enum FP8Format {
    case e4m3fn  // 4 exponent bits, 3 mantissa bits, bias 7
    case e5m2    // 5 exponent bits, 2 mantissa bits, bias 15
  }

  /// Check if a tensor metadata entry represents FP8 data, and which format.
  ///
  /// CivitAI FP8 checkpoints use dtype "F8_E4M3" or "F8_E4M3FN" or "F8_E5M2"
  /// in the safetensors header. This loader reads them as raw uint8 bytes and
  /// preserves the original dtype string in rawDType for explicit decoding here.
  private static func fp8Format(_ meta: SafeTensorMetadata) -> FP8Format? {
    let raw = meta.rawDType.uppercased()
    switch raw {
    case "F8_E4M3", "F8_E4M3FN": return .e4m3fn
    case "F8_E5M2": return .e5m2
    default: return nil
    }
  }

  private static func isFP8Tensor(_ meta: SafeTensorMetadata) -> Bool {
    return fp8Format(meta) != nil
  }

  private static func mapCivitAIDType(_ value: String) throws -> DType {
    let raw = value.uppercased()
    switch raw {
    case "F8_E4M3", "F8_E4M3FN", "F8_E5M2":
      return .uint8
    default:
      return try SafeTensorsReader.mapDType(value)
    }
  }

  /// Decode FP8 E4M3FN tensor to float32.
  ///
  /// FP8 E4M3FN: 1 sign bit, 4 exponent bits, 3 mantissa bits.
  /// Bias = 7, no infinity representation, NaN = 0x7F.
  ///
  ///   value = (-1)^sign * 2^(exponent - 7) * (1 + mantissa/8)   [normal]
  ///   value = (-1)^sign * 2^(-6) * (mantissa/8)                  [subnormal, exp=0]
  ///   value = NaN                                                 [exp=15, mantissa=7]
  private static func decodeFP8E4M3(_ tensor: MLXArray) -> MLXArray {
    let u = tensor.asType(.int32)

    let sign = (u >> 7) & 1
    let exponent = (u >> 3) & 0xF
    let mantissa = u & 0x7

    let signF = MLX.where(sign .== 0, MLXArray(Float(1.0)), MLXArray(Float(-1.0)))

    let mantissaF = mantissa.asType(.float32) / MLXArray(Float(8.0))
    let expF = exponent.asType(.float32)

    let isSubnormal = exponent .== 0
    let isNaN = (exponent .== 15) & (mantissa .== 7)

    let normalPow = MLX.pow(MLXArray(Float(2.0)), expF - MLXArray(Float(7.0)))
    let normalVal = signF * normalPow * (MLXArray(Float(1.0)) + mantissaF)

    let subnormalVal = signF * MLXArray(Float(1.0 / 64.0)) * mantissaF

    var result = MLX.where(isSubnormal, subnormalVal, normalVal)
    result = MLX.where(isNaN, MLXArray(Float(0.0)), result)

    return result
  }

  /// Decode FP8 E5M2 tensor to float32.
  ///
  /// FP8 E5M2: 1 sign bit, 5 exponent bits, 2 mantissa bits.
  /// Bias = 15, has infinity (exp=31, mantissa=0), NaN (exp=31, mantissa!=0).
  ///
  ///   value = (-1)^sign * 2^(exponent - 15) * (1 + mantissa/4)  [normal]
  ///   value = (-1)^sign * 2^(-14) * (mantissa/4)                 [subnormal, exp=0]
  ///   value = Inf / NaN                                           [exp=31]
  private static func decodeFP8E5M2(_ tensor: MLXArray) -> MLXArray {
    let u = tensor.asType(.int32)

    let sign = (u >> 7) & 1
    let exponent = (u >> 2) & 0x1F  // 5 exponent bits
    let mantissa = u & 0x3          // 2 mantissa bits

    let signF = MLX.where(sign .== 0, MLXArray(Float(1.0)), MLXArray(Float(-1.0)))

    let mantissaF = mantissa.asType(.float32) / MLXArray(Float(4.0))
    let expF = exponent.asType(.float32)

    let isSubnormal = exponent .== 0
    let isSpecial = exponent .== 31  // Inf or NaN — treat both as zero for weights

    let normalPow = MLX.pow(MLXArray(Float(2.0)), expF - MLXArray(Float(15.0)))
    let normalVal = signF * normalPow * (MLXArray(Float(1.0)) + mantissaF)

    let subnormalVal = signF * MLXArray(Float(1.0 / 16384.0)) * mantissaF  // 2^(-14)

    var result = MLX.where(isSubnormal, subnormalVal, normalVal)
    result = MLX.where(isSpecial, MLXArray(Float(0.0)), result)

    return result
  }

  /// Decode FP8 tensor to float32 using the appropriate format rules.
  private static func decodeFP8(_ tensor: MLXArray, format: FP8Format = .e4m3fn) -> MLXArray {
    switch format {
    case .e4m3fn: return decodeFP8E4M3(tensor)
    case .e5m2: return decodeFP8E5M2(tensor)
    }
  }
}
