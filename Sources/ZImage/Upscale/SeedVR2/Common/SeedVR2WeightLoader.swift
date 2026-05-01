import Foundation
import Logging
import MLX
import MLXNN

/// Loads and applies weights from safetensors files to SeedVR2 transformer and VAE modules.
///
/// Supports both 3B and 7B model variants with automatic weight file detection.
public enum SeedVR2WeightLoader {

  /// Default 3B transformer weight file names.
  public static let transformerWeightFile3B = "seedvr2_ema_3b_fp16.safetensors"
  public static let transformerWeightFile3BFP8 = "seedvr2_ema_3b_fp8.safetensors"

  /// 7B transformer weight file names.
  public static let transformerWeightFile7B = "seedvr2_ema_7b_fp16.safetensors"
  public static let transformerWeightFile7BFP8 = "seedvr2_ema_7b_fp8.safetensors"

  /// Legacy aliases for backward compatibility.
  public static let transformerWeightFile = transformerWeightFile3B
  public static let transformerWeightFileFP8 = transformerWeightFile3BFP8

  /// Default VAE weight file name.
  public static let vaeWeightFile = "ema_vae_fp16.safetensors"

  /// Text embedding file name.
  public static let textEmbeddingFile = SeedVR2TextEmbeddings.fileName

  /// Errors that can occur during weight loading.
  public enum LoadError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case weightApplicationFailed(String, Error)
    case safetensorsReadFailed(String, Error)
    case noTransformerWeightsFound(String)

    public var description: String {
      switch self {
      case .fileNotFound(let path):
        return "Weight file not found: \(path)"
      case .weightApplicationFailed(let component, let error):
        return "Failed to apply \(component) weights: \(error)"
      case .safetensorsReadFailed(let path, let error):
        return "Failed to read safetensors file \(path): \(error)"
      case .noTransformerWeightsFound(let dir):
        return "No SeedVR2 transformer weights (3B or 7B) found in: \(dir)"
      }
    }
  }

  // MARK: - Public API

  /// Detects which transformer weight file is available in the directory.
  ///
  /// Returns the file name and detected model config. Prefers fp16 over fp8.
  /// Prefers 7B over 3B if both are present.
  public static func detectTransformerWeights(
    in directory: URL
  ) -> (fileName: String, config: SeedVR2ModelConfig)? {
    let fm = FileManager.default

    // Check 7B first
    if fm.fileExists(atPath: directory.appendingPathComponent(transformerWeightFile7B).path) {
      return (transformerWeightFile7B, .preset7B)
    }
    if fm.fileExists(atPath: directory.appendingPathComponent(transformerWeightFile7BFP8).path) {
      return (transformerWeightFile7BFP8, .preset7B)
    }

    // Then 3B
    if fm.fileExists(atPath: directory.appendingPathComponent(transformerWeightFile3B).path) {
      return (transformerWeightFile3B, .preset3B)
    }
    if fm.fileExists(atPath: directory.appendingPathComponent(transformerWeightFile3BFP8).path) {
      return (transformerWeightFile3BFP8, .preset3B)
    }

    return nil
  }

  /// Loads transformer weights from a safetensors file and applies them to the model.
  public static func loadTransformerWeights(
    into transformer: SeedVR2Transformer,
    from directory: URL,
    fileName: String,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws {
    let fileURL = directory.appendingPathComponent(fileName)
    guard FileManager.default.fileExists(atPath: fileURL.path) else {
      throw LoadError.fileNotFound(fileURL.path)
    }

    logger.info("Loading SeedVR2 transformer weights from \(fileName)")

    let rawTensors: [String: MLXArray]
    do {
      let reader = try SafeTensorsReader(fileURL: fileURL)
      rawTensors = try reader.loadAllTensors(as: dtype)
    } catch {
      throw LoadError.safetensorsReadFailed(fileURL.path, error)
    }

    logger.info("Read \(rawTensors.count) tensors from \(fileName)")

    // Apply key remapping.
    let remapped = remapTransformerKeys(rawTensors, hasOutputNorm: transformer.hasOutputNorm)
    logger.info("Remapped to \(remapped.count) transformer weight keys")

    // Apply to the module.
    do {
      let params = ModuleParameters.unflattened(remapped)
      try transformer.update(parameters: params, verify: [.shapeMismatch])
    } catch {
      logger.error("Transformer weight application error: \(error)")
      throw LoadError.weightApplicationFailed("transformer", error)
    }

    logger.info("Applied transformer weights successfully")
  }

  /// Loads VAE weights from a safetensors file and applies them to the model.
  public static func loadVAEWeights(
    into vae: SeedVR2VAE,
    from directory: URL,
    fileName: String = vaeWeightFile,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws {
    let fileURL = directory.appendingPathComponent(fileName)
    guard FileManager.default.fileExists(atPath: fileURL.path) else {
      throw LoadError.fileNotFound(fileURL.path)
    }

    logger.info("Loading SeedVR2 VAE weights from \(fileName)")

    let rawTensors: [String: MLXArray]
    do {
      let reader = try SafeTensorsReader(fileURL: fileURL)
      rawTensors = try reader.loadAllTensors(as: dtype)
    } catch {
      throw LoadError.safetensorsReadFailed(fileURL.path, error)
    }

    logger.info("Read \(rawTensors.count) tensors from \(fileName)")

    let transposed = transposeVAEWeights(rawTensors)
    logger.info("Processed \(transposed.count) VAE weight keys")

    do {
      let params = ModuleParameters.unflattened(transposed)
      try vae.update(parameters: params, verify: [.shapeMismatch])
    } catch {
      throw LoadError.weightApplicationFailed("vae", error)
    }

    logger.info("Applied VAE weights successfully")
  }

  // MARK: - Transformer Key Remapping

  private static func remapTransformerKeys(
    _ tensors: [String: MLXArray],
    hasOutputNorm: Bool
  ) -> [(String, MLXArray)] {
    var result: [(String, MLXArray)] = []

    for (key, value) in tensors {
      let remappedKeys = remapSingleTransformerKey(key, hasOutputNorm: hasOutputNorm)
      for rk in remappedKeys {
        result.append((rk, value))
      }
    }

    return result
  }

  private static func remapSingleTransformerKey(
    _ key: String,
    hasOutputNorm: Bool
  ) -> [String] {
    // Top-level output adaptive norm renames (3B only).
    if key == "vid_out_ada.out_shift" {
      return hasOutputNorm ? ["out_shift"] : []
    }
    if key == "vid_out_ada.out_scale" {
      return hasOutputNorm ? ["out_scale"] : []
    }

    // RoPE frequency rename
    if key.contains(".attn.rope.rope.freqs") {
      return [key.replacingOccurrences(of: ".attn.rope.rope.freqs", with: ".attn.rope.freqs")]
    }

    // AdaLN parameter renames
    if key.contains(".ada.vid.") {
      return [key.replacingOccurrences(of: ".ada.vid.", with: ".ada.params_vid.")]
    }
    if key.contains(".ada.txt.") {
      return [key.replacingOccurrences(of: ".ada.txt.", with: ".ada.params_txt.")]
    }
    if key.contains(".ada.all.") {
      return [key.replacingOccurrences(of: ".ada.all.", with: ".ada.params_all.")]
    }

    // Attention weight sharing for blocks with .all. pattern (3B blocks 10-31)
    let sharedAttnPatterns = [
      ".attn.proj_qkv.all.": (".attn.proj_qkv_vid.", ".attn.proj_qkv_txt."),
      ".attn.norm_q.all.": (".attn.norm_q_vid.", ".attn.norm_q_txt."),
      ".attn.norm_k.all.": (".attn.norm_k_vid.", ".attn.norm_k_txt."),
      ".attn.proj_out.all.": (".attn.proj_out_vid.", ".attn.proj_out_txt."),
    ]

    for (pattern, (vidReplace, txtReplace)) in sharedAttnPatterns {
      if key.contains(pattern) {
        return [
          key.replacingOccurrences(of: pattern, with: vidReplace),
          key.replacingOccurrences(of: pattern, with: txtReplace),
        ]
      }
    }

    // Non-shared attention renames
    let singleAttnPatterns = [
      ".attn.proj_qkv.vid.": ".attn.proj_qkv_vid.",
      ".attn.proj_qkv.txt.": ".attn.proj_qkv_txt.",
      ".attn.norm_q.vid.": ".attn.norm_q_vid.",
      ".attn.norm_q.txt.": ".attn.norm_q_txt.",
      ".attn.norm_k.vid.": ".attn.norm_k_vid.",
      ".attn.norm_k.txt.": ".attn.norm_k_txt.",
      ".attn.proj_out.vid.": ".attn.proj_out_vid.",
      ".attn.proj_out.txt.": ".attn.proj_out_txt.",
    ]

    for (pattern, replacement) in singleAttnPatterns {
      if key.contains(pattern) {
        return [key.replacingOccurrences(of: pattern, with: replacement)]
      }
    }

    // MLP shared weight duplication for blocks with .all. pattern (3B blocks 10-31)
    if key.contains(".mlp.all.") {
      return [
        key.replacingOccurrences(of: ".mlp.all.", with: ".mlp.vid."),
        key.replacingOccurrences(of: ".mlp.all.", with: ".mlp.txt."),
      ]
    }

    // Default: key passes through unchanged.
    return [key]
  }

  // MARK: - VAE Weight Transposition

  private static func transposeVAEWeights(_ tensors: [String: MLXArray]) -> [(String, MLXArray)] {
    var result: [(String, MLXArray)] = []

    for (key, value) in tensors {
      if value.ndim == 5 && isConvWeightKey(key) {
        let transposed = value.transposed(0, 2, 3, 4, 1)
        result.append((key, transposed))
      } else {
        result.append((key, value))
      }
    }

    return result
  }

  private static func isConvWeightKey(_ key: String) -> Bool {
    key.contains("conv") && key.hasSuffix(".weight")
  }
}
