import Foundation
import Logging
import MLX
import MLXNN

/// Loads and applies weights from safetensors files to LTX-2 VAE modules.
///
/// Handles key remapping from HuggingFace/PyTorch format to the Swift module
/// hierarchy, and transposes Conv3d weights from PyTorch's channels-first format
/// `(O, I, D, H, W)` to MLX's channels-last format `(O, D, H, W, I)`.
///
/// ## Key Remapping
///
/// PyTorch weight keys follow the pattern:
/// ```
/// vae.encoder.conv_in.weight
/// vae.encoder.down_blocks.0.res_blocks.0.conv1.weight
/// vae.decoder.conv_in.conv.weight
/// vae.decoder.up_blocks.0.res_blocks.0.conv1.conv.weight
/// vae.per_channel_statistics.mean-of-means
/// vae.per_channel_statistics.std-of-means
/// ```
///
/// These are remapped to match the Swift module tree:
/// ```
/// encoder.conv_in.weight
/// encoder.down_blocks.0.res_blocks.0.conv1.weight
/// decoder.conv_in.conv.weight
/// decoder.up_blocks.0.res_blocks.0.conv1.conv.weight
/// encoder.per_channel_statistics.mean
/// decoder.per_channel_statistics.mean
/// ```
public enum LTX2WeightLoader {

  /// Default weight file names.
  public static let encoderWeightFile = "ltx-video-2-0.7b.safetensors"
  public static let decoderWeightFile = "ltx-video-2-0.7b.safetensors"

  /// Errors that can occur during weight loading.
  public enum LoadError: Error, CustomStringConvertible {
    case fileNotFound(String)
    case weightApplicationFailed(String, Error)
    case safetensorsReadFailed(String, Error)

    public var description: String {
      switch self {
      case .fileNotFound(let path):
        return "Weight file not found: \(path)"
      case .weightApplicationFailed(let component, let error):
        return "Failed to apply \(component) weights: \(error)"
      case .safetensorsReadFailed(let path, let error):
        return "Failed to read safetensors file \(path): \(error)"
      }
    }
  }

  // MARK: - Public API

  /// Loads VAE weights from safetensors files and applies them to the model.
  ///
  /// - Parameters:
  ///   - vae: The LTX2VAE instance to load weights into.
  ///   - directory: Directory containing the safetensors files.
  ///   - dtype: Target dtype for weights. Default `.bfloat16`.
  ///   - logger: Logger for progress reporting.
  public static func loadVAEWeights(
    into vae: LTX2VAE,
    from directory: URL,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws {
    // Collect all safetensors files
    let fm = FileManager.default
    let files = try fm.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
      .filter { $0.pathExtension == "safetensors" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !files.isEmpty else {
      throw LoadError.fileNotFound("No safetensors files found in \(directory.path)")
    }

    logger.info("Found \(files.count) safetensors file(s) in \(directory.lastPathComponent)")

    // Load all tensors
    var rawTensors: [String: MLXArray] = [:]
    for file in files {
      do {
        let reader = try SafeTensorsReader(fileURL: file)
        let tensors = try reader.loadAllTensors(as: dtype)
        rawTensors.merge(tensors) { _, new in new }
      } catch {
        throw LoadError.safetensorsReadFailed(file.path, error)
      }
    }

    logger.info("Read \(rawTensors.count) tensors total")

    // Remap and apply encoder weights
    let encoderWeights = remapEncoderKeys(rawTensors)
    logger.info("Remapped \(encoderWeights.count) encoder weight keys")

    do {
      let params = ModuleParameters.unflattened(encoderWeights)
      try vae.encoder.update(parameters: params, verify: [.shapeMismatch])
    } catch {
      logger.error("Encoder weight application error: \(error)")
      throw LoadError.weightApplicationFailed("encoder", error)
    }

    // Remap and apply decoder weights
    let decoderWeights = remapDecoderKeys(rawTensors)
    logger.info("Remapped \(decoderWeights.count) decoder weight keys")

    do {
      let params = ModuleParameters.unflattened(decoderWeights)
      try vae.decoder.update(parameters: params, verify: [.shapeMismatch])
    } catch {
      logger.error("Decoder weight application error: \(error)")
      throw LoadError.weightApplicationFailed("decoder", error)
    }

    logger.info("Applied LTX-2 VAE weights successfully")
  }

  // MARK: - Encoder Key Remapping

  private static func remapEncoderKeys(
    _ tensors: [String: MLXArray]
  ) -> [(String, MLXArray)] {
    var result: [(String, MLXArray)] = []

    for (key, value) in tensors {
      // Only process VAE encoder weights
      guard key.hasPrefix("vae.") else { continue }
      guard !key.hasPrefix("vae.decoder.") else { continue }

      var newKey = key
      var newValue = value

      // Per-channel statistics
      if key == "vae.per_channel_statistics.mean-of-means" {
        newKey = "per_channel_statistics.mean"
      } else if key == "vae.per_channel_statistics.std-of-means" {
        newKey = "per_channel_statistics.std"
      } else if key.hasPrefix("vae.per_channel_statistics.") {
        continue  // Skip other statistics
      } else if key.hasPrefix("vae.encoder.") {
        newKey = key.replacingOccurrences(of: "vae.encoder.", with: "")
      } else {
        continue
      }

      // Conv3d weight transpose: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
      if isConvWeightKey(newKey) && newValue.ndim == 5 {
        newValue = newValue.transposed(0, 2, 3, 4, 1)
      }

      result.append((newKey, newValue))
    }

    return result
  }

  // MARK: - Decoder Key Remapping

  private static func remapDecoderKeys(
    _ tensors: [String: MLXArray]
  ) -> [(String, MLXArray)] {
    var result: [(String, MLXArray)] = []

    for (key, value) in tensors {
      // Only process VAE decoder weights + per-channel stats
      guard key.hasPrefix("vae.") else { continue }
      guard !key.hasPrefix("vae.encoder.") else { continue }

      var newKey = key
      var newValue = value

      // Per-channel statistics
      if key == "vae.per_channel_statistics.mean-of-means" {
        newKey = "per_channel_statistics.mean"
      } else if key == "vae.per_channel_statistics.std-of-means" {
        newKey = "per_channel_statistics.std"
      } else if key.hasPrefix("vae.per_channel_statistics.") {
        continue
      } else if key.hasPrefix("vae.decoder.") {
        newKey = key.replacingOccurrences(of: "vae.decoder.", with: "")
      } else {
        continue
      }

      // Conv3d weight transpose: PyTorch (O, I, D, H, W) -> MLX (O, D, H, W, I)
      if isConvWeightKey(newKey) && newValue.ndim == 5 {
        newValue = newValue.transposed(0, 2, 3, 4, 1)
      }

      // Ensure conv_in/conv_out have nested .conv. path for decoder
      // PyTorch: conv_in.conv.weight -> already has .conv.
      // The decoder wrapper structure expects this nesting

      result.append((newKey, newValue))
    }

    return result
  }

  // MARK: - Helpers

  private static func isConvWeightKey(_ key: String) -> Bool {
    key.contains("conv") && key.hasSuffix(".weight")
  }
}
