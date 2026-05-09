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


  /// Loads VAE weights from a pre-loaded tensor dictionary.
  ///
  /// The keys must use the `vae.encoder.X` / `vae.decoder.X` format,
  /// with `vae.per_channel_statistics.mean-of-means` and
  /// `vae.per_channel_statistics.std-of-means` for channel stats.
  ///
  /// - Parameters:
  ///   - vae: The LTX2VAE instance to load weights into.
  ///   - tensors: Dictionary of raw tensors with `vae.` prefixed keys.
  ///   - logger: Logger for progress reporting.
  public static func loadVAEWeightsFromTensors(
    into vae: LTX2VAE,
    tensors: [String: MLXArray],
    logger: Logger
  ) throws {
    // The encoder/decoder use [Int: Module] for down_blocks/up_blocks,
    // which is incompatible with ModuleParameters.unflattened (it creates arrays).
    // Work around this by loading block-level weights individually.

    let encoderWeights = remapEncoderKeys(tensors)
    logger.info("Remapped \(encoderWeights.count) encoder weight keys")

    if !encoderWeights.isEmpty {
      applyBlockWeights(to: vae.encoder, weights: encoderWeights, label: "encoder", logger: logger)
    }

    let decoderWeights = remapDecoderKeys(tensors)
    logger.info("Remapped \(decoderWeights.count) decoder weight keys")

    if !decoderWeights.isEmpty {
      applyBlockWeights(to: vae.decoder, weights: decoderWeights, label: "decoder", logger: logger)
    }

    logger.info("Applied LTX-2 VAE weights (best-effort)")
  }

  /// Apply weights to a module, handling dict-typed sub-modules by loading
  /// each top-level group separately.
  private static func applyBlockWeights(
    to module: Module,
    weights: [(String, MLXArray)],
    label: String,
    logger: Logger
  ) {
    // Group weights by top-level key (before first dot)
    var groups: [String: [(String, MLXArray)]] = [:]
    for (key, value) in weights {
      let parts = key.split(separator: ".", maxSplits: 1)
      let topKey = String(parts[0])
      let subKey = parts.count > 1 ? String(parts[1]) : ""
      groups[topKey, default: []].append((subKey, value))
    }

    var loadedCount = 0

    for (topKey, subWeights) in groups {
      // Try to find the child module
      let children = module.children()

      if let childItems = children[topKey] {
        // Handle dict-typed children like down_blocks/up_blocks
        if let dictChildren = childItems as? [Int: Module] {
          // Group sub-weights by block index
          var blockGroups: [Int: [(String, MLXArray)]] = [:]
          for (subKey, value) in subWeights {
            let parts = subKey.split(separator: ".", maxSplits: 1)
            if let idx = Int(parts[0]) {
              let remainder = parts.count > 1 ? String(parts[1]) : ""
              blockGroups[idx, default: []].append((remainder, value))
            }
          }

          for (idx, blockWeights) in blockGroups {
            if let blockModule = dictChildren[idx] {
              let weightList = blockWeights.filter { !$0.0.isEmpty }
              if !weightList.isEmpty {
                do {
                  let params = ModuleParameters.unflattened(weightList)
                  try blockModule.update(parameters: params, verify: [])
                  loadedCount += weightList.count
                } catch {
                  logger.warning("\(label).\(topKey).\(idx) weight load failed: \(error)")
                }
              }
            }
          }
        } else {
          // Regular module child -- load all sub-weights
          let weightList = subWeights.filter { !$0.0.isEmpty }
          if !weightList.isEmpty {
            do {
              if let childModule = childItems as? Module {
                let params = ModuleParameters.unflattened(weightList)
                try childModule.update(parameters: params, verify: [])
                loadedCount += weightList.count
              }
            } catch {
              logger.warning("\(label).\(topKey) weight load failed: \(error)")
            }
          }
        }
      } else {
        // Top-level parameter (e.g., standalone weight/bias)
        let allTopLevel = subWeights.map { ($0.0.isEmpty ? topKey : topKey + "." + $0.0, $0.1) }
        do {
          let params = ModuleParameters.unflattened(allTopLevel)
          try module.update(parameters: params, verify: [])
          loadedCount += allTopLevel.count
        } catch {
          // Silently skip if can't apply
        }
      }
    }

    logger.info("\(label): loaded \(loadedCount)/\(weights.count) weight tensors")
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
