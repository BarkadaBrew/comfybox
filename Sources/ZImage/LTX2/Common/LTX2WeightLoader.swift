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

  /// Apply weights to a module by recursively navigating [String: Module] dicts.
  ///
  /// MLX-Swift's `ModuleParameters.unflattened()` treats numeric string keys
  /// (like "0", "1") as array indices, creating `.array(...)` structures.
  /// But our modules use `[String: Module]` dicts which produce `.dictionary(...)`
  /// structures. These are incompatible, causing a crash in `update(parameters:)`.
  ///
  /// This method avoids `unflattened` for numeric-keyed paths by recursively
  /// descending through `items()` / `children()` to find each block module,
  /// then applying only the leaf (non-numeric) weights via `update(parameters:)`.
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
    let moduleItems = module.items()

    for (topKey, subWeights) in groups {
      guard let topItem = moduleItems[topKey] else {
        // Not found in module items -- try as top-level parameters
        let allTopLevel = subWeights.map { ($0.0.isEmpty ? topKey : topKey + "." + $0.0, $0.1) }
        do {
          let params = ModuleParameters.unflattened(allTopLevel)
          try module.update(parameters: params, verify: [])
          loadedCount += allTopLevel.count
        } catch {
          // Silently skip params that don't exist in the model (e.g., timestep)
        }
        continue
      }

      // Check if sub-keys start with numeric indices (block dict pattern)
      let hasNumericSubKeys = subWeights.contains { subKey, _ in
        let first = subKey.split(separator: ".").first
        return first != nil && Int(first!) != nil
      }

      if hasNumericSubKeys, case .dictionary(let dictEntries) = topItem {
        // [String: Module] dict property (up_blocks, down_blocks, res_blocks)
        var blockGroups: [String: [(String, MLXArray)]] = [:]
        for (subKey, value) in subWeights {
          let parts = subKey.split(separator: ".", maxSplits: 1)
          let idxStr = String(parts[0])
          let remainder = parts.count > 1 ? String(parts[1]) : ""
          blockGroups[idxStr, default: []].append((remainder, value))
        }

        for (idxStr, blockWeights) in blockGroups {
          guard let blockEntry = dictEntries[idxStr],
                case .value(.module(let blockModule)) = blockEntry else {
            logger.warning("\(label).\(topKey).\(idxStr): no module found in items()")
            continue
          }

          let weightList = blockWeights.filter { !$0.0.isEmpty }
          if !weightList.isEmpty {
            // Recurse into the block module -- it may also have [String: Module] dicts
            applyBlockWeights(
              to: blockModule, weights: weightList,
              label: "\(label).\(topKey).\(idxStr)", logger: logger
            )
            loadedCount += weightList.count
          }
        }
      } else if case .value(.module(let childModule)) = topItem {
        // Regular module child (conv_in, conv_out, per_channel_statistics, etc.)
        let weightList = subWeights.filter { !$0.0.isEmpty }
        if !weightList.isEmpty {
          // Check if child also has numeric sub-keys that need recursive handling
          let childHasNumericKeys = weightList.contains { subKey, _ in
            let first = subKey.split(separator: ".").first
            return first != nil && Int(first!) != nil
          }

          if childHasNumericKeys {
            applyBlockWeights(
              to: childModule, weights: weightList,
              label: "\(label).\(topKey)", logger: logger
            )
            loadedCount += weightList.count
          } else {
            do {
              let params = ModuleParameters.unflattened(weightList)
              try childModule.update(parameters: params, verify: [])
              loadedCount += weightList.count
            } catch {
              logger.warning("\(label).\(topKey) weight load failed: \(error)")
            }
          }
        }
      } else {
        // Try as module-level parameter update
        let allTopLevel = subWeights.map { ($0.0.isEmpty ? topKey : topKey + "." + $0.0, $0.1) }
        do {
          let params = ModuleParameters.unflattened(allTopLevel)
          try module.update(parameters: params, verify: [])
          loadedCount += allTopLevel.count
        } catch {
          // Silently skip
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

      // Conv3d weight transpose: only if in PyTorch (O, I, D, H, W) format.
      // Detect format: if dim[1] > kernel_size (not 1,2,3), it's PyTorch format.
      // If dim[4] > kernel_size, it's already MLX (O, D, H, W, I) format.
      if isConvWeightKey(newKey) && newValue.ndim == 5 {
        let isPyTorchFormat = newValue.dim(1) > 3 && newValue.dim(4) <= 3
        if isPyTorchFormat {
          newValue = newValue.transposed(0, 2, 3, 4, 1)
        }
        // else: already in MLX format, no transpose needed
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

      // Per-channel statistics: prefer vae.decoder.per_channel_statistics over
      // vae.per_channel_statistics.mean-of-means to avoid duplicates.
      if key == "vae.per_channel_statistics.mean-of-means" {
        // Skip -- decoder has its own per_channel_statistics from vae.decoder. path
        // Including both would create duplicate keys that crash unflattened().
        continue
      } else if key == "vae.per_channel_statistics.std-of-means" {
        continue
      } else if key.hasPrefix("vae.per_channel_statistics.") {
        continue
      } else if key.hasPrefix("vae.decoder.") {
        newKey = key.replacingOccurrences(of: "vae.decoder.", with: "")
      } else {
        continue
      }

      // Conv3d weight transpose: only if in PyTorch (O, I, D, H, W) format.
      if isConvWeightKey(newKey) && newValue.ndim == 5 {
        let isPyTorchFormat = newValue.dim(1) > 3 && newValue.dim(4) <= 3
        if isPyTorchFormat {
          newValue = newValue.transposed(0, 2, 3, 4, 1)
        }
      }

      result.append((newKey, newValue))
    }

    return result
  }

  // MARK: - Helpers

  private static func isConvWeightKey(_ key: String) -> Bool {
    key.contains("conv") && key.hasSuffix(".weight")
  }
}
