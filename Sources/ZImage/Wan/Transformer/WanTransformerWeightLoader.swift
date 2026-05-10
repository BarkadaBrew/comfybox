import Foundation
import Logging
import MLX
import MLXNN

/// Loads sharded safetensors weights into the Wan 2.2 I2V-A14B transformer.
///
/// The model weights are split across 6 safetensors shards (~53 GB total, FP32).
/// An index JSON maps each weight key to its shard file.
///
/// ## Loading Strategy
/// 1. Parse index JSON to get key -> shard mapping
/// 2. Group keys by shard, load each shard once
/// 3. Transpose Conv3d weights (PyTorch NCTHW -> MLX NTHWC)
/// 4. Convert FP32 -> BF16
/// 5. Apply via `model.update(parameters:verify:)`
///
/// ## Key Identity
/// The safetensors keys map directly to the Swift module hierarchy
/// (`@ModuleInfo` keys match). No key remapping is required.
///
/// ```
/// patch_embedding.weight  ->  model.patchEmbedding.weight  (Conv3d, needs transpose)
/// blocks.0.self_attn.q.weight  ->  model.blocks[0].selfAttn.qProj.weight
/// time_projection.1.weight  ->  model.timeProjection.layers[1].weight
/// ```
public enum WanTransformerWeightLoader {

  /// Default index file name for sharded weights.
  public static let indexFileName = "diffusion_pytorch_model.safetensors.index.json"

  // MARK: - Errors

  public enum LoadError: Error, CustomStringConvertible {
    case indexFileNotFound(String)
    case indexFileInvalid(String)
    case shardFileNotFound(String)
    case safetensorsReadFailed(String, Error)
    case weightApplicationFailed(Error)
    case missingKeys([String])

    public var description: String {
      switch self {
      case .indexFileNotFound(let path):
        return "Transformer weight index not found: \(path)"
      case .indexFileInvalid(let reason):
        return "Invalid index JSON: \(reason)"
      case .shardFileNotFound(let path):
        return "Shard file not found: \(path)"
      case .safetensorsReadFailed(let path, let error):
        return "Failed to read safetensors shard \(path): \(error)"
      case .weightApplicationFailed(let error):
        return "Failed to apply transformer weights: \(error)"
      case .missingKeys(let keys):
        return "Missing \(keys.count) weight keys: \(keys.prefix(5).joined(separator: ", "))..."
      }
    }
  }

  // MARK: - Index Parsing

  /// Parsed weight index: maps each weight key to its shard filename.
  public struct WeightIndex {
    /// Maps weight key name to shard filename.
    public let weightMap: [String: String]
    /// Total size in bytes (metadata).
    public let totalSize: Int

    /// Unique shard filenames, sorted.
    public var shardFiles: [String] {
      Array(Set(weightMap.values)).sorted()
    }

    /// Keys grouped by shard file.
    public var keysByShard: [String: [String]] {
      var grouped: [String: [String]] = [:]
      for (key, shard) in weightMap {
        grouped[shard, default: []].append(key)
      }
      return grouped
    }
  }

  /// Parses the safetensors index JSON file.
  ///
  /// - Parameter url: Path to `diffusion_pytorch_model.safetensors.index.json`.
  /// - Returns: Parsed weight index.
  public static func parseIndex(at url: URL) throws -> WeightIndex {
    guard FileManager.default.fileExists(atPath: url.path) else {
      throw LoadError.indexFileNotFound(url.path)
    }

    let data = try Data(contentsOf: url)
    guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      throw LoadError.indexFileInvalid("Root is not a dictionary")
    }

    guard let weightMap = json["weight_map"] as? [String: String] else {
      throw LoadError.indexFileInvalid("Missing or invalid weight_map")
    }

    let totalSize: Int
    if let metadata = json["metadata"] as? [String: Any],
       let size = metadata["total_size"] as? Int {
      totalSize = size
    } else {
      totalSize = 0
    }

    return WeightIndex(weightMap: weightMap, totalSize: totalSize)
  }

  // MARK: - Weight Loading

  /// Loads all transformer weights from sharded safetensors files.
  ///
  /// Reads each shard once, extracts only the keys belonging to it,
  /// transposes Conv3d weights, and converts to the target dtype.
  ///
  /// - Parameters:
  ///   - directory: Directory containing the shard files and index JSON.
  ///   - indexFileName: Name of the index JSON file.
  ///   - dtype: Target data type for weights. Default `.bfloat16`.
  ///   - logger: Logger for progress output.
  /// - Returns: Dictionary of weight key to MLXArray.
  public static func loadWeights(
    from directory: URL,
    indexFileName: String = WanTransformerWeightLoader.indexFileName,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws -> [String: MLXArray] {
    let indexURL = directory.appendingPathComponent(indexFileName)
    let index = try parseIndex(at: indexURL)

    logger.info("Wan transformer weight index: \(index.weightMap.count) keys across \(index.shardFiles.count) shards")
    if index.totalSize > 0 {
      let gb = Double(index.totalSize) / (1024 * 1024 * 1024)
      logger.info("  Total weight size: \(String(format: "%.1f", gb)) GB (source FP32)")
    }

    var allWeights: [String: MLXArray] = [:]
    allWeights.reserveCapacity(index.weightMap.count)

    let keysByShard = index.keysByShard

    for (shardIdx, shardFile) in index.shardFiles.enumerated() {
      let shardURL = directory.appendingPathComponent(shardFile)
      guard FileManager.default.fileExists(atPath: shardURL.path) else {
        throw LoadError.shardFileNotFound(shardURL.path)
      }

      logger.info("  Loading shard \(shardIdx + 1)/\(index.shardFiles.count): \(shardFile)")

      let reader: SafeTensorsReader
      do {
        reader = try SafeTensorsReader(fileURL: shardURL)
      } catch {
        throw LoadError.safetensorsReadFailed(shardURL.path, error)
      }

      // Load only the keys that belong to this shard
      guard let shardKeys = keysByShard[shardFile] else { continue }

      for key in shardKeys {
        var tensor = try reader.tensor(named: key)

        // Transpose Conv3d weights: PyTorch [outCh, inCh, kT, kH, kW]
        // -> MLX [outCh, kT, kH, kW, inCh]
        if key == "patch_embedding.weight" && tensor.ndim == 5 {
          tensor = tensor.transposed(0, 2, 3, 4, 1)
        }

        // Convert to target dtype
        if tensor.dtype != dtype {
          tensor = tensor.asType(dtype)
        }

        allWeights[key] = tensor
      }

      logger.info("    Loaded \(shardKeys.count) tensors from \(shardFile)")
    }

    return allWeights
  }

  /// Loads weights and applies them to a WanTransformer3D model.
  ///
  /// - Parameters:
  ///   - transformer: The model to load weights into.
  ///   - directory: Directory containing the shard files and index JSON.
  ///   - indexFileName: Name of the index JSON file.
  ///   - dtype: Target data type. Default `.bfloat16`.
  ///   - logger: Logger for progress output.
  public static func loadTransformerWeights(
    into transformer: WanTransformer3D,
    from directory: URL,
    indexFileName: String = WanTransformerWeightLoader.indexFileName,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws {
    let weights = try loadWeights(
      from: directory, indexFileName: indexFileName,
      dtype: dtype, logger: logger
    )

    // Validate keys
    let (missing, unexpected) = WanTransformerWeightMapping.validateKeys(
      weights, config: transformer.config
    )
    if !unexpected.isEmpty {
      logger.warning("  \(unexpected.count) unexpected weight keys (ignored)")
    }
    if !missing.isEmpty {
      logger.error("  Missing \(missing.count) weight keys: \(missing.prefix(5))")
      throw LoadError.missingKeys(missing)
    }

    // Build flat key-value pairs for ModuleParameters
    let flat: [(String, MLXArray)] = weights.map { ($0.key, $0.value) }

    // Apply to model
    do {
      let params = ModuleParameters.unflattened(flat)
      try transformer.update(parameters: params, verify: [.shapeMismatch])
    } catch {
      throw LoadError.weightApplicationFailed(error)
    }

    let totalParams = weights.values.reduce(0) { $0 + $1.size }
    let paramGB = Double(totalParams * 2) / (1024 * 1024 * 1024)  // BF16 = 2 bytes
    logger.info("Wan transformer weights applied: \(weights.count) tensors, \(String(format: "%.1f", paramGB)) GB (BF16)")
  }

  // MARK: - Convenience

  /// Checks whether transformer weights can be loaded from the given directory.
  ///
  /// Looks for the index JSON at `directory/indexFileName`.
  ///
  /// - Parameter directory: Model directory (e.g., `.../high_noise_model/`).
  /// - Returns: Whether the index file exists.
  public static func canLoadWeights(from directory: URL) -> Bool {
    let indexURL = directory.appendingPathComponent(indexFileName)
    return FileManager.default.fileExists(atPath: indexURL.path)
  }
}
