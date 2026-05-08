import Foundation
import Logging
import MLX
import MLXNN

/// Loads BasicSR/Real-ESRGAN RRDBNet weights from safetensors files.
public enum ESRGANWeightLoader {
  public enum LoadError: Error, CustomStringConvertible {
    case pathNotFound(String)
    case noSafeTensorsFound(String)
    case noCompatibleWeightsFound(String)
    case safetensorsReadFailed(String, Error)
    case weightApplicationFailed(Error)

    public var description: String {
      switch self {
      case .pathNotFound(let path):
        return "ESRGAN weights path not found: \(path)"
      case .noSafeTensorsFound(let path):
        return "No ESRGAN safetensors files found in: \(path)"
      case .noCompatibleWeightsFound(let path):
        return "No compatible ESRGAN weights found in: \(path)"
      case .safetensorsReadFailed(let path, let error):
        return "Failed to read safetensors file \(path): \(error)"
      case .weightApplicationFailed(let error):
        return "Failed to apply ESRGAN weights: \(error)"
      }
    }
  }

  public static func loadWeights(
    into model: RRDBNet,
    from directory: URL,
    dtype: DType = .float32,
    logger: Logger? = nil
  ) throws {
    let files = try safetensorsFiles(in: directory)
    guard !files.isEmpty else {
      throw LoadError.noSafeTensorsFound(directory.path)
    }

    var rawTensors: [String: MLXArray] = [:]
    for file in files {
      do {
        let reader = try SafeTensorsReader(fileURL: file)
        let tensors = try reader.loadAllTensors(as: dtype)
        for (key, value) in tensors {
          rawTensors[key] = value
        }
        logger?.info("Read \(tensors.count) ESRGAN tensors from \(file.lastPathComponent)")
      } catch {
        throw LoadError.safetensorsReadFailed(file.path, error)
      }
    }

    let remapped = remapAndTranspose(rawTensors, for: model)
    guard !remapped.isEmpty else {
      throw LoadError.noCompatibleWeightsFound(directory.path)
    }

    do {
      let params = ModuleParameters.unflattened(remapped)
      try model.update(parameters: params, verify: [.shapeMismatch])
    } catch {
      throw LoadError.weightApplicationFailed(error)
    }

    logger?.info("Applied \(remapped.count) ESRGAN tensors")
  }

  public static func detectConfig(from directory: URL) -> ESRGANConfig {
    ESRGANConfig.detect(from: directory)
  }

  static func remapAndTranspose(
    _ tensors: [String: MLXArray],
    for model: RRDBNet
  ) -> [(String, MLXArray)] {
    let expectedShapes = Dictionary(
      uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) }
    )

    var result: [(String, MLXArray)] = []
    result.reserveCapacity(tensors.count)

    for (rawKey, value) in tensors {
      let key = canonicalKey(rawKey)
      guard let expectedShape = expectedShapes[key] else {
        continue
      }

      let prepared: MLXArray
      if value.shape == expectedShape {
        prepared = value
      } else if value.ndim == 4 && key.hasSuffix(".weight") {
        let transposed = value.transposed(0, 2, 3, 1)
        prepared = transposed.shape == expectedShape ? transposed : value
      } else {
        prepared = value
      }

      result.append((key, prepared))
    }

    return result.sorted { $0.0 < $1.0 }
  }

  private static func canonicalKey(_ key: String) -> String {
    for prefix in ["params_ema.", "params.", "model."] where key.hasPrefix(prefix) {
      return String(key.dropFirst(prefix.count))
    }
    return key
  }

  private static func safetensorsFiles(in url: URL) throws -> [URL] {
    let fm = FileManager.default
    var isDirectory: ObjCBool = false
    guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
      throw LoadError.pathNotFound(url.path)
    }

    if !isDirectory.boolValue {
      return url.pathExtension == "safetensors" ? [url] : []
    }

    let contents = try fm.contentsOfDirectory(
      at: url,
      includingPropertiesForKeys: [.isRegularFileKey],
      options: [.skipsHiddenFiles]
    )

    return contents
      .filter { $0.pathExtension == "safetensors" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
  }
}
