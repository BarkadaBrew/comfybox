import Foundation

/// Configuration for ESRGAN/Real-ESRGAN RRDBNet upscalers.
public struct ESRGANConfig: Equatable, Sendable {
  public let numInCh: Int
  public let numOutCh: Int
  public let scale: Int
  public let numFeat: Int
  public let numBlock: Int
  public let numGrowCh: Int

  public init(
    numInCh: Int = 3,
    numOutCh: Int = 3,
    scale: Int = 4,
    numFeat: Int = 64,
    numBlock: Int = 23,
    numGrowCh: Int = 32
  ) {
    self.numInCh = numInCh
    self.numOutCh = numOutCh
    self.scale = scale
    self.numFeat = numFeat
    self.numBlock = numBlock
    self.numGrowCh = numGrowCh
  }

  public static let ultraSharp4x = ESRGANConfig(
    numInCh: 3,
    numOutCh: 3,
    scale: 4,
    numFeat: 64,
    numBlock: 23,
    numGrowCh: 32
  )

  public static let realESRGAN_x4plus = ESRGANConfig(
    numInCh: 3,
    numOutCh: 3,
    scale: 4,
    numFeat: 64,
    numBlock: 23,
    numGrowCh: 32
  )

  /// Detects the RRDB block count from safetensors keys in a weights directory.
  ///
  /// Other architecture dimensions default to the common 4x RRDBNet preset.
  public static func detect(from directory: URL) -> ESRGANConfig {
    let blockCount = detectBlockCount(from: directory)
    guard let blockCount, blockCount > 0 else {
      return .ultraSharp4x
    }

    return ESRGANConfig(
      numInCh: ultraSharp4x.numInCh,
      numOutCh: ultraSharp4x.numOutCh,
      scale: ultraSharp4x.scale,
      numFeat: ultraSharp4x.numFeat,
      numBlock: blockCount,
      numGrowCh: ultraSharp4x.numGrowCh
    )
  }

  private static func detectBlockCount(from directory: URL) -> Int? {
    let files = ESRGANConfig.safetensorsFiles(in: directory)
    guard !files.isEmpty else { return nil }

    var blockIndices = Set<Int>()
    for file in files {
      guard let reader = try? SafeTensorsReader(fileURL: file) else { continue }
      for name in reader.tensorNames {
        if let index = bodyBlockIndex(from: name) {
          blockIndices.insert(index)
        }
      }
    }

    guard let maxIndex = blockIndices.max() else { return nil }
    return maxIndex + 1
  }

  private static func safetensorsFiles(in url: URL) -> [URL] {
    let fm = FileManager.default
    var isDirectory: ObjCBool = false
    guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
      return []
    }

    if !isDirectory.boolValue {
      return url.pathExtension == "safetensors" ? [url] : []
    }

    let contents = (try? fm.contentsOfDirectory(
      at: url,
      includingPropertiesForKeys: [.isRegularFileKey],
      options: [.skipsHiddenFiles]
    )) ?? []

    return contents
      .filter { $0.pathExtension == "safetensors" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
  }

  private static func bodyBlockIndex(from key: String) -> Int? {
    let parts = key.split(separator: ".").map(String.init)
    guard !parts.isEmpty else { return nil }

    let offset: Int
    switch parts.first {
    case "params", "params_ema", "model":
      offset = 1
    default:
      offset = 0
    }

    guard parts.count > offset + 1, parts[offset] == "body" else {
      return nil
    }

    return Int(parts[offset + 1])
  }
}
