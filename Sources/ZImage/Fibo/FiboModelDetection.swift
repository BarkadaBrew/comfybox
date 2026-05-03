// FiboModelDetection.swift — Detect FIBO models from HF cache or snapshot directories
// Reference: briaai/FIBO transformer/config.json

import Foundation

/// Information about a detected FIBO model.
public struct FiboDetectedModel: Sendable {
  /// The snapshot directory URL containing the model.
  public let snapshotURL: URL
  /// Parsed transformer configuration.
  public let transformerConfig: FiboTransformerConfig
  /// Parsed VAE configuration.
  public let vaeConfig: FiboVAEConfig
  /// Parsed text encoder configuration.
  public let textEncoderConfig: FiboTextEncoderConfig
  /// Component subdirectory paths.
  public let componentPaths: ComponentPaths

  public struct ComponentPaths: Sendable {
    public let transformer: URL
    public let vae: URL
    public let textEncoder: URL
    public let tokenizer: URL?
  }
}

/// Detects FIBO models from snapshot directories.
///
/// FIBO detection uses the transformer config.json because there is no
/// model_index.json. Detection signals:
///
/// 1. `transformer/config.json` contains `"_class_name": "Bria4Transformer2DModel"`
/// 2. `in_channels` is 48 (from Wan 2.2 VAE z_dim)
/// 3. 8 joint blocks + 38 single blocks
/// 4. `text_encoder/config.json` has `"SmolLM3ForCausalLM"` in architectures
public enum FiboModelDetection {

  /// Known FIBO HuggingFace model IDs.
  public static let knownModelIds: [String] = [
    "briaai/FIBO",
  ]

  /// Check if a model spec string refers to a known FIBO model.
  public static func isKnownFiboModel(_ modelSpec: String) -> Bool {
    let normalized = modelSpec.lowercased()
    return knownModelIds.contains { $0.lowercased() == normalized }
      || normalized.contains("briaai/fibo")
      || normalized.hasSuffix("/fibo")
  }

  /// Detect whether a model snapshot directory contains a FIBO model.
  ///
  /// - Parameter snapshot: Root URL of the model snapshot directory.
  /// - Returns: Detected model info, or nil if not a FIBO model.
  public static func detect(at snapshot: URL) -> FiboDetectedModel? {
    let fm = FileManager.default

    // 1. Check transformer config for Bria4Transformer2DModel
    let transformerConfigURL = snapshot
      .appendingPathComponent("transformer")
      .appendingPathComponent("config.json")

    guard fm.fileExists(atPath: transformerConfigURL.path),
          let configData = try? Data(contentsOf: transformerConfigURL),
          let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
          let className = configDict["_class_name"] as? String,
          className == "Bria4Transformer2DModel" else {
      return nil
    }

    // 2. Verify FIBO-specific architecture params
    let inChannels = configDict["in_channels"] as? Int ?? 0
    let numLayers = configDict["num_layers"] as? Int ?? 0
    let numSingleLayers = configDict["num_single_layers"] as? Int ?? 0

    // FIBO signature: 48 in_channels, 8 joint blocks, 38 single blocks
    guard inChannels == 48, numLayers == 8, numSingleLayers == 38 else {
      return nil
    }

    // 3. Verify text encoder is SmolLM3 (not Qwen3, not CLIP, not T5)
    let textEncoderConfigURL = snapshot
      .appendingPathComponent("text_encoder")
      .appendingPathComponent("config.json")

    guard fm.fileExists(atPath: textEncoderConfigURL.path),
          let teData = try? Data(contentsOf: textEncoderConfigURL),
          let teDict = try? JSONSerialization.jsonObject(with: teData) as? [String: Any],
          let architectures = teDict["architectures"] as? [String],
          architectures.contains("SmolLM3ForCausalLM") else {
      return nil
    }

    // 4. Verify VAE directory exists with safetensors
    let vaeDir = snapshot.appendingPathComponent("vae")
    guard fm.fileExists(atPath: vaeDir.path),
          let vaeContents = try? fm.contentsOfDirectory(at: vaeDir, includingPropertiesForKeys: nil),
          vaeContents.contains(where: { $0.pathExtension == "safetensors" }) else {
      return nil
    }

    // 5. Parse configs
    let decoder = JSONDecoder()

    let transformerConfig: FiboTransformerConfig
    if let tc = try? decoder.decode(FiboTransformerConfig.self, from: configData) {
      transformerConfig = tc
    } else {
      transformerConfig = FiboTransformerConfig()
    }

    let vaeConfig: FiboVAEConfig
    let vaeConfigURL = vaeDir.appendingPathComponent("config.json")
    if let vaeData = try? Data(contentsOf: vaeConfigURL),
       let vc = try? decoder.decode(FiboVAEConfig.self, from: vaeData) {
      vaeConfig = vc
    } else {
      vaeConfig = FiboVAEConfig()
    }

    let textEncoderConfig: FiboTextEncoderConfig
    if let tec = try? decoder.decode(FiboTextEncoderConfig.self, from: teData) {
      textEncoderConfig = tec
    } else {
      textEncoderConfig = FiboTextEncoderConfig()
    }

    // 6. Resolve component paths
    let tokenizerDir = snapshot.appendingPathComponent("tokenizer")
    let hasTokenizer = fm.fileExists(atPath: tokenizerDir.path)

    let componentPaths = FiboDetectedModel.ComponentPaths(
      transformer: snapshot.appendingPathComponent("transformer"),
      vae: vaeDir,
      textEncoder: snapshot.appendingPathComponent("text_encoder"),
      tokenizer: hasTokenizer ? tokenizerDir : nil
    )

    return FiboDetectedModel(
      snapshotURL: snapshot,
      transformerConfig: transformerConfig,
      vaeConfig: vaeConfig,
      textEncoderConfig: textEncoderConfig,
      componentPaths: componentPaths
    )
  }

  /// Attempt to find a cached FIBO model in the HuggingFace cache directory.
  ///
  /// - Parameter cacheRoot: HuggingFace hub cache directory (default: ~/.cache/huggingface/hub).
  /// - Returns: Detected model info, or nil if not found.
  public static func findInCache(
    cacheRoot: URL? = nil
  ) -> FiboDetectedModel? {
    let fm = FileManager.default
    let root = cacheRoot ?? defaultHFCacheDirectory()

    let repoDir = root.appendingPathComponent("models--briaai--FIBO")
    let snapshotsDir = repoDir.appendingPathComponent("snapshots")

    guard fm.fileExists(atPath: snapshotsDir.path),
          let snapshots = try? fm.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: nil
          ) else {
      return nil
    }

    for snapshot in snapshots {
      if let detected = detect(at: snapshot) {
        return detected
      }
    }

    return nil
  }

  // MARK: - Private

  private static func defaultHFCacheDirectory() -> URL {
    let env = ProcessInfo.processInfo.environment
    if let hubCache = env["HF_HUB_CACHE"], !hubCache.isEmpty {
      return URL(fileURLWithPath: hubCache)
    }
    if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
      return URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
    }
    return FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".cache/huggingface/hub")
  }
}
