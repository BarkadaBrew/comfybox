// ChromaModelDetection.swift — Detect Chroma models from HF cache or snapshot directories
// Reference: lodestones/Chroma (chroma/ directory with Approximator, T5-XXL text encoder)

import Foundation

/// Information about a detected Chroma model.
public struct ChromaDetectedModel: Sendable {
  /// Chroma model configuration.
  public let config: ChromaConfig
  /// The snapshot directory URL containing the model.
  public let snapshotURL: URL
  /// Component subdirectory paths.
  public let componentPaths: ChromaComponentPaths
}

/// Resolved file paths for Chroma model components.
public struct ChromaComponentPaths: Sendable {
  /// Path to the transformer weights (chroma/chroma.safetensors).
  public let transformerPath: URL
  /// Path to the VAE weights (vae/ae.safetensors).
  public let vaePath: URL
  /// Paths to T5-XXL text encoder shards.
  public let t5Paths: [URL]
  /// Path to the T5 tokenizer directory (t5/tokenizer_2/).
  public let tokenizerPath: URL
}

/// Detects Chroma models from snapshot directories.
///
/// Chroma detection uses structural signals unique to this architecture:
///
/// 1. `chroma/` subdirectory with safetensors (Approximator + transformer weights)
/// 2. `t5/` subdirectory (T5-XXL text encoder — Flux2/Klein uses Qwen3, not T5)
/// 3. Does NOT have `transformer/config.json` with `Flux2Transformer2DModel`
///
/// Chroma replaces the 3.3B AdaLN modulation layer with a 250M Approximator FFN.
/// Uses T5-XXL (not CLIP) as its sole text encoder.
public enum ChromaModelDetection {

  /// Known Chroma HuggingFace model IDs.
  public static let knownModelIds: [String] = [
    "lodestones/Chroma",
    "jack813liu/mlx-chroma",
  ]

  /// Check if a model spec string refers to a known Chroma model.
  public static func isKnownChromaModel(_ modelSpec: String) -> Bool {
    let normalized = modelSpec.lowercased()
    return knownModelIds.contains { $0.lowercased() == normalized }
      || normalized == "chroma"
      || normalized.contains("chroma")
  }

  /// Detect whether a model snapshot directory contains a Chroma model.
  ///
  /// Detection signals:
  /// 1. `chroma/` subdirectory exists with at least one safetensors file
  /// 2. `t5/` subdirectory exists (T5-XXL text encoder)
  /// 3. Not a Flux 2 model (no `transformer/config.json` with `Flux2Transformer2DModel`)
  ///
  /// - Parameter snapshot: Root URL of the model snapshot directory.
  /// - Returns: Detected model info, or nil if not a Chroma model.
  public static func detect(at snapshot: URL) -> ChromaDetectedModel? {
    let fm = FileManager.default

    // 1. Check for chroma/ subdirectory with safetensors
    let chromaDir = snapshot.appendingPathComponent("chroma")
    guard fm.fileExists(atPath: chromaDir.path),
          let chromaContents = try? fm.contentsOfDirectory(at: chromaDir, includingPropertiesForKeys: nil),
          chromaContents.contains(where: { $0.pathExtension == "safetensors" }) else {
      return nil
    }

    // 2. Check for t5/ subdirectory (T5-XXL text encoder, unique to Chroma)
    let t5Dir = snapshot.appendingPathComponent("t5")
    guard fm.fileExists(atPath: t5Dir.path) else {
      return nil
    }

    // 3. Exclude Flux 2 Klein models (they have transformer/config.json with Flux2Transformer2DModel)
    let transformerConfigURL = snapshot
      .appendingPathComponent("transformer")
      .appendingPathComponent("config.json")
    if fm.fileExists(atPath: transformerConfigURL.path),
       let configData = try? Data(contentsOf: transformerConfigURL),
       let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any],
       let className = configDict["_class_name"] as? String,
       className == "Flux2Transformer2DModel" {
      return nil
    }

    // 4. Resolve component paths
    guard let componentPaths = resolveComponentPaths(at: snapshot) else {
      return nil
    }

    // 5. Parse config from chroma/config.json if available, otherwise use standard
    let config = parseChromaConfig(at: snapshot) ?? ChromaConfig.standard

    return ChromaDetectedModel(
      config: config,
      snapshotURL: snapshot,
      componentPaths: componentPaths
    )
  }

  /// Resolve paths to all Chroma model components.
  ///
  /// Expected layout:
  /// ```
  /// snapshot/
  ///   chroma/chroma.safetensors
  ///   vae/ae.safetensors
  ///   t5/text_encoder_2/model-00001-of-00002.safetensors
  ///   t5/text_encoder_2/model-00002-of-00002.safetensors
  ///   t5/tokenizer_2/
  /// ```
  ///
  /// - Parameter snapshot: Root URL of the model snapshot directory.
  /// - Returns: Resolved component paths, or nil if required files are missing.
  public static func resolveComponentPaths(at snapshot: URL) -> ChromaComponentPaths? {
    let fm = FileManager.default

    // Transformer: chroma/chroma.safetensors
    let chromaDir = snapshot.appendingPathComponent("chroma")
    let transformerPath = chromaDir.appendingPathComponent("chroma.safetensors")
    guard fm.fileExists(atPath: transformerPath.path) else {
      // Fall back to any safetensors file in chroma/
      guard let contents = try? fm.contentsOfDirectory(at: chromaDir, includingPropertiesForKeys: nil),
            let firstSafetensors = contents.first(where: { $0.pathExtension == "safetensors" }) else {
        return nil
      }
      return resolveWithTransformer(firstSafetensors, at: snapshot)
    }

    return resolveWithTransformer(transformerPath, at: snapshot)
  }

  /// Attempt to find a cached Chroma model in the HuggingFace cache directory.
  ///
  /// - Parameter cacheRoot: HuggingFace hub cache directory (default: ~/.cache/huggingface/hub).
  /// - Returns: Detected model info, or nil if not found.
  public static func findInCache(
    cacheRoot: URL? = nil
  ) -> ChromaDetectedModel? {
    let fm = FileManager.default
    let root = cacheRoot ?? defaultHFCacheDirectory()

    // Check known repo directory names
    let repoDirNames = [
      "models--lodestones--Chroma",
      "models--jack813liu--mlx-chroma",
    ]

    for repoDirName in repoDirNames {
      let snapshotsDir = root
        .appendingPathComponent(repoDirName)
        .appendingPathComponent("snapshots")

      guard fm.fileExists(atPath: snapshotsDir.path),
            let snapshots = try? fm.contentsOfDirectory(
              at: snapshotsDir,
              includingPropertiesForKeys: nil
            ) else {
        continue
      }

      for snapshot in snapshots {
        if let detected = detect(at: snapshot) {
          return detected
        }
      }
    }

    return nil
  }

  // MARK: - Private

  private static func resolveWithTransformer(
    _ transformerPath: URL,
    at snapshot: URL
  ) -> ChromaComponentPaths? {
    let fm = FileManager.default

    // VAE: vae/ae.safetensors
    let vaePath = snapshot
      .appendingPathComponent("vae")
      .appendingPathComponent("ae.safetensors")
    guard fm.fileExists(atPath: vaePath.path) else {
      return nil
    }

    // T5 encoder shards: resolve from index.json if present, otherwise use known paths
    let t5Paths = resolveT5Paths(at: snapshot)
    guard !t5Paths.isEmpty else {
      return nil
    }

    // Tokenizer: t5/tokenizer_2/
    let tokenizerPath = snapshot
      .appendingPathComponent("t5")
      .appendingPathComponent("tokenizer_2")
    guard fm.fileExists(atPath: tokenizerPath.path) else {
      return nil
    }

    return ChromaComponentPaths(
      transformerPath: transformerPath,
      vaePath: vaePath,
      t5Paths: t5Paths,
      tokenizerPath: tokenizerPath
    )
  }

  /// Resolve T5-XXL encoder shard paths from index.json or known filenames.
  private static func resolveT5Paths(at snapshot: URL) -> [URL] {
    let fm = FileManager.default
    let t5EncoderDir = snapshot
      .appendingPathComponent("t5")
      .appendingPathComponent("text_encoder_2")

    // Try index.json first for shard resolution
    let indexURL = t5EncoderDir.appendingPathComponent("model.safetensors.index.json")
    if fm.fileExists(atPath: indexURL.path),
       let indexData = try? Data(contentsOf: indexURL),
       let indexDict = try? JSONSerialization.jsonObject(with: indexData) as? [String: Any],
       let weightMap = indexDict["weight_map"] as? [String: String] {
      // Collect unique shard filenames, preserving order
      var seen = Set<String>()
      var shardFiles: [String] = []
      for (_, filename) in weightMap.sorted(by: { $0.key < $1.key }) {
        if seen.insert(filename).inserted {
          shardFiles.append(filename)
        }
      }
      let paths = shardFiles.map { t5EncoderDir.appendingPathComponent($0) }
      if paths.allSatisfy({ fm.fileExists(atPath: $0.path) }) {
        return paths
      }
    }

    // Fall back to known shard filenames
    let knownShards = [
      "model-00001-of-00002.safetensors",
      "model-00002-of-00002.safetensors",
    ]
    let paths = knownShards.map { t5EncoderDir.appendingPathComponent($0) }
    if paths.allSatisfy({ fm.fileExists(atPath: $0.path) }) {
      return paths
    }

    // Last resort: any safetensors files in the directory
    if let contents = try? fm.contentsOfDirectory(at: t5EncoderDir, includingPropertiesForKeys: nil) {
      let safetensorsPaths = contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
      if !safetensorsPaths.isEmpty {
        return safetensorsPaths
      }
    }

    return []
  }

  /// Parse Chroma-specific config from the snapshot directory.
  private static func parseChromaConfig(at snapshot: URL) -> ChromaConfig? {
    let configURL = snapshot
      .appendingPathComponent("chroma")
      .appendingPathComponent("config.json")

    guard let data = try? Data(contentsOf: configURL),
          let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      return nil
    }

    let inChannels = dict["in_channels"] as? Int ?? 64
    let outChannels = dict["out_channels"] as? Int ?? 64
    let contextInDim = dict["context_in_dim"] as? Int ?? 4096
    let hiddenSize = dict["hidden_size"] as? Int ?? 3072
    let mlpRatio = (dict["mlp_ratio"] as? NSNumber)?.floatValue ?? 4.0
    let numHeads = dict["num_heads"] as? Int ?? 24
    let depth = dict["depth"] as? Int ?? 19
    let depthSingleBlocks = dict["depth_single_blocks"] as? Int ?? 38
    let theta = dict["theta"] as? Int ?? 10_000
    let patchSize = dict["patch_size"] as? Int ?? 2
    let qkvBias = dict["qkv_bias"] as? Bool ?? true

    let axesDim: [Int]
    if let axes = dict["axes_dim"] as? [Int] {
      axesDim = axes
    } else {
      axesDim = [16, 56, 56]
    }

    let approxInDim = dict["approx_in_dim"] as? Int ?? 64
    let approxOutDim = dict["approx_out_dim"] as? Int ?? 3072
    let approxHiddenDim = dict["approx_hidden_dim"] as? Int ?? 5120
    let approxNLayers = dict["approx_n_layers"] as? Int ?? 5

    return ChromaConfig(
      inChannels: inChannels,
      outChannels: outChannels,
      contextInDim: contextInDim,
      hiddenSize: hiddenSize,
      mlpRatio: mlpRatio,
      numHeads: numHeads,
      depth: depth,
      depthSingleBlocks: depthSingleBlocks,
      axesDim: axesDim,
      theta: theta,
      patchSize: patchSize,
      qkvBias: qkvBias,
      approxInDim: approxInDim,
      approxOutDim: approxOutDim,
      approxHiddenDim: approxHiddenDim,
      approxNLayers: approxNLayers
    )
  }

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
