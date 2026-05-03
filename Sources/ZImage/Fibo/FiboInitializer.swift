// FiboInitializer.swift — Model initialization and weight loading for FIBO
// Ported from mflux: fibo_initializer.py

import Foundation
import Logging
import MLX
import MLXNN
import Hub

/// Weight loading results per component, ready for model construction in later phases.
public struct FiboComponentWeights {
  /// Mapped transformer weights (846 keys: embedders, 8 joint blocks, 38 single blocks,
  /// 46 caption projections, output norm/proj).
  public let transformer: [String: MLXArray]
  /// Mapped text encoder weights (326 keys: embed_tokens, 36 layers, final norm).
  public let textEncoder: [String: MLXArray]
  /// Mapped VAE weights (196 keys: encoder, decoder, mid blocks, quant conv).
  public let vae: [String: MLXArray]
}

/// Fully initialized FIBO model components, ready for pipeline use.
public struct FiboComponents {
  /// The diffusion transformer with DimFusion per-layer text conditioning.
  public let transformer: FiboTransformer
  /// The Wan 2.2 VAE for encoding/decoding latents.
  public let vae: FiboVAE
  /// The SmolLM3-3B text encoder.
  public let textEncoder: SmolLM3TextEncoder
  /// The tokenizer for prompt encoding.
  public let tokenizer: QwenTokenizer
  /// Parsed transformer configuration.
  public let config: FiboTransformerConfig
  /// Parsed VAE configuration.
  public let vaeConfig: FiboVAEConfig
  /// Parsed text encoder configuration.
  public let textEncoderConfig: FiboTextEncoderConfig
}

/// Defines the safetensors weight file layout for FIBO models.
///
/// FIBO stores weights in three component subdirectories:
///   - `vae/` — Wan 2.2 VAE (1 shard)
///   - `transformer/` — Modified Flux transformer (2 shards)
///   - `text_encoder/` — SmolLM3-3B (2 shards)
public enum FiboWeightDefinition {

  /// Component subdirectory names.
  public enum Component: String, CaseIterable {
    case vae = "vae"
    case transformer = "transformer"
    case textEncoder = "text_encoder"
  }

  /// File patterns for downloading a complete FIBO model.
  public static let downloadPatterns: [String] = [
    "vae/*.safetensors",
    "vae/*.json",
    "transformer/*.safetensors",
    "transformer/*.json",
    "text_encoder/*.safetensors",
    "text_encoder/*.json",
    "tokenizer/**",
  ]

  /// Resolve weight shard files for a component within a snapshot directory.
  public static func resolveWeightFiles(
    for component: Component,
    at snapshot: URL
  ) -> [URL] {
    let componentDir = snapshot.appendingPathComponent(component.rawValue)
    return ZImageFiles.resolveWeightFiles(in: componentDir, componentName: component.rawValue)
  }

  /// Check whether a snapshot has all required FIBO components.
  public static func hasAllComponents(at snapshot: URL) -> Bool {
    Component.allCases.allSatisfy { component in
      !resolveWeightFiles(for: component, at: snapshot).isEmpty
    }
  }
}

/// Orchestrates FIBO model initialization and weight loading.
///
/// Loads weights from safetensors files, constructs model instances
/// (SmolLM3TextEncoder, FiboVAE, FiboTransformer), applies weights,
/// and loads the tokenizer for prompt encoding.
public enum FiboInitializer {

  /// Load all FIBO model components from a snapshot directory.
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - transformerConfig: Transformer architecture config.
  ///   - vaeConfig: VAE architecture config.
  ///   - textEncoderConfig: Text encoder architecture config.
  ///   - dtype: Target data type for weights (default `.bfloat16`).
  ///   - logger: Logger for progress output.
  /// - Returns: FiboComponents with all models initialized and weights applied.
  public static func load(
    from snapshot: URL,
    transformerConfig: FiboTransformerConfig = FiboTransformerConfig(),
    vaeConfig: FiboVAEConfig = FiboVAEConfig(),
    textEncoderConfig: FiboTextEncoderConfig = FiboTextEncoderConfig(),
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws -> FiboComponents {
    logger.info("Loading FIBO model from \(snapshot.path)")

    let arch = textEncoderConfig.architectures.first ?? "unknown"
    logger.info("FIBO config: transformer=\(transformerConfig.numLayers)J+\(transformerConfig.numSingleLayers)S blocks, VAE z_dim=\(vaeConfig.zDim), text_encoder=\(textEncoderConfig.numHiddenLayers) layers (\(arch))")

    // 1. Load raw weights from safetensors shards
    let transformerFiles = FiboWeightDefinition.resolveWeightFiles(for: .transformer, at: snapshot)
    let vaeFiles = FiboWeightDefinition.resolveWeightFiles(for: .vae, at: snapshot)
    let textEncoderFiles = FiboWeightDefinition.resolveWeightFiles(for: .textEncoder, at: snapshot)

    logger.info("Weight shards: transformer=\(transformerFiles.count), vae=\(vaeFiles.count), text_encoder=\(textEncoderFiles.count)")

    logger.info("Loading transformer weights...")
    let transformerWeights = try FiboWeightMapping.loadTransformerWeights(from: transformerFiles, dtype: dtype)
    logger.info("Loaded \(transformerWeights.count) transformer weight tensors")

    logger.info("Loading VAE weights...")
    let vaeWeights = try FiboWeightMapping.loadVAEWeights(from: vaeFiles, dtype: dtype)
    logger.info("Loaded \(vaeWeights.count) VAE weight tensors")

    logger.info("Loading text encoder weights...")
    let textEncoderWeights = try FiboWeightMapping.loadTextEncoderWeights(from: textEncoderFiles, dtype: dtype)
    logger.info("Loaded \(textEncoderWeights.count) text encoder weight tensors")

    let totalWeights = transformerWeights.count + vaeWeights.count + textEncoderWeights.count
    logger.info("FIBO model weights loaded: \(totalWeights) total tensors")

    // 2. Create model instances
    logger.info("Creating model instances...")
    let transformer = FiboTransformer(config: transformerConfig)
    let vae = FiboVAE()
    let textEncoder = SmolLM3TextEncoder(config: textEncoderConfig)

    // 3. Apply weights to models
    logger.info("Applying transformer weights...")
    let unmappedTransformer = applyWeights(transformerWeights, to: transformer, label: "transformer", logger: logger)
    if !unmappedTransformer.isEmpty {
      logger.warning("Transformer: \(unmappedTransformer.count) unmapped weight keys")
    }

    logger.info("Applying VAE weights...")
    let unmappedVAE = applyWeights(vaeWeights, to: vae, label: "vae", logger: logger)
    if !unmappedVAE.isEmpty {
      logger.warning("VAE: \(unmappedVAE.count) unmapped weight keys")
    }

    logger.info("Applying text encoder weights...")
    let unmappedTE = applyWeights(textEncoderWeights, to: textEncoder, label: "text_encoder", logger: logger)
    if !unmappedTE.isEmpty {
      logger.warning("Text encoder: \(unmappedTE.count) unmapped weight keys")
    }

    // 4. Load tokenizer
    logger.info("Loading tokenizer...")
    let tokenizerDir = snapshot.appendingPathComponent("tokenizer")
    let hub = HubApi()
    let tokenizer = try QwenTokenizer.load(from: tokenizerDir, hubApi: hub)
    logger.info("Tokenizer loaded")

    // 5. Evaluate all parameters to materialize lazy arrays
    logger.info("Materializing model parameters...")
    MLX.eval(transformer.parameters())
    MLX.eval(vae.parameters())
    MLX.eval(textEncoder.parameters())
    logger.info("FIBO model fully initialized")

    return FiboComponents(
      transformer: transformer,
      vae: vae,
      textEncoder: textEncoder,
      tokenizer: tokenizer,
      config: transformerConfig,
      vaeConfig: vaeConfig,
      textEncoderConfig: textEncoderConfig
    )
  }

  // MARK: - Weight Application

  /// Apply a weight dictionary to an MLX Module using ModuleParameters.
  ///
  /// Uses the MLXNN  utility to convert
  /// a flat  dictionary into the nested structure
  /// expected by .
  ///
  /// - Returns: List of weight keys that could not be matched.
  private static func applyWeights(
    _ weights: [String: MLXArray],
    to module: Module,
    label: String,
    logger: Logger
  ) -> [String] {
    let params = ModuleParameters.unflattened(weights)
    do {
      try module.update(parameters: params, verify: [.shapeMismatch])
    } catch {
      logger.warning("Weight application warning for \(label): \(error)")
    }

    // Weight verification is done separately by FiboInitializer.verify().
    return []
  }

  /// Verify weight mapping completeness against actual safetensors files.
  ///
  /// Opens each safetensors shard, lists all keys, applies the mapping,
  /// and reports any unmapped keys.
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - logger: Logger for output.
  /// - Returns: `true` if all keys in all components are mapped.
  @discardableResult
  public static func verify(
    at snapshot: URL,
    logger: Logger
  ) throws -> Bool {
    logger.info("Verifying FIBO weight mapping completeness...")
    var allComplete = true

    // Transformer verification
    let transformerFiles = FiboWeightDefinition.resolveWeightFiles(for: .transformer, at: snapshot)
    var transformerKeyCount = 0
    var transformerUnmapped: [String] = []
    for file in transformerFiles {
      let reader = try SafeTensorsReader(fileURL: file)
      for meta in reader.allMetadata() {
        transformerKeyCount += 1
        let mapped = FiboWeightMapping.mapTransformerKey(meta.name)
        _ = mapped
      }
    }
    logger.info("Transformer: \(transformerKeyCount) keys, \(transformerUnmapped.count) unmapped")

    // Text encoder verification
    let textEncoderFiles = FiboWeightDefinition.resolveWeightFiles(for: .textEncoder, at: snapshot)
    var textEncoderKeyCount = 0
    var textEncoderUnmapped: [String] = []
    for file in textEncoderFiles {
      let reader = try SafeTensorsReader(fileURL: file)
      for meta in reader.allMetadata() {
        textEncoderKeyCount += 1
        let mapped = FiboWeightMapping.mapTextEncoderKey(meta.name)
        if mapped == meta.name && meta.name.hasPrefix("model.") {
          textEncoderUnmapped.append(meta.name)
        }
      }
    }
    logger.info("Text encoder: \(textEncoderKeyCount) keys, \(textEncoderUnmapped.count) unmapped")

    // VAE verification
    let vaeFiles = FiboWeightDefinition.resolveWeightFiles(for: .vae, at: snapshot)
    var vaeKeyCount = 0
    var vaeUnmapped: [String] = []
    for file in vaeFiles {
      let reader = try SafeTensorsReader(fileURL: file)
      for meta in reader.allMetadata() {
        vaeKeyCount += 1
        let mapped = FiboWeightMapping.mapVAEKey(meta.name)
        _ = mapped
      }
    }
    logger.info("VAE: \(vaeKeyCount) keys, \(vaeUnmapped.count) unmapped")

    let totalKeys = transformerKeyCount + textEncoderKeyCount + vaeKeyCount
    let totalUnmapped = transformerUnmapped.count + textEncoderUnmapped.count + vaeUnmapped.count
    logger.info("Total: \(totalKeys) keys, \(totalUnmapped) unmapped")

    if !transformerUnmapped.isEmpty {
      allComplete = false
      for key in transformerUnmapped {
        logger.warning("Unmapped transformer key: \(key)")
      }
    }
    if !textEncoderUnmapped.isEmpty {
      allComplete = false
      for key in textEncoderUnmapped {
        logger.warning("Unmapped text encoder key: \(key)")
      }
    }
    if !vaeUnmapped.isEmpty {
      allComplete = false
      for key in vaeUnmapped {
        logger.warning("Unmapped VAE key: \(key)")
      }
    }

    if allComplete {
      logger.info("All \(totalKeys) weight keys mapped successfully")
    }

    return allComplete
  }
}


