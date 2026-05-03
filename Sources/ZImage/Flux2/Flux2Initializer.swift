// Flux2Initializer.swift — Model initialization and weight loading orchestration
// Ported from mflux: flux2_initializer.py

import Foundation
import Logging
import MLX
import MLXNN

/// Orchestrates initialization and weight loading for the Flux 2 Klein pipeline.
///
/// The initializer handles:
/// 1. Resolving the model snapshot (local path or HuggingFace download)
/// 2. Loading safetensors weight shards for each component
/// 3. Constructing the model modules (transformer, VAE, text encoder)
/// 4. Applying mapped weights to each module
///
/// ## Usage
///
/// ```swift
/// let components = try Flux2Initializer.load(
///   from: snapshotURL,
///   transformerConfig: config,
///   dtype: .bfloat16,
///   logger: logger
/// )
/// // components.transformer, components.vae, components.textEncoder are ready
/// ```
public enum Flux2Initializer {

  /// Loaded Flux 2 model components, ready for inference.
  public struct Components {
    /// The denoising transformer backbone.
    public let transformer: Flux2Transformer
    /// The VAE for latent encode/decode.
    public let vae: Flux2VAE
    /// The Qwen3 text encoder.
    public let textEncoder: Qwen3TextEncoder
  }

  /// Load and initialize all Flux 2 Klein model components from a snapshot.
  ///
  /// This is the primary entry point for model loading. It resolves weight files,
  /// constructs modules, loads weights from safetensors, and applies them.
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - transformerConfig: Configuration for the Flux2 transformer.
  ///   - textEncoderConfig: Configuration for the Qwen3 text encoder.
  ///   - dtype: Target data type for weights (default `.bfloat16`).
  ///   - logger: Logger for progress and diagnostic output.
  /// - Returns: Initialized `Components` with all weights loaded.
  public static func load(
    from snapshot: URL,
    transformerConfig: Flux2TransformerConfig = Flux2TransformerConfig(),
    textEncoderConfig: Qwen3TextEncoderConfiguration = Qwen3TextEncoderConfiguration(),
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws -> Components {
    logger.info("Loading Flux 2 Klein model from \(snapshot.path)")

    // 1. Resolve weight files for each component
    let transformerFiles = Flux2WeightDefinition.resolveWeightFiles(for: .transformer, at: snapshot)
    let vaeFiles = Flux2WeightDefinition.resolveWeightFiles(for: .vae, at: snapshot)
    let textEncoderFiles = Flux2WeightDefinition.resolveWeightFiles(for: .textEncoder, at: snapshot)

    logger.info("Transformer shards: \(transformerFiles.count), VAE shards: \(vaeFiles.count), Text encoder shards: \(textEncoderFiles.count)")

    // 2. Construct model modules
    let transformer = Flux2Transformer(config: transformerConfig)
    let vae = Flux2VAE()
    let textEncoder = Qwen3TextEncoder(configuration: textEncoderConfig)

    // 3. Load and map weights from safetensors
    logger.info("Loading transformer weights...")
    let transformerWeights = try Flux2WeightMapping.loadTransformerWeights(from: transformerFiles, dtype: dtype)
    logger.info("Loaded \(transformerWeights.count) transformer weight tensors")

    logger.info("Loading VAE weights...")
    let vaeWeights = try Flux2WeightMapping.loadVAEWeights(from: vaeFiles, dtype: dtype)
    logger.info("Loaded \(vaeWeights.count) VAE weight tensors")

    logger.info("Loading text encoder weights...")
    let textEncoderWeights = try Flux2WeightMapping.loadTextEncoderWeights(from: textEncoderFiles, dtype: dtype)
    logger.info("Loaded \(textEncoderWeights.count) text encoder weight tensors")

    // 4. Apply weights to modules
    logger.info("Applying weights to transformer...")
    try Flux2WeightMapping.applyTransformerWeights(transformerWeights, to: transformer)

    logger.info("Applying weights to VAE...")
    try Flux2WeightMapping.applyVAEWeights(vaeWeights, to: vae)

    logger.info("Applying weights to text encoder...")
    try Flux2WeightMapping.applyTextEncoderWeights(textEncoderWeights, to: textEncoder)

    logger.info("Flux 2 Klein model loaded successfully")

    return Components(
      transformer: transformer,
      vae: vae,
      textEncoder: textEncoder
    )
  }

  /// Load only the transformer weights from a snapshot.
  ///
  /// Useful when the transformer needs to be reloaded independently
  /// (e.g., swapping transformer checkpoints while keeping VAE/encoder).
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - transformer: The Flux2Transformer module to load weights into.
  ///   - dtype: Target data type (default `.bfloat16`).
  ///   - logger: Logger for progress output.
  public static func loadTransformer(
    from snapshot: URL,
    into transformer: Flux2Transformer,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws {
    let files = Flux2WeightDefinition.resolveWeightFiles(for: .transformer, at: snapshot)
    let weights = try Flux2WeightMapping.loadTransformerWeights(from: files, dtype: dtype)
    logger.info("Loaded \(weights.count) transformer weight tensors")
    try Flux2WeightMapping.applyTransformerWeights(weights, to: transformer)
  }

  /// Load only the VAE weights from a snapshot.
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - vae: The Flux2VAE module to load weights into.
  ///   - dtype: Target data type (default `.bfloat16`).
  ///   - logger: Logger for progress output.
  public static func loadVAE(
    from snapshot: URL,
    into vae: Flux2VAE,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws {
    let files = Flux2WeightDefinition.resolveWeightFiles(for: .vae, at: snapshot)
    let weights = try Flux2WeightMapping.loadVAEWeights(from: files, dtype: dtype)
    logger.info("Loaded \(weights.count) VAE weight tensors")
    try Flux2WeightMapping.applyVAEWeights(weights, to: vae)
  }

  /// Load only the text encoder weights from a snapshot.
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - textEncoder: The Qwen3TextEncoder module to load weights into.
  ///   - dtype: Target data type (default `.bfloat16`).
  ///   - logger: Logger for progress output.
  public static func loadTextEncoder(
    from snapshot: URL,
    into textEncoder: Qwen3TextEncoder,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws {
    let files = Flux2WeightDefinition.resolveWeightFiles(for: .textEncoder, at: snapshot)
    let weights = try Flux2WeightMapping.loadTextEncoderWeights(from: files, dtype: dtype)
    logger.info("Loaded \(weights.count) text encoder weight tensors")
    try Flux2WeightMapping.applyTextEncoderWeights(weights, to: textEncoder)
  }
}
