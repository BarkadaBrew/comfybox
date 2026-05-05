// ChromaInitializer.swift — Model initialization and weight loading for Chroma
// Follows the Flux2Initializer pattern: resolve files, load weights, apply to modules.

import Foundation
import Logging
import MLX
import MLXNN

/// Orchestrates initialization and weight loading for the Chroma pipeline.
///
/// The initializer handles:
/// 1. Locating weight files via ChromaModelDetection.resolveComponentPaths
/// 2. Constructing empty model modules (transformer, T5, VAE)
/// 3. Sanitizing and mapping weight keys to Swift module paths
/// 4. Applying weights to each module
///
/// Chroma reuses the FLUX.1 VAE decoder (AutoencoderKL) and has its own T5-XXL text encoder
/// and Chroma transformer with Approximator.
///
/// ## Usage
///
/// ```swift
/// let components = try ChromaInitializer.load(
///   from: snapshotURL,
///   config: .standard,
///   dtype: .bfloat16,
///   logger: logger
/// )
/// // components.transformer, components.t5, components.vae are ready
/// ```
public enum ChromaInitializer {

  /// Loaded Chroma model components, ready for inference.
  public struct Components {
    /// The Chroma denoising transformer (with Approximator).
    public let transformer: ChromaTransformer
    /// The T5-XXL text encoder.
    public let t5: T5Encoder
    /// The VAE for latent decode (FLUX.1 AutoencoderKL).
    public let vae: AutoencoderKL
  }

  /// Load and initialize all Chroma model components from a snapshot directory.
  ///
  /// - Parameters:
  ///   - snapshot: Root URL of the model snapshot directory.
  ///   - paths: Pre-resolved component paths. If nil, resolves automatically.
  ///   - config: Configuration for the Chroma transformer (default `.standard`).
  ///   - t5Config: Configuration for the T5-XXL encoder (default `.xxl`).
  ///   - dtype: Target data type for weights (default `.bfloat16`).
  ///   - logger: Logger for progress and diagnostic output.
  /// - Returns: Initialized `Components` with all weights loaded.
  public static func load(
    from snapshot: URL,
    paths: ChromaComponentPaths? = nil,
    config: ChromaConfig = .standard,
    t5Config: T5Config = .xxl,
    dtype: DType = .bfloat16,
    logger: Logger
  ) throws -> Components {
    logger.info("Loading Chroma model from \(snapshot.path)")

    // Resolve component paths
    guard let resolvedPaths = paths ?? ChromaModelDetection.resolveComponentPaths(at: snapshot) else {
      throw ChromaInitializerError.componentsNotFound(
        "Could not resolve Chroma model components at \(snapshot.path)"
      )
    }

    // 1. Construct empty model modules
    let transformer = ChromaTransformer(config: config)
    let t5 = T5Encoder(config: t5Config)
    let vae = AutoencoderKL()

    // 2. Load and apply transformer weights
    logger.info("Loading Chroma transformer weights from \(resolvedPaths.transformerPath.lastPathComponent)...")
    let rawTransformerWeights = try MLX.loadArrays(url: resolvedPaths.transformerPath)
    let transformerWeights = ChromaWeightMapping.sanitize(rawTransformerWeights)
    logger.info("Loaded \(transformerWeights.count) transformer weight tensors")

    let transformerParams = ModuleParameters.unflattened(transformerWeights)
    try transformer.update(parameters: transformerParams, verify: [.shapeMismatch])
    logger.info("Chroma transformer weights applied")

    // 3. Load and apply T5-XXL text encoder weights (sharded)
    logger.info("Loading T5-XXL text encoder weights (\(resolvedPaths.t5Paths.count) shards)...")
    var rawT5Weights: [String: MLXArray] = [:]
    for t5Path in resolvedPaths.t5Paths {
      logger.info("  Loading shard: \(t5Path.lastPathComponent)")
      let shard = try MLX.loadArrays(url: t5Path)
      for (key, value) in shard {
        rawT5Weights[key] = value
      }
    }
    let t5Weights = T5Encoder.sanitizeWeights(rawT5Weights)
    logger.info("Loaded \(t5Weights.count) T5 weight tensors")

    let t5Params = ModuleParameters.unflattened(t5Weights)
    try t5.update(parameters: t5Params, verify: [.shapeMismatch])
    logger.info("T5-XXL weights applied")

    // 4. Load and apply VAE weights
    //    Conv2d: PyTorch NCHW -> MLX NHWC transpose (same as Flux2)
    logger.info("Loading VAE weights from \(resolvedPaths.vaePath.lastPathComponent)...")
    let rawVaeWeights = try MLX.loadArrays(url: resolvedPaths.vaePath)
    var vaeWeights: [String: MLXArray] = [:]
    for (key, var tensor) in rawVaeWeights {
      // Transpose conv2d weights for MLX layout
      if key.hasSuffix(".weight") && tensor.ndim == 4
        && (key.contains(".conv") || key.contains("quant_conv") || key.contains("post_quant_conv"))
      {
        tensor = tensor.transposed(0, 2, 3, 1)
      }
      if tensor.dtype != dtype {
        tensor = tensor.asType(dtype)
      }
      // Same key mapping as Flux2 VAE: .to_out.0. -> .to_out.
      var mappedKey = key
      mappedKey = mappedKey.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
      vaeWeights[mappedKey] = tensor
    }
    logger.info("Loaded \(vaeWeights.count) VAE weight tensors")

    let vaeParams = ModuleParameters.unflattened(vaeWeights)
    try vae.update(parameters: vaeParams, verify: [.shapeMismatch])
    logger.info("VAE weights applied")

    let msg = "Chroma model loaded successfully (transformer: \(transformerWeights.count), T5: \(t5Weights.count), VAE: \(vaeWeights.count) tensors)"
    logger.info("\(msg)")

    return Components(transformer: transformer, t5: t5, vae: vae)
  }
}

public enum ChromaInitializerError: Error, LocalizedError {
  case componentsNotFound(String)
  case weightLoadFailed(String)

  public var errorDescription: String? {
    switch self {
    case .componentsNotFound(let path):
      return "Chroma model components not found: \(path)"
    case .weightLoadFailed(let msg):
      return "Weight loading failed: \(msg)"
    }
  }
}
