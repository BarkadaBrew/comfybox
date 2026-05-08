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
    //    Chroma's ae.safetensors uses original SD/LDKM key format while the Swift
    //    AutoencoderKL uses diffusers format. Requires comprehensive key remapping
    //    and 1x1 conv weight squeezing for attention Linear layers.
    logger.info("Loading VAE weights from \(resolvedPaths.vaePath.lastPathComponent)...")
    let rawVaeWeights = try MLX.loadArrays(url: resolvedPaths.vaePath)
    var vaeWeights: [String: MLXArray] = [:]
    var remappedCount = 0
    for (key, var tensor) in rawVaeWeights {
      // Remap key from original format to diffusers format
      let mappedKey = mapChromaVAEKey(key)
      if mappedKey != key { remappedCount += 1 }

      // Handle 4D weight tensor transformations
      if tensor.ndim == 4 {
        let isAttention = mappedKey.contains(".to_q.") || mappedKey.contains(".to_k.")
          || mappedKey.contains(".to_v.") || mappedKey.contains(".to_out.")
        if isAttention {
          // 1x1 conv weights for attention Linear layers: squeeze [out, in, 1, 1] -> [out, in]
          tensor = tensor.squeezed(axis: 3).squeezed(axis: 2)
        } else {
          // Regular conv weights: transpose NCHW -> NHWC for MLX Conv2d
          tensor = tensor.transposed(0, 2, 3, 1)
        }
      }

      if tensor.dtype != dtype {
        tensor = tensor.asType(dtype)
      }
      vaeWeights[mappedKey] = tensor
    }
    logger.info("Loaded \(vaeWeights.count) VAE weight tensors (\(remappedCount) keys remapped)")

    let vaeParams = ModuleParameters.unflattened(vaeWeights)
    try vae.update(parameters: vaeParams, verify: [.shapeMismatch])
    logger.info("VAE weights applied")

    let msg = "Chroma model loaded successfully (transformer: \(transformerWeights.count), T5: \(t5Weights.count), VAE: \(vaeWeights.count) tensors)"
    logger.info("\(msg)")

    return Components(transformer: transformer, t5: t5, vae: vae)
  }

  /// Map original-format VAE weight keys (from ae.safetensors) to diffusers-format
  /// module paths expected by the Swift AutoencoderKL.
  ///
  /// The Chroma ae.safetensors uses original SD/LDKM naming convention while
  /// the Swift AutoencoderKL uses diffusers naming. Key differences:
  ///
  /// | Original | Diffusers |
  /// |----------|-----------|
  /// | `.mid.attn_1.{q,k,v}` | `.mid_block.attentions.0.to_{q,k,v}` |
  /// | `.mid.attn_1.proj_out` | `.mid_block.attentions.0.to_out.0` |
  /// | `.mid.attn_1.norm` | `.mid_block.attentions.0.group_norm` |
  /// | `.mid.block_{1,2}` | `.mid_block.resnets.{0,1}` |
  /// | `decoder.up.N` | `decoder.up_blocks.(3-N)` *(reversed)* |
  /// | `encoder.down.N` | `encoder.down_blocks.N` *(same order)* |
  /// | `.block.M` | `.resnets.M` |
  /// | `.nin_shortcut` | `.conv_shortcut` |
  /// | `.upsample.` | `.upsamplers.0.` |
  /// | `.downsample.` | `.downsamplers.0.` |
  /// | `.norm_out.` | `.conv_norm_out.` |
  private static func mapChromaVAEKey(_ key: String) -> String {
    var k = key

    // --- Mid block (decoder and encoder share the same pattern) ---

    // Attention projections: .mid.attn_1.X -> .mid_block.attentions.0.to_X
    // proj_out must come before q/k/v to avoid partial match on "proj_out" containing no overlap,
    // but ordering is safe since patterns are distinct — kept for clarity.
    k = k.replacingOccurrences(of: ".mid.attn_1.proj_out.", with: ".mid_block.attentions.0.to_out.0.")
    k = k.replacingOccurrences(of: ".mid.attn_1.q.", with: ".mid_block.attentions.0.to_q.")
    k = k.replacingOccurrences(of: ".mid.attn_1.k.", with: ".mid_block.attentions.0.to_k.")
    k = k.replacingOccurrences(of: ".mid.attn_1.v.", with: ".mid_block.attentions.0.to_v.")
    k = k.replacingOccurrences(of: ".mid.attn_1.norm.", with: ".mid_block.attentions.0.group_norm.")

    // Resnets: .mid.block_1 -> .mid_block.resnets.0, .mid.block_2 -> .mid_block.resnets.1
    k = k.replacingOccurrences(of: ".mid.block_1.", with: ".mid_block.resnets.0.")
    k = k.replacingOccurrences(of: ".mid.block_2.", with: ".mid_block.resnets.1.")

    // --- Decoder up blocks (REVERSED ordering) ---
    // Original up.0 = highest resolution = diffusers up_blocks.3
    // Original up.3 = lowest resolution  = diffusers up_blocks.0
    for origIdx in 0...3 {
      let diffIdx = 3 - origIdx
      k = k.replacingOccurrences(
        of: "decoder.up.\(origIdx).block.",
        with: "decoder.up_blocks.\(diffIdx).resnets.")
      k = k.replacingOccurrences(
        of: "decoder.up.\(origIdx).upsample.",
        with: "decoder.up_blocks.\(diffIdx).upsamplers.0.")
    }

    // --- Encoder down blocks (SAME ordering) ---
    for idx in 0...3 {
      k = k.replacingOccurrences(
        of: "encoder.down.\(idx).block.",
        with: "encoder.down_blocks.\(idx).resnets.")
      k = k.replacingOccurrences(
        of: "encoder.down.\(idx).downsample.",
        with: "encoder.down_blocks.\(idx).downsamplers.0.")
    }

    // --- Shared mappings ---
    k = k.replacingOccurrences(of: ".nin_shortcut.", with: ".conv_shortcut.")
    k = k.replacingOccurrences(of: ".norm_out.", with: ".conv_norm_out.")

    return k
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
