// LTX2Pipeline.swift -- Main pipeline orchestrator for LTX-2 video generation
// Phase 4 of the LTX-2 Swift/MLX port
//
// Wires the VAE (Phase 1), Text Encoder (Phase 2), and Transformer (Phase 3)
// into a complete video generation pipeline supporting:
// - Text-to-Video (T2V): prompt -> noise -> denoise -> decode -> frames
// - Image-to-Video (I2V): prompt + image -> encode -> condition -> denoise -> decode -> frames
// - Two-stage distilled: low-res denoise -> upsample latents -> high-res denoise -> decode
//
// The pipeline handles the full flow:
// 1. Encode prompt via TextEncoder -> embeddings
// 2. Create or condition initial noise
// 3. Build position grid for RoPE
// 4. Denoise via Transformer (Euler or res_2s sampler)
// 5. Apply CFG guidance (dev pipeline)
// 6. Decode latents via VAE -> RGB frames
// 7. Post-process frames to MP4
//
// Reference: generate.py — the full pipeline logic

import Foundation
import Logging
import MLX
import MLXRandom
import MLXNN

// MARK: - Pipeline Output

/// Output from the LTX-2 video generation pipeline.
public struct LTX2PipelineOutput {
  /// Decoded video tensor `(B, 3, F, H, W)` in float32, range [0, 1].
  public let decoded: MLXArray

  /// Number of generated frames.
  public let numFrames: Int

  /// Output width in pixels.
  public let width: Int

  /// Output height in pixels.
  public let height: Int

  /// Total generation time in seconds.
  public let elapsedSeconds: Double
}

// MARK: - Pipeline

/// Main LTX-2 video generation pipeline.
///
/// Orchestrates VAE, TextEncoder, and Transformer into a complete
/// generation pipeline for T2V and I2V.
public final class LTX2Pipeline {

  /// The 3D Video VAE (encode/decode).
  public let vae: LTX2VAE

  /// The text encoder (Gemma 3 + connectors).
  public let textEncoder: LTX2TextEncoder

  /// The denoising transformer.
  public let transformer: LTX2Transformer

  /// Pipeline configuration.
  public let config: LTX2PipelineConfig

  /// Optional latent upsampler for two-stage pipeline.
  public let upsampler: LTX2LatentUpsampler?

  /// Logger.
  private let logger: Logger

  /// VAE spatial compression factor.
  public var spatialCompression: Int { vae.spatialCompression }

  /// VAE temporal compression factor.
  public var temporalCompression: Int { vae.temporalCompression }

  /// Transformer inner dimension.
  public var innerDim: Int { transformer.innerDim }

  /// Initialize the pipeline with pre-created components.
  ///
  /// - Parameters:
  ///   - vae: The LTX-2 VAE.
  ///   - textEncoder: The text encoder.
  ///   - transformer: The denoising transformer.
  ///   - config: Pipeline configuration.
  ///   - upsampler: Optional latent upsampler for two-stage.
  ///   - logger: Logger instance.
  public init(
    vae: LTX2VAE,
    textEncoder: LTX2TextEncoder,
    transformer: LTX2Transformer,
    config: LTX2PipelineConfig,
    upsampler: LTX2LatentUpsampler? = nil,
    logger: Logger = Logger(label: "ltx2.pipeline")
  ) {
    self.vae = vae
    self.textEncoder = textEncoder
    self.transformer = transformer
    self.config = config
    self.upsampler = upsampler
    self.logger = logger
  }

  // MARK: - Text-to-Video

  /// Generate a video from a text prompt.
  ///
  /// - Parameters:
  ///   - inputIds: Pre-tokenized input IDs `[B, S]`.
  ///   - attentionMask: Attention mask `[B, S]`.
  ///   - width: Output width in pixels (must be divisible by 32).
  ///   - height: Output height in pixels (must be divisible by 32).
  ///   - numFrames: Number of output frames. Must be `1 + 8k`.
  ///   - steps: Number of denoising steps.
  ///   - seed: Optional random seed for reproducibility.
  ///   - guidance: CFG guidance scale. Overrides config if provided.
  ///   - negativeInputIds: Negative prompt token IDs for CFG.
  ///   - negativeAttentionMask: Negative prompt attention mask.
  ///   - progressCallback: Called after each denoising step with `(currentStep, totalSteps)`.
  /// - Returns: Pipeline output with decoded frames.
  public func generateT2V(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    width: Int,
    height: Int,
    numFrames: Int,
    steps: Int,
    seed: UInt64? = nil,
    guidance: Float? = nil,
    negativeInputIds: MLXArray? = nil,
    negativeAttentionMask: MLXArray? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) -> LTX2PipelineOutput {
    let startTime = CFAbsoluteTimeGetCurrent()
    let cfgScale = guidance ?? config.guidance

    // Validate frame count
    precondition((numFrames - 1) % 8 == 0, "numFrames must be 1 + 8k (got \(numFrames))")
    precondition(width % spatialCompression == 0, "width must be divisible by \(spatialCompression)")
    precondition(height % spatialCompression == 0, "height must be divisible by \(spatialCompression)")

    logger.info("T2V generation: \(width)x\(height), \(numFrames) frames, \(steps) steps")

    // Step 1: Encode prompt
    logger.info("Encoding prompt...")
    let textOutput = textEncoder.encode(
      inputIds: inputIds,
      attentionMask: attentionMask,
      returnAudioEmbeddings: false
    )
    eval(textOutput.videoEmbeddings)

    // Encode negative prompt for CFG (dev pipeline)
    var negativeEmbeddings: MLXArray? = nil
    if cfgScale > 1.0, let negIds = negativeInputIds, let negMask = negativeAttentionMask {
      logger.info("Encoding negative prompt for CFG (scale=\(cfgScale))...")
      let negOutput = textEncoder.encode(
        inputIds: negIds,
        attentionMask: negMask,
        returnAudioEmbeddings: false
      )
      negativeEmbeddings = negOutput.videoEmbeddings
      eval(negativeEmbeddings!)
    }

    // Step 2: Compute latent dimensions
    let latH = height / spatialCompression
    let latW = width / spatialCompression
    let latF = (numFrames - 1) / temporalCompression + 1

    logger.info("Latent dimensions: \(latF) frames x \(latH) x \(latW)")

    // Step 3: Create initial noise
    if let seed = seed {
      MLXRandom.seed(seed)
    }

    let sigmas = getSigmaSchedule(steps: steps, latF: latF, latH: latH, latW: latW)

    // Scale initial noise by sigma_max
    let noise = MLXRandom.normal([1, 128, latF, latH, latW], dtype: .float32)
    var latents = noise * MLXArray(sigmas[0])
    eval(latents)

    // Step 4: Build position grid for RoPE
    let positions = createPositionGrid(
      batchSize: 1, latF: latF, latH: latH, latW: latW
    )

    // Step 5: Precompute RoPE
    let precomputedPE = ltx2PrecomputeFreqsCIS(
      indicesGrid: positions,
      dim: innerDim,
      theta: transformer.positionalEmbeddingTheta,
      maxPos: transformer.positionalEmbeddingMaxPos,
      useMiddleIndicesGrid: transformer.useMiddleIndicesGrid,
      numAttentionHeads: transformer.numHeads,
      ropeMode: transformer.ropeMode,
      doublePrecision: transformer.doublePrecisionRoPE
    )
    eval(precomputedPE.cos, precomputedPE.sin)

    // Step 6: Denoising loop
    logger.info("Denoising (\(config.sampler) sampler, \(sigmas.count - 1) steps)...")
    latents = denoisingLoop(
      latents: latents,
      positions: positions,
      precomputedPE: precomputedPE,
      textEmbeddings: textOutput.videoEmbeddings,
      negativeEmbeddings: negativeEmbeddings,
      sigmas: sigmas,
      cfgScale: cfgScale,
      state: nil,
      progressCallback: progressCallback
    )

    // Step 7: Decode latents via VAE
    logger.info("Decoding latents via VAE...")
    let decoded: MLXArray
    if config.tiledDecode {
      decoded = vae.decodeTiled(latents.asType(.bfloat16))
    } else {
      decoded = vae.decode(latents.asType(.bfloat16))
    }
    eval(decoded)

    // Convert from [-1, 1] to [0, 1] range (VAE outputs centered at 0)
    let rescaled = (decoded.asType(.float32) + 1.0) / 2.0
    let clamped = MLX.clip(rescaled, min: 0, max: 1)
    eval(clamped)

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("Generation complete in \(String(format: "%.1f", elapsed))s")

    return LTX2PipelineOutput(
      decoded: clamped,
      numFrames: numFrames,
      width: width,
      height: height,
      elapsedSeconds: elapsed
    )
  }

  // MARK: - Text-to-Video with Pre-computed Embeddings

  /// Generate a video from pre-computed text embeddings, bypassing the text encoder.
  ///
  /// Use this when you want to supply your own embeddings (e.g. from a cached
  /// encoder run, or dummy embeddings for pipeline testing) without loading
  /// the Gemma 3 text encoder weights.
  ///
  /// - Parameters:
  ///   - videoEmbeddings: Pre-computed text embeddings `[B, S, 4096]`.
  ///   - negativeEmbeddings: Optional negative embeddings for CFG.
  ///   - width: Output width in pixels (must be divisible by 32).
  ///   - height: Output height in pixels (must be divisible by 32).
  ///   - numFrames: Number of output frames. Must be `1 + 8k`.
  ///   - steps: Number of denoising steps.
  ///   - seed: Optional random seed for reproducibility.
  ///   - guidance: CFG guidance scale. Overrides config if provided.
  ///   - progressCallback: Called after each denoising step with `(currentStep, totalSteps)`.
  /// - Returns: Pipeline output with decoded frames.
  public func generateT2VWithEmbeddings(
    videoEmbeddings: MLXArray,
    negativeEmbeddings: MLXArray? = nil,
    width: Int,
    height: Int,
    numFrames: Int,
    steps: Int,
    seed: UInt64? = nil,
    guidance: Float? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) -> LTX2PipelineOutput {
    let startTime = CFAbsoluteTimeGetCurrent()
    let cfgScale = guidance ?? config.guidance

    // Validate frame count
    precondition((numFrames - 1) % 8 == 0, "numFrames must be 1 + 8k (got \(numFrames))")
    precondition(width % spatialCompression == 0, "width must be divisible by \(spatialCompression)")
    precondition(height % spatialCompression == 0, "height must be divisible by \(spatialCompression)")

    logger.info("T2V (embeddings) generation: \(width)x\(height), \(numFrames) frames, \(steps) steps")

    // Use embeddings directly -- skip text encoder
    eval(videoEmbeddings)

    var negEmb: MLXArray? = nil
    if cfgScale > 1.0, let neg = negativeEmbeddings {
      negEmb = neg
      eval(negEmb!)
    }

    // Compute latent dimensions
    let latH = height / spatialCompression
    let latW = width / spatialCompression
    let latF = (numFrames - 1) / temporalCompression + 1

    logger.info("Latent dimensions: \(latF) frames x \(latH) x \(latW)")

    // Create initial noise
    if let seed = seed {
      MLXRandom.seed(seed)
    }

    let sigmas = getSigmaSchedule(steps: steps, latF: latF, latH: latH, latW: latW)

    // Scale initial noise by sigma_max
    let noise = MLXRandom.normal([1, 128, latF, latH, latW], dtype: .float32)
    var latents = noise * MLXArray(sigmas[0])
    eval(latents)

    // Build position grid for RoPE
    let positions = createPositionGrid(
      batchSize: 1, latF: latF, latH: latH, latW: latW
    )

    // Precompute RoPE
    let precomputedPE = ltx2PrecomputeFreqsCIS(
      indicesGrid: positions,
      dim: innerDim,
      theta: transformer.positionalEmbeddingTheta,
      maxPos: transformer.positionalEmbeddingMaxPos,
      useMiddleIndicesGrid: transformer.useMiddleIndicesGrid,
      numAttentionHeads: transformer.numHeads,
      ropeMode: transformer.ropeMode,
      doublePrecision: transformer.doublePrecisionRoPE
    )
    eval(precomputedPE.cos, precomputedPE.sin)

    // Denoising loop
    logger.info("Denoising (\(config.sampler) sampler, \(sigmas.count - 1) steps)...")
    latents = denoisingLoop(
      latents: latents,
      positions: positions,
      precomputedPE: precomputedPE,
      textEmbeddings: videoEmbeddings,
      negativeEmbeddings: negEmb,
      sigmas: sigmas,
      cfgScale: cfgScale,
      state: nil,
      progressCallback: progressCallback
    )

    // Decode latents via VAE
    logger.info("Decoding latents via VAE...")
    let decoded: MLXArray
    if config.tiledDecode {
      decoded = vae.decodeTiled(latents.asType(.bfloat16))
    } else {
      decoded = vae.decode(latents.asType(.bfloat16))
    }
    eval(decoded)

    // Convert from [-1, 1] to [0, 1] range (VAE outputs centered at 0)
    let rescaled = (decoded.asType(.float32) + 1.0) / 2.0
    let clamped = MLX.clip(rescaled, min: 0, max: 1)
    eval(clamped)

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("Generation complete in \(String(format: "%.1f", elapsed))s")

    return LTX2PipelineOutput(
      decoded: clamped,
      numFrames: numFrames,
      width: width,
      height: height,
      elapsedSeconds: elapsed
    )
  }

  // MARK: - Image-to-Video

  /// Generate a video from a text prompt and input image.
  ///
  /// The input image is encoded via the VAE, injected at frame 0, and
  /// the remaining frames are denoised conditioned on it.
  ///
  /// - Parameters:
  ///   - inputIds: Pre-tokenized prompt IDs `[B, S]`.
  ///   - attentionMask: Attention mask `[B, S]`.
  ///   - image: Input image tensor `(1, 3, H, W)` in float32 [0, 1].
  ///   - strength: Motion strength (0.0 = static, 1.0 = full motion). Default 1.0.
  ///   - width: Output width in pixels.
  ///   - height: Output height in pixels.
  ///   - numFrames: Number of output frames. Must be `1 + 8k`.
  ///   - steps: Number of denoising steps.
  ///   - seed: Optional random seed.
  ///   - guidance: CFG guidance scale.
  ///   - negativeInputIds: Negative prompt token IDs for CFG.
  ///   - negativeAttentionMask: Negative prompt attention mask.
  ///   - progressCallback: Called after each step.
  /// - Returns: Pipeline output with decoded frames.
  public func generateI2V(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    image: MLXArray,
    strength: Float = 1.0,
    width: Int,
    height: Int,
    numFrames: Int,
    steps: Int,
    seed: UInt64? = nil,
    guidance: Float? = nil,
    negativeInputIds: MLXArray? = nil,
    negativeAttentionMask: MLXArray? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) -> LTX2PipelineOutput {
    let startTime = CFAbsoluteTimeGetCurrent()
    let cfgScale = guidance ?? config.guidance

    precondition((numFrames - 1) % 8 == 0, "numFrames must be 1 + 8k (got \(numFrames))")

    logger.info("I2V generation: \(width)x\(height), \(numFrames) frames, strength=\(strength)")

    // Step 1: Encode prompt
    logger.info("Encoding prompt...")
    let textOutput = textEncoder.encode(
      inputIds: inputIds,
      attentionMask: attentionMask,
      returnAudioEmbeddings: false
    )
    eval(textOutput.videoEmbeddings)

    var negativeEmbeddings: MLXArray? = nil
    if cfgScale > 1.0, let negIds = negativeInputIds, let negMask = negativeAttentionMask {
      let negOutput = textEncoder.encode(
        inputIds: negIds, attentionMask: negMask, returnAudioEmbeddings: false
      )
      negativeEmbeddings = negOutput.videoEmbeddings
      eval(negativeEmbeddings!)
    }

    // Step 2: Encode input image via VAE
    logger.info("Encoding input image via VAE...")
    var imageInput = image
    if imageInput.ndim == 4 {
      imageInput = imageInput.expandedDimensions(axis: 2)
    }
    let imageLatent = vae.encode(imageInput)
    eval(imageLatent)

    // Step 3: Compute latent dimensions
    let latH = height / spatialCompression
    let latW = width / spatialCompression
    let latF = (numFrames - 1) / temporalCompression + 1

    // Step 4: Create initial noisy state with I2V conditioning
    if let seed = seed {
      MLXRandom.seed(seed)
    }

    let sigmas = getSigmaSchedule(steps: steps, latF: latF, latH: latH, latW: latW)

    // Create initial state
    var state = LTX2Conditioning.createInitialState(
      shape: [1, 128, latF, latH, latW],
      noiseScale: sigmas[0]
    )

    // Apply image conditioning at frame 0
    let condition = LTX2VideoCondition(
      latent: imageLatent,
      frameIndex: 0,
      strength: strength
    )
    state = LTX2Conditioning.applyConditioning(state: state, conditions: [condition])
    eval(state.latent, state.cleanLatent, state.denoiseMask)

    // Step 5: Build position grid
    let positions = createPositionGrid(
      batchSize: 1, latF: latF, latH: latH, latW: latW
    )

    // Precompute RoPE
    let precomputedPE = ltx2PrecomputeFreqsCIS(
      indicesGrid: positions,
      dim: innerDim,
      theta: transformer.positionalEmbeddingTheta,
      maxPos: transformer.positionalEmbeddingMaxPos,
      useMiddleIndicesGrid: transformer.useMiddleIndicesGrid,
      numAttentionHeads: transformer.numHeads,
      ropeMode: transformer.ropeMode,
      doublePrecision: transformer.doublePrecisionRoPE
    )
    eval(precomputedPE.cos, precomputedPE.sin)

    // Step 6: Denoising loop with I2V state
    logger.info("Denoising with I2V conditioning...")
    let latents = denoisingLoop(
      latents: state.latent,
      positions: positions,
      precomputedPE: precomputedPE,
      textEmbeddings: textOutput.videoEmbeddings,
      negativeEmbeddings: negativeEmbeddings,
      sigmas: sigmas,
      cfgScale: cfgScale,
      state: state,
      progressCallback: progressCallback
    )

    // Step 7: Decode
    logger.info("Decoding latents via VAE...")
    let decoded: MLXArray
    if config.tiledDecode {
      decoded = vae.decodeTiled(latents.asType(.bfloat16))
    } else {
      decoded = vae.decode(latents.asType(.bfloat16))
    }
    eval(decoded)

    let rescaled = (decoded.asType(.float32) + 1.0) / 2.0
    let clamped = MLX.clip(rescaled, min: 0, max: 1)
    eval(clamped)

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("I2V generation complete in \(String(format: "%.1f", elapsed))s")

    return LTX2PipelineOutput(
      decoded: clamped,
      numFrames: numFrames,
      width: width,
      height: height,
      elapsedSeconds: elapsed
    )
  }

  // MARK: - Multi-Keyframe ("tween") Generation

  /// One keyframe image anchoring the generated video at a specific point in
  /// its timeline — the primitive behind multi-keyframe "tween" generation.
  public struct Keyframe {
    /// Normalized pixel image, same format `generateI2V`'s `image:` expects.
    public let image: MLXArray
    /// Target position in VIDEO frames (0 = first frame) — converted
    /// internally to a latent-frame index via the VAE's temporal compression.
    public let videoFrameIndex: Int
    /// Denoising strength at this keyframe (1.0 = fully replace with the image).
    public let strength: Float

    public init(image: MLXArray, videoFrameIndex: Int, strength: Float = 1.0) {
      self.image = image
      self.videoFrameIndex = videoFrameIndex
      self.strength = strength
    }
  }

  /// Generate a video conditioned on MULTIPLE keyframe images placed at
  /// different points in the timeline, letting the transformer's own
  /// temporal self-attention interpolate ("tween") between them during
  /// denoising. There is no separate interpolation step — each keyframe is
  /// spliced into the dense latent grid exactly like `generateI2V`'s single
  /// frame-0 image, just at its own frame index. `LTX2Conditioning.
  /// applyConditioning` already accepted a list of conditions; until this
  /// method, it was only ever called with one (frame 0). Positions here come
  /// from `createPositionGrid` over the WHOLE dense grid uniformly (not a
  /// separately-offset token sequence, unlike the Lightricks Python
  /// reference's frame>0 path) — mid-sequence keyframes are architecturally
  /// consistent with frame-0 conditioning in this port, but see
  /// docs/ltx2-multi-keyframe-fdd.md for the empirical verification (a real
  /// two-keyframe render, both keyframes landed correctly with no
  /// corruption) and the caveats around transition smoothness.
  public func generateMultiKeyframe(
    inputIds: MLXArray,
    attentionMask: MLXArray,
    keyframes: [Keyframe],
    width: Int,
    height: Int,
    numFrames: Int,
    steps: Int,
    seed: UInt64? = nil,
    guidance: Float? = nil,
    negativeInputIds: MLXArray? = nil,
    negativeAttentionMask: MLXArray? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) -> LTX2PipelineOutput {
    precondition(!keyframes.isEmpty, "generateMultiKeyframe requires at least one keyframe")
    precondition((numFrames - 1) % 8 == 0, "numFrames must be 1 + 8k (got \(numFrames))")

    let startTime = CFAbsoluteTimeGetCurrent()
    let cfgScale = guidance ?? config.guidance

    logger.info("Multi-keyframe generation: \(width)x\(height), \(numFrames) frames, \(keyframes.count) keyframe(s)")

    // Step 1: Encode prompt
    logger.info("Encoding prompt...")
    let textOutput = textEncoder.encode(
      inputIds: inputIds, attentionMask: attentionMask, returnAudioEmbeddings: false
    )
    eval(textOutput.videoEmbeddings)

    var negativeEmbeddings: MLXArray? = nil
    if cfgScale > 1.0, let negIds = negativeInputIds, let negMask = negativeAttentionMask {
      let negOutput = textEncoder.encode(
        inputIds: negIds, attentionMask: negMask, returnAudioEmbeddings: false
      )
      negativeEmbeddings = negOutput.videoEmbeddings
      eval(negativeEmbeddings!)
    }

    // Step 2: Compute latent dimensions (needed to convert video-frame ->
    // latent-frame indices before encoding each keyframe).
    let latH = height / spatialCompression
    let latW = width / spatialCompression
    let latF = (numFrames - 1) / temporalCompression + 1

    // Step 3: Encode each keyframe image via VAE.
    logger.info("Encoding \(keyframes.count) keyframe image(s) via VAE...")
    let conditions: [LTX2VideoCondition] = keyframes.map { kf in
      var imageInput = kf.image
      if imageInput.ndim == 4 {
        imageInput = imageInput.expandedDimensions(axis: 2)
      }
      let latent = vae.encode(imageInput)
      eval(latent)
      let latentFrameIndex = min(kf.videoFrameIndex / temporalCompression, latF - 1)
      return LTX2VideoCondition(latent: latent, frameIndex: latentFrameIndex, strength: kf.strength)
    }

    // Step 4: Create initial noisy state, apply ALL keyframe conditions.
    if let seed = seed {
      MLXRandom.seed(seed)
    }
    let sigmas = getSigmaSchedule(steps: steps, latF: latF, latH: latH, latW: latW)
    var state = LTX2Conditioning.createInitialState(
      shape: [1, 128, latF, latH, latW],
      noiseScale: sigmas[0]
    )
    state = LTX2Conditioning.applyConditioning(state: state, conditions: conditions)
    eval(state.latent, state.cleanLatent, state.denoiseMask)

    // Step 5: Build position grid
    let positions = createPositionGrid(
      batchSize: 1, latF: latF, latH: latH, latW: latW
    )
    let precomputedPE = ltx2PrecomputeFreqsCIS(
      indicesGrid: positions,
      dim: innerDim,
      theta: transformer.positionalEmbeddingTheta,
      maxPos: transformer.positionalEmbeddingMaxPos,
      useMiddleIndicesGrid: transformer.useMiddleIndicesGrid,
      numAttentionHeads: transformer.numHeads,
      ropeMode: transformer.ropeMode,
      doublePrecision: transformer.doublePrecisionRoPE
    )
    eval(precomputedPE.cos, precomputedPE.sin)

    // Step 6: Denoise
    logger.info("Denoising with \(keyframes.count)-keyframe conditioning...")
    let latents = denoisingLoop(
      latents: state.latent,
      positions: positions,
      precomputedPE: precomputedPE,
      textEmbeddings: textOutput.videoEmbeddings,
      negativeEmbeddings: negativeEmbeddings,
      sigmas: sigmas,
      cfgScale: cfgScale,
      state: state,
      progressCallback: progressCallback
    )

    // Step 7: Decode
    logger.info("Decoding latents via VAE...")
    let decoded: MLXArray
    if config.tiledDecode {
      decoded = vae.decodeTiled(latents.asType(.bfloat16))
    } else {
      decoded = vae.decode(latents.asType(.bfloat16))
    }
    eval(decoded)

    let rescaled = (decoded.asType(.float32) + 1.0) / 2.0
    let clamped = MLX.clip(rescaled, min: 0, max: 1)
    eval(clamped)

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("Multi-keyframe generation complete in \(String(format: "%.1f", elapsed))s")

    return LTX2PipelineOutput(
      decoded: clamped,
      numFrames: numFrames,
      width: width,
      height: height,
      elapsedSeconds: elapsed
    )
  }

  // MARK: - Internal: Denoising Loop

  /// Core denoising loop shared by T2V and I2V.
  private func denoisingLoop(
    latents: MLXArray,
    positions: MLXArray,
    precomputedPE: (cos: MLXArray, sin: MLXArray),
    textEmbeddings: MLXArray,
    negativeEmbeddings: MLXArray?,
    sigmas: [Float],
    cfgScale: Float,
    state: LTX2LatentState?,
    progressCallback: ((Int, Int) -> Void)?
  ) -> MLXArray {
    let dtype: DType = .bfloat16
    let useCFG = cfgScale > 1.0 && negativeEmbeddings != nil
    let numSteps = sigmas.count - 1

    // Keep latents in float32 throughout for precision
    var currentLatents = latents.asType(.float32)

    for i in 0..<numSteps {
      let sigma = sigmas[i]
      let sigmaNext = sigmas[i + 1]

      let b = currentLatents.dim(0)
      let c = currentLatents.dim(1)
      let f = currentLatents.dim(2)
      let h = currentLatents.dim(3)
      let w = currentLatents.dim(4)
      let numTokens = f * h * w

      // Flatten latents to token space: (B, C, F, H, W) -> (B, T, C)
      let latentsFlat = currentLatents
        .reshaped(b, c, -1)
        .transposed(0, 2, 1)
        .asType(dtype)

      // Compute per-token timesteps
      let timesteps: MLXArray
      if let s = state {
        // I2V: per-token timesteps based on denoise mask
        let maskBroadcast = MLX.broadcast(
          s.denoiseMask.reshaped(b, 1, f, 1, 1),
          to: [b, 1, f, h, w]
        )
        let maskFlat = maskBroadcast.reshaped(b, numTokens)
        timesteps = MLXArray(sigma).asType(dtype) * maskFlat.asType(dtype)
      } else {
        timesteps = MLXArray(sigma).reshaped(1, 1).asType(dtype)
      }

      let sigmaArray = MLXArray([sigma]).asType(dtype)

      // Positive pass through transformer.
      // Pass PER-TOKEN timesteps so the model sees conditioned frames (mask=0)
      // at timestep 0 (clean) and generated frames at `sigma`. Without this the
      // transformer treats every token as equally noisy and cannot tell which
      // frame is the I2V conditioning frame — it discounts the image and
      // produces T2V-like output. Matches reference guided_denoise_loop which
      // passes `video_timesteps = denoise_mask * sigma` when the mask is
      // non-uniform. For T2V, `timesteps` is a scalar (1,1) → broadcast, so this
      // is a no-op there.
      let velocityPos = transformer(
        latent: latentsFlat,
        timestep: timesteps,
        context: textEmbeddings.asType(dtype),
        positions: positions,
        sigma: sigmaArray,
        precomputedPE: precomputedPE
      )

      // Compute x0 (denoised) from velocity using per-token timesteps
      let latentsFlatF32 = currentLatents
        .reshaped(b, c, -1)
        .transposed(0, 2, 1)
      let timestepsF32 = timesteps.asType(.float32).expandedDimensions(axis: -1)
      var x0GuidedF32 = latentsFlatF32 - timestepsF32 * velocityPos.asType(.float32)

      // CFG: negative pass
      if useCFG, let negEmb = negativeEmbeddings {
        let velocityNeg = transformer(
          latent: latentsFlat,
          timestep: timesteps,
          context: negEmb.asType(dtype),
          positions: positions,
          sigma: sigmaArray,
          precomputedPE: precomputedPE
        )
        eval(velocityNeg)

        let x0NegF32 = latentsFlatF32 - timestepsF32 * velocityNeg.asType(.float32)
        x0GuidedF32 = LTX2Guidance.applyCFG(
          conditioned: x0GuidedF32,
          unconditioned: x0NegF32,
          scale: cfgScale
        )
      }

      // Reshape x0 from token space to spatial
      var denoised = x0GuidedF32
        .transposed(0, 2, 1)
        .reshaped(b, c, f, h, w)

      // Apply I2V denoise mask
      if let s = state {
        denoised = LTX2Conditioning.applyDenoiseMask(
          denoised: denoised,
          clean: s.cleanLatent.asType(.float32),
          denoiseMask: s.denoiseMask
        )
      }

      // Euler step in float32
      if sigmaNext > 0 {
        let sigmaF32 = MLXArray(sigma)
        let sigmaNextF32 = MLXArray(sigmaNext)
        currentLatents = denoised + sigmaNextF32 * (currentLatents - denoised) / sigmaF32
      } else {
        currentLatents = denoised
      }

      eval(currentLatents)
      progressCallback?(i + 1, numSteps)
    }

    return currentLatents
  }

  // MARK: - Internal: Sigma Schedule

  /// Get sigma schedule based on pipeline type.
  private func getSigmaSchedule(
    steps: Int, latF: Int, latH: Int, latW: Int
  ) -> [Float] {
    switch config.pipelineType {
    case .distilled:
      return LTX2PipelineConfig.stage1Sigmas
    case .dev, .devTwoStage:
      let numTokens = latF * latH * latW
      return LTX2PipelineConfig.devSigmaSchedule(
        steps: steps,
        numTokens: numTokens
      )
    }
  }

  // MARK: - Internal: Position Grid

  /// Create a 3D position grid for RoPE in pixel space.
  ///
  /// Matches the Python `create_position_grid` function with causal fix
  /// and bfloat16 precision quantization.
  private func createPositionGrid(
    batchSize: Int,
    latF: Int,
    latH: Int,
    latW: Int
  ) -> MLXArray {
    let temporalScale = Float(temporalCompression)
    let spatialScale = Float(spatialCompression)
    let fps = Float(config.fps)
    let numPatches = latF * latH * latW

    // Build position indices
    var positions = [Float](repeating: 0, count: batchSize * 3 * numPatches * 2)

    for b in 0..<batchSize {
      for f in 0..<latF {
        for h in 0..<latH {
          for w in 0..<latW {
            let tokenIdx = f * latH * latW + h * latW + w
            let baseIdx = b * 3 * numPatches * 2

            // Time start/end in pixel space
            var tStart = Float(f) * temporalScale
            var tEnd = Float(f + 1) * temporalScale

            // Causal fix: shift temporal coordinates
            tStart = max(0, tStart + 1 - temporalScale)
            tEnd = max(0, tEnd + 1 - temporalScale)

            // Divide temporal by fps
            tStart /= fps
            tEnd /= fps

            // Spatial in pixel space
            let hStart = Float(h) * spatialScale
            let hEnd = Float(h + 1) * spatialScale
            let wStart = Float(w) * spatialScale
            let wEnd = Float(w + 1) * spatialScale

            // Time dimension
            positions[baseIdx + 0 * numPatches * 2 + tokenIdx * 2 + 0] = tStart
            positions[baseIdx + 0 * numPatches * 2 + tokenIdx * 2 + 1] = tEnd

            // Height dimension
            positions[baseIdx + 1 * numPatches * 2 + tokenIdx * 2 + 0] = hStart
            positions[baseIdx + 1 * numPatches * 2 + tokenIdx * 2 + 1] = hEnd

            // Width dimension
            positions[baseIdx + 2 * numPatches * 2 + tokenIdx * 2 + 0] = wStart
            positions[baseIdx + 2 * numPatches * 2 + tokenIdx * 2 + 1] = wEnd
          }
        }
      }
    }

    // Cast through bfloat16 to match PyTorch precision behavior
    let posArray = MLXArray(positions, [batchSize, 3, numPatches, 2])
    let bf16 = posArray.asType(.bfloat16)
    eval(bf16)
    return bf16.asType(.float32)
  }
}
