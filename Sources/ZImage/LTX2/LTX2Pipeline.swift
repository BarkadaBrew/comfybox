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

    // Encode negative prompt for CFG, or for CFG++ (needs it every step at cfg=1).
    var negativeEmbeddings: MLXArray? = nil
    if cfgScale > 1.0 || LTX2PipelineConfig.envCfgPP,
       let negIds = negativeInputIds, let negMask = negativeAttentionMask {
      logger.info("Encoding negative prompt (scale=\(cfgScale), cfgPP=\(LTX2PipelineConfig.envCfgPP))...")
      let negOutput = textEncoder.encode(
        inputIds: negIds,
        attentionMask: negMask,
        returnAudioEmbeddings: false
      )
      negativeEmbeddings = negOutput.videoEmbeddings
      eval(negativeEmbeddings!)
    }

    // NAG conditioning. The reference wires this as a separate model-patch
    // input (`LTX2_NAG {nag_cond_video}`), NOT as the CFG negative — in that
    // recipe CFG sits at 1.0 so the CFG negative is inert. With no dedicated
    // NAG prompt we reuse the negative embeddings (same text in practice),
    // encoding them here if CFG/CFG++ didn't already.
    let nagConfig = LTX2NAGConfig.fromEnvironment()
    var nagEmbeddings: MLXArray? = nil
    if nagConfig.isEnabled {
      if let negIds = negativeInputIds, let negMask = negativeAttentionMask {
        if let already = negativeEmbeddings {
          nagEmbeddings = already
        } else {
          let out = textEncoder.encode(
            inputIds: negIds, attentionMask: negMask, returnAudioEmbeddings: false)
          eval(out.videoEmbeddings)
          nagEmbeddings = out.videoEmbeddings
        }
        logger.info("NAG enabled (scale \(nagConfig.scale), alpha \(nagConfig.alpha), tau \(nagConfig.tau)).")
      } else {
        logger.warning("NAG configured but no negative prompt supplied — NAG requires a negative conditioning; skipping.")
      }
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
      nagEmbeddings: nagEmbeddings, nag: nagConfig,
      progressCallback: progressCallback
    )

    // Step 6b: Two-stage refine (env LTX2_TWO_STAGE) — identical machinery to the
    // i2v refine, minus the frame-0 identity re-anchor (t2v has no source frame,
    // so the refine runs free with state:nil). Upsample x2 -> flow re-noise ->
    // short deterministic refine denoise.
    if let ups = self.upsampler, ProcessInfo.processInfo.environment["LTX2_TWO_STAGE"] == "1" {
      MLX.GPU.clearCache()
      logger.info("T2V two-stage refine: upsampling latents 2x...")
      let stats = vae.decoder.perChannelStatistics
      let upLatent = stats.normalize(ups(stats.unNormalize(latents.asType(.float32)))).asType(.float32)
      eval(upLatent)
      // Refine-volume OOM gate — same guard as the i2v path (Codex review
      // 2026-07-26: t2v could still crash the process at large formats).
      let t2vRefineVolume = upLatent.dim(2) * (latH * 2) * (latW * 2)
      let t2vRefineMax = Int(ProcessInfo.processInfo.environment["LTX2_REFINE_MAX_VOL"] ?? "") ?? 12_000
      if t2vRefineVolume > t2vRefineMax {
        logger.info("T2V two-stage refine: SKIPPED denoise (volume \(t2vRefineVolume) > \(t2vRefineMax)) — decoding upsampled latent directly (OOM guard).")
        latents = upLatent
      } else {
      let rLatH = latH * 2, rLatW = latW * 2
      let refineSigmas: [Float] = (ProcessInfo.processInfo.environment["LTX2_REFINE_SIGMAS"]).flatMap { s -> [Float]? in
        let v = s.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        return v.count >= 2 ? v : nil
      } ?? [0.85, 0.7250, 0.4219, 0.0]
      if let seed = seed { MLXRandom.seed(seed &+ 1000) }
      let refNoise = MLXRandom.normal(upLatent.shape).asType(.float32)
      let sigma0 = refineSigmas[0]
      let refineInit = MLXArray(1 - sigma0) * upLatent + refNoise * MLXArray(sigma0)  // flow re-noise
      let refinePos = createPositionGrid(batchSize: 1, latF: latF, latH: rLatH, latW: rLatW)
      let refinePE = ltx2PrecomputeFreqsCIS(
        indicesGrid: refinePos, dim: innerDim,
        theta: transformer.positionalEmbeddingTheta,
        maxPos: transformer.positionalEmbeddingMaxPos,
        useMiddleIndicesGrid: transformer.useMiddleIndicesGrid,
        numAttentionHeads: transformer.numHeads,
        ropeMode: transformer.ropeMode,
        doublePrecision: transformer.doublePrecisionRoPE)
      eval(refinePE.cos, refinePE.sin)
      logger.info("T2V two-stage refine: denoising at \(rLatW * spatialCompression)x\(rLatH * spatialCompression), \(refineSigmas.count - 1) steps...")
      latents = denoisingLoop(
        latents: refineInit, positions: refinePos, precomputedPE: refinePE,
        textEmbeddings: textOutput.videoEmbeddings, negativeEmbeddings: negativeEmbeddings,
        sigmas: refineSigmas, cfgScale: cfgScale, state: nil, nagEmbeddings: nagEmbeddings, nag: nagConfig,
        forceDeterministic: ProcessInfo.processInfo.environment["LTX2_REFINE_DETERMINISTIC"] != "0",
        progressCallback: progressCallback)
      eval(latents)
      MLX.GPU.clearCache()
      logger.info("T2V two-stage refine complete.")
      }
    }

    // Step 7: Decode latents via VAE
    logger.info("Decoding latents via VAE...")
    let decoded = decodeAdaptive(latents)
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
    let decoded = decodeAdaptive(latents)
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
    faceAnchorMask: MLXArray? = nil,
    faceAnchorStrength: Float = 0,
    refineAnchorImage: MLXArray? = nil,
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
    // CFG++ samplers need the negative embeddings every step even at cfg=1.
    if cfgScale > 1.0 || LTX2PipelineConfig.envCfgPP,
       let negIds = negativeInputIds, let negMask = negativeAttentionMask {
      let negOutput = textEncoder.encode(
        inputIds: negIds, attentionMask: negMask, returnAudioEmbeddings: false
      )
      negativeEmbeddings = negOutput.videoEmbeddings
      eval(negativeEmbeddings!)
    }

    // NAG conditioning — see generateT2V for the rationale. The reference
    // recipe is i2v-centric, so this path matters at least as much.
    let nagConfig = LTX2NAGConfig.fromEnvironment()
    var nagEmbeddings: MLXArray? = nil
    if nagConfig.isEnabled {
      if let negIds = negativeInputIds, let negMask = negativeAttentionMask {
        if let already = negativeEmbeddings {
          nagEmbeddings = already
        } else {
          let out = textEncoder.encode(
            inputIds: negIds, attentionMask: negMask, returnAudioEmbeddings: false)
          eval(out.videoEmbeddings)
          nagEmbeddings = out.videoEmbeddings
        }
        logger.info("NAG enabled (scale \(nagConfig.scale), alpha \(nagConfig.alpha), tau \(nagConfig.tau)).")
      } else {
        logger.warning("NAG configured but no negative prompt supplied — NAG requires a negative conditioning; skipping.")
      }
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
    if let fm = faceAnchorMask, faceAnchorStrength > 0 {
      state.faceMask = fm
      state.faceRef = imageLatent  // source-encoded latent = the face identity reference
      state.faceAnchorStrength = faceAnchorStrength
    }

    // IC-control (union-control IC-LoRA): append the source-encoded latent as
    // extra reference frames at temporal frame_idx 0, FROZEN (mask=1-strength).
    // The merged union-control IC-LoRA makes self-attention lock structure to
    // these reference frames -> coherent motion instead of morph. Env-gated
    // LTX2_IC_CONTROL; MVP uses reference_downscale_factor=1 (no dilation).
    // Mirrors ComfyUI LTXVAddGuide.append_keyframe (frame axis cat) +
    // add_keyframe_index (temporal position -> frame_idx 0, handled in
    // createPositionGrid via refFrames). Slice back to latF before decode.
    var icRefFrames = 0
    // IC-control (identity anchor) defaults ON for i2v: without it the subject
    // morphs into a different person over the render. LTX2_IC_CONTROL=0 disables.
    if ProcessInfo.processInfo.environment["LTX2_IC_CONTROL"] != "0" {
      let refStrength = Float(ProcessInfo.processInfo.environment["LTX2_IC_REF_STRENGTH"] ?? "1.0") ?? 1.0
      icRefFrames = imageLatent.dim(2)
      state.latent = MLX.concatenated([state.latent, imageLatent.asType(state.latent.dtype)], axis: 2)
      state.cleanLatent = MLX.concatenated([state.cleanLatent, imageLatent.asType(state.cleanLatent.dtype)], axis: 2)
      let refMask = MLX.broadcast(
        MLXArray(Float(1.0 - refStrength)).reshaped(1, 1, 1, 1, 1),
        to: [1, 1, icRefFrames, 1, 1])
      state.denoiseMask = MLX.concatenated([state.denoiseMask, refMask], axis: 2)
      logger.info("IC-control: appended \(icRefFrames) reference frame(s) @ strength \(refStrength)")
    }
    eval(state.latent, state.cleanLatent, state.denoiseMask)

    // Step 5: Build position grid (extended by icRefFrames; ref frames get
    // temporal frame_idx 0 inside createPositionGrid).
    let positions = createPositionGrid(
      batchSize: 1, latF: latF, latH: latH, latW: latW, refFrames: icRefFrames
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
    let latentsAll = denoisingLoop(
      latents: state.latent,
      positions: positions,
      precomputedPE: precomputedPE,
      textEmbeddings: textOutput.videoEmbeddings,
      negativeEmbeddings: negativeEmbeddings,
      sigmas: sigmas,
      cfgScale: cfgScale,
      state: state,
      nagEmbeddings: nagEmbeddings, nag: nagConfig,
      progressCallback: progressCallback
    )

    // Drop the appended IC-control reference frames before decode (keep latF).
    var latents = icRefFrames > 0
      ? latentsAll[0..., 0..., 0..<latF, 0..., 0...]
      : latentsAll

    // Two-stage refine — shared with continuation chunks (applyTwoStageRefine).
    latents = applyTwoStageRefine(
      latents, latF: latF, latH: latH, latW: latW,
      textEmbeddings: textOutput.videoEmbeddings,
      negativeEmbeddings: negativeEmbeddings,
      cfgScale: cfgScale, seed: seed,
      refineAnchorImage: refineAnchorImage,
      progressCallback: progressCallback)

    // Step 7: Decode
    logger.info("Decoding latents via VAE...")
    let decoded = decodeAdaptive(latents)
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
    if cfgScale > 1.0 || LTX2PipelineConfig.envCfgPP, let negIds = negativeInputIds, let negMask = negativeAttentionMask {
      let negOutput = textEncoder.encode(
        inputIds: negIds, attentionMask: negMask, returnAudioEmbeddings: false
      )
      negativeEmbeddings = negOutput.videoEmbeddings
      eval(negativeEmbeddings!)
    }

    // NAG conditioning. The reference wires this as a separate model-patch
    // input (`LTX2_NAG {nag_cond_video}`), NOT as the CFG negative — in that
    // recipe CFG sits at 1.0 so the CFG negative is inert. With no dedicated
    // NAG prompt we reuse the negative embeddings (same text in practice),
    // encoding them here if CFG/CFG++ didn't already.
    let nagConfig = LTX2NAGConfig.fromEnvironment()
    var nagEmbeddings: MLXArray? = nil
    if nagConfig.isEnabled {
      if let negIds = negativeInputIds, let negMask = negativeAttentionMask {
        if let already = negativeEmbeddings {
          nagEmbeddings = already
        } else {
          let out = textEncoder.encode(
            inputIds: negIds, attentionMask: negMask, returnAudioEmbeddings: false)
          eval(out.videoEmbeddings)
          nagEmbeddings = out.videoEmbeddings
        }
        logger.info("NAG enabled (scale \(nagConfig.scale), alpha \(nagConfig.alpha), tau \(nagConfig.tau)).")
      } else {
        logger.warning("NAG configured but no negative prompt supplied — NAG requires a negative conditioning; skipping.")
      }
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

    // Step 6b: Two-stage refine — SAME treatment as generateI2V. Continuation
    // chunks previously skipped refine entirely, so chunks 2+ of every
    // multi-chunk render were base-resolution decodes upscaled by the muxer
    // (visible quality cliff at each chunk boundary; blockiness rose 1.14 ->
    // 1.43 across the 2026-07-26 03:59 filed video).
    let refinedLatents = applyTwoStageRefine(
      latents, latF: latF, latH: latH, latW: latW,
      textEmbeddings: textOutput.videoEmbeddings,
      negativeEmbeddings: negativeEmbeddings,
      cfgScale: cfgScale, seed: seed,
      refineAnchorImage: nil,
      progressCallback: progressCallback)

    // Step 7: Decode
    logger.info("Decoding latents via VAE...")
    let decoded = decodeAdaptive(refinedLatents)
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
    nagEmbeddings: MLXArray? = nil,
    nag: LTX2NAGConfig = .disabled,
    forceDeterministic: Bool = false,
    progressCallback: ((Int, Int) -> Void)?
  ) -> MLXArray {
    let dtype: DType = .bfloat16
    let numSteps = sigmas.count - 1
    // Per-step CFG schedule (community "CFG ramp": e.g. LTX2_CFG_SCHEDULE=3,2,1 —
    // CFG only on the first steps, where broad motion forms, then back to 1;
    // 10Eros per-step guider pattern). Missing entries extend the last value.
    let cfgSchedule: [Float]? = ProcessInfo.processInfo.environment["LTX2_CFG_SCHEDULE"].flatMap { s in
      let v = s.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
      return v.isEmpty ? nil : v
    }
    func cfgAt(_ step: Int) -> Float {
      guard let sch = cfgSchedule else { return cfgScale }
      return sch[min(step, sch.count - 1)]
    }
    let useCFG = (cfgScale > 1.0 || (cfgSchedule?.contains { $0 > 1.0 } ?? false)) && negativeEmbeddings != nil

    // --- Denoise-trajectory instrumentation (env-gated, off by default) ---
    // Set LTX2_TRAJ_DUMP=<path-prefix> to append one JSON line per step to
    // <path-prefix>.jsonl with latent/x0/velocity statistics. Used to diff
    // sampler behavior numerically against reference implementations
    // (e.g. ComfyUI). Zero overhead when the env var is unset.
    let trajPrefix = ProcessInfo.processInfo.environment["LTX2_TRAJ_DUMP"]
    var trajHandle: FileHandle? = nil
    // Mean over all elements.
    func trajMean(_ x: MLXArray) -> MLXArray { x.mean() }
    // Std over all elements (population).
    func trajStd(_ x: MLXArray) -> MLXArray {
      let m = x.mean()
      return MLX.sqrt(((x - m) * (x - m)).mean())
    }
    // Weighted mean/std over token subset. x: (B, T, C) f32, w: (B, T) f32 0/1.
    func trajMaskedStats(_ x: MLXArray, _ w: MLXArray) -> (mean: MLXArray, std: MLXArray) {
      let w3 = w.expandedDimensions(axis: -1)
      let channels = Float(x.dim(-1))
      let count = w.sum() * channels + 1e-6
      let m = (x * w3).sum() / count
      let v = (((x - m) * (x - m)) * w3).sum() / count
      return (m, MLX.sqrt(v))
    }

    // Keep latents in float32 throughout for precision
    var currentLatents = latents.asType(.float32)
    // Refine pass uses the deterministic member of the sampler family (validated
    // PinkCherry recipe: base=euler_ancestral_cfg_pp, refine=euler_cfg_pp), so
    // forceDeterministic drops the ancestral/SDE noise but keeps CFG++ mode.
    let useSDE = LTX2PipelineConfig.envAncestral && !forceDeterministic
    // CFG++ (euler[_ancestral]_cfg_pp): needs an unconditional (negative) pass
    // every step regardless of cfgScale.
    let useCfgPP = LTX2PipelineConfig.envCfgPP && negativeEmbeddings != nil

    for i in 0..<numSteps {
      let sigma = sigmas[i]
      let sigmaNext = sigmas[i + 1]

      // INPUT-side conditioning clamp (matches ComfyUI KSamplerX0Inpaint +
      // LTXV.scale_latent_inpaint which returns the CLEAN latent, never
      // re-noised): conditioned tokens (denoiseMask < 1) enter the model as
      // clean latent every step, consistent with their per-token timestep of
      // ~0. Without this the step update re-mixes noise into conditioned
      // tokens, so the model reads a NOISY frame while told it's clean —
      // misreading the reference each step (appearance divergence, i2v
      // transition pop). Output-side x0 clamping alone is insufficient.
      if let s = state {
        let m = s.denoiseMask.asType(.float32)
        currentLatents = currentLatents * m + s.cleanLatent.asType(currentLatents.dtype) * (MLXArray(Float(1)) - m)
      }

      // Snapshot pre-step latents for delta_norm (post input-clamp so the
      // delta measures the denoise update; no compute or eval unless
      // trajectory dumping is enabled).
      let trajLatentsBefore: MLXArray? = trajPrefix != nil ? currentLatents : nil

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
      // NAG rides ONLY the positive pass: it steers the conditional prediction
      // away from the negative concept inside cross-attention. Applying it to
      // the unconditional pass too would guide the baseline that CFG++ steps
      // along, double-counting the guidance.
      let velocityPos = transformer(
        latent: latentsFlat,
        timestep: timesteps,
        context: textEmbeddings.asType(dtype),
        positions: positions,
        sigma: sigmaArray,
        precomputedPE: precomputedPE,
        nagContext: nagEmbeddings?.asType(dtype),
        nag: nag
      )

      // Compute x0 (denoised) from velocity using per-token timesteps
      let latentsFlatF32 = currentLatents
        .reshaped(b, c, -1)
        .transposed(0, 2, 1)
      let timestepsF32 = timesteps.asType(.float32).expandedDimensions(axis: -1)
      var x0GuidedF32 = latentsFlatF32 - timestepsF32 * velocityPos.asType(.float32)
      let x0CondF32 = x0GuidedF32  // pure conditional x0, saved for the STG delta

      // Negative pass: needed for classic CFG (scale>1) and for CFG++ (every
      // step, even at cfg=1 — the CFG++ update steps along the UNCONDITIONAL
      // noise direction; see ComfyUI sample_euler_ancestral_cfg_pp which sets
      // disable_cfg1_optimization=True for exactly this reason).
      var x0UncondF32: MLXArray? = nil
      if useCFG || useCfgPP, let negEmb = negativeEmbeddings {
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
        x0UncondF32 = x0NegF32
        if useCFG && cfgAt(i) > 1.0 {
          x0GuidedF32 = LTX2Guidance.applyCFG(
            conditioned: x0GuidedF32,
            unconditioned: x0NegF32,
            scale: cfgAt(i)
          )
          // Guidance rescale (Lin et al.; counters CFG over-saturation /
          // shadow-crush). High CFG inflates the prediction's std -> boosted
          // color/contrast (measured: cfg2 saturation +19% vs seed). Rescale
          // the guided x0 back toward the CONDITIONAL prediction's std, then
          // blend by phi — keeps the action-motion boost, restores seed color.
          // env LTX2_GUIDANCE_RESCALE (0 = off); per-request override next.
          let phi = Float(ProcessInfo.processInfo.environment["LTX2_GUIDANCE_RESCALE"] ?? "") ?? 0
          if phi > 0 {
            let stdCond = MLX.sqrt(((x0CondF32 - x0CondF32.mean()) * (x0CondF32 - x0CondF32.mean())).mean())
            let stdGuided = MLX.sqrt(((x0GuidedF32 - x0GuidedF32.mean()) * (x0GuidedF32 - x0GuidedF32.mean())).mean())
            let rescaled = x0GuidedF32 * (stdCond / (stdGuided + 1e-6))
            x0GuidedF32 = MLXArray(phi) * rescaled + MLXArray(1 - phi) * x0GuidedF32
          }
        }
      }

      // STG: spatiotemporal guidance. A perturbed pass skips self-attention in a
      // block subset; steer away from it to restore high-frequency motion detail.
      // For distilled cfg=1 this is the primary guidance lever (anti-haze).
      let stgBase = LTX2PipelineConfig.envSTGScale
      if stgBase > 0 {
        let stgScale = LTX2PipelineConfig.stgScaleForStep(i, base: stgBase)
        let velocitySTG = transformer(
          latent: latentsFlat,
          timestep: timesteps,
          context: textEmbeddings.asType(dtype),
          positions: positions,
          sigma: sigmaArray,
          precomputedPE: precomputedPE,
          stgBlocks: LTX2PipelineConfig.envSTGBlocks
        )
        eval(velocitySTG)
        let x0STGF32 = latentsFlatF32 - timestepsF32 * velocitySTG.asType(.float32)
        x0GuidedF32 = x0GuidedF32 + MLXArray(stgScale) * (x0CondF32 - x0STGF32)
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
      // Face-region identity hold: pull ONLY the masked face latents (all frames)
      // toward the source face, softly + every step. No full-frame splicing, so
      // nothing to collapse between anchors — holds a stationary partner's face.
      if let s = state, let fm = s.faceMask, let ref = s.faceRef, s.faceAnchorStrength > 0 {
        let m = fm.asType(.float32)
        let wm = m * s.faceAnchorStrength
        denoised = denoised * (MLXArray(Float(1)) - wm) + ref.asType(.float32) * wm
      }

      // Euler step (ancestral/SDE when LTX2_SAMPLER=euler_ancestral) in float32
      if sigmaNext > 0 {
        if useCfgPP, let x0Neg = x0UncondF32 {
          // CFG++ step (ComfyUI sample_euler_ancestral_cfg_pp, flow/CONST model:
          // exp(lambda(σ)) = (1-σ)/σ so alpha_s = 1-σ, alpha_t = 1-σ_next).
          // Direction from the UNCONDITIONAL x0, target the conditional x0:
          //   d = (x - alpha_s·x0_uncond) / σ
          //   x = alpha_t·x0_cond + σ_down·d  (+ alpha_t·σ_up·noise, ancestral)
          let x0UncondSpatial = x0Neg
            .transposed(0, 2, 1)
            .reshaped(b, c, f, h, w)
          // Guard the σ=1.0 first step: alphaS→0 makes sf=σ/alphaS blow up to inf
          // and NaN the ancestral term. Clamp alphaS to a small floor.
          let alphaS = max(1.0 - sigma, 1e-4)
          let alphaT = 1.0 - sigmaNext
          let d = (currentLatents - MLXArray(alphaS) * x0UncondSpatial) / MLXArray(sigma)
          var sigmaDown = sigmaNext
          var sigmaUp: Float = 0
          if useSDE {
            // get_ancestral_step(σ/alpha_s, σ_next/alpha_t, eta=1), then
            // σ_down = alpha_t · σ_down'
            let sf = sigma / alphaS
            let st = sigmaNext / alphaT
            let inner = st * st * (sf * sf - st * st) / (sf * sf)
            let up = min(st, (inner > 0 ? inner : 0).squareRoot())
            sigmaDown = alphaT * (max(st * st - up * up, 0)).squareRoot()
            sigmaUp = up
          }
          currentLatents = MLXArray(alphaT) * denoised + MLXArray(sigmaDown) * d
          if useSDE && sigmaUp > 0 {
            // Plain Gaussian ancestral noise, matching ComfyUI's noise_sampler
            // (randn_like). getNewNoise's per-channel normalization flattens the
            // stochasticity that drives inter-frame motion — do NOT use it here.
            let noise = MLXRandom.normal(currentLatents.shape, dtype: .float32)
            currentLatents = currentLatents + MLXArray(alphaT) * noise * MLXArray(sigmaUp)
          }
        } else if useSDE {
          let sigmaF32 = MLXArray(sigma)
          let eps = (currentLatents - denoised) / sigmaF32
          let (alphaRatio, sigmaDown, sigmaUp) = getSdeCoeff(sigmaNext: sigmaNext)
          let noise = MLXRandom.normal(currentLatents.shape, dtype: .float32)
          currentLatents = MLXArray(alphaRatio) * (denoised + MLXArray(sigmaDown) * eps)
            + MLXArray(sigmaUp) * noise
        } else {
          let sigmaF32 = MLXArray(sigma)
          let sigmaNextF32 = MLXArray(sigmaNext)
          currentLatents = denoised + sigmaNextF32 * (currentLatents - denoised) / sigmaF32
        }
      } else {
        currentLatents = denoised
      }
      // CFG++ steps along the uncond direction, which drifts frames the denoise
      // mask pins at timestep 0 (I2V conditioning / refine re-inject); re-snap
      // them to the clean latent after every step (ComfyUI does the equivalent
      // in its outer inpaint wrapper).
      if useCfgPP, let s = state {
        currentLatents = LTX2Conditioning.applyDenoiseMask(
          denoised: currentLatents,
          clean: s.cleanLatent.asType(.float32),
          denoiseMask: s.denoiseMask
        )
      }

      eval(currentLatents)

      // --- Trajectory dump (after step update) ---
      if let prefix = trajPrefix, let before = trajLatentsBefore {
        // Scalar stats via small MLX reductions; only scalars are evaluated.
        let velocityF32 = velocityPos.asType(.float32)
        let deltaAbs = MLX.abs(currentLatents - before)

        var fields: [(String, Float)] = [
          ("sigma", sigma),
          ("sigmaNext", sigmaNext),
          ("latent_mean", trajMean(currentLatents).item(Float.self)),
          ("latent_std", trajStd(currentLatents).item(Float.self)),
          ("x0_mean", trajMean(denoised).item(Float.self)),
          ("x0_std", trajStd(denoised).item(Float.self)),
          ("velocity_mean_abs", trajMean(MLX.abs(velocityF32)).item(Float.self)),
          ("delta_norm", trajMean(deltaAbs).item(Float.self)),
          // Plain Euler step: no ancestral noise injection (sigmaUp == 0).
          ("noise_injected", 0),
        ]

        // Conditioned vs generated token subsets (I2V only).
        if let s = state {
          let maskFlat = MLX.broadcast(
            s.denoiseMask.reshaped(b, 1, f, 1, 1),
            to: [b, 1, f, h, w]
          ).reshaped(b, numTokens).asType(.float32)
          let genW = (maskFlat .> 0.5).asType(.float32)
          let condW = 1.0 - genW

          // Token-space (B, T, C) views for subset reductions.
          let curFlat = currentLatents.reshaped(b, c, -1).transposed(0, 2, 1)
          let x0Flat = denoised.reshaped(b, c, -1).transposed(0, 2, 1)
          let deltaFlat = deltaAbs.reshaped(b, c, -1).transposed(0, 2, 1)

          for (suffix, weight) in [("cond", condW), ("gen", genW)] {
            let latentStats = trajMaskedStats(curFlat, weight)
            let x0Stats = trajMaskedStats(x0Flat, weight)
            let deltaMean = (deltaFlat * weight.expandedDimensions(axis: -1)).sum()
              / (weight.sum() * Float(c) + 1e-6)
            fields.append(("latent_mean_\(suffix)", latentStats.mean.item(Float.self)))
            fields.append(("latent_std_\(suffix)", latentStats.std.item(Float.self)))
            fields.append(("x0_mean_\(suffix)", x0Stats.mean.item(Float.self)))
            fields.append(("x0_std_\(suffix)", x0Stats.std.item(Float.self)))
            fields.append(("delta_norm_\(suffix)", deltaMean.item(Float.self)))
          }
        }

        // Serialize one JSON line and append (create file on first step).
        var jsonParts = ["\"step\":\(i)"]
        for (key, value) in fields {
          let v = value.isFinite ? String(format: "%.8e", value) : "null"
          jsonParts.append("\"\(key)\":\(v)")
        }
        let line = "{" + jsonParts.joined(separator: ",") + "}\n"

        if trajHandle == nil {
          // Each denoise pass gets its own file (prefix.jsonl, prefix.2.jsonl, …)
          // so a two-stage render's refine pass doesn't clobber the stage-1 dump.
          var path = prefix + ".jsonl"
          var pass = 1
          while FileManager.default.fileExists(atPath: path) {
            pass += 1
            path = prefix + ".\(pass).jsonl"
          }
          FileManager.default.createFile(atPath: path, contents: nil)
          trajHandle = FileHandle(forWritingAtPath: path)
          if trajHandle == nil {
            logger.warning("LTX2_TRAJ_DUMP: cannot open \(path) for writing")
          }
        }
        if let handle = trajHandle, let data = line.data(using: .utf8) {
          handle.seekToEndOfFile()
          handle.write(data)
          try? handle.synchronize()
        }
      }

      progressCallback?(i + 1, numSteps)
    }

    if let handle = trajHandle {
      try? handle.close()
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
      return LTX2PipelineConfig.envStage1Sigmas ?? LTX2PipelineConfig.stage1Sigmas
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
  /// Adaptive VAE decode. Plain single-pass decode is clean (no spatial-tile
  /// mosaic, no temporal-window jitter) but memory-heavy; tiled decode is
  /// OOM-safe for long/large clips. Decide by latent frame count: plain for
  /// normal clips, exact streamed decode above the safety gate. Mode switch:
  /// LTX2_DECODE_MODE=auto|stream|tile|plain. Threshold tunable
  /// via LTX2_PLAIN_DECODE_MAX_LATF (latent frames; ~8x fewer than output frames).
  private func decodeAdaptive(_ latents: MLXArray) -> MLXArray {
    let latF = latents.dim(2)
    let latH = latents.dim(3)
    let latW = latents.dim(4)
    // Gate on TOTAL latent volume, not frame count alone: a 768x1280 clip at
    // latF 13 (24x40 spatial) is 3x the volume of 448x704 and plain decode at
    // that size drove 40GB+ of swap with critical memory pressure mid-decode,
    // corrupting the output (blank/gray frames). Budget = the proven-safe
    // 448x704x97f working point (13*14*22 = 4004) with ~2x headroom.
    let volume = latF * latH * latW
    // Default budget derived from the Metal max-buffer ceiling, measured
    // empirically (2026-07-24): plain-decoding 49f @ 768x1280 (volume 6720,
    // ~25GB peak intermediate) SILENTLY CORRUPTS the last ~60% of frames —
    // no error, just garbage — while tiled decode of the same tensor is clean.
    // 4004 (97f @ 448x704) is proven-safe; 4500 keeps margin below the cliff.
    let plainMaxVolume = Int(ProcessInfo.processInfo.environment["LTX2_PLAIN_DECODE_MAX_VOL"] ?? "") ?? 4500
    // LTX2_DUMP_LATENT=<path>: save the pre-decode latent for offline decode
    // debugging (real-values reproduction harness, #36).
    if let dumpPath = ProcessInfo.processInfo.environment["LTX2_DUMP_LATENT"] {
      try? MLX.save(arrays: ["latent": latents.asType(.float32)], url: URL(fileURLWithPath: dumpPath))
      logger.info("VAE decode: dumped pre-decode latent to \(dumpPath)")
    }
    // LTX2_DECODE_F32=1: decode in float32 instead of bf16 (debug probe).
    let decodeF32 = ProcessInfo.processInfo.environment["LTX2_DECODE_F32"] == "1"
    let bf = latents.asType(decodeF32 ? .float32 : .bfloat16)

    // Decode mode — ONE switch (LTX2_DECODE_MODE): auto | stream | tile | plain.
    //   auto (default): plain under the corruption-safe volume gate, exact
    //     streamed chunked-io above it (bit-identical to plain, zero seams).
    //   stream: streamed always. tile: legacy spatial tiling (LTX2_DECODE_TILE).
    //   plain: force single-pass — EXPERT/DEBUG ONLY: above the gate, MLX Metal
    //     silently corrupts boundary frames (int32 offset overflow, mlx #3836;
    //     unfixed in any mlx-swift release as of 0.31.6).
    // Legacy alias honored: LTX2_DECODE_STREAM=0 -> tile. config.tiledDecode
    // no longer disables the safety path (Codex review 2026-07-26: a default-
    // constructed config could force a corrupting plain decode).
    let env = ProcessInfo.processInfo.environment
    let mode = (env["LTX2_DECODE_MODE"] ?? (env["LTX2_DECODE_STREAM"] == "0" ? "tile" : "auto")).lowercased()
    switch mode {
    case "plain":
      if volume > plainMaxVolume {
        logger.warning("VAE decode: FORCED plain above the safety gate (volume \(volume) > \(plainMaxVolume)) — output may be silently corrupt (mlx #3836).")
      }
      logger.info("VAE decode: plain single-pass (forced; volume \(volume) [\(latF)x\(latH)x\(latW)]).")
      return vae.decode(bf)
    case "tile":
      let tile = Int(env["LTX2_DECODE_TILE"] ?? "") ?? 16
      logger.info("VAE decode: tiled (mode=tile; volume \(volume) [\(latF)x\(latH)x\(latW)]) — tile \(tile).")
      return vae.decodeTiled(bf, tileSize: tile, tileStride: tile - 2)
    case "stream":
      logger.info("VAE decode: streamed exact chunked-io (mode=stream; volume \(volume) [\(latF)x\(latH)x\(latW)]) — zero seams.")
      return vae.decodeStreamed(bf)
    default:  // auto
      if volume > plainMaxVolume {
        logger.info("VAE decode: streamed exact chunked-io (volume \(volume) [\(latF)x\(latH)x\(latW)] > \(plainMaxVolume)) — zero seams.")
        return vae.decodeStreamed(bf)
      }
      logger.info("VAE decode: plain single-pass (volume \(volume) [\(latF)x\(latH)x\(latW)]) — under the safety gate.")
      return vae.decode(bf)
    }
  }

  /// Shared two-stage refine (Phase 3): latent upsample x2 -> re-noise ->
  /// short refine denoise at high res, with the refine-volume OOM gate and
  /// optional native-res frame-0 re-anchor. Used by generateI2V AND
  /// generateMultiKeyframe (continuation chunks) so every chunk of a
  /// multi-chunk render gets identical refine treatment — previously chunks
  /// 2+ skipped refine entirely, dropping to base resolution mid-video.
  /// Returns the refined (or gated/upsampled) latents; caller decodes.
  private func applyTwoStageRefine(
    _ latents: MLXArray,
    latF: Int, latH: Int, latW: Int,
    textEmbeddings: MLXArray,
    negativeEmbeddings: MLXArray?,
    cfgScale: Float,
    seed: UInt64?,
    refineAnchorImage: MLXArray?,
    progressCallback: ((Int, Int) -> Void)?
  ) -> MLXArray {
    guard let ups = self.upsampler,
          ProcessInfo.processInfo.environment["LTX2_TWO_STAGE"] == "1" else {
      return latents
    }
    logger.info("Two-stage refine: upsampling latents 2x...")
    // The latent upsampler is trained on UN-normalized (VAE-scale) latents.
    let stats = vae.decoder.perChannelStatistics
    let denorm = stats.unNormalize(latents.asType(.float32))
    let upDenorm = ups(denorm)
    let upLatent = stats.normalize(upDenorm).asType(.float32)
    eval(upLatent)
    // --- Refine bisection instrumentation (env-gated, 2026-08-01 band bug) ---
    // LTX2_REFINE_ROWSTATS=1 logs per-latent-row energy at each refine boundary
    // so the vertical-truncation fault can be localized (upsample vs re-noise
    // vs denoise vs decode) from a single short render.
    // LTX2_REFINE_DUMP_DIR=<dir> additionally saves the boundary latents as
    // .npy for offline decode/parity tests.
    let rowStatsOn = ProcessInfo.processInfo.environment["LTX2_REFINE_ROWSTATS"] == "1"
    let dumpDir = ProcessInfo.processInfo.environment["LTX2_REFINE_DUMP_DIR"]
    func rowStats(_ name: String, _ x: MLXArray) {
      guard rowStatsOn else { return }
      let f32 = x.asType(.float32)
      let energy = MLX.mean(f32 * f32, axes: [0, 1, 2, 4])  // (H,)
      eval(energy)
      let rows = energy.asArray(Float.self).map { String(format: "%.4f", $0) }
      logger.info("REFINE ROWSTATS \(name) [\(x.dim(3)) rows]: \(rows.joined(separator: " "))")
      if let dir = dumpDir {
        try? MLX.save(array: f32, url: URL(fileURLWithPath: dir).appendingPathComponent("\(name).npy"))
      }
    }
    rowStats("base", latents)
    rowStats("upsampled", upLatent)
    // Refine-volume OOM gate (2026-07-25 23:24 crash): above it, decode the
    // upsampled latent directly — upscaled-but-unrefined beats a dead server.
    let refineVolume = upLatent.dim(2) * (latH * 2) * (latW * 2)
    let refineMaxVolume = Int(ProcessInfo.processInfo.environment["LTX2_REFINE_MAX_VOL"] ?? "") ?? 12_000
    if ProcessInfo.processInfo.environment["LTX2_REFINE_DECODE_ONLY"] == "1" {
      logger.info("Two-stage refine: DECODE_ONLY (skipped denoise) — decoding upsampled latent directly.")
      return upLatent
    }
    if refineVolume > refineMaxVolume {
      logger.info("Two-stage refine: SKIPPED denoise (refine latent volume \(refineVolume) > \(refineMaxVolume)) — decoding upsampled latent directly (OOM guard).")
      return upLatent
    }
    MLX.GPU.clearCache()
    let rLatH = latH * 2, rLatW = latW * 2
    let refineSigmas: [Float] = (ProcessInfo.processInfo.environment["LTX2_REFINE_SIGMAS"]).flatMap { s -> [Float]? in
      let v = s.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
      return v.count >= 2 ? v : nil
    } ?? [0.85, 0.7250, 0.4219, 0.0]  // PinkCherry v1.5 pass-2 sigmas
    if let seed = seed { MLXRandom.seed(seed &+ 1000) }
    // Flow-matching re-noise: x_σ = (1-σ)·x0 + σ·ε (ComfyUI CONST noise_scaling).
    let refNoise = MLXRandom.normal(upLatent.shape).asType(.float32)
    let sigma0 = refineSigmas[0]
    let mixed = MLXArray(1 - sigma0) * upLatent + refNoise * MLXArray(sigma0)
    // Frame-0 re-anchor: raw source re-encoded at refine res when available
    // (workflow nodes 19/20); else the upsampled latent's own frame 0.
    let rF = upLatent.dim(2)
    var anchorFrame = upLatent[0..., 0..., 0..<1, 0..., 0...]
    if var refImg = refineAnchorImage {
      if refImg.ndim == 4 { refImg = refImg.expandedDimensions(axis: 2) }
      anchorFrame = vae.encode(refImg).asType(.float32)
      eval(anchorFrame)
      logger.info("Two-stage refine: frame 0 re-anchored to source re-encoded at \(rLatW * spatialCompression)x\(rLatH * spatialCompression).")
    }
    rowStats("mixed", mixed)
    let refineInit = MLX.concatenated(
      [anchorFrame, mixed[0..., 0..., 1..<rF, 0..., 0...]], axis: 2)
    let refClean = MLX.concatenated(
      [anchorFrame, upLatent[0..., 0..., 1..<rF, 0..., 0...]], axis: 2)
    let refMask = MLX.concatenated(
      [MLXArray.zeros([1, 1, 1, 1, 1]), MLXArray.ones([1, 1, rF - 1, 1, 1])],
      axis: 2).asType(.float32)
    let refState = LTX2LatentState(
      latent: refineInit, cleanLatent: refClean, denoiseMask: refMask)
    let refinePos = createPositionGrid(batchSize: 1, latF: latF, latH: rLatH, latW: rLatW)
    let refinePE = ltx2PrecomputeFreqsCIS(
      indicesGrid: refinePos, dim: innerDim,
      theta: transformer.positionalEmbeddingTheta,
      maxPos: transformer.positionalEmbeddingMaxPos,
      useMiddleIndicesGrid: transformer.useMiddleIndicesGrid,
      numAttentionHeads: transformer.numHeads,
      ropeMode: transformer.ropeMode,
      doublePrecision: transformer.doublePrecisionRoPE)
    eval(refinePE.cos, refinePE.sin)
    logger.info("Two-stage refine: denoising at \(rLatW * spatialCompression)x\(rLatH * spatialCompression), \(refineSigmas.count - 1) steps...")
    // base=euler_ancestral_cfg_pp -> refine=euler_cfg_pp (the author's pairing).
    var refined = denoisingLoop(
      latents: refineInit, positions: refinePos, precomputedPE: refinePE,
      textEmbeddings: textEmbeddings, negativeEmbeddings: negativeEmbeddings,
      sigmas: refineSigmas, cfgScale: cfgScale, state: refState,
      forceDeterministic: ProcessInfo.processInfo.environment["LTX2_REFINE_DETERMINISTIC"] != "0",
      progressCallback: progressCallback)
    eval(refined)
    rowStats("refined", refined)
    MLX.GPU.clearCache()
    logger.info("Two-stage refine complete.")
    return refined
  }

  private func createPositionGrid(
    batchSize: Int,
    latF: Int,
    latH: Int,
    latW: Int,
    refFrames: Int = 0,   // IC-control: appended reference frames positioned at temporal frame_idx 0
    spatialScaleMul: Float = 1.0   // refine: scale spatial positions to keep RoPE in-distribution at 2x
  ) -> MLXArray {
    let temporalScale = Float(temporalCompression)
    let spatialScale = Float(spatialCompression) * spatialScaleMul
    // Conditioning frame_rate drives the temporal RoPE spacing, which is the
    // motion-vs-coherence dial: LOWER fps → coords spread further apart → the
    // model assumes bigger inter-frame time gaps → MORE motion per frame (until
    // coords exceed the trained temporal max_pos ~20 → shimmer). LTX2_COND_FPS
    // decouples this from the OUTPUT/playback fps (config.fps) so we can drive
    // motion from a sharp seed without making the mp4 choppy. Default: match
    // playback fps (temporally-correct). See QA-CAMPAIGN-2026-07-26 motion sweep.
    let fps = Float(ProcessInfo.processInfo.environment["LTX2_COND_FPS"] ?? "") ?? Float(config.fps)
    let totalF = latF + refFrames
    let numPatches = totalF * latH * latW

    // Build position indices
    var positions = [Float](repeating: 0, count: batchSize * 3 * numPatches * 2)

    for b in 0..<batchSize {
      for f in 0..<totalF {
        // IC-control reference frames (grid index >= latF) take temporal
        // frame_idx 0 (their own 0-based reference index) so RoPE treats them as
        // reference context at the sequence start, not trailing frames. Mirrors
        // ComfyUI add_keyframe_index: pixel_coords[:,0] += frame_idx(=0).
        let tf = f < latF ? f : (f - latF)
        for h in 0..<latH {
          for w in 0..<latW {
            let tokenIdx = f * latH * latW + h * latW + w
            let baseIdx = b * 3 * numPatches * 2

            // Time start/end in pixel space
            var tStart = Float(tf) * temporalScale
            var tEnd = Float(tf + 1) * temporalScale

            // Causal fix: shift temporal coordinates (matches reference
            // latent_to_pixel_coords: pixel_coords[:,0] = (pc + 1 - scale[0]).clamp(0)).
            tStart = max(0, tStart + 1 - temporalScale)
            tEnd = max(0, tEnd + 1 - temporalScale)

            // fps division RESTORED (2026-07-25): ComfyUI's
            // _prepare_positional_embeddings (ldm/lightricks/model.py) does
            // `fractional_coords[:, 0] *= 1/frame_rate` before RoPE — temporal
            // coordinates are in SECONDS. Without it our coords span 0..48
            // against the trained temporal max_pos of 20 (out of range) and
            // adjacent frames sit 24x further apart than trained — the model
            // renders them near-independently: motion survives but fine detail
            // decorrelates frame to frame (the shimmer: flicker 0.45-0.8 vs
            // reference 0.157). The earlier removal ("fixed near-static
            // motion") misattributed the freeze — that was the pristine-still
            // conditioning bug (LTX2_I2V_COMPRESSION=0 freezing i2v), fixed
            // separately.
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
