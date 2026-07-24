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
        sigmas: refineSigmas, cfgScale: cfgScale, state: nil,
        forceDeterministic: ProcessInfo.processInfo.environment["LTX2_REFINE_DETERMINISTIC"] != "0",
        progressCallback: progressCallback)
      eval(latents)
      MLX.GPU.clearCache()
      logger.info("T2V two-stage refine complete.")
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
      progressCallback: progressCallback
    )

    // Drop the appended IC-control reference frames before decode (keep latF).
    var latents = icRefFrames > 0
      ? latentsAll[0..., 0..., 0..<latF, 0..., 0...]
      : latentsAll

    // Two-stage refine (Phase 3): latent upsample x2 -> re-noise -> short refine
    // denoise at high res. Faithful to the proven two-stage i2v workflow (LTX2.3
    // I2V GGUF 12GB): LTXVLatentUpsampler(spatial x2) -> RandomNoise ->
    // SamplerCustomAdvanced with refine sigmas [0.8025,0.6332,0.4525,0.2425,0.0]
    // (starts at 0.80 = partial denoise preserving the upsampled structure).
    // Env-gated LTX2_TWO_STAGE; requires the loaded upsampler.
    if let ups = self.upsampler, ProcessInfo.processInfo.environment["LTX2_TWO_STAGE"] == "1" {
      logger.info("Two-stage refine: upsampling latents 2x...")
      // The latent upsampler is trained on UN-normalized (VAE-scale) latents.
      // The pipeline's working latents are per-channel normalized, so denormalize
      // -> upsample -> renormalize (matches reference ltx2UpsampleLatents /
      // ComfyUI LTXVLatentUpsampler). Feeding normalized latents raw produces
      // structured garbage.
      let stats = vae.decoder.perChannelStatistics
      let denorm = stats.unNormalize(latents.asType(.float32))
      let upDenorm = ups(denorm)
      let upLatent = stats.normalize(upDenorm).asType(.float32)
      eval(upLatent)
      // DIAGNOSTIC: decode the upsampled latent directly (skip refine denoise) to
      // isolate whether artifacts originate in the upsampler or the refine loop.
      if ProcessInfo.processInfo.environment["LTX2_REFINE_DECODE_ONLY"] == "1" {
        latents = upLatent
        logger.info("Two-stage refine: DECODE_ONLY (skipped denoise) — decoding upsampled latent directly.")
      } else {
      // Free base-pass intermediates before the memory-heavy refine (12s/289f).
      MLX.GPU.clearCache()
      let rLatH = latH * 2, rLatW = latW * 2
      let refineSigmas: [Float] = (ProcessInfo.processInfo.environment["LTX2_REFINE_SIGMAS"]).flatMap { s -> [Float]? in
        let v = s.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        return v.count >= 2 ? v : nil
      } ?? [0.85, 0.7250, 0.4219, 0.0]  // PinkCherry v1.5 pass-2 sigmas (validated
      // on ComfyUI 2026-07-22 with euler_cfg_pp; link-traced from the author's
      // shipped workflow, SamplerCustomAdvanced #119).
      if let seed = seed { MLXRandom.seed(seed &+ 1000) }
      // Flow-matching re-noise: x_σ = (1-σ)·x0 + σ·ε (matches ComfyUI CONST
      // noise_scaling). NOT x0 + σ·ε — that unnormalized mix corrupts the refine.
      let refNoise = MLXRandom.normal(upLatent.shape).asType(.float32)
      let sigma0 = refineSigmas[0]
      let mixed = MLXArray(1 - sigma0) * upLatent + refNoise * MLXArray(sigma0)
      // Identity re-anchor (author's pass-2 LTXVImgToVideoInplace @ strength 1):
      // frame 0 of the upsampled latent is the clean upscaled source frame —
      // keep it CLEAN through the refine via a conditioning state (mask 0 at
      // frame 0, per-token timestep 0), generated frames re-noised at σ0.
      let rF = upLatent.dim(2)
      let refineInit = MLX.concatenated(
        [upLatent[0..., 0..., 0..<1, 0..., 0...], mixed[0..., 0..., 1..<rF, 0..., 0...]],
        axis: 2)
      let refMask = MLX.concatenated(
        [MLXArray.zeros([1, 1, 1, 1, 1]), MLXArray.ones([1, 1, rF - 1, 1, 1])],
        axis: 2).asType(.float32)
      let refState = LTX2LatentState(
        latent: refineInit, cleanLatent: upLatent, denoiseMask: refMask)
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
      // forceDeterministic drops the ancestral noise but keeps the CFG++ family:
      // base=euler_ancestral_cfg_pp -> refine=euler_cfg_pp (the author's pairing).
      latents = denoisingLoop(
        latents: refineInit, positions: refinePos, precomputedPE: refinePE,
        textEmbeddings: textOutput.videoEmbeddings, negativeEmbeddings: negativeEmbeddings,
        sigmas: refineSigmas, cfgScale: cfgScale, state: refState,
        forceDeterministic: ProcessInfo.processInfo.environment["LTX2_REFINE_DETERMINISTIC"] != "0",
        progressCallback: progressCallback)
      eval(latents)
      MLX.GPU.clearCache()
      logger.info("Two-stage refine complete.")
      }
    }

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
    let decoded = decodeAdaptive(latents)
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
  /// normal clips, tiled only when long enough to risk OOM. `config.tiledDecode
  /// == false` (LTX2_TILED_DECODE=0) forces plain everywhere. Threshold tunable
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
    let bf = latents.asType(.bfloat16)
    if config.tiledDecode && volume > plainMaxVolume {
      logger.info("VAE decode: tiled (latent volume \(volume) [\(latF)x\(latH)x\(latW)] > \(plainMaxVolume)) — OOM-safe path.")
      return vae.decodeTiled(bf)
    }
    logger.info("VAE decode: plain single-pass (volume \(volume) [\(latF)x\(latH)x\(latW)]) — clean, no tile mosaic.")
    return vae.decode(bf)
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
    let fps = Float(config.fps)
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

            // NOTE: NO fps division. The reference ComfyUI LTX pipeline
            // (symmetric_patchifier.latent_to_pixel_coords + get_fractional_positions)
            // scales temporal coords by the VAE factor and normalizes ONLY by
            // max_pos — never by fps. The previous /= fps compressed the temporal
            // RoPE ~24x, flattening frame-to-frame progression → near-static motion.

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
