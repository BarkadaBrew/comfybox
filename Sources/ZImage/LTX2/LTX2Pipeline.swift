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
/// Positional-embedding bundle for the joint A/V forward.
public typealias LTX2AVPEs = (
  videoPE: (cos: MLXArray, sin: MLXArray),
  audioPE: (cos: MLXArray, sin: MLXArray),
  crossVideoPE: (cos: MLXArray, sin: MLXArray),
  crossAudioPE: (cos: MLXArray, sin: MLXArray))

/// Audio stream state threaded through the joint denoise (task #21 wire 4).
/// v1: audio rides the POSITIVE pass only (plain Euler, no SDE/CFG/STG/NAG
/// on the audio stream); the video stream inside callAV still receives the
/// a2v cross-modal contribution, so video output with audio enabled is a
/// different (joint) model than video-only — by design.
final class LTX2AVDenoiseState {
  var audioLatents: MLXArray  // (B, 8, Ta, 16) float32
  let audioContext: MLXArray  // (B, S, 2048)
  /// Negative audio embeddings (task #26): when present, the CFG/CFG++
  /// negative pass runs dual-stream and the audio step is guided.
  var negativeAudioContext: MLXArray?
  /// Keyed noise chain for audio ancestral steps — keeps the GLOBAL RNG
  /// stream untouched so video renders stay bit-identical with audio on/off.
  var audioNoiseKey: MLXArray?
  let pe: LTX2AVPEs
  init(audioLatents: MLXArray, audioContext: MLXArray, pe: LTX2AVPEs,
       negativeAudioContext: MLXArray? = nil, audioNoiseKey: MLXArray? = nil) {
    self.audioLatents = audioLatents
    self.audioContext = audioContext
    self.negativeAudioContext = negativeAudioContext
    self.audioNoiseKey = audioNoiseKey
    self.pe = pe
  }
}

public struct LTX2PipelineOutput {
  /// Decoded video tensor `(B, 3, F, H, W)` in float32, range [0, 1].
  public let decoded: MLXArray

  /// Final denoised AUDIO latents `(B, 8, Ta, 16)` when audio generation was
  /// requested (task #21); decode via LTX2AudioVAE.decodeToWaveform. Nil for
  /// video-only renders.
  public var audioLatents: MLXArray? = nil

  /// Number of generated frames.
  public let numFrames: Int

  /// Output width in pixels.
  public let width: Int

  /// Output height in pixels.
  public let height: Int

  /// Total generation time in seconds.
  public let elapsedSeconds: Double
}

/// #1479: what one pipeline-level render produced — a finished clip, or a
/// checkpoint because a preemption signal was raised. Only the `*Resumable`
/// entry points can yield; the legacy entry points pass `preemption: nil` and
/// keep returning `LTX2PipelineOutput` so existing call sites are untouched.
public enum LTX2PipelineOutcome {
  case completed(LTX2PipelineOutput)
  case yielded(LTX2ResumeState)
}

/// Outcome of the shared two-stage refine: refined latents, or a checkpoint
/// taken inside the refine loop.
enum LTX2RefineOutcome {
  case completed(MLXArray)
  case yielded(LTX2ResumeState)
}

// MARK: - Pipeline

/// Main LTX-2 video generation pipeline.
///
/// Orchestrates VAE, TextEncoder, and Transformer into a complete
/// generation pipeline for T2V and I2V.
public final class LTX2Pipeline {

  /// The authoritative per-render configuration (task #9 Phase 2, Codex
  /// finding #14). Refreshed by the generator before each render; defaults
  /// to a no-request/no-preset resolution so standalone pipeline use behaves
  /// exactly like the old inline env reads. Single-render server (admission
  /// gate) — no concurrent mutation.
  public var resolvedConfig: LTX2ResolvedVideoConfig =
    LTX2ConfigResolver.resolveTyped(request: nil, preset: nil)

  /// The 3D Video VAE (encode/decode).
  public let vae: LTX2VAE

  /// The text encoder (Gemma 3 + connectors).
  public let textEncoder: LTX2TextEncoder

  /// The denoising transformer.
  public let transformer: LTX2Transformer

  /// Pipeline configuration.
  public let config: LTX2PipelineConfig

  /// Optional latent upsampler for two-stage pipeline.
  public var upsampler: LTX2LatentUpsampler?  // var: lazy-loaded on first two_stage request (finding #18)

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

  // MARK: - #1479 non-preemptible bridge

  /// Run a render that supplied NEITHER a preemption signal NOR a resume
  /// state, and unwrap its outcome.
  ///
  /// Both non-completed results are structurally unreachable here:
  /// `denoisingLoop` yields only when a signal is supplied AND raised (and the
  /// callers below pass no signal at all), and the resume validation that
  /// throws only runs when a checkpoint is supplied. Reaching either arm means
  /// the render genuinely did not finish, so there is no clip to hand back —
  /// fail loudly rather than dress half-denoised latents up as a result
  /// (spec, Error handling: never hide a real bug behind a plausible output).
  private func nonPreemptible(
    _ site: String, _ body: () throws -> LTX2PipelineOutcome
  ) -> LTX2PipelineOutput {
    let outcome: LTX2PipelineOutcome
    do {
      outcome = try body()
    } catch {
      logger.error("\(site): non-preemptible render threw \(error) — impossible without a resume state.")
      preconditionFailure("\(site): non-preemptible render threw \(error)")
    }
    switch outcome {
    case .completed(let output):
      return output
    case .yielded(let s):
      logger.error("\(site): non-preemptible render yielded at step \(s.stepIndex) — no preemption signal was supplied.")
      preconditionFailure("\(site): yielded with no preemption signal supplied")
    }
  }

  /// Checkpoint taken at a FREE unwind point (a phase or chunk boundary), not
  /// inside a loop. `stepIndex: 0` plus `phase` = "this phase has not started";
  /// `videoLatents` are whatever the next phase consumes.
  private func boundaryCheckpoint(
    phase: LTX2Phase, latents: MLXArray, sigmas: [Float], fingerprint: String,
    chunkIndex: Int, seed: UInt64?, avState: LTX2AVDenoiseState?
  ) -> LTX2ResumeState {
    LTX2ResumeState(
      videoLatents: latents, stepIndex: 0, sigmas: sigmas, phase: phase,
      chunkIndex: chunkIndex, seed: seed,
      audioLatents: avState?.audioLatents, audioNoiseKey: avState?.audioNoiseKey,
      configFingerprint: fingerprint)
  }

  /// Re-stamp a checkpoint taken INSIDE a refine loop with the two things the
  /// loop cannot know: the clean (upsampled, pre-re-noise) base latent its
  /// denoise mask re-injects every step, and — when audio bypassed the refine
  /// (the default) so the loop had no AV state — the BASE pass's finished
  /// audio track. Without that fallback a resume would rebuild the audio from
  /// fresh noise and never denoise it: a silent-noise track on every preempted
  /// A/V render.
  private func refineCheckpoint(
    _ s: LTX2ResumeState, cleanBase: MLXArray, baseAV: LTX2AVDenoiseState?
  ) -> LTX2ResumeState {
    LTX2ResumeState(
      videoLatents: s.videoLatents, stepIndex: s.stepIndex, sigmas: s.sigmas,
      phase: s.phase, chunkIndex: s.chunkIndex, seed: s.seed,
      audioLatents: s.audioLatents ?? baseAV?.audioLatents,
      audioNoiseKey: s.audioNoiseKey ?? baseAV?.audioNoiseKey,
      configFingerprint: s.configFingerprint,
      refineCleanLatents: cleanBase,
      context: s.context)
  }

  /// Restore a checkpoint's audio latents + ancestral chain head into the
  /// freshly rebuilt AV state. Reassignment only — never subscript-assignment,
  /// which would mutate the checkpoint's own tensors through the shared handle.
  private func restoreAudio(_ r: LTX2ResumeState, into av: LTX2AVDenoiseState?) {
    guard let av else { return }
    if let al = r.audioLatents { av.audioLatents = al }
    if let key = r.audioNoiseKey { av.audioNoiseKey = key }
  }

  /// Validate a checkpoint against the freshly resolved BASE basis, unless it
  /// was taken inside the refine loop — that loop resolves its own fingerprint
  /// and sigma schedule and validates there.
  private func validateResumeUnlessRefineOwned(
    _ r: LTX2ResumeState, fingerprint: String, sigmas: [Float]
  ) throws {
    if r.phase == .refineDenoise, r.refineCleanLatents != nil { return }
    // A checkpoint taken before any loop started restarts that phase from its
    // deterministic beginning — there is no in-flight trajectory to protect.
    if r.configFingerprint == ltx2NotStartedFingerprint { return }
    try LTX2ResumeValidator.validate(
      checkpointFingerprint: r.configFingerprint, currentFingerprint: fingerprint,
      checkpointSigmas: r.sigmas, currentSigmas: sigmas, stepIndex: r.stepIndex)
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
    audioSeconds: Float? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) -> LTX2PipelineOutput {
    nonPreemptible("generateT2V") {
      try generateT2VResumable(
        inputIds: inputIds, attentionMask: attentionMask,
        width: width, height: height, numFrames: numFrames, steps: steps,
        seed: seed, guidance: guidance,
        negativeInputIds: negativeInputIds, negativeAttentionMask: negativeAttentionMask,
        audioSeconds: audioSeconds, progressCallback: progressCallback)
    }
  }

  /// #1479: `generateT2V` with preemption/resume plumbing. Defaults reproduce
  /// the legacy call exactly — no signal, no checkpoint, no telemetry.
  func generateT2VResumable(
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
    audioSeconds: Float? = nil,
    preemption: PreemptionSignal? = nil,
    telemetry: LTX2PhaseTelemetry? = nil,
    resume: LTX2ResumeState? = nil,
    chunkIndex: Int = 0,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) throws -> LTX2PipelineOutcome {
    let startTime = CFAbsoluteTimeGetCurrent()
    let cfgScale = guidance ?? config.guidance

    // Validate frame count
    precondition((numFrames - 1) % 8 == 0, "numFrames must be 1 + 8k (got \(numFrames))")
    precondition(width % spatialCompression == 0, "width must be divisible by \(spatialCompression)")
    precondition(height % spatialCompression == 0, "height must be divisible by \(spatialCompression)")

    logger.info("T2V generation: \(width)x\(height), \(numFrames) frames, \(steps) steps")

    // Step 1: Encode prompt
    logger.info("Encoding prompt...")
    telemetry?.begin(.textEncode)
    let wantAudio = audioSeconds != nil && transformer.hasAudio
    let textOutput = textEncoder.encode(
      inputIds: inputIds,
      attentionMask: attentionMask,
      returnAudioEmbeddings: wantAudio
    )
    eval(textOutput.videoEmbeddings)

    // Encode negative prompt for CFG, or for CFG++ (needs it every step at cfg=1).
    var negativeEmbeddings: MLXArray? = nil
    var negativeAudioEmbeddings: MLXArray? = nil
    if cfgScale > 1.0 || resolvedConfig.samplerIsCfgPP,
       let negIds = negativeInputIds, let negMask = negativeAttentionMask {
      logger.info("Encoding negative prompt (scale=\(cfgScale), cfgPP=\(resolvedConfig.samplerIsCfgPP))...")
      let negOutput = textEncoder.encode(
        inputIds: negIds,
        attentionMask: negMask,
        returnAudioEmbeddings: wantAudio
      )
      negativeEmbeddings = negOutput.videoEmbeddings
      if wantAudio { negativeAudioEmbeddings = negOutput.audioEmbeddings }
      eval(negativeEmbeddings!)
    }

    // NAG conditioning. The reference wires this as a separate model-patch
    // input (`LTX2_NAG {nag_cond_video}`), NOT as the CFG negative — in that
    // recipe CFG sits at 1.0 so the CFG negative is inert. With no dedicated
    // NAG prompt we reuse the negative embeddings (same text in practice),
    // encoding them here if CFG/CFG++ didn't already.
    let nagConfig = resolvedConfig.nagConfig ?? .disabled
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
    telemetry?.end(.textEncode)

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

    // Step 5b: Audio stream setup (task #21). Isolated RNG key so video
    // renders are bit-identical with audio on/off; latent rate 25/s
    // (sampleRate / (hop * downsample) = 16000/640).
    var avState: LTX2AVDenoiseState? = nil
    if let seconds = audioSeconds, wantAudio {
      let ta = max(1, Int((seconds * 25).rounded(.up)))
      // Seeded: isolated key (video bit-identical with audio on/off).
      // Unseeded: global stream, so audio noise varies like everything else
      // (Codex #9: keying off seed-0 froze unseeded audio noise).
      let audioKey = seed.map { MLXRandom.key($0 &+ 0xA0D10) }
      let audioNoise = MLXRandom.normal([1, 8, ta, 16], key: audioKey).asType(.float32)
      let audioInit = audioNoise * MLXArray(sigmas[0])
      let (_, audioCoords) = LTX2AudioPatchifier.patchify(audioInit)
      let avPE = transformer.precomputeAVPositionalEmbeddings(
        positions: positions, audioCoords: audioCoords)
      avState = LTX2AVDenoiseState(
        audioLatents: audioInit,
        audioContext: textOutput.audioEmbeddings,
        pe: avPE,
        negativeAudioContext: negativeAudioEmbeddings,
        audioNoiseKey: seed.map { MLXRandom.key($0 &+ 0xA0D12) })
      logger.info("Audio stream enabled: \(ta) latent frames (\(seconds)s, negatives \(negativeAudioEmbeddings != nil ? "on" : "off")).")
    }

    // #1479 resume dispatch. Everything above is deterministic given the seed
    // (`MLXRandom.seed(seed)` precedes the initial draw, the text encoder and
    // the VAE carry no RNG), so re-running setup reproduces the pre-loop state
    // bit-for-bit — only the loop position and its latents come from the
    // checkpoint. A checkpoint whose config drifted is REFUSED here, loudly.
    let baseFingerprint = denoiseConfigFingerprint(
      width: latW * spatialCompression, height: latH * spatialCompression,
      frames: latF, steps: sigmas.count - 1, cfg: cfgScale)
    var baseStartStep = 0
    var skipBaseLoop = false
    if let r = resume {
      try validateResumeUnlessRefineOwned(r, fingerprint: baseFingerprint, sigmas: sigmas)
      restoreAudio(r, into: avState)
      if r.phase == .baseDenoise {
        // stepIndex 0 = the phase never started (a free unwind point between
        // chunks/phases stamps a placeholder latent); rebuild from the
        // deterministic initial noise already computed above.
        if r.stepIndex > 0 {
          latents = r.videoLatents
          baseStartStep = r.stepIndex
        }
      } else {
        skipBaseLoop = true
        latents = r.videoLatents
      }
      logger.info("#1479 resume: T2V re-entering \(r.phase.rawValue) at step \(r.stepIndex) (chunk \(r.chunkIndex)).")
    }

    // Step 6: Denoising loop
    if !skipBaseLoop {
      logger.info("Denoising (\(config.sampler) sampler, \(sigmas.count - 1) steps)...")
      switch denoisingLoop(
        latents: latents,
        positions: positions,
        precomputedPE: precomputedPE,
        textEmbeddings: textOutput.videoEmbeddings,
        negativeEmbeddings: negativeEmbeddings,
        sigmas: sigmas,
        cfgScale: cfgScale,
        state: nil,
        nagEmbeddings: nagEmbeddings, nag: nagConfig,
        avState: avState,
        startStep: baseStartStep,
        seed: seed,
        preemption: preemption,
        telemetry: telemetry,
        chunkIndex: chunkIndex,
        progressCallback: progressCallback
      ) {
      case .completed(let l): latents = l
      case .yielded(let s): return .yielded(s)
      }
    }

    // #1479 free unwind point: the base→refine boundary. Yielding here costs
    // nothing and keeps a raised signal from buying a whole refine pass —
    // but ONLY when it moves this render forward (see LTX2UnwindGuard).
    if preemption?.isRaised == true,
       LTX2UnwindGuard.mayCheckpoint(at: .refineDenoise, whileResuming: resume?.phase) {
      return .yielded(boundaryCheckpoint(
        phase: .refineDenoise, latents: latents, sigmas: sigmas,
        fingerprint: baseFingerprint, chunkIndex: chunkIndex, seed: seed, avState: avState))
    }

    // Step 6b: Two-stage refine (env LTX2_TWO_STAGE) — identical machinery to the
    // i2v refine, minus the frame-0 identity re-anchor (t2v has no source frame,
    // so the refine runs free with state:nil). Upsample x2 -> flow re-noise ->
    // short deterministic refine denoise.
    // #1479: `.refineDenoise` WITH a clean base latent = a checkpoint taken
    // inside this loop; the upsample/re-noise already happened, so re-doing it
    // would restart the pass. Without one it is the base→refine boundary and
    // the refine runs normally from the (finished) base latents.
    let resumeRefine: LTX2ResumeState? = {
      guard let r = resume, r.phase == .refineDenoise, r.refineCleanLatents != nil else { return nil }
      return r
    }()
    // #1479: the config is re-resolved every render, so two-stage can be off
    // (or the upsampler absent) by the time a refine checkpoint comes back.
    // Skipping the refine for a MID-REFINE checkpoint would send a partially
    // denoised, REFINE-resolution tensor straight to the decoder as if it were
    // finished. Refuse loudly instead (spec, Error handling). A base→refine
    // BOUNDARY checkpoint is not affected — see LTX2RefineAvailability.
    if LTX2RefineAvailability.mustRefuse(
        resume: resume, twoStage: resolvedConfig.twoStage,
        upsamplerLoaded: self.upsampler != nil) {
      throw LTX2ResumeError.refineUnavailableOnResume(
        twoStage: resolvedConfig.twoStage, upsamplerLoaded: self.upsampler != nil)
    }
    // A `.vaeDecode` checkpoint means base AND refine already finished — the
    // checkpointed latents are final, so the refine must not run again.
    if resume?.phase != .vaeDecode, let ups = self.upsampler, resolvedConfig.twoStage {
      MLX.GPU.clearCache()
      // Gate-skip decode fix (2026-08-05): check BEFORE upsampling — see
      // applyTwoStageRefine. Above the gate, keep base latents (native size).
      // Same 1.5x refine logic as applyTwoStageRefine (Todd 2026-08-07).
      let rScale = max(1.0, min(2.0, resolvedConfig.refineScale))
      let sLatH = max(1, Int((Float(latH) * rScale).rounded()))
      let sLatW = max(1, Int((Float(latW) * rScale).rounded()))
      let t2vPreVolume = latents.dim(2) * sLatH * sLatW
      let t2vRefineMax = resolvedConfig.refineMaxVol
      if resumeRefine == nil, t2vPreVolume > t2vRefineMax,
         ProcessInfo.processInfo.environment["LTX2_REFINE_UPSCALE_ON_SKIP"] != "1" {
        logger.info("T2V two-stage refine: SKIPPED entirely (volume \(t2vPreVolume) > \(t2vRefineMax)) — native-size decode (gate-skip fix 2026-08-05).")
      } else {
      var upLatent: MLXArray
      if let rr = resumeRefine, let clean = rr.refineCleanLatents {
        upLatent = clean
        logger.info("#1479 resume: T2V refine re-entering at step \(rr.stepIndex) from the checkpointed clean base latent.")
      } else {
      logger.info("T2V two-stage refine: upsampling latents 2x...")
      let stats = vae.decoder.perChannelStatistics
      upLatent = stats.normalize(ups(stats.unNormalize(latents.asType(.float32)))).asType(.float32)
      if sLatH < latH * 2 || sLatW < latW * 2 {
        logger.info("T2V two-stage refine: resizing 2x latent to \(rScale)x (\(sLatH)x\(sLatW) latent) before denoise.")
        upLatent = LTX2Conditioning.resizeLatentBilinear(upLatent, height: sLatH, width: sLatW)
      }
      eval(upLatent)
      }
      let t2vRefineVolume = upLatent.dim(2) * sLatH * sLatW
      if resumeRefine == nil, t2vRefineVolume > t2vRefineMax {
        logger.info("T2V two-stage refine: SKIPPED denoise (volume \(t2vRefineVolume) > \(t2vRefineMax)) — decoding upsampled latent directly (LTX2_REFINE_UPSCALE_ON_SKIP path).")
        latents = upLatent
      } else {
      let rLatH = sLatH, rLatW = sLatW
      let refineSigmas: [Float] = resolvedConfig.refineSigmasEffective
      // AUDIO BYPASSES PASS 2 (final listen verdict, Todd 2026-08-05:
      // "refined is not better — it picks up and enhances ambient [noise]",
      // agreeing with PinkCherry HF #8 and the grit metric; matched-seed
      // pairs audio-ab-t2v-* / audio-ab-speech-*). Audio keeps its fully
      // denoised stage-1 track; video still refines. LTX2_AUDIO_REFINE=1
      // re-enables the joint refine for A/B.
      // Now resolved config (env -> preset -> request), not ProcessInfo:
      // the two-pass quality tier needs to request audio refine PER RENDER,
      // and an env read here could not see a per-request value.
      // LTX2_AUDIO_REFINE still works as the global default.
      let refineAudio = resolvedConfig.audioRefine
      // #1479: the refine loop owns its own validation basis — its fingerprint
      // and sigmas are the REFINE pass's, not the base pass's.
      let refineFingerprint = denoiseConfigFingerprint(
        width: rLatW * spatialCompression, height: rLatH * spatialCompression,
        frames: upLatent.dim(2), steps: refineSigmas.count - 1, cfg: cfgScale)
      var refineStartStep = 0
      let refineInit: MLXArray
      if let rr = resumeRefine {
        try LTX2ResumeValidator.validate(
          checkpointFingerprint: rr.configFingerprint, currentFingerprint: refineFingerprint,
          checkpointSigmas: rr.sigmas, currentSigmas: refineSigmas, stepIndex: rr.stepIndex)
        refineInit = rr.videoLatents
        refineStartStep = rr.stepIndex
      } else {
      if let seed = seed { MLXRandom.seed(seed &+ 1000) }
      let refNoise = MLXRandom.normal(upLatent.shape).asType(.float32)
      let sigma0 = refineSigmas[0]
      refineInit = MLXArray(1 - sigma0) * upLatent + refNoise * MLXArray(sigma0)  // flow re-noise
      if refineAudio, let av = avState {
        let aKey = seed.map { MLXRandom.key($0 &+ 0xA0D11) }
        let aNoise = MLXRandom.normal(av.audioLatents.shape, key: aKey).asType(.float32)
        av.audioLatents = MLXArray(1 - sigma0) * av.audioLatents + aNoise * MLXArray(sigma0)
      }
      }
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
      // Refine PEs change with the upsampled grid; rebuild the AV bundle too
      // (audio-refine A/B path only — default keeps stage-1 audio untouched).
      var refineAVState: LTX2AVDenoiseState? = nil
      if refineAudio, let av = avState {
        let (_, aCoords) = LTX2AudioPatchifier.patchify(av.audioLatents)
        let avPE2 = transformer.precomputeAVPositionalEmbeddings(
          positions: refinePos, audioCoords: aCoords)
        refineAVState = LTX2AVDenoiseState(
          audioLatents: av.audioLatents, audioContext: av.audioContext, pe: avPE2)
      }
      switch denoisingLoop(
        latents: refineInit, positions: refinePos, precomputedPE: refinePE,
        textEmbeddings: textOutput.videoEmbeddings, negativeEmbeddings: negativeEmbeddings,
        sigmas: refineSigmas, cfgScale: cfgScale, state: nil, nagEmbeddings: nagEmbeddings, nag: nagConfig,
        forceDeterministic: ProcessInfo.processInfo.environment["LTX2_REFINE_DETERMINISTIC"] != "0",
        avState: refineAVState,
        // #1479 codex review: matches the file's existing convention of
        // decorrelating the refine pass's RNG from the base pass via
        // `seed &+ 1000` (see the `MLXRandom.seed(seed &+ 1000)` reseed just
        // above, for `refNoise`) — keeps refine's per-step video noise on a
        // distinct key sequence from base's, deterministic and resume-safe.
        startStep: refineStartStep,
        seed: seed.map { $0 &+ 1000 },
        preemption: preemption,
        telemetry: telemetry,
        loopPhase: .refineDenoise,
        chunkIndex: chunkIndex,
        progressCallback: progressCallback) {
      case .completed(let l):
        latents = l
        // #1479 codex review: same fix as applyTwoStageRefine — only commit
        // the refined audio writeback on genuine completion, never past a
        // .yielded arm's already-captured snapshot.
        if refineAudio, let av = avState, let rav = refineAVState {
          av.audioLatents = rav.audioLatents
        }
      case .yielded(let s):
        return .yielded(refineCheckpoint(s, cleanBase: upLatent, baseAV: avState))
      }
      eval(latents)
      MLX.GPU.clearCache()
      logger.info("T2V two-stage refine complete.")
      }
      }
    }

    // #1479 free unwind point: the refine→decode boundary. The VAE decode is
    // the longest uninterruptible op in the render, so never start one with a
    // preemption already pending.
    if preemption?.isRaised == true,
       LTX2UnwindGuard.mayCheckpoint(at: .vaeDecode, whileResuming: resume?.phase) {
      return .yielded(boundaryCheckpoint(
        phase: .vaeDecode, latents: latents, sigmas: sigmas,
        fingerprint: baseFingerprint, chunkIndex: chunkIndex, seed: seed, avState: avState))
    }

    // Step 7: Decode latents via VAE
    logger.info("Decoding latents via VAE...")
    telemetry?.begin(.vaeDecode)
    let decoded = decodeAdaptive(latents)
    eval(decoded)

    // Convert from [-1, 1] to [0, 1] range (VAE outputs centered at 0)
    let rescaled = (decoded.asType(.float32) + 1.0) / 2.0
    let clamped = MLX.clip(rescaled, min: 0, max: 1)
    eval(clamped)
    telemetry?.end(.vaeDecode)

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("Generation complete in \(String(format: "%.1f", elapsed))s")

    var output = LTX2PipelineOutput(
      decoded: clamped,
      numFrames: numFrames,
      width: width,
      height: height,
      elapsedSeconds: elapsed
    )
    output.audioLatents = avState?.audioLatents
    return .completed(output)
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
    nonPreemptible("generateT2VWithEmbeddings") {
      try generateT2VWithEmbeddingsResumable(
        videoEmbeddings: videoEmbeddings, negativeEmbeddings: negativeEmbeddings,
        width: width, height: height, numFrames: numFrames, steps: steps,
        seed: seed, guidance: guidance, progressCallback: progressCallback)
    }
  }

  /// #1479: `generateT2VWithEmbeddings` with preemption/resume plumbing.
  func generateT2VWithEmbeddingsResumable(
    videoEmbeddings: MLXArray,
    negativeEmbeddings: MLXArray? = nil,
    width: Int,
    height: Int,
    numFrames: Int,
    steps: Int,
    seed: UInt64? = nil,
    guidance: Float? = nil,
    preemption: PreemptionSignal? = nil,
    telemetry: LTX2PhaseTelemetry? = nil,
    resume: LTX2ResumeState? = nil,
    chunkIndex: Int = 0,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) throws -> LTX2PipelineOutcome {
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

    // #1479 resume dispatch — see generateT2VResumable. This path has no
    // refine and no audio, so the only re-entry is the base loop (or, at the
    // free unwind point below, the decode).
    let baseFingerprint = denoiseConfigFingerprint(
      width: latW * spatialCompression, height: latH * spatialCompression,
      frames: latF, steps: sigmas.count - 1, cfg: cfgScale)
    var baseStartStep = 0
    var skipBaseLoop = false
    if let r = resume {
      try validateResumeUnlessRefineOwned(r, fingerprint: baseFingerprint, sigmas: sigmas)
      if r.phase == .baseDenoise {
        if r.stepIndex > 0 {
          latents = r.videoLatents
          baseStartStep = r.stepIndex
        }
      } else {
        skipBaseLoop = true
        latents = r.videoLatents
      }
      logger.info("#1479 resume: T2V(embeddings) re-entering \(r.phase.rawValue) at step \(r.stepIndex).")
    }

    // Denoising loop
    if !skipBaseLoop {
      logger.info("Denoising (\(config.sampler) sampler, \(sigmas.count - 1) steps)...")
      switch denoisingLoop(
        latents: latents,
        positions: positions,
        precomputedPE: precomputedPE,
        textEmbeddings: videoEmbeddings,
        negativeEmbeddings: negEmb,
        sigmas: sigmas,
        cfgScale: cfgScale,
        state: nil,
        startStep: baseStartStep,
        seed: seed,
        preemption: preemption,
        telemetry: telemetry,
        chunkIndex: chunkIndex,
        progressCallback: progressCallback
      ) {
      case .completed(let l): latents = l
      case .yielded(let s): return .yielded(s)
      }
    }

    // #1479 free unwind point: never start the decode with a pending preemption.
    if preemption?.isRaised == true,
       LTX2UnwindGuard.mayCheckpoint(at: .vaeDecode, whileResuming: resume?.phase) {
      return .yielded(boundaryCheckpoint(
        phase: .vaeDecode, latents: latents, sigmas: sigmas,
        fingerprint: baseFingerprint, chunkIndex: chunkIndex, seed: seed, avState: nil))
    }

    // Decode latents via VAE
    logger.info("Decoding latents via VAE...")
    telemetry?.begin(.vaeDecode)
    let decoded = decodeAdaptive(latents)
    eval(decoded)

    // Convert from [-1, 1] to [0, 1] range (VAE outputs centered at 0)
    let rescaled = (decoded.asType(.float32) + 1.0) / 2.0
    let clamped = MLX.clip(rescaled, min: 0, max: 1)
    eval(clamped)
    telemetry?.end(.vaeDecode)

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("Generation complete in \(String(format: "%.1f", elapsed))s")

    return .completed(LTX2PipelineOutput(
      decoded: clamped,
      numFrames: numFrames,
      width: width,
      height: height,
      elapsedSeconds: elapsed
    ))
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
    audioSeconds: Float? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) -> LTX2PipelineOutput {
    nonPreemptible("generateI2V") {
      try generateI2VResumable(
        inputIds: inputIds, attentionMask: attentionMask, image: image, strength: strength,
        width: width, height: height, numFrames: numFrames, steps: steps,
        seed: seed, guidance: guidance,
        negativeInputIds: negativeInputIds, negativeAttentionMask: negativeAttentionMask,
        faceAnchorMask: faceAnchorMask, faceAnchorStrength: faceAnchorStrength,
        refineAnchorImage: refineAnchorImage, audioSeconds: audioSeconds,
        progressCallback: progressCallback)
    }
  }

  /// #1479: `generateI2V` with preemption/resume plumbing.
  func generateI2VResumable(
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
    audioSeconds: Float? = nil,
    preemption: PreemptionSignal? = nil,
    telemetry: LTX2PhaseTelemetry? = nil,
    resume: LTX2ResumeState? = nil,
    chunkIndex: Int = 0,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) throws -> LTX2PipelineOutcome {
    let startTime = CFAbsoluteTimeGetCurrent()
    let cfgScale = guidance ?? config.guidance

    precondition((numFrames - 1) % 8 == 0, "numFrames must be 1 + 8k (got \(numFrames))")

    logger.info("I2V generation: \(width)x\(height), \(numFrames) frames, strength=\(strength)")

    // Step 1: Encode prompt
    logger.info("Encoding prompt...")
    telemetry?.begin(.textEncode)
    let wantAudio = audioSeconds != nil && transformer.hasAudio
    let textOutput = textEncoder.encode(
      inputIds: inputIds,
      attentionMask: attentionMask,
      returnAudioEmbeddings: wantAudio
    )
    eval(textOutput.videoEmbeddings)

    var negativeEmbeddings: MLXArray? = nil
    var negativeAudioEmbeddings: MLXArray? = nil
    // CFG++ samplers need the negative embeddings every step even at cfg=1.
    if cfgScale > 1.0 || resolvedConfig.samplerIsCfgPP,
       let negIds = negativeInputIds, let negMask = negativeAttentionMask {
      let negOutput = textEncoder.encode(
        inputIds: negIds, attentionMask: negMask, returnAudioEmbeddings: wantAudio
      )
      negativeEmbeddings = negOutput.videoEmbeddings
      if wantAudio { negativeAudioEmbeddings = negOutput.audioEmbeddings }
      eval(negativeEmbeddings!)
    }

    // NAG conditioning — see generateT2V for the rationale. The reference
    // recipe is i2v-centric, so this path matters at least as much.
    let nagConfig = resolvedConfig.nagConfig ?? .disabled
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

    telemetry?.end(.textEncode)

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
      // Temporal decay (2026-08-15, "stuck/floating face" reports on partnered
      // action clips): fm was a single spatial mask (frame-dim 1), broadcast
      // identically across all latF frames — every frame got pulled toward the
      // EXACT SAME static source-frame latent, at every step, for the whole
      // clip. Frame 0 matching the source is correct (that's the actual
      // condition frame); holding every LATER frame to that same static
      // appearance fights any real motion under the mask once the anchored
      // subject moves even slightly — reads as the face freezing (no temporal
      // variation at all inside the mask) and detaching from a body that keeps
      // moving under it ("floating"). Ramp the effective strength down across
      // the frame axis instead of applying it flat; a floor keeps SOME identity
      // hold late in the clip (the reason this exists — an unanchored partner
      // drifts into a different person) without pinning motion the whole way.
      let decayFloor = Float(ProcessInfo.processInfo.environment["LTX2_FACE_ANCHOR_DECAY_FLOOR"] ?? "") ?? 0.3
      var temporalFm = fm
      if latF > 1 {
        let decay = (0..<latF).map { t -> Float in
          let frac = Float(t) / Float(latF - 1)
          return 1.0 - (1.0 - decayFloor) * frac
        }
        let decayArray = MLXArray(decay, [1, 1, latF, 1, 1])
        temporalFm = MLX.broadcast(fm, to: [1, 1, latF, latH, latW]) * decayArray
      }
      state.faceMask = temporalFm
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
    if resolvedConfig.icControl {
      let refStrength = resolvedConfig.icRefStrength
      icRefFrames = imageLatent.dim(2)
      state.latent = MLX.concatenated([state.latent, imageLatent.asType(state.latent.dtype)], axis: 2)
      state.cleanLatent = MLX.concatenated([state.cleanLatent, imageLatent.asType(state.cleanLatent.dtype)], axis: 2)
      let refMask = MLX.broadcast(
        MLXArray(Float(1.0 - refStrength)).reshaped(1, 1, 1, 1, 1),
        to: [1, 1, icRefFrames, 1, 1])
      state.denoiseMask = MLX.concatenated([state.denoiseMask, refMask], axis: 2)
      logger.info("IC-control: appended \(icRefFrames) reference frame(s) @ strength \(refStrength)")
      // Face-anchor mask covers only the latF video frames, but the ref-frame
      // append above extends the latent frame axis to latF+icRefFrames. Pad the
      // mask with zeros over the ref frames (frozen references — nothing to
      // anchor) or the identity-hold multiply in the denoising loop hits an MLX
      // broadcast fatal: (1,1,latF,H,W) vs (1,128,latF+ref,H,W). Regression
      // from the 2026-08-15 temporal-decay expansion (mask was frame-dim 1 and
      // broadcast fine before); crash-looped every face-anchored i2v 2026-08-18.
      if let fmT = state.faceMask {
        let pad = MLX.zeros([1, 1, icRefFrames, fmT.dim(3), fmT.dim(4)]).asType(fmT.dtype)
        state.faceMask = MLX.concatenated([fmT, pad], axis: 2)
      }
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

    // Step 5b: Audio stream (task #21 i2v). Same setup as t2v; the per-token
    // av_ca video conditioning inside callAV handles the i2v denoise mask
    // (conditioned tokens at ~0). Cross PEs use the FULL positions grid incl.
    // IC-control ref frames (they sit at t=0s, matching their video PE).
    var avState: LTX2AVDenoiseState? = nil
    if let seconds = audioSeconds, wantAudio {
      let ta = max(1, Int((seconds * 25).rounded(.up)))
      let audioKey = seed.map { MLXRandom.key($0 &+ 0xA0D10) }
      let audioNoise = MLXRandom.normal([1, 8, ta, 16], key: audioKey).asType(.float32)
      let audioInit = audioNoise * MLXArray(sigmas[0])
      let (_, audioCoords) = LTX2AudioPatchifier.patchify(audioInit)
      let avPE = transformer.precomputeAVPositionalEmbeddings(
        positions: positions, audioCoords: audioCoords)
      avState = LTX2AVDenoiseState(
        audioLatents: audioInit,
        audioContext: textOutput.audioEmbeddings,
        pe: avPE,
        negativeAudioContext: negativeAudioEmbeddings,
        audioNoiseKey: seed.map { MLXRandom.key($0 &+ 0xA0D12) })
      logger.info("Audio stream enabled (i2v): \(ta) latent frames (\(seconds)s, negatives \(negativeAudioEmbeddings != nil ? "on" : "off")).")
    }

    // #1479 resume dispatch. The whole setup above is deterministic given the
    // seed (`MLXRandom.seed` precedes `createInitialState`'s draw; the text
    // encoder and the VAE encode carry no RNG), so a resume reproduces the
    // conditioning state bit-for-bit and only the loop position is restored.
    // NOTE the fingerprint's frame count is the FULL grid — IC-control's
    // appended reference frames included — matching what `denoisingLoop`
    // stamps from `latents.dim(2)`.
    let baseFingerprint = denoiseConfigFingerprint(
      width: latW * spatialCompression, height: latH * spatialCompression,
      frames: latF + icRefFrames, steps: sigmas.count - 1, cfg: cfgScale)
    var baseStartStep = 0
    var skipBaseLoop = false
    var resumedLatents: MLXArray? = nil
    if let r = resume {
      try validateResumeUnlessRefineOwned(r, fingerprint: baseFingerprint, sigmas: sigmas)
      restoreAudio(r, into: avState)
      if r.phase == .baseDenoise {
        if r.stepIndex > 0 {
          resumedLatents = r.videoLatents
          baseStartStep = r.stepIndex
        }
      } else {
        skipBaseLoop = true
        resumedLatents = r.videoLatents
      }
      logger.info("#1479 resume: I2V re-entering \(r.phase.rawValue) at step \(r.stepIndex) (chunk \(r.chunkIndex)).")
    }

    // Step 6: Denoising loop with I2V state
    let latentsAll: MLXArray
    if skipBaseLoop, let resumed = resumedLatents {
      latentsAll = resumed
    } else {
      logger.info("Denoising with I2V conditioning...")
      switch denoisingLoop(
        latents: resumedLatents ?? state.latent,
        positions: positions,
        precomputedPE: precomputedPE,
        textEmbeddings: textOutput.videoEmbeddings,
        negativeEmbeddings: negativeEmbeddings,
        sigmas: sigmas,
        cfgScale: cfgScale,
        state: state,
        nagEmbeddings: nagEmbeddings, nag: nagConfig,
        avState: avState,
        startStep: baseStartStep,
        seed: seed,
        preemption: preemption,
        telemetry: telemetry,
        chunkIndex: chunkIndex,
        progressCallback: progressCallback
      ) {
      case .completed(let l): latentsAll = l
      case .yielded(let s): return .yielded(s)
      }
    }

    // Drop the appended IC-control reference frames before decode (keep latF).
    // A post-base checkpoint (.refineDenoise / .vaeDecode) was taken AFTER this
    // slice, so it is already at latF frames — slicing again would over-trim.
    var latents = (icRefFrames > 0 && !skipBaseLoop)
      ? latentsAll[0..., 0..., 0..<latF, 0..., 0...]
      : latentsAll

    // #1479 free unwind point: the base→refine boundary.
    if preemption?.isRaised == true,
       LTX2UnwindGuard.mayCheckpoint(at: .refineDenoise, whileResuming: resume?.phase) {
      return .yielded(boundaryCheckpoint(
        phase: .refineDenoise, latents: latents, sigmas: sigmas,
        fingerprint: baseFingerprint, chunkIndex: chunkIndex, seed: seed, avState: avState))
    }

    // Two-stage refine — shared with continuation chunks (applyTwoStageRefine).
    if resume?.phase != .vaeDecode {
      switch try applyTwoStageRefine(
        latents, latF: latF, latH: latH, latW: latW,
        textEmbeddings: textOutput.videoEmbeddings,
        negativeEmbeddings: negativeEmbeddings,
        cfgScale: cfgScale, seed: seed,
        refineAnchorImage: refineAnchorImage,
        avState: avState,
        preemption: preemption, telemetry: telemetry,
        resume: resume, chunkIndex: chunkIndex,
        progressCallback: progressCallback) {
      case .completed(let l): latents = l
      case .yielded(let s): return .yielded(s)
      }
    }

    // #1479 free unwind point: never start the decode with a pending preemption.
    if preemption?.isRaised == true,
       LTX2UnwindGuard.mayCheckpoint(at: .vaeDecode, whileResuming: resume?.phase) {
      return .yielded(boundaryCheckpoint(
        phase: .vaeDecode, latents: latents, sigmas: sigmas,
        fingerprint: baseFingerprint, chunkIndex: chunkIndex, seed: seed, avState: avState))
    }

    // Step 7: Decode
    logger.info("Decoding latents via VAE...")
    telemetry?.begin(.vaeDecode)
    let decoded = decodeAdaptive(latents)
    eval(decoded)

    let rescaled = (decoded.asType(.float32) + 1.0) / 2.0
    let clamped = MLX.clip(rescaled, min: 0, max: 1)
    eval(clamped)
    telemetry?.end(.vaeDecode)

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("I2V generation complete in \(String(format: "%.1f", elapsed))s")

    var output = LTX2PipelineOutput(
      decoded: clamped,
      numFrames: numFrames,
      width: width,
      height: height,
      elapsedSeconds: elapsed
    )
    output.audioLatents = avState?.audioLatents
    return .completed(output)
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
    nonPreemptible("generateMultiKeyframe") {
      try generateMultiKeyframeResumable(
        inputIds: inputIds, attentionMask: attentionMask, keyframes: keyframes,
        width: width, height: height, numFrames: numFrames, steps: steps,
        seed: seed, guidance: guidance,
        negativeInputIds: negativeInputIds, negativeAttentionMask: negativeAttentionMask,
        progressCallback: progressCallback)
    }
  }

  /// #1479: `generateMultiKeyframe` with preemption/resume plumbing.
  func generateMultiKeyframeResumable(
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
    preemption: PreemptionSignal? = nil,
    telemetry: LTX2PhaseTelemetry? = nil,
    resume: LTX2ResumeState? = nil,
    chunkIndex: Int = 0,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) throws -> LTX2PipelineOutcome {
    precondition(!keyframes.isEmpty, "generateMultiKeyframe requires at least one keyframe")
    precondition((numFrames - 1) % 8 == 0, "numFrames must be 1 + 8k (got \(numFrames))")

    let startTime = CFAbsoluteTimeGetCurrent()
    let cfgScale = guidance ?? config.guidance

    logger.info("Multi-keyframe generation: \(width)x\(height), \(numFrames) frames, \(keyframes.count) keyframe(s)")

    // Step 1: Encode prompt
    logger.info("Encoding prompt...")
    telemetry?.begin(.textEncode)
    let textOutput = textEncoder.encode(
      inputIds: inputIds, attentionMask: attentionMask, returnAudioEmbeddings: false
    )
    eval(textOutput.videoEmbeddings)

    var negativeEmbeddings: MLXArray? = nil
    if cfgScale > 1.0 || resolvedConfig.samplerIsCfgPP, let negIds = negativeInputIds, let negMask = negativeAttentionMask {
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
    let nagConfig = resolvedConfig.nagConfig ?? .disabled
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

    telemetry?.end(.textEncode)

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

    // #1479 resume dispatch — see generateI2VResumable.
    let baseFingerprint = denoiseConfigFingerprint(
      width: latW * spatialCompression, height: latH * spatialCompression,
      frames: latF, steps: sigmas.count - 1, cfg: cfgScale)
    var baseStartStep = 0
    var skipBaseLoop = false
    var resumedLatents: MLXArray? = nil
    if let r = resume {
      try validateResumeUnlessRefineOwned(r, fingerprint: baseFingerprint, sigmas: sigmas)
      if r.phase == .baseDenoise {
        if r.stepIndex > 0 {
          resumedLatents = r.videoLatents
          baseStartStep = r.stepIndex
        }
      } else {
        skipBaseLoop = true
        resumedLatents = r.videoLatents
      }
      logger.info("#1479 resume: multi-keyframe re-entering \(r.phase.rawValue) at step \(r.stepIndex) (chunk \(r.chunkIndex)).")
    }

    // Step 6: Denoise
    let latents: MLXArray
    if skipBaseLoop, let resumed = resumedLatents {
      latents = resumed
    } else {
      logger.info("Denoising with \(keyframes.count)-keyframe conditioning...")
      switch denoisingLoop(
        latents: resumedLatents ?? state.latent,
        positions: positions,
        precomputedPE: precomputedPE,
        textEmbeddings: textOutput.videoEmbeddings,
        negativeEmbeddings: negativeEmbeddings,
        sigmas: sigmas,
        cfgScale: cfgScale,
        state: state,
        startStep: baseStartStep,
        seed: seed,
        preemption: preemption,
        telemetry: telemetry,
        chunkIndex: chunkIndex,
        progressCallback: progressCallback
      ) {
      case .completed(let l): latents = l
      case .yielded(let s): return .yielded(s)
      }
    }

    // #1479 free unwind point: the base→refine boundary.
    if preemption?.isRaised == true,
       LTX2UnwindGuard.mayCheckpoint(at: .refineDenoise, whileResuming: resume?.phase) {
      return .yielded(boundaryCheckpoint(
        phase: .refineDenoise, latents: latents, sigmas: sigmas,
        fingerprint: baseFingerprint, chunkIndex: chunkIndex, seed: seed, avState: nil))
    }

    // Step 6b: Two-stage refine — SAME treatment as generateI2V. Continuation
    // chunks previously skipped refine entirely, so chunks 2+ of every
    // multi-chunk render were base-resolution decodes upscaled by the muxer
    // (visible quality cliff at each chunk boundary; blockiness rose 1.14 ->
    // 1.43 across the 2026-07-26 03:59 filed video).
    var refinedLatents = latents
    if resume?.phase != .vaeDecode {
      switch try applyTwoStageRefine(
        latents, latF: latF, latH: latH, latW: latW,
        textEmbeddings: textOutput.videoEmbeddings,
        negativeEmbeddings: negativeEmbeddings,
        cfgScale: cfgScale, seed: seed,
        refineAnchorImage: nil,
        preemption: preemption, telemetry: telemetry,
        resume: resume, chunkIndex: chunkIndex,
        progressCallback: progressCallback) {
      case .completed(let l): refinedLatents = l
      case .yielded(let s): return .yielded(s)
      }
    }

    // #1479 free unwind point: never start the decode with a pending preemption.
    if preemption?.isRaised == true,
       LTX2UnwindGuard.mayCheckpoint(at: .vaeDecode, whileResuming: resume?.phase) {
      return .yielded(boundaryCheckpoint(
        phase: .vaeDecode, latents: refinedLatents, sigmas: sigmas,
        fingerprint: baseFingerprint, chunkIndex: chunkIndex, seed: seed, avState: nil))
    }

    // Step 7: Decode
    logger.info("Decoding latents via VAE...")
    telemetry?.begin(.vaeDecode)
    let decoded = decodeAdaptive(refinedLatents)
    eval(decoded)

    let rescaled = (decoded.asType(.float32) + 1.0) / 2.0
    let clamped = MLX.clip(rescaled, min: 0, max: 1)
    eval(clamped)
    telemetry?.end(.vaeDecode)

    let elapsed = CFAbsoluteTimeGetCurrent() - startTime
    logger.info("Multi-keyframe generation complete in \(String(format: "%.1f", elapsed))s")

    return .completed(LTX2PipelineOutput(
      decoded: clamped,
      numFrames: numFrames,
      width: width,
      height: height,
      elapsedSeconds: elapsed
    ))
  }

  // MARK: - Internal: Denoising Loop

  /// #1479: mismatch-detection fingerprint for resume. `LTX2ResolvedVideoConfig`
  /// (the type of `resolvedConfig`) carries sampler/cfg-schedule provenance but
  /// not the per-render width/height/frames/steps — those are runtime locals
  /// derived from the caller's `latents` shape and sigma schedule, so this is a
  /// method taking them as parameters rather than a zero-arg stored property.
  /// Its only job is making a config-changed-between-pause-and-resume resumption
  /// detectable (compared by Task 4); the exact string format is not a contract.
  func denoiseConfigFingerprint(
    width: Int, height: Int, frames: Int, steps: Int, cfg: Float
  ) -> String {
    "\(width)x\(height)f\(frames)s\(steps)-\(resolvedConfig.sampler)-cfg\(cfg)"
  }

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
    avState: LTX2AVDenoiseState? = nil,
    // #1479: preemption/resume plumbing. All defaulted so pre-existing
    // callers compile unchanged; loopPhase's default (.baseDenoise) is
    // overridden explicitly at refine call sites.
    startStep: Int = 0,
    seed: UInt64? = nil,
    preemption: PreemptionSignal? = nil,
    telemetry: LTX2PhaseTelemetry? = nil,
    loopPhase: LTX2Phase = .baseDenoise,
    chunkIndex: Int = 0,
    progressCallback: ((Int, Int) -> Void)?
  ) -> LTX2DenoiseResult {
    // #1479 phase timing. `defer` so a yield closes the phase too: a preempted
    // pass then contributes two partial samples that sum to the true duration,
    // which is why the refusal guard reads `meanStepSec` (exact per step) for
    // the denoise phases and only uses phase means for the uninterruptible ones.
    telemetry?.begin(loopPhase)
    defer { telemetry?.end(loopPhase) }
    let dtype: DType = .bfloat16
    let numSteps = sigmas.count - 1
    // Per-step CFG schedule (community "CFG ramp": e.g. LTX2_CFG_SCHEDULE=3,2,1 —
    // CFG only on the first steps, where broad motion forms, then back to 1;
    // 10Eros per-step guider pattern). Missing entries extend the last value.
    let cfgSchedule: [Float]? = resolvedConfig.cfgSchedule.isEmpty ? nil : resolvedConfig.cfgSchedule
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
    let useSDE = resolvedConfig.samplerIsAncestral && !forceDeterministic
    // CFG++ (euler[_ancestral]_cfg_pp): needs an unconditional (negative) pass
    // every step regardless of cfgScale.
    let useCfgPP = resolvedConfig.samplerIsCfgPP && negativeEmbeddings != nil

    // #1479: fingerprint the render's shape/config once, up front — cheap,
    // and every yield point below needs the same value. `latents` is
    // (B, C, F, H, W); dims are stable across the whole loop.
    let renderFingerprint = denoiseConfigFingerprint(
      width: latents.dim(4) * spatialCompression,
      height: latents.dim(3) * spatialCompression,
      frames: latents.dim(2),
      steps: numSteps,
      cfg: cfgScale)

    for i in startStep..<numSteps {
      // #1479: yield at the step boundary. Checked first so a raised signal
      // costs zero model passes. currentLatents at this point is the
      // end-of-step-(i-1) state — exactly what resume needs to re-enter at i.
      if let p = preemption, p.isRaised {
        return .yielded(LTX2ResumeState(
          videoLatents: currentLatents,
          stepIndex: i,
          sigmas: sigmas,
          phase: loopPhase,
          chunkIndex: chunkIndex,
          seed: seed,
          audioLatents: avState?.audioLatents,
          audioNoiseKey: avState?.audioNoiseKey,
          configFingerprint: renderFingerprint))
      }
      let stepStart = Date().timeIntervalSince1970
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
      let velocityPos: MLXArray
      var avVelocityPos: MLXArray? = nil
      var avVelocityNeg: MLXArray? = nil
      if let av = avState {
        // Joint A/V pass (task #21): both velocities in one forward. NAG is
        // not available on this path yet (spec open item) — the joint model's
        // a2v cross-modal coupling replaces it as the video conditioning
        // difference. Audio updates HERE (positive pass only, plain Euler,
        // deterministic): CFG/STG passes below remain video-only and steer
        // only the video x0.
        let (vv, va) = transformer.callAV(
          latent: latentsFlat,
          audioLatents: av.audioLatents.asType(dtype),
          timestep: timesteps,
          videoSigmaMax: sigma,
          audioSigma: sigma,
          context: textEmbeddings.asType(dtype),
          audioContext: av.audioContext.asType(dtype),
          sigma: sigmaArray,
          pe: av.pe)
        velocityPos = vv
        avVelocityPos = va
      } else {
        velocityPos = transformer(
          latent: latentsFlat,
          timestep: timesteps,
          context: textEmbeddings.asType(dtype),
          positions: positions,
          sigma: sigmaArray,
          precomputedPE: precomputedPE,
          nagContext: nagEmbeddings?.asType(dtype),
          nag: nag
        )
      }

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
        let velocityNeg: MLXArray
        if let av = avState, let negACtx = av.negativeAudioContext {
          // Audio joins the negative pass (task #26): the reference recipe's
          // negative ("...distorted sound, saturated sound, loud") flows
          // through BOTH streams; ours never steered audio until now. The
          // dual negative also makes the video-side negative prediction
          // reference-faithful (it carries a2v cross-modal contributions).
          let (vv, va) = transformer.callAV(
            latent: latentsFlat,
            audioLatents: av.audioLatents.asType(dtype),
            timestep: timesteps,
            videoSigmaMax: sigma,
            audioSigma: sigma,
            context: negEmb.asType(dtype),
            audioContext: negACtx.asType(dtype),
            sigma: sigmaArray,
            pe: av.pe)
          velocityNeg = vv
          avVelocityNeg = va
        } else {
          velocityNeg = transformer(
            latent: latentsFlat,
            timestep: timesteps,
            context: negEmb.asType(dtype),
            positions: positions,
            sigma: sigmaArray,
            precomputedPE: precomputedPE
          )
        }
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
          // resolvedConfig.guidanceRescale (0 = off) — request/preset capable.
          let phi = resolvedConfig.guidanceRescale
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
      let stgBase = resolvedConfig.stgScale
      if stgBase > 0 {
        let stgScale = LTX2PipelineConfig.stgScaleForStep(i, base: stgBase)
        let velocitySTG = transformer(
          latent: latentsFlat,
          timestep: timesteps,
          context: textEmbeddings.asType(dtype),
          positions: positions,
          sigma: sigmaArray,
          precomputedPE: precomputedPE,
          stgBlocks: resolvedConfig.stgBlockSet
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
            // #1479: per-step keyed when seeded (bit-identical resume); falls
            // back to the plain global stream when unseeded (byte-identical
            // to pre-#1479 behaviour).
            let noise = ancestralVideoNoise(shape: currentLatents.shape, seed: seed, step: i)
            currentLatents = currentLatents + MLXArray(alphaT) * noise * MLXArray(sigmaUp)
          }
        } else if useSDE {
          let sigmaF32 = MLXArray(sigma)
          let eps = (currentLatents - denoised) / sigmaF32
          let (alphaRatio, sigmaDown, sigmaUp) = getSdeCoeff(sigmaNext: sigmaNext)
          // #1479: see ancestral-branch comment above.
          let noise = ancestralVideoNoise(shape: currentLatents.shape, seed: seed, step: i)
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
      // Audio step (task #26): guided when the negative pass ran dual-stream,
      // plain Euler otherwise. Ancestral noise comes from the state's keyed
      // chain so the global RNG (video noise sequence) is untouched.
      if let av = avState, let velA = avVelocityPos {
        var aNoise: MLXArray? = nil
        if useSDE, avVelocityNeg != nil, let key = av.audioNoiseKey {
          let keys = MLXRandom.split(key: key)
          av.audioNoiseKey = keys.0
          aNoise = MLXRandom.normal(av.audioLatents.shape, key: keys.1).asType(.float32)
        }
        // Audio guidance is DECOUPLED from video cfg (2026-08-05): Kira's
        // video cfg 3.5 over-guided the audio stream — dragged, slow-motion
        // sound on otherwise-correct 5s clips; the author's recipes never
        // run audio above cfg 1. Negatives still steer via CFG++ stepping.
        // LTX2_AUDIO_CFG overrides (default 1.0).
        let audioCfg = Float(ProcessInfo.processInfo.environment["LTX2_AUDIO_CFG"] ?? "") ?? 1.0
        av.audioLatents = ltx2AudioStep(
          audio: av.audioLatents, velocityPos: velA, velocityNeg: avVelocityNeg,
          sigma: sigma, sigmaNext: sigmaNext, cfgScale: min(cfgAt(i), audioCfg),
          useCfgPP: useCfgPP, useSDE: useSDE && !forceDeterministic,
          ancestralNoise: aNoise)
        eval(av.audioLatents)
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
      telemetry?.recordStep(seconds: Date().timeIntervalSince1970 - stepStart)
    }

    if let handle = trajHandle {
      try? handle.close()
    }

    return .completed(currentLatents)
  }

  // MARK: - Internal: Sigma Schedule

  /// Get sigma schedule based on pipeline type.
  private func getSigmaSchedule(
    steps: Int, latF: Int, latH: Int, latW: Int
  ) -> [Float] {
    switch config.pipelineType {
    case .distilled:
      // NOTE (2026-08-05): the pinned schedule IGNORES `steps` by design (the
      // author's recipe pins sigmas). Log loudly so a steps override never
      // silently no-ops again (the vocal-16 mechanism was a placebo for a day).
      let pinned = resolvedConfig.stage1SigmasOrNil ?? LTX2PipelineConfig.stage1Sigmas
      if steps != pinned.count - 1 {
        logger.warning("Distilled pipeline: requested steps=\(steps) IGNORED — pinned sigma schedule (\(pinned.count - 1) steps) governs. Use tuning.stage1_sigmas to change step density.")
      }
      return pinned
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
    let plainMaxVolume = resolvedConfig.plainDecodeMaxVol
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
    avState: LTX2AVDenoiseState? = nil,
    preemption: PreemptionSignal? = nil,
    telemetry: LTX2PhaseTelemetry? = nil,
    resume: LTX2ResumeState? = nil,
    chunkIndex: Int = 0,
    progressCallback: ((Int, Int) -> Void)?
  ) throws -> LTX2RefineOutcome {
    // #1479: a checkpoint taken INSIDE this loop carries the clean (upsampled,
    // pre-re-noise) base latent; the upsample and flow re-noise already ran, so
    // redoing them would restart the pass rather than continue it.
    let resumeRefine: LTX2ResumeState? = {
      guard let r = resume, r.phase == .refineDenoise, r.refineCleanLatents != nil else { return nil }
      return r
    }()
    // #1479: this availability guard sits ABOVE the resume validation, so a
    // two-stage/upsampler config drift between checkpoint and resume would
    // otherwise return a MID-REFINE checkpoint's partially denoised,
    // REFINE-resolution latents to the caller as finished work — a silently
    // degraded decode. Refuse loudly instead (spec, Error handling). A
    // base→refine BOUNDARY checkpoint carries clean base-resolution latents
    // and resumes fine through the `guard` below — see LTX2RefineAvailability.
    if LTX2RefineAvailability.mustRefuse(
        resume: resume, twoStage: resolvedConfig.twoStage,
        upsamplerLoaded: self.upsampler != nil) {
      throw LTX2ResumeError.refineUnavailableOnResume(
        twoStage: resolvedConfig.twoStage, upsamplerLoaded: self.upsampler != nil)
    }
    guard let ups = self.upsampler,
          resolvedConfig.twoStage else {
      return .completed(latents)
    }
    // Gate-skip decode fix (Todd 2026-08-05): the volume is knowable from
    // dims BEFORE upsampling. Above the gate the refine denoise never runs,
    // so the old path paid the upsampler + a 4x-pixel decode for a soft
    // unrefined upscale — at current production formats (5s 480p = 24,960 >
    // 20,000) that was EVERY Kira clip, ~30% of each render wasted. Skip
    // early and decode base latents at the requested size.
    // LTX2_REFINE_UPSCALE_ON_SKIP=1 restores the soft-upscale path for A/B.
    // Refine scale (Todd 2026-08-07: "1.5 is enough" / "render times are
    // already too long"): the learned upsampler is fixed 2x, but the refine
    // denoise — the expensive stage on long clips — runs at rScale. Below 2
    // the upsampled latent is bilinearly resized down first: refine area
    // drops to (rScale/2)^2 (56% at 1.5) and the volume gates reflect the
    // REAL refine size, so more clips refine instead of skipping.
    let rScale = max(1.0, min(2.0, resolvedConfig.refineScale))
    let sLatH = max(1, Int((Float(latH) * rScale).rounded()))
    let sLatW = max(1, Int((Float(latW) * rScale).rounded()))
    let preVolume = latents.dim(2) * sLatH * sLatW
    let preMaxVolume = resolvedConfig.refineMaxVol
    if resumeRefine == nil, preVolume > preMaxVolume,
       ProcessInfo.processInfo.environment["LTX2_REFINE_UPSCALE_ON_SKIP"] != "1",
       ProcessInfo.processInfo.environment["LTX2_REFINE_DECODE_ONLY"] != "1" {
      logger.info("Two-stage refine: SKIPPED entirely (volume \(preVolume) > \(preMaxVolume)) — native-size decode (gate-skip fix 2026-08-05).")
      return .completed(latents)
    }
    var upLatent: MLXArray
    if let rr = resumeRefine, let clean = rr.refineCleanLatents {
      upLatent = clean
      logger.info("#1479 resume: refine re-entering at step \(rr.stepIndex) from the checkpointed clean base latent.")
    } else {
    logger.info("Two-stage refine: upsampling latents 2x...")
    // The latent upsampler is trained on UN-normalized (VAE-scale) latents.
    let stats = vae.decoder.perChannelStatistics
    let denorm = stats.unNormalize(latents.asType(.float32))
    let upDenorm = ups(denorm)
    upLatent = stats.normalize(upDenorm).asType(.float32)
    if sLatH < latH * 2 || sLatW < latW * 2 {
      logger.info("Two-stage refine: resizing 2x latent to \(rScale)x (\(sLatH)x\(sLatW) latent) before denoise.")
      upLatent = LTX2Conditioning.resizeLatentBilinear(upLatent, height: sLatH, width: sLatW)
    }
    eval(upLatent)
    }
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
    let refineVolume = upLatent.dim(2) * sLatH * sLatW
    let refineMaxVolume = resolvedConfig.refineMaxVol
    if resumeRefine == nil, ProcessInfo.processInfo.environment["LTX2_REFINE_DECODE_ONLY"] == "1" {
      logger.info("Two-stage refine: DECODE_ONLY (skipped denoise) — decoding upsampled latent directly.")
      return .completed(upLatent)
    }
    if resumeRefine == nil, refineVolume > refineMaxVolume {
      logger.info("Two-stage refine: SKIPPED denoise (refine latent volume \(refineVolume) > \(refineMaxVolume)) — decoding upsampled latent directly (OOM guard).")
      return .completed(upLatent)
    }
    MLX.GPU.clearCache()
    // The refine denoise runs at the RESIZED dims (rScale, 1.5x default), not
    // the upsampler fixed 2x. The T2V refine block already uses sLat*; this
    // shared path (i2v + continuation chunks) was missed, so the position
    // grid was built for a different token count than the latent.
    let rLatH = sLatH, rLatW = sLatW
    let refineSigmas: [Float] = resolvedConfig.refineSigmasEffective
    let sigma0 = refineSigmas[0]
    // Audio bypasses pass 2 by default — see the t2v refine block comment
    // (final listen verdict 2026-08-05). LTX2_AUDIO_REFINE=1 re-enables.
    // Resolved config, not ProcessInfo — see the t2v refine block.
    let refineAudio = resolvedConfig.audioRefine
    // #1479: the re-noise and the audio re-noise already happened before the
    // checkpoint — running them again would restart the pass.
    var mixed = upLatent
    if resumeRefine == nil {
      if let seed = seed { MLXRandom.seed(seed &+ 1000) }
      // Flow-matching re-noise: x_σ = (1-σ)·x0 + σ·ε (ComfyUI CONST noise_scaling).
      let refNoise = MLXRandom.normal(upLatent.shape).asType(.float32)
      mixed = MLXArray(1 - sigma0) * upLatent + refNoise * MLXArray(sigma0)
      if refineAudio, let av = avState {
        let aKey = seed.map { MLXRandom.key($0 &+ 0xA0D11) }
        let aNoise = MLXRandom.normal(av.audioLatents.shape, key: aKey).asType(.float32)
        av.audioLatents = MLXArray(1 - sigma0) * av.audioLatents + aNoise * MLXArray(sigma0)
      }
    }
    // Frame-0 re-anchor: raw source re-encoded at refine res when available
    // (workflow nodes 19/20); else the upsampled latent's own frame 0.
    let rF = upLatent.dim(2)
    var anchorFrame = upLatent[0..., 0..., 0..<1, 0..., 0...]
    if var refImg = refineAnchorImage {
      if refImg.ndim == 4 { refImg = refImg.expandedDimensions(axis: 2) }
      anchorFrame = vae.encode(refImg).asType(.float32)
      // The re-encoded anchor lands at the source image own resolution (the
      // learned upsampler fixed 2x); upLatent may have been resized down to
      // rScale (1.5x default). Match it, or the frame-axis concat below traps:
      // "[concatenate] All the input array dimensions must match exactly except
      // for the concatenation axis" — this killed EVERY i2v refine and took the
      // server down with it (36 restarts, 2026-08-09).
      if anchorFrame.dim(3) != upLatent.dim(3) || anchorFrame.dim(4) != upLatent.dim(4) {
        anchorFrame = LTX2Conditioning.resizeLatentBilinear(
          anchorFrame, height: upLatent.dim(3), width: upLatent.dim(4))
      }
      eval(anchorFrame)
      logger.info("Two-stage refine: frame 0 re-anchored to source re-encoded at \(rLatW * spatialCompression)x\(rLatH * spatialCompression).")
    }
    rowStats("mixed", mixed)
    // #1479: on resume the refine loop re-enters from its checkpointed
    // latents; `mixed` (the fresh flow re-noise) is bypassed entirely.
    let refineInit = resumeRefine?.videoLatents ?? MLX.concatenated(
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
    // Rebuild the AV PE bundle against the refine grid (video token count and
    // spatial coords change at 2x; audio coords are unchanged).
    var refineAVState: LTX2AVDenoiseState? = nil
    if refineAudio, let av = avState {
      let (_, aCoords) = LTX2AudioPatchifier.patchify(av.audioLatents)
      let avPE2 = transformer.precomputeAVPositionalEmbeddings(
        positions: refinePos, audioCoords: aCoords)
      refineAVState = LTX2AVDenoiseState(
        audioLatents: av.audioLatents, audioContext: av.audioContext, pe: avPE2)
    }
    logger.info("Two-stage refine: denoising at \(rLatW * spatialCompression)x\(rLatH * spatialCompression), \(refineSigmas.count - 1) steps...")
    // #1479: the refine loop owns its validation basis — its fingerprint and
    // sigma schedule are the REFINE pass's, not the base pass's.
    let refineFingerprint = denoiseConfigFingerprint(
      width: rLatW * spatialCompression, height: rLatH * spatialCompression,
      frames: refineInit.dim(2), steps: refineSigmas.count - 1, cfg: cfgScale)
    var refineStartStep = 0
    if let rr = resumeRefine {
      try LTX2ResumeValidator.validate(
        checkpointFingerprint: rr.configFingerprint, currentFingerprint: refineFingerprint,
        checkpointSigmas: rr.sigmas, currentSigmas: refineSigmas, stepIndex: rr.stepIndex)
      refineStartStep = rr.stepIndex
      restoreAudio(rr, into: refineAVState)
    }
    // base=euler_ancestral_cfg_pp -> refine=euler_cfg_pp (the author's pairing).
    var refined: MLXArray
    switch denoisingLoop(
      latents: refineInit, positions: refinePos, precomputedPE: refinePE,
      textEmbeddings: textEmbeddings, negativeEmbeddings: negativeEmbeddings,
      sigmas: refineSigmas, cfgScale: cfgScale, state: refState,
      forceDeterministic: ProcessInfo.processInfo.environment["LTX2_REFINE_DETERMINISTIC"] != "0",
      avState: refineAVState,
      // #1479 codex review: see the T2V inline refine call site's comment —
      // matches this file's existing `seed &+ 1000` refine-decorrelation
      // convention (the `MLXRandom.seed(seed &+ 1000)` reseed above, for
      // `refNoise`).
      startStep: refineStartStep, seed: seed.map { $0 &+ 1000 },
      preemption: preemption,
      telemetry: telemetry,
      loopPhase: .refineDenoise,
      chunkIndex: chunkIndex,
      progressCallback: progressCallback) {
    case .completed(let l):
      refined = l
      // #1479 codex review: this writeback must only happen on a genuine
      // completion. Kept inside .completed so the .yielded arm never commits
      // partially-stepped refine audio into the caller's AV state past the
      // point the LTX2ResumeState snapshot already captured it.
      if let av = avState, let refAV = refineAVState {
        av.audioLatents = refAV.audioLatents  // propagate refined audio back to the caller's state
      }
    case .yielded(let s):
      return .yielded(refineCheckpoint(s, cleanBase: upLatent, baseAV: avState))
    }
    eval(refined)
    rowStats("refined", refined)
    MLX.GPU.clearCache()
    logger.info("Two-stage refine complete.")
    return .completed(refined)
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
    let fps = resolvedConfig.condFps ?? Float(config.fps)
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
