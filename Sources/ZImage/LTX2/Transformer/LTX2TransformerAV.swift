// LTX2TransformerAV.swift — top-level audio conditioning + I/O for the joint
// A/V forward (task #21 wire 1).
//
// Mirrors LTXAVModel's audio-side plumbing (av_model.py) using the hasAudio
// members LTX2Transformer already binds: the audio adaln singles, the four
// av_ca adaln singles (with CROSSED gate inputs and the
// av_ca_timestep_scale_multiplier / timestep_scale_multiplier factor), the
// audio patchify projection, and the audio output processing
// (LayerNorm -> table+embedded modulation -> proj_out -> unpatchify).

import Foundation
import MLX
import MLXFast
import MLXNN

/// Per-step audio-side conditioning bundle consumed by `callDualStream`
/// across all blocks. Field names match the block's parameter labels.
public struct LTX2AVConditioning {
  /// Audio 9-coeff AdaLN timestep `(B, 1, 9*audioDim)`.
  public let audioTimestep: MLXArray
  /// Audio embedded timestep for output modulation `(B, 1, audioDim)`.
  public let audioEmbeddedTimestep: MLXArray
  /// Audio prompt AdaLN timestep `(B, 1, 2*audioDim)`.
  public let audioPromptTimestep: MLXArray
  /// av_ca video scale-shift `(B, 1, 4*dim)`.
  public let crossScaleShiftTimestep: MLXArray
  /// av_ca audio scale-shift `(B, 1, 4*audioDim)`.
  public let audioCrossScaleShiftTimestep: MLXArray
  /// av_ca a2v gate `(B, 1, dim)` — driven by the AUDIO sigma (crossed).
  public let crossGateTimestep: MLXArray
  /// av_ca v2a gate `(B, 1, audioDim)` — driven by the VIDEO sigma (crossed).
  public let audioCrossGateTimestep: MLXArray
}

extension LTX2Transformer {
  /// LTX-2.3 audio latent geometry (fixed by the audio VAE).
  public static let audioLatentChannels = 8
  public static let audioLatentFrequencyBins = 16
  /// Reference av_ca_timestep_scale_multiplier (v16b config default).
  public static let avCaTimestepScaleMultiplier: Float = 1.0

  /// All audio-side timestep conditioning for one denoise step (scalar
  /// sigmas per batch). Requires `hasAudio`.
  ///
  /// Gate inputs are CROSSED per the reference: the a2v gate (applied on the
  /// video stream) is driven by the max AUDIO sigma, the v2a gate by the max
  /// VIDEO sigma — both times avCaTimestepScaleMultiplier, NOT the 1000x
  /// timestep multiplier (av_ca_factor = avCaMult / tsMult cancels it).
  public func prepareAVConditioning(
    videoSigma: Float, audioSigma: Float, batchSize: Int
  ) -> LTX2AVConditioning {
    precondition(hasAudio, "prepareAVConditioning requires hasAudio")
    func rep(_ value: Float) -> MLXArray {
      MLXArray(Array(repeating: value, count: batchSize))
    }
    let avCaMult = Self.avCaTimestepScaleMultiplier
    let aScaled = rep(audioSigma * timestepScaleMultiplier)
    let vScaled = rep(videoSigma * timestepScaleMultiplier)

    let (aTs, aEmbedded) = audioAdaLNSingle!(aScaled, hiddenDtype: .float32)
    let (aPrompt, _) = audioPromptAdaLNSingle!(aScaled, hiddenDtype: .float32)
    let (vSS, _) = avCaVideoScaleShiftAdaLN!(vScaled, hiddenDtype: .float32)
    let (aSS, _) = avCaAudioScaleShiftAdaLN!(aScaled, hiddenDtype: .float32)
    let (a2vGate, _) = avCaA2vGateAdaLN!(rep(audioSigma * avCaMult), hiddenDtype: .float32)
    let (v2aGate, _) = avCaV2aGateAdaLN!(rep(videoSigma * avCaMult), hiddenDtype: .float32)

    func shape3(_ x: MLXArray) -> MLXArray { x.reshaped([batchSize, 1, -1]) }
    return LTX2AVConditioning(
      audioTimestep: shape3(aTs),
      audioEmbeddedTimestep: shape3(aEmbedded),
      audioPromptTimestep: shape3(aPrompt),
      crossScaleShiftTimestep: shape3(vSS),
      audioCrossScaleShiftTimestep: shape3(aSS),
      crossGateTimestep: shape3(a2vGate),
      audioCrossGateTimestep: shape3(v2aGate))
  }

  /// `(B, C, T, F)` audio latents -> projected tokens `(B, T, audioDim)` +
  /// real-time start/end coords `(B, 1, T, 2)`. Requires `hasAudio`.
  public func projectAudioTokens(_ audioLatents: MLXArray) -> (tokens: MLXArray, coords: MLXArray) {
    precondition(hasAudio, "projectAudioTokens requires hasAudio")
    let (tokens, coords) = LTX2AudioPatchifier.patchify(audioLatents)
    return (audioPatchifyProj!(tokens), coords)
  }

  /// All four positional-embedding sets for the joint forward, computed once
  /// per render. `positions` are video pixel coords `(B, 3, Nv, 2)` from
  /// `makePositionGrid`; `audioCoords` are the seconds-valued start/end
  /// coords from `projectAudioTokens`.
  ///
  /// Cross-modal PEs run in REAL SECONDS on both sides (video time row ÷
  /// frameRate; audio coords already in seconds), dim = audioInnerDim,
  /// audio head count, shared maxPos.
  public func precomputeAVPositionalEmbeddings(
    positions: MLXArray,
    audioCoords: MLXArray,
    frameRate: Float
  ) -> (videoPE: (cos: MLXArray, sin: MLXArray),
        audioPE: (cos: MLXArray, sin: MLXArray),
        crossVideoPE: (cos: MLXArray, sin: MLXArray),
        crossAudioPE: (cos: MLXArray, sin: MLXArray)) {
    precondition(hasAudio, "precomputeAVPositionalEmbeddings requires hasAudio")
    let videoPE = ltx2PrecomputeFreqsCIS(
      indicesGrid: positions, dim: innerDim, theta: positionalEmbeddingTheta,
      maxPos: positionalEmbeddingMaxPos, useMiddleIndicesGrid: useMiddleIndicesGrid,
      numAttentionHeads: numHeads, ropeMode: ropeMode, doublePrecision: doublePrecisionRoPE)
    // Audio self PE: 1-D temporal, audio max_pos [20].
    let audioPE = ltx2PrecomputeFreqsCIS(
      indicesGrid: audioCoords, dim: audioInnerDim, theta: positionalEmbeddingTheta,
      maxPos: [20], useMiddleIndicesGrid: useMiddleIndicesGrid,
      numAttentionHeads: audioHeads, ropeMode: ropeMode, doublePrecision: doublePrecisionRoPE)
    // Cross PEs: video time row in seconds; middle-grid ALWAYS on (reference
    // passes use_middle_indices_grid=True explicitly for av cross attention).
    let posF32 = positions.asType(.float32)
    let timeSecs = posF32[0..., 0..<1] * (1.0 / frameRate)
    let maxPos = max(positionalEmbeddingMaxPos[0], 20)
    let crossVideoPE = ltx2PrecomputeFreqsCIS(
      indicesGrid: timeSecs, dim: audioInnerDim, theta: positionalEmbeddingTheta,
      maxPos: [maxPos], useMiddleIndicesGrid: true,
      numAttentionHeads: audioHeads, ropeMode: ropeMode, doublePrecision: doublePrecisionRoPE)
    let crossAudioPE = ltx2PrecomputeFreqsCIS(
      indicesGrid: audioCoords[0..., 0..<1], dim: audioInnerDim, theta: positionalEmbeddingTheta,
      maxPos: [maxPos], useMiddleIndicesGrid: true,
      numAttentionHeads: audioHeads, ropeMode: ropeMode, doublePrecision: doublePrecisionRoPE)
    return (videoPE, audioPE, crossVideoPE, crossAudioPE)
  }

  /// Joint dual-stream forward: video + audio velocities in one pass.
  ///
  /// Mirrors LTXAVModel.forward: per-modality timesteps, the av_ca cross
  /// conditioning from `prepareAVConditioning`, all 48 blocks via
  /// `callDualStream`, then both output projections.
  ///
  /// v1 scope: audio timestep is a SCALAR sigma (t2v audio; per-token audio
  /// timesteps arrive with i2v mask integration); no NAG/STG on this path
  /// yet. `videoSigmaMax` drives the av_ca video scale-shift and the v2a
  /// gate — pass the max video sigma (== sigma for uniform timesteps).
  public func callAV(
    latent: MLXArray,
    audioLatents: MLXArray,
    timestep: MLXArray,
    videoSigmaMax: Float,
    audioSigma: Float,
    context: MLXArray,
    audioContext: MLXArray,
    contextMask: MLXArray? = nil,
    sigma: MLXArray? = nil,
    pe: (videoPE: (cos: MLXArray, sin: MLXArray),
         audioPE: (cos: MLXArray, sin: MLXArray),
         crossVideoPE: (cos: MLXArray, sin: MLXArray),
         crossAudioPE: (cos: MLXArray, sin: MLXArray))
  ) -> (video: MLXArray, audio: MLXArray) {
    precondition(hasAudio, "callAV requires hasAudio")
    let batchSize = latent.dim(0)

    // ---- Video side: identical prep to the video-only forward ----
    var vx = patchifyProj(latent)
    let scaledTimestep = timestep * timestepScaleMultiplier
    let (timestepEmb, embeddedTimestep) = adaLNSingle(
      scaledTimestep.reshaped(-1), hiddenDtype: vx.dtype)
    let vTsEmb = timestepEmb.reshaped(batchSize, -1, timestepEmb.dim(-1))
    let vEmbTS = embeddedTimestep.reshaped(batchSize, -1, embeddedTimestep.dim(-1))

    var ctx = context
    if let captionProj = captionProjection { ctx = captionProj(ctx) }
    ctx = ctx.reshaped(batchSize, -1, vx.dim(-1))

    var attnMask = contextMask
    if let mask = attnMask {
      if mask.dtype != .float32 && mask.dtype != .float16 && mask.dtype != .bfloat16 {
        let floatMask = (mask.asType(latent.dtype) - 1) * 1e9
        attnMask = floatMask.reshaped(mask.dim(0), 1, -1, mask.dim(-1))
      }
    }

    var promptTS: MLXArray? = nil
    if let promptAda = promptAdaLNSingle, let s = sigma {
      let scaledSigma = s * timestepScaleMultiplier
      let (pTS, _) = promptAda(scaledSigma.reshaped(-1), hiddenDtype: vx.dtype)
      promptTS = pTS.reshaped(batchSize, -1, pTS.dim(-1))
    }

    // ---- Audio side ----
    var (ax, _) = projectAudioTokens(audioLatents)
    ax = ax.asType(vx.dtype)
    let av = prepareAVConditioning(
      videoSigma: videoSigmaMax, audioSigma: audioSigma, batchSize: batchSize)
    let aCtx = audioContext.reshaped(batchSize, -1, audioInnerDim).asType(vx.dtype)

    // ---- Joint block loop ----
    for block in transformerBlocks {
      (vx, ax) = block.callDualStream(
        video: vx, audio: ax,
        context: ctx, audioContext: aCtx,
        contextMask: attnMask, audioContextMask: attnMask,
        timestep: vTsEmb, audioTimestep: av.audioTimestep,
        pe: pe.videoPE, audioPE: pe.audioPE,
        crossPE: pe.crossVideoPE, audioCrossPE: pe.crossAudioPE,
        crossScaleShiftTimestep: av.crossScaleShiftTimestep,
        audioCrossScaleShiftTimestep: av.audioCrossScaleShiftTimestep,
        crossGateTimestep: av.crossGateTimestep,
        audioCrossGateTimestep: av.audioCrossGateTimestep,
        promptTimestep: promptTS,
        audioPromptTimestep: av.audioPromptTimestep)
    }

    // ---- Outputs ----
    let videoOut = processOutput(x: vx, embeddedTimestep: vEmbTS)
    let audioOut = processAudioOutput(ax, embeddedTimestep: av.audioEmbeddedTimestep)
    return (videoOut, audioOut)
  }

  /// Audio output processing: LayerNorm (no affine, eps 1e-6) ->
  /// table+embedded modulation (row 0 shift, row 1 scale) -> proj_out ->
  /// unpatchify to `(B, 8, T, 16)`. Requires `hasAudio`.
  public func processAudioOutput(_ ax: MLXArray, embeddedTimestep: MLXArray) -> MLXArray {
    precondition(hasAudio, "processAudioOutput requires hasAudio")
    let ssv = audioScaleShiftTable!.expandedDimensions(axes: [0, 1])
      + embeddedTimestep.expandedDimensions(axis: 2)  // (B, 1, 2, audioDim)
    let shift = ssv[0..., 0..., 0]
    let scale = ssv[0..., 0..., 1]
    var x = MLXFast.layerNorm(ax, weight: nil, bias: nil, eps: 1e-6)
    x = x * (1 + scale) + shift
    x = audioProjOut!(x)
    return LTX2AudioPatchifier.unpatchify(
      x, channels: Self.audioLatentChannels, freq: Self.audioLatentFrequencyBins)
  }
}
