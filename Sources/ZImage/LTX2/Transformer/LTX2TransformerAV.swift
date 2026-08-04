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
