// LTX2TransformerBlock.swift — BasicAVTransformerBlock for the DiT transformer
// Phase 3 of the LTX-2 Swift/MLX port
//
// Each of the 48 blocks contains:
// 1. RMSNorm -> AdaLN modulate (scale1, shift1, gate1) -> Self-Attention (+RoPE) -> gate * output -> residual
// 2. RMSNorm -> Cross-Attention(text) -> residual
//    (LTX-2.3: Q modulated by timestep, context modulated by prompt_adaln)
// 3. [Stub] Cross-Modal Attention (A2V/V2A) — deferred to Phase 5
// 4. RMSNorm -> AdaLN modulate (scale2, shift2, gate2) -> GEGLU FFN -> gate * output -> residual
//
// AdaLN produces 6 params (LTX-2) or 9 params (LTX-2.3) from the scale_shift_table
// + timestep embedding. Applied as: x = gate * op(norm(x) * (1 + scale) + shift)
//
// Reference: transformer.py class BasicAVTransformerBlock

import Foundation
import MLX
import MLXFast
import MLXNN

/// A single transformer block in the LTX-2 DiT architecture.
///
/// Processes video-only in Phase 3 (audio support deferred to Phase 5).
/// Each block has self-attention with RoPE, cross-attention with text,
/// and a feed-forward network, all with adaptive layer normalization.
///
/// Weight key mapping:
/// - `transformer_blocks.N.attn1.*` (self-attention)
/// - `transformer_blocks.N.attn2.*` (cross-attention)
/// - `transformer_blocks.N.ff.*` (feed-forward)
/// - `transformer_blocks.N.scale_shift_table` (AdaLN parameters)
/// - `transformer_blocks.N.prompt_scale_shift_table` (LTX-2.3 only)
public final class LTX2TransformerBlock: Module {
  let normEps: Float
  let hasPromptAdaLN: Bool
  let numAdaParams: Int
  private let rmsNormOnes: MLXArray

  // Self-attention
  @ModuleInfo(key: "attn1") var attn1: LTX2Attention

  // Cross-attention with text
  @ModuleInfo(key: "attn2") var attn2: LTX2Attention

  // Feed-forward network
  @ModuleInfo(key: "ff") var ff: LTX2FeedForward

  // AdaLN scale-shift table: (numAdaParams, dim)
  // LTX-2: 6 params (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
  // LTX-2.3: 9 params (adds shift_q, scale_q, gate_q for cross-attention)
  @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

  // LTX-2.3: prompt-conditioned scale-shift for cross-attention context
  @ParameterInfo(key: "prompt_scale_shift_table") var promptScaleShiftTable: MLXArray?

  // ---- Audio branch (JoyAI-Echo, Phase 2) — built only when `hasAudio`. ----
  // All optional so the video-only path (and its weight layout) is untouched.
  public let hasAudio: Bool
  let audioDim: Int
  private let audioRmsNormOnes: MLXArray

  /// Audio self-attention (`audio_attn1`, dim 2048, 32×64).
  @ModuleInfo(key: "audio_attn1") var audioAttn1: LTX2Attention?
  /// Audio cross-attention with the audio text embeddings (`audio_attn2`).
  @ModuleInfo(key: "audio_attn2") var audioAttn2: LTX2Attention?
  /// Audio feed-forward (`audio_ff`).
  @ModuleInfo(key: "audio_ff") var audioFF: LTX2FeedForward?
  /// Audio AdaLN table (`audio_scale_shift_table`, [9, 2048]).
  @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray?
  /// Audio prompt AdaLN table (`audio_prompt_scale_shift_table`, [2, 2048]).
  @ParameterInfo(key: "audio_prompt_scale_shift_table") var audioPromptScaleShiftTable: MLXArray?
  /// A2V cross-modal attention: Q=video(4096), KV=audio(2048), out→video(4096).
  @ModuleInfo(key: "audio_to_video_attn") var audioToVideoAttn: LTX2Attention?
  /// V2A cross-modal attention: Q=audio(2048), KV=video(4096), out→audio(2048).
  @ModuleInfo(key: "video_to_audio_attn") var videoToAudioAttn: LTX2Attention?
  /// Per-block cross-modal AdaLN tables ([5, 4096] / [5, 2048]).
  @ParameterInfo(key: "scale_shift_table_a2v_ca_video") var scaleShiftTableA2VCaVideo: MLXArray?
  @ParameterInfo(key: "scale_shift_table_a2v_ca_audio") var scaleShiftTableA2VCaAudio: MLXArray?

  /// Initialize a transformer block.
  ///
  /// - Parameters:
  ///   - dim: Hidden dimension (inner_dim, e.g. 4096).
  ///   - contextDim: Text embedding dimension for cross-attention (e.g. 4096).
  ///   - heads: Number of attention heads. Default 32.
  ///   - dimHead: Dimension per head. Default 128.
  ///   - normEps: Epsilon for RMSNorm. Default 1e-6.
  ///   - ropeMode: RoPE mode. Default `.split`.
  ///   - hasPromptAdaLN: Whether to use 9-param AdaLN (LTX-2.3). Default false.
  public init(
    dim: Int,
    contextDim: Int,
    heads: Int = 32,
    dimHead: Int = 128,
    normEps: Float = 1e-6,
    ropeMode: LTX2RoPEMode = .split,
    hasPromptAdaLN: Bool = false,
    hasAudio: Bool = false,
    audioDim: Int = 2048,
    audioHeads: Int = 32,
    audioDimHead: Int = 64
  ) {
    self.normEps = normEps
    self.hasPromptAdaLN = hasPromptAdaLN
    self.numAdaParams = hasPromptAdaLN ? 9 : 6
    self.rmsNormOnes = MLXArray.ones([dim]).asType(.bfloat16)
    self.hasAudio = hasAudio
    self.audioDim = audioDim
    self.audioRmsNormOnes = MLXArray.ones([audioDim]).asType(.bfloat16)

    // Self-attention (no context_dim = self-attention)
    self._attn1.wrappedValue = LTX2Attention(
      queryDim: dim,
      contextDim: nil,
      heads: heads,
      dimHead: dimHead,
      normEps: normEps,
      ropeMode: ropeMode,
      hasGateLogits: hasPromptAdaLN
    )

    // Cross-attention with text
    self._attn2.wrappedValue = LTX2Attention(
      queryDim: dim,
      contextDim: contextDim,
      heads: heads,
      dimHead: dimHead,
      normEps: normEps,
      ropeMode: ropeMode,
      hasGateLogits: hasPromptAdaLN
    )

    // Feed-forward
    self._ff.wrappedValue = LTX2FeedForward(dim: dim, dimOut: dim)

    // AdaLN parameters
    self._scaleShiftTable.wrappedValue = MLXArray.zeros([numAdaParams, dim])

    if hasPromptAdaLN {
      self._promptScaleShiftTable.wrappedValue = MLXArray.zeros([2, dim])
    }

    if hasAudio {
      // Audio self / cross / FF (dim 2048, 32×64, gated like the 2.3 video attns).
      self._audioAttn1.wrappedValue = LTX2Attention(
        queryDim: audioDim, contextDim: nil, heads: audioHeads, dimHead: audioDimHead,
        normEps: normEps, ropeMode: ropeMode, hasGateLogits: true)
      self._audioAttn2.wrappedValue = LTX2Attention(
        queryDim: audioDim, contextDim: audioDim, heads: audioHeads, dimHead: audioDimHead,
        normEps: normEps, ropeMode: ropeMode, hasGateLogits: true)
      self._audioFF.wrappedValue = LTX2FeedForward(dim: audioDim, dimOut: audioDim)
      self._audioScaleShiftTable.wrappedValue = MLXArray.zeros([9, audioDim])
      self._audioPromptScaleShiftTable.wrappedValue = MLXArray.zeros([2, audioDim])
      // Cross-modal: A2V updates video (Q=video), V2A updates audio (Q=audio).
      self._audioToVideoAttn.wrappedValue = LTX2Attention(
        queryDim: dim, contextDim: audioDim, heads: audioHeads, dimHead: audioDimHead,
        normEps: normEps, ropeMode: ropeMode, hasGateLogits: true)
      self._videoToAudioAttn.wrappedValue = LTX2Attention(
        queryDim: audioDim, contextDim: dim, heads: audioHeads, dimHead: audioDimHead,
        normEps: normEps, ropeMode: ropeMode, hasGateLogits: true)
      self._scaleShiftTableA2VCaVideo.wrappedValue = MLXArray.zeros([5, dim])
      self._scaleShiftTableA2VCaAudio.wrappedValue = MLXArray.zeros([5, audioDim])
    }
  }

  /// Weight-free RMSNorm over the audio hidden width.
  private func audioRMSNorm(_ x: MLXArray) -> MLXArray {
    MLXFast.rmsNorm(x, weight: audioRmsNormOnes, eps: normEps)
  }

  /// Dual-stream forward: run the video sub-steps (self-attn, text cross-attn,
  /// FF) alongside the audio sub-steps and the bidirectional a2v/v2a cross-modal
  /// attention, returning updated `(video, audio)`.
  ///
  /// Mirrors `BasicAVTransformerBlock.forward` (av_model.py):
  /// - Cross-modal AdaLN uses the [5, dim] `scale_shift_table_a2v_ca_*` tables
  ///   with SCALE-FIRST row order: rows 0/1 = a2v scale/shift, rows 2/3 = v2a
  ///   scale/shift, row 4 = gate. (The main 9-row table is shift-first.)
  /// - Both cross-modal attention inputs are norm+modulated; the norms are
  ///   taken ONCE before a2v and reused for v2a (the reference computes
  ///   vx_norm3/ax_norm3 before either update).
  /// - a2v runs with `pe: crossPE, kPE: audioCrossPE`; v2a with the mirror.
  ///
  /// - Parameters:
  ///   - video: Video hidden states `(B, T_v, dim)`.
  ///   - audio: Audio hidden states `(B, T_a, audioDim)`.
  ///   - context: Video text embeddings.
  ///   - audioContext: Audio text embeddings `(B, S_a, audioDim)`.
  ///   - timestep: Video timestep embedding `(B, 1, 9*dim)`.
  ///   - audioTimestep: Audio timestep embedding `(B, 1, 9*audioDim)`.
  ///   - pe: Video RoPE.
  ///   - audioPE: Audio RoPE.
  ///   - crossPE: Video-side RoPE for the cross-modal attentions.
  ///   - audioCrossPE: Audio-side RoPE for the cross-modal attentions.
  ///   - crossScaleShiftTimestep: Video av_ca scale-shift timestep `(B, 1, 4*dim)`.
  ///   - audioCrossScaleShiftTimestep: Audio av_ca scale-shift timestep `(B, 1, 4*audioDim)`.
  ///   - crossGateTimestep: Video av_ca gate timestep `(B, 1, dim)`.
  ///   - audioCrossGateTimestep: Audio av_ca gate timestep `(B, 1, audioDim)`.
  ///     Nil cross timesteps degrade to table-rows-only modulation.
  public func callDualStream(
    video: MLXArray,
    audio: MLXArray,
    context: MLXArray,
    audioContext: MLXArray,
    contextMask: MLXArray? = nil,
    audioContextMask: MLXArray? = nil,
    timestep: MLXArray,
    audioTimestep: MLXArray,
    pe: (cos: MLXArray, sin: MLXArray)? = nil,
    audioPE: (cos: MLXArray, sin: MLXArray)? = nil,
    crossPE: (cos: MLXArray, sin: MLXArray)? = nil,
    audioCrossPE: (cos: MLXArray, sin: MLXArray)? = nil,
    crossScaleShiftTimestep: MLXArray? = nil,
    audioCrossScaleShiftTimestep: MLXArray? = nil,
    crossGateTimestep: MLXArray? = nil,
    audioCrossGateTimestep: MLXArray? = nil,
    promptTimestep: MLXArray? = nil,
    audioPromptTimestep: MLXArray? = nil
  ) -> (video: MLXArray, audio: MLXArray) {
    precondition(hasAudio, "callDualStream requires hasAudio")
    let b = video.dim(0)

    // ---- Video stream: self-attn, text cross-attn (reuse video sub-logic) ----
    var v = video
    let vMSA = getAdaValues(table: scaleShiftTable, batchSize: b, timestep: timestep, range: 0..<3)
    let vNorm = weightFreeRMSNorm(v) * (1 + vMSA[1]) + vMSA[0]
    v = v + attn1(vNorm, pe: pe) * vMSA[2]

    let vCross = getAdaValues(table: scaleShiftTable, batchSize: b, timestep: timestep, range: 6..<9)
    var vCtx = context
    if let pTS = promptTimestep, let pTable = promptScaleShiftTable {
      let pp = getAdaValues(table: pTable, batchSize: b, timestep: pTS, range: 0..<2)
      vCtx = context * (1 + pp[1]) + pp[0]
    }
    v = v + attn2(weightFreeRMSNorm(v) * (1 + vCross[1]) + vCross[0],
                  context: vCtx, mask: contextMask) * vCross[2]

    // ---- Audio stream: self-attn, text cross-attn ----
    var a = audio
    let aMSA = getAdaValues(table: audioScaleShiftTable!, batchSize: b, timestep: audioTimestep, range: 0..<3)
    a = a + audioAttn1!(audioRMSNorm(a) * (1 + aMSA[1]) + aMSA[0], pe: audioPE) * aMSA[2]

    let aCross = getAdaValues(table: audioScaleShiftTable!, batchSize: b, timestep: audioTimestep, range: 6..<9)
    var aCtx = audioContext
    if let pTS = audioPromptTimestep, let pTable = audioPromptScaleShiftTable {
      let pp = getAdaValues(table: pTable, batchSize: b, timestep: pTS, range: 0..<2)
      aCtx = audioContext * (1 + pp[1]) + pp[0]
    }
    a = a + audioAttn2!(audioRMSNorm(a) * (1 + aCross[1]) + aCross[0],
                        context: aCtx, mask: audioContextMask) * aCross[2]

    // ---- Cross-modal: A2V (update video from audio) + V2A (update audio) ----
    // Norms taken once, BEFORE either update (reference vx_norm3/ax_norm3).
    let vxNorm3 = weightFreeRMSNorm(v)
    let axNorm3 = audioRMSNorm(a)
    let (vCA, vGateCA) = avCrossAdaValues(
      table: scaleShiftTableA2VCaVideo!, batchSize: b,
      scaleShiftTimestep: crossScaleShiftTimestep, gateTimestep: crossGateTimestep)
    let (aCA, aGateCA) = avCrossAdaValues(
      table: scaleShiftTableA2VCaAudio!, batchSize: b,
      scaleShiftTimestep: audioCrossScaleShiftTimestep, gateTimestep: audioCrossGateTimestep)

    // a2v: Q = modulated video, KV = modulated audio (scale row 0, shift row 1).
    let vxScaledA2V = vxNorm3 * (1 + vCA[0]) + vCA[1]
    let axScaledA2V = axNorm3 * (1 + aCA[0]) + aCA[1]
    v = v + audioToVideoAttn!(vxScaledA2V, context: axScaledA2V,
                              pe: crossPE, kPE: audioCrossPE) * vGateCA

    // v2a: Q = modulated audio, KV = modulated video (scale row 2, shift row 3).
    let axScaledV2A = axNorm3 * (1 + aCA[2]) + aCA[3]
    let vxScaledV2A = vxNorm3 * (1 + vCA[2]) + vCA[3]
    a = a + videoToAudioAttn!(axScaledV2A, context: vxScaledV2A,
                              pe: audioCrossPE, kPE: crossPE) * aGateCA

    // ---- Feed-forward on both streams ----
    let vMLP = getAdaValues(table: scaleShiftTable, batchSize: b, timestep: timestep, range: 3..<6)
    v = v + ff(weightFreeRMSNorm(v) * (1 + vMLP[1]) + vMLP[0]) * vMLP[2]
    let aMLP = getAdaValues(table: audioScaleShiftTable!, batchSize: b, timestep: audioTimestep, range: 3..<6)
    a = a + audioFF!(audioRMSNorm(a) * (1 + aMLP[1]) + aMLP[0]) * aMLP[2]

    return (v, a)
  }

  /// AdaLN values for a cross-modal [5, dim] table: rows 0..3 combined with
  /// the 4-row scale-shift timestep, row 4 with the 1-row gate timestep
  /// (reference `get_av_ca_ada_values`). Nil timesteps contribute zero.
  private func avCrossAdaValues(
    table: MLXArray,
    batchSize: Int,
    scaleShiftTimestep: MLXArray?,
    gateTimestep: MLXArray?
  ) -> (values: [MLXArray], gate: MLXArray) {
    let dim = table.dim(1)
    var ss = table[0..<4].expandedDimensions(axes: [0, 1])  // (1, 1, 4, dim)
    if let ts = scaleShiftTimestep {
      ss = ss + ts.reshaped(batchSize, ts.dim(1), 4, dim)
    }
    var values: [MLXArray] = []
    for i in 0..<4 { values.append(ss[0..., 0..., i]) }
    var g = table[4..<5].expandedDimensions(axes: [0, 1])  // (1, 1, 1, dim)
    if let ts = gateTimestep {
      g = g + ts.reshaped(batchSize, ts.dim(1), 1, dim)
    }
    return (values, g[0..., 0..., 0])
  }

  /// Load a single dual-stream (audio+video) block from a block-local extract
  /// (keys already stripped of `model.diffusion_model.transformer_blocks.N.`).
  /// Applies the standard key remaps (`to_out.0` → `to_out`, GEGLU
  /// `ff.net.0.proj`/`ff.net.2` → `ff.proj_in`/`ff.proj_out` — the substring
  /// replace also covers `audio_ff.*`). LTX-2.3 A/V dims are fixed by the
  /// architecture: video 4096/32×128, audio 2048/32×64.
  public static func loadAVBlock(blockExtractPath: String) throws -> LTX2TransformerBlock {
    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: blockExtractPath))
    var weights: [String: MLXArray] = [:]
    for (key, value) in raw {
      var k = key
      k = k.replacingOccurrences(of: "to_out.0.", with: "to_out.")
      k = k.replacingOccurrences(of: "ff.net.0.proj.", with: "ff.proj_in.")
      k = k.replacingOccurrences(of: "ff.net.2.", with: "ff.proj_out.")
      weights[k] = value
    }
    let block = LTX2TransformerBlock(
      dim: 4096, contextDim: 4096, heads: 32, dimHead: 128,
      hasPromptAdaLN: true, hasAudio: true,
      audioDim: 2048, audioHeads: 32, audioDimHead: 64)
    let params = ModuleParameters.unflattened(weights.map { ($0.key, $0.value) })
    try block.update(parameters: params, verify: [.shapeMismatch])
    return block
  }

  /// Extract adaptive normalization values from the scale-shift table.
  ///
  /// Combines the learned table parameters with timestep embeddings.
  ///
  /// - Parameters:
  ///   - table: Scale-shift table `(numParams, dim)`.
  ///   - batchSize: Batch size.
  ///   - timestep: Timestep embeddings `(B, 1, numParams * dim)`.
  ///   - range: Range of parameters to extract.
  /// - Returns: Tuple of extracted parameters, each `(B, 1, dim)`.
  private func getAdaValues(
    table: MLXArray,
    batchSize: Int,
    timestep: MLXArray,
    range: Range<Int>
  ) -> [MLXArray] {
    let numTotalParams = table.dim(0)
    let dim = table.dim(1)

    // table[range]: (numSelected, dim) -> (1, 1, numSelected, dim)
    let tableSlice = table[range.lowerBound..<range.upperBound]
    let tableExpanded = tableSlice.expandedDimensions(axes: [0, 1])

    // Reshape timestep: (B, 1, numParams * dim) -> (B, 1, numParams, dim)
    let tsReshaped = timestep.reshaped(batchSize, timestep.dim(1), numTotalParams, dim)

    // Extract relevant indices: (B, 1, numSelected, dim)
    let tsSlice = tsReshaped[0..., 0..., range.lowerBound..<range.upperBound]

    // Add table + timestep: (B, 1, numSelected, dim)
    let adaValues = tableExpanded + tsSlice

    // Unbind along parameter dimension
    var result: [MLXArray] = []
    for i in 0..<(range.upperBound - range.lowerBound) {
      result.append(adaValues[0..., 0..., i])
    }
    return result
  }

  /// Forward pass through the transformer block.
  ///
  /// - Parameters:
  ///   - x: Hidden states `(B, T, dim)`.
  ///   - context: Text embeddings for cross-attention `(B, S, contextDim)`.
  ///   - contextMask: Optional attention mask for cross-attention.
  ///   - timestep: Timestep embeddings `(B, 1, numAdaParams * dim)`.
  ///   - pe: Position embeddings `(cos, sin)` for self-attention RoPE.
  ///   - skipSelfAttn: Skip self-attention (STG perturbation). Default false.
  ///   - promptTimestep: Prompt-conditioned timestep for LTX-2.3 cross-attention.
  /// - Returns: Updated hidden states `(B, T, dim)`.
  /// - Parameters:
  ///   - negativeContext: NAG's own negative conditioning. The reference wires
  ///     this as a model patch (`LTX2_NAG {nag_cond_video}`) separate from the
  ///     CFG negative, which is inert in that recipe because CFG runs at 1.0.
  ///     Nil (or a disabled `nag`) leaves the block byte-identical to before.
  ///   - nag: Guidance strength/blend/clamp. `.disabled` is a true no-op.
  public func callAsFunction(
    _ x: MLXArray,
    context: MLXArray,
    contextMask: MLXArray? = nil,
    timestep: MLXArray,
    pe: (cos: MLXArray, sin: MLXArray)? = nil,
    skipSelfAttn: Bool = false,
    promptTimestep: MLXArray? = nil,
    negativeContext: MLXArray? = nil,
    nag: LTX2NAGConfig = .disabled
  ) -> MLXArray {
    // Cross-attention against the positive context, then — when NAG is on —
    // the same computation against the negative context, combined by
    // ltx2ApplyNAG. Modulation is applied to BOTH contexts identically so the
    // two attention outputs are comparable.
    let nagActive = nag.isEnabled && negativeContext != nil
    let batchSize = x.dim(0)
    var h = x

    // ---- 1. Self-Attention with AdaLN ----

    let msaParams = getAdaValues(
      table: scaleShiftTable,
      batchSize: batchSize,
      timestep: timestep,
      range: 0..<3
    )
    let shiftMSA = msaParams[0]
    let scaleMSA = msaParams[1]
    let gateMSA = msaParams[2]

    // Pre-norm + modulate
    var normH = weightFreeRMSNorm(h)
    normH = normH * (1 + scaleMSA) + shiftMSA

    // Self-attention with RoPE
    let attnOut = attn1(normH, pe: pe, skipAttention: skipSelfAttn)
    h = h + attnOut * gateMSA

    // ---- 2. Cross-Attention with Text ----

    if hasPromptAdaLN {
      // LTX-2.3: Q modulated by timestep (indices 6-8), context modulated by prompt_adaln
      let crossParams = getAdaValues(
        table: scaleShiftTable,
        batchSize: batchSize,
        timestep: timestep,
        range: 6..<9
      )
      let shiftQ = crossParams[0]
      let scaleQ = crossParams[1]
      let gateQ = crossParams[2]

      let attnInput = weightFreeRMSNorm(h) * (1 + scaleQ) + shiftQ

      // Modulate context with prompt timestep
      var encoderStates = context
      if let promptTS = promptTimestep, let promptTable = promptScaleShiftTable {
        let promptParams = getAdaValues(
          table: promptTable,
          batchSize: batchSize,
          timestep: promptTS,
          range: 0..<2
        )
        let promptShift = promptParams[0]
        let promptScale = promptParams[1]
        encoderStates = context * (1 + promptScale) + promptShift
      }

      var crossOut = attn2(attnInput, context: encoderStates, mask: contextMask)
      if nagActive, let negCtx = negativeContext {
        var negStates = negCtx
        if let promptTS = promptTimestep, let promptTable = promptScaleShiftTable {
          let p = getAdaValues(
            table: promptTable, batchSize: batchSize, timestep: promptTS, range: 0..<2)
          negStates = negCtx * (1 + p[1]) + p[0]
        }
        let negOut = attn2(attnInput, context: negStates, mask: contextMask)
        crossOut = ltx2ApplyNAG(
          positive: crossOut, negative: negOut,
          scale: nag.scale, alpha: nag.alpha, tau: nag.tau)
      }
      h = h + crossOut * gateQ
    } else {
      // LTX-2: simple cross-attention with RMSNorm
      let normed = weightFreeRMSNorm(h)
      var crossOut = attn2(normed, context: context, mask: contextMask)
      if nagActive, let negCtx = negativeContext {
        let negOut = attn2(normed, context: negCtx, mask: contextMask)
        crossOut = ltx2ApplyNAG(
          positive: crossOut, negative: negOut,
          scale: nag.scale, alpha: nag.alpha, tau: nag.tau)
      }
      h = h + crossOut
    }

    // ---- 3. Feed-Forward with AdaLN ----

    let mlpParams = getAdaValues(
      table: scaleShiftTable,
      batchSize: batchSize,
      timestep: timestep,
      range: 3..<6
    )
    let shiftMLP = mlpParams[0]
    let scaleMLP = mlpParams[1]
    let gateMLP = mlpParams[2]

    var normFF = weightFreeRMSNorm(h)
    normFF = normFF * (1 + scaleMLP) + shiftMLP
    let ffOut = ff(normFF)
    h = h + ffOut * gateMLP

    return h
  }

  /// Weight-free RMSNorm using unit weight vector.
  ///
  /// Matches Python's `rms_norm(x, eps)` utility which uses `mx.ones` as weight.
  private func weightFreeRMSNorm(_ x: MLXArray) -> MLXArray {
    return MLXFast.rmsNorm(x, weight: rmsNormOnes, eps: normEps)
  }
}
