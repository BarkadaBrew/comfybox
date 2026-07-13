// LTX2Transformer.swift — Top-level X0Model denoising transformer
// Phase 3 of the LTX-2 Swift/MLX port
//
// The main LTX-2 denoising model. Contains:
// - Patchify projection: Linear(128, 4096) to project latent tokens
// - AdaLayerNormSingle: timestep conditioning
// - TextProjection or PromptAdaLN: text conditioning
// - 48 BasicAVTransformerBlock instances
// - Output: AdaLN modulate -> LayerNorm -> Linear(4096, 128)
//
// Forward pass:
//   Input: flattened latent (B, T, 128), timestep, text_embeddings, positions
//   1. Patchify: Linear(128, 4096)
//   2. Timestep -> AdaLayerNormSingle -> modulation params
//   3. Caption -> TextProjection -> pooled conditioning (LTX-2 only)
//   4. Precompute RoPE from positions
//   5. Loop 48 blocks
//   6. Final: scale_shift_table + embedded_timestep -> LayerNorm -> Linear(4096, 128)
//   Output: velocity prediction (B, T, 128)
//
// Reference: ltx_2.py classes LTXModel, X0Model, TransformerArgsPreprocessor

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - LTX2Transformer (LTXModel)

/// The LTX-2 diffusion transformer (video-only configuration).
///
/// Contains the full denoising model: patchify projection, AdaLN conditioning,
/// 48 transformer blocks, and output projection.
///
/// Weight key mapping:
/// - `patchify_proj.weight`, `.bias`
/// - `adaln_single.*`
/// - `caption_projection.*` (LTX-2) or `prompt_adaln_single.*` (LTX-2.3)
/// - `transformer_blocks.N.*`
/// - `scale_shift_table`
/// - `proj_out.weight`, `.bias`
public final class LTX2Transformer: Module {
  let innerDim: Int
  let numHeads: Int
  let headDim: Int
  let numLayers: Int
  let inChannels: Int
  let outChannels: Int
  let normEps: Float
  let hasPromptAdaLN: Bool
  let timestepScaleMultiplier: Float
  let positionalEmbeddingTheta: Float
  let positionalEmbeddingMaxPos: [Int]
  let useMiddleIndicesGrid: Bool
  let ropeMode: LTX2RoPEMode
  let doublePrecisionRoPE: Bool

  // Patchify projection
  @ModuleInfo(key: "patchify_proj") var patchifyProj: Linear

  // Timestep conditioning
  @ModuleInfo(key: "adaln_single") var adaLNSingle: LTX2AdaLayerNormSingle

  // Text conditioning (LTX-2 only, nil for LTX-2.3)
  @ModuleInfo(key: "caption_projection") var captionProjection: LTX2TextProjection?

  // Prompt-conditioned AdaLN (LTX-2.3 only)
  @ModuleInfo(key: "prompt_adaln_single") var promptAdaLNSingle: LTX2AdaLayerNormSingle?

  // Transformer blocks
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [LTX2TransformerBlock]

  // Output projection
  @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
  @ModuleInfo(key: "norm_out") var normOut: LayerNorm
  @ModuleInfo(key: "proj_out") var projOut: Linear

  // ---- Top-level audio branch (JoyAI-Echo, Phase 2) — built when `hasAudio`. ----
  public let hasAudio: Bool
  let audioInnerDim: Int
  @ModuleInfo(key: "audio_patchify_proj") var audioPatchifyProj: Linear?
  @ModuleInfo(key: "audio_proj_out") var audioProjOut: Linear?
  @ModuleInfo(key: "audio_adaln_single") var audioAdaLNSingle: LTX2AdaLayerNormSingle?
  @ModuleInfo(key: "audio_prompt_adaln_single") var audioPromptAdaLNSingle: LTX2AdaLayerNormSingle?
  @ParameterInfo(key: "audio_scale_shift_table") var audioScaleShiftTable: MLXArray?
  // Cross-modal gate / scale-shift AdaLN modules (shared across blocks).
  @ModuleInfo(key: "av_ca_a2v_gate_adaln_single") var avCaA2vGateAdaLN: LTX2AdaLayerNormSingle?
  @ModuleInfo(key: "av_ca_v2a_gate_adaln_single") var avCaV2aGateAdaLN: LTX2AdaLayerNormSingle?
  @ModuleInfo(key: "av_ca_video_scale_shift_adaln_single") var avCaVideoScaleShiftAdaLN: LTX2AdaLayerNormSingle?
  @ModuleInfo(key: "av_ca_audio_scale_shift_adaln_single") var avCaAudioScaleShiftAdaLN: LTX2AdaLayerNormSingle?

  /// Initialize the LTX-2 transformer.
  ///
  /// - Parameters:
  ///   - numHeads: Number of attention heads. Default 32.
  ///   - headDim: Dimension per head. Default 128.
  ///   - inChannels: Latent input channels. Default 128.
  ///   - outChannels: Output channels. Default 128.
  ///   - numLayers: Number of transformer blocks. Default 48.
  ///   - crossAttentionDim: Text embedding dimension. Default 4096.
  ///   - captionChannels: Caption input channels for text projection. Default 3840.
  ///   - normEps: Epsilon for layer norms. Default 1e-6.
  ///   - hasPromptAdaLN: Whether to use LTX-2.3 prompt AdaLN. Default false.
  ///   - timestepScaleMultiplier: Scale factor for timestep. Default 1000.
  ///   - positionalEmbeddingTheta: RoPE theta. Default 10000.
  ///   - positionalEmbeddingMaxPos: Max positions per dim. Default [20, 2048, 2048].
  ///   - useMiddleIndicesGrid: Use midpoint of position ranges. Default true.
  ///   - ropeMode: RoPE mode. Default `.split`.
  public init(
    numHeads: Int = 32,
    headDim: Int = 128,
    inChannels: Int = 128,
    outChannels: Int = 128,
    numLayers: Int = 48,
    crossAttentionDim: Int = 4096,
    captionChannels: Int = 3840,
    normEps: Float = 1e-6,
    hasPromptAdaLN: Bool = false,
    timestepScaleMultiplier: Float = 1000,
    positionalEmbeddingTheta: Float = 10000,
    positionalEmbeddingMaxPos: [Int] = [20, 2048, 2048],
    useMiddleIndicesGrid: Bool = true,
    ropeMode: LTX2RoPEMode = .split,
    doublePrecisionRoPE: Bool = false,
    hasAudio: Bool = false,
    audioInnerDim: Int = 2048,
    audioInChannels: Int = 128
  ) {
    self.hasAudio = hasAudio
    self.audioInnerDim = audioInnerDim
    self.innerDim = numHeads * headDim
    self.numHeads = numHeads
    self.headDim = headDim
    self.numLayers = numLayers
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.normEps = normEps
    self.hasPromptAdaLN = hasPromptAdaLN
    self.timestepScaleMultiplier = timestepScaleMultiplier
    self.positionalEmbeddingTheta = positionalEmbeddingTheta
    self.positionalEmbeddingMaxPos = positionalEmbeddingMaxPos
    self.useMiddleIndicesGrid = useMiddleIndicesGrid
    self.ropeMode = ropeMode
    self.doublePrecisionRoPE = doublePrecisionRoPE

    // Patchify projection: latent channels -> inner dim
    self._patchifyProj.wrappedValue = Linear(inChannels, innerDim, bias: true)

    // Timestep conditioning AdaLN
    let adaLNCoefficient = hasPromptAdaLN ? 9 : 6
    self._adaLNSingle.wrappedValue = LTX2AdaLayerNormSingle(
      embeddingDim: innerDim,
      embeddingCoefficient: adaLNCoefficient
    )

    // Text conditioning
    if hasPromptAdaLN {
      self._captionProjection.wrappedValue = nil
      self._promptAdaLNSingle.wrappedValue = LTX2AdaLayerNormSingle(
        embeddingDim: innerDim,
        embeddingCoefficient: 2
      )
    } else {
      self._captionProjection.wrappedValue = LTX2TextProjection(
        inFeatures: captionChannels,
        hiddenSize: innerDim
      )
      self._promptAdaLNSingle.wrappedValue = nil
    }

    // 48 transformer blocks
    var blocks: [LTX2TransformerBlock] = []
    for _ in 0..<numLayers {
      blocks.append(LTX2TransformerBlock(
        dim: innerDim,
        contextDim: crossAttentionDim,
        heads: numHeads,
        dimHead: headDim,
        normEps: normEps,
        ropeMode: ropeMode,
        hasPromptAdaLN: hasPromptAdaLN,
        hasAudio: hasAudio,
        audioDim: audioInnerDim
      ))
    }
    self._transformerBlocks.wrappedValue = blocks

    // Output projection
    self._scaleShiftTable.wrappedValue = MLXArray.zeros([2, innerDim])
    self._normOut.wrappedValue = LayerNorm(dimensions: innerDim, eps: normEps, affine: false)
    self._projOut.wrappedValue = Linear(innerDim, outChannels)

    // Top-level audio branch.
    if hasAudio {
      self._audioPatchifyProj.wrappedValue = Linear(audioInChannels, audioInnerDim, bias: true)
      self._audioProjOut.wrappedValue = Linear(audioInnerDim, audioInChannels, bias: true)
      self._audioAdaLNSingle.wrappedValue =
        LTX2AdaLayerNormSingle(embeddingDim: audioInnerDim, embeddingCoefficient: 9)
      self._audioPromptAdaLNSingle.wrappedValue =
        LTX2AdaLayerNormSingle(embeddingDim: audioInnerDim, embeddingCoefficient: 2)
      self._audioScaleShiftTable.wrappedValue = MLXArray.zeros([2, audioInnerDim])
      self._avCaA2vGateAdaLN.wrappedValue =
        LTX2AdaLayerNormSingle(embeddingDim: innerDim, embeddingCoefficient: 1)
      self._avCaV2aGateAdaLN.wrappedValue =
        LTX2AdaLayerNormSingle(embeddingDim: audioInnerDim, embeddingCoefficient: 1)
      self._avCaVideoScaleShiftAdaLN.wrappedValue =
        LTX2AdaLayerNormSingle(embeddingDim: innerDim, embeddingCoefficient: 4)
      self._avCaAudioScaleShiftAdaLN.wrappedValue =
        LTX2AdaLayerNormSingle(embeddingDim: audioInnerDim, embeddingCoefficient: 4)
    }

    super.init()
  }

  /// Forward pass through the transformer.
  ///
  /// - Parameters:
  ///   - latent: Flattened latent tokens `(B, numTokens, inChannels)`.
  ///   - timestep: Raw timestep value (will be scaled by timestepScaleMultiplier).
  ///   - context: Text embeddings `(B, seqLen, captionChannels or crossAttentionDim)`.
  ///   - positions: 3D position grid from `LTX2PatchEmbed.makePositionGrid`.
  ///   - contextMask: Optional attention mask for cross-attention.
  ///   - sigma: Raw sigma for prompt AdaLN (LTX-2.3). Default nil.
  ///   - precomputedPE: Optional precomputed RoPE to avoid recomputation.
  ///   - stgBlocks: Block indices where self-attention is skipped (STG).
  /// - Returns: Velocity prediction `(B, numTokens, outChannels)`.
  public func callAsFunction(
    latent: MLXArray,
    timestep: MLXArray,
    context: MLXArray,
    positions: MLXArray,
    contextMask: MLXArray? = nil,
    sigma: MLXArray? = nil,
    precomputedPE: (cos: MLXArray, sin: MLXArray)? = nil,
    stgBlocks: Set<Int>? = nil
  ) -> MLXArray {
    let batchSize = latent.dim(0)

    // 1. Patchify projection
    var x = patchifyProj(latent)

    // 2. Timestep conditioning
    let scaledTimestep = timestep * timestepScaleMultiplier
    let (timestepEmb, embeddedTimestep) = adaLNSingle(
      scaledTimestep.reshaped(-1),
      hiddenDtype: x.dtype
    )
    // Reshape to (B, 1, dim) for broadcasting
    let tsEmb = timestepEmb.reshaped(batchSize, -1, timestepEmb.dim(-1))
    let embTS = embeddedTimestep.reshaped(batchSize, -1, embeddedTimestep.dim(-1))

    // 3. Text conditioning
    var ctx = context
    if let captionProj = captionProjection {
      ctx = captionProj(ctx)
    }
    ctx = ctx.reshaped(batchSize, -1, x.dim(-1))

    // Prepare attention mask
    var attnMask = contextMask
    if let mask = attnMask {
      if mask.dtype != .float32 && mask.dtype != .float16 && mask.dtype != .bfloat16 {
        // Convert boolean/int mask to float: 0 -> -1e9, 1 -> 0
        let floatMask = (mask.asType(latent.dtype) - 1) * 1e9
        attnMask = floatMask.reshaped(mask.dim(0), 1, -1, mask.dim(-1))
      }
    }

    // 4. RoPE
    let pe: (cos: MLXArray, sin: MLXArray)
    if let precomputed = precomputedPE {
      pe = precomputed
    } else {
      pe = ltx2PrecomputeFreqsCIS(
        indicesGrid: positions,
        dim: innerDim,
        theta: positionalEmbeddingTheta,
        maxPos: positionalEmbeddingMaxPos,
        useMiddleIndicesGrid: useMiddleIndicesGrid,
        numAttentionHeads: numHeads,
        ropeMode: ropeMode
      )
    }

    // Prompt-conditioned timestep for LTX-2.3
    var promptTS: MLXArray? = nil
    if let promptAda = promptAdaLNSingle, let s = sigma {
      let scaledSigma = s * timestepScaleMultiplier
      let (pTS, _) = promptAda(scaledSigma.reshaped(-1), hiddenDtype: x.dtype)
      promptTS = pTS.reshaped(batchSize, -1, pTS.dim(-1))
    }

    // 5. Process through all transformer blocks
    let stgSet = stgBlocks ?? Set<Int>()
    for (idx, block) in transformerBlocks.enumerated() {
      x = block(
        x,
        context: ctx,
        contextMask: attnMask,
        timestep: tsEmb,
        pe: pe,
        skipSelfAttn: stgSet.contains(idx),
        promptTimestep: promptTS
      )
    }

    // 6. Output projection with AdaLN modulation
    x = processOutput(x: x, embeddedTimestep: embTS)

    return x
  }

  /// Apply final output modulation and projection.
  ///
  /// Combines the learned scale_shift_table with the embedded timestep,
  /// then applies LayerNorm, modulation, and final Linear.
  ///
  /// - Parameters:
  ///   - x: Hidden states `(B, T, innerDim)`.
  ///   - embeddedTimestep: Timestep embedding `(B, 1, innerDim)`.
  /// - Returns: Output `(B, T, outChannels)`.
  private func processOutput(x: MLXArray, embeddedTimestep: MLXArray) -> MLXArray {
    // scale_shift_table: (2, dim) -> (1, 1, 2, dim)
    // embedded_timestep: (B, 1, dim) -> (B, 1, 1, dim)
    let tableExpanded = scaleShiftTable.expandedDimensions(axes: [0, 1])
    let tsExpanded = embeddedTimestep.expandedDimensions(axis: 2)

    // Combine: broadcasts to (B, 1, 2, dim)
    let scaleShiftValues = tableExpanded + tsExpanded

    // Extract shift (index 0) and scale (index 1)
    let shift = scaleShiftValues[0..., 0..., 0]  // (B, 1, dim)
    let scale = scaleShiftValues[0..., 0..., 1]  // (B, 1, dim)

    // Apply: LayerNorm -> modulate -> project
    var output = normOut(x)
    output = output * (1 + scale) + shift
    output = projOut(output)

    return output
  }

  /// Weight sanitization for loading from safetensors.
  ///
  /// Handles key remapping from HuggingFace format:
  /// - Strips `model.diffusion_model.` prefix
  /// - Remaps `.to_out.0.` -> `.to_out.`
  /// - Remaps `.ff.net.0.proj.` -> `.ff.proj_in.`
  /// - Remaps `.ff.net.2.` -> `.ff.proj_out.`
  /// - Remaps `.linear_1.` -> `.linear1.`
  /// - Remaps `.linear_2.` -> `.linear2.`
  ///
  /// - Parameter weights: Raw weight dictionary.
  /// - Returns: Sanitized weight dictionary.
  /// Audio-inclusive weight sanitization for the JoyAI-Echo dual-stream load.
  ///
  /// Same as `sanitizeWeights` for the video keys, but instead of DROPPING the
  /// `model.diffusion_model.` audio branch it KEEPS and remaps it onto this
  /// module's (hasAudio) parameter namespace: strips the prefix, applies the
  /// shared name remaps (`.to_out.0.`→`.to_out.`, video `.ff.net.*`→`.ff.proj_*`)
  /// plus the audio-FF remaps (`.audio_ff.net.0.proj.`→`.audio_ff.proj_in.`,
  /// `.audio_ff.net.2.`→`.audio_ff.proj_out.`). Connectors and aggregate embeds
  /// (text-encoder concerns) are still skipped.
  ///
  /// Use only with a transformer built `hasAudio: true`.
  public static func sanitizeWeightsWithAudio(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var sanitized: [String: MLXArray] = [:]
    for (key, value) in weights where key.hasPrefix("model.diffusion_model.") {
      // Connectors + aggregate embeds live on the text-encoder side.
      if key.contains("embeddings_connector") || key.contains("aggregate_embed") { continue }

      var newKey = String(key.dropFirst("model.diffusion_model.".count))
      newKey = newKey.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
      // Audio FF first (more specific), then video FF.
      newKey = newKey.replacingOccurrences(of: ".audio_ff.net.0.proj.", with: ".audio_ff.proj_in.")
      newKey = newKey.replacingOccurrences(of: ".audio_ff.net.2.", with: ".audio_ff.proj_out.")
      newKey = newKey.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
      newKey = newKey.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
      newKey = newKey.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
      newKey = newKey.replacingOccurrences(of: ".linear_2.", with: ".linear2.")
      sanitized[newKey] = value
    }
    return sanitized
  }

  public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    // Detect which prefix format the checkpoint uses.
    // LTX-2.3 distilled/dev weights use "transformer." prefix.
    // Older checkpoints use "model.diffusion_model." prefix.
    let hasTransformerPrefix = weights.keys.contains { $0.hasPrefix("transformer.") }
    let hasModelPrefix = weights.keys.contains { $0.hasPrefix("model.diffusion_model.") }

    guard hasTransformerPrefix || hasModelPrefix else { return weights }

    var sanitized: [String: MLXArray] = [:]

    for (key, value) in weights {
      var newKey: String

      if hasTransformerPrefix && key.hasPrefix("transformer.") {
        // Skip audio-only modules — our Swift transformer is video-only.
        // Audio blocks have keys under audio_* or av_ca_* top-level names.
        let stripped = String(key.dropFirst("transformer.".count))
        let isAudioOnly = stripped.hasPrefix("audio_")
          || stripped.hasPrefix("av_ca_")
          || stripped.contains(".audio_attn")
          || stripped.contains(".audio_ff")
          || stripped.contains(".audio_prompt_scale_shift_table")
          || stripped.contains(".audio_scale_shift_table")
          || stripped.contains(".audio_to_video_attn")
          || stripped.contains(".video_to_audio_attn")
          || stripped.contains(".scale_shift_table_a2v")
        if isAudioOnly { continue }

        newKey = stripped

        // Rename norm_out -> norm_out (already correct in weights)
        // No renames needed for transformer. prefix format.

      } else if hasModelPrefix && key.hasPrefix("model.diffusion_model.") {
        // Skip audio/video embedding connectors (handled in text encoder)
        if key.contains("audio_embeddings_connector") || key.contains("video_embeddings_connector") {
          continue
        }

        // Skip the audio / cross-modal DiT branch — our Swift transformer is
        // video-only (audio arrives in Phase 2). JoyAI-Echo's monolith carries
        // the full dual-stream half under this `model.diffusion_model.` prefix;
        // without this the audio keys would be silently dropped by
        // `verify:[.shapeMismatch]`. Make the video-only load explicit so it's
        // intentional, mirroring the skip already present on the `transformer.`
        // branch above.
        let ditKey = String(key.dropFirst("model.diffusion_model.".count))
        let isAudioOnly = ditKey.hasPrefix("audio_")
          || ditKey.hasPrefix("av_ca_")
          || ditKey.contains(".audio_attn")
          || ditKey.contains(".audio_ff")
          || ditKey.contains(".audio_prompt_scale_shift_table")
          || ditKey.contains(".audio_scale_shift_table")
          || ditKey.contains(".audio_to_video_attn")
          || ditKey.contains(".video_to_audio_attn")
          || ditKey.contains(".scale_shift_table_a2v")
        if isAudioOnly { continue }

        newKey = key.replacingOccurrences(of: "model.diffusion_model.", with: "")
        newKey = newKey.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
        newKey = newKey.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.proj_in.")
        newKey = newKey.replacingOccurrences(of: ".ff.net.2.", with: ".ff.proj_out.")
        newKey = newKey.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
        newKey = newKey.replacingOccurrences(of: ".linear_2.", with: ".linear2.")

      } else {
        continue
      }

      sanitized[newKey] = value
    }

    return sanitized
  }
}
