// FiboTransformer.swift — FIBO transformer with DimFusion per-layer text conditioning
// Ported from mflux: transformer.py (FiboTransformer)
//
// FIBO's transformer is derived from Flux but with key architectural differences:
//
// 1. DimFusion: 46 caption_projection layers inject per-layer text encoder hidden states
//    into the transformer. Each block gets its own projected text conditioning, replacing
//    Flux's single global vec embedding.
//
// 2. Reduced joint blocks: 8 (vs Flux's 19), same 38 single blocks.
// 3. 48 input channels (from Wan 2.2 VAE, vs Flux's 64/128).
// 4. No guidance embeddings (guidance_embeds=false).
// 5. Attention masks: explicit masks computed from prompt + latent attention masks.
// 6. GELU-approximate FFN with bias (vs Flux's SwiGLU without bias).
// 7. RoPE with theta=10000 and axes_dim=[16, 56, 56] (3 axes, not 4).
//
// Forward pass:
// 1. Embed latents (x_embedder) and text (context_embedder)
// 2. Compute timestep embeddings (time_embed)
// 3. Compute RoPE position embeddings (pos_embed)
// 4. Compute attention mask
// 5. Project per-layer text encoder hidden states (caption_projection × 46)
// 6. For each of 8 joint blocks: inject DimFusion conditioning, run block
// 7. For each of 38 single blocks: inject DimFusion conditioning, concat+run block
// 8. Output norm + projection

import Foundation
import MLX
import MLXNN

// MARK: - FIBO Position Embeddings

/// N-dimensional rotary position embeddings for FIBO.
///
/// Computes (cos, sin) pairs per axis from position IDs, concatenated across axes.
/// Uses 3 axes with dims [16, 56, 56] and theta=10000 (different from Flux 2's
/// 4 axes with [32,32,32,32] and theta=2000).
///
/// No learnable parameters.
struct FiboPosEmbed {
  let theta: Int
  let axesDim: [Int]

  init(theta: Int = 10000, axesDim: [Int] = [16, 56, 56]) {
    self.theta = theta
    self.axesDim = axesDim
  }

  /// - Parameter ids: Position IDs `[seqLen, numAxes]` (supports optional batch dim).
  /// - Returns: `(cos, sin)` each `[seqLen, totalDim]` where totalDim = sum(axesDim).
  func callAsFunction(_ ids: MLXArray) -> (MLXArray, MLXArray) {
    var idsFlat = ids
    if idsFlat.ndim == 3 && idsFlat.dim(0) == 1 {
      idsFlat = idsFlat[0]
    }

    let nAxes = idsFlat.dim(-1)
    let pos = idsFlat.asType(.float32)

    var cosOut: [MLXArray] = []
    var sinOut: [MLXArray] = []

    for i in 0..<nAxes {
      let axisDim = axesDim[i]
      let (cos, sin) = get1dRotaryPosEmbed(
        dim: axisDim,
        pos: pos[0..., i],
        theta: Float(theta)
      )
      cosOut.append(cos)
      sinOut.append(sin)
    }

    return (
      MLX.concatenated(cosOut, axis: -1),
      MLX.concatenated(sinOut, axis: -1)
    )
  }

  /// Compute 1D rotary position embeddings.
  private func get1dRotaryPosEmbed(
    dim: Int,
    pos: MLXArray,
    theta: Float
  ) -> (MLXArray, MLXArray) {
    var posFlat = pos
    if posFlat.ndim != 1 {
      posFlat = posFlat.reshaped(-1)
    }
    posFlat = posFlat.asType(.float32)

    // freqs = 1 / (theta ^ (arange(0, dim, 2) / dim))
    let scale = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) / Float(dim) })
    let freqs = 1.0 / MLX.pow(MLXArray(theta), scale)

    // angles = pos[:, None] * freqs[None, :]
    let angles = posFlat.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)
    let cosBase = MLX.cos(angles)
    let sinBase = MLX.sin(angles)

    // Interleave: [cosBase, cosBase] and [sinBase, sinBase] -> [seqLen, dim]
    let cosInterleaved = MLX.stacked([cosBase, cosBase], axis: -1).reshaped(posFlat.dim(0), -1)
    let sinInterleaved = MLX.stacked([sinBase, sinBase], axis: -1).reshaped(posFlat.dim(0), -1)
    return (cosInterleaved, sinInterleaved)
  }
}

// MARK: - FiboTransformer

/// FIBO transformer — denoising backbone with DimFusion per-layer text conditioning.
///
/// Architecture:
/// - 8 joint blocks + 38 single blocks
/// - 46 DimFusion caption projections (one per block)
/// - 48 input channels, 3072 hidden dim, 24 attention heads
/// - No guidance embeddings
///
/// Weight key paths:
/// - `x_embedder.{weight,bias}` — latent embedding
/// - `context_embedder.{weight,bias}` — text embedding
/// - `time_embed.timestep_embedder.*` — timestep MLP
/// - `transformer_blocks.{0-7}.*` — joint blocks
/// - `single_transformer_blocks.{0-37}.*` — single blocks
/// - `caption_projection.{0-45}.linear.weight` — DimFusion projections
/// - `norm_out.{linear,norm}.*` — output adaptive layer norm
/// - `proj_out.{weight,bias}` — output projection
public final class FiboTransformer: Module {
  public let config: FiboTransformerConfig

  let posEmbed: FiboPosEmbed

  @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
  @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
  @ModuleInfo(key: "time_embed") var timeEmbed: FiboTimestepProjEmbedding
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [FiboJointTransformerBlock]
  @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [FiboSingleTransformerBlock]
  @ModuleInfo(key: "caption_projection") var captionProjection: [FiboTextProjection]
  @ModuleInfo(key: "norm_out") var normOut: FiboAdaLayerNormContinuous
  @ModuleInfo(key: "proj_out") var projOut: Linear

  public init(config: FiboTransformerConfig = FiboTransformerConfig()) {
    self.config = config
    let hiddenDim = config.hiddenDim  // 3072

    self.posEmbed = FiboPosEmbed(
      theta: config.ropeTheta,
      axesDim: config.axesDimsRope
    )

    self._xEmbedder.wrappedValue = Linear(config.inChannels, hiddenDim)
    self._contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, hiddenDim)
    self._timeEmbed.wrappedValue = FiboTimestepProjEmbedding(
      embeddingDim: hiddenDim,
      timeTheta: config.timeTheta
    )

    var jointBlocks: [FiboJointTransformerBlock] = []
    for _ in 0..<config.numLayers {
      jointBlocks.append(FiboJointTransformerBlock(
        dim: hiddenDim,
        numAttentionHeads: config.numAttentionHeads,
        attentionHeadDim: config.attentionHeadDim
      ))
    }
    self._transformerBlocks.wrappedValue = jointBlocks

    var singleBlocks: [FiboSingleTransformerBlock] = []
    for _ in 0..<config.numSingleLayers {
      singleBlocks.append(FiboSingleTransformerBlock(
        dim: hiddenDim,
        numAttentionHeads: config.numAttentionHeads,
        attentionHeadDim: config.attentionHeadDim
      ))
    }
    self._singleTransformerBlocks.wrappedValue = singleBlocks

    var captions: [FiboTextProjection] = []
    for _ in 0..<config.numCaptionProjectionLayers {
      captions.append(FiboTextProjection(
        inFeatures: config.textEncoderDim,
        hiddenSize: hiddenDim / 2  // 1536 = 3072 / 2
      ))
    }
    self._captionProjection.wrappedValue = captions

    self._normOut.wrappedValue = FiboAdaLayerNormContinuous(
      embeddingDim: hiddenDim,
      conditioningEmbeddingDim: hiddenDim
    )
    self._projOut.wrappedValue = Linear(hiddenDim, config.inChannels)

    super.init()
  }

  /// Run the FIBO transformer forward pass.
  ///
  /// - Parameters:
  ///   - hiddenStates: Latent image tokens `[batch, latentSeq, inChannels]`.
  ///   - encoderHiddenStates: Pooled text embeddings `[batch, txtSeq, jointAttentionDim]`.
  ///   - timestep: Scalar or `[batch]` timestep values.
  ///   - textEncoderLayers: Per-layer text encoder hidden states, 46 arrays each
  ///     `[batch, txtSeq, textEncoderDim]`.
  ///   - height: Image height in pixels (for latent ID computation).
  ///   - width: Image width in pixels (for latent ID computation).
  /// - Returns: Denoised output `[batch, latentSeq, inChannels]`.
  public func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    timestep: MLXArray,
    textEncoderLayers: [MLXArray],
    height: Int,
    width: Int
  ) -> MLXArray {
    let dtype = hiddenStates.dtype
    let batchSize = hiddenStates.dim(0)

    // 1. Handle classifier-free guidance (duplicate latents if encoder has 2x batch)
    var hidden = Self.handleClassifierFreeGuidance(hiddenStates, encoderHiddenStates: encoderHiddenStates)

    // 2. Embed inputs
    hidden = xEmbedder(hidden)
    var encoder = contextEmbedder(encoderHiddenStates)

    // 3. Compute timestep embedding
    let t = MLX.broadcast(timestep, to: [hidden.dim(0)]).asType(dtype)
    let timeEmbeddings = timeEmbed(timestep: t, dtype: dtype)

    // 4. Compute rotary position embeddings
    let imageRotaryEmb = Self.computeRotaryEmbeddings(
      posEmbed: posEmbed,
      encoderHiddenStates: encoder,
      height: height,
      width: width,
      dtype: dtype
    )

    // 5. Compute attention mask
    let attentionMask = Self.computeAttentionMask(
      batchSize: hidden.dim(0),
      encoderHiddenStates: encoder,
      maxTokens: encoder.dim(1),
      height: height,
      width: width
    )

    // 6. Project per-layer text encoder hidden states via DimFusion
    let projectedTextLayers = textEncoderLayers.enumerated().map { (i, layer) in
      captionProjection[i](layer)
    }

    // 7. Joint transformer blocks
    var blockId = 0
    for block in transformerBlocks {
      (encoder, hidden) = Self.applyJointBlock(
        block: block,
        timeEmbeddings: timeEmbeddings,
        hiddenStates: hidden,
        encoderHiddenStates: encoder,
        textEncoderLayer: projectedTextLayers[blockId],
        imageRotaryEmb: imageRotaryEmb,
        attentionMask: attentionMask
      )
      blockId += 1
    }

    // 8. Single transformer blocks
    for block in singleTransformerBlocks {
      (encoder, hidden) = Self.applySingleBlock(
        block: block,
        timeEmbeddings: timeEmbeddings,
        hiddenStates: hidden,
        encoderHiddenStates: encoder,
        textEncoderLayer: projectedTextLayers[blockId],
        imageRotaryEmb: imageRotaryEmb,
        attentionMask: attentionMask
      )
      blockId += 1
    }

    // 9. Output norm + projection
    hidden = normOut(hidden, conditioning: timeEmbeddings)
    hidden = projOut(hidden)

    return hidden
  }

  // MARK: - DimFusion Block Application

  /// Apply a joint transformer block with DimFusion conditioning.
  ///
  /// DimFusion replaces the first half of the encoder hidden states (1536 dims)
  /// with the projected text encoder layer, then concatenates to form 3072-dim input.
  private static func applyJointBlock(
    block: FiboJointTransformerBlock,
    timeEmbeddings: MLXArray,
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    textEncoderLayer: MLXArray,
    imageRotaryEmb: (MLXArray, MLXArray),
    attentionMask: MLXArray?
  ) -> (MLXArray, MLXArray) {
    // DimFusion: take first half of encoder states (1536-dim), concat with projected text layer
    let encoderHalf = encoderHiddenStates[0..., 0..., ..<1536]
    let fusedEncoder = MLX.concatenated([encoderHalf, textEncoderLayer], axis: -1)

    return block(
      temb: timeEmbeddings,
      hiddenStates: hiddenStates,
      encoderHiddenStates: fusedEncoder,
      imageRotaryEmb: imageRotaryEmb,
      attentionMask: attentionMask
    )
  }

  /// Apply a single transformer block with DimFusion conditioning.
  ///
  /// For single blocks, DimFusion conditioning is applied to the encoder portion,
  /// then encoder + image are concatenated along the sequence dimension.
  private static func applySingleBlock(
    block: FiboSingleTransformerBlock,
    timeEmbeddings: MLXArray,
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    textEncoderLayer: MLXArray,
    imageRotaryEmb: (MLXArray, MLXArray),
    attentionMask: MLXArray?
  ) -> (MLXArray, MLXArray) {
    // DimFusion: same as joint — fuse encoder half with projected text layer
    let encoderHalf = encoderHiddenStates[0..., 0..., ..<1536]
    let fusedEncoder = MLX.concatenated([encoderHalf, textEncoderLayer], axis: -1)

    // Concatenate encoder + image along sequence dim for single block processing
    let combined = MLX.concatenated([fusedEncoder, hiddenStates], axis: 1)

    let result = block(
      temb: timeEmbeddings,
      hiddenStates: combined,
      imageRotaryEmb: imageRotaryEmb,
      attentionMask: attentionMask
    )

    // Split back into encoder and image portions
    let encoderLen = fusedEncoder.dim(1)
    let newEncoder = result[0..., ..<encoderLen, 0...]
    let newHidden = result[0..., encoderLen..., 0...]

    return (newEncoder, newHidden)
  }

  // MARK: - Static Helpers

  /// Handle classifier-free guidance by duplicating latents if needed.
  private static func handleClassifierFreeGuidance(
    _ hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray
  ) -> MLXArray {
    let batchSize = hiddenStates.dim(0)
    let encoderBatchSize = encoderHiddenStates.dim(0)
    if encoderBatchSize == batchSize * 2 {
      return MLX.concatenated([hiddenStates, hiddenStates], axis: 0)
    }
    return hiddenStates
  }

  /// Compute rotary position embeddings for the full text+image sequence.
  private static func computeRotaryEmbeddings(
    posEmbed: FiboPosEmbed,
    encoderHiddenStates: MLXArray,
    height: Int,
    width: Int,
    dtype: DType
  ) -> (MLXArray, MLXArray) {
    let maxTokens = encoderHiddenStates.dim(1)
    // Text position IDs: all zeros (3 axes)
    let txtIds = MLX.zeros([maxTokens, 3], dtype: dtype)
    // Image position IDs: [0, row, col] for each latent patch
    let imgIds = prepareLatentImageIds(height: height, width: width, dtype: dtype)
    let ids = MLX.concatenated([txtIds, imgIds], axis: 0)
    return posEmbed(ids.expandedDimensions(axis: 0))
  }

  /// Prepare latent image position IDs: [0, row, col] per pixel.
  private static func prepareLatentImageIds(
    height: Int,
    width: Int,
    dtype: DType = .float32
  ) -> MLXArray {
    let vaeScaleFactor = 16
    let latentHeight = height / vaeScaleFactor
    let latentWidth = width / vaeScaleFactor

    let rowIndices = MLXArray(stride(from: 0, to: latentHeight, by: 1).map { Float($0) })
      .expandedDimensions(axis: -1)
    let rowBroadcast = MLX.broadcast(rowIndices, to: [latentHeight, latentWidth])

    let colIndices = MLXArray(stride(from: 0, to: latentWidth, by: 1).map { Float($0) })
      .expandedDimensions(axis: 0)
    let colBroadcast = MLX.broadcast(colIndices, to: [latentHeight, latentWidth])

    let zeros = MLX.zeros([latentHeight, latentWidth], dtype: .float32)
    let stacked = MLX.stacked([zeros, rowBroadcast, colBroadcast], axis: -1)
    return stacked.reshaped(latentHeight * latentWidth, 3).asType(dtype)
  }

  /// Compute attention mask from prompt + latent sequence lengths.
  ///
  /// FIBO uses explicit attention masks (unlike Flux which uses none).
  /// The mask allows all-to-all attention between prompt and latent tokens.
  private static func computeAttentionMask(
    batchSize: Int,
    encoderHiddenStates: MLXArray,
    maxTokens: Int,
    height: Int,
    width: Int
  ) -> MLXArray {
    let vaeScaleFactor = 16
    let latentHeight = height / vaeScaleFactor
    let latentWidth = width / vaeScaleFactor
    let latentSeqLen = latentHeight * latentWidth

    let promptMask = MLX.ones([batchSize, maxTokens], dtype: .float32)
    let latentMask = MLX.ones([batchSize, latentSeqLen], dtype: .float32)
    let mask2d = MLX.concatenated([promptMask, latentMask], axis: 1)

    return prepareAttentionMask(mask2d).asType(encoderHiddenStates.dtype)
  }

  /// Convert 2D attention mask to 4D additive mask.
  ///
  /// - Parameter mask2d: `[batch, seqTotal]` binary mask.
  /// - Returns: `[batch, 1, seqTotal, seqTotal]` additive mask
  ///   (0 for attend, large negative for mask).
  private static func prepareAttentionMask(_ mask2d: MLXArray) -> MLXArray {
    // Outer product: [B, S] x [B, S] -> [B, S, S]
    let attentionMatrix = MLX.einsum("bi,bj->bij", mask2d, mask2d)
    let maskDtype = mask2d.dtype

    // Where mask == 1: 0 (attend), where mask == 0: -inf (mask out)
    // Use a very large negative value instead of -inf for numerical stability
    let minVal = MLXArray(Float(-1e9))
    let zerosMatrix = MLX.zeros(like: attentionMatrix).asType(maskDtype)
    let onesMatrix = (MLX.ones(like: attentionMatrix) * minVal).asType(maskDtype)
    let result = MLX.where(attentionMatrix .== 1, zerosMatrix, onesMatrix)
    return result.expandedDimensions(axis: 1)
  }
}
