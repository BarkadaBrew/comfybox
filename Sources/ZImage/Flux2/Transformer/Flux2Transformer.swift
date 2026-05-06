import Foundation
import MLX
import MLXNN

/// Adaptive layer norm with continuous conditioning for the Flux 2 output projection.
///
/// Applies SiLU-activated linear projection of the conditioning signal to produce
/// scale and shift parameters for layer normalization.
final class Flux2AdaLayerNormContinuous: Module {
  let embeddingDim: Int

  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm

  init(embeddingDim: Int, conditioningEmbeddingDim: Int) {
    self.embeddingDim = embeddingDim
    self._linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim * 2, bias: false)
    self._norm.wrappedValue = LayerNorm(dimensions: embeddingDim, eps: 1e-6, affine: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
    let mod = linear(silu(conditioning))
    let chunkSize = embeddingDim
    let scale = mod[0..., (0 * chunkSize)..<(1 * chunkSize)]
    let shift = mod[0..., (1 * chunkSize)..<(2 * chunkSize)]
    return norm(x) * (1 + scale).expandedDimensions(axis: 1) + shift.expandedDimensions(axis: 1)
  }
}

// MARK: - Flux2Transformer Configuration

/// Configuration for the Flux 2 transformer.
public struct Flux2TransformerConfig {
  public let patchSize: Int
  public let inChannels: Int
  public let outChannels: Int
  public let numLayers: Int
  public let numSingleLayers: Int
  public let attentionHeadDim: Int
  public let numAttentionHeads: Int
  public let jointAttentionDim: Int
  public let timestepGuidanceChannels: Int
  public let mlpRatio: Float
  public let axesDimsRope: [Int]
  public let ropeTheta: Int
  public let guidanceEmbeds: Bool

  public init(
    patchSize: Int = 1,
    inChannels: Int = 128,
    outChannels: Int? = nil,
    numLayers: Int = 5,
    numSingleLayers: Int = 20,
    attentionHeadDim: Int = 128,
    numAttentionHeads: Int = 24,
    jointAttentionDim: Int = 7680,
    timestepGuidanceChannels: Int = 256,
    mlpRatio: Float = 3.0,
    axesDimsRope: [Int] = [32, 32, 32, 32],
    ropeTheta: Int = 2000,
    guidanceEmbeds: Bool = false
  ) {
    self.patchSize = patchSize
    self.inChannels = inChannels
    self.outChannels = outChannels ?? inChannels
    self.numLayers = numLayers
    self.numSingleLayers = numSingleLayers
    self.attentionHeadDim = attentionHeadDim
    self.numAttentionHeads = numAttentionHeads
    self.jointAttentionDim = jointAttentionDim
    self.timestepGuidanceChannels = timestepGuidanceChannels
    self.mlpRatio = mlpRatio
    self.axesDimsRope = axesDimsRope
    self.ropeTheta = ropeTheta
    self.guidanceEmbeds = guidanceEmbeds
  }
}

// MARK: - Flux2Transformer

/// Flux 2 transformer — the denoising backbone for Flux 2 image generation.
///
/// Architecture:
/// 1. Embed timestep (+ optional guidance) into a conditioning vector
/// 2. Embed image latents and text embeddings into the model dimension
/// 3. Compute RoPE for image and text positions
/// 4. Run double-stream (joint) transformer blocks with cross-attention
/// 5. Concatenate streams and run single-stream blocks with parallel self-attention
/// 6. Apply adaptive layer norm and project to output
public final class Flux2Transformer: Module {
  public let config: Flux2TransformerConfig
  let innerDim: Int

  /// DyPE configuration — set before generation to enable high-res RoPE scaling.
  /// When enabled, spatial axes (h, w) use NTK-aware interpolation while
  /// temporal and layer axes remain vanilla.
  public var dyPEConfig: DyPEConfig = .disabled {
    didSet { posEmbed.dyPE = dyPEConfig }
  }

  @ModuleInfo(key: "pos_embed") var posEmbed: Flux2PosEmbed
  @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2TimestepEmbedding
  @ModuleInfo(key: "double_stream_modulation_img") var doubleStreamModulationImg: Flux2Modulation
  @ModuleInfo(key: "double_stream_modulation_txt") var doubleStreamModulationTxt: Flux2Modulation
  @ModuleInfo(key: "single_stream_modulation") var singleStreamModulation: Flux2Modulation

  @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
  @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
  @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [Flux2TransformerBlock]
  @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [Flux2SingleTransformerBlock]
  @ModuleInfo(key: "norm_out") var normOut: Flux2AdaLayerNormContinuous
  @ModuleInfo(key: "proj_out") var projOut: Linear

  public init(config: Flux2TransformerConfig = Flux2TransformerConfig()) {
    self.config = config
    self.innerDim = config.numAttentionHeads * config.attentionHeadDim

    self._posEmbed.wrappedValue = Flux2PosEmbed(
      theta: config.ropeTheta,
      axesDim: config.axesDimsRope
    )
    self._timeGuidanceEmbed.wrappedValue = Flux2TimestepEmbedding(
      inChannels: config.timestepGuidanceChannels,
      embeddingDim: innerDim,
      guidanceEmbeds: config.guidanceEmbeds
    )
    self._doubleStreamModulationImg.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2)
    self._doubleStreamModulationTxt.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 2)
    self._singleStreamModulation.wrappedValue = Flux2Modulation(dim: innerDim, modParamSets: 1)

    self._xEmbedder.wrappedValue = Linear(config.inChannels, innerDim, bias: false)
    self._contextEmbedder.wrappedValue = Linear(config.jointAttentionDim, innerDim, bias: false)

    var doubleBlocks: [Flux2TransformerBlock] = []
    for _ in 0..<config.numLayers {
      doubleBlocks.append(Flux2TransformerBlock(
        dim: innerDim,
        numAttentionHeads: config.numAttentionHeads,
        attentionHeadDim: config.attentionHeadDim,
        mlpRatio: config.mlpRatio
      ))
    }
    self._transformerBlocks.wrappedValue = doubleBlocks

    var singleBlocks: [Flux2SingleTransformerBlock] = []
    for _ in 0..<config.numSingleLayers {
      singleBlocks.append(Flux2SingleTransformerBlock(
        dim: innerDim,
        numAttentionHeads: config.numAttentionHeads,
        attentionHeadDim: config.attentionHeadDim,
        mlpRatio: config.mlpRatio
      ))
    }
    self._singleTransformerBlocks.wrappedValue = singleBlocks

    self._normOut.wrappedValue = Flux2AdaLayerNormContinuous(
      embeddingDim: innerDim,
      conditioningEmbeddingDim: innerDim
    )
    self._projOut.wrappedValue = Linear(
      innerDim,
      config.patchSize * config.patchSize * config.outChannels,
      bias: false
    )

    super.init()
  }

  /// Run the Flux 2 transformer forward pass.
  ///
  /// - Parameters:
  ///   - hiddenStates: Image latents `[batch, seq, inChannels]`.
  ///   - encoderHiddenStates: Text embeddings `[batch, txtSeq, jointAttentionDim]`.
  ///   - timestep: Scalar or `[batch]` timestep values.
  ///   - imgIds: Image position IDs `[seqLen, numAxes]` or `[1, seqLen, numAxes]`.
  ///   - txtIds: Text position IDs `[txtSeqLen, numAxes]` or `[1, txtSeqLen, numAxes]`.
  ///   - guidance: Optional guidance scale (for distilled models).
  /// - Returns: Denoised output `[batch, seq, outChannels]`.
  public func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    timestep: MLXArray,
    imgIds: MLXArray,
    txtIds: MLXArray,
    guidance: MLXArray? = nil
  ) -> MLXArray {
    let dtype = hiddenStates.dtype

    // Prepare timestep: ensure [batch] shape, scale to [0, 1000] if needed
    var t = timestep.ndim == 0
      ? MLX.broadcast(timestep, to: [hiddenStates.dim(0)]).asType(dtype)
      : timestep.asType(dtype)
    let tScale = MLX.where(MLX.max(t) .<= 1.0, MLXArray(1000.0), MLXArray(1.0)).asType(dtype)
    t = t * tScale

    // Prepare guidance similarly
    var g: MLXArray? = nil
    if var guidance = guidance {
      if guidance.ndim == 0 {
        guidance = MLX.broadcast(guidance, to: [hiddenStates.dim(0)]).asType(dtype)
      } else {
        guidance = guidance.asType(dtype)
      }
      let gScale = MLX.where(MLX.max(guidance) .<= 1.0, MLXArray(1000.0), MLXArray(1.0)).asType(dtype)
      g = guidance * gScale
    }

    // Timestep embedding
    var temb = timeGuidanceEmbed(timestep: t, guidance: g)
    temb = temb.asType(.bfloat16)

    // Embed inputs
    var hidden = xEmbedder(hiddenStates)
    var encoder = contextEmbedder(encoderHiddenStates)

    // Squeeze batch dimension from position IDs if present
    let imgIdsFlat = imgIds.ndim == 3 ? imgIds[0] : imgIds
    let txtIdsFlat = txtIds.ndim == 3 ? txtIds[0] : txtIds

    // Compute DyPE scale factors for spatial axes when resolution exceeds training base.
    // Flux 2 has 4 RoPE axes: [t, h, w, layer]. Only h (axis 1) and w (axis 2) get scaled.
    let ropeScales: [Float]?
    if dyPEConfig.enabled {
      // imgIds columns: [t, h, w, layer]. Max h/w position = number of patches on that axis.
      // Base patch count = baseResolution / 16 (VAE=8x downscale, patchSize=2 for Klein is 1
      // but Flux2 packs 2x2 patches into channels, so effective spatial tokens = pixels/16).
      let basePatches = Float(dyPEConfig.baseResolution) / 16.0
      let hMax = Float(imgIdsFlat[0..., 1].max().item(Int.self)) + 1.0
      let wMax = Float(imgIdsFlat[0..., 2].max().item(Int.self)) + 1.0
      let hScale = hMax / basePatches
      let wScale = wMax / basePatches
      // [t=1.0, h=hScale, w=wScale, layer=1.0] — only scale spatial axes
      ropeScales = [1.0, hScale, wScale, 1.0]
    } else {
      ropeScales = nil
    }

    // Compute RoPE (with DyPE scales when enabled)
    let imageRotaryEmb = posEmbed(imgIdsFlat, scales: ropeScales)
    let textRotaryEmb = posEmbed(txtIdsFlat)
    let concatRotaryEmb = (
      MLX.concatenated([textRotaryEmb.0, imageRotaryEmb.0], axis: 0),
      MLX.concatenated([textRotaryEmb.1, imageRotaryEmb.1], axis: 0)
    )

    // Compute modulation params (shared across all blocks)
    let tembModParamsImg = doubleStreamModulationImg(temb)
    let tembModParamsTxt = doubleStreamModulationTxt(temb)

    // Double-stream blocks
    for block in transformerBlocks {
      (encoder, hidden) = block(
        hiddenStates: hidden,
        encoderHiddenStates: encoder,
        tembModParamsImg: tembModParamsImg,
        tembModParamsTxt: tembModParamsTxt,
        imageRotaryEmb: concatRotaryEmb
      )
    }

    // Merge streams for single-stream blocks
    hidden = MLX.concatenated([encoder, hidden], axis: 1)

    // Single-stream modulation: extract the first (and only) set
    let singleModParams = singleStreamModulation(temb)[0]
    let singleMod = (singleModParams[0], singleModParams[1], singleModParams[2])

    for block in singleTransformerBlocks {
      hidden = block(
        hiddenStates: hidden,
        tembModParams: singleMod,
        imageRotaryEmb: concatRotaryEmb
      )
    }

    // Slice off encoder tokens, keep only image tokens
    hidden = hidden[0..., encoder.dim(1)..., 0...]

    // Final norm + projection
    hidden = normOut(hidden, conditioning: temb)
    hidden = projOut(hidden)

    return hidden
  }
}
