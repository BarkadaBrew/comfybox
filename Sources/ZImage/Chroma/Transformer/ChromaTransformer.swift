import Foundation
import MLX
import MLXFast
import MLXNN

/// Timestep embedding — sinusoidal positional encoding for scalar timesteps.
///
/// Matches the Python `timestep_embedding(t, dim, max_period=10000, time_factor=1000)`.
func chromaTimestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Int = 10000, timeFactor: Float = 1000.0) -> MLXArray {
  let half = dim / 2
  let freqs = MLXArray(stride(from: 0, to: half, by: 1).map { Float($0) / Float(half) })
    * MLXArray(-log(Float(maxPeriod)))
  let freqsExp = MLX.exp(freqs)

  // [B] * timeFactor -> [B, 1] * [1, half] -> [B, half]
  let x = (timeFactor * t).expandedDimensions(axis: -1) * freqsExp.expandedDimensions(axis: 0)
  return MLX.concatenated([MLX.cos(x), MLX.sin(x)], axis: -1).asType(t.dtype)
}

/// Last layer — adaptive norm + linear projection for Chroma output.
///
/// Weight keys: `final_layer.norm_final`, `final_layer.linear`
final class ChromaLastLayer: Module {
  @ModuleInfo(key: "norm_final") var normFinal: LayerNorm
  @ModuleInfo(key: "linear") var linear: Linear

  init(hiddenSize: Int, patchSize: Int, outChannels: Int) {
    self._normFinal.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
    self._linear.wrappedValue = Linear(hiddenSize, patchSize * patchSize * outChannels, bias: true)
    super.init()
  }

  /// Apply final layer with modulation.
  ///
  /// - Parameters:
  ///   - x: Hidden states `[B, seq, dim]`
  ///   - mod: Tuple of (shift, scale) from the last 2 modulation vectors
  func callAsFunction(_ x: MLXArray, mod: (MLXArray, MLXArray)) -> MLXArray {
    let (shift, scale) = mod
    let shiftSqueezed = shift.squeezed(axis: 1)
    let scaleSqueezed = scale.squeezed(axis: 1)
    let normalized = (1 + scaleSqueezed.expandedDimensions(axis: 1)) * normFinal(x)
      + shiftSqueezed.expandedDimensions(axis: 1)
    return linear(normalized)
  }
}

/// Chroma transformer — the full denoising backbone.
///
/// Architecture:
/// 1. Embed image latents and text through linear projections
/// 2. Compute timestep + guidance embeddings, broadcast with modulation indices
/// 3. Run Approximator to produce 344 modulation vectors
/// 4. Run 19 double-stream blocks with per-block modulations
/// 5. Concatenate streams, run 38 single-stream blocks
/// 6. Apply final layer with modulation
///
/// Weight key prefix: none (top-level keys like `img_in`, `txt_in`, `distilled_guidance_layer`, etc.)
public final class ChromaTransformer: Module {
  public let config: ChromaConfig

  @ModuleInfo(key: "img_in") var imgIn: Linear
  @ModuleInfo(key: "txt_in") var txtIn: Linear
  @ModuleInfo(key: "distilled_guidance_layer") var approximator: ChromaApproximator
  @ModuleInfo(key: "double_blocks") var doubleBlocks: [ChromaDoubleStreamBlock]
  @ModuleInfo(key: "single_blocks") var singleBlocks: [ChromaSingleStreamBlock]
  @ModuleInfo(key: "final_layer") var finalLayer: ChromaLastLayer

  let posEmbed: Flux2PosEmbed

  public init(config: ChromaConfig = .standard) {
    self.config = config

    self._imgIn.wrappedValue = Linear(config.inChannels, config.hiddenSize, bias: true)
    self._txtIn.wrappedValue = Linear(config.contextInDim, config.hiddenSize, bias: true)
    self._approximator.wrappedValue = ChromaApproximator(
      inDim: config.approxInDim,
      outDim: config.approxOutDim,
      hiddenDim: config.approxHiddenDim,
      nLayers: config.approxNLayers
    )

    var dblBlocks: [ChromaDoubleStreamBlock] = []
    for _ in 0..<config.depth {
      dblBlocks.append(ChromaDoubleStreamBlock(
        hiddenSize: config.hiddenSize,
        numHeads: config.numHeads,
        mlpRatio: config.mlpRatio,
        qkvBias: config.qkvBias
      ))
    }
    self._doubleBlocks.wrappedValue = dblBlocks

    var snglBlocks: [ChromaSingleStreamBlock] = []
    for _ in 0..<config.depthSingleBlocks {
      snglBlocks.append(ChromaSingleStreamBlock(
        hiddenSize: config.hiddenSize,
        numHeads: config.numHeads,
        mlpRatio: config.mlpRatio
      ))
    }
    self._singleBlocks.wrappedValue = snglBlocks

    self._finalLayer.wrappedValue = ChromaLastLayer(
      hiddenSize: config.hiddenSize,
      patchSize: 1,
      outChannels: config.outChannels
    )

    self.posEmbed = Flux2PosEmbed(theta: config.theta, axesDim: config.axesDim)

    super.init()
  }

  // MARK: - Modulation Slicing

  /// Extract modulations for a specific block from the Approximator output.
  ///
  /// Layout: single(depthSingle*3) | double_img(depth*6) | double_txt(depth*6) | final(2)
  func getModulations(_ tensor: MLXArray, blockType: String, idx: Int = 0) -> Any {
    let singleCount = config.depthSingleBlocks
    let doubleCount = config.depth

    switch blockType {
    case "final":
      let shift = tensor[0..., (tensor.dim(1) - 2)..<(tensor.dim(1) - 1), 0...]
      let scale = tensor[0..., (tensor.dim(1) - 1)..<tensor.dim(1), 0...]
      return (shift, scale) as (MLXArray, MLXArray)

    case "single":
      let offset = 3 * idx
      return ChromaModulation.fromOffset(tensor, offset: offset)

    case "double_img":
      let offset = 3 * singleCount + 6 * idx
      let mod1 = ChromaModulation.fromOffset(tensor, offset: offset)
      let mod2 = ChromaModulation.fromOffset(tensor, offset: offset + 3)
      return (mod1, mod2)

    case "double_txt":
      let offset = 3 * singleCount + 6 * doubleCount + 6 * idx
      let mod1 = ChromaModulation.fromOffset(tensor, offset: offset)
      let mod2 = ChromaModulation.fromOffset(tensor, offset: offset + 3)
      return (mod1, mod2)

    default:
      fatalError("Unknown block type: \(blockType)")
    }
  }

  // MARK: - Forward Pass

  /// Run the Chroma transformer.
  ///
  /// - Parameters:
  ///   - img: Packed image latents `[B, seq, inChannels]`
  ///   - imgIds: Image position IDs `[B, seq, 3]`
  ///   - txt: T5 text embeddings `[B, txtSeq, 4096]`
  ///   - txtIds: Text position IDs `[B, txtSeq, 3]`
  ///   - timesteps: Denoising timestep `[B]`
  ///   - guidance: Guidance scale `[B]`
  /// - Returns: Denoised prediction `[B, seq, outChannels]`
  public func callAsFunction(
    img: MLXArray,
    imgIds: MLXArray,
    txt: MLXArray,
    txtIds: MLXArray,
    timesteps: MLXArray,
    guidance: MLXArray
  ) -> MLXArray {
    let dtype = MLX.DType.bfloat16
    let batch = img.dim(0)
    let modIndexLength = config.modIndexLength

    // Project inputs
    var imgH = imgIn(img)
    var txtH = txtIn(txt).asType(dtype)

    // Build Approximator input: [B, 344, 64]
    let distillTimestep = chromaTimestepEmbedding(
      MLX.stopGradient(timesteps), dim: 16
    ).asType(dtype)
    let distillGuidance = chromaTimestepEmbedding(
      MLX.stopGradient(guidance), dim: 16
    ).asType(dtype)

    // Modulation index: sinusoidal embedding of 0..343
    let modIndexValues = MLXArray(stride(from: 0, to: modIndexLength, by: 1).map { Float($0) })
    var modIndex = chromaTimestepEmbedding(modIndexValues, dim: 32).asType(dtype)
    modIndex = MLX.broadcast(modIndex.expandedDimensions(axis: 0), to: [batch, modIndexLength, 32])

    // Combine timestep + guidance -> [B, 32], broadcast to [B, 344, 32]
    let tg = MLX.concatenated([distillTimestep, distillGuidance], axis: 1).asType(dtype)
    let tgExpanded = MLX.broadcast(tg.expandedDimensions(axis: 1), to: [batch, modIndexLength, 32])

    // Final input: [B, 344, 64]
    let inputVec = MLX.concatenated([tgExpanded, modIndex], axis: -1).asType(dtype)

    // Run Approximator -> [B, 344, 3072]
    let modVectors = approximator(inputVec).asType(dtype)

    // Compute positional encodings
    let ids = MLX.concatenated([txtIds, imgIds], axis: 1)
    let idsFlat = ids.ndim == 3 ? ids[0] : ids
    let rawPE = posEmbed(idsFlat)
    let pe = rawPE  // Both Python and Swift use R(+θ) convention — no negation needed

    // Double-stream blocks
    for i in 0..<doubleBlocks.count {
      let imgMod = getModulations(modVectors, blockType: "double_img", idx: i)
        as! (ChromaModulation, ChromaModulation)
      let txtMod = getModulations(modVectors, blockType: "double_txt", idx: i)
        as! (ChromaModulation, ChromaModulation)
      (imgH, txtH) = doubleBlocks[i](img: imgH, txt: txtH, imgMod: imgMod, txtMod: txtMod, pe: pe)
    }

    // Merge streams for single-stream blocks
    var unified = MLX.concatenated([txtH, imgH], axis: 1)
    let txtLen = txtH.dim(1)

    for i in 0..<singleBlocks.count {
      let mod = getModulations(modVectors, blockType: "single", idx: i) as! ChromaModulation
      unified = singleBlocks[i](unified, mod: mod, pe: pe)
    }

    // Slice off text tokens, keep only image
    let imgOut = unified[0..., txtLen..., 0...]

    // Final layer with modulation
    let finalMod = getModulations(modVectors, blockType: "final") as! (MLXArray, MLXArray)
    return finalLayer(imgOut, mod: finalMod)
  }
}
