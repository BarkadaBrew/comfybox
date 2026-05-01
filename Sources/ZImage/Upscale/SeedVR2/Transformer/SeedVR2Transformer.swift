import Foundation
import MLX
import MLXFast
import MLXNN

/// Top-level NaDiT (Neighborhood-Aware DiT) transformer for SeedVR2 upscaling.
///
/// Supports both 3B and 7B model variants:
///
/// 3B: vid_out_norm + out_shift/out_scale applied after blocks, before vid_out.
/// 7B: No output norm/shift/scale. Blocks output goes directly to vid_out.
public final class SeedVR2Transformer: Module {

  public let vidDim: Int
  public let embDim: Int
  public let mmLayers: Int
  public let numLayers: Int
  public let hasOutputNorm: Bool

  // Input modules
  @ModuleInfo(key: "vid_in") var vidIn: SeedVR2PatchIn
  @ModuleInfo(key: "txt_in") var txtIn: Linear
  @ModuleInfo(key: "emb_in") var embIn: SeedVR2TimeEmbedding

  // Transformer blocks
  @ModuleInfo(key: "blocks") var blocks: [SeedVR2TransformerBlock]

  // Output modules (always present for weight loading; may not be used for 7B)
  @ModuleInfo(key: "vid_out_norm") var vidOutNorm: RMSNorm?

  @ParameterInfo(key: "out_shift") var outShift: MLXArray?
  @ParameterInfo(key: "out_scale") var outScale: MLXArray?

  @ModuleInfo(key: "vid_out") var vidOut: SeedVR2PatchOut

  /// Creates the SeedVR2 transformer from a model config.
  public convenience init(config: SeedVR2ModelConfig) {
    self.init(
      vidDim: config.vidDim,
      txtInDim: config.txtInDim,
      heads: config.heads,
      headDim: config.headDim,
      expandRatio: config.expandRatio,
      numLayers: config.numLayers,
      mmLayers: config.mmLayers,
      ropeDim: config.ropeDim,
      mlpType: config.mlpType,
      hasOutputNorm: config.hasOutputNorm,
      hasLastLayerFreeze: config.hasLastLayerFreeze
    )
  }

  /// Creates the SeedVR2 transformer with 3B defaults.
  public override convenience init() {
    self.init(config: .preset3B)
  }

  public init(
    vidInChannels: Int = 33,
    vidOutChannels: Int = 16,
    vidDim: Int = 2560,
    txtInDim: Int = 5120,
    txtDim: Int? = nil,
    embDim: Int? = nil,
    heads: Int = 20,
    headDim: Int = 128,
    expandRatio: Int = 4,
    normEps: Float = 1e-5,
    patchSize: (Int, Int, Int) = (1, 2, 2),
    numLayers: Int = 32,
    mmLayers: Int = 10,
    ropeDim: Int = 128,
    window: (Int, Int, Int) = (4, 3, 3),
    mlpType: SeedVR2MLPType = .swiglu,
    hasOutputNorm: Bool = true,
    hasLastLayerFreeze: Bool = true
  ) {
    let resolvedTxtDim = txtDim ?? vidDim
    let resolvedEmbDim = embDim ?? (6 * vidDim)
    self.vidDim = vidDim
    self.embDim = resolvedEmbDim
    self.mmLayers = mmLayers
    self.numLayers = numLayers
    self.hasOutputNorm = hasOutputNorm

    // Input modules
    self._vidIn.wrappedValue = SeedVR2PatchIn(
      inChannels: vidInChannels, patchSize: patchSize, dim: vidDim
    )
    self._txtIn.wrappedValue = Linear(txtInDim, resolvedTxtDim)
    self._embIn.wrappedValue = SeedVR2TimeEmbedding(
      sinusoidalDim: 256,
      hiddenDim: max(vidDim, resolvedTxtDim),
      outputDim: resolvedEmbDim
    )

    // Transformer blocks
    var blockList: [SeedVR2TransformerBlock] = []
    for i in 0 ..< numLayers {
      let shared = i >= mmLayers
      let isLast = hasLastLayerFreeze && (i == numLayers - 1)
      let shifted = i % 2 == 1

      blockList.append(SeedVR2TransformerBlock(
        vidDim: vidDim, txtDim: resolvedTxtDim,
        heads: heads, headDim: headDim, expandRatio: expandRatio,
        normEps: normEps, qkBias: false, ropeDim: ropeDim,
        sharedWeights: shared, isLastLayer: isLast,
        window: window, shift: shifted,
        mlpType: mlpType
      ))
    }
    self._blocks.wrappedValue = blockList

    // Output modules
    if hasOutputNorm {
      self._vidOutNorm.wrappedValue = RMSNorm(dimensions: vidDim, eps: normEps)
      self._outShift.wrappedValue = MLXArray.zeros([vidDim])
      self._outScale.wrappedValue = MLXArray.ones([vidDim])
    } else {
      self._vidOutNorm.wrappedValue = nil
      self._outShift.wrappedValue = nil
      self._outScale.wrappedValue = nil
    }

    self._vidOut.wrappedValue = SeedVR2PatchOut(
      outChannels: vidOutChannels, patchSize: patchSize, dim: vidDim
    )

    super.init()
  }

  /// Runs the full transformer forward pass.
  public func callAsFunction(
    vid: MLXArray,
    txt: MLXArray,
    timestep: MLXArray
  ) -> MLXArray {
    // Project text embeddings
    let txtProjected = txtIn(txt)
    let txtShape = MLXArray(
      [Int32](repeating: Int32(txtProjected.dim(1)), count: txtProjected.dim(0))
    ).reshaped(-1, 1)

    // Patchify video
    let (vidTokens, vidShape) = vidIn(vid)

    // Timestep embedding -> modulation parameters
    let embFlat = embIn(timestep)
    let emb = embFlat.reshaped(-1, vidDim, 2, 3)

    // Run through all transformer blocks
    var vidStream = vidTokens
    var txtStream = txtProjected
    for block in blocks {
      let result = block(vid: vidStream, txt: txtStream, emb: emb, vidShape: vidShape, txtShape: txtShape)
      vidStream = result.0
      txtStream = result.1
    }

    // Output head
    if hasOutputNorm {
      // 3B: RMSNorm + adaptive shift/scale
      vidStream = vidOutNorm!(vidStream)
      vidStream = applyOutAda(vidStream, emb: emb)
    }
    // 7B: no output norm, go straight to unpatchify

    // Un-patchify
    let (vidResult, _) = vidOut(vidStream, vidShape)
    return vidResult
  }

  /// Applies the final output adaptive modulation (3B only).
  private func applyOutAda(_ hidden: MLXArray, emb: MLXArray) -> MLXArray {
    let shiftA = emb[0..., 0..., 0, 0].expandedDimensions(axis: 1)
    let scaleA = emb[0..., 0..., 0, 1].expandedDimensions(axis: 1)
    return hidden * (outScale! + scaleA) + (outShift! + shiftA)
  }
}
