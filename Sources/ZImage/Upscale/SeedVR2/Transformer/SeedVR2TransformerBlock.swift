import Foundation
import MLX
import MLXFast
import MLXNN

/// A single transformer block in the SeedVR2 architecture.
///
/// Each block processes both video and text streams through attention and MLP
/// with adaptive layer normalization.
public final class SeedVR2TransformerBlock: Module {

  public let sharedWeights: Bool
  public let isLastLayer: Bool
  public let normEps: Float

  @ModuleInfo(key: "attn") var attn: SeedVR2MMAttention
  @ModuleInfo(key: "mlp") var mlp: SeedVR2MMSwiGLU
  @ModuleInfo(key: "ada") var ada: SeedVR2AdaModulation

  public init(
    vidDim: Int = 2560,
    txtDim: Int = 2560,
    heads: Int = 20,
    headDim: Int = 128,
    expandRatio: Int = 4,
    normEps: Float = 1e-5,
    qkBias: Bool = false,
    ropeDim: Int = 128,
    sharedWeights: Bool = false,
    isLastLayer: Bool = false,
    window: (Int, Int, Int) = (4, 3, 3),
    shift: Bool = false,
    mlpType: SeedVR2MLPType = .swiglu
  ) {
    self.sharedWeights = sharedWeights
    self.isLastLayer = isLastLayer
    self.normEps = normEps

    self._attn.wrappedValue = SeedVR2MMAttention(
      vidDim: vidDim, txtDim: txtDim, heads: heads, headDim: headDim,
      qkBias: qkBias, qkNormEps: normEps, ropeDim: ropeDim,
      sharedWeights: sharedWeights, window: window, shift: shift
    )

    self._mlp.wrappedValue = SeedVR2MMSwiGLU(
      vidDim: vidDim, txtDim: txtDim, expandRatio: expandRatio,
      sharedWeights: sharedWeights, isLastLayer: isLastLayer,
      mlpType: mlpType
    )

    self._ada.wrappedValue = SeedVR2AdaModulation(
      dim: vidDim, sharedWeights: sharedWeights, isLastLayer: isLastLayer
    )

    super.init()
  }

  public func callAsFunction(
    vid: MLXArray,
    txt: MLXArray,
    emb: MLXArray,
    vidShape: MLXArray,
    txtShape: MLXArray
  ) -> (MLXArray, MLXArray) {
    var vidStream = vid
    var txtStream = txt

    // ---- Attention ----

    // Pre-norm (weight-free RMSNorm)
    var vidAttn = SeedVR2RMSNorm.apply(vidStream, eps: normEps)
    var txtAttn = SeedVR2RMSNorm.apply(txtStream, eps: normEps)

    // In-modulation (shift + scale)
    vidAttn = ada.modulateVid(vidAttn, emb: emb, layer: "attn", mode: "in")
    txtAttn = ada.modulateTxt(txtAttn, emb: emb, layer: "attn", mode: "in")

    // Multi-modal attention
    let attnResult = attn(vidAttn, txtAttn, vidShape, txtShape)
    vidAttn = attnResult.0
    txtAttn = attnResult.1

    // Out-modulation (gate)
    vidAttn = ada.modulateVid(vidAttn, emb: emb, layer: "attn", mode: "out")
    txtAttn = ada.modulateTxt(txtAttn, emb: emb, layer: "attn", mode: "out")

    // Residual
    vidStream = vidStream + vidAttn
    if !isLastLayer {
      txtStream = txtStream + txtAttn
    }

    // ---- MLP ----

    // Pre-norm
    var vidMlp = SeedVR2RMSNorm.apply(vidStream, eps: normEps)
    var txtMlp: MLXArray
    if isLastLayer {
      txtMlp = txtStream
    } else {
      txtMlp = SeedVR2RMSNorm.apply(txtStream, eps: normEps)
    }

    // In-modulation
    vidMlp = ada.modulateVid(vidMlp, emb: emb, layer: "mlp", mode: "in")
    txtMlp = ada.modulateTxt(txtMlp, emb: emb, layer: "mlp", mode: "in")

    // Feed-forward MLP
    let mlpResult = mlp(vidMlp, txtMlp)
    vidMlp = mlpResult.0
    txtMlp = mlpResult.1

    // Out-modulation
    vidMlp = ada.modulateVid(vidMlp, emb: emb, layer: "mlp", mode: "out")
    txtMlp = ada.modulateTxt(txtMlp, emb: emb, layer: "mlp", mode: "out")

    // Residual
    vidStream = vidStream + vidMlp
    if !isLastLayer {
      txtStream = txtStream + txtMlp
    }

    return (vidStream, txtStream)
  }
}
