import Foundation
import MLX
import MLXFast
import MLXNN

/// Modulation parameters for a Chroma transformer block.
///
/// Each set contains shift, scale, and gate — all of shape `[B, 1, dim]`.
public struct ChromaModulation {
  public let shift: MLXArray
  public let scale: MLXArray
  public let gate: MLXArray

  /// Extract a modulation triplet from the Approximator output at a given offset.
  ///
  /// - Parameters:
  ///   - tensor: Full modulation tensor `[B, 344, dim]`
  ///   - offset: Starting index into dimension 1
  public static func fromOffset(_ tensor: MLXArray, offset: Int) -> ChromaModulation {
    ChromaModulation(
      shift: tensor[0..., offset..<(offset + 1), 0...],
      scale: tensor[0..., (offset + 1)..<(offset + 2), 0...],
      gate: tensor[0..., (offset + 2)..<(offset + 3), 0...]
    )
  }
}

/// Double-stream (joint attention) block for Chroma.
///
/// Processes image and text hidden states through parallel norm → modulate → attention → FFN paths.
/// Both streams share joint attention but have separate projections and feed-forward networks.
///
/// Weight key prefix: `double_blocks.N.`
/// Sub-keys: `img_norm1`, `img_attn`, `img_norm2`, `img_mlp`, `txt_norm1`, `txt_attn`, `txt_norm2`, `txt_mlp`
public final class ChromaDoubleStreamBlock: Module {
  let numHeads: Int
  let hiddenSize: Int
  let headDim: Int

  @ModuleInfo(key: "img_norm1") var imgNorm1: LayerNorm
  @ModuleInfo(key: "img_attn") var imgAttn: ChromaSelfAttention
  @ModuleInfo(key: "img_norm2") var imgNorm2: LayerNorm
  @ModuleInfo(key: "img_mlp") var imgMlp: Sequential

  @ModuleInfo(key: "txt_norm1") var txtNorm1: LayerNorm
  @ModuleInfo(key: "txt_attn") var txtAttn: ChromaSelfAttention
  @ModuleInfo(key: "txt_norm2") var txtNorm2: LayerNorm
  @ModuleInfo(key: "txt_mlp") var txtMlp: Sequential

  public init(hiddenSize: Int, numHeads: Int, mlpRatio: Float, qkvBias: Bool = true) {
    self.numHeads = numHeads
    self.hiddenSize = hiddenSize
    self.headDim = hiddenSize / numHeads
    let mlpHiddenDim = Int(Float(hiddenSize) * mlpRatio)

    self._imgNorm1.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
    self._imgAttn.wrappedValue = ChromaSelfAttention(dim: hiddenSize, numHeads: numHeads, qkvBias: qkvBias)
    self._imgNorm2.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
    self._imgMlp.wrappedValue = Sequential {
      Linear(hiddenSize, mlpHiddenDim, bias: true)
      GELU(approximation: .tanh)
      Linear(mlpHiddenDim, hiddenSize, bias: true)
    }

    self._txtNorm1.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
    self._txtAttn.wrappedValue = ChromaSelfAttention(dim: hiddenSize, numHeads: numHeads, qkvBias: qkvBias)
    self._txtNorm2.wrappedValue = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false)
    self._txtMlp.wrappedValue = Sequential {
      Linear(hiddenSize, mlpHiddenDim, bias: true)
      GELU(approximation: .tanh)
      Linear(mlpHiddenDim, hiddenSize, bias: true)
    }

    super.init()
  }

  /// Run double-stream block.
  ///
  /// - Parameters:
  ///   - img: Image hidden states `[B, imgSeq, dim]`
  ///   - txt: Text hidden states `[B, txtSeq, dim]`
  ///   - imgMod: Tuple of (mod1, mod2) modulations for image stream
  ///   - txtMod: Tuple of (mod1, mod2) modulations for text stream
  ///   - pe: Positional encoding (cos, sin) for RoPE
  /// - Returns: Updated (img, txt)
  public func callAsFunction(
    img: MLXArray,
    txt: MLXArray,
    imgMod: (ChromaModulation, ChromaModulation),
    txtMod: (ChromaModulation, ChromaModulation),
    pe: (MLXArray, MLXArray)
  ) -> (MLXArray, MLXArray) {
    let (imgMod1, imgMod2) = imgMod
    let (txtMod1, txtMod2) = txtMod

    let b = img.dim(0)
    let imgSeq = img.dim(1)
    let txtSeq = txt.dim(1)

    // Prepare image for attention
    var imgModulated = imgNorm1(img)
    imgModulated = (1 + imgMod1.scale) * imgModulated + imgMod1.shift
    let (imgQ, imgK, imgV) = imgAttn.projectQKV(imgModulated)

    // Prepare text for attention
    var txtModulated = txtNorm1(txt)
    txtModulated = (1 + txtMod1.scale) * txtModulated + txtMod1.shift
    let (txtQ, txtK, txtV) = txtAttn.projectQKV(txtModulated)

    // Joint attention: concatenate [txt, img]
    let q = MLX.concatenated([txtQ, imgQ], axis: 2)
    let k = MLX.concatenated([txtK, imgK], axis: 2)
    let v = MLX.concatenated([txtV, imgV], axis: 2)

    // Apply RoPE
    let (cos, sin) = pe
    let (qRoped, kRoped) = Flux2AttentionUtils.applyRopeBSHD(query: q, key: k, cos: cos, sin: sin)

    // Scaled dot-product attention
    let scale = Float(1.0 / sqrt(Float(headDim)))
    let attnOut = MLXFast.scaledDotProductAttention(
      queries: qRoped, keys: kRoped, values: v,
      scale: scale, mask: nil
    )
    // [B, H, S, D] -> [B, S, H*D]
    let combined = attnOut.transposed(0, 2, 1, 3).reshaped(b, -1, hiddenSize)

    // Split back
    let txtAttnOut = combined[0..., ..<txtSeq, 0...]
    let imgAttnOut = combined[0..., txtSeq..., 0...]

    // Image residual + gating
    var imgOut = img + imgMod1.gate * imgAttn.projectOut(imgAttnOut)
    let imgNormed2 = (1 + imgMod2.scale) * imgNorm2(imgOut) + imgMod2.shift
    imgOut = imgOut + imgMod2.gate * imgMlp(imgNormed2)

    // Text residual + gating
    var txtOut = txt + txtMod1.gate * txtAttn.projectOut(txtAttnOut)
    let txtNormed2 = (1 + txtMod2.scale) * txtNorm2(txtOut) + txtMod2.shift
    txtOut = txtOut + txtMod2.gate * txtMlp(txtNormed2)

    return (imgOut, txtOut)
  }
}
