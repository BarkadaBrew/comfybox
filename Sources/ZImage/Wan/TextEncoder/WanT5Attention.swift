import Foundation
import MLX
import MLXNN

/// T5 multi-head self-attention for the Wan 2.2 UMT5 encoder.
///
/// Implements scaled dot-product attention with relative position bias.
/// All linear projections have no bias, matching the T5 architecture.
///
/// In the Wan 2.2 checkpoint, every layer has its own relative position bias,
/// so this module always includes a `WanT5RelativePositionBias`.
///
/// ## Forward Pass
///
/// ```
/// Q = qProj(x) → [B, seqLen, numHeads, headDim]
/// K = kProj(x) → [B, seqLen, numHeads, headDim]
/// V = vProj(x) → [B, seqLen, numHeads, headDim]
///
/// attnWeights = Q @ K^T / sqrt(headDim) + positionBias
/// attnWeights = softmax(attnWeights + maskBias)
/// output = attnWeights @ V
/// output = oProj(output)
/// ```
public final class WanT5Attention: Module {

  /// Query projection. Linear(hiddenSize, hiddenSize, bias: false).
  @ModuleInfo(key: "q") var qProj: Linear

  /// Key projection. Linear(hiddenSize, hiddenSize, bias: false).
  @ModuleInfo(key: "k") var kProj: Linear

  /// Value projection. Linear(hiddenSize, hiddenSize, bias: false).
  @ModuleInfo(key: "v") var vProj: Linear

  /// Output projection. Linear(hiddenSize, hiddenSize, bias: false).
  @ModuleInfo(key: "o") var oProj: Linear

  /// Number of attention heads.
  public let numHeads: Int

  /// Dimension per attention head.
  public let headDim: Int

  /// Attention scale factor: 1/sqrt(headDim).
  public let scale: Float

  /// Creates a T5 self-attention module.
  ///
  /// - Parameters:
  ///   - hiddenSize: Model hidden dimension. Default 4096.
  ///   - numHeads: Number of attention heads. Default 64.
  ///   - headDim: Per-head dimension. Default 64.
  public init(hiddenSize: Int = 4096, numHeads: Int = 64, headDim: Int = 64) {
    self.numHeads = numHeads
    self.headDim = headDim
    self.scale = 1.0 / Float(headDim).squareRoot()

    self._qProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
    self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
    self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
    self._oProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)

    super.init()
  }

  /// Applies self-attention with relative position bias.
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `[B, seqLen, hiddenSize]`.
  ///   - positionBias: Relative position bias of shape `[1, numHeads, seqLen, seqLen]`.
  ///   - mask: Optional attention mask of shape `[B, 1, 1, seqLen]` where 0 = attend, large negative = mask.
  /// - Returns: Output tensor of shape `[B, seqLen, hiddenSize]`.
  public func callAsFunction(_ x: MLXArray, positionBias: MLXArray, mask: MLXArray? = nil) -> MLXArray {
    let b = x.dim(0)
    let seqLen = x.dim(1)

    // Project Q, K, V: [B, seqLen, hiddenSize] -> [B, seqLen, numHeads, headDim]
    let q = qProj(x).reshaped(b, seqLen, numHeads, headDim)
    let k = kProj(x).reshaped(b, seqLen, numHeads, headDim)
    let v = vProj(x).reshaped(b, seqLen, numHeads, headDim)

    // Transpose to [B, numHeads, seqLen, headDim]
    let qT = q.transposed(0, 2, 1, 3)
    let kT = k.transposed(0, 2, 1, 3)
    let vT = v.transposed(0, 2, 1, 3)

    // Attention scores: [B, numHeads, Q, K]
    var scores = MLX.matmul(qT, kT.transposed(0, 1, 3, 2)) * MLXArray(scale)

    // Add position bias: [1, numHeads, Q, K]
    scores = scores + positionBias

    // Apply attention mask if provided
    if let mask = mask {
      scores = scores + mask
    }

    // Softmax
    let weights = MLX.softmax(scores, axis: -1)

    // Weighted sum: [B, numHeads, seqLen, headDim]
    let attended = MLX.matmul(weights, vT)

    // Transpose back and reshape: [B, seqLen, hiddenSize]
    let output = attended.transposed(0, 2, 1, 3).reshaped(b, seqLen, numHeads * headDim)

    return oProj(output)
  }
}
