import Foundation
import MLX
import MLXNN

/// Transformer block for the Wan 2.2 model with DiT-style modulation.
///
/// Contains self-attention, cross-attention, and FFN, each with adaptive
/// normalization controlled by time embeddings.
///
/// ## Modulation Flow
///
/// The modulation parameter `[1, 6, dim]` is broadcast with the per-position
/// time embedding `e [B, seqLen, 6, dim]` to produce 6 modulation vectors:
/// ```
/// e[0]=shift1, e[1]=scale1, e[2]=gate1 (self-attention)
/// e[3]=shift2, e[4]=scale2, e[5]=gate2 (FFN)
/// ```
///
/// ## Key Architectural Details
/// - norm1, norm2: WanLayerNorm WITHOUT learnable parameters (elementwise_affine=false)
/// - norm3: WanLayerNorm WITH weight+bias (elementwise_affine=true, when crossAttnNorm=true)
/// - FFN: Linear(dim,ffnDim) -> GELU(tanh) -> Linear(ffnDim,dim) [2 projections, NOT 3]
/// - All attention projections have bias
/// - QK-norm via WanRMSNorm
///
/// Weight keys per block:
/// ```
/// blocks.{i}.self_attn.*    (q/k/v/o weight+bias, norm_q/k weight)
/// blocks.{i}.cross_attn.*   (same as self_attn)
/// blocks.{i}.ffn.0.weight   [ffnDim, dim]
/// blocks.{i}.ffn.0.bias     [ffnDim]
/// blocks.{i}.ffn.2.weight   [dim, ffnDim]
/// blocks.{i}.ffn.2.bias     [dim]
/// blocks.{i}.norm3.weight   [dim]
/// blocks.{i}.norm3.bias     [dim]
/// blocks.{i}.modulation     [1, 6, dim]
/// ```
public final class WanTransformerBlock: Module {

  // MARK: - Components

  @ModuleInfo(key: "norm1") var norm1: WanLayerNorm
  @ModuleInfo(key: "self_attn") var selfAttn: WanSelfAttention
  @ModuleInfo(key: "norm3") var norm3: WanLayerNorm
  @ModuleInfo(key: "cross_attn") var crossAttn: WanCrossAttention
  @ModuleInfo(key: "norm2") var norm2: WanLayerNorm
  @ModuleInfo(key: "ffn") var ffn: WanFFN

  /// Per-block modulation parameter `[1, 6, dim]`.
  public var modulation: MLXArray

  public let dim: Int

  // MARK: - Init

  /// Creates a transformer block.
  ///
  /// - Parameters:
  ///   - dim: Model dimension.
  ///   - ffnDim: FFN intermediate dimension.
  ///   - numHeads: Number of attention heads.
  ///   - qkNorm: Enable QK normalization.
  ///   - crossAttnNorm: Enable cross-attention normalization (norm3 gets weight+bias).
  ///   - eps: Normalization epsilon.
  public init(
    dim: Int,
    ffnDim: Int,
    numHeads: Int,
    qkNorm: Bool = true,
    crossAttnNorm: Bool = true,
    eps: Float = 1e-6
  ) {
    self.dim = dim

    // norm1, norm2: no learnable params
    self._norm1.wrappedValue = WanLayerNorm(dim: dim, eps: eps, elementwiseAffine: false)
    self._norm2.wrappedValue = WanLayerNorm(dim: dim, eps: eps, elementwiseAffine: false)

    // norm3: has weight+bias when crossAttnNorm=true
    self._norm3.wrappedValue = WanLayerNorm(
      dim: dim, eps: eps, elementwiseAffine: crossAttnNorm
    )

    self._selfAttn.wrappedValue = WanSelfAttention(
      dim: dim, numHeads: numHeads, qkNorm: qkNorm, eps: eps
    )
    self._crossAttn.wrappedValue = WanCrossAttention(
      dim: dim, numHeads: numHeads, qkNorm: qkNorm, eps: eps
    )

    // FFN: Linear -> GELU(tanh) -> Linear
    self._ffn.wrappedValue = WanFFN(dim: dim, ffnDim: ffnDim)

    // Modulation: initialized as randn/sqrt(dim), matching Python
    self.modulation = MLXRandom.normal([1, 6, dim]) / MLXArray(Float(dim).squareRoot())

    super.init()
  }

  // MARK: - Forward

  /// Forward pass through the transformer block.
  ///
  /// - Parameters:
  ///   - x: Input tensor `[B, seqLen, dim]`.
  ///   - e: Per-position time embedding `[B, seqLen, 6, dim]` (float32).
  ///   - seqLens: Actual sequence lengths per sample.
  ///   - gridSizes: 3D grid dimensions per sample.
  ///   - freqs: Precomputed RoPE frequencies.
  ///   - context: Text embeddings `[B, textLen, dim]`.
  ///   - contextLens: Actual context lengths per sample.
  /// - Returns: Output tensor `[B, seqLen, dim]`.
  public func callAsFunction(
    _ x: MLXArray,
    e: MLXArray,
    seqLens: [Int],
    gridSizes: [[Int]],
    freqs: MLXArray,
    context: MLXArray,
    contextLens: [Int]?
  ) -> MLXArray {
    var h = x

    // Compute modulation vectors in float32
    // modulation: [1, 6, dim] -> [1, 1, 6, dim] broadcast with e: [B, seqLen, 6, dim]
    let modBroadcast = modulation.expandedDimensions(axis: 0)  // [1, 1, 6, dim]
    let eMod = (modBroadcast + e).asType(.float32)

    // Split into 6 modulation vectors along axis 2
    // Each: [B, seqLen, 1, dim] -> squeeze -> [B, seqLen, dim]
    let e0 = eMod[0..., 0..., 0...0, 0...].squeezed(axis: 2)  // shift1
    let e1 = eMod[0..., 0..., 1...1, 0...].squeezed(axis: 2)  // scale1
    let e2 = eMod[0..., 0..., 2...2, 0...].squeezed(axis: 2)  // gate1
    let e3 = eMod[0..., 0..., 3...3, 0...].squeezed(axis: 2)  // shift2
    let e4 = eMod[0..., 0..., 4...4, 0...].squeezed(axis: 2)  // scale2
    let e5 = eMod[0..., 0..., 5...5, 0...].squeezed(axis: 2)  // gate2

    // Self-attention with modulation
    let normed1 = norm1(h).asType(.float32) * (1 + e1) + e0
    let y = selfAttn(normed1, seqLens: seqLens, gridSizes: gridSizes, freqs: freqs)
    h = (h.asType(.float32) + y.asType(.float32) * e2).asType(x.dtype)

    // Cross-attention (no modulation on input, just norm3)
    h = h + crossAttn(norm3(h), context: context, contextLens: contextLens)

    // FFN with modulation
    let normed2 = norm2(h).asType(.float32) * (1 + e4) + e3
    let ffnOut = ffn(normed2)
    h = (h.asType(.float32) + ffnOut.asType(.float32) * e5).asType(x.dtype)

    return h
  }
}

/// Feed-forward network for the Wan 2.2 transformer.
///
/// Architecture: Linear(dim, ffnDim) -> GELU(approximate='tanh') -> Linear(ffnDim, dim)
///
/// Two projections only (NOT SiLU-gated with 3 projections).
///
/// Weight keys use Sequential indices:
/// ```
/// ffn.0.weight  [ffnDim, dim]
/// ffn.0.bias    [ffnDim]
/// ffn.2.weight  [dim, ffnDim]
/// ffn.2.bias    [dim]
/// ```
/// Index 1 is GELU (no parameters).
public final class WanFFN: Module {

  /// Sequential layers array for index-based weight key mapping.
  /// Index 0: Linear(dim, ffnDim)
  /// Index 1: GELUPlaceholder (no params)
  /// Index 2: Linear(ffnDim, dim)
  public let layers: [Module]

  public init(dim: Int, ffnDim: Int) {
    self.layers = [
      Linear(dim, ffnDim),        // index 0
      GELUPlaceholder(),          // index 1 (GELU, no params)
      Linear(ffnDim, dim),        // index 2
    ]
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = (layers[0] as! Linear)(x)
    h = geluApprox(h)
    h = (layers[2] as! Linear)(h)
    return h
  }
}

/// Placeholder for GELU in Sequential index mapping (no learnable parameters).
public final class GELUPlaceholder: Module {
  public override init() {
    super.init()
  }
}

/// GELU activation with tanh approximation.
///
/// Matches PyTorch `nn.GELU(approximate='tanh')`:
/// ```
/// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// ```
public func geluApprox(_ x: MLXArray) -> MLXArray {
  let coeff = Float(0.7978845608028654)  // sqrt(2/pi)
  let inner = coeff * (x + 0.044715 * x * x * x)
  return 0.5 * x * (1 + MLX.tanh(inner))
}

// Import for modulation init
import MLXRandom
