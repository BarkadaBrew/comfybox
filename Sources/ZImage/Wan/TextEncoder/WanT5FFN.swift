import Foundation
import MLX
import MLXNN

/// Gated FFN for the Wan 2.2 UMT5 encoder (T5DenseGatedActDense variant).
///
/// Uses gated GELU activation. The gate projection determines which features
/// to pass through, while the up projection computes feature values.
///
/// ## Architecture
///
/// ```
/// gate = gelu(wi0(x))   — gate projection with GELU
/// up   = wi1(x)         — up projection (no activation)
/// out  = wo(gate * up)  — down projection after element-wise multiply
/// ```
///
/// ## Weight Mapping (Wan naming)
///
/// The Wan checkpoint stores the gate linear as `ffn.gate.0.weight` (with a
/// numeric `.0.` segment). Because MLX-Swift's unflatten logic treats numeric
/// path components as array indices (incompatible with Module.update), the
/// loading code remaps `ffn.gate.0.weight` → `ffn.gate.weight` before
/// unflattening. This allows a plain Linear property keyed as `gate`.
///
/// | Wan Key | Remapped Key | Role | Shape |
/// |---------|-------------|------|-------|
/// | `ffn.gate.0.weight` | `ffn.gate.weight` | Gate (wi0) | [10240, 4096] |
/// | `ffn.fc1.weight` | `ffn.fc1.weight` | Up (wi1) | [10240, 4096] |
/// | `ffn.fc2.weight` | `ffn.fc2.weight` | Down (wo) | [4096, 10240] |
///
/// All projections have no bias.
public final class WanT5FFN: Module {

  /// Gate projection: Linear(hiddenSize, ffnHiddenSize, bias: false).
  /// Wan weight key: `ffn.gate.0.weight` (remapped to `ffn.gate.weight` at load time).
  @ModuleInfo(key: "gate") var gateProj: Linear

  /// Up projection: Linear(hiddenSize, ffnHiddenSize, bias: false).
  /// Wan weight key: `ffn.fc1.weight`
  @ModuleInfo(key: "fc1") var upProj: Linear

  /// Down projection: Linear(ffnHiddenSize, hiddenSize, bias: false).
  /// Wan weight key: `ffn.fc2.weight`
  @ModuleInfo(key: "fc2") var downProj: Linear

  /// Creates a gated FFN.
  ///
  /// - Parameters:
  ///   - hiddenSize: Model hidden dimension. Default 4096.
  ///   - ffnHiddenSize: FFN intermediate dimension. Default 10240.
  public init(hiddenSize: Int = 4096, ffnHiddenSize: Int = 10240) {
    self._gateProj.wrappedValue = Linear(hiddenSize, ffnHiddenSize, bias: false)
    self._upProj.wrappedValue = Linear(hiddenSize, ffnHiddenSize, bias: false)
    self._downProj.wrappedValue = Linear(ffnHiddenSize, hiddenSize, bias: false)
    super.init()
  }

  /// Applies gated FFN.
  ///
  /// - Parameter x: Input tensor of shape `[B, seqLen, hiddenSize]`.
  /// - Returns: Output tensor of shape `[B, seqLen, hiddenSize]`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let gate = gelu(gateProj(x))  // gelu(wi0(x))
    let up = upProj(x)            // wi1(x)
    return downProj(gate * up)
  }
}
