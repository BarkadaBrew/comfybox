import Foundation
import MLX
import MLXFast
import MLXNN

/// Root mean square layer normalization utilities for SeedVR2.
///
/// SeedVR2 uses RMSNorm in two ways:
///
/// 1. **As a learnable module** -- via MLXNN's built-in RMSNorm (used in attention
///    qk-norms and the top-level transformer norms). These are stored with a learnable
///    weight vector and map naturally to the checkpoint key paths.
///
/// 2. **As a weight-free operation** -- the transformer block applies inline RMSNorm
///    using a ones vector (no learned parameters). This is the static helper below.
///
/// For learnable norms, use MLXNN's RMSNorm directly (matching existing ZImage style):
///
///     @ModuleInfo(key: "norm_q_vid") var normQVid: RMSNorm
///
/// For the weight-free norm used inside transformer block forward passes, use:
///
///     SeedVR2RMSNorm.apply(x, eps: normEps)
///
public enum SeedVR2RMSNorm {

  /// Applies weight-free RMS normalization using a unit weight vector.
  ///
  /// This matches the Python SeedVR2 transformer block's inline norm:
  ///
  ///     mx.fast.rms_norm(x, mx.ones(x.shape[-1]), eps)
  ///
  /// - Parameters:
  ///   - x: Input array of arbitrary shape.
  ///   - eps: Numerical stability constant. Default 1e-5.
  /// - Returns: The RMS-normalized array with the same shape and dtype.
  public static func apply(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let dim = x.dim(-1)
    let ones = MLXArray.ones([dim])
    return MLXFast.rmsNorm(x, weight: ones, eps: eps)
  }
}
