import Foundation
import MLX
import MLXNN

/// Pre-computed batch normalization statistics for the Flux2 VAE packed latent pipeline.
///
/// Unlike standard BatchNorm, this module does not learn gamma/beta parameters.
/// It stores only running mean and variance loaded from model weights, used to
/// un-normalize packed latents before decoding.
///
/// ## Weight Keys
///
/// - `running_mean`: Shape `(num_features,)` -- loaded from checkpoint
/// - `running_var`: Shape `(num_features,)` -- loaded from checkpoint
public final class Flux2BatchNormStats: Module {

  /// Running mean loaded from checkpoint. Shape `(num_features,)`.
  @ModuleInfo(key: "running_mean") var runningMean: MLXArray

  /// Running variance loaded from checkpoint. Shape `(num_features,)`.
  @ModuleInfo(key: "running_var") var runningVar: MLXArray

  /// Small constant for numerical stability.
  public let eps: Float

  /// Creates a batch norm statistics container.
  ///
  /// - Parameters:
  ///   - numFeatures: Number of channels/features.
  ///   - eps: Epsilon for numerical stability. Default `1e-4`.
  public init(numFeatures: Int, eps: Float = 1e-4) {
    self.eps = eps
    self._runningMean.wrappedValue = MLXArray.zeros([numFeatures])
    self._runningVar.wrappedValue = MLXArray.ones([numFeatures])
    super.init()
  }
}
