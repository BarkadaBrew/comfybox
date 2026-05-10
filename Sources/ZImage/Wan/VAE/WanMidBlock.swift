import Foundation
import MLX
import MLXNN

/// Mid block for the Wan 2.2 VAE.
///
/// Sits between encoder/decoder blocks. Applies a ResidualBlock, then
/// alternating AttentionBlock + ResidualBlock pairs.
///
/// ## Architecture
///
/// ```
/// Input (B, C, T, H, W)
///   ├─ ResidualBlock
///   ├─ AttentionBlock → ResidualBlock  (repeated numLayers times)
///   └─ Output (B, C, T, H, W)
/// ```
public final class WanMidBlock: Module {

  /// Residual blocks (first one + one per attention layer).
  @ModuleInfo(key: "resnets") var resnets: [WanResidualBlock]

  /// Attention blocks.
  @ModuleInfo(key: "attentions") var attentions: [WanAttentionBlock]

  /// Creates a Wan mid block.
  ///
  /// - Parameters:
  ///   - dim: Channel dimension.
  ///   - numLayers: Number of attention + resblock pairs. Default `1`.
  public init(dim: Int, numLayers: Int = 1) {
    var resBlocks: [WanResidualBlock] = [WanResidualBlock(inDim: dim, outDim: dim)]
    var attnBlocks: [WanAttentionBlock] = []

    for _ in 0..<numLayers {
      attnBlocks.append(WanAttentionBlock(dim: dim))
      resBlocks.append(WanResidualBlock(inDim: dim, outDim: dim))
    }

    self._resnets.wrappedValue = resBlocks
    self._attentions.wrappedValue = attnBlocks

    super.init()
  }

  /// Applies the mid block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C, T, H, W)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = resnets[0](x)
    for i in 0..<attentions.count {
      h = attentions[i](h)
      h = resnets[i + 1](h)
    }
    return h
  }
}
