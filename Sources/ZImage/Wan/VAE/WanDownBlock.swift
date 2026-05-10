import Foundation
import MLX
import MLXNN

/// Down block for the Wan 2.2 VAE encoder.
///
/// Contains a sequence of residual blocks, a spatial/temporal downsampler on
/// the main path, and an average-pooling shortcut (AvgDown3D) that bypasses
/// the residual blocks.
///
/// ## Architecture
///
/// ```
/// Input (B, C_in, T, H, W)
///   ├─ Shortcut: AvgDown3D(C_in → C_out, downsample)
///   ├─ Main: ResBlock × N → [optional] Resample(downsample)
///   ├─ main + shortcut
///   └─ Output (B, C_out, T_out, H_out, W_out)
/// ```
///
/// When `isLast=true`, no spatial downsampling is applied.
public final class WanDownBlock: Module {

  /// Residual blocks on the main path.
  @ModuleInfo(key: "resnets") var resnets: [WanResidualBlock]

  /// Optional spatial downsampler on the main path.
  @ModuleInfo(key: "downsampler") var downsampler: WanResample?

  /// Average-pooling shortcut.
  @ModuleInfo(key: "avg_shortcut") var avgShortcut: WanAvgDown3D

  /// Creates a Wan down block.
  ///
  /// - Parameters:
  ///   - inDim: Input channel dimension.
  ///   - outDim: Output channel dimension.
  ///   - numResBlocks: Number of residual blocks.
  ///   - temporalDownsample: Whether to also downsample temporally.
  ///   - isLast: If true, no spatial downsampling is applied.
  public init(
    inDim: Int,
    outDim: Int,
    numResBlocks: Int,
    temporalDownsample: Bool = false,
    isLast: Bool = false
  ) {
    // Build residual blocks
    var blocks: [WanResidualBlock] = []
    var currentDim = inDim
    for _ in 0..<numResBlocks {
      blocks.append(WanResidualBlock(inDim: currentDim, outDim: outDim))
      currentDim = outDim
    }
    self._resnets.wrappedValue = blocks

    // Shortcut path
    self._avgShortcut.wrappedValue = WanAvgDown3D(
      inChannels: inDim,
      outChannels: outDim,
      factorT: temporalDownsample ? 2 : 1,
      factorS: isLast ? 1 : 2
    )

    // Main path downsampler
    if !isLast {
      let mode: WanResampleMode = temporalDownsample ? .downsample3d : .downsample2d
      self._downsampler.wrappedValue = WanResample(dim: outDim, mode: mode)
    }

    super.init()
  }

  /// Applies the down block.
  ///
  /// - Parameter x: Input tensor of shape `(B, C_in, T, H, W)`.
  /// - Returns: Output tensor of shape `(B, C_out, T_out, H_out, W_out)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let xCopy = x

    // Main path
    var h = x
    for resnet in resnets {
      h = resnet(h)
    }
    if let ds = downsampler {
      h = ds(h)
    }

    // Shortcut path + residual
    return h + avgShortcut(xCopy)
  }
}
