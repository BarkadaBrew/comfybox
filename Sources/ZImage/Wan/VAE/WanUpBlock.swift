import Foundation
import MLX
import MLXNN

/// Residual up block for the Wan 2.2 VAE decoder.
///
/// Contains a sequence of residual blocks, an optional spatial/temporal
/// upsampler on the main path, and an optional DupUp3D shortcut that
/// bypasses the residual blocks.
///
/// ## Architecture
///
/// ```
/// Input (B, C_in, T, H, W)
///   ├─ [optional] Shortcut: DupUp3D(C_in → C_out, upsample)
///   ├─ Main: ResBlock × (N+1) → [optional] Resample(upsample)
///   ├─ main + shortcut (if upFlag)
///   └─ Output (B, C_out, T_out, H_out, W_out)
/// ```
///
/// When `upFlag=false`, no upsampling or shortcut is applied.
public final class WanResidualUpBlock: Module {

  /// Residual blocks on the main path.
  @ModuleInfo(key: "resnets") var resnets: [WanResidualBlock]

  /// Optional spatial upsampler on the main path.
  @ModuleInfo(key: "upsampler") var upsampler: WanResample?

  /// Optional duplicate-upsample shortcut.
  @ModuleInfo(key: "avg_shortcut") var avgShortcut: WanDupUp3D?

  /// Whether upsampling is enabled.
  public let upFlag: Bool

  /// Creates a Wan residual up block.
  ///
  /// - Parameters:
  ///   - inDim: Input channel dimension.
  ///   - outDim: Output channel dimension.
  ///   - numResBlocks: Number of residual blocks (actual count is numResBlocks + 1).
  ///   - temporalUpsample: Whether to also upsample temporally.
  ///   - upFlag: If true, enables upsampling and shortcut.
  public init(
    inDim: Int,
    outDim: Int,
    numResBlocks: Int,
    temporalUpsample: Bool = false,
    upFlag: Bool = false
  ) {
    self.upFlag = upFlag

    // Build residual blocks (N + 1 blocks)
    var blocks: [WanResidualBlock] = []
    var currentDim = inDim
    for _ in 0..<(numResBlocks + 1) {
      blocks.append(WanResidualBlock(inDim: currentDim, outDim: outDim))
      currentDim = outDim
    }
    self._resnets.wrappedValue = blocks

    // Shortcut path (only if upsampling)
    if upFlag {
      self._avgShortcut.wrappedValue = WanDupUp3D(
        inChannels: inDim,
        outChannels: outDim,
        factorT: temporalUpsample ? 2 : 1,
        factorS: 2
      )

      let mode: WanResampleMode = temporalUpsample ? .upsample3d : .upsample2d
      self._upsampler.wrappedValue = WanResample(
        dim: outDim, mode: mode, upsampleOutDim: outDim
      )
    }

    super.init()
  }

  /// Applies the up block.
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `(B, C_in, T, H, W)`.
  ///   - firstChunk: Whether this is the first temporal chunk (affects DupUp3D trim).
  /// - Returns: Output tensor of shape `(B, C_out, T_out, H_out, W_out)`.
  public func callAsFunction(_ x: MLXArray, firstChunk: Bool = false) -> MLXArray {
    let xCopy = x

    // Main path
    var h = x
    for resnet in resnets {
      h = resnet(h)
    }
    if let us = upsampler {
      h = us(h)
    }

    // Shortcut path
    if let shortcut = avgShortcut {
      h = h + shortcut(xCopy, firstChunk: firstChunk)
    }

    return h
  }
}
