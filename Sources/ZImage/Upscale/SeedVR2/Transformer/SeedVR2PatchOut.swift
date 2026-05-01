import Foundation
import MLX
import MLXNN

/// Reverse patchification: maps transformer tokens back to a 5D video tensor.
///
/// ## Architecture
///
/// ```
/// Input: (B, L, dim), vid_shape (B, 3) = [T', H', W']
///   → Linear(dim, out_channels * pt * ph * pw)
///   → reshape to (B, T', H', W', out_channels, pt, ph, pw)
///   → transpose to (B, out_channels, T'*pt, H'*ph, W'*pw)
///   → output (B, out_channels, T, H, W)
/// ```
///
/// With default parameters (out_channels=16, patch_size=(1,2,2), dim=2560):
///   - Output features per patch: 16 * 1 * 2 * 2 = 64
///   - Linear: 2560 -> 64
///
/// ## Weight Key Paths
///
/// - `vid_out.proj.weight`, `vid_out.proj.bias`
public final class SeedVR2PatchOut: Module {

  /// Spatial/temporal patch size (pt, ph, pw).
  public let patchSize: (Int, Int, Int)

  /// Number of output channels per voxel.
  public let outChannels: Int

  /// Linear projection from model dim to flattened patch.
  @ModuleInfo(key: "proj") var proj: Linear

  /// Creates a patch-out module.
  ///
  /// - Parameters:
  ///   - outChannels: Number of output channels per voxel. Default `16`.
  ///   - patchSize: (temporal, height, width) patch size. Default `(1, 2, 2)`.
  ///   - dim: Input feature dimension. Default `2560`.
  public init(outChannels: Int = 16, patchSize: (Int, Int, Int) = (1, 2, 2), dim: Int = 2560) {
    self.outChannels = outChannels
    self.patchSize = patchSize
    let (pt, ph, pw) = patchSize
    self._proj.wrappedValue = Linear(dim, outChannels * pt * ph * pw)
    super.init()
  }

  /// Converts tokens back to a 5D video tensor.
  ///
  /// - Parameters:
  ///   - vid: Token tensor of shape `(B, L, dim)`.
  ///   - vidShape: Patch grid shape, `(B, 3)` containing `[T', H', W']`.
  /// - Returns: Tuple of (video, vidShape) where video is `(B, C, T, H, W)`.
  public func callAsFunction(_ vid: MLXArray, _ vidShape: MLXArray) -> (MLXArray, MLXArray) {
    let (pt, ph, pw) = patchSize

    // Linear projection: (B, L, dim) → (B, L, outChannels * pt * ph * pw)
    var x = proj(vid)

    let bSize = x.dim(0)
    let tPatches = Int(vidShape[0, 0].item(Int32.self))
    let hPatches = Int(vidShape[0, 1].item(Int32.self))
    let wPatches = Int(vidShape[0, 2].item(Int32.self))

    // (B, T', H', W', outChannels, pt, ph, pw)
    x = x.reshaped(bSize, tPatches, hPatches, wPatches, outChannels, pt, ph, pw)

    // → (B, outChannels, T', pt, H', ph, W', pw)
    x = x.transposed(0, 4, 1, 5, 2, 6, 3, 7)

    // → (B, outChannels, T'*pt, H'*ph, W'*pw)
    x = x.reshaped(bSize, outChannels, tPatches * pt, hPatches * ph, wPatches * pw)

    return (x, vidShape)
  }
}
