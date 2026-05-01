import Foundation
import MLX
import MLXNN

/// Patchifies 5D video tensors into a flat token sequence for the SeedVR2 transformer.
///
/// ## Architecture
///
/// ```
/// Input: (B, C=33, T, H, W)
///   → reshape to (B, C, T/pt, pt, H/ph, ph, W/pw, pw)
///   → transpose to (B, T/pt, H/ph, W/pw, C*pt*ph*pw)
///   → Linear(C*pt*ph*pw, dim)
///   → reshape to (B, T'*H'*W', dim)
///   → output tokens (B, L, dim) and vid_shape (B, 3) = [T', H', W']
/// ```
///
/// With default parameters (C=33, patch_size=(1,2,2), dim=2560):
///   - Patch volume: 1*2*2 = 4
///   - Input features per patch: 33 * 4 = 132
///   - Linear: 132 -> 2560
///
/// ## Weight Key Paths
///
/// - `vid_in.proj.weight`, `vid_in.proj.bias`
public final class SeedVR2PatchIn: Module {

  /// Spatial/temporal patch size (pt, ph, pw).
  public let patchSize: (Int, Int, Int)

  /// Linear projection from flattened patch to model dim.
  @ModuleInfo(key: "proj") var proj: Linear

  /// Creates a patch-in module.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels. Default `33`.
  ///   - patchSize: (temporal, height, width) patch size. Default `(1, 2, 2)`.
  ///   - dim: Output feature dimension. Default `2560`.
  public init(inChannels: Int = 33, patchSize: (Int, Int, Int) = (1, 2, 2), dim: Int = 2560) {
    self.patchSize = patchSize
    let (pt, ph, pw) = patchSize
    self._proj.wrappedValue = Linear(inChannels * pt * ph * pw, dim)
    super.init()
  }

  /// Patchifies a 5D video tensor into a token sequence.
  ///
  /// - Parameter vid: Input tensor of shape `(B, C, T, H, W)`.
  /// - Returns: Tuple of (tokens, vid_shape) where tokens is `(B, L, dim)` and
  ///   vid_shape is `(B, 3)` containing `[T/pt, H/ph, W/pw]` per batch element.
  public func callAsFunction(_ vid: MLXArray) -> (MLXArray, MLXArray) {
    let (pt, ph, pw) = patchSize
    let bSize = vid.dim(0)
    let channels = vid.dim(1)
    let tSize = vid.dim(2)
    let hSize = vid.dim(3)
    let wSize = vid.dim(4)

    let tPatches = tSize / pt
    let hPatches = hSize / ph
    let wPatches = wSize / pw

    // (B, C, T, H, W) → (B, C, T/pt, pt, H/ph, ph, W/pw, pw)
    var x = vid.reshaped(bSize, channels, tPatches, pt, hPatches, ph, wPatches, pw)

    // → (B, T/pt, H/ph, W/pw, pt, ph, pw, C)
    x = x.transposed(0, 2, 4, 6, 3, 5, 7, 1)

    // → (B, T/pt, H/ph, W/pw, pt*ph*pw*C)
    x = x.reshaped(bSize, tPatches, hPatches, wPatches, pt * ph * pw * channels)

    // Linear projection
    x = proj(x)

    // Flatten spatial dims: (B, T*H*W, dim)
    x = x.reshaped(bSize, -1, x.dim(-1))

    // vid_shape: (B, 3) = broadcast [tPatches, hPatches, wPatches]
    let shape = MLXArray([Int32(tPatches), Int32(hPatches), Int32(wPatches)])
    let vidShape = MLX.broadcast(shape.reshaped(1, 3), to: [bSize, 3])

    return (x, vidShape)
  }
}
