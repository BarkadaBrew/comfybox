// FiboVAEPatchify.swift — Patchify and unpatchify operations for the Wan 2.2 VAE
// Ported from mflux: wan_2_2_vae.py (_patchify / _unpatchify)
//
// The Wan 2.2 VAE uses spatial patchification to reduce the input resolution
// before encoding, and unpatchification after decoding to restore it. With
// patch_size=2:
//
//   Patchify:   (B, 3, T, H, W) → (B, 3*2*2, T, H/2, W/2) = (B, 12, T, H/2, W/2)
//   Unpatchify: (B, 12, T, H/2, W/2) → (B, 3, T, H, W)
//
// This is a simple spatial pixel-shuffle/unshuffle — the patch pixels are
// folded into the channel dimension.

import MLX

/// Patchify and unpatchify operations for the Wan 2.2 VAE.
///
/// These are static utility functions used by ``FiboVAE`` during encode and decode.
/// The operations fold/unfold spatial patches into the channel dimension.
public enum FiboVAEPatchify {

  /// Folds spatial patches into the channel dimension.
  ///
  /// Equivalent to a pixel-unshuffle (space-to-depth) operation:
  /// - Splits each spatial dimension by `patchSize`
  /// - Moves the patch pixels into the channel dimension
  ///
  /// Example with patchSize=2:
  ///   (B, 3, T, 256, 256) → (B, 12, T, 128, 128)
  ///
  /// - Parameters:
  ///   - x: Input tensor of shape `(B, C, T, H, W)`.
  ///   - patchSize: Spatial patch size. Default `2`.
  /// - Returns: Patchified tensor of shape `(B, C*P*P, T, H/P, W/P)`.
  public static func patchify(_ x: MLXArray, patchSize: Int = 2) -> MLXArray {
    if patchSize == 1 { return x }

    let (batchSize, channels, frames, height, width) = (
      x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4)
    )

    // Reshape to separate patch pixels:
    // (B, C, T, H, W) → (B, C, T, H/P, P, W/P, P)
    var result = x.reshaped(
      batchSize, channels, frames,
      height / patchSize, patchSize,
      width / patchSize, patchSize
    )

    // Reorder to move patch pixels to channel dim:
    // (B, C, T, H/P, P_h, W/P, P_w) → (B, C, P_w, P_h, T, H/P, W/P)
    result = result.transposed(0, 1, 6, 4, 2, 3, 5)

    // Collapse channels and patch pixels:
    // (B, C, P_w, P_h, T, H/P, W/P) → (B, C*P*P, T, H/P, W/P)
    result = result.reshaped(
      batchSize,
      channels * patchSize * patchSize,
      frames,
      height / patchSize,
      width / patchSize
    )

    return result
  }

  /// Unfolds the channel dimension back into spatial patches.
  ///
  /// Equivalent to a pixel-shuffle (depth-to-space) operation:
  /// - Extracts patch pixels from the channel dimension
  /// - Distributes them back to spatial dimensions
  ///
  /// Example with patchSize=2:
  ///   (B, 12, T, 128, 128) → (B, 3, T, 256, 256)
  ///
  /// - Parameters:
  ///   - x: Patchified tensor of shape `(B, C*P*P, T, H/P, W/P)`.
  ///   - patchSize: Spatial patch size. Default `2`.
  /// - Returns: Unpatchified tensor of shape `(B, C, T, H, W)`.
  public static func unpatchify(_ x: MLXArray, patchSize: Int = 2) -> MLXArray {
    if patchSize == 1 { return x }

    let (batchSize, cPatches, frames, height, width) = (
      x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4)
    )
    let channels = cPatches / (patchSize * patchSize)

    // Reshape to separate patch pixels:
    // (B, C*P*P, T, H, W) → (B, C, P_h, P_w, T, H, W)
    var result = x.reshaped(
      batchSize, channels, patchSize, patchSize, frames, height, width
    )

    // Reorder to distribute patches back to spatial dims:
    // (B, C, P_h, P_w, T, H, W) → (B, C, T, H, P_w, W, P_h)
    result = result.transposed(0, 1, 4, 5, 3, 6, 2)

    // Collapse spatial + patch dims:
    // (B, C, T, H, P_w, W, P_h) → (B, C, T, H*P, W*P)
    result = result.reshaped(
      batchSize, channels, frames,
      height * patchSize, width * patchSize
    )

    return result
  }
}
