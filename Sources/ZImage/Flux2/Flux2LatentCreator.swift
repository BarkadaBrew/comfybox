// Flux2LatentCreator.swift — Latent noise creation, patchify, pack for Flux 2
// Ported from mflux: flux2_latent_creator.py

import Foundation
import MLX
import MLXRandom

/// Creates and manipulates latent tensors for the Flux 2 diffusion pipeline.
///
/// Flux 2 uses a patchified latent representation where the spatial dimensions
/// are halved and the channels are quadrupled (2x2 patches folded into channels).
/// The packing step then flattens the spatial dims into a sequence.
///
/// ## Latent Layout Pipeline
///
/// ```
/// Random noise:      (B, 4*C, H/2, W/2)  — patchified at creation
///   -> pack:         (B, H/2 * W/2, 4*C)  — spatial flattened to sequence
///   -> transformer:  (B, seq, dim)         — denoising
///   -> unpack:       (B, 4*C, H/2, W/2)   — back to spatial
///   -> unpatchify:   (B, C, H, W)          — full resolution latent
///   -> VAE decode:   (B, 3, H*8, W*8)     — pixel image
/// ```
public enum Flux2LatentCreator {

  /// Default number of latent channels for Flux 2.
  public static let defaultNumLatentChannels: Int = 32

  /// Default VAE scale factor (spatial downsampling ratio).
  public static let defaultVAEScaleFactor: Int = 8

  // MARK: - Patchify / Unpatchify

  /// Patchify latents by folding 2x2 spatial patches into the channel dimension.
  ///
  /// Converts `(B, C, H, W)` -> `(B, 4*C, H/2, W/2)` by rearranging 2x2 spatial
  /// neighborhoods into the channel axis. Height and width must be even.
  ///
  /// - Parameter latents: Latent tensor in NCHW layout `(B, C, H, W)`.
  /// - Returns: Patchified tensor `(B, 4*C, H/2, W/2)`.
  public static func patchifyLatents(_ latents: MLXArray) -> MLXArray {
    var x = latents
    // Remove temporal dimension if present
    if x.ndim == 5 && x.shape[2] == 1 {
      x = x.squeezed(axis: 2)
    }
    precondition(x.ndim == 4, "Expected latents with ndim=4, got shape=\(x.shape)")

    let batchSize = x.shape[0]
    let numChannels = x.shape[1]
    let height = x.shape[2]
    let width = x.shape[3]

    // (B, C, H, W) -> (B, C, H/2, 2, W/2, 2)
    x = x.reshaped(batchSize, numChannels, height / 2, 2, width / 2, 2)
    // (B, C, H/2, 2, W/2, 2) -> (B, C, 2, 2, H/2, W/2)
    x = x.transposed(0, 1, 3, 5, 2, 4)
    // (B, C, 2, 2, H/2, W/2) -> (B, 4*C, H/2, W/2)
    x = x.reshaped(batchSize, numChannels * 4, height / 2, width / 2)

    return x
  }

  /// Unpatchify latents by unfolding channel patches back to spatial dimensions.
  ///
  /// Inverse of `patchifyLatents`. Converts `(B, 4*C, H, W)` -> `(B, C, 2*H, 2*W)`.
  ///
  /// - Parameter latents: Patchified latent tensor `(B, 4*C, H, W)`.
  /// - Returns: Unpatchified tensor `(B, C, 2*H, 2*W)`.
  public static func unpatchifyLatents(_ latents: MLXArray) -> MLXArray {
    let batchSize = latents.shape[0]
    let numChannels = latents.shape[1]
    let height = latents.shape[2]
    let width = latents.shape[3]

    // (B, 4C, H, W) -> (B, C, 2, 2, H, W)
    var x = latents.reshaped(batchSize, numChannels / 4, 2, 2, height, width)
    // (B, C, 2, 2, H, W) -> (B, C, H, 2, W, 2)
    x = x.transposed(0, 1, 4, 2, 5, 3)
    // (B, C, H, 2, W, 2) -> (B, C, 2H, 2W)
    x = x.reshaped(batchSize, numChannels / 4, height * 2, width * 2)

    return x
  }

  // MARK: - Pack / Unpack

  /// Pack latents by flattening spatial dimensions into a sequence.
  ///
  /// Converts `(B, C, H, W)` -> `(B, H*W, C)` for transformer processing.
  ///
  /// - Parameter latents: Spatial latent tensor `(B, C, H, W)`.
  /// - Returns: Packed sequence tensor `(B, H*W, C)`.
  public static func packLatents(_ latents: MLXArray) -> MLXArray {
    let batchSize = latents.shape[0]
    let numChannels = latents.shape[1]
    let height = latents.shape[2]
    let width = latents.shape[3]
    return latents.reshaped(batchSize, numChannels, height * width).transposed(0, 2, 1)
  }

  /// Unpack latents from sequence back to spatial layout.
  ///
  /// Converts `(B, seq, C)` -> `(B, C, H, W)`. If the input is already 4D,
  /// returns it unchanged.
  ///
  /// - Parameters:
  ///   - latents: Packed sequence tensor `(B, seq, C)` or already spatial `(B, C, H, W)`.
  ///   - height: Target image height in pixels.
  ///   - width: Target image width in pixels.
  ///   - vaeScaleFactor: VAE spatial downsampling factor (default 8).
  /// - Returns: Spatial latent tensor `(B, C, H, W)`.
  public static func unpackLatents(
    _ latents: MLXArray,
    height: Int,
    width: Int,
    vaeScaleFactor: Int = defaultVAEScaleFactor
  ) -> MLXArray {
    if latents.ndim == 4 {
      return latents
    }
    precondition(latents.ndim == 3, "Expected packed latents with ndim=3, got shape=\(latents.shape)")

    let batchSize = latents.shape[0]
    let seqLen = latents.shape[1]
    let channels = latents.shape[2]

    let latentHeight = height / (vaeScaleFactor * 2)
    let latentWidth = width / (vaeScaleFactor * 2)
    let expectedSeqLen = latentHeight * latentWidth

    precondition(
      expectedSeqLen == seqLen,
      "Packed latent seq_len mismatch: got \(seqLen), expected \(expectedSeqLen) "
      + "for height=\(height) width=\(width) (latent \(latentHeight)x\(latentWidth))"
    )

    return latents.reshaped(batchSize, latentHeight, latentWidth, channels)
      .transposed(0, 3, 1, 2)
  }

  // MARK: - Grid IDs

  /// Prepare spatial grid position IDs for the transformer's RoPE.
  ///
  /// Each spatial position gets a 4D coordinate `[t, h, w, layer]` where:
  ///   - `t`: temporal coordinate (always `tCoord` for images)
  ///   - `h`: height index
  ///   - `w`: width index
  ///   - `layer`: always 0 (reserved for future use)
  ///
  /// - Parameters:
  ///   - latents: Latent tensor `(B, C, H, W)` — used for shape only.
  ///   - tCoord: Temporal coordinate value (default 0 for static images).
  /// - Returns: Grid IDs tensor `(B, H*W, 4)`.
  public static func prepareGridIds(
    _ latents: MLXArray,
    tCoord: Int = 0
  ) -> MLXArray {
    let batchSize = latents.shape[0]
    let height = latents.shape[2]
    let width = latents.shape[3]

    let hIds = MLXArray(0..<height)
    let wIds = MLXArray(0..<width)

    // Create H x W grid
    let hGrid = MLX.broadcast(
      hIds.expandedDimensions(axis: 1),
      to: [height, width]
    )
    let wGrid = MLX.broadcast(
      wIds.expandedDimensions(axis: 0),
      to: [height, width]
    )

    let flatH = hGrid.reshaped(-1)
    let flatW = wGrid.reshaped(-1)
    let t = MLX.full([flatH.shape[0]], values: MLXArray(Int32(tCoord)), dtype: .int32)
    let layerIds = MLX.zeros(like: flatH).asType(.int32)

    // Stack [t, h, w, layer] -> (H*W, 4)
    var coords = MLX.stacked([t, flatH.asType(.int32), flatW.asType(.int32), layerIds], axis: 1)

    // Expand to batch: (1, H*W, 4) -> (B, H*W, 4)
    coords = coords.expandedDimensions(axis: 0)
    return MLX.broadcast(coords, to: [batchSize, coords.shape[1], coords.shape[2]])
  }

  // MARK: - Full Preparation

  /// Prepare initial noise latents for the diffusion process.
  ///
  /// Generates random noise, already patchified (2x2 patches folded into channels).
  /// The noise is drawn from a standard normal distribution with the given seed.
  ///
  /// - Parameters:
  ///   - seed: Random seed for reproducibility.
  ///   - height: Target image height in pixels.
  ///   - width: Target image width in pixels.
  ///   - batchSize: Number of images to generate.
  ///   - numLatentChannels: Number of base latent channels (default 32).
  ///   - vaeScaleFactor: VAE spatial downsampling factor (default 8).
  ///   - dtype: Data type for the latent tensor (default `.bfloat16`).
  /// - Returns: Tuple of `(latents, latentIds, latentHeight, latentWidth)` where
  ///   latents is `(B, 4*C, H', W')` in patchified NCHW layout.
  public static func prepareLatents(
    seed: UInt64,
    height: Int,
    width: Int,
    batchSize: Int = 1,
    numLatentChannels: Int = defaultNumLatentChannels,
    vaeScaleFactor: Int = defaultVAEScaleFactor,
    dtype: DType = .bfloat16
  ) -> (latents: MLXArray, latentIds: MLXArray, latentHeight: Int, latentWidth: Int) {
    // Align to patch grid: height/width must be multiples of vaeScaleFactor * 2
    let alignedHeight = 2 * (height / (vaeScaleFactor * 2))
    let alignedWidth = 2 * (width / (vaeScaleFactor * 2))
    let latentHeight = alignedHeight / 2
    let latentWidth = alignedWidth / 2

    // Generate noise in patchified shape: (B, 4*C, H/2, W/2)
    let shape = [batchSize, numLatentChannels * 4, latentHeight, latentWidth]
    let key = MLXRandom.key(seed)
    let latents = MLXRandom.normal(shape, key: key).asType(dtype)

    let latentIds = prepareGridIds(latents, tCoord: 0)

    return (latents, latentIds, latentHeight, latentWidth)
  }

  /// Prepare initial noise latents in packed (sequence) format.
  ///
  /// Same as `prepareLatents` but additionally packs the spatial dimensions
  /// into a sequence for direct consumption by the transformer.
  ///
  /// - Returns: Tuple of `(packedLatents, latentIds, latentHeight, latentWidth)` where
  ///   packedLatents is `(B, H'*W', 4*C)` in packed sequence layout.
  public static func preparePackedLatents(
    seed: UInt64,
    height: Int,
    width: Int,
    batchSize: Int = 1,
    numLatentChannels: Int = defaultNumLatentChannels,
    vaeScaleFactor: Int = defaultVAEScaleFactor,
    dtype: DType = .bfloat16
  ) -> (packedLatents: MLXArray, latentIds: MLXArray, latentHeight: Int, latentWidth: Int) {
    let (latents, latentIds, latentHeight, latentWidth) = prepareLatents(
      seed: seed,
      height: height,
      width: width,
      batchSize: batchSize,
      numLatentChannels: numLatentChannels,
      vaeScaleFactor: vaeScaleFactor,
      dtype: dtype
    )
    return (packLatents(latents), latentIds, latentHeight, latentWidth)
  }
}
