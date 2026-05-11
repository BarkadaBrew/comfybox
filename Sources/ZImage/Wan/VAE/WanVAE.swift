import Foundation
import MLX
import MLXNN

/// Wan 2.1 Video VAE -- top-level variational autoencoder.
///
/// Encodes RGB video to a 16-channel latent space with 8x spatial and
/// 4x temporal compression, and decodes latents back to video.
///
/// No patchify/unpatchify -- encoder takes raw RGB (3ch), decoder outputs RGB (3ch).
public final class WanVAE: Module {

  // MARK: - Constants

  public static let zDim = 16
  public static let baseDim = 96
  public static let dimMult = [1, 2, 4, 4]
  public static let numResBlocks = 2
  public static let spatialScale = 8
  public static let temporalScale = 4

  public static let latentMean: [Float] = [
    -0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632, -0.1922, -0.9497,  0.2503, -0.2921,
  ]

  public static let latentStd: [Float] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
  ]

  // MARK: - Modules

  @ModuleInfo(key: "encoder") var encoder: WanEncoder3d
  @ModuleInfo(key: "conv1") var conv1: WanCausalConv3d
  @ModuleInfo(key: "conv2") var conv2: WanCausalConv3d
  @ModuleInfo(key: "decoder") var decoder: WanDecoder3d

  // MARK: - Init

  public override init() {
    self._encoder.wrappedValue = WanEncoder3d(
      dim: Self.baseDim,
      zDim: Self.zDim,
      dimMult: Self.dimMult,
      numResBlocks: Self.numResBlocks,
      temporalDownsample: [false, true, true]
    )
    self._conv1.wrappedValue = WanCausalConv3d(
      inChannels: Self.zDim * 2,
      outChannels: Self.zDim * 2,
      kernelSize: 1,
      padding: 0
    )
    self._conv2.wrappedValue = WanCausalConv3d(
      inChannels: Self.zDim,
      outChannels: Self.zDim,
      kernelSize: 1,
      padding: 0
    )
    self._decoder.wrappedValue = WanDecoder3d(
      dim: Self.baseDim,
      zDim: Self.zDim,
      dimMult: Self.dimMult,
      numResBlocks: Self.numResBlocks,
      temporalUpsample: [true, true, false]
    )
    super.init()
  }

  // MARK: - Encode / Decode

  /// Encodes a video tensor [B, 3, T, H, W] to latent space.
  ///
  /// Uses chunk-by-chunk processing to match Python's cached encoding:
  /// - Chunk 0: first 1 frame (cached, no temporal downsample)
  /// - Chunk 1: remaining 4 frames (uses cached data for temporal downsample)
  /// The outputs are concatenated along the temporal axis.
  ///
  /// This produces the correct latent temporal dimension: for T=5 input,
  /// the output has latentT = (T-1)/4 + 1 = 2.
  public func encode(_ images: MLXArray) -> MLXArray {
    var x = images
    if x.ndim == 4 {
      x = x.expandedDimensions(axis: 2)
    }

    let t = x.dim(2)

    // For a single frame, no temporal processing needed
    if t == 1 {
      var h = encoder(x)
      h = conv1(h)
      let mu = h[0..., ..<Self.zDim, 0..., 0..., 0...]
      let latMean = MLXArray(Self.latentMean).reshaped(1, Self.zDim, 1, 1, 1)
      let latInvStd = MLXArray(Self.latentStd.map { 1.0 / $0 }).reshaped(1, Self.zDim, 1, 1, 1)
      return (mu - latMean) * latInvStd
    }

    // Multi-frame: chunk-by-chunk encoding matching Python's approach.
    // Python splits as: chunk 0 = frame 0 (1 frame), chunk 1 = frames 1..4 (4 frames)
    // iter_ = 1 + (t - 1) // 4
    let numChunks = 1 + (t - 1) / 4
    let cacheLayerCount = encoder.countCacheLayers()
    let cache = WanEncoderCache(layerCount: cacheLayerCount)

    var encodedChunks: [MLXArray] = []

    for i in 0..<numChunks {
      cache.idx = 0  // Reset index for each chunk

      let chunk: MLXArray
      if i == 0 {
        chunk = x[0..., 0..., 0..<1, 0..., 0...]  // First frame [B, 3, 1, H, W]
      } else {
        let start = 1 + 4 * (i - 1)
        let end = min(1 + 4 * i, t)
        chunk = x[0..., 0..., start..<end, 0..., 0...]  // 4 frames [B, 3, 4, H, W]
      }

      let encoded = encoder.forward(chunk, cache: cache)
      encodedChunks.append(encoded)
    }

    // Concatenate all chunk outputs along temporal axis
    var h: MLXArray
    if encodedChunks.count == 1 {
      h = encodedChunks[0]
    } else {
      h = MLX.concatenated(encodedChunks, axis: 2)
    }

    h = conv1(h)

    let mu = h[0..., ..<Self.zDim, 0..., 0..., 0...]
    let latMean = MLXArray(Self.latentMean).reshaped(1, Self.zDim, 1, 1, 1)
    let latInvStd = MLXArray(Self.latentStd.map { 1.0 / $0 }).reshaped(1, Self.zDim, 1, 1, 1)
    let encoded = (mu - latMean) * latInvStd

    return encoded
  }

  public func decode(_ latents: MLXArray) -> MLXArray {
    var z = latents
    if z.ndim == 4 {
      z = z.expandedDimensions(axis: 2)
    }

    let latMean = MLXArray(Self.latentMean).reshaped(1, Self.zDim, 1, 1, 1)
    let latStd = MLXArray(Self.latentStd).reshaped(1, Self.zDim, 1, 1, 1)
    z = z * latStd + latMean

    z = conv2(z)
    let decoded = decoder(z)
    return MLX.clip(decoded, min: -1.0, max: 1.0)
  }
}
