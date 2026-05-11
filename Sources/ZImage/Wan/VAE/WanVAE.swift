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

  public func encode(_ images: MLXArray) -> MLXArray {
    var x = images
    if x.ndim == 4 {
      x = x.expandedDimensions(axis: 2)
    }

    var h = encoder(x)
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
    let latInvStd = MLXArray(Self.latentStd.map { 1.0 / $0 }).reshaped(1, Self.zDim, 1, 1, 1)
    z = z / latInvStd + latMean

    z = conv2(z)
    let decoded = decoder(z)
    return MLX.clip(decoded, min: -1.0, max: 1.0)
  }
}
