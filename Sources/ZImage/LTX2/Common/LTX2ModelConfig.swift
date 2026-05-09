import Foundation

/// Configuration for LTX-2 Video VAE model.
///
/// Holds all architecture parameters for the 3D Video VAE used in
/// the LTX-2 video generation pipeline. The encoder compresses
/// video to 128-channel latents with 32x spatial and 8x temporal
/// compression (via patchify + strided convolutions).
///
/// ## Default Architecture
///
/// ```
/// Encoder: conv_in -> 9 blocks (res_x / compress) -> norm -> conv_out
/// Decoder: conv_in -> 7 blocks (res_x / upsample)  -> norm -> conv_out
/// ```
///
/// Encoder blocks use SpaceToDepthDownsample with residual connections.
/// Decoder blocks use DepthToSpaceUpsample with residual connections.
public struct LTX2VideoVAEConfig: Equatable {

  /// Number of input channels (RGB = 3).
  public let inChannels: Int

  /// Number of latent channels.
  public let latentChannels: Int

  /// Spatial patch size used in patchify/unpatchify.
  public let patchSize: Int

  /// Whether timestep conditioning is enabled in the decoder.
  public let timestepConditioning: Bool

  /// Noise scale for decoder timestep conditioning.
  public let decodeNoiseScale: Float

  /// Default timestep for decoder.
  public let decodeTimestep: Float

  /// Timestep scale multiplier.
  public let timestepScaleMultiplier: Float

  /// Encoder block definitions.
  /// Each entry is a `(blockType, config)` pair.
  public let encoderBlocks: [EncoderBlockDef]

  /// Decoder block definitions.
  /// Each entry is a `(blockType, config)` pair.
  public let decoderBlocks: [DecoderBlockDef]

  /// Encoder block definition.
  public enum EncoderBlockDef: Equatable {
    /// Stack of residual blocks.
    case resX(numLayers: Int)
    /// SpaceToDepth downsample: spatial only.
    case compressSpaceRes(multiplier: Int)
    /// SpaceToDepth downsample: temporal only.
    case compressTimeRes(multiplier: Int)
    /// SpaceToDepth downsample: all dimensions.
    case compressAllRes(multiplier: Int)
  }

  /// Decoder block definition.
  public enum DecoderBlockDef: Equatable {
    /// Stack of residual blocks.
    case resX(numLayers: Int)
    /// DepthToSpace upsample: spatial only (1, 2, 2).
    case compressSpace(multiplier: Int)
    /// DepthToSpace upsample: temporal only (2, 1, 1).
    case compressTime(multiplier: Int)
    /// DepthToSpace upsample: all dimensions with residual.
    case compressAll(multiplier: Int, residual: Bool)
  }

  /// Default LTX-2 Video VAE configuration.
  public static let `default` = LTX2VideoVAEConfig(
    inChannels: 3,
    latentChannels: 128,
    patchSize: 4,
    timestepConditioning: true,
    decodeNoiseScale: 0.025,
    decodeTimestep: 0.05,
    timestepScaleMultiplier: 1000.0,
    encoderBlocks: [
      .resX(numLayers: 4),
      .compressSpaceRes(multiplier: 2),
      .resX(numLayers: 6),
      .compressTimeRes(multiplier: 2),
      .resX(numLayers: 6),
      .compressAllRes(multiplier: 2),
      .resX(numLayers: 2),
      .compressAllRes(multiplier: 2),
      .resX(numLayers: 2),
    ],
    decoderBlocks: [
      .resX(numLayers: 5),
      .compressAll(multiplier: 2, residual: true),
      .resX(numLayers: 5),
      .compressAll(multiplier: 2, residual: true),
      .resX(numLayers: 5),
      .compressAll(multiplier: 2, residual: true),
      .resX(numLayers: 5),
    ]
  )

  /// LTX-2 v2.3 Video VAE configuration.
  ///
  /// v2.3 uses separate spatial and temporal upsample blocks in the decoder
  /// instead of the default all-dimension blocks. Timestep conditioning is
  /// disabled in the VAE.
  public static let v23 = LTX2VideoVAEConfig(
    inChannels: 3,
    latentChannels: 128,
    patchSize: 4,
    timestepConditioning: false,
    decodeNoiseScale: 0.025,
    decodeTimestep: 0.05,
    timestepScaleMultiplier: 1000.0,
    encoderBlocks: [
      .resX(numLayers: 4),
      .compressSpaceRes(multiplier: 2),
      .resX(numLayers: 6),
      .compressTimeRes(multiplier: 2),
      .resX(numLayers: 4),
      .compressAllRes(multiplier: 2),
      .resX(numLayers: 2),
      .compressAllRes(multiplier: 1),
      .resX(numLayers: 2),
    ],
    decoderBlocks: [
      .resX(numLayers: 4),
      .compressSpace(multiplier: 2),
      .resX(numLayers: 6),
      .compressTime(multiplier: 2),
      .resX(numLayers: 4),
      .compressAll(multiplier: 1, residual: false),
      .resX(numLayers: 2),
      .compressAll(multiplier: 2, residual: false),
      .resX(numLayers: 2),
    ]
  )


  /// Spatial compression factor: patchSize (4) * 2^(number of spatial downsamples).
  /// Default: 4 * 8 = 32.
  public var spatialCompression: Int {
    var factor = patchSize
    for block in encoderBlocks {
      switch block {
      case .compressSpaceRes: factor *= 2
      case .compressAllRes: factor *= 2
      default: break
      }
    }
    return factor
  }

  /// Temporal compression factor: product of temporal strides.
  /// Default: 2 * 2 * 2 = 8.
  public var temporalCompression: Int {
    var factor = 1
    for block in encoderBlocks {
      switch block {
      case .compressTimeRes: factor *= 2
      case .compressAllRes: factor *= 2
      default: break
      }
    }
    return factor
  }
}
