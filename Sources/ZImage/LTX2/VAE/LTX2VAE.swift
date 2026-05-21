import Foundation
import MLX
import MLXNN

/// Top-level LTX-2 Video VAE with encode and decode operations.
///
/// The LTX-2 VAE compresses RGB video to a 128-channel latent space with
/// 32x spatial compression and 8x temporal compression. Unlike SeedVR2 (which
/// uses 16 latent channels and GroupNorm), LTX-2 uses 128 latent channels and
/// PixelNorm throughout.
///
/// ## Encode
///
/// ```
/// Input (B, 3, F, H, W)
///   → patchify (4x spatial)
///   → Encoder3D → conv blocks + SpaceToDepth downsamples
///   → per-channel normalize
///   → Output (B, 128, F_lat, H/32, W/32)
/// ```
///
/// ## Decode
///
/// ```
/// Input (B, 128, F_lat, H/32, W/32)
///   → per-channel denormalize
///   → Decoder3D → res blocks + DepthToSpace upsamples (with timestep conditioning)
///   → unpatchify (4x spatial)
///   → Output (B, 3, F, H, W)
/// ```
///
/// ## Tiled Decode
///
/// For large videos, ``decodeTiled`` splits latents into overlapping spatial
/// tiles, decodes each independently, and blends with cosine-ramp weights.
/// Tile size, stride, and blending ramp are configurable.
public final class LTX2VAE: Module {

  /// Number of latent channels (128).
  public let latentChannels: Int

  /// Spatial compression factor (32x).
  public let spatialCompression: Int

  /// Temporal compression factor (8x).
  public let temporalCompression: Int

  /// The 3D encoder.
  @ModuleInfo(key: "encoder") var encoder: LTX2Encoder3D

  /// The 3D decoder.
  @ModuleInfo(key: "decoder") var decoder: LTX2Decoder3D

  /// Configuration.
  public let config: LTX2VideoVAEConfig

  /// Creates an LTX-2 Video VAE.
  ///
  /// - Parameter config: Video VAE configuration. Defaults to `.default`.
  public init(config: LTX2VideoVAEConfig = .default) {
    self.config = config
    self.latentChannels = config.latentChannels
    self.spatialCompression = config.spatialCompression
    self.temporalCompression = config.temporalCompression

    self._encoder.wrappedValue = LTX2Encoder3D(config: config)
    self._decoder.wrappedValue = LTX2Decoder3D(config: config)

    super.init()
  }

  /// Encodes a video to the latent space.
  ///
  /// - Parameter x: Input tensor of shape `(B, 3, F, H, W)` or `(B, 3, H, W)`.
  ///   F must satisfy `(F - 1) % 8 == 0` (i.e., 1, 9, 17, 25, ...).
  /// - Returns: Latent tensor of shape `(B, 128, F_lat, H/32, W/32)`.
  public func encode(_ x: MLXArray) -> MLXArray {
    var input = x
    // Add temporal dimension if single image
    if input.ndim == 4 {
      input = input.expandedDimensions(axis: 2)
    }
    return encoder(input)
  }

  /// Decodes latents back to video.
  ///
  /// - Parameters:
  ///   - z: Latent tensor of shape `(B, 128, F_lat, H_lat, W_lat)`.
  ///   - timestep: Optional timestep for conditioning.
  /// - Returns: Decoded video tensor of shape `(B, 3, F, H, W)`.
  public func decode(_ z: MLXArray, timestep: MLXArray? = nil) -> MLXArray {
    var latent = z
    if latent.ndim == 4 {
      latent = latent.expandedDimensions(axis: 2)
    }
    return decoder(latent, timestep: timestep)
  }

  /// Decodes latents using spatial tiling to reduce memory usage.
  ///
  /// Splits the latent tensor into overlapping tiles, decodes each independently,
  /// and blends with cosine-ramp weights. This pattern matches SeedVR2's tiled
  /// decode approach.
  ///
  /// - Parameters:
  ///   - z: Latent tensor of shape `(B, 128, F_lat, H_lat, W_lat)`.
  ///   - tileSize: Tile size in latent pixels. Default `16` (= 512 output pixels).
  ///   - tileStride: Tile stride in latent pixels. Default `14` (= 2 pixel overlap).
  ///   - rampSize: Cosine ramp blending width in latent pixels. Default `2`.
  ///   - timestep: Optional timestep for conditioning.
  /// - Returns: Decoded video tensor of shape `(B, 3, F, H, W)`.
  public func decodeTiled(
    _ z: MLXArray,
    tileSize: Int = 16,
    tileStride: Int = 14,
    rampSize: Int = 2,
    timestep: MLXArray? = nil
  ) -> MLXArray {
    var latent = z
    if latent.ndim == 4 {
      latent = latent.expandedDimensions(axis: 2)
    }

    let hLat = latent.dim(3)
    let wLat = latent.dim(4)

    // If the latent fits in a single tile, decode directly
    if hLat <= tileSize && wLat <= tileSize {
      return decoder(latent, timestep: timestep)
    }

    // Compute output dimensions
    let outH = hLat * spatialCompression
    let outW = wLat * spatialCompression
    let outF = 1 + (latent.dim(2) - 1) * temporalCompression

    // Allocate output and weight accumulators
    var output = MLXArray.zeros([latent.dim(0), 3, outF, outH, outW])
    let weights = MLXArray.zeros([latent.dim(0), 1, outF, outH, outW])

    // Generate tiles
    var hStart = 0
    while hStart < hLat {
      let hEnd = min(hStart + tileSize, hLat)
      var wStart = 0
      while wStart < wLat {
        let wEnd = min(wStart + tileSize, wLat)

        // Extract tile
        let tile = latent[0..., 0..., 0..., hStart..<hEnd, wStart..<wEnd]

        // Decode tile
        let decoded = decoder(tile, timestep: timestep)
        eval(decoded)

        // Compute output coordinates
        let outHStart = hStart * spatialCompression
        let outHEnd = outHStart + decoded.dim(3)
        let outWStart = wStart * spatialCompression
        let outWEnd = outWStart + decoded.dim(4)

        // Build blending mask with cosine ramps
        let tileH = decoded.dim(3)
        let tileW = decoded.dim(4)
        let rampPixels = rampSize * spatialCompression

        let hMask = Self.buildRamp1D(
          length: tileH,
          rampLeft: hStart > 0 ? rampPixels : 0,
          rampRight: hEnd < hLat ? rampPixels : 0
        )
        let wMask = Self.buildRamp1D(
          length: tileW,
          rampLeft: wStart > 0 ? rampPixels : 0,
          rampRight: wEnd < wLat ? rampPixels : 0
        )

        // Outer product: (H,) x (W,) -> (1, 1, 1, H, W)
        let blendMask = hMask.reshaped(1, 1, 1, -1, 1) * wMask.reshaped(1, 1, 1, 1, -1)

        // Accumulate
        let outSlice = output[0..., 0..., 0..., outHStart..<outHEnd, outWStart..<outWEnd]
        output[0..., 0..., 0..., outHStart..<outHEnd, outWStart..<outWEnd] =
          outSlice + decoded.asType(.float32) * blendMask
        let wSlice = weights[0..., 0..., 0..., outHStart..<outHEnd, outWStart..<outWEnd]
        weights[0..., 0..., 0..., outHStart..<outHEnd, outWStart..<outWEnd] =
          wSlice + blendMask
        eval(output, weights)

        wStart += tileStride
      }
      hStart += tileStride
    }

    // Normalize by weights
    let safeWeights = MLX.maximum(weights, MLXArray(Float(1e-8)))
    output = output / safeWeights

    return output.asType(latent.dtype)
  }

  /// Builds a 1D cosine ramp mask for blending.
  ///
  /// - Parameters:
  ///   - length: Total length of the mask.
  ///   - rampLeft: Fade-in length on the left.
  ///   - rampRight: Fade-out length on the right.
  /// - Returns: 1D array of shape `(length,)` with values in `[0, 1]`.
  private static func buildRamp1D(
    length: Int,
    rampLeft: Int,
    rampRight: Int
  ) -> MLXArray {
    var values = [Float](repeating: 1.0, count: length)

    // Left cosine ramp (0 → 1)
    if rampLeft > 0 {
      for i in 0..<min(rampLeft, length) {
        let t = Float(i) / Float(rampLeft)
        values[i] = 0.5 * (1.0 - cos(t * .pi))
      }
    }

    // Right cosine ramp (1 → 0)
    if rampRight > 0 {
      for i in 0..<min(rampRight, length) {
        let idx = length - rampRight + i
        if idx >= 0 && idx < length {
          let t = Float(rampRight - i) / Float(rampRight)
          values[idx] *= 0.5 * (1.0 - cos(t * .pi))
        }
      }
    }

    return MLX.clip(MLXArray(values), min: 0, max: 1)
  }
}
