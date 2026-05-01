import Foundation
import MLX
import MLXNN

/// Top-level 3D Video VAE for the SeedVR2 upscaler.
///
/// Provides encode and decode operations for mapping between RGB image/video
/// tensors and the latent space used by the SeedVR2 diffusion transformer.
///
/// ## Encode
///
/// Accepts an image `(B, 3, H, W)` or video `(B, 3, T, H, W)`. For single images,
/// a temporal dimension of 1 is inserted automatically. The encoder produces both
/// mean and log-variance channels; only the mean is used, scaled by ``scalingFactor``.
///
/// ```
/// Input (B, 3, H, W)
///   → (B, 3, 1, H, W)           // add temporal dim
///   → Encoder3D → (B, 32, 1, H/8, W/8)
///   → split first 16 channels   // take mean, discard logvar
///   → × scalingFactor
///   → Output (B, 16, 1, H/8, W/8)
/// ```
///
/// ## Decode
///
/// Accepts latent `(B, 16, T, H/8, W/8)`. Divides by ``scalingFactor`` then
/// passes through the decoder to reconstruct RGB.
///
/// ```
/// Input (B, 16, 1, H/8, W/8)
///   → ÷ scalingFactor
///   → Decoder3D
///   → Output (B, 3, 1, H, W)
/// ```
///
/// ## Spatial Scale
///
/// The VAE provides an 8x spatial reduction: an H x W input produces H/8 x W/8
/// latents. This is exposed via ``spatialScale`` for use by the diffusion pipeline.
public final class SeedVR2VAE: Module {

  /// Scaling factor applied to latents after encoding.
  /// Normalizes the latent distribution for the downstream diffusion model.
  public let scalingFactor: Float = 0.9152

  /// Spatial downsampling factor (8x for the default architecture).
  public let spatialScale: Int = 8

  /// Number of latent channels (mean only, not including log-variance).
  public let latentChannels: Int

  /// The 3D encoder.
  @ModuleInfo(key: "encoder") var encoder: SeedVR2Encoder3D

  /// The 3D decoder.
  @ModuleInfo(key: "decoder") var decoder: SeedVR2Decoder3D

  /// Creates a SeedVR2 3D Video VAE.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels (RGB = 3). Default `3`.
  ///   - outChannels: Number of output channels (RGB = 3). Default `3`.
  ///   - latentChannels: Number of latent channels. Default `16`.
  ///   - blockOutChannels: Channel counts for each encoder/decoder stage.
  ///     Default `[128, 256, 512, 512]`.
  public init(
    inChannels: Int = 3,
    outChannels: Int = 3,
    latentChannels: Int = 16,
    blockOutChannels: [Int] = [128, 256, 512, 512]
  ) {
    self.latentChannels = latentChannels

    self._encoder.wrappedValue = SeedVR2Encoder3D(
      inChannels: inChannels,
      outChannels: latentChannels,
      blockOutChannels: blockOutChannels,
      layersPerBlock: 2,
      temporalDownBlocks: 2
    )

    self._decoder.wrappedValue = SeedVR2Decoder3D(
      inChannels: latentChannels,
      outChannels: outChannels,
      blockOutChannels: blockOutChannels,
      layersPerBlock: 3,
      temporalUpBlocks: 2
    )

    super.init()
  }

  /// Encodes an image or video into the scaled latent space.
  ///
  /// - Parameter x: Input tensor of shape `(B, 3, H, W)` or `(B, 3, T, H, W)`.
  /// - Returns: Scaled latent tensor of shape `(B, latentChannels, T_out, H/8, W/8)`.
  public func encode(_ x: MLXArray) -> MLXArray {
    // Add temporal dimension if input is a single image (4D → 5D).
    var input = x
    if input.ndim == 4 {
      input = input.expandedDimensions(axis: 2)
    }

    // Encode → (B, 2*latentChannels, T_out, H/8, W/8)
    let encoded = encoder(input)

    // Split into mean and logvar, keep only the mean.
    let parts = split(encoded, parts: 2, axis: 1)
    let mean = parts[0]

    // Scale the latents.
    return mean * scalingFactor
  }

  /// Decodes a latent tensor back to an image or video.
  ///
  /// - Parameter z: Latent tensor of shape `(B, latentChannels, T, H/8, W/8)`
  ///   or `(B, latentChannels, H/8, W/8)`.
  /// - Returns: Decoded tensor of shape `(B, 3, T, H, W)`.
  public func decode(_ z: MLXArray) -> MLXArray {
    // Add temporal dimension if input is 4D.
    var latent = z
    if latent.ndim == 4 {
      latent = latent.expandedDimensions(axis: 2)
    }

    // Unscale the latents.
    latent = latent / scalingFactor

    // Decode → (B, 3, T, H, W)
    let decoded = decoder(latent)
    return decoded
  }


  // MARK: - Tiled Encoding

  /// Pixel-space tiling parameters for tiled encoding.
  private static let encodeTileSize = 512     // pixels per tile side
  private static let encodeTileOverlap = 64   // pixels of overlap between tiles

  /// Encodes an image tensor using tiled VAE encoding to prevent OOM at large
  /// spatial sizes (2048px+).
  ///
  /// When the input image is small enough (H <= 512 and W <= 512 pixels), falls
  /// through to the standard `encode()` path. For larger images, the pixel tensor
  /// is split into overlapping 512x512 pixel tiles, each encoded independently,
  /// then blended together using cosine-ramped weights in latent space.
  ///
  /// This matches the tiled VAE encoding behavior of Python mflux: tiles are in
  /// pixel space (512px, stride 448px), but blending is done in latent space
  /// (8 latent units of overlap).
  ///
  /// - Parameter x: Input tensor of shape `(B, 3, 1, H, W)` or `(B, 3, H, W)`.
  /// - Returns: Encoded latent tensor of shape `(B, 16, 1, H/8, W/8)`.
  public func tiledEncode(_ x: MLXArray) -> MLXArray {
    // Ensure 5D: [B, C, T, H, W]
    var input = x
    if input.ndim == 4 {
      input = input.expandedDimensions(axis: 2)
    }

    let hPix = input.dim(3)
    let wPix = input.dim(4)

    // Bypass tiling for small images — no benefit, just overhead.
    if hPix <= Self.encodeTileSize && wPix <= Self.encodeTileSize {
      print("[ZImage] Tiled encode: bypass (image \(hPix)x\(wPix) fits in single tile)")
      return encode(input)
    }

    let stride = Self.encodeTileSize - Self.encodeTileOverlap  // 448 pixels
    let latentOverlap = Self.encodeTileOverlap / spatialScale  // 8 latent units

    // Output dimensions in latent space.
    let outC = latentChannels  // 16
    let outH = hPix / spatialScale
    let outW = wPix / spatialScale
    let latentTileH = Self.encodeTileSize / spatialScale  // 64
    let latentTileW = Self.encodeTileSize / spatialScale  // 64

    // Compute tile grid positions matching Python mflux: range(0, dim - tileSize + 1, stride)
    var hPositions = [Int]()
    var y = 0
    while y <= hPix - Self.encodeTileSize {
      hPositions.append(y)
      y += stride
    }
    if hPositions.isEmpty || hPositions.last! + Self.encodeTileSize < hPix {
      hPositions.append(max(hPix - Self.encodeTileSize, 0))
    }

    var wPositions = [Int]()
    var xPos = 0
    while xPos <= wPix - Self.encodeTileSize {
      wPositions.append(xPos)
      xPos += stride
    }
    if wPositions.isEmpty || wPositions.last! + Self.encodeTileSize < wPix {
      wPositions.append(max(wPix - Self.encodeTileSize, 0))
    }

    let hTiles = hPositions.count
    let wTiles = wPositions.count
    let totalTiles = hTiles * wTiles
    print("[ZImage] Tiled encode: \(totalTiles) tiles (\(hTiles)x\(wTiles)), image \(hPix)x\(wPix)")

    // CPU accumulation buffers in LATENT space: weighted sum and weight counts.
    let totalLatents = outC * outH * outW
    var accumulator = [Float](repeating: 0.0, count: totalLatents)
    var counts = [Float](repeating: 0.0, count: totalLatents)

    for (iy, yStart) in hPositions.enumerated() {
      for (ix, xStart) in wPositions.enumerated() {
        let yEnd = min(yStart + Self.encodeTileSize, hPix)
        let xEnd = min(xStart + Self.encodeTileSize, wPix)
        let tileHPix = yEnd - yStart
        let tileWPix = xEnd - xStart

        // Skip degenerate sliver tiles smaller than the overlap.
        if tileHPix < Self.encodeTileOverlap || tileWPix < Self.encodeTileOverlap {
          continue
        }

        let tileHLat = tileHPix / spatialScale
        let tileWLat = tileWPix / spatialScale

        // Extract the pixel tile: [1, 3, 1, tileHPix, tileWPix]
        let tile = input[0..., 0..., 0..., yStart..<yEnd, xStart..<xEnd]

        // Encode this tile through the existing VAE encoder.
        // encode() handles temporal dim and scalingFactor internally.
        let encodedTile = encode(tile)
        MLX.eval(encodedTile)

        // Convert encoded tile to CPU Float array.
        // encodedTile shape: [1, 16, 1, tileHLat, tileWLat]
        let tileF32 = encodedTile.asType(.float32)
        MLX.eval(tileF32)
        let tileArr = tileF32.asArray(Float.self)

        // Build the 2D blending weight mask in LATENT space.
        //
        // Height weight: 1.0 everywhere, but:
        //   - If not the first row: leading latentOverlap values ramp 0→1 (cosine)
        //   - If not the last row: trailing latentOverlap values ramp 1→0
        var hWeights = [Float](repeating: 1.0, count: tileHLat)
        if iy > 0 {
          let ramp = Self.cosRamp(latentOverlap)
          for i in 0..<min(latentOverlap, tileHLat) {
            hWeights[i] = ramp[i]
          }
        }
        if iy < hTiles - 1 {
          let ramp = Self.cosRamp(latentOverlap)
          for i in 0..<min(latentOverlap, tileHLat) {
            hWeights[tileHLat - 1 - i] = ramp[i]
          }
        }

        // Width weight: same logic as height.
        var wWeights = [Float](repeating: 1.0, count: tileWLat)
        if ix > 0 {
          let ramp = Self.cosRamp(latentOverlap)
          for i in 0..<min(latentOverlap, tileWLat) {
            wWeights[i] = ramp[i]
          }
        }
        if ix < wTiles - 1 {
          let ramp = Self.cosRamp(latentOverlap)
          for i in 0..<min(latentOverlap, tileWLat) {
            wWeights[tileWLat - 1 - i] = ramp[i]
          }
        }

        // Latent-space offsets for this tile in the output buffer.
        let lyStart = yStart / spatialScale
        let lxStart = xStart / spatialScale

        // Accumulate: output[region] += encoded_tile * weight2d
        //             counts[region] += weight2d
        //
        // tileArr layout: [1, 16, 1, tileHLat, tileWLat] flattened as BCTHW.
        for c in 0..<outC {
          for th in 0..<tileHLat {
            let w2dH = hWeights[th]
            for tw in 0..<tileWLat {
              let weight = w2dH * wWeights[tw]

              // Index into the tile flat array: [1, 16, 1, tileHLat, tileWLat]
              // Offset = c * (1 * tileHLat * tileWLat) + th * tileWLat + tw
              let tileIdx = c * tileHLat * tileWLat + th * tileWLat + tw

              // Index into the output buffer: [16, outH, outW] (no batch/temporal)
              let outIdx = c * (outH * outW) + (lyStart + th) * outW + (lxStart + tw)

              accumulator[outIdx] += tileArr[tileIdx] * weight
              counts[outIdx] += weight
            }
          }
        }
      }
    }

    // Normalize: output = accumulator / max(counts, 1e-6)
    for i in 0..<totalLatents {
      accumulator[i] /= max(counts[i], 1e-6)
    }

    // Convert back to MLXArray with shape [1, 16, 1, outH, outW].
    let resultShape = [1, outC, 1, outH, outW]
    let result = MLXArray(accumulator, resultShape).asType(x.dtype)
    return result
  }

  // MARK: - Tiled Decoding

  /// Tiling parameters matching Python mflux behavior.
  private static let tileSize = 64       // latent units per tile side
  private static let tileOverlap = 8     // latent units of overlap between tiles

  /// Generates a cosine ramp from 0.0 to 1.0 over `n` steps.
  ///
  /// Used to blend overlapping tile edges smoothly:
  /// - Leading edge of a non-first tile ramps from 0 → 1 (fade in)
  /// - Trailing edge of a non-last tile ramps from 1 → 0 (fade out)
  private static func cosRamp(_ n: Int) -> [Float] {
    (0..<n).map { i in
      let t = Float(i) / max(Float(n - 1), 1.0)
      return 0.5 - 0.5 * cos(t * .pi)
    }
  }

  /// Decodes a latent tensor using tiled VAE decoding to prevent intermediate
  /// value blowup at large spatial sizes.
  ///
  /// When the latent tensor is small enough (H <= 64 and W <= 64), falls through
  /// to the standard `decode()` path. For larger tensors, the latent is split into
  /// overlapping 64x64 tiles, each decoded independently, then blended together
  /// using cosine-ramped weights in pixel space.
  ///
  /// This matches the tiled VAE decoding behavior of Python mflux, which decodes
  /// in 64x64 latent tiles rather than the full tensor at once.
  ///
  /// - Parameter z: Latent tensor of shape `(B, 16, T, H, W)` or `(B, 16, H, W)`.
  /// - Returns: Decoded tensor of shape `(B, 3, T, H*8, W*8)`.
  public func tiledDecode(_ z: MLXArray) -> MLXArray {
    // Ensure 5D: [B, C, T, H, W]
    var latent = z
    if latent.ndim == 4 {
      latent = latent.expandedDimensions(axis: 2)
    }

    let hLat = latent.dim(3)
    let wLat = latent.dim(4)

    // Bypass tiling for small tensors — no benefit, just overhead.
    if hLat <= Self.tileSize && wLat <= Self.tileSize {
      print("[ZImage] Tiled decode: bypass (latent \(hLat)x\(wLat) fits in single tile)")
      return decode(latent)
    }

    let stride = Self.tileSize - Self.tileOverlap  // 56 latent units
    let pixelScale = spatialScale                   // 8
    let pixelOverlap = Self.tileOverlap * pixelScale  // 64 pixels

    // Output dimensions in pixel space.
    let outC = 3
    let outH = hLat * pixelScale
    let outW = wLat * pixelScale

    // Compute tile grid positions matching Python mflux: range(0, dim - tileSize + 1, stride)
    var hPositions = [Int]()
    var y = 0
    while y <= hLat - Self.tileSize {
      hPositions.append(y)
      y += stride
    }
    if hPositions.isEmpty || hPositions.last! + Self.tileSize < hLat {
      hPositions.append(max(hLat - Self.tileSize, 0))
    }

    var wPositions = [Int]()
    var x = 0
    while x <= wLat - Self.tileSize {
      wPositions.append(x)
      x += stride
    }
    if wPositions.isEmpty || wPositions.last! + Self.tileSize < wLat {
      wPositions.append(max(wLat - Self.tileSize, 0))
    }

    let hTiles = hPositions.count
    let wTiles = wPositions.count
    let totalTiles = hTiles * wTiles
    print("[ZImage] Tiled decode: \(totalTiles) tiles (\(hTiles)x\(wTiles)), latent \(hLat)x\(wLat)")

    // CPU accumulation buffers: weighted sum and weight counts.
    let totalPixels = outC * outH * outW
    var accumulator = [Float](repeating: 0.0, count: totalPixels)
    var counts = [Float](repeating: 0.0, count: totalPixels)

    for (iy, yStart) in hPositions.enumerated() {
      for (ix, xStart) in wPositions.enumerated() {
        let yEnd = min(yStart + Self.tileSize, hLat)
        let xEnd = min(xStart + Self.tileSize, wLat)
        let tileH = yEnd - yStart
        let tileW = xEnd - xStart

        // Skip degenerate sliver tiles smaller than the overlap.
        if tileH < Self.tileOverlap || tileW < Self.tileOverlap {
          continue
        }

        let pixTileH = tileH * pixelScale
        let pixTileW = tileW * pixelScale

        // Extract the latent tile: [1, 16, 1, tileH, tileW]
        let tile = latent[0..., 0..., 0..., yStart..<yEnd, xStart..<xEnd]

        // Decode this tile through the existing VAE decoder.
        // decode() handles scalingFactor division and temporal dim internally.
        let decodedTile = decode(tile)
        MLX.eval(decodedTile)

        // Convert decoded tile to CPU Float array.
        // decodedTile shape: [1, 3, 1, pixTileH, pixTileW]
        let tileF32 = decodedTile.asType(.float32)
        MLX.eval(tileF32)
        let tileArr = tileF32.asArray(Float.self)

        // Build the 2D blending weight mask for this tile.
        //
        // Height weight: 1.0 everywhere, but:
        //   - If not the first row: leading pixelOverlap values ramp 0→1 (cosine)
        //   - If not the last row: trailing pixelOverlap values ramp 1→0
        var hWeights = [Float](repeating: 1.0, count: pixTileH)
        if iy > 0 {
          let ramp = Self.cosRamp(pixelOverlap)
          for i in 0..<min(pixelOverlap, pixTileH) {
            hWeights[i] = ramp[i]
          }
        }
        if iy < hTiles - 1 {
          let ramp = Self.cosRamp(pixelOverlap)
          for i in 0..<min(pixelOverlap, pixTileH) {
            hWeights[pixTileH - 1 - i] = ramp[i]
          }
        }

        // Width weight: same logic as height.
        var wWeights = [Float](repeating: 1.0, count: pixTileW)
        if ix > 0 {
          let ramp = Self.cosRamp(pixelOverlap)
          for i in 0..<min(pixelOverlap, pixTileW) {
            wWeights[i] = ramp[i]
          }
        }
        if ix < wTiles - 1 {
          let ramp = Self.cosRamp(pixelOverlap)
          for i in 0..<min(pixelOverlap, pixTileW) {
            wWeights[pixTileW - 1 - i] = ramp[i]
          }
        }

        // Pixel-space offsets for this tile in the output buffer.
        let pyStart = yStart * pixelScale
        let pxStart = xStart * pixelScale

        // Accumulate: output[region] += decoded_tile * weight2d
        //             counts[region] += weight2d
        //
        // tileArr layout: [1, 3, 1, pixTileH, pixTileW] flattened as BCTHW.
        // We iterate over C, H, W and index into the flat arrays.
        for c in 0..<outC {
          for th in 0..<pixTileH {
            let w2dH = hWeights[th]
            for tw in 0..<pixTileW {
              let weight = w2dH * wWeights[tw]

              // Index into the tile's flat array: [1, 3, 1, pixTileH, pixTileW]
              // Offset = c * (1 * pixTileH * pixTileW) + th * pixTileW + tw
              let tileIdx = c * pixTileH * pixTileW + th * pixTileW + tw

              // Index into the output buffer: [3, outH, outW] (no batch/temporal)
              let outIdx = c * (outH * outW) + (pyStart + th) * outW + (pxStart + tw)

              accumulator[outIdx] += tileArr[tileIdx] * weight
              counts[outIdx] += weight
            }
          }
        }
      }
    }

    // Normalize: output = accumulator / max(counts, 1e-6)
    for i in 0..<totalPixels {
      accumulator[i] /= max(counts[i], 1e-6)
    }

    // Convert back to MLXArray with shape [1, 3, 1, outH, outW].
    let resultShape = [1, outC, 1, outH, outW]
    let result = MLXArray(accumulator, resultShape).asType(z.dtype)
    return result
  }
}
