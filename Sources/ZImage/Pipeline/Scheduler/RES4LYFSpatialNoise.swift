// RES4LYFSpatialNoise.swift — RES4LYF's `fractal` and `pyramid` SDE noise
// generators, ported as OPT-IN alternatives to the gaussian noise stream.
//
// Faithful to `beta/noise_classes.py` at the pinned upstream commit
// `26036f647ca15d3048a193daf99a40cecfc3820d` (`FractalNoiseGenerator`,
// `PyramidNoiseGenerator`). The DEFAULT gaussian path is untouched: these are
// selected only when a caller asks for `noise_type: fractal|pyramid`, and
// `fractal` with `alpha == 0` short-circuits to the plain z-scored gaussian so
// it is byte-identical to the gaussian stream.
//
// Both generators operate on the 2D latent GRID `(1, C, latH, latW)`, but the
// Krea 2 SDE loop runs on the PATCHIFIED latent `(1, tokens, C·p·p)`. Each
// stream therefore round-trips: draw the gaussian in the token shape (the SAME
// draw the gaussian stream makes, for reproducibility) → unpatchify to the grid
// → transform → patchify back → z-score. `p`/`latH`/`latW` are derived from the
// sample's trailing axis and the plumbed-in token grid.

import Foundation
import MLX
import MLXRandom

// MARK: - Noise type selector

/// Which RES4LYF noise generator drives the SDE re-noise. `gaussian` is the
/// production default and byte-identical to the pre-existing stream; `fractal`
/// and `pyramid` are opt-in spatial alternatives.
public enum RES4LYFNoiseType: String, Sendable, CaseIterable {
  case gaussian
  case fractal
  case pyramid
}

extension RES4LYFNoiseLayout {
  /// The channel count the layout carries, when it names one. The spatial
  /// generators need it to unpatchify onto the `(1, C, latH, latW)` grid; a
  /// layout that carries no channel structure cannot drive them and falls back
  /// to gaussian at the injector.
  var channelCount: Int? {
    switch self {
    case .patchifiedTrailing(let channels): return channels
    case .channelsAtAxis1, .whole: return nil
    }
  }
}

// MARK: - Shared grid geometry

/// The grid geometry a spatial stream reconstructs from one patchified sample
/// plus the plumbed-in token grid.
///
/// `p = round(sqrt(trailing / channels))`, `latH = hTok·p`, `latW = wTok·p` —
/// exactly the inverse of ``Krea2Pipeline/patchify(_:patch:)``.
struct RES4LYFSpatialGeometry {
  let p: Int
  let latH: Int
  let latW: Int

  init(sample: MLXArray, hTok: Int, wTok: Int, channels: Int) {
    let trailing = sample.dim(sample.ndim - 1)
    self.p = Int((Double(trailing / channels)).squareRoot().rounded())
    self.latH = hTok * p
    self.latW = wTok * p
  }
}

// MARK: - Fractal

/// RES4LYF's `FractalNoiseGenerator` (`noise_classes.py:145`), on the Krea 2
/// patchified latent.
///
/// `out = ifft2(fft2(gaussian) · density).real`, `density = k / freq^(α·scale)`
/// with `density[0,0] = 0` and `freq` the radial FFT frequency magnitude on the
/// `(latH, latW)` grid. With `alpha == 0` the spectrum is flat, so the transform
/// is the identity up to floating error — upstream returns `noise / std(noise)`
/// which is then z-scored, and the flat-density product does the same — so the
/// port short-circuits to the plain z-scored gaussian and is byte-identical to
/// the gaussian stream for the same seed/shape.
///
/// The gaussian draw uses the SAME key operations as
/// ``RES4LYFGaussianNoiseStream`` (split into 2, advance on `keys[0]`, draw on
/// `keys[1]`), so `alpha == 0` reproduces the gaussian stream to the bit.
public final class RES4LYFFractalNoiseStream: RES4LYFNoiseStream {
  private var key: MLXArray
  private let layout: RES4LYFNoiseLayout
  private let hTok: Int
  private let wTok: Int
  private let channels: Int
  private let alpha: Double
  private let k: Double
  private let scale: Double

  public init(
    seed: UInt64, layout: RES4LYFNoiseLayout, hTok: Int, wTok: Int, channels: Int,
    alpha: Double, k: Double = 1.0, scale: Double = 0.1
  ) {
    self.key = MLXRandom.key(seed)
    self.layout = layout
    self.hTok = hTok
    self.wTok = wTok
    self.channels = channels
    self.alpha = alpha
    self.k = k
    self.scale = scale
  }

  public func next(like sample: MLXArray) -> MLXArray {
    // The gaussian draw, identical to RES4LYFGaussianNoiseStream.
    let keys = MLXRandom.split(key: key, into: 2)
    key = keys[0]
    let raw = MLXRandom.normal(sample.shape, dtype: .float32, key: keys[1])

    // alpha == 0 ⇒ flat spectrum ⇒ fractal ≡ gaussian. Byte-identical.
    guard alpha != 0 else {
      return RES4LYFNoiseNormalization.zscore(raw, layout: layout)
    }

    let geo = RES4LYFSpatialGeometry(sample: sample, hTok: hTok, wTok: wTok, channels: channels)
    let grid = Krea2Sampling.unpatchify(raw, patch: geo.p, h: hTok, w: wTok, c: channels)
    let transformed = Self.fractal(
      grid, latH: geo.latH, latW: geo.latW, alpha: alpha, k: k, scale: scale)
    let repatched = Krea2Sampling.patchify(transformed, patch: geo.p)
    return RES4LYFNoiseNormalization.zscore(repatched, layout: layout)
  }

  /// `torch.fft.fftfreq(n, 1/n)` — the integer cycle counts
  /// `[0, 1, …, ⌈n/2⌉−1, −⌊n/2⌋, …, −1]`.
  static func fftFreq(_ n: Int) -> [Double] {
    var out = [Double](repeating: 0, count: n)
    let half = (n + 1) / 2
    for i in 0..<n { out[i] = Double(i < half ? i : i - n) }
    return out
  }

  /// The spectral transform on `(1, C, latH, latW)`. `density` is built
  /// host-side on the `(latH, latW)` grid and broadcast over batch/channel.
  static func fractal(
    _ grid: MLXArray, latH: Int, latW: Int, alpha: Double, k: Double, scale: Double
  ) -> MLXArray {
    let fy = fftFreq(latH)
    let fx = fftFreq(latW)
    let exponent = alpha * scale
    var density = [Float](repeating: 0, count: latH * latW)
    for y in 0..<latH {
      let fy2 = fy[y] * fy[y]
      for x in 0..<latW {
        var freq = (fy2 + fx[x] * fx[x]).squareRoot()
        if freq < 1e-10 { freq = 1e-10 }  // .clamp(min=1e-10)
        density[y * latW + x] = Float(k / pow(freq, exponent))
      }
    }
    density[0] = 0  // spectral_density[0, 0] = 0

    let densityArray = MLXArray(density, [latH, latW])
    let noiseFFT = MLX.fft2(grid)                 // over the trailing (latH, latW) axes
    let modified = noiseFFT * densityArray         // broadcast (latH,latW) over (1,C,latH,latW)
    return MLX.ifft2(modified).realPart()
  }
}

// MARK: - Pyramid

/// RES4LYF's `PyramidNoiseGenerator` (`noise_classes.py:252`), on the Krea 2
/// patchified latent.
///
/// Five DOWN-sampled octaves summed with a geometric `discount` falloff, then
/// normalised. Each octave draws `normal(std = 0.5^i)` at `r = 2^{i+1}` times
/// the grid resolution and `nearest-exact`-downsamples back — upstream's
/// `torch.nn.functional.interpolate(size=origSize, mode='nearest-exact')`,
/// whose integer-factor index rule is `floor((i+0.5)·r) = i·r + r/2`.
public final class RES4LYFPyramidNoiseStream: RES4LYFNoiseStream {
  private var key: MLXArray
  private let layout: RES4LYFNoiseLayout
  private let hTok: Int
  private let wTok: Int
  private let channels: Int
  private let discount: Double

  public init(
    seed: UInt64, layout: RES4LYFNoiseLayout, hTok: Int, wTok: Int, channels: Int,
    discount: Double = 0.8
  ) {
    self.key = MLXRandom.key(seed)
    self.layout = layout
    self.hTok = hTok
    self.wTok = wTok
    self.channels = channels
    self.discount = discount
  }

  public func next(like sample: MLXArray) -> MLXArray {
    let geo = RES4LYFSpatialGeometry(sample: sample, hTok: hTok, wTok: wTok, channels: channels)
    let latH = geo.latH, latW = geo.latW

    // One advance key + five octave keys, drawn from the stream's own chain.
    let keys = MLXRandom.split(key: key, into: 6)
    key = keys[0]

    var x = MLXArray.zeros([1, channels, latH, latW], dtype: .float32)
    var r = 1
    for i in 0..<5 {
      r *= 2  // r = 2, 4, 8, 16, 32
      let std = Float(pow(0.5, Double(i)))
      let hi = MLXRandom.normal(
        [1, channels, latH * r, latW * r], dtype: .float32, scale: std, key: keys[i + 1])
      let down = Self.nearestExactDownsample(hi, toH: latH, toW: latW)
      x = x + down * Float(pow(discount, Double(i)))
    }

    let normalised = x / MLX.std(x, ddof: 1)      // upstream's `x / x.std()`
    let repatched = Krea2Sampling.patchify(normalised, patch: geo.p)
    return RES4LYFNoiseNormalization.zscore(repatched, layout: layout)
  }

  /// `nearest-exact` downsample of `(b, c, toH·r, toW·r)` to `(b, c, toH, toW)`.
  ///
  /// For an integer factor `r`, torch's `nearest-exact` picks source index
  /// `floor((i + 0.5)·r) = i·r + r/2`. Reshaping the axis `toH·r → (toH, r)`
  /// makes `r/2` a strided pick on the inserted `r` axis; the same on width.
  static func nearestExactDownsample(_ x: MLXArray, toH: Int, toW: Int) -> MLXArray {
    precondition(x.ndim == 4, "nearestExactDownsample expects (b, c, H, W)")
    let b = x.dim(0), c = x.dim(1), h = x.dim(2), w = x.dim(3)
    let rH = h / toH, rW = w / toW
    let reshaped = x.reshaped(b, c, toH, rH, toW, rW)
    // Int index on the rH/rW axes removes them: (b, c, toH, toW).
    return reshaped[0..., 0..., 0..., rH / 2, 0..., rW / 2]
  }
}
