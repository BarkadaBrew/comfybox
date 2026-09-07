import Foundation

/// "Phone look" post-process for 4-step distilled renders (Todd 2026-08-24).
///
/// The TDM 4-step distill quits before the final denoise steps that settle
/// dynamic range: renders come out hazy — lifted blacks (~0.07–0.23), no true
/// white (~0.71–0.78), compressed contrast (σ ~0.10–0.16 vs ~0.30 for the
/// 12-step res_2s reference). This deterministic CPU pass restores "a
/// reasonable mobile phone look — on par with an android phone or iPhone 8":
///
///   1. Percentile auto-levels: black = P0.5 of luminance, white = P99.5,
///      uniform RGB stretch (hue-preserving). Skipped when the window is
///      degenerate so a near-flat image is never stretched into noise.
///   2. S-contrast ×1.15 around mid-gray — the "pinch of contrast".
///   3. Adaptive saturation: measured AFTER the stretch (which itself
///      recovers chroma); nudged toward a 0.32 mean only when under, boost
///      capped at ×1.25 so warm golden-hour scenes don't overcook.
///   4. Unsharp mask (σ 1.6, amount 0.45, threshold 2/255) — the "pinch of
///      sharpness"; the threshold spares smooth skin from gritting up.
///
/// Pure function over an interleaved HWC RGB float buffer in [0, 1] — no
/// server, no weights, fully unit-testable (`PhoneLookTests`). Recipe was
/// validated against real renders (stacktest a1–a3, 2026-08-24) before being
/// ported here.
public enum PhoneLook {

  static let contrastGain: Float = 1.15
  static let saturationTarget: Float = 0.32
  static let saturationCap: Float = 1.25
  static let unsharpSigma: Float = 1.6
  static let unsharpAmount: Float = 0.45
  static let unsharpThreshold: Float = 2.0 / 255.0
  /// Below this luminance window the levels stretch is skipped (degenerate).
  static let minLevelsWindow: Float = 0.05

  /// Apply the full recipe in place. `pixels` is HWC interleaved RGB, [0,1].
  public static func apply(pixels: inout [Float], width: Int, height: Int) {
    let n = width * height
    guard n > 0, pixels.count == n * 3 else { return }

    // ── luminance + percentile window ──
    var lum = [Float](repeating: 0, count: n)
    for i in 0..<n {
      lum[i] = 0.2126 * pixels[i * 3] + 0.7152 * pixels[i * 3 + 1] + 0.0722 * pixels[i * 3 + 2]
    }
    let sorted = lum.sorted()
    let lo = sorted[min(n - 1, Int(0.005 * Float(n)))]
    let hi = sorted[min(n - 1, Int(0.995 * Float(n)))]

    // ── 1+2: levels stretch (guarded) + S-contrast, fused per component ──
    let window = hi - lo
    let stretch = window >= minLevelsWindow
    let inv = stretch ? 1 / window : 1
    for i in 0..<(n * 3) {
      var v = pixels[i]
      if stretch { v = (v - lo) * inv }
      v = 0.5 + (v - 0.5) * contrastGain
      pixels[i] = min(1, max(0, v))
    }

    // ── 3: adaptive saturation ──
    var satSum: Float = 0
    for i in 0..<n {
      let r = pixels[i * 3], g = pixels[i * 3 + 1], b = pixels[i * 3 + 2]
      let mx = max(r, g, b), mn = min(r, g, b)
      satSum += mx > 0 ? (mx - mn) / (mx + 1e-6) : 0
    }
    let meanSat = satSum / Float(n)
    let boost = min(saturationCap, max(1, meanSat > 1e-4 ? saturationTarget / meanSat : 1))
    if boost > 1 {
      for i in 0..<n {
        let r = pixels[i * 3], g = pixels[i * 3 + 1], b = pixels[i * 3 + 2]
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b
        pixels[i * 3] = min(1, max(0, l + (r - l) * boost))
        pixels[i * 3 + 1] = min(1, max(0, l + (g - l) * boost))
        pixels[i * 3 + 2] = min(1, max(0, l + (b - l) * boost))
      }
    }

    // ── 4: unsharp mask (separable gaussian, thresholded) ──
    let blurred = gaussianBlur(pixels, width: width, height: height, sigma: unsharpSigma)
    for i in 0..<(n * 3) {
      let diff = pixels[i] - blurred[i]
      if abs(diff) > unsharpThreshold {
        pixels[i] = min(1, max(0, pixels[i] + unsharpAmount * diff))
      }
    }
  }

  /// Separable gaussian blur over an HWC RGB buffer. Edge-clamped.
  static func gaussianBlur(_ src: [Float], width: Int, height: Int, sigma: Float) -> [Float] {
    let radius = max(1, Int((3 * sigma).rounded(.up)))
    var kernel = [Float](repeating: 0, count: 2 * radius + 1)
    var sum: Float = 0
    for k in -radius...radius {
      let v = expf(-Float(k * k) / (2 * sigma * sigma))
      kernel[k + radius] = v
      sum += v
    }
    for i in kernel.indices { kernel[i] /= sum }

    var tmp = [Float](repeating: 0, count: src.count)
    // horizontal
    for y in 0..<height {
      for x in 0..<width {
        var acc: (Float, Float, Float) = (0, 0, 0)
        for k in -radius...radius {
          let xs = min(width - 1, max(0, x + k))
          let si = (y * width + xs) * 3
          let kw = kernel[k + radius]
          acc.0 += src[si] * kw
          acc.1 += src[si + 1] * kw
          acc.2 += src[si + 2] * kw
        }
        let di = (y * width + x) * 3
        tmp[di] = acc.0; tmp[di + 1] = acc.1; tmp[di + 2] = acc.2
      }
    }
    var out = [Float](repeating: 0, count: src.count)
    // vertical
    for y in 0..<height {
      for x in 0..<width {
        var acc: (Float, Float, Float) = (0, 0, 0)
        for k in -radius...radius {
          let ys = min(height - 1, max(0, y + k))
          let si = (ys * width + x) * 3
          let kw = kernel[k + radius]
          acc.0 += tmp[si] * kw
          acc.1 += tmp[si + 1] * kw
          acc.2 += tmp[si + 2] * kw
        }
        let di = (y * width + x) * 3
        out[di] = acc.0; out[di + 1] = acc.1; out[di + 2] = acc.2
      }
    }
    return out
  }
}
