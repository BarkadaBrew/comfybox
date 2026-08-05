// LTX2AudioEnhance.swift — in-engine audio mastering (task #26).
//
// Raw LTX audio decodes ~10dB quiet with BigVGAN/BWE high-frequency grain
// (measured 2026-08-04 across the e2e set; no clipping — peaks ~-12dBFS).
// This chain runs between codec decode and AAC mux, replacing the external
// ffmpeg prototype (highpass 50 -> eq -2.5dB@7.5kHz -> loudnorm I=-16):
//
//   1. High-pass biquad (50Hz, Q .707) — rumble/DC guard.
//   2. Peaking biquad (7.5kHz, Q 1.2, -2.5dB) — softens the BWE grain band.
//   3. RMS loudness raise toward -18dBFS (~-16 LUFS for this content) with
//      a bounded gain and a tanh soft-knee ceiling at 0.85 FS — loud input
//      passes through nearly unity, quiet input comes up, nothing clips.
//
// Deliberately NO spectral denoise in v1 (the biquads + loudness carry most
// of the audible win; a gate can misfire on breath/ambience). Pure function,
// CPU-side [Float] math — a 12s stereo track is ~1.2M samples, negligible.

import Foundation
import MLX

public enum LTX2AudioEnhance {

  /// RBJ biquad, direct form 1, processed per channel.
  private struct Biquad {
    let b0, b1, b2, a1, a2: Float
    func run(_ x: [Float]) -> [Float] {
      var y = [Float](repeating: 0, count: x.count)
      var x1: Float = 0, x2: Float = 0, y1: Float = 0, y2: Float = 0
      for i in 0..<x.count {
        let v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1; x1 = x[i]; y2 = y1; y1 = v
        y[i] = v
      }
      return y
    }

    static func highPass(hz: Float, q: Float, rate: Float) -> Biquad {
      let w = 2 * Float.pi * hz / rate
      let alpha = sin(w) / (2 * q)
      let c = cos(w)
      let a0 = 1 + alpha
      return Biquad(
        b0: (1 + c) / 2 / a0, b1: -(1 + c) / a0, b2: (1 + c) / 2 / a0,
        a1: -2 * c / a0, a2: (1 - alpha) / a0)
    }

    static func peaking(hz: Float, q: Float, gainDB: Float, rate: Float) -> Biquad {
      let A = pow(10, gainDB / 40)
      let w = 2 * Float.pi * hz / rate
      let alpha = sin(w) / (2 * q)
      let c = cos(w)
      let a0 = 1 + alpha / A
      return Biquad(
        b0: (1 + alpha * A) / a0, b1: -2 * c / a0, b2: (1 - alpha * A) / a0,
        a1: -2 * c / a0, a2: (1 - alpha / A) / a0)
    }
  }

  /// `(channels, N)` float in [-1, 1] -> mastered, same shape/dtype domain.
  public static func process(_ samples: MLXArray, sampleRate: Int) -> MLXArray {
    let channels = samples.dim(0)
    let n = samples.dim(1)
    guard n > 0 else { return samples }
    let rate = Float(sampleRate)
    let hp = Biquad.highPass(hz: 50, q: 0.707, rate: rate)
    let dip = Biquad.peaking(hz: 7500, q: 1.2, gainDB: -2.5, rate: rate)

    var outChannels: [[Float]] = []
    outChannels.reserveCapacity(channels)
    for c in 0..<channels {
      var ch = samples[c].asType(.float32).asArray(Float.self)
      ch = hp.run(ch)
      ch = dip.run(ch)
      outChannels.append(ch)
    }

    // Loudness: one gain for ALL channels (stereo image preserved).
    var sumSq: Float = 0
    for ch in outChannels { for v in ch { sumSq += v * v } }
    let rms = sqrt(sumSq / Float(channels * n))
    let targetRMS: Float = pow(10, -18.0 / 20.0)  // -18 dBFS
    // Raise quiet content, cap the boost (+20dB), and never attenuate more
    // than a touch — already-loud tracks pass ~unity into the soft ceiling.
    let gain = min(max(targetRMS / max(rms, 1e-6), 1.0), 10.0)

    let ceiling: Float = 0.85
    var flat = [Float]()
    flat.reserveCapacity(channels * n)
    for ch in outChannels {
      for v in ch {
        // Soft-knee: linear below ~60% of ceiling, tanh into the ceiling.
        let g = v * gain
        flat.append(ceiling * tanh(g / ceiling))
      }
    }
    return MLXArray(flat).reshaped([channels, n])
  }
}
