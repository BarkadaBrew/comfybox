// LTX2AudioPatchifier.swift — audio latent <-> token conversion + real-time coords
// Task #21 (LTX-2 audio), top-level joint forward.
//
// Reference: symmetric_patchifier.py class AudioPatchifier, constructed as
// AudioPatchifier(1, start_end=True) with defaults sample_rate=16000,
// hop_length=160, audio_latent_downsample_factor=4, is_causal=True.
//
// patchify:   (B, C, T, F) -> tokens (B, T, C*F)  [channel-major (c f) flatten]
//             + timings (B, 1, T, 2) — per-token [start, end] in SECONDS.
// Times map latent frame t to its causal mel frame: mel = max(4t + 1 - 4, 0),
// seconds = mel * hop / sampleRate. These real-time coords drive the
// cross-modal RoPE shared with the video stream (which converts its frame
// axis to seconds via 1/frameRate).

import Foundation
import MLX

public enum LTX2AudioPatchifier {
  public static let sampleRate = 16000
  public static let hopLength = 160
  public static let downsampleFactor = 4

  /// Seconds for latent frame indices `start..<end` (causal mel mapping).
  static func latentTimesSeconds(from start: Int, to end: Int) -> [Float] {
    (start..<end).map { t in
      let mel = max(t * downsampleFactor + 1 - downsampleFactor, 0)
      return Float(mel) * Float(hopLength) / Float(sampleRate)
    }
  }

  /// `(B, C, T, F)` audio latents -> `(tokens (B, T, C*F), timings (B, 1, T, 2))`.
  public static func patchify(_ audioLatents: MLXArray) -> (tokens: MLXArray, timings: MLXArray) {
    let b = audioLatents.dim(0)
    let c = audioLatents.dim(1)
    let t = audioLatents.dim(2)
    let f = audioLatents.dim(3)

    // b c t f -> b t (c f): channel-major within each token.
    let tokens = audioLatents.transposed(0, 2, 1, 3).reshaped([b, t, c * f])

    let starts = MLXArray(latentTimesSeconds(from: 0, to: t))
    let ends = MLXArray(latentTimesSeconds(from: 1, to: t + 1))
    // stack -> (T, 2), broadcast to (B, 1, T, 2)
    let stacked = MLX.stacked([starts, ends], axis: -1).reshaped([1, 1, t, 2])
    let timings = MLX.broadcast(stacked, to: [b, 1, t, 2])
    return (tokens, timings)
  }

  /// `(B, T, C*F)` tokens -> `(B, C, T, F)` audio latents (exact inverse).
  public static func unpatchify(_ tokens: MLXArray, channels: Int, freq: Int) -> MLXArray {
    let b = tokens.dim(0)
    let t = tokens.dim(1)
    return tokens.reshaped([b, t, channels, freq]).transposed(0, 2, 1, 3)
  }
}
