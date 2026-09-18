// LTX2MelAnalysis.swift — waveform -> mel, using the checkpoint's OWN basis
// (FDD-ltx-director-tab §4.7, WP11: audio-driven chunks).
//
// The stack has always had mel -> waveform (LTX2Vocoder). Conditioning a render
// on an existing voice needs the other direction, and the risk the FDD names is
// exactly this step: "a wrong mel normalization makes the video follow noise."
//
// So nothing here is guessed. JoyAI-Echo ships its analysis front-end in the
// same checkpoint as three buffers the vocoder never uses on the synthesis path:
//
//   vocoder.mel_stft.stft_fn.forward_basis  [514, 1, 512]  — the STFT as a conv:
//       257 real + 257 imaginary filters of length 512 (n_fft 512).
//   vocoder.mel_stft.mel_basis              [64, 257]      — 64 mel bins.
//   vocoder.mel_stft.stft_fn.inverse_basis  [514, 1, 512]  — the synthesis side.
//
// We load the first two and apply them. The only convention not carried by a
// tensor is the hop, and that is pinned by the rest of the pipeline rather than
// chosen: the audio VAE compresses time 4x, the audio latent runs at 25 frames
// per second, so mel runs at 100 frames per second, which at 16 kHz is hop 160.
//
// Shapes: `mel(waveform:)` takes `(channels, samples)` at 16 kHz and returns
// `(1, 2, 64, T)` — exactly what `LTX2AudioVAE.encode` expects.

import Foundation
import Logging
import MLX

public struct LTX2MelAnalysis: Sendable {
  /// 16 kHz: the rate the audio VAE and vocoder were trained at.
  public static let sampleRate = 16_000
  /// n_fft, from `forward_basis`'s filter length.
  public static let fftSize = 512
  /// 160 samples = 100 mel frames/s = 4 x the 25/s audio latent rate.
  public static let hop = 160
  public static let melBins = 64
  /// The floor before the log, matching the usual BigVGAN/HiFi-GAN analysis.
  public static let logFloor: Float = 1e-5

  /// `(514, 512)` — 257 real then 257 imaginary rows.
  let forwardBasis: MLXArray
  /// `(64, 257)`
  let melBasis: MLXArray

  public init(forwardBasis: MLXArray, melBasis: MLXArray) {
    self.forwardBasis = forwardBasis
    self.melBasis = melBasis
  }

  /// Load the two analysis buffers from a JoyAI-Echo checkpoint.
  /// Returns nil when the checkpoint predates them, so a caller can fall back
  /// to generated audio rather than failing a render.
  public static func load(
    weights: [String: MLXArray], logger: Logger? = nil
  ) -> LTX2MelAnalysis? {
    guard let forward = weights["vocoder.mel_stft.stft_fn.forward_basis"],
          let mel = weights["vocoder.mel_stft.mel_basis"]
    else {
      logger?.warning("LTX-2 audio: checkpoint carries no mel analysis basis — cannot condition on audio")
      return nil
    }
    // forward_basis ships as (514, 1, 512); the middle axis is the conv's
    // input channel and is always 1 here.
    let flattened = forward.ndim == 3 ? forward.reshaped([forward.dim(0), forward.dim(2)]) : forward
    guard flattened.dim(0) == 514, flattened.dim(1) == fftSize,
          mel.dim(0) == melBins, mel.dim(1) == fftSize / 2 + 1
    else {
      logger?.warning(
        "LTX-2 audio: unexpected mel analysis shapes (forward \(flattened.shape), mel \(mel.shape))")
      return nil
    }
    return LTX2MelAnalysis(
      forwardBasis: flattened.asType(.float32), melBasis: mel.asType(.float32))
  }

  /// The audio latent rate: 25 frames/s, the VAE's 4x compression of the
  /// 100 frames/s mel.
  public static let latentRate = 25

  /// Audio latent frames for a duration — the ONE definition of `ta`.
  ///
  /// The pipeline sizes its audio noise with this and the generator sizes the
  /// analysis window with it; if the two ever disagreed by a frame, a supplied
  /// voice would be silently padded or clipped against the stream it is meant
  /// to replace.
  public static func latentFrames(seconds: Double) -> Int {
    max(1, Int((seconds * Double(latentRate)).rounded(.up)))
  }

  /// Number of mel frames a sample count produces (centre-padded, as the
  /// reference analysis does: `1 + samples / hop`).
  public static func frameCount(samples: Int) -> Int {
    max(1, samples / hop + 1)
  }

  /// Samples that cover `frames` mel frames.
  public static func sampleCount(frames: Int) -> Int {
    max(0, (frames - 1) * hop)
  }

  /// `(channels, samples)` float waveform in [-1, 1] -> `(1, 2, 64, T)` log-mel.
  /// A mono input is duplicated; more than two channels keep the first two.
  public func mel(waveform: MLXArray) -> MLXArray {
    let channels = waveform.dim(0)
    let samples = waveform.dim(1)
    let frames = Self.frameCount(samples: samples)
    // Centre padding: the reference pads n_fft/2 each side by reflection, which
    // puts frame 0 at sample 0 rather than half a window in.
    let pad = Self.fftSize / 2
    let padded = MLX.padded(waveform, widths: [IntOrPair(0), IntOrPair((pad, pad))], mode: .edge)

    var perChannel: [MLXArray] = []
    for channel in 0..<min(channels, 2) {
      perChannel.append(logMel(padded[channel], frames: frames))
    }
    if perChannel.count == 1 { perChannel.append(perChannel[0]) }  // mono -> stereo
    // (2, 64, T) -> (1, 2, 64, T)
    return MLX.stacked(perChannel, axis: 0).expandedDimensions(axis: 0)
  }

  /// One channel: STFT by the shipped basis, magnitude, mel, log.
  private func logMel(_ signal: MLXArray, frames: Int) -> MLXArray {
    // Frame the signal: (T, fftSize).
    var windows: [MLXArray] = []
    windows.reserveCapacity(frames)
    let available = signal.dim(0)
    for index in 0..<frames {
      let start = index * Self.hop
      let end = start + Self.fftSize
      if end <= available {
        windows.append(signal[start..<end])
      } else {
        // The tail frame is zero-padded rather than dropped, so the mel length
        // matches the audio latent length exactly.
        let tail = start < available ? signal[start..<available] : MLX.zeros([0], dtype: .float32)
        let missing = Self.fftSize - tail.dim(0)
        windows.append(
          missing > 0
            ? MLX.concatenated([tail, MLX.zeros([missing], dtype: .float32)], axis: 0)
            : tail)
      }
    }
    let framed = MLX.stacked(windows, axis: 0).asType(.float32)  // (T, 512)

    // (T, 512) @ (512, 514) -> (T, 514): real rows then imaginary rows.
    let spectrum = framed.matmul(forwardBasis.transposed(1, 0))
    let bins = Self.fftSize / 2 + 1
    let real = spectrum[0..., 0..<bins]
    let imaginary = spectrum[0..., bins...]
    // No epsilon inside the root: at 1e-9 it floors silence at ~3e-5, which
    // then lands ABOVE the log floor and makes silence look like quiet noise.
    let magnitude = MLX.sqrt(real * real + imaginary * imaginary)  // (T, 257)

    // (T, 257) @ (257, 64) -> (T, 64), then log with a floor.
    let mel = magnitude.matmul(melBasis.transposed(1, 0))
    let logged = MLX.log(MLX.maximum(mel, MLXArray(Self.logFloor)))
    return logged.transposed(1, 0)  // (64, T)
  }

  /// Resample a waveform to 16 kHz by linear interpolation. Voice tracks arrive
  /// at 44.1 or 48 kHz; the model only knows 16.
  public static func resample(
    _ waveform: [[Float]], from sourceRate: Int, to targetRate: Int = sampleRate
  ) -> [[Float]] {
    guard sourceRate != targetRate, sourceRate > 0, targetRate > 0 else { return waveform }
    let ratio = Double(sourceRate) / Double(targetRate)
    return waveform.map { channel -> [Float] in
      guard channel.count > 1 else { return channel }
      let outCount = max(1, Int(Double(channel.count) / ratio))
      var out = [Float](repeating: 0, count: outCount)
      for index in 0..<outCount {
        let position = Double(index) * ratio
        let left = Int(position)
        let right = min(left + 1, channel.count - 1)
        let fraction = Float(position - Double(left))
        out[index] = channel[left] * (1 - fraction) + channel[right] * fraction
      }
      return out
    }
  }
}
