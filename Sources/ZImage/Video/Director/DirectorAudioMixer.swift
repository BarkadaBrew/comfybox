// DirectorAudioMixer.swift — the Director timeline's audio mixer (WP2d).
//
// Pure CPU arithmetic over [Float]: a silent canvas the length of the
// timeline at settings.fps, and every placed clip is trimmed, gained and
// ADDED into it (overlaps sum), then hard-clipped to [-1, 1]. Deliberately no
// LTX2AudioEnhance / loudness / EQ — those are mastering for BigVGAN output;
// a user's imported clip is never recoloured.
//
// Two placement sources:
//   imported mode  one PlacedAudio per timeline audio_clip (start/length/trim/
//                  gain straight from the timeline; pcm from
//                  DirectorAudioIngest.decode).
//   generated mode chunkPlacement(): each chunk's own generated audio placed
//                  at its start frame, with the shared boundary frame's
//                  1/fps of audio dropped for k > 0 — mirroring the stitcher
//                  dropping that frame from the picture.

import Foundation

/// One clip on the timeline. Frame fields are timeline frames at the mixer's
/// fps; `pcm` is expected at the mixer's sample rate (the ingest decodes at
/// DirectorAudioIngest.timelineSampleRate; no resampling happens here).
public struct PlacedAudio: Sendable, Equatable {
  public var pcm: StereoPCM
  public var startFrame: Int
  public var lengthFrames: Int
  public var trimStartFrames: Int
  public var gain: Float

  public init(pcm: StereoPCM, startFrame: Int, lengthFrames: Int, trimStartFrames: Int = 0, gain: Float = 1.0) {
    self.pcm = pcm
    self.startFrame = startFrame
    self.lengthFrames = lengthFrames
    self.trimStartFrames = trimStartFrames
    self.gain = gain
  }
}

public enum DirectorAudioMixer {

  /// Samples for `frames` video frames: ceil(frames / fps * sampleRate). The
  /// canvas length and the generator's own audio trim use the same formula
  /// (LTX2VideoGenerator trims generated audio to ceil(frames/fps*sr)).
  public static func sampleCount(frames: Int, fps: Int, sampleRate: Int) -> Int {
    guard frames > 0, fps > 0, sampleRate > 0 else { return 0 }
    return Int((Double(frames) / Double(fps) * Double(sampleRate)).rounded(.up))
  }

  /// Sample OFFSET of a timeline frame (rounded to nearest; exact whenever
  /// fps divides the sample rate, e.g. 24 fps @ 48 kHz = 2000 samples/frame).
  static func sampleOffset(frame: Int, fps: Int, sampleRate: Int) -> Int {
    guard frame > 0, fps > 0 else { return 0 }
    return Int((Double(frame) * Double(sampleRate) / Double(fps)).rounded())
  }

  /// Mix `clips` onto a silent stereo canvas of `lengthFrames` at `fps`.
  /// Per clip: skip `trimStartFrames` of the source, place at `startFrame`,
  /// take `lengthFrames` (silence-padded when the source is shorter),
  /// multiply by `gain`, add. Writes never leave the canvas. Final hard clip.
  public static func mix(
    _ clips: [PlacedAudio],
    fps: Int,
    lengthFrames: Int,
    sampleRate: Int = 48000
  ) -> StereoPCM {
    let n = sampleCount(frames: lengthFrames, fps: fps, sampleRate: sampleRate)
    var left = [Float](repeating: 0, count: n)
    var right = [Float](repeating: 0, count: n)
    guard n > 0 else { return StereoPCM(left: left, right: right, sampleRate: sampleRate) }

    for clip in clips {
      guard clip.lengthFrames > 0, clip.gain != 0 else { continue }
      let start = sampleOffset(frame: clip.startFrame, fps: fps, sampleRate: sampleRate)
      guard start < n else { continue }
      let slot = sampleOffset(frame: clip.lengthFrames, fps: fps, sampleRate: sampleRate)
      let trim = sampleOffset(frame: max(0, clip.trimStartFrames), fps: fps, sampleRate: sampleRate)
      let available = clip.pcm.frames - trim
      guard available > 0 else { continue }  // trimmed past the end: silence
      // Real samples to copy: bounded by the slot, the source, and the canvas.
      let count = min(slot, available, n - start)
      guard count > 0 else { continue }
      let gain = clip.gain
      clip.pcm.left.withUnsafeBufferPointer { src in
        for i in 0..<count { left[start + i] += src[trim + i] * gain }
      }
      clip.pcm.right.withUnsafeBufferPointer { src in
        for i in 0..<count { right[start + i] += src[trim + i] * gain }
      }
    }

    for i in 0..<n {
      left[i] = min(1, max(-1, left[i]))
      right[i] = min(1, max(-1, right[i]))
    }
    return StereoPCM(left: left, right: right, sampleRate: sampleRate)
  }

  /// Generated-mode placement. Chunk k's frame 0 IS global frame start_k (the
  /// boundary shared with chunk k-1), and the stitcher keeps chunk k-1's copy
  /// of that frame — so for k > 0 the first frame's worth of chunk audio is
  /// trimmed, the remainder lands at start_k + 1, and the length is frames_k
  /// - 1. Placing the trimmed audio at start_k itself would shift it one
  /// frame early and double up the boundary frame. Zip semantics: a chunk
  /// without PCM (audio failed) is simply absent from the result.
  public static func chunkPlacement(
    chunkPCM: [StereoPCM],
    spans: [DirectorMath.ChunkSpan],
    fps: Int
  ) -> [PlacedAudio] {
    zip(chunkPCM, spans).map { pcm, span in
      let drop = span.index > 0 ? 1 : 0
      return PlacedAudio(
        pcm: pcm,
        startFrame: span.startFrame + drop,
        lengthFrames: span.frames - drop,
        trimStartFrames: drop,
        gain: 1.0)
    }
  }
}
