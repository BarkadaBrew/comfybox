// DirectorMath.swift — pure frame arithmetic for the Director timeline.
//
// Shared by the engine (validator/compiler), the CLI and the desktop ruler so
// every surface snaps, chunks and maps frames identically. No engine
// dependencies; every function is a pure static. Cross-checked in
// DirectorMathTests against LTX2VideoGenerator.chunkPlan and
// WarmServer.resolvedLTX2Frames so the plan a client sees is what renders.

import Foundation

public enum DirectorMath {

  /// Video frames per latent frame (LTX-2.3 VAE temporal compression).
  public static let latentStride = 8
  /// The trained single-pass ceiling: 289 frames (12 s @ 24 fps).
  public static let maxChunkFrames = 289
  /// Latent steps per chunk at the ceiling: (289 - 1) / 8.
  public static let maxChunkSteps = (maxChunkFrames - 1) / latentStride
  /// Phase 1 cap on chunks per timeline.
  public static let maxChunks = 16
  /// Production floor for a video render (WarmServer.resolvedLTX2Frames).
  public static let minTimelineFrames = 97
  /// 1 + 8 * 36 * 16.
  public static let maxTimelineFrames = 1 + latentStride * maxChunkSteps * maxChunks
  /// Keyframes closer than this (in frames) trigger the jump-cut warning.
  public static let closeKeyframeFrames = 24
  /// A prompt segment shorter than this has a zero-width beat bias window
  /// (buildVideoBias window = max(span/2 - 2, 0) latent frames).
  public static let biasWindowFrames = 32

  /// One rendered chunk of the timeline. `endFrame` is inclusive and is
  /// shared with the next chunk's `startFrame` (the carry-over frame).
  public struct ChunkSpan: Sendable, Equatable, Hashable {
    public let index: Int
    public let startFrame: Int
    public let endFrame: Int
    public let frames: Int

    public init(index: Int, startFrame: Int, frames: Int) {
      self.index = index
      self.startFrame = startFrame
      self.frames = frames
      self.endFrame = startFrame + frames - 1
    }
  }

  // MARK: Length

  /// Smallest 1 + 8k (k >= 1) that is >= L. Anything below 2 snaps to 9.
  public static func snapLengthUp(_ length: Int) -> Int {
    if length <= 1 { return 9 }
    return ((length - 2) / latentStride) * latentStride + 9
  }

  /// 1 + 8k with k >= 1.
  public static func isValidLength(_ length: Int) -> Bool {
    length >= 9 && (length - 1) % latentStride == 0
  }

  /// Latent frames for a 1 + 8k video: (n - 1) / 8 + 1.
  public static func latentFrames(for numFrames: Int) -> Int {
    max(1, (numFrames - 1) / latentStride + 1)
  }

  // MARK: Keyframes

  /// Video frame -> latent row with integer FLOOR, clamped to the last row.
  /// Mirrors LTX2Pipeline.generateMultiKeyframeResumable
  /// (`min(kf.videoFrameIndex / 8, latF - 1)`). Upstream ceils; this port
  /// floors, which is why the validator forbids off-grid keyframes — on the
  /// grid the two agree.
  public static func latentIndex(frame: Int, latF: Int) -> Int {
    min(max(0, frame) / latentStride, max(0, latF - 1))
  }

  /// Nearest multiple of 8 (ties round DOWN: 100 -> 96, 101 -> 104),
  /// clamped to [0, length - 1] (the desktop ruler).
  public static func snapToGrid(frame: Int, length: Int) -> Int {
    let f = max(0, frame)
    let snapped = ((f + latentStride / 2 - 1) / latentStride) * latentStride
    let upper = max(0, length - 1)
    return min(snapped, upper)
  }

  /// true when any two frames share a latent bucket (frame / 8): the later
  /// condition would silently overwrite the earlier in applyConditioning.
  public static func keyframeBucketsCollide(_ frames: [Int]) -> Bool {
    var seen = Set<Int>()
    for f in frames {
      if !seen.insert(f / latentStride).inserted { return true }
    }
    return false
  }

  // MARK: Time

  public static func frames(seconds: Double, fps: Int) -> Int {
    Int((seconds * Double(fps)).rounded())
  }

  public static func seconds(frames: Int, fps: Int) -> Double {
    guard fps > 0 else { return 0 }
    return Double(frames) / Double(fps)
  }

  // MARK: Chunks

  /// Split a 1 + 8k timeline into the fewest balanced chunks of at most 289
  /// frames, each 1 + 8k, sharing one boundary frame with its neighbour.
  ///
  /// S = (L - 1) / 8 latent steps; n = ceil(S / 36); base = S / n,
  /// extra = S % n; chunk i has base + (i < extra ? 1 : 0) steps and
  /// 8 * steps + 1 frames; start_{i+1} = start_i + frames_i - 1. The frames
  /// sum back to L. For n >= 2 every chunk is in [145, 289]; for n == 1 the
  /// single chunk is L.
  /// The ceiling for an AUDIO-DRIVEN timeline (WP11). Measured 2026-09-18 on
  /// the same seed/keyframe/prompt, as the ratio of mouth motion during speech
  /// to mouth motion during silence — "does the mouth go still when the voice
  /// stops", which unlike a correlation does not depend on how much of the
  /// track is speech:
  ///
  ///     145-frame chunks   1.92 (LTX voice), 1.73 (TTS voice)
  ///     273-frame chunks   1.03, 1.27, 1.30  (the 3 chunks of a 34 s take)
  ///
  /// CORRECTED 2026-09-18 05:40 after rendering the A/B this was supposed to
  /// justify. The numbers above are all FIRST chunks — every one of those
  /// measurements was a single-chunk render conditioning off a real keyframe —
  /// and the generalisation to every chunk does not hold:
  ///
  ///     34 s take, same script/voice/keyframe/seed, only the chunking changed
  ///     3 x 273 f:  chunk0 1.03 | continuation chunks pooled 1.31
  ///     6 x 137 f:  chunk0 3.05 | continuation chunks pooled 1.10
  ///
  /// Shortening chunks transforms the FIRST chunk (1.03 -> 3.05) and does
  /// nothing for the rest. Net it is still a clear win — whole-clip response
  /// 1.20 -> 1.45, r 0.132 -> 0.195 — so the ceiling stays. But the honest
  /// claim is narrower than "chunk length is the lever": a continuation chunk
  /// conditions POORLY ON AUDIO AT ANY LENGTH, and why is the open question
  /// (the carry-over frame is the obvious suspect, since it is the only thing
  /// that distinguishes those chunks from the ones that score 1.7-3.0).
  public static let audioDrivenChunkFrames = 145
  public static var audioDrivenChunkSteps: Int { (audioDrivenChunkFrames - 1) / latentStride }

  public static func chunkLayout(
    lengthFrames: Int, maxFrames: Int = maxChunkFrames
  ) -> [ChunkSpan] {
    let length = max(9, lengthFrames)
    let ceilingSteps = max(1, (max(9, maxFrames) - 1) / latentStride)
    let steps = (length - 1) / latentStride
    let count = max(1, (steps + ceilingSteps - 1) / ceilingSteps)
    let base = steps / count
    let extra = steps % count
    var spans: [ChunkSpan] = []
    spans.reserveCapacity(count)
    var start = 0
    for i in 0..<count {
      let s = base + (i < extra ? 1 : 0)
      let frames = latentStride * s + 1
      spans.append(ChunkSpan(index: i, startFrame: start, frames: frames))
      start += frames - 1
    }
    return spans
  }

  /// Chunk-local frame index (frame 0 of chunk k is global start_k).
  public static func localFrame(global: Int, chunk: ChunkSpan) -> Int {
    global - chunk.startFrame
  }

  /// Ownership rule: chunk k conditions a user keyframe when
  /// start_k < frame <= end_k, plus frame == 0 for chunk 0. A keyframe on a
  /// boundary (frame == start_k, k > 0) belongs to chunk k-1 as its LAST
  /// frame; chunk k conditions its frame 0 on the rendered carry-over.
  public static func chunkOwnsKeyframe(_ chunk: ChunkSpan, frame: Int) -> Bool {
    if chunk.index == 0 && frame == 0 { return true }
    return frame > chunk.startFrame && frame <= chunk.endFrame
  }

  /// The chunk that conditions on a keyframe at `frame`, per the ownership
  /// rule above; nil when the frame is outside the timeline.
  public static func owningChunk(frame: Int, layout: [ChunkSpan]) -> ChunkSpan? {
    layout.first { chunkOwnsKeyframe($0, frame: frame) }
  }

  // MARK: Beats

  /// Fractions of chunk k's duration covered by segment [start, start+len):
  /// lo = max(start, start_k), hi = min(start + len, end_k + 1); nil when the
  /// segment does not overlap the chunk. A segment spanning a boundary
  /// appears in both chunks (ending at 1.0 in k, starting at 0.0 in k+1).
  ///
  /// The shared boundary frame alone is NOT an overlap: a segment starting
  /// on start_{k+1} (the grid-snapped normal case) would otherwise leak into
  /// chunk k as a zero-width beat whose text biases chunk k's last frame —
  /// the carry-over chunk k+1 is conditioned on. Likewise a segment ending at
  /// start_k + 1 does not put a (0, 1/f) sliver into chunk k > 0.
  /// `sharesEndFrame` is false for the timeline's last chunk (its end frame
  /// belongs to no one else).
  public static func beatFractions(
    segmentStart: Int, segmentLength: Int, chunk: ChunkSpan, sharesEndFrame: Bool = true
  ) -> (startFrac: Float, endFrac: Float)? {
    let lo = max(segmentStart, chunk.startFrame)
    let hi = min(segmentStart + segmentLength, chunk.endFrame + 1)
    guard hi > lo, chunk.frames > 0 else { return nil }
    if hi - lo == 1 {
      if sharesEndFrame && lo == chunk.endFrame { return nil }
      if chunk.index > 0 && lo == chunk.startFrame { return nil }
    }
    let frames = Float(chunk.frames)
    return (Float(lo - chunk.startFrame) / frames, Float(hi - chunk.startFrame) / frames)
  }
}
