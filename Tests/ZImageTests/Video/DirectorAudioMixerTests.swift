import Foundation
import XCTest

@testable import ZImage

/// WP2d: the timeline mixer is a pure CPU function over [Float]. No
/// AVFoundation, no mastering — a user's clip is placed, trimmed, gained
/// and summed, nothing else.
final class DirectorAudioMixerTests: XCTestCase {

  private let sr = 48000

  /// `seconds` of a 440 Hz tone at `amplitude` on both channels.
  private func tone(seconds: Double, amplitude: Float = 0.5, sampleRate: Int = 48000) -> StereoPCM {
    let n = Int(seconds * Double(sampleRate))
    var s = [Float](repeating: 0, count: n)
    for i in 0..<n { s[i] = amplitude * sin(2 * .pi * 440 * Float(i) / Float(sampleRate)) }
    return StereoPCM(left: s, right: s, sampleRate: sampleRate)
  }

  private func constant(_ value: Float, frames: Int, sampleRate: Int = 48000) -> StereoPCM {
    let s = [Float](repeating: value, count: frames)
    return StereoPCM(left: s, right: s, sampleRate: sampleRate)
  }

  private func maxAbs(_ x: ArraySlice<Float>) -> Float {
    x.reduce(0) { max($0, abs($1)) }
  }

  func testCanvasLengthMatchesTimeline() {
    let out = DirectorAudioMixer.mix([], fps: 24, lengthFrames: 289, sampleRate: sr)
    XCTAssertEqual(out.frames, 578000, "ceil(289/24*48000)")
    XCTAssertEqual(out.left.count, 578000)
    XCTAssertEqual(out.right.count, 578000)
    XCTAssertEqual(out.sampleRate, sr)
    XCTAssertEqual(maxAbs(out.left[...]), 0, "empty mix is silence")
    // Non-integral: 97 f @ 24 = 4.041666 s -> ceil = 194000.
    XCTAssertEqual(DirectorAudioMixer.mix([], fps: 24, lengthFrames: 97).frames, 194000)
    XCTAssertEqual(DirectorAudioMixer.sampleCount(frames: 97, fps: 24, sampleRate: 48000), 194000)
  }

  func testPlacementTrimGainAndPadding() {
    // A 1 s tone (48000 real samples) placed at frame 48 with 12 frames
    // trimmed off its head, occupying 96 frames on the timeline, at gain 0.5.
    let clip = PlacedAudio(pcm: tone(seconds: 1, amplitude: 0.5), startFrame: 48, lengthFrames: 96, trimStartFrames: 12, gain: 0.5)
    let out = DirectorAudioMixer.mix([clip], fps: 24, lengthFrames: 289, sampleRate: sr)
    XCTAssertEqual(out.frames, 578000)

    let start = 96000       // 48 / 24 * 48000
    let realEnd = start + (48000 - 24000)  // 24000 samples survive the 12-frame trim
    let slotEnd = start + 192000           // 96 frames

    XCTAssertEqual(maxAbs(out.left[0..<start]), 0, "silence before the clip")
    let peak = maxAbs(out.left[start..<realEnd])
    XCTAssertEqual(peak, 0.25, accuracy: 0.01, "amplitude halved by gain 0.5")
    XCTAssertGreaterThan(maxAbs(out.right[start..<realEnd]), 0.2, "right channel placed too")
    XCTAssertEqual(maxAbs(out.left[realEnd..<slotEnd]), 0, "padded with silence once the clip runs out")
    XCTAssertEqual(maxAbs(out.left[slotEnd...]), 0, "silence after the slot")

    // The trimmed head is really skipped: sample `start` is the tone's
    // sample 24000, not its sample 0 (which is exactly 0 for a sine).
    let expected = 0.5 * 0.5 * sin(2 * .pi * 440 * Float(24000) / Float(sr))
    XCTAssertEqual(out.left[start], expected, accuracy: 1e-5)
  }

  func testClipIsCutAtTheTimelineEnd() {
    // A clip that runs past the end never writes outside the canvas.
    let clip = PlacedAudio(pcm: constant(0.5, frames: 48000 * 4), startFrame: 96, lengthFrames: 96, trimStartFrames: 0, gain: 1)
    let out = DirectorAudioMixer.mix([clip], fps: 24, lengthFrames: 97, sampleRate: sr)
    XCTAssertEqual(out.frames, 194000)
    XCTAssertEqual(out.left[192000], 0.5, accuracy: 1e-6)
    XCTAssertEqual(out.left[193999], 0.5, accuracy: 1e-6)
  }

  func testOverlapsSumAndClip() {
    let a = PlacedAudio(pcm: constant(0.8, frames: 48000), startFrame: 0, lengthFrames: 24, trimStartFrames: 0, gain: 1)
    let b = PlacedAudio(pcm: constant(0.8, frames: 48000), startFrame: 12, lengthFrames: 24, trimStartFrames: 0, gain: 1)
    let out = DirectorAudioMixer.mix([a, b], fps: 24, lengthFrames: 97, sampleRate: sr)
    XCTAssertEqual(out.left[0], 0.8, accuracy: 1e-6, "only a")
    XCTAssertEqual(out.left[24000], 1.0, accuracy: 1e-6, "a + b = 1.6 hard-clipped to 1.0")
    XCTAssertEqual(out.right[24000], 1.0, accuracy: 1e-6)
    XCTAssertEqual(out.left[60000], 0.8, accuracy: 1e-6, "only b")
    XCTAssertEqual(out.left[72000], 0, accuracy: 1e-6, "past b")

    let neg = PlacedAudio(pcm: constant(-0.9, frames: 48000), startFrame: 0, lengthFrames: 24, trimStartFrames: 0, gain: 2)
    let out2 = DirectorAudioMixer.mix([neg], fps: 24, lengthFrames: 97, sampleRate: sr)
    XCTAssertEqual(out2.left[100], -1.0, accuracy: 1e-6, "negative side clips at -1")
  }

  func testTrimPastClipEndYieldsSilence() {
    let clip = PlacedAudio(pcm: constant(0.5, frames: 4800), startFrame: 0, lengthFrames: 24, trimStartFrames: 48, gain: 1)
    let out = DirectorAudioMixer.mix([clip], fps: 24, lengthFrames: 97, sampleRate: sr)
    XCTAssertEqual(maxAbs(out.left[...]), 0)
  }

  func testChunkPlacementDropsBoundaryAudio() {
    let spans = DirectorMath.chunkLayout(lengthFrames: 577)
    XCTAssertEqual(spans.map(\.frames), [289, 289])
    let pcm = [tone(seconds: 289.0 / 24.0), tone(seconds: 289.0 / 24.0)]
    let placed = DirectorAudioMixer.chunkPlacement(chunkPCM: pcm, spans: spans, fps: 24)
    XCTAssertEqual(placed.count, 2)
    XCTAssertEqual(placed[0].startFrame, 0)
    XCTAssertEqual(placed[0].trimStartFrames, 0)
    XCTAssertEqual(placed[0].lengthFrames, 289)
    XCTAssertEqual(placed[0].gain, 1)
    XCTAssertEqual(placed[1].trimStartFrames, 1, "the dropped boundary frame's 1/fps of audio is skipped")
    XCTAssertEqual(placed[1].startFrame, 289, "chunk 1's surviving audio starts one past the shared boundary frame (288)")
    XCTAssertEqual(placed[1].lengthFrames, 288)
    XCTAssertEqual(placed[1].startFrame + placed[1].lengthFrames - 1, 576, "ends on the timeline's last frame")

    // Placement covers exactly the timeline: 289 + 288 = 577 frames.
    XCTAssertEqual(placed.map(\.lengthFrames).reduce(0, +), 577)
    let mixed = DirectorAudioMixer.mix(placed, fps: 24, lengthFrames: 577)
    XCTAssertEqual(mixed.frames, DirectorAudioMixer.sampleCount(frames: 577, fps: 24, sampleRate: 48000))
  }

  func testChunkPlacementTruncatesToShorterList() {
    // A chunk whose audio failed (no PCM) is simply absent: zip semantics.
    let spans = DirectorMath.chunkLayout(lengthFrames: 577)
    let placed = DirectorAudioMixer.chunkPlacement(chunkPCM: [tone(seconds: 1)], spans: spans, fps: 24)
    XCTAssertEqual(placed.count, 1)
  }

  func testNegativeStartTrimsTheHeadInsteadOfShifting() {
    // A ramp so each sample is identifiable: sample i has value i / n.
    let n = 480000  // 10 s
    let ramp = (0..<n).map { Float($0) / Float(n) }
    let pcm = StereoPCM(left: ramp, right: ramp, sampleRate: sr)
    // start -48 (2 s before the timeline), 120 frames: only 72 frames
    // (3 s) are on the timeline, and they are the file's seconds 2..5.
    let clip = PlacedAudio(pcm: pcm, startFrame: -48, lengthFrames: 120)
    let out = DirectorAudioMixer.mix([clip], fps: 24, lengthFrames: 289, sampleRate: sr)
    XCTAssertEqual(out.left[0], ramp[96000], accuracy: 1e-6, "timeline 0 is the file at 2 s")
    XCTAssertEqual(out.left[143999], ramp[239999], accuracy: 1e-6, "last sample of the 72-frame slot")
    XCTAssertEqual(out.left[144000], 0, "nothing past the shortened slot")
    // Equivalent to the explicitly normalised clip.
    let normalised = DirectorAudioMixer.mix(
      [PlacedAudio(pcm: pcm, startFrame: 0, lengthFrames: 72, trimStartFrames: 48)],
      fps: 24, lengthFrames: 289, sampleRate: sr)
    XCTAssertEqual(out.left, normalised.left)
    // Entirely before the timeline: silence.
    let gone = DirectorAudioMixer.mix(
      [PlacedAudio(pcm: pcm, startFrame: -200, lengthFrames: 100)], fps: 24, lengthFrames: 289, sampleRate: sr)
    XCTAssertEqual(maxAbs(gone.left[...]), 0)
  }
}
