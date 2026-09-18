import Foundation
import XCTest

@testable import ZImage

/// "Use thoughtful pauses to span the joins" (Todd, 2026-09-18) — WP11.
///
/// LTX-2 does coarse audio-visual correspondence, not phoneme-to-viseme; its
/// own ceiling measures ~1.23, so exact lip sync is not available. But a viewer
/// forgives phonemes that do not match. What they cannot forgive is a mouth
/// caught mid-word jumping at a seam — and a seam is the ONLY discontinuity a
/// sequence has. Put the join inside a pause and there is nothing there to
/// perceive as wrong.
final class DirectorSilentJoinsTests: XCTestCase {

  private let length = 817
  private let ceiling = DirectorMath.maxChunkFrames

  private var layout: [DirectorMath.ChunkSpan] {
    DirectorMath.chunkLayout(lengthFrames: length, maxFrames: ceiling)
  }

  private func assertLegal(
    _ spans: [DirectorMath.ChunkSpan], file: StaticString = #filePath, line: UInt = #line
  ) {
    for span in spans {
      XCTAssertEqual((span.frames - 1) % 8, 0, "chunk \(span.index) is not 1+8k", file: file, line: line)
      XCTAssertLessThanOrEqual(span.frames, ceiling, file: file, line: line)
      XCTAssertGreaterThanOrEqual(span.frames, 9, file: file, line: line)
    }
    XCTAssertEqual(spans.first?.startFrame, 0, file: file, line: line)
    let covered = spans.reduce(0) { $0 + $1.frames } - (spans.count - 1)
    XCTAssertEqual(covered, length, "chunks must tile the timeline", file: file, line: line)
  }

  // MARK: the objective is MARGIN, not mere silence

  func testAJoinLandsInTheMIDDLEOfAPauseNotItsEdge() {
    // Two reachable candidates: one at the very edge of a pause, one deep
    // inside a wider one further away. "Span the join" means the deep one —
    // a boundary on the last silent frame before speech resumes is silent and
    // perceptually useless.
    let nominal = layout[1].startFrame
    let edge = nominal - 8          // last frame of a short pause
    let deep = nominal + 16         // centre of a wide pause
    let pauses = [(edge - 1)..<(edge + 1), (deep - 20)..<(deep + 20)]
    let snapped = DirectorMath.spanningPauses(
      layout, lengthFrames: length, maxFrames: ceiling, pauses: pauses)
    XCTAssertEqual(snapped[1].startFrame, deep, "the deeper silence wins over the nearer one")
    assertLegal(snapped)
  }

  func testTiesGoToTheSMALLERMove() {
    // Equal silence either side: the boundary must not wander further than it
    // has to.
    let nominal = layout[1].startFrame
    let near = nominal - 8, far = nominal + 24
    let pauses = [(near - 10)..<(near + 10), (far - 10)..<(far + 10)]
    let snapped = DirectorMath.spanningPauses(
      layout, lengthFrames: length, maxFrames: ceiling, pauses: pauses)
    XCTAssertEqual(snapped[1].startFrame, near)
    assertLegal(snapped)
  }

  // MARK: degrading safely

  func testNoPausesLeavesTheLayoutEXACTLYAsItWas() {
    let snapped = DirectorMath.spanningPauses(
      layout, lengthFrames: length, maxFrames: ceiling, pauses: [])
    XCTAssertEqual(snapped.map(\.startFrame), layout.map(\.startFrame))
    XCTAssertEqual(snapped.map(\.frames), layout.map(\.frames))
  }

  func testAPauseOutsideTheSearchWindowIsIgnored() {
    let nominal = layout[1].startFrame
    let snapped = DirectorMath.spanningPauses(
      layout, lengthFrames: length, maxFrames: ceiling,
      pauses: [(nominal - 200)..<(nominal - 150)], searchFrames: 32)
    XCTAssertEqual(snapped[1].startFrame, nominal, "a distant pause must not drag the join")
    assertLegal(snapped)
  }

  func testAPauseThatWouldMakeAnIllegalChunkIsRefused() {
    // Silence only at the very start: moving there leaves the next chunk over
    // the ceiling, so the join must stay put.
    let snapped = DirectorMath.spanningPauses(
      layout, lengthFrames: length, maxFrames: ceiling, pauses: [0..<40])
    assertLegal(snapped)
    XCTAssertEqual(snapped.map(\.startFrame), layout.map(\.startFrame))
  }

  func testOnlyMultiplesOfTheLatentStrideAreReachable() {
    let nominal = layout[1].startFrame
    // A pause so narrow that its only frames are off the 8-grid.
    let snapped = DirectorMath.spanningPauses(
      layout, lengthFrames: length, maxFrames: ceiling,
      pauses: [(nominal - 3)..<(nominal - 1)])
    XCTAssertEqual(snapped[1].startFrame, nominal)
    assertLegal(snapped)
  }

  // MARK: pause DETECTION — a pause is a run, not a frame

  func testAnInterSyllableDipIsNotAPause() {
    // The mistake that made a first pass report every join safe when two of
    // five were mid-word: at a loose threshold the dips BETWEEN syllables
    // qualify. A pause must be continuous quiet.
    let rate = 48_000, fps = 24
    let perFrame = rate / fps
    var left = [Float](repeating: 0, count: perFrame * 60)
    // Alternate loud/quiet every 2 frames — speech, not pauses.
    for frame in 0..<60 where (frame / 2) % 2 == 0 {
      for i in (frame * perFrame)..<((frame + 1) * perFrame) { left[i] = 0.8 }
    }
    let pcm = StereoPCM(left: left, right: left, sampleRate: rate)
    let runs = DirectorAudioIngest.pauseRuns(pcm, fps: fps, frameCount: 60, minRunFrames: 8)
    XCTAssertTrue(runs.isEmpty, "2-frame gaps between syllables are not pauses: \(runs)")
  }

  func testARealPauseIsFound() {
    let rate = 48_000, fps = 24
    let perFrame = rate / fps
    var left = [Float](repeating: 0, count: perFrame * 60)
    for frame in 0..<60 where !(20..<40).contains(frame) {
      for i in (frame * perFrame)..<((frame + 1) * perFrame) { left[i] = 0.8 }
    }
    let pcm = StereoPCM(left: left, right: left, sampleRate: rate)
    let runs = DirectorAudioIngest.pauseRuns(pcm, fps: fps, frameCount: 60, minRunFrames: 8)
    XCTAssertEqual(runs.count, 1)
    XCTAssertEqual(runs.first?.lowerBound, 20)
    XCTAssertEqual(runs.first?.upperBound, 40)
  }

  func testASilentTrackIsOnePauseNotZero() {
    let pcm = StereoPCM.silence(frames: 48_000, sampleRate: 48_000)
    let runs = DirectorAudioIngest.pauseRuns(pcm, fps: 24, frameCount: 24)
    XCTAssertEqual(runs, [0..<24])
  }

  // MARK: end to end

  func testTheMovedJoinsREACHTheRenderedBodies() throws {
    let timeline = DirectorTimeline(
      settings: .init(fps: 24, width: 576, height: 896, lengthFrames: length),
      globalPrompt: "she talks to camera",
      keyframes: [.init(id: "k1", imagePath: "/img/a.png", frame: 0)],
      audioClips: [
        .init(id: "voice", audioPath: "/audio/v.wav", startFrame: 0,
              lengthFrames: length, drivesVideo: true)
      ],
      audio: .init(mode: .imported))
    let nominal = layout.dropFirst().map(\.startFrame)
    let pauses = nominal.map { ($0 - 12)..<($0 + 12) }

    let validation = DirectorValidator.validate(
      timeline, fileExists: { _ in true }, pauseRuns: { _, _, _ in pauses })
    XCTAssertTrue(validation.ok, "\(validation.issues)")
    let plan = try XCTUnwrap(validation.plan)
    XCTAssertTrue(plan.joinsInPauses || plan.boundaryFrames == nominal)

    let compilation = try DirectorCompiler.compile(validation, session: "T", source: "test")
    XCTAssertEqual(
      compilation.chunks.map(\.span.startFrame), plan.chunks.map(\.startFrame),
      "the rendered chunks must start where the plan says — no silent recompute")
  }

  func testANonDrivingTimelineIsNeverMovedEvenWithPausesEverywhere() throws {
    let timeline = DirectorTimeline(
      settings: .init(fps: 24, width: 576, height: 896, lengthFrames: length),
      globalPrompt: "x",
      audioClips: [
        .init(id: "a", audioPath: "/audio/v.wav", startFrame: 0,
              lengthFrames: length, drivesVideo: false)
      ],
      audio: .init(mode: .imported))
    let validation = DirectorValidator.validate(
      timeline, fileExists: { _ in true }, pauseRuns: { _, _, _ in [0..<length] })
    XCTAssertEqual(validation.plan?.joinsInPauses, false)
    XCTAssertEqual(validation.plan?.boundaryFrames, layout.dropFirst().map(\.startFrame))
  }
}
