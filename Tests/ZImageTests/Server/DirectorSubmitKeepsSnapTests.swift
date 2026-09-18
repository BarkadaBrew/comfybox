import Foundation
import XCTest

@testable import ZImage

/// What `/v1/video/director/validate` promises must be what renders.
///
/// The submit route re-resolves fps and seed and then recompiles. That
/// recompile used `DirectorCompiler.plan` with no layout, which recomputes the
/// plain arithmetic split — silently discarding boundaries the validator had
/// moved into pauses. `/validate` reported joins at [240, 496] while the
/// engine accepted chunks of 249 frames, and the only reason the rendered clip
/// was still correct is that the voice had been composed to put pauses at the
/// arithmetic boundaries anyway.
final class DirectorSubmitKeepsSnapTests: XCTestCase {

  private func timeline(fps: Int?) -> DirectorTimeline {
    DirectorTimeline(
      settings: .init(fps: fps, width: 576, height: 896, lengthFrames: 737),
      globalPrompt: "she talks to camera",
      keyframes: [.init(id: "k1", imagePath: "/img/a.png", frame: 0)],
      audioClips: [
        .init(id: "voice", audioPath: "/audio/v.wav", startFrame: 0,
              lengthFrames: 737, drivesVideo: true)
      ],
      audio: .init(mode: .imported))
  }

  /// Pauses centred 8 frames BEFORE each arithmetic boundary, so a correct
  /// snap visibly moves the joins and a lost one visibly does not.
  private func pauses(for timeline: DirectorTimeline) -> [Range<Int>] {
    DirectorMath.chunkLayout(lengthFrames: 737, maxFrames: DirectorMath.maxChunkFrames)
      .dropFirst()
      .map { ($0.startFrame - 8 - 10)..<($0.startFrame - 8 + 10) }
  }

  func testTheSnappedLayoutSurvivesARECOMPILE() throws {
    let original = timeline(fps: nil)
    let runs = pauses(for: original)
    let validated = DirectorValidator.validate(
      original, fileExists: { _ in true }, pauseRuns: { _, _, _ in runs })
    let promised = try XCTUnwrap(validated.plan).boundaryFrames
    XCTAssertTrue(validated.plan?.joinsInPauses == true, "the fixture must actually move them")

    // What the submit route does: pin fps/seed, then build the plan again.
    let resolved = DirectorCompiler.resolvingDefaults(validated.snapped, fps: 24, seed: 707070)
    let revalidated = DirectorValidator.validate(
      resolved, fileExists: { _ in true }, pauseRuns: { _, _, _ in runs })
    let rendered = try XCTUnwrap(revalidated.plan).boundaryFrames

    XCTAssertEqual(
      rendered, promised,
      "the joins that render must be the joins /validate promised")
  }

  func testAPlainRecompileIsWhatUSEDToLoseThem() throws {
    // Pins the failure mode, so a future refactor that reaches for
    // `plan(for:)` here fails loudly instead of quietly un-snapping.
    let original = timeline(fps: 24)
    let runs = pauses(for: original)
    let validated = DirectorValidator.validate(
      original, fileExists: { _ in true }, pauseRuns: { _, _, _ in runs })
    let promised = try XCTUnwrap(validated.plan).boundaryFrames
    let naive = DirectorCompiler.plan(for: validated.snapped).boundaryFrames
    XCTAssertNotEqual(
      naive, promised,
      "a layout-less replan recomputes the arithmetic split — that is the bug")
  }

  func testTheBODIESFollowTheRevalidatedPlan() throws {
    let original = timeline(fps: 24)
    let runs = pauses(for: original)
    let validated = DirectorValidator.validate(
      original, fileExists: { _ in true }, pauseRuns: { _, _, _ in runs })
    let compilation = try DirectorCompiler.compile(validated, session: "T", source: "test")
    XCTAssertEqual(
      compilation.chunks.map(\.span.startFrame),
      try XCTUnwrap(validated.plan).chunks.map(\.startFrame))
  }
}
