import Foundation
import XCTest

@testable import ZImage

/// The drafter (FDD §4.9.4, WP15). The contract under test: the PRESET owns
/// every number, the model owns only prose.
final class SequenceDrafterTests: XCTestCase {

  private func preset(
    seconds: Double = 30, cadence: Double? = 10, keyframes: String = "first_only",
    audio: String = "generated", notes: String? = nil
  ) -> SequencePreset {
    SequencePreset(
      id: "apple-30", name: "Apple 30s", lengthSeconds: seconds, fps: 24,
      width: 576, height: 896, videoPresetId: "kira-video-apple", steps: 10,
      negativePrompt: "blurry", character: nil,
      keyframePolicy: keyframes, segmentCadenceSeconds: cadence, audio: audio,
      seedPolicy: "fixed:4242", authorNotes: notes)
  }

  private func prose(segments: Int = 3) -> SequenceDraftProse {
    SequenceDraftProse(
      globalPrompt: "  A woman on a stool by a window, late light.  ",
      segments: (1...segments).map { "  beat \($0)  " },
      keyframeDescriptions: ["she sits down, looking off-camera"])
  }

  // MARK: assembly

  func testEveryNumberComesFromThePresetNotTheModel() {
    let p = preset()
    let timeline = SequenceDrafter.assemble(
      prose: prose(), preset: p,
      request: SequenceDraftRequest(presetId: p.id, brief: "her day"))

    XCTAssertEqual(timeline.settings.lengthFrames, p.snappedFrames)
    XCTAssertEqual(timeline.settings.width, 576)
    XCTAssertEqual(timeline.settings.fps, 24)
    XCTAssertEqual(timeline.settings.seed, 4242)
    XCTAssertEqual(timeline.settings.steps, 10)
    XCTAssertEqual(timeline.settings.negativePrompt, "blurry")
    XCTAssertEqual(timeline.settings.preset, "kira-video-apple")

    let spans = p.segmentSpans()
    XCTAssertEqual(timeline.promptSegments.count, spans.count)
    XCTAssertEqual(timeline.promptSegments.map(\.startFrame), spans.map(\.start))
    XCTAssertEqual(timeline.promptSegments.map(\.lengthFrames), spans.map(\.length))
  }

  func testProseIsTrimmedAndOrdered() {
    let timeline = SequenceDrafter.assemble(
      prose: prose(), preset: preset(),
      request: SequenceDraftRequest(presetId: "apple-30", brief: "x"))
    XCTAssertEqual(timeline.globalPrompt, "A woman on a stool by a window, late light.")
    XCTAssertEqual(timeline.promptSegments.map(\.prompt), ["beat 1", "beat 2", "beat 3"])
  }

  func testTooFewSegmentsReuseTheLastRatherThanLeavingAGap() {
    let p = preset()
    let timeline = SequenceDrafter.assemble(
      prose: prose(segments: 1), preset: p,
      request: SequenceDraftRequest(presetId: p.id, brief: "x"))
    XCTAssertEqual(timeline.promptSegments.count, p.segmentSpans().count)
    XCTAssertTrue(timeline.promptSegments.allSatisfy { !$0.prompt.isEmpty })
  }

  func testKeyframesArePlacedOnlyWhereTheCallerSuppliedAnImage() {
    let p = preset(keyframes: "fflf")
    let none = SequenceDrafter.assemble(
      prose: prose(), preset: p, request: SequenceDraftRequest(presetId: p.id, brief: "x"))
    XCTAssertTrue(none.keyframes.isEmpty, "no image, no keyframe — never a dangling path")

    let one = SequenceDrafter.assemble(
      prose: prose(), preset: p,
      request: SequenceDraftRequest(presetId: p.id, brief: "x", keyframePaths: ["/img/a.png"]))
    XCTAssertEqual(one.keyframes.count, 1)
    XCTAssertEqual(one.keyframes.first?.frame, 0)
    XCTAssertEqual(one.keyframes.first?.imagePath, "/img/a.png")
  }

  func testAVoiceTrackBecomesAnImportedAudioClipSpanningTheClip() {
    let p = preset(audio: "driven")
    let timeline = SequenceDrafter.assemble(
      prose: prose(), preset: p,
      request: SequenceDraftRequest(
        presetId: p.id, brief: "x", keyframePaths: ["/img/a.png"], audioPath: "/audio/voice.wav"))
    XCTAssertEqual(timeline.audio.mode, .imported)
    XCTAssertEqual(timeline.audioClips.first?.audioPath, "/audio/voice.wav")
    XCTAssertEqual(timeline.audioClips.first?.lengthFrames, p.snappedFrames)
  }

  func testAnAssembledDraftPassesTheValidator() {
    let p = preset()
    let timeline = SequenceDrafter.assemble(
      prose: prose(), preset: p, request: SequenceDraftRequest(presetId: p.id, brief: "x"))
    let validation = DirectorValidator.validate(timeline, fileExists: { _ in true })
    XCTAssertTrue(
      validation.ok,
      "the drafter must not produce timelines the compiler refuses: \(validation.issues.map(\.code))")
  }

  // MARK: the template

  func testSystemPromptStatesTheExactCountsAndForbidsNumbers() {
    let text = SequenceDrafter.systemPrompt(preset: preset(), segmentCount: 3, keyframeSlots: 2)
    XCTAssertTrue(text.contains("EXACTLY 3 line(s)"))
    XCTAssertTrue(text.contains("EXACTLY 2 line(s)"))
    XCTAssertTrue(text.contains("The engine owns every frame count."))
    XCTAssertTrue(text.contains("No meta language"))
  }

  func testHouseStyleAndAudioDrivenNotesAreCarried() {
    let styled = SequenceDrafter.systemPrompt(
      preset: preset(audio: "driven", notes: "static camera, no zooms"),
      segmentCount: 1, keyframeSlots: 0)
    XCTAssertTrue(styled.contains("static camera, no zooms"))
    XCTAssertTrue(styled.contains("the voice already exists"))
    XCTAssertFalse(
      SequenceDrafter.systemPrompt(preset: preset(), segmentCount: 1, keyframeSlots: 0)
        .contains("the voice already exists"))
  }

  func testUserPromptDescribesTheBeatsInSeconds() {
    let p = preset()
    let text = SequenceDrafter.userPrompt(
      brief: "she talks about her day", preset: p,
      segmentSeconds: p.segmentSpans().map { Double($0.length) / 24.0 })
    XCTAssertTrue(text.contains("she talks about her day"))
    XCTAssertTrue(text.contains("beats"))
    XCTAssertTrue(text.contains("30.0s"))
  }

  func testRepairPromptCarriesTheValidatorsOwnWords() {
    let issues = [
      DirectorIssue(severity: .error, code: "keyframe_off_grid", message: "frame 100 is not a multiple of 8", ids: ["k1"])
    ]
    let text = SequenceDrafter.repairPrompt(issues: issues)
    XCTAssertTrue(text.contains("keyframe_off_grid"))
    XCTAssertTrue(text.contains("frame 100 is not a multiple of 8"))
    XCTAssertTrue(text.contains("Keep the same number of segments"))
  }

  // MARK: parsing the answer

  func testParseAcceptsPlainJSON() throws {
    let parsed = try SequenceDrafter.parse(
      #"{"global_prompt":"a room","segments":["one","two"]}"#)
    XCTAssertEqual(parsed.globalPrompt, "a room")
    XCTAssertEqual(parsed.segments, ["one", "two"])
  }

  func testParseAcceptsFencedJSONAndSurroundingChatter() throws {
    let content = """
      Sure! Here's the timeline:

      ```json
      {"global_prompt": "a room", "segments": ["one"], "keyframe_descriptions": ["a still"]}
      ```

      Let me know if you want it warmer.
      """
    let parsed = try SequenceDrafter.parse(content)
    XCTAssertEqual(parsed.globalPrompt, "a room")
    XCTAssertEqual(parsed.keyframeDescriptions, ["a still"])
  }

  func testParseRefusesAnswersWithNoTimelineInThem() {
    XCTAssertThrowsError(try SequenceDrafter.parse("I can't do that."))
    XCTAssertThrowsError(try SequenceDrafter.parse(#"{"global_prompt":"","segments":[]}"#))
    XCTAssertThrowsError(try SequenceDrafter.parse(#"{"segments":["one"]}"#))
  }
}
