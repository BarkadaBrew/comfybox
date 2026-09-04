import XCTest

@testable import ZImage

/// Winner actions (2026-08-10): the scheduler standardizes on 480p/4s and
/// gallery winners get improved post-hoc — a 720p re-render of the exact same
/// clip, or a 4s continuation chained from the last frame. These tests pin
/// the pure JSON transforms that make a stored trace replayable.
final class VideoWinnerActionsTests: XCTestCase {

  private func obj(_ data: Data) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
  }

  private let originalBody: [String: Any] = [
    "prompt": "raw daemon prompt",
    "image_path": "/tmp/seed.png",
    "width": 480, "height": 832,
    "frames": 97, "fps": 24, "seed": 4242,
    "preset": "kira-video-avocado",
    "audio": true,
    "output_path": "/Users/toddwalderman/Pictures/ComfyBox/orig.mp4",
    "source": "api",
  ]

  private func originalJSON() throws -> String {
    let data = try JSONSerialization.data(withJSONObject: originalBody)
    return try XCTUnwrap(String(data: data, encoding: .utf8))
  }

  // MARK: - sanitizedRequestJSON (trace storage)

  func testSanitizedRequestJSONDropsInlineImageBytes() throws {
    var body = originalBody
    body["image_base64"] = String(repeating: "A", count: 4096)
    let data = try JSONSerialization.data(withJSONObject: body)
    let json = try XCTUnwrap(VideoWinnerActions.sanitizedRequestJSON(fromBody: data))
    let round = try obj(Data(json.utf8))
    XCTAssertNil(round["image_base64"])
    XCTAssertEqual(round["prompt"] as? String, "raw daemon prompt")
    XCTAssertEqual(round["seed"] as? Int, 4242)
  }

  func testSanitizedRequestJSONRejectsNonObjectBodies() {
    XCTAssertNil(VideoWinnerActions.sanitizedRequestJSON(fromBody: Data("[]".utf8)))
    XCTAssertNil(VideoWinnerActions.sanitizedRequestJSON(fromBody: Data("nonsense".utf8)))
  }

  // MARK: - rerenderBody (720p winner replay)

  func testRerenderBodySwapsResolutionAndDropsExplicitDims() throws {
    let data = try VideoWinnerActions.rerenderBody(
      requestJSON: originalJSON(), resolvedSeed: "4242",
      effectivePrompt: "the effective prompt", resolution: "720p")
    let body = try obj(data)
    XCTAssertEqual(body["resolution"] as? String, "720p")
    // Explicit dims would pin the OLD budget — the named budget must win.
    XCTAssertNil(body["width"])
    XCTAssertNil(body["height"])
  }

  func testRerenderBodyPinsResolvedSeedAndEffectivePrompt() throws {
    // Same seed + already-enhanced prompt = the same clip, larger.
    let data = try VideoWinnerActions.rerenderBody(
      requestJSON: originalJSON(), resolvedSeed: "777",
      effectivePrompt: "the effective prompt", resolution: "720p")
    let body = try obj(data)
    XCTAssertEqual(body["seed"] as? Int, 777)
    XCTAssertEqual(body["prompt"] as? String, "the effective prompt")
    XCTAssertEqual(body["enhance"] as? Bool, false)
    XCTAssertEqual(body["skip_character_injection"] as? Bool, true)
  }

  func testRerenderBodyKeepsOriginalSeedAndPromptWhenTraceHasNone() throws {
    let data = try VideoWinnerActions.rerenderBody(
      requestJSON: originalJSON(), resolvedSeed: nil,
      effectivePrompt: nil, resolution: "720p")
    let body = try obj(data)
    XCTAssertEqual(body["seed"] as? Int, 4242)
    XCTAssertEqual(body["prompt"] as? String, "raw daemon prompt")
  }

  func testRerenderBodyMintsFreshOutputAndTagsSource() throws {
    let data = try VideoWinnerActions.rerenderBody(
      requestJSON: originalJSON(), resolvedSeed: "4242",
      effectivePrompt: "p", resolution: "720p")
    let body = try obj(data)
    XCTAssertNil(body["output_path"])
    XCTAssertEqual(body["source"] as? String, "winner-rerender")
    // Everything that shaped the winner stays.
    XCTAssertEqual(body["preset"] as? String, "kira-video-avocado")
    XCTAssertEqual(body["frames"] as? Int, 97)
    XCTAssertEqual(body["audio"] as? Bool, true)
    XCTAssertEqual(body["image_path"] as? String, "/tmp/seed.png")
  }

  func testRerenderBodyRestoresInitImageStrippedFromBase64Submissions() throws {
    // An i2v winner submitted via image_base64 has no image_path in the
    // sanitized request — without the trace's resolved init image the replay
    // silently flips to t2v (review finding 1).
    var original = originalBody
    original.removeValue(forKey: "image_path")
    let json = String(
      data: try JSONSerialization.data(withJSONObject: original), encoding: .utf8)!
    let body = try obj(
      try VideoWinnerActions.rerenderBody(
        requestJSON: json, resolvedSeed: "1", effectivePrompt: "p",
        resolution: "720p", initImagePath: "/tmp/resolved-init.png"))
    XCTAssertEqual(body["image_path"] as? String, "/tmp/resolved-init.png")
  }

  func testRerenderBodyPrefersTheRequestsOwnImagePath() throws {
    let body = try obj(
      try VideoWinnerActions.rerenderBody(
        requestJSON: originalJSON(), resolvedSeed: "1", effectivePrompt: "p",
        resolution: "720p", initImagePath: "/tmp/resolved-init.png"))
    XCTAssertEqual(body["image_path"] as? String, "/tmp/seed.png")
  }

  func testRerenderBodySuppressesPresetPromptRewrap() throws {
    // The stored effective prompt already carries the preset prefix/suffix —
    // re-wrapping would condition on "prefix, prefix, …" (review finding 2).
    let body = try obj(
      try VideoWinnerActions.rerenderBody(
        requestJSON: originalJSON(), resolvedSeed: "1", effectivePrompt: "p",
        resolution: "720p"))
    XCTAssertEqual(body["skip_preset_prompt"] as? Bool, true)
  }

  func testRerenderBodySynthesizesAspectRatioFromStoredDims() throws {
    // Portrait winner (480x832) with no aspect_ratio key: stripping dims
    // without recording orientation replays landscape (review finding 3).
    let portrait = try obj(
      try VideoWinnerActions.rerenderBody(
        requestJSON: originalJSON(), resolvedSeed: "1", effectivePrompt: "p",
        resolution: "720p"))
    XCTAssertEqual(portrait["aspect_ratio"] as? String, "9:16")

    var landscapeBody = originalBody
    landscapeBody["width"] = 832
    landscapeBody["height"] = 480
    let json = String(
      data: try JSONSerialization.data(withJSONObject: landscapeBody), encoding: .utf8)!
    let landscape = try obj(
      try VideoWinnerActions.rerenderBody(
        requestJSON: json, resolvedSeed: "1", effectivePrompt: "p", resolution: "720p"))
    XCTAssertEqual(landscape["aspect_ratio"] as? String, "16:9")
  }

  func testRerenderBodyKeepsAnExplicitAspectRatio() throws {
    var original = originalBody
    original["aspect_ratio"] = "16:9"  // explicit wins over dims inference
    let json = String(
      data: try JSONSerialization.data(withJSONObject: original), encoding: .utf8)!
    let body = try obj(
      try VideoWinnerActions.rerenderBody(
        requestJSON: json, resolvedSeed: "1", effectivePrompt: "p", resolution: "720p"))
    XCTAssertEqual(body["aspect_ratio"] as? String, "16:9")
  }

  func testRerenderBodyRejectsGarbage() {
    XCTAssertThrowsError(try VideoWinnerActions.rerenderBody(
      requestJSON: "not json", resolvedSeed: nil, effectivePrompt: nil, resolution: "720p"))
  }

  // MARK: - extendBody (last-frame continuation)

  func testExtendBodyChainsFromExtractedFrameAtStandardLength() throws {
    let data = try VideoWinnerActions.extendBody(
      requestJSON: originalJSON(), framePath: "/tmp/last-frame.png",
      seconds: 4, prompt: nil, effectivePrompt: "the effective prompt")
    let body = try obj(data)
    XCTAssertEqual(body["image_path"] as? String, "/tmp/last-frame.png")
    XCTAssertEqual(body["frames"] as? Int, 97)  // 4s @ 24fps → 1+8k
    XCTAssertEqual(body["resolution"] as? String, "480p")
    XCTAssertEqual(body["prompt"] as? String, "the effective prompt")
    XCTAssertEqual(body["source"] as? String, "winner-extend")
    XCTAssertEqual(body["enhance"] as? Bool, false)
    XCTAssertEqual(body["skip_character_injection"] as? Bool, true)
  }

  func testExtendBodySnapsFramesToTrainedGrid() throws {
    for (seconds, frames) in [(4, 97), (5, 121), (8, 193), (12, 289)] {
      let body = try obj(
        try VideoWinnerActions.extendBody(
          requestJSON: originalJSON(), framePath: "/tmp/f.png",
          seconds: seconds, prompt: nil, effectivePrompt: "p"))
      XCTAssertEqual(body["frames"] as? Int, frames, "\(seconds)s")
    }
  }

  func testExtendBodyMintsAFreshSeed() throws {
    // The local video path defaults a missing seed to a CONSTANT (42 or the
    // preset seed) — merely stripping the key is not fresh sampling (review
    // finding 4). The mint is injectable so this stays deterministic.
    let body = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: originalJSON(), framePath: "/tmp/f.png",
        seconds: 4, prompt: nil, effectivePrompt: "p", freshSeed: 987_654))
    XCTAssertEqual(body["seed"] as? Int, 987_654)
    XCTAssertNotEqual(body["seed"] as? Int, 4242)

    // Un-injected, the mint must still replace the winner's seed.
    let minted = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: originalJSON(), framePath: "/tmp/f.png",
        seconds: 4, prompt: nil, effectivePrompt: "p"))
    XCTAssertNotNil(minted["seed"])
    XCTAssertNotEqual(minted["seed"] as? Int, 4242)
  }

  func testExtendBodyPresetWrapFollowsThePromptSource() throws {
    // Stored effective prompt: already wrapped → suppress the re-wrap.
    let stored = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: originalJSON(), framePath: "/tmp/f.png",
        seconds: 4, prompt: nil, effectivePrompt: "p"))
    XCTAssertEqual(stored["skip_preset_prompt"] as? Bool, true)

    // Caller-supplied prompt: raw text → the preset prefix/suffix (trigger
    // words) should compose exactly as they did for the original render.
    let fresh = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: originalJSON(), framePath: "/tmp/f.png",
        seconds: 4, prompt: "she turns away", effectivePrompt: "p"))
    XCTAssertNil(fresh["skip_preset_prompt"])
  }

  func testExtendBodyPromptOverrideWins() throws {
    let data = try VideoWinnerActions.extendBody(
      requestJSON: originalJSON(), framePath: "/tmp/f.png",
      seconds: 4, prompt: "she turns and walks away", effectivePrompt: "old")
    let body = try obj(data)
    XCTAssertEqual(body["prompt"] as? String, "she turns and walks away")
  }

  func testExtendBodyDropsSeedDurationAndDims() throws {
    var original = originalBody
    original["duration"] = 8
    original["extend_to_seconds"] = 8
    let json = String(
      data: try JSONSerialization.data(withJSONObject: original), encoding: .utf8)!
    let body = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: json, framePath: "/tmp/f.png",
        seconds: 4, prompt: nil, effectivePrompt: "p"))
    // A continuation is NEW content — reusing the winner's seed would bias
    // the sampler toward replaying the same motion from the new anchor.
    XCTAssertNotEqual(body["seed"] as? Int, 4242)
    XCTAssertNil(body["duration"])
    XCTAssertNil(body["extend_to_seconds"])
    XCTAssertNil(body["width"])
    XCTAssertNil(body["height"])
    XCTAssertNil(body["output_path"])
  }

  func testExtendBodyWorksWithoutAnOriginalRequest() throws {
    // Pre-upgrade clips have no request_json in the trace — extend must still
    // work from just the frame + a caller prompt.
    let body = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: nil, framePath: "/tmp/f.png",
        seconds: 4, prompt: "she smiles", effectivePrompt: nil))
    XCTAssertEqual(body["image_path"] as? String, "/tmp/f.png")
    XCTAssertEqual(body["prompt"] as? String, "she smiles")
    XCTAssertEqual(body["frames"] as? Int, 97)
  }

  func testExtendBodyRequiresAPrompt() {
    XCTAssertThrowsError(try VideoWinnerActions.extendBody(
      requestJSON: nil, framePath: "/tmp/f.png",
      seconds: 4, prompt: nil, effectivePrompt: nil))
  }

  // MARK: - extendBody beat_schedule (comfybox#328, Codex round 1 finding 6)

  private let originalBodyWithBeats: [String: Any] = [
    "prompt": "raw daemon prompt",
    "image_path": "/tmp/seed.png",
    "width": 480, "height": 832,
    "frames": 97, "fps": 24, "seed": 4242,
    "beat_schedule": [
      ["text": "she walks closer", "start_frac": 0.0, "end_frac": 0.5] as [String: Any]
    ],
    "source": "api",
  ]

  private func originalJSONWithBeats() throws -> String {
    let data = try JSONSerialization.data(withJSONObject: originalBodyWithBeats)
    return try XCTUnwrap(String(data: data, encoding: .utf8))
  }

  func testExtendBodyDropsInheritedBeatScheduleWhenCallerSuppliesAReplacementPrompt() throws {
    // The inherited beats' text is anchored to the OLD prompt's exact
    // wording — a fresh caller prompt invalidates every span.
    let body = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: originalJSONWithBeats(), framePath: "/tmp/f.png",
        seconds: 4, prompt: "she turns and walks away", effectivePrompt: "old"))
    XCTAssertNil(body["beat_schedule"], "a replacement prompt invalidates the inherited beats' spans")
  }

  func testExtendBodyKeepsInheritedBeatScheduleWhenPromptIsUnchanged() throws {
    // No caller-supplied prompt: the effective prompt carries forward
    // unchanged, so the beats' text still matches.
    let body = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: originalJSONWithBeats(), framePath: "/tmp/f.png",
        seconds: 4, prompt: nil, effectivePrompt: "raw daemon prompt"))
    XCTAssertNotNil(body["beat_schedule"], "beats stay valid when the prompt they were anchored to is unchanged")
  }

  func testExtendBodyDropsCamelCaseBeatScheduleVariantToo() throws {
    var original = originalBodyWithBeats
    original.removeValue(forKey: "beat_schedule")
    original["beatSchedule"] = [
      ["text": "she walks closer", "start_frac": 0.0, "end_frac": 0.5] as [String: Any]
    ]
    let json = try XCTUnwrap(
      String(data: try JSONSerialization.data(withJSONObject: original), encoding: .utf8))
    let body = try obj(
      try VideoWinnerActions.extendBody(
        requestJSON: json, framePath: "/tmp/f.png",
        seconds: 4, prompt: "she turns and walks away", effectivePrompt: "old"))
    XCTAssertNil(body["beatSchedule"])
    XCTAssertNil(body["beat_schedule"])
  }
}
