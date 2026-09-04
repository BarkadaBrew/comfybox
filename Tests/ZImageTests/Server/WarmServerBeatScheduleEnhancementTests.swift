// WarmServerBeatScheduleEnhancementTests.swift — pins comfybox#328: a
// `beat_schedule` must survive server-side prompt enhancement.
//
// Root cause: `beat_schedule` beats are located as VERBATIM substrings of
// the composed prompt (LTX2BeatScheduleLocator, matched against the exact
// text handed to the tokenizer). WarmServer's video prep used to overwrite
// `effectivePrompt` with the LLM-rewritten prompt from the dolphin
// prompt-optimizer BEFORE beats are ever located downstream in
// `LTX2VideoGenerator` — the rewrite shares no verbatim substrings with the
// caller's original beat text, so every beat drops (fail-open, one warning
// each) and the feature is silently dead whenever enhancement is on (the
// default: `req.enhance != false`). A separate, now-fixed tokenizer bug
// (`LTX2BeatScheduleLocatorRealTokenizerTests`) produced the identical 0/4
// symptom even with `enhance:false` — this suite is only about the
// enhancement interaction, not that bug.
//
// Fix: beats are the more specific ask — a non-empty, ACTIVE (kill-switch
// on) `beat_schedule` skips enhancement outright, but ONLY when enhancement
// would otherwise actually have run (`enhance != false` AND a provider is
// configured) — so the `enhancement_skipped` marker recorded on the
// response/trace is never a false claim (Codex round 1, finding 3).
//
// `WarmServer.resolveVideoEnhancement` is the pure(ish) decision + effect
// WarmServer's video prep consults; its ONLY side effect is the injected
// `optimize` closure, so these tests use a spy closure to prove — without
// any network call, model weights, or running server — that the optimizer
// is (or is not) invoked, and that the composed prompt retains every beat's
// exact text when it is not.

import Logging
import XCTest
@testable import ZImage

final class WarmServerBeatScheduleEnhancementTests: XCTestCase {

  private let logger = Logger(label: "test.beat-enhancement")
  private let endpoint = AIProviderEndpoint(baseUrl: "http://localhost:1234/v1", model: "dolphin3.0")

  private func beat(_ text: String = "she walks closer") -> BeatSegment {
    BeatSegment(text: text, startFrac: 0, endFrac: 1)
  }

  /// A prompt containing beat phrasing the enhancement rewrite (if it ran)
  /// would NOT preserve verbatim — matching the production dolphin rewrite
  /// #328 reports.
  private let promptWithBeats = "she walks closer. hips sway slowly. she turns toward camera."

  /// Records every call; returns a rewrite that would fail beat-locate if it
  /// were ever used as `effectivePrompt` (proves the optimizer path was, or
  /// wasn't, actually taken — not just that some field says so).
  private final class OptimizeSpy: @unchecked Sendable {
    private(set) var callCount = 0
    func call(
      endpoint: AIProviderEndpoint, prompt: String, character: String?,
      characterDescription: String?, contentMode: String, mediaKind: String
    ) async -> OptimizeResult {
      callCount += 1
      return OptimizeResult(prompt: "a completely rewritten scene with none of the original wording", enhanced: true, note: nil)
    }
  }

  // MARK: - Finding 1: the optimizer must not run, and beats must survive

  func testBeatScheduleOmittedEnhanceSkipsOptimizerAndRetainsEveryBeat() async {
    let spy = OptimizeSpy()
    let outcome = await WarmServer.resolveVideoEnhancement(
      prompt: promptWithBeats, enhance: nil, beatSchedule: [beat("she walks closer"), beat("hips sway slowly")],
      beatScheduleEnabled: true, characterName: nil, characterDesc: nil,
      contentMode: "neutral", mediaKind: "video", optimizerEndpoint: endpoint,
      logger: logger, optimize: spy.call)

    XCTAssertEqual(spy.callCount, 0, "optimizer must not be invoked when beats are present")
    XCTAssertEqual(outcome.effectivePrompt, promptWithBeats, "every beat's exact text must survive verbatim")
    XCTAssertFalse(outcome.enhancedApplied)
    XCTAssertEqual(outcome.enhancementSkippedReason, "beat_schedule")
  }

  func testBeatScheduleWithEnhanceExplicitlyTrueStillSkipsOptimizer() async {
    // The scenario #328 reports verbatim: caller asks for both at once.
    let spy = OptimizeSpy()
    let outcome = await WarmServer.resolveVideoEnhancement(
      prompt: promptWithBeats, enhance: true, beatSchedule: [beat()],
      beatScheduleEnabled: true, characterName: nil, characterDesc: nil,
      contentMode: "neutral", mediaKind: "video", optimizerEndpoint: endpoint,
      logger: logger, optimize: spy.call)

    XCTAssertEqual(spy.callCount, 0, "beats win even when enhance:true is explicit")
    XCTAssertEqual(outcome.effectivePrompt, promptWithBeats)
    XCTAssertEqual(outcome.enhancementSkippedReason, "beat_schedule")
  }

  // MARK: - No beat_schedule: enhancement runs normally

  func testNoBeatScheduleInvokesTheOptimizer() async {
    let spy = OptimizeSpy()
    let outcome = await WarmServer.resolveVideoEnhancement(
      prompt: "a plain scene", enhance: nil, beatSchedule: nil,
      beatScheduleEnabled: true, characterName: nil, characterDesc: nil,
      contentMode: "neutral", mediaKind: "video", optimizerEndpoint: endpoint,
      logger: logger, optimize: spy.call)

    XCTAssertEqual(spy.callCount, 1, "no beats — enhancement must run as before")
    XCTAssertTrue(outcome.enhancedApplied)
    XCTAssertNil(outcome.enhancementSkippedReason)
  }

  func testEmptyBeatScheduleInvokesTheOptimizer() async {
    let spy = OptimizeSpy()
    let outcome = await WarmServer.resolveVideoEnhancement(
      prompt: "a plain scene", enhance: nil, beatSchedule: [],
      beatScheduleEnabled: true, characterName: nil, characterDesc: nil,
      contentMode: "neutral", mediaKind: "video", optimizerEndpoint: endpoint,
      logger: logger, optimize: spy.call)

    XCTAssertEqual(spy.callCount, 1)
    XCTAssertNil(outcome.enhancementSkippedReason)
  }

  // MARK: - Finding 3: the marker must be truthful — never claim a skip that
  // caused nothing, i.e. enhancement wouldn't have run anyway.

  func testEnhanceFalseWithBeatsSkipsOptimizerButRecordsNoBeatMarker() async {
    // enhance:false already means "don't optimize" — beats aren't why the
    // optimizer didn't run, so the marker must not claim they are.
    let spy = OptimizeSpy()
    let outcome = await WarmServer.resolveVideoEnhancement(
      prompt: promptWithBeats, enhance: false, beatSchedule: [beat()],
      beatScheduleEnabled: true, characterName: nil, characterDesc: nil,
      contentMode: "neutral", mediaKind: "video", optimizerEndpoint: endpoint,
      logger: logger, optimize: spy.call)

    XCTAssertEqual(spy.callCount, 0)
    XCTAssertNil(outcome.enhancementSkippedReason, "enhance:false is the actual cause, not beat_schedule")
  }

  func testNoOptimizerConfiguredWithBeatsRecordsNoBeatMarker() async {
    // No provider configured — enhancement was never going to run, so beats
    // did not cause anything to be skipped.
    let spy = OptimizeSpy()
    let outcome = await WarmServer.resolveVideoEnhancement(
      prompt: promptWithBeats, enhance: nil, beatSchedule: [beat()],
      beatScheduleEnabled: true, characterName: nil, characterDesc: nil,
      contentMode: "neutral", mediaKind: "video", optimizerEndpoint: nil,
      logger: logger, optimize: spy.call)

    XCTAssertEqual(spy.callCount, 0)
    XCTAssertNil(outcome.enhancementSkippedReason)
  }

  func testBeatKillSwitchOffWithBeatsRunsEnhancementNormally() async {
    // LTX2_BEAT_SCHEDULE=0: beats are already inert engine-wide, so there is
    // no reason to sacrifice enhancement for them — and no truthful marker
    // to record, since nothing beat-related was skipped.
    let spy = OptimizeSpy()
    let outcome = await WarmServer.resolveVideoEnhancement(
      prompt: promptWithBeats, enhance: nil, beatSchedule: [beat()],
      beatScheduleEnabled: false, characterName: nil, characterDesc: nil,
      contentMode: "neutral", mediaKind: "video", optimizerEndpoint: endpoint,
      logger: logger, optimize: spy.call)

    XCTAssertEqual(spy.callCount, 1, "kill switch off — beats are inert, enhancement proceeds")
    XCTAssertTrue(outcome.enhancedApplied)
    XCTAssertNil(outcome.enhancementSkippedReason)
  }
}
