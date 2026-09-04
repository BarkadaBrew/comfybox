// WarmServerBeatScheduleEnhancementTests.swift — pins comfybox#328: a
// `beat_schedule` must survive server-side prompt enhancement.
//
// Root cause: `beat_schedule` beats are located as VERBATIM substrings of
// the composed prompt (LTX2BeatScheduleLocator, matched against the exact
// text handed to the tokenizer). WarmServer's video prep
// (`prepareLocalVideo`, WarmServer.swift ~2367-2389) used to overwrite
// `effectivePrompt` with the LLM-rewritten prompt from the dolphin
// prompt-optimizer BEFORE beats are ever located downstream in
// `LTX2VideoGenerator` (~952-964) — the rewrite shares no verbatim
// substrings with the caller's original beat text, so every beat drops
// (fail-open, one warning each) and the feature is silently dead whenever
// enhancement is on (which is the default: `req.enhance != false`).
//
// Fix: beats are the more specific ask — when a request carries a
// non-empty `beat_schedule`, prompt enhancement is skipped outright
// (logged loudly) rather than left to neutralize the schedule. This test
// pins the pure decision function `WarmServer.beatScheduleEnhancementSkip`
// that WarmServer's video-prep path now consults before calling the
// optimizer.

import XCTest
@testable import ZImage

final class WarmServerBeatScheduleEnhancementTests: XCTestCase {

  private func beat(_ text: String = "she walks closer") -> BeatSegment {
    BeatSegment(text: text, startFrac: 0, endFrac: 1)
  }

  func testNonEmptyBeatScheduleSkipsEnhancement() {
    XCTAssertEqual(
      WarmServer.beatScheduleEnhancementSkip(beatSchedule: [beat()]),
      "beat_schedule",
      "a request carrying beats must skip enhancement so the beat text survives verbatim in the composed prompt")
  }

  func testNonEmptyBeatScheduleSkipsEnhancementEvenWhenEnhanceRequestedExplicitly() {
    // Beats win regardless of `enhance` — the caller asking for both at once
    // is exactly the scenario #328 reports; beats are the more specific ask.
    XCTAssertEqual(
      WarmServer.beatScheduleEnhancementSkip(beatSchedule: [beat(), beat("hips sway slowly")]),
      "beat_schedule")
  }

  func testEmptyBeatScheduleDoesNotSkipEnhancement() {
    XCTAssertNil(WarmServer.beatScheduleEnhancementSkip(beatSchedule: []))
  }

  func testNilBeatScheduleDoesNotSkipEnhancement() {
    XCTAssertNil(WarmServer.beatScheduleEnhancementSkip(beatSchedule: nil))
  }
}
