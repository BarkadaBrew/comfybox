// LTX2BeatScheduleLocatorLazyPrefixTests.swift — comfybox#340.
//
// #340 reported every LTX-2 render wedging at "loading text encoder (Gemma 3
// 12B)…" and named `standalonePrefix = tokenize("")` (added by the #335 /
// PR #338 beat-locate fix) as the prime suspect, on the theory that it ran
// during encoder construction.
//
// It does not: `LTX2BeatScheduleLocator.locate` is called from exactly one
// place — the `resolvedBeats` binding in `LTX2VideoGenerator.render`, gated on
// `typedConfig.beatScheduleEnabled && request.beatSchedule` being non-empty —
// and that is reached long after `LTX2VideoGenerator.load` logs
// "LTX-2: models ready." (Deliberately named by symbol, not line: Codex r1
// caught this comment's line numbers already stale one round in.) See the PR
// for the production-log evidence.
//
// What these tests DO pin is the invariant the ticket asks for, at the one
// layer that can be tested without weights: the locator performs NO tokenizer
// work at all unless there is a beat to place, and the standalone prefix is
// computed lazily and at most once per call. A counting/hostile stub stands in
// for the tokenizer, so a future refactor that hoists tokenizer work to a
// caller-independent position fails here instead of in production.

import XCTest

@testable import ZImage

final class LTX2BeatScheduleLocatorLazyPrefixTests: XCTestCase {

  /// Counting tokenizer: records every string it is handed, so a test can
  /// assert not just how many calls happened but exactly which ones.
  private final class RecordingTokenizer {
    private(set) var calls: [String] = []
    private var table: [String: Int] = [:]
    private var next = 100

    /// `""` encodes to the standalone special-token prefix alone (Gemma: BOS
    /// id 2), like the real Gemma 3 tokenizer.
    func tokenize(_ text: String) -> [Int] {
      calls.append(text)
      var ids = [2]
      for word in text.split(separator: " ") {
        let w = String(word)
        if let id = table[w] {
          ids.append(id)
        } else {
          table[w] = next
          ids.append(next)
          next += 1
        }
      }
      return ids
    }

    var emptyStringCalls: Int { calls.filter { $0.isEmpty }.count }
  }

  private func beat(_ text: String, _ s: Float, _ e: Float) -> BeatSegment {
    BeatSegment(text: text, startFrac: s, endFrac: e)
  }

  /// The #340 invariant: a render that did NOT ask for a beat schedule must
  /// not touch the tokenizer through this path at all — not even once, not
  /// even for the empty string. If `tokenize("")` ever spins or blocks in a
  /// production tokenizer, a beat-less render must never be able to reach it.
  func testNoBeatsPerformsNoTokenizerWorkAtAll() {
    let tok = RecordingTokenizer()
    let full = tok.tokenize("she walks slowly toward the counter")
    XCTAssertEqual(tok.calls.count, 1, "only the fixture's own prompt encode so far")

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [],
      fullPromptTokenIds: full,
      maxLength: 32,
      tokenize: { tok.tokenize($0) })

    XCTAssertEqual(resolved, [])
    XCTAssertEqual(
      tok.calls.count, 1,
      "locate([]) must not call the tokenizer at all (got: \(tok.calls.dropFirst()))")
    XCTAssertEqual(tok.emptyStringCalls, 0, "the empty-string encode must not run with no beats")
  }

  /// Same invariant one layer in: beats that never reach tokenization (frac
  /// hygiene drops them first) must not trigger the standalone-prefix encode
  /// either. Otherwise "has a beat_schedule field" — rather than "has a beat
  /// worth placing" — is what decides whether the tokenizer runs.
  func testBeatsDroppedByFracHygieneNeverReachTheTokenizer() {
    let tok = RecordingTokenizer()
    let full = tok.tokenize("she walks slowly toward the counter")
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [
        beat("she walks", .nan, 1),
        beat("toward the counter", 0.5, 0.5),
      ],
      fullPromptTokenIds: full,
      maxLength: 32,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: { tok.tokenize($0) })

    XCTAssertEqual(resolved, [])
    XCTAssertEqual(dropped.count, 2, "both beats fail-open before any tokenization")
    XCTAssertEqual(
      tok.calls.count, 1,
      "no beat survived frac hygiene, so nothing may be tokenized (got: \(tok.calls.dropFirst()))")
    XCTAssertEqual(tok.emptyStringCalls, 0)
  }

  /// A hostile tokenizer that traps on `""` proves the empty-string encode is
  /// genuinely unreachable for a beat-less locate, rather than merely counted.
  func testEmptyStringEncodeIsUnreachableWithoutBeats() {
    let full = [2, 100, 101, 102]

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [],
      fullPromptTokenIds: full,
      maxLength: 8,
      tokenize: { text in
        XCTAssertFalse(
          text.isEmpty, "tokenize(\"\") must never be reached when there are no beats")
        return [2]
      })

    XCTAssertEqual(resolved, [])
  }

  /// With beats present the prefix IS needed — but exactly once per call, no
  /// matter how many beats there are. (Pins the memoization: the fix must not
  /// turn one encode per call into one per beat.)
  func testStandalonePrefixIsComputedExactlyOnceAcrossManyBeats() {
    let tok = RecordingTokenizer()
    let full = tok.tokenize("she walks closer she sways slowly she turns away")
    let beats = [
      beat("she walks closer", 0, 0.3),
      beat("she sways slowly", 0.3, 0.6),
      beat("she turns away", 0.6, 1),
    ]
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: beats,
      fullPromptTokenIds: full,
      maxLength: 16,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: { tok.tokenize($0) })

    XCTAssertEqual(dropped, [], "verbatim beats must still locate (comfybox#335 unchanged)")
    XCTAssertEqual(resolved.count, 3)
    XCTAssertEqual(
      tok.emptyStringCalls, 1,
      "the standalone prefix must be computed once per locate call, not once per beat")
  }

  /// The prefix encode must be cheap to get wrong safely: a tokenizer that
  /// returns nothing for `""` (no special tokens at all) still locates, and a
  /// beat whose text is nothing BUT the prefix still fail-opens rather than
  /// matching a zero-length run anywhere.
  func testEmptyPrefixAndPrefixOnlyBeatBothStayFailOpen() {
    var table: [String: Int] = [:]
    var next = 100
    let tokenize: (String) -> [Int] = { text in
      text.split(separator: " ").map { word in
        let w = String(word)
        if let id = table[w] { return id }
        table[w] = next
        next += 1
        return next - 1
      }
    }
    let full = tokenize("she walks closer")
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("   ", 0, 0.5), beat("she walks closer", 0.5, 1)],
      fullPromptTokenIds: full,
      maxLength: 8,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: tokenize)

    XCTAssertEqual(resolved.count, 1, "the locatable beat survives")
    XCTAssertEqual(dropped.count, 1)
    guard dropped.count == 1 else { return }
    XCTAssertTrue(
      dropped[0].contains("no tokens"),
      "a whitespace-only beat drops as token-less, never matches an empty run: \(dropped[0])")
  }
}
