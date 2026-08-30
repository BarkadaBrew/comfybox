// LTX2BeatScheduleLocatorTests.swift — token-range locate coverage for the
// temporal beat scheduling locator (comfybox#310, adversarial review F12(b)).
// Pure Swift (no MLX tensors, no real tokenizer): the tokenize closure is a
// word -> id table, which makes every fixture hand-checkable.

import XCTest
@testable import ZImage

final class LTX2BeatScheduleLocatorTests: XCTestCase {

  /// word -> token id: stable, whitespace-split, unknown words get fresh ids.
  private func makeTokenizer() -> (String) -> [Int] {
    var table: [String: Int] = [:]
    var next = 100
    return { text in
      text.split(separator: " ").map { word in
        let w = String(word)
        if let id = table[w] { return id }
        table[w] = next
        next += 1
        return next - 1
      }
    }
  }

  private func beat(_ text: String, _ s: Float, _ e: Float) -> BeatSegment {
    BeatSegment(text: text, startFrac: s, endFrac: e)
  }

  // MARK: left-pad offset

  func testLeftPadOffsetShiftsTokenRanges() {
    let tokenize = makeTokenizer()
    let full = tokenize("she walks closer she sways slowly")  // 6 tokens
    let maxLength = 16  // padOffset = 16 - 6 = 10

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("she walks closer", 0, 0.5), beat("she sways slowly", 0.5, 1)],
      fullPromptTokenIds: full, maxLength: maxLength, tokenize: tokenize)

    XCTAssertEqual(resolved.count, 2)
    // Beat 1 sits at local 0..<3 → padded 10..<13; beat 2 local 3..<6 → 13..<16.
    XCTAssertEqual(resolved[0].tokenStart, 10)
    XCTAssertEqual(resolved[0].tokenEnd, 13)
    XCTAssertEqual(resolved[1].tokenStart, 13)
    XCTAssertEqual(resolved[1].tokenEnd, 16)
  }

  // MARK: monotonic left-to-right re-match

  func testRepeatedBeatTextMatchesSecondOccurrenceForSecondBeat() {
    let tokenize = makeTokenizer()
    // "hips sway" appears twice; the second beat must land on the SECOND
    // occurrence because the search cursor advances past the first match.
    let full = tokenize("hips sway then hips sway again")  // 6 tokens
    let maxLength = 6  // no pad, ranges are local

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("hips sway", 0, 0.5), beat("hips sway", 0.5, 1)],
      fullPromptTokenIds: full, maxLength: maxLength, tokenize: tokenize)

    XCTAssertEqual(resolved.count, 2)
    XCTAssertEqual(resolved[0].tokenStart, 0)
    XCTAssertEqual(resolved[0].tokenEnd, 2)
    XCTAssertEqual(resolved[1].tokenStart, 3, "second identical beat must re-match AFTER the first")
    XCTAssertEqual(resolved[1].tokenEnd, 5)
  }

  // MARK: per-beat fail-open

  func testUnlocatableBeatDropsWithoutFailingOthers() {
    let tokenize = makeTokenizer()
    let full = tokenize("she walks closer she sways slowly")
    var dropped: [(String, String)] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [
        beat("she walks closer", 0, 0.33),
        beat("totally absent phrase", 0.33, 0.66),  // never tokenized into `full`
        beat("she sways slowly", 0.66, 1),
      ],
      fullPromptTokenIds: full, maxLength: 6,
      onDrop: { b, reason in dropped.append((b.text, reason)) },
      tokenize: tokenize)

    XCTAssertEqual(resolved.count, 2, "the two locatable beats must survive")
    XCTAssertEqual(resolved[0].tokenStart, 0)
    XCTAssertEqual(resolved[1].tokenStart, 3)
    XCTAssertEqual(dropped.count, 1, "exactly one drop logged")
    XCTAssertEqual(dropped[0].0, "totally absent phrase")
    XCTAssertTrue(dropped[0].1.contains("could not locate"))
  }

  // MARK: truncation-window clamp

  func testTruncationClampsStraddlingBeatAndDropsFullyOutsideBeat() {
    let tokenize = makeTokenizer()
    let full = tokenize("a b c d e f g h")  // 8 tokens
    var dropped: [String] = []

    // maxLength 4 < count 8 → padOffset 0, truncatedLen 4. Beat "c d e f"
    // (local 2..<6) straddles the window → clamped to 2..<4. Beat "g h"
    // (local 6..<8) is fully outside → dropped.
    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("c d e f", 0, 0.5), beat("g h", 0.5, 1)],
      fullPromptTokenIds: full, maxLength: 4,
      onDrop: { b, _ in dropped.append(b.text) },
      tokenize: tokenize)

    XCTAssertEqual(resolved.count, 1)
    XCTAssertEqual(resolved[0].tokenStart, 2)
    XCTAssertEqual(resolved[0].tokenEnd, 4, "span must clamp at the truncation window")
    XCTAssertEqual(dropped, ["g h"], "a beat entirely past the window must drop fail-open")
  }

  // MARK: frac hygiene (F10)

  func testFracHygieneClampsOutOfRangeAndDropsDegenerateSpans() {
    let tokenize = makeTokenizer()
    let full = tokenize("one two three four")
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [
        beat("one two", -0.5, 0.5),          // clamps to [0, 0.5]
        beat("three", 0.9, 0.4),             // endFrac <= startFrac → drop
        BeatSegment(text: "four", startFrac: Float.nan, endFrac: 1),  // non-finite → drop
      ],
      fullPromptTokenIds: full, maxLength: 4,
      onDrop: { b, _ in dropped.append(b.text) },
      tokenize: tokenize)

    XCTAssertEqual(resolved.count, 1)
    XCTAssertEqual(resolved[0].startFrac, 0, "negative startFrac must clamp to 0")
    XCTAssertEqual(resolved[0].endFrac, 0.5)
    XCTAssertEqual(dropped.sorted(), ["four", "three"],
                   "degenerate and non-finite spans must drop fail-open, not fail the render")
  }
}
