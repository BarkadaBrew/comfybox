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
    guard resolved.count == 2 else { return }
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
    guard resolved.count == 2 else { return }
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
    guard resolved.count == 2 else { return }
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
    guard resolved.count == 1 else { return }
    XCTAssertEqual(resolved[0].tokenStart, 2)
    XCTAssertEqual(resolved[0].tokenEnd, 4, "span must clamp at the truncation window")
    XCTAssertEqual(dropped, ["g h"], "a beat entirely past the window must drop fail-open")
  }

  // MARK: comfybox#335 — production tokenizer semantics (BOS + leading space)

  /// Mimics the two Gemma 3 behaviors the whitespace fake above cannot,
  /// which are exactly what broke every production locate (comfybox#335):
  /// 1. a BOS special token (id 2) prepended to EVERY standalone encode —
  ///    including the empty string, and
  /// 2. SentencePiece-style pretokens that keep their leading space, so the
  ///    same word gets a DIFFERENT id at text start ("She") vs mid-text
  ///    (" She"), matching the real ids observed in the #335 repro
  ///    (bare "She" = 5778 vs in-context "▁She" = 2625).
  /// Newlines are standalone pretokens (real Gemma: "\n" = 107), so a beat
  /// after a newline appears in its BARE form in context.
  private func makeGemmaLikeTokenizer() -> (String) -> [Int] {
    var table: [String: Int] = [:]
    var next = 100
    return { text in
      var pretokens: [String] = []
      var current = ""
      for ch in text {
        if ch == "\n" {
          if !current.isEmpty { pretokens.append(current); current = "" }
          pretokens.append("\n")
        } else if ch == " " {
          if !current.isEmpty { pretokens.append(current) }
          current = " "
        } else {
          current.append(ch)
        }
      }
      if !current.isEmpty { pretokens.append(current) }
      var ids = [2]  // BOS — prepended to every encode, even ""
      for p in pretokens {
        if let id = table[p] {
          ids.append(id)
        } else {
          table[p] = next
          ids.append(next)
          next += 1
        }
      }
      return ids
    }
  }

  func testGemmaLikeMidPromptBeatLocates() {
    let tokenize = makeGemmaLikeTokenizer()
    let prompt = "Visual: A woman in a sunlit kitchen. She walks slowly toward the counter."
    let full = tokenize(prompt)
    // Pretokens: [Visual:][ A][ woman][ in][ a][ sunlit][ kitchen.]
    //            [ She][ walks][ slowly][ toward][ the][ counter.]
    // = 13 pretokens + BOS = 14 ids. Beat = last 6 pretokens, local 8..<14.
    XCTAssertEqual(full.count, 14)
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("She walks slowly toward the counter.", 0, 1)],
      fullPromptTokenIds: full, maxLength: 16,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: tokenize)

    XCTAssertEqual(dropped, [], "verbatim mid-prompt beat must locate (comfybox#335)")
    XCTAssertEqual(resolved.count, 1)
    guard resolved.count == 1 else { return }
    // padOffset = 16 - 14 = 2, local 8..<14 → padded 10..<16.
    XCTAssertEqual(resolved[0].tokenStart, 10)
    XCTAssertEqual(resolved[0].tokenEnd, 16)
  }

  func testGemmaLikeBeatAtPromptStartLocates() {
    let tokenize = makeGemmaLikeTokenizer()
    let full = tokenize("She walks closer. She sways slowly.")
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("She walks closer.", 0, 0.5), beat("She sways slowly.", 0.5, 1)],
      fullPromptTokenIds: full, maxLength: 16,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: tokenize)

    XCTAssertEqual(dropped, [])
    XCTAssertEqual(resolved.count, 2)
    guard resolved.count == 2 else { return }
    // Prompt: BOS + [She][ walks][ closer.][ She][ sways][ slowly.] = 7 ids,
    // padOffset 9. Beat 1 (bare form) local 1..<4 → 10..<13; beat 2 (leading-
    // space form) local 4..<7 → 13..<16.
    XCTAssertEqual(resolved[0].tokenStart, 10)
    XCTAssertEqual(resolved[0].tokenEnd, 13)
    XCTAssertEqual(resolved[1].tokenStart, 13)
    XCTAssertEqual(resolved[1].tokenEnd, 16)
  }

  func testGemmaLikeNewlineBoundaryBeatLocates() {
    let tokenize = makeGemmaLikeTokenizer()
    // After a newline the next word is a standalone "\n" pretoken followed by
    // the BARE word form — the leading-space variant must NOT be required.
    let full = tokenize("A sunlit kitchen.\nShe walks closer.")
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("She walks closer.", 0, 1)],
      fullPromptTokenIds: full, maxLength: 16,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: tokenize)

    XCTAssertEqual(dropped, [], "beat after a newline must locate via its bare form")
    XCTAssertEqual(resolved.count, 1)
    guard resolved.count == 1 else { return }
    // BOS + [A][ sunlit][ kitchen.][\n][She][ walks][ closer.] = 8 ids,
    // padOffset 8; beat local 5..<8 → 13..<16.
    XCTAssertEqual(resolved[0].tokenStart, 13)
    XCTAssertEqual(resolved[0].tokenEnd, 16)
  }

  func testGemmaLikeQuotedBeatLocates() {
    let tokenize = makeGemmaLikeTokenizer()
    // Mirrors real Gemma: bare '"Come' and in-context ' "Come' are different
    // tokens (236775 '"' vs 623 '▁"') — the leading-space variant matches.
    let full = tokenize("She reaches the counter. \"Come here,\" she whispers.")
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("\"Come here,\" she whispers.", 0, 1)],
      fullPromptTokenIds: full, maxLength: 16,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: tokenize)

    XCTAssertEqual(dropped, [], "quoted mid-prompt beat must locate")
    XCTAssertEqual(resolved.count, 1)
    guard resolved.count == 1 else { return }
  }

  func testGemmaLikeAbsentBeatStillFailOpens() {
    let tokenize = makeGemmaLikeTokenizer()
    let full = tokenize("She walks slowly toward the counter.")
    var dropped: [(String, String)] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("totally absent phrase", 0, 1)],
      fullPromptTokenIds: full, maxLength: 32,
      onDrop: { b, reason in dropped.append((b.text, reason)) },
      tokenize: tokenize)

    XCTAssertEqual(resolved.count, 0)
    guard resolved.count == 0 else { return }
    XCTAssertEqual(dropped.count, 1, "genuinely absent beat must fail-open, exactly one warning")
    XCTAssertTrue(dropped[0].1.contains("could not locate"),
                  "fail-open contract keeps the greppable 'could not locate' reason")
  }

  func testGemmaLikeBeatBeyondTruncationFailOpensNotMislocates() {
    let tokenize = makeGemmaLikeTokenizer()
    // 1 + 12 pretokens = 13 ids; maxLength 7 truncates mid-prompt. The second
    // sentence starts at local 7 — entirely past the window. It must now
    // LOCATE (the #335 fix) and then drop via the truncation guard, never
    // silently mislocate into the surviving window.
    let full = tokenize("She walks slowly toward the counter. She lifts the avocado up high.")
    XCTAssertEqual(full.count, 13)
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [beat("She lifts the avocado up high.", 0.5, 1)],
      fullPromptTokenIds: full, maxLength: 7,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: tokenize)

    XCTAssertEqual(resolved.count, 0)
    guard resolved.count == 0 else { return }
    XCTAssertEqual(dropped.count, 1)
    XCTAssertTrue(dropped[0].contains("window"),
                  "beat past the truncation window must drop via the window guard, not a locate miss: \(dropped[0])")
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
    guard resolved.count == 1 else { return }
    XCTAssertEqual(resolved[0].startFrac, 0, "negative startFrac must clamp to 0")
    XCTAssertEqual(resolved[0].endFrac, 0.5)
    XCTAssertEqual(dropped.sorted(), ["four", "three"],
                   "degenerate and non-finite spans must drop fail-open, not fail the render")
  }
}
