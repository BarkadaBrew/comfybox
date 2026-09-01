// LTX2BeatScheduleLocatorRealTokenizerTests.swift — comfybox#335 regression
// coverage against the REAL production Gemma 3 tokenizer (no GPU, no MLX
// eval: LTX2GemmaTokenizer.load only parses tokenizer.json).
//
// Production repro (serve.err.log 2026-08-31T19:46:49): enhance:false, beat
// texts verbatim substrings of the submitted prompt, and ALL FOUR beats
// dropped with "could not locate beat text as a contiguous run in the
// composed prompt tokens". Two mismatches between a beat's STANDALONE
// tokenization and its in-context run caused this:
//   1. BOS: encode prepends <bos> (id 2) to every standalone encode, and 2
//      appears only at position 0 of the full prompt — so no beat could
//      EVER match (this alone broke 100% of production locates).
//   2. Leading space: bare "She" = 5778 ('She') vs in-context 2625 ('▁She').
// Skips (never fails) when the Gemma snapshot isn't on this machine.

import XCTest
@testable import ZImage

final class LTX2BeatScheduleLocatorRealTokenizerTests: XCTestCase {

  /// Tokenizer files from the same mlx-community Gemma 3 snapshot the warm
  /// server loads. Preferred location is `~/.comfybox/reference/` — like the
  /// audio-VAE parity assets, because the HF hub cache under `~/.cache` is
  /// NOT readable from xctest (NSCocoaErrorDomain 257, same class of
  /// restriction as Bolt being TCC-invisible). The hub cache is the
  /// fallback for environments without that restriction.
  static func tokenizerDir() -> URL? {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let reference = home.appendingPathComponent(".comfybox/reference/gemma3-tokenizer")
    if FileManager.default.fileExists(
      atPath: reference.appendingPathComponent("tokenizer.json").path) {
      return reference
    }
    let snapshots = home.appendingPathComponent(
      ".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-8bit/snapshots")
    guard let entries = try? FileManager.default.contentsOfDirectory(
      at: snapshots, includingPropertiesForKeys: nil) else { return nil }
    return entries.first {
      FileManager.default.fileExists(
        atPath: $0.appendingPathComponent("tokenizer.json").path)
    }
  }

  private func loadTokenizer() throws -> LTX2GemmaTokenizer {
    guard let dir = Self.tokenizerDir() else {
      throw XCTSkip("Gemma 3 tokenizer snapshot not present on this machine")
    }
    return try LTX2GemmaTokenizer.load(from: dir, maxLength: 1024)
  }

  // The production failure shape: the kira-video-avocado A/B render
  // (traces/renders-202608.jsonl, effective_prompt_hash=273b53407efb),
  // enhance:false. The Audio/Visual segments and all four beat_schedule
  // texts are VERBATIM from that request; the preset's character preamble
  // is stood in by a neutral description of the same shape (what matters
  // to locate is that every beat sits mid-prompt, past the preamble and
  // the audio marker — audio_marker_token_index was 113 in production).
  static let avocadoPrompt =
    "A woman with a slim build and long straight black hair past mid-back, "
    + "warm dark eyes, natural bare face, real skin texture, a few flyaway "
    + "hairs, lived-in candid look. "
    + "Audio: soft kitchen ambience, gentle knife sounds on wood. Visual: A "
    + "woman in a sunlit kitchen. She walks slowly toward the counter. She "
    + "picks up a ripe avocado and turns it in her hand. She slices the "
    + "avocado open on the wooden board. She smiles and holds both halves up "
    + "toward the camera. Static camera, natural light."

  static let avocadoBeats: [BeatSegment] = [
    BeatSegment(text: "She walks slowly toward the counter.", startFrac: 0, endFrac: 0.25),
    BeatSegment(text: "She picks up a ripe avocado and turns it in her hand.", startFrac: 0.25, endFrac: 0.5),
    BeatSegment(text: "She slices the avocado open on the wooden board.", startFrac: 0.5, endFrac: 0.75),
    BeatSegment(text: "She smiles and holds both halves up toward the camera.", startFrac: 0.75, endFrac: 1),
  ]

  /// The exact production failure: all four verbatim beats must locate.
  /// On unmodified main this fails with zero located / four drops.
  func testAvocadoProductionFixtureLocatesAllFourBeats() throws {
    let tokenizer = try loadTokenizer()
    let fullIds = tokenizer.untruncatedTokenIds(prompt: Self.avocadoPrompt)
    var dropped: [(String, String)] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: Self.avocadoBeats,
      fullPromptTokenIds: fullIds,
      maxLength: tokenizer.maxLength,
      onDrop: { b, reason in dropped.append((String(b.text.prefix(40)), reason)) },
      tokenize: { tokenizer.untruncatedTokenIds(prompt: $0) })

    XCTAssertEqual(
      dropped.map { "\($0.0) — \($0.1)" }, [],
      "verbatim beats must never drop on the production tokenizer (comfybox#335)")
    XCTAssertEqual(resolved.count, 4, "all four production beats must locate")
    guard resolved.count == 4 else { return }

    // Ranges must be monotonic, non-overlapping, and inside the padded axis.
    let padOffset = tokenizer.maxLength - fullIds.count
    var prevEnd = padOffset
    for r in resolved {
      XCTAssertGreaterThanOrEqual(r.tokenStart, prevEnd, "beat ranges must be ordered left-to-right")
      XCTAssertGreaterThan(r.tokenEnd, r.tokenStart)
      XCTAssertLessThanOrEqual(r.tokenEnd, tokenizer.maxLength)
      prevEnd = r.tokenEnd
    }

    // The located run must decode back to the beat text itself (modulo the
    // leading space the in-context form carries).
    let first = resolved[0]
    let span = Array(fullIds[(first.tokenStart - padOffset)..<(first.tokenEnd - padOffset)])
    let decoded = tokenizer.decode(tokens: span).trimmingCharacters(in: .whitespaces)
    XCTAssertEqual(decoded, Self.avocadoBeats[0].text,
                   "located token run must decode back to the beat text")
  }

  /// A genuinely absent beat still fail-opens with the greppable warning.
  func testAbsentBeatFailOpensOnRealTokenizer() throws {
    let tokenizer = try loadTokenizer()
    let fullIds = tokenizer.untruncatedTokenIds(prompt: Self.avocadoPrompt)
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: [BeatSegment(text: "A phrase that appears nowhere in the prompt at all.",
                          startFrac: 0, endFrac: 1)],
      fullPromptTokenIds: fullIds,
      maxLength: tokenizer.maxLength,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: { tokenizer.untruncatedTokenIds(prompt: $0) })

    XCTAssertEqual(resolved.count, 0)
    guard resolved.count == 0 else { return }
    XCTAssertEqual(dropped.count, 1)
    XCTAssertTrue(dropped[0].contains("could not locate"),
                  "absent beat keeps the fail-open 'could not locate' warning")
  }

  /// A beat wholly past the truncation window fail-opens via the window
  /// guard (never a bogus locate miss, never a mislocated range).
  func testBeatBeyondTruncationWindowFailOpensOnRealTokenizer() throws {
    let tokenizer = try loadTokenizer()
    let fullIds = tokenizer.untruncatedTokenIds(prompt: Self.avocadoPrompt)
    // The first beat starts at token 63 in this prompt (probe-verified), so
    // a 48-token window excludes every beat without straddling any.
    let tinyMax = 48
    XCTAssertGreaterThan(fullIds.count, tinyMax)
    var dropped: [String] = []

    let resolved = LTX2BeatScheduleLocator.locate(
      beats: Self.avocadoBeats,
      fullPromptTokenIds: fullIds,
      maxLength: tinyMax,
      onDrop: { _, reason in dropped.append(reason) },
      tokenize: { tokenizer.untruncatedTokenIds(prompt: $0) })

    XCTAssertEqual(resolved.count, 0, "no beat may mislocate into the surviving window")
    guard resolved.count == 0 else { return }
    XCTAssertEqual(dropped.count, Self.avocadoBeats.count)
    for reason in dropped {
      XCTAssertTrue(reason.contains("window"),
                    "past-window beats must drop via the truncation guard: \(reason)")
    }
  }
}
