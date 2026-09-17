import XCTest
@testable import ZImage

/// WP2b: a compiled chunk's prompt + beat_schedule must be locatable by the
/// engine's beat locator (each beat text is a verbatim substring of the
/// composed prompt, matched left-to-right). A word-per-token stub tokenizer
/// stands in for Gemma; the real-tokenizer behaviour is pinned elsewhere.
final class DirectorBeatLocateTests: XCTestCase {

  /// Word-per-token: ids are assigned on first sight; "" tokenizes to []
  /// (no BOS), whitespace (including the compiler's "\n" joins) separates.
  private final class WordTokenizer {
    private var ids: [String: Int] = [:]
    func tokenize(_ text: String) -> [Int] {
      text.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).map { word in
        let w = String(word)
        if let id = ids[w] { return id }
        let id = ids.count + 100
        ids[w] = id
        return id
      }
    }
  }

  private func compileChunks(
    length: Int, global: String, segments: [DirectorTimeline.PromptSegment]
  ) throws -> DirectorCompilation {
    let t = DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: length),
      globalPrompt: global,
      keyframes: [.init(id: "k1", imagePath: "/img/a.png", frame: 0)],
      promptSegments: segments)
    let v = DirectorValidator.validate(t, fileExists: { _ in true })
    XCTAssertTrue(v.ok, "\(v.issues)")
    return try DirectorCompiler.compile(v, session: "s", source: "test")
  }

  private func decodeLocal(_ body: [String: Any]) throws -> WarmServer.LocalVideoRequest {
    let data = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(WarmServer.LocalVideoRequest.self, from: data)
  }

  func testCompiledBeatsLocateWithStubTokenizer() throws {
    let c = try compileChunks(
      length: 577,
      global: "a woman walks through a crowded market at dusk",
      segments: [
        .init(id: "p1", startFrame: 0, lengthFrames: 120, prompt: "she inspects the fruit"),
        .init(id: "p2", startFrame: 120, lengthFrames: 120, prompt: "she smiles at the vendor"),
        .init(id: "p3", startFrame: 240, lengthFrames: 120, prompt: "she smiles at the vendor"),
        .init(id: "p4", startFrame: 360, lengthFrames: 217, prompt: "she walks away"),
      ])
    XCTAssertEqual(c.chunks.count, 2)

    for chunk in c.chunks {
      let tok = WordTokenizer()
      let req = try decodeLocal(chunk.body)
      let beats = try XCTUnwrap(req.beatSchedule)
      XCTAssertFalse(beats.isEmpty)
      let full = tok.tokenize(req.prompt)
      // Resolved ranges index the LEFT-PADDED axis (Gemma pads at the start).
      let maxLength = 1024
      let padOffset = max(0, maxLength - full.count)
      var dropped: [String] = []
      let resolved = LTX2BeatScheduleLocator.locate(
        beats: beats,
        fullPromptTokenIds: full,
        maxLength: maxLength,
        onDrop: { beat, reason in dropped.append("\(beat.text): \(reason)") },
        tokenize: { tok.tokenize($0) })

      XCTAssertEqual(dropped, [], "chunk \(chunk.index) dropped beats")
      XCTAssertEqual(resolved.count, beats.count, "chunk \(chunk.index): every beat resolves")
      // Monotonic, non-overlapping token ranges in beat order, and each
      // range is exactly the beat's own words (not the global prompt's).
      var cursor = padOffset
      for (beat, r) in zip(beats, resolved) {
        XCTAssertGreaterThanOrEqual(r.tokenStart, cursor, "chunk \(chunk.index) beat '\(beat.text)' went backwards")
        XCTAssertGreaterThan(r.tokenEnd, r.tokenStart)
        XCTAssertEqual(Array(full[(r.tokenStart - padOffset)..<(r.tokenEnd - padOffset)]), tok.tokenize(beat.text))
        XCTAssertEqual(r.startFrac, beat.startFrac)
        XCTAssertEqual(r.endFrac, beat.endFrac)
        cursor = r.tokenEnd
      }
      // The global prompt occupies the head of the token stream; no beat
      // may land inside it.
      let globalLen = tok.tokenize("a woman walks through a crowded market at dusk").count
      XCTAssertTrue(resolved.allSatisfy { $0.tokenStart - padOffset >= globalLen })
    }

    // Duplicate-text segments p2/p3 both live in chunk 0 and resolve to two
    // DISTINCT token runs in time order.
    let chunk0Beats = c.plan.chunks[0].beatSchedule
    XCTAssertEqual(chunk0Beats.map(\.text), [
      "she inspects the fruit", "she smiles at the vendor", "she smiles at the vendor",
    ])
    // Chunk 1 carries p3 (crosses the boundary at 288) and p4.
    XCTAssertEqual(c.plan.chunks[1].beatSchedule.map(\.text), ["she smiles at the vendor", "she walks away"])
    XCTAssertEqual(c.plan.chunks[1].beatSchedule[0].startFrac, 0.0, accuracy: 1e-6)
    XCTAssertEqual(c.plan.chunks[1].beatSchedule[0].endFrac, 72.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(c.plan.chunks[1].beatSchedule[1].startFrac, 72.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(c.plan.chunks[1].beatSchedule[1].endFrac, 1.0, accuracy: 1e-6)
  }

  func testSegmentSpanningBoundaryAppearsInBothChunksAndLocates() throws {
    let c = try compileChunks(
      length: 577,
      global: "night rain on a city street",
      segments: [
        .init(id: "p1", startFrame: 200, lengthFrames: 200, prompt: "a taxi pulls up to the kerb"),
      ])
    let tok = WordTokenizer()
    for chunk in c.chunks {
      let req = try decodeLocal(chunk.body)
      let beats = try XCTUnwrap(req.beatSchedule)
      XCTAssertEqual(beats.map(\.text), ["a taxi pulls up to the kerb"])
      let full = tok.tokenize(req.prompt)
      // maxLength == prompt length => padOffset 0, ranges index `full` directly.
      let resolved = LTX2BeatScheduleLocator.locate(
        beats: beats, fullPromptTokenIds: full, maxLength: full.count, tokenize: { tok.tokenize($0) })
      XCTAssertEqual(resolved.count, 1)
      XCTAssertEqual(Array(full[resolved[0].tokenStart..<resolved[0].tokenEnd]), tok.tokenize("a taxi pulls up to the kerb"))
    }
    XCTAssertEqual(c.plan.chunks[0].beatSchedule[0].startFrac, 200.0 / 289.0, accuracy: 1e-6)
    XCTAssertEqual(c.plan.chunks[0].beatSchedule[0].endFrac, 1.0, accuracy: 1e-6)
    XCTAssertEqual(c.plan.chunks[1].beatSchedule[0].startFrac, 0.0, accuracy: 1e-6)
    XCTAssertEqual(c.plan.chunks[1].beatSchedule[0].endFrac, 112.0 / 289.0, accuracy: 1e-6)
  }
}
