import XCTest
@testable import ZImage

/// comfybox#322 (review r1, item 2b) — every LTX-2 render stage keeps its
/// step-boundary cancellation check.
///
/// **Why this is a source scan.** Root cause 2 of #322 was literally "the LTX-2
/// render path contains zero `Task.checkCancellation()` sites", and the fix is
/// the placement of those checks: at the TOP of each long loop, before the
/// model pass, and — in `denoisingLoop` — before the #1479 preemption yield.
/// Deleting or reordering any one of them silently restores the bug for that
/// stage. Nothing else in the suite would notice, because the loops themselves
/// are unreachable without weights: `LTX2Pipeline.denoisingLoop` needs a real
/// transformer + VAE, and the streamed decoder needs a loaded `LTX2Decoder3D`.
/// Agents run unit tests only (`intent.md`), so there is no behavioural test
/// that can stand in.
///
/// The repo already relies on this technique where a behavioural test cannot
/// reach the invariant — see `ControlSurfaceParityTests`, which parses
/// `WarmServer.swift`'s route table and pins its tuple count. This is the same
/// bargain: brittle by construction, and that brittleness IS the guard. If a
/// legitimate refactor moves a boundary, update the pin in the same review.
///
/// The complementary behavioural coverage lives in `LTX2CancellationTests`
/// (the boundary's own decision, under real task cancellation) and
/// `LocalVideoInterruptTests` (the coordinator publishes a cancellable task).
final class LTX2CancellationBoundaryTests: XCTestCase {

  private static let repoRoot: URL = {
    // <root>/Tests/ZImageTests/LTX2CancellationBoundaryTests.swift
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }()

  private func source(_ relativePath: String) throws -> [String] {
    let url = Self.repoRoot.appendingPathComponent(relativePath)
    let text = try String(contentsOf: url, encoding: .utf8)
    return text.components(separatedBy: "\n")
  }

  /// The first line of `lines` after `index` that is neither blank nor a `//`
  /// comment — i.e. the first thing the loop body actually executes.
  private func firstStatement(after index: Int, in lines: [String]) -> String {
    for line in lines[(index + 1)...] {
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      if trimmed.isEmpty || trimmed.hasPrefix("//") { continue }
      return trimmed
    }
    return ""
  }

  private func indexOfLine(containing needle: String, in lines: [String]) -> Int? {
    lines.firstIndex { $0.contains(needle) }
  }

  // MARK: - The sampler step loop (both stages)

  /// `LTX2Pipeline.denoisingLoop` is the ONE driver both the base pass and the
  /// 1.5x refine walk, so this single site bounds abort latency for both to one
  /// step. Two properties are pinned, and both are load-bearing:
  ///
  ///   1. the gate is the FIRST statement in the loop body — a check placed
  ///      after the transformer call would cost a full model pass per abort,
  ///      which at HQ is minutes, not the "within one step" #322 asks for;
  ///   2. it goes through `LTX2LoopBoundary`, which evaluates cancellation
  ///      BEFORE the preemption yield. A bare `if preemption.isRaised` here
  ///      would bank a #1479 checkpoint for a render the operator just killed,
  ///      and the coordinator would dutifully resume it.
  func testSamplerStepLoopChecksCancellationFirst() throws {
    let lines = try source("Sources/ZImage/LTX2/LTX2Pipeline.swift")
    guard let loop = indexOfLine(containing: "for i in startStep..<numSteps {", in: lines) else {
      return XCTFail("the denoising step loop moved — update this pin in the same review")
    }
    let first = firstStatement(after: loop, in: lines)
    XCTAssertTrue(
      first.contains("LTX2LoopBoundary.decide(preemption:"),
      """
      the sampler step boundary must gate on LTX2LoopBoundary (cancel before yield) \
      as its first statement; found: \(first)
      """)
    XCTAssertTrue(
      first.hasPrefix("if try "),
      "the gate must be `try`d so CancellationError leaves the loop; found: \(first)")
  }

  // MARK: - The chunk loop

  /// Between chunks — the other place a multi-chunk render can be stopped
  /// without waiting out a whole pass. Cancellation is checked BEFORE the
  /// #1479 free unwind point for the same reason as the step loop.
  func testChunkLoopChecksCancellationBeforeThePreemptionUnwind() throws {
    let lines = try source("Sources/ZImage/LTX2/LTX2VideoGenerator.swift")
    guard let loop = indexOfLine(containing: "for chunk in startChunk..<plan.totalChunks {", in: lines) else {
      return XCTFail("the chunk loop moved — update this pin in the same review")
    }
    XCTAssertEqual(
      firstStatement(after: loop, in: lines), "try Task.checkCancellation()",
      "the chunk boundary must check cancellation before anything else in the body")

    guard let unwind = indexOfLine(containing: "if chunkResume == nil, preemption?.isRaised == true {", in: lines) else {
      return XCTFail("the #1479 chunk unwind point moved — update this pin in the same review")
    }
    XCTAssertLessThan(
      loop, unwind,
      "cancellation must be evaluated before the preemption unwind, never after")
  }

  // MARK: - Decode and audio

  /// Every stage that can burn minutes on its own keeps a boundary. Named
  /// individually rather than counted, so a failure says WHICH stage lost its
  /// check rather than "the number changed".
  func testEveryLongStageKeepsACancellationBoundary() throws {
    let stages: [(file: String, marker: String, why: String)] = [
      ("Sources/ZImage/LTX2/LTX2VideoGenerator.swift",
       "try Task.checkCancellation()\n        telemetry?.begin(.modelLoad)",
       "before the model load — tens of GB, uninterruptible once started"),
      ("Sources/ZImage/LTX2/VAE/LTX2Decoder3D.swift",
       "func runUp(_ idx: Int, _ input: MLXArray, _ ended: Bool) throws {",
       "streamed decode recurses per output volume; runUp must be able to throw"),
      ("Sources/ZImage/LTX2/VAE/LTX2VAE.swift",
       "try Task.checkCancellation()",
       "tiled decode, per tile and per temporal window"),
      ("Sources/ZImage/LTX2/LTX2Pipeline.swift",
       "try Task.checkCancellation()",
       "decode dispatch and the single-pass decode branches"),
    ]
    for stage in stages {
      let text = try String(
        contentsOf: Self.repoRoot.appendingPathComponent(stage.file), encoding: .utf8)
      XCTAssertTrue(
        text.contains(stage.marker),
        "\(stage.file) lost its cancellation boundary (\(stage.why))")
    }

    // The audio pass and the MP4 write are single, multi-second calls with no
    // inner loop; their boundaries sit immediately before them.
    let generator = try String(
      contentsOf: Self.repoRoot.appendingPathComponent(
        "Sources/ZImage/LTX2/LTX2VideoGenerator.swift"), encoding: .utf8)
    XCTAssertTrue(
      generator.contains("try LTX2PostProcess.writeMP4("),
      "the write call moved — update this pin")
    let lines = generator.components(separatedBy: "\n")
    guard let write = indexOfLine(containing: "try LTX2PostProcess.writeMP4(", in: lines) else {
      return XCTFail("writeMP4 call not found")
    }
    let preceding = lines[max(0, write - 6)..<write].map { $0.trimmingCharacters(in: .whitespaces) }
    XCTAssertTrue(
      preceding.contains("try Task.checkCancellation()"),
      """
      an interrupted render must not write an MP4 — this check is what guarantees it \
      (the write itself is synchronous and cannot be preempted part-way)
      """)
  }

  /// The blunt backstop for root cause 2 as it was actually reported: "zero
  /// checkCancellation sites under Sources/ZImage/LTX2". A refactor that
  /// removed the boundaries wholesale would pass every pin above that happens
  /// to key on a moved marker; it cannot pass this.
  func testLTX2SourcesRetainCancellationBoundaries() throws {
    let ltx2 = Self.repoRoot.appendingPathComponent("Sources/ZImage/LTX2")
    var sites = 0
    guard let files = FileManager.default.enumerator(at: ltx2, includingPropertiesForKeys: nil) else {
      return XCTFail("cannot enumerate \(ltx2.path)")
    }
    for case let url as URL in files where url.pathExtension == "swift" {
      let text = try String(contentsOf: url, encoding: .utf8)
      sites += text.components(separatedBy: "Task.checkCancellation()").count - 1
      sites += text.components(separatedBy: "LTX2LoopBoundary.decide(preemption:").count - 1
    }
    XCTAssertGreaterThanOrEqual(
      sites, 8,
      "the LTX-2 render path is back to too few cancellation boundaries (issue #322 was 'zero')")
  }
}
