import XCTest
@testable import ZImage

/// WP2a (docs/FDD-ltx-director-tab.md) — source pin, in the
/// `LTX2CancellationBoundaryTests` style: the multi-keyframe pipeline path
/// must carry the audio (AV) state end to end, and the generator must route
/// any frame > 0 keyframe through it.
///
/// **Why a source scan.** `generateMultiKeyframeResumable` needs a loaded
/// transformer + VAE, and agents run unit tests only, so no behavioural test
/// can observe whether `avState` reached the denoise loop, both boundary
/// checkpoints and the refine. Before WP2a it did not (the path passed
/// `avState: nil` everywhere and encoded text without audio embeddings), which
/// is exactly the regression this pins until the WP5 live ladder runs.
final class LTX2MultiKeyframeAudioPlumbingTests: XCTestCase {

  private static let repoRoot: URL = {
    // <root>/Tests/ZImageTests/LTX2/LTX2MultiKeyframeAudioPlumbingTests.swift
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }()

  private func source(_ relativePath: String) throws -> String {
    try String(contentsOf: Self.repoRoot.appendingPathComponent(relativePath), encoding: .utf8)
  }

  private func occurrences(of needle: String, in text: String) -> Int {
    text.components(separatedBy: needle).count - 1
  }

  private func multiKeyframeSlice() throws -> String {
    let text = try source("Sources/ZImage/LTX2/LTX2Pipeline.swift")
    guard let start = text.range(of: "func generateMultiKeyframeResumable(") else {
      throw XCTSkip("generateMultiKeyframeResumable moved — update this pin in the same review")
    }
    guard let end = text.range(of: "// MARK: - Internal: Denoising Loop", range: start.upperBound..<text.endIndex) else {
      throw XCTSkip("the denoising-loop MARK moved — update this pin in the same review")
    }
    return String(text[start.lowerBound..<end.lowerBound])
  }

  func testMultiKeyframePathCarriesAudioState() throws {
    let slice = try multiKeyframeSlice()
    XCTAssertTrue(slice.contains("audioSeconds: Float?"), "generateMultiKeyframeResumable must accept audioSeconds")
    XCTAssertTrue(slice.contains("returnAudioEmbeddings: wantAudio"), "text encode must request audio embeddings when audio is wanted")
    XCTAssertEqual(occurrences(of: "avState: nil", in: slice), 0, "no call in the multi-keyframe path may drop the AV state")
    let carried = occurrences(of: "avState: avState", in: slice) + occurrences(of: "avState: refineAVState", in: slice)
    XCTAssertGreaterThanOrEqual(carried, 4, "denoise + two boundary checkpoints + refine must all receive avState (got \(carried))")
    XCTAssertTrue(slice.contains("output.audioLatents = avState?.audioLatents"), "the output must surface the audio latents like generateI2VResumable does")
    XCTAssertTrue(slice.contains("restoreAudio(r, into: avState)"), "a resume must restore the checkpoint's audio latents")
  }

  func testPublicMultiKeyframeForwardsAudioSeconds() throws {
    let text = try source("Sources/ZImage/LTX2/LTX2Pipeline.swift")
    guard let start = text.range(of: "public func generateMultiKeyframe("),
          let end = text.range(of: "func generateMultiKeyframeResumable(", range: start.upperBound..<text.endIndex) else {
      return XCTFail("generateMultiKeyframe moved — update this pin in the same review")
    }
    let slice = String(text[start.lowerBound..<end.lowerBound])
    XCTAssertTrue(slice.contains("audioSeconds: Float? = nil"))
    XCTAssertTrue(slice.contains("audioSeconds: audioSeconds"))
  }

  /// The generator's chunk dispatch: the multi-keyframe arm is checked FIRST so
  /// any frame > 0 keyframe takes it, and it passes audioSeconds on chunk 0.
  func testGeneratorDispatchesExtraKeyframesFirstWithAudio() throws {
    let text = try source("Sources/ZImage/LTX2/LTX2VideoGenerator.swift")
    guard let loop = text.range(of: "for chunk in startChunk..<plan.totalChunks {") else {
      return XCTFail("the chunk loop moved — update this pin in the same review")
    }
    let body = String(text[loop.upperBound...])
    guard let arm = body.range(of: "if !extraKeyframes.isEmpty") else {
      return XCTFail("the chunk loop must dispatch on extraKeyframes")
    }
    guard let i2v = body.range(of: "if let image = currentImage") else {
      return XCTFail("the i2v arm moved — update this pin in the same review")
    }
    XCTAssertTrue(arm.lowerBound < i2v.lowerBound, "the multi-keyframe arm must be checked before the i2v arm")
    let armSlice = String(body[arm.lowerBound..<i2v.lowerBound])
    XCTAssertTrue(armSlice.contains("generateMultiKeyframeResumable("))
    XCTAssertTrue(armSlice.contains("audioSeconds: wantAudio && chunk == 0"))
    XCTAssertTrue(armSlice.contains("videoFrameIndex: 0, strength: request.strength"), "a frame-0 init image must be prepended at request.strength")
  }

  /// Every keyframe image must go through the same loader as the init image
  /// (center-crop + resize + H.264 round-trip) so applyConditioning never sees
  /// a shape mismatch.
  func testKeyframesShareTheConditioningLoader() throws {
    let text = try source("Sources/ZImage/LTX2/LTX2VideoGenerator.swift")
    XCTAssertTrue(text.contains("func loadConditioningImage("))
    XCTAssertGreaterThanOrEqual(occurrences(of: "loadConditioningImage(", in: text), 3, "declaration + init image + extras")
  }

  /// The writeMP4 cancellation adjacency (LTX2CancellationBoundaryTests) is a
  /// separate pin; this package does not touch that region, so it must still hold.
  func testWriteMP4CancellationAdjacencyUntouched() throws {
    let lines = try source("Sources/ZImage/LTX2/LTX2VideoGenerator.swift").components(separatedBy: "\n")
    guard let write = lines.firstIndex(where: { $0.contains("try LTX2PostProcess.writeMP4(") }) else {
      return XCTFail("writeMP4 call moved")
    }
    let preceding = lines[max(0, write - 6)..<write].map { $0.trimmingCharacters(in: .whitespaces) }
      .filter { !$0.isEmpty && !$0.hasPrefix("//") }
    XCTAssertTrue(preceding.contains("try Task.checkCancellation()"), "checkCancellation must stay adjacent to writeMP4: \(preceding)")
  }
}
