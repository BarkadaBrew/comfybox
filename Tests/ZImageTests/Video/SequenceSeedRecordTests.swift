import Foundation
import XCTest

@testable import ZImage

/// A sequence sidecar exists so a render can be reproduced (FDD §4.9.2).
///
/// It recorded the seed for chunk 0 and `nil` for every other chunk, which is
/// worse than recording nothing: the file LOOKS complete, so a replay of a
/// three-chunk sequence silently produces two different chunks. Found by
/// reading the sidecar of the first real demo.
final class SequenceSeedRecordTests: XCTestCase {

  private func document(chunks: [SequenceChunkRecord]) -> SequenceDocument {
    SequenceDocument(
      id: "s", name: "s",
      timeline: DirectorTimeline(
        settings: .init(width: 576, height: 896, lengthFrames: 737), globalPrompt: "x"),
      chunks: chunks)
  }

  func testAMissingSeedIsNAMEDNotIgnored() {
    let doc = document(chunks: [
      SequenceChunkRecord(index: 0, startFrame: 0, frames: 249, seed: 707070, recipeHash: "r"),
      SequenceChunkRecord(index: 1, startFrame: 248, frames: 249, seed: nil, recipeHash: "r"),
      SequenceChunkRecord(index: 2, startFrame: 496, frames: 241, seed: nil, recipeHash: "r"),
    ])
    let check = SequenceSidecar.check(doc, currentRecipeHash: "r", currentEngineBuild: nil)
    XCTAssertEqual(check.chunksMissingSeed, [1, 2])
    XCTAssertFalse(check.ok, "a sequence that cannot be replayed exactly is not ok")
  }

  func testAFullyRecordedSequenceReplaysClean() {
    let doc = document(chunks: (0..<3).map {
      SequenceChunkRecord(index: $0, startFrame: $0 * 248, frames: 249,
                          seed: UInt64(707070 + $0), recipeHash: "r")
    })
    let check = SequenceSidecar.check(doc, currentRecipeHash: "r", currentEngineBuild: nil)
    XCTAssertTrue(check.chunksMissingSeed.isEmpty)
    XCTAssertFalse(check.chunksDisagree)
    XCTAssertTrue(check.ok)
  }

  func testChunksRenderedUNDERDIFFERENTRECIPESAreCaught() {
    // A long sequence spans hours. A preset edited halfway through changes
    // only the later chunks, so the clip was never reproducible as one thing
    // — whatever the recipe happens to be now. Undetectable while every chunk
    // reported chunk 0's hash.
    let doc = document(chunks: [
      SequenceChunkRecord(index: 0, startFrame: 0, frames: 249, seed: 1, recipeHash: "before"),
      SequenceChunkRecord(index: 1, startFrame: 248, frames: 249, seed: 2, recipeHash: "before"),
      SequenceChunkRecord(index: 2, startFrame: 496, frames: 241, seed: 3, recipeHash: "AFTER"),
    ])
    let check = SequenceSidecar.check(doc, currentRecipeHash: "before", currentEngineBuild: nil)
    XCTAssertTrue(check.chunksDisagree, "the render itself was not internally consistent")
    XCTAssertFalse(check.recipeDrift, "chunk 0 still matches today's recipe — this is a different fault")
    XCTAssertFalse(check.ok)
  }

  func testSeedsSurviveTheWire() throws {
    let doc = document(chunks: [
      SequenceChunkRecord(index: 0, startFrame: 0, frames: 249, seed: 18_446_744_073_709_551_615)
    ])
    let data = try JSONEncoder().encode(doc)
    let back = try JSONDecoder().decode(SequenceDocument.self, from: data)
    XCTAssertEqual(
      back.chunks.first?.seed, 18_446_744_073_709_551_615,
      "a UInt64 seed must survive JSON — a lossy round trip is a different render")
  }
}
