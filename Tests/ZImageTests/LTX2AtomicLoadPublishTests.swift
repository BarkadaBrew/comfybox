// LTX2AtomicLoadPublishTests.swift — comfybox#340, Codex review r1.
//
// The one-copy memory invariant on the LTX-2 load path. Before this, `load()`
// assigned `self.pipeline` — the fully built transformer + VAE + text encoder —
// and only THEN ran the throwing tokenizer stage. A tokenizer failure therefore
// left `isLoaded == false` while the generator still retained a complete model
// stack, and because `load()` unloads only when `isLoaded` is true, the retry
// built a SECOND stack alongside the first. On a box where one stack is ~54GB
// that is not a leak, it is an OOM.
//
// `atomicallyPublishLoad` is the fix's testable core: nothing is published
// until every throwing stage has succeeded, and a throw discards instead.
// Tested here with plain values — no weights, no MLX, no 54GB.

import XCTest

@testable import ZImage

final class LTX2AtomicLoadPublishTests: XCTestCase {

  private struct StageFailure: Error, Equatable { let stage: String }

  func testSuccessPublishesExactlyOnceAndNeverDiscards() throws {
    var published: [String] = []
    var discards = 0

    try LTX2VideoGenerator.atomicallyPublishLoad(
      build: { "stack" },
      publish: { published.append($0) },
      discard: { discards += 1 })

    XCTAssertEqual(published, ["stack"])
    XCTAssertEqual(discards, 0, "a successful load must not discard anything")
  }

  /// The #340/r1 regression: a stage that throws must publish NOTHING. If
  /// `publish` ran here, the generator would retain a full model stack that
  /// `isLoaded == false` says it does not have.
  func testThrowingStagePublishesNothingAndDiscardsOnce() {
    var published: [String] = []
    var discards = 0

    XCTAssertThrowsError(
      try LTX2VideoGenerator.atomicallyPublishLoad(
        build: { () throws -> String in throw StageFailure(stage: "tokenizer") },
        publish: { published.append($0) },
        discard: { discards += 1 })
    ) { error in
      XCTAssertEqual(error as? StageFailure, StageFailure(stage: "tokenizer"),
                     "the original error must propagate unchanged")
    }

    XCTAssertEqual(published, [], "a failed load must publish no partial stack")
    XCTAssertEqual(discards, 1, "the partial load must be discarded exactly once")
  }

  /// A stage that throws AFTER earlier stages succeeded is the exact shape of
  /// the reviewed bug (pipeline built, tokenizer then fails). Everything the
  /// build produced must be dropped together.
  func testLateStageFailureStillPublishesNothing() {
    var built: [String] = []
    var published: [[String]] = []
    var discards = 0

    XCTAssertThrowsError(
      try LTX2VideoGenerator.atomicallyPublishLoad(
        build: { () throws -> [String] in
          built.append("transformer")
          built.append("vae")
          built.append("textEncoder")
          built.append("pipeline")
          throw StageFailure(stage: "tokenizer")
        },
        publish: { published.append($0) },
        discard: { discards += 1 })
    )

    XCTAssertEqual(built.count, 4, "the earlier stages did run")
    XCTAssertEqual(published, [], "…and none of them may be published")
    XCTAssertEqual(discards, 1)
  }

  /// End to end on the real generator, without weights: a load that cannot find
  /// its checkpoint must leave the generator empty, and a retry must find it
  /// empty too (never two stacks).
  func testFailedLoadLeavesGeneratorEmptyAndRetryableWithoutWeights() throws {
    let tmp = URL(fileURLWithPath: NSTemporaryDirectory())
      .appendingPathComponent("ltx2-340-empty-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: tmp) }

    let generator = LTX2VideoGenerator(
      config: .init(weightsDir: tmp.path, gemmaPath: tmp.path))

    for attempt in 1...2 {
      XCTAssertThrowsError(try generator.load(), "attempt \(attempt) must fail: no checkpoint")
      XCTAssertFalse(generator.isLoaded, "attempt \(attempt): a failed load is not loaded")
      XCTAssertNil(generator.loadedPipeline, "attempt \(attempt): no partial pipeline retained")
      XCTAssertNil(generator.loadedTokenizer, "attempt \(attempt): no partial tokenizer retained")
    }
  }
}
