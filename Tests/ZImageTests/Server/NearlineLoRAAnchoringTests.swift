// NearlineLoRAAnchoringTests.swift — #273 fix round 1 (C1): anchoring must
// fix what the issue actually named — a LoRALibraryEntry whose
// `relative_path` points at a detachable attached volume (212 of 213
// entries, measured 2026-08-21). Proves the render-path claim end to end:
// after anchoring, `LoRALibrary.resolve(id)` yields the internal path, and
// that path keeps resolving to a real file even after the simulated
// attached volume ("Bolt") is removed.

import XCTest
@testable import ZImage

final class NearlineLoRAAnchoringTests: XCTestCase {

  private var tempRoot: URL!
  /// Stands in for the detachable attached volume (e.g. /Volumes/Bolt).
  private var boltDir: URL!
  /// LoRALibrary's own (internal, always-resident) root.
  private var libraryRoot: URL!
  private var loraCacheDir: URL!
  private var modelCacheDir: URL!

  private var loraLibrary: LoRALibrary!
  private var nearlineLibrary: NearlineLibrary!

  private let filename = "bolt_lora.safetensors"
  private let entryId = "bolt-lora"

  override func setUpWithError() throws {
    tempRoot = FileManager.default.temporaryDirectory
      .appendingPathComponent("nearline-lora-anchor-\(UUID().uuidString)", isDirectory: true)
    boltDir = tempRoot.appendingPathComponent("bolt", isDirectory: true)
    libraryRoot = tempRoot.appendingPathComponent("library", isDirectory: true)
    loraCacheDir = tempRoot.appendingPathComponent("nearline-loras", isDirectory: true)
    modelCacheDir = tempRoot.appendingPathComponent("nearline-models", isDirectory: true)
    try FileManager.default.createDirectory(at: boltDir, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: libraryRoot, withIntermediateDirectories: true)

    // The file lives on "Bolt".
    let sourceFile = boltDir.appendingPathComponent(filename)
    try Data(repeating: 0xAB, count: 2 * 1_048_576).write(to: sourceFile)

    // Seed library.json exactly the way Todd's real one is shaped today:
    // `relative_path` is an ABSOLUTE path pointing at the attached volume,
    // not a path relative to `libraryRoot`.
    let libraryJSON = """
      {
        "version": 1,
        "updated_at": "2026-01-01T00:00:00Z",
        "entries": [{
          "id": "\(entryId)",
          "filename": "\(filename)",
          "relative_path": "\(sourceFile.path)",
          "size_bytes": 2097152,
          "model_compatibility": ["z-image"],
          "format": "lora",
          "rank": 16,
          "key_count": 10,
          "layer_targets": ["attention"],
          "triggerwords": [],
          "recommended_scale": 1.0,
          "scale_range": [0.0, 2.0],
          "tags": [],
          "category": "uncategorized",
          "notes": "",
          "date_added": "2026-01-01",
          "quarantined": false
        }]
      }
      """
    try Data(libraryJSON.utf8).write(to: libraryRoot.appendingPathComponent("library.json"))

    loraLibrary = try LoRALibrary(root: libraryRoot)
    nearlineLibrary = NearlineLibrary(
      statePath: tempRoot.appendingPathComponent("nearline.json"),
      loraCacheDir: loraCacheDir, modelCacheDir: modelCacheDir)
    nearlineLibrary.updateConfiguration(.init(roots: [boltDir.path], cacheLimitGB: 1))
    nearlineLibrary.scan()
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempRoot)
  }

  // MARK: - Sanity: the pre-anchor state matches the real 212/213 problem

  func testPreconditionEntryPointsAtAbsoluteBoltPath() throws {
    let entry = try XCTUnwrap(loraLibrary.entry(for: entryId))
    XCTAssertTrue(entry.relativePath.hasPrefix(boltDir.path))
    XCTAssertFalse(entry.anchored)
  }

  // MARK: - C1: anchoring rewrites the library entry through LoRALibrary's own API

  func testAnchoringRewritesRelativePathToInternalStagedLocation() throws {
    let nearlineItem = try NearlineLoRAAnchoring.setAnchored(
      name: filename, anchored: true, loraLibrary: loraLibrary, nearlineLibrary: nearlineLibrary)

    let stagedPath = try XCTUnwrap(nearlineItem.stagedPath)
    XCTAssertTrue(stagedPath.hasPrefix(loraCacheDir.path), "staged into the nearline LoRA cache, not Bolt")

    let entry = try XCTUnwrap(loraLibrary.entry(for: entryId))
    XCTAssertEqual(entry.relativePath, stagedPath)
    XCTAssertTrue(entry.anchored)
  }

  /// The load-bearing claim: the render path resolves from internal storage
  /// even once the attached volume is gone.
  func testResolveLoadsFromInternalPathWithBoltDetached() throws {
    _ = try NearlineLoRAAnchoring.setAnchored(
      name: filename, anchored: true, loraLibrary: loraLibrary, nearlineLibrary: nearlineLibrary)

    let internalURL = try loraLibrary.resolve(entryId)
    XCTAssertTrue(internalURL.path.hasPrefix(loraCacheDir.path))
    XCTAssertTrue(FileManager.default.fileExists(atPath: internalURL.path))

    // Simulate detaching Bolt.
    try FileManager.default.removeItem(at: boltDir)

    let resolvedAfterDetach = try loraLibrary.resolve(entryId)
    XCTAssertEqual(resolvedAfterDetach.path, internalURL.path)
    XCTAssertTrue(
      FileManager.default.fileExists(atPath: resolvedAfterDetach.path),
      "render path must still find the file locally once the attached volume is gone")
  }

  func testUnanchoringClearsBothFlagsButLeavesTheFileWhereItIs() throws {
    _ = try NearlineLoRAAnchoring.setAnchored(
      name: filename, anchored: true, loraLibrary: loraLibrary, nearlineLibrary: nearlineLibrary)
    let internalPath = try loraLibrary.resolve(entryId).path

    _ = try NearlineLoRAAnchoring.setAnchored(
      name: filename, anchored: false, loraLibrary: loraLibrary, nearlineLibrary: nearlineLibrary)

    let entry = try XCTUnwrap(loraLibrary.entry(for: entryId))
    XCTAssertFalse(entry.anchored)
    XCTAssertEqual(entry.relativePath, internalPath, "un-anchor must not move the file back or touch the path")
    XCTAssertEqual(nearlineLibrary.item(named: filename)?.anchored, false)
    XCTAssertTrue(FileManager.default.fileExists(atPath: internalPath), "un-anchor must not delete the staged copy")
  }

  // MARK: - Scope: items with no matching library entry

  func testAnchoringAnItemWithNoLibraryEntryOnlySetsTheNearlineFlag() throws {
    let orphanName = "no_library_entry.safetensors"
    try Data(repeating: 0xCD, count: 1_048_576).write(to: boltDir.appendingPathComponent(orphanName))
    nearlineLibrary.scan()

    let result = try NearlineLoRAAnchoring.setAnchored(
      name: orphanName, anchored: true, loraLibrary: loraLibrary, nearlineLibrary: nearlineLibrary)

    XCTAssertTrue(result.anchored)
    XCTAssertNil(loraLibrary.entry(for: orphanName))
  }

  func testAnchoringWithNoLoRALibraryPresentStillAnchorsAtTheNearlineLevel() throws {
    // WarmServer's loraLibrary is nil when init failed — the orchestration
    // must degrade to nearline-only anchoring, not crash or throw.
    let result = try NearlineLoRAAnchoring.setAnchored(
      name: filename, anchored: true, loraLibrary: nil, nearlineLibrary: nearlineLibrary)
    XCTAssertTrue(result.anchored)
  }

  // MARK: - Failure propagation

  func testInsufficientCapacityPropagatesThroughOrchestration() throws {
    nearlineLibrary.updateConfiguration(.init(roots: [boltDir.path], cacheLimitGB: 1.0 / 1024.0))  // 1 MB, file is 2 MB
    XCTAssertThrowsError(
      try NearlineLoRAAnchoring.setAnchored(
        name: filename, anchored: true, loraLibrary: loraLibrary, nearlineLibrary: nearlineLibrary)
    ) { error in
      guard case NearlineError.insufficientCapacity = error else {
        return XCTFail("expected insufficientCapacity, got \(error)")
      }
    }
    // Neither side should have been mutated by the failed attempt.
    XCTAssertEqual(loraLibrary.entry(for: entryId)?.anchored, false)
    XCTAssertEqual(nearlineLibrary.item(named: filename)?.anchored, false)
  }
}
