// LoRALibraryScanAnchorTests.swift — #273 fix round 2 (N1): `LoRALibrary
// .scan()` walks `libraryRoot` only. An anchored entry's `relativePath` was
// rewritten (by `NearlineLoRAAnchoring`) to an absolute internal path OUTSIDE
// `libraryRoot` (the nearline-staged cache dir) — the walk never finds that
// file, so on rescan the entry either silently reverts to whatever the walk
// matched by id (if a same-named file happens to exist under libraryRoot) or
// is dropped from the index entirely (if nothing matches). Anchored entries
// must survive `scan()` unchanged as long as their internal file is still
// there, and gracefully un-anchor (with a warning) rather than vanish if it
// is gone.

import Foundation
import XCTest

@testable import ZImage

final class LoRALibraryScanAnchorTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("lora-library-anchor-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  @discardableResult
  private func writeSafetensors(at url: URL, keys: [String: [Int]]) throws -> Int {
    var tensors: [String: Any] = [:]
    var offset = 0
    var dataBlobs: [Data] = []
    for (key, shape) in keys.sorted(by: { $0.key < $1.key }) {
      let count = max(shape.reduce(1, *), 1)
      let byteLen = count * 4
      tensors[key] = [
        "dtype": "F32",
        "shape": shape,
        "data_offsets": [offset, offset + byteLen],
      ]
      dataBlobs.append(Data(repeating: 0, count: byteLen))
      offset += byteLen
    }
    let header = try JSONSerialization.data(withJSONObject: tensors, options: [.sortedKeys])
    var headerLen = UInt64(header.count)
    var file = Data()
    withUnsafeBytes(of: &headerLen) { file.append(contentsOf: $0) }
    file.append(header)
    for blob in dataBlobs { file.append(blob) }
    try file.write(to: url)
    return file.count
  }

  /// Anchors the on-disk-under-libraryRoot entry `id` to a fresh internal
  /// path (mimicking what `NearlineLoRAAnchoring.setAnchored` does), and
  /// returns that internal path.
  private func anchorEntry(_ id: String, in library: LoRALibrary, internalDir: URL) throws -> String {
    let internalPath = internalDir.appendingPathComponent("\(id).safetensors")
    try FileManager.default.createDirectory(at: internalDir, withIntermediateDirectories: true)
    try writeSafetensors(at: internalPath, keys: ["something.else.lora_A.weight": [4, 8]])
    try library.update(id, patch: LoRAEntryPatch(relativePath: internalPath.path, anchored: true))
    return internalPath.path
  }

  // MARK: - N1: rescan after anchoring keeps the internal path

  func testRescanAfterAnchoringKeepsTheInternalPath() throws {
    let fileURL = tempDir.appendingPathComponent("anchor-me.safetensors")
    try writeSafetensors(at: fileURL, keys: ["something.else.lora_A.weight": [4, 8]])
    let library = try LoRALibrary(root: tempDir)
    _ = try library.scan()
    guard let firstPass = library.entry(for: "anchor-me") else {
      return XCTFail("entry not indexed after first scan")
    }

    let internalDir = tempDir.deletingLastPathComponent().appendingPathComponent("internal-\(UUID().uuidString)")
    let internalPath = try anchorEntry(firstPass.id, in: library, internalDir: internalDir)

    // The original file under libraryRoot is still there too (anchoring
    // never deletes the source) — the walk will still find IT at the old
    // relative path and, pre-fix, could stitch it back onto this id.
    let result = try library.scan()
    _ = result

    guard let afterRescan = library.entry(for: firstPass.id) else {
      return XCTFail("anchored entry must not be dropped by scan()")
    }
    XCTAssertEqual(afterRescan.relativePath, internalPath, "scan() must not overwrite an anchored entry's internal path")
    XCTAssertTrue(afterRescan.anchored)
  }

  // MARK: - N1: rescan with the Bolt root removed keeps the anchored entry

  func testRescanWithLibraryRootFileRemovedKeepsTheAnchoredEntry() throws {
    let fileURL = tempDir.appendingPathComponent("anchor-me-2.safetensors")
    try writeSafetensors(at: fileURL, keys: ["something.else.lora_A.weight": [4, 8]])
    let library = try LoRALibrary(root: tempDir)
    _ = try library.scan()
    guard let firstPass = library.entry(for: "anchor-me-2") else {
      return XCTFail("entry not indexed after first scan")
    }

    let internalDir = tempDir.deletingLastPathComponent().appendingPathComponent("internal-\(UUID().uuidString)")
    let internalPath = try anchorEntry(firstPass.id, in: library, internalDir: internalDir)

    // Simulate the attached volume ("Bolt") going away: the file the walk
    // used to see under libraryRoot is now gone. The anchored entry's own
    // (internal, still-present) file must keep it alive in the index.
    try FileManager.default.removeItem(at: fileURL)

    let result = try library.scan()
    _ = result

    guard let afterRescan = library.entry(for: firstPass.id) else {
      return XCTFail("anchored entry must survive scan() even when nothing under libraryRoot matches it")
    }
    XCTAssertEqual(afterRescan.relativePath, internalPath)
    XCTAssertTrue(afterRescan.anchored)
    XCTAssertTrue(FileManager.default.fileExists(atPath: internalPath))

    try? FileManager.default.removeItem(at: internalDir)
  }

  // MARK: - N1: missing internal file un-anchors instead of vanishing

  func testRescanWithInternalFileGoneUnanchorsRatherThanDropping() throws {
    let fileURL = tempDir.appendingPathComponent("anchor-me-3.safetensors")
    try writeSafetensors(at: fileURL, keys: ["something.else.lora_A.weight": [4, 8]])
    let library = try LoRALibrary(root: tempDir)
    _ = try library.scan()
    guard let firstPass = library.entry(for: "anchor-me-3") else {
      return XCTFail("entry not indexed after first scan")
    }

    let internalDir = tempDir.deletingLastPathComponent().appendingPathComponent("internal-\(UUID().uuidString)")
    let internalPath = try anchorEntry(firstPass.id, in: library, internalDir: internalDir)

    // The internal (staged) copy itself is gone — e.g. deleted out of band —
    // while the original file under libraryRoot is untouched.
    try FileManager.default.removeItem(atPath: internalPath)

    let result = try library.scan()
    _ = result

    guard let afterRescan = library.entry(for: firstPass.id) else {
      return XCTFail("entry must fall back to the walk result, not vanish, when the internal file is gone")
    }
    XCTAssertFalse(afterRescan.anchored, "a missing internal file must clear anchored rather than keep claiming it")
    // Falls back to the walk-found (library-relative) path — not the exact
    // string, which macOS's /private/var symlinking on temp dirs can affect,
    // but definitely no longer the internal path that's now gone.
    XCTAssertNotEqual(afterRescan.relativePath, internalPath)
    XCTAssertTrue(afterRescan.relativePath.hasSuffix("anchor-me-3.safetensors"))
  }
}
