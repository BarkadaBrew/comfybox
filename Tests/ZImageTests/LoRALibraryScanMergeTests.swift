import Foundation
import XCTest

@testable import ZImage

/// #313: `LoRALibrary.scan()` rebuilt every entry from `LoRAScanner.analyze()`
/// on every pass, including entries whose `model_compatibility` a user had
/// manually corrected via `POST /v1/loras/{id}/update` — silently clobbering
/// it back to the auto-detected value (`unknown`, in the reported case) the
/// next time the library rescanned. `LoRALibrary.scan()`
/// (Sources/ZImage/LoRA/LoRALibrary.swift, the "Update:" branch) unconditionally
/// set `updated.modelCompatibility = scanResult.compatibility` with no
/// provenance check.
final class LoRALibraryScanMergeTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("lora-library-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  /// Write a minimal, valid safetensors file at `url` with the given keys
  /// (each a `[key: shape]` pair, float32 data). Padding bytes let us grow
  /// the file between scans while staying a well-formed file the scanner can
  /// re-open (a real "the file on disk changed" repro, not corrupted bytes).
  @discardableResult
  private func writeSafetensors(
    at url: URL,
    keys: [String: [Int]],
    paddingBytes: Int = 0
  ) throws -> Int {
    var tensors: [String: Any] = [:]
    var offset = 0
    var dataBlobs: [Data] = []
    for (key, shape) in keys.sorted(by: { $0.key < $1.key }) {
      let count = max(shape.reduce(1, *), 1)
      let byteLen = count * 4  // float32
      tensors[key] = [
        "dtype": "F32",
        "shape": shape,
        "data_offsets": [offset, offset + byteLen],
      ]
      dataBlobs.append(Data(repeating: 0, count: byteLen))
      offset += byteLen
    }
    if paddingBytes > 0 {
      // A harmless extra tensor absorbs the padding so data_offsets stay
      // internally consistent — SafeTensorsReader validates every tensor's
      // byte range against the declared shape.
      tensors["__pad__"] = [
        "dtype": "F32",
        "shape": [paddingBytes / 4],
        "data_offsets": [offset, offset + paddingBytes],
      ]
      dataBlobs.append(Data(repeating: 0, count: paddingBytes))
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

  func testRescanPreservesManuallySetCompatibilityAndRefreshesFileFields() throws {
    let fileURL = tempDir.appendingPathComponent("manual-tag-test.safetensors")
    try writeSafetensors(at: fileURL, keys: ["something.else.lora_A.weight": [4, 8]])
    let library = try LoRALibrary(root: tempDir)

    // First scan: auto-detects (falls to "unknown" for this synthetic key).
    _ = try library.scan()
    guard let firstPass = library.entry(for: "manual-tag-test") else {
      return XCTFail("entry not indexed after first scan")
    }
    XCTAssertEqual(firstPass.modelCompatibility, ["unknown"])
    let originalSize = firstPass.sizeBytes

    // User manually corrects it, the way POST /v1/loras/{id}/update does.
    try library.update("manual-tag-test", patch: LoRAEntryPatch(modelCompatibility: ["ltx"]))
    guard let afterManualEdit = library.entry(for: "manual-tag-test") else {
      return XCTFail("entry missing after update")
    }
    XCTAssertEqual(afterManualEdit.modelCompatibility, ["ltx"])
    XCTAssertEqual(afterManualEdit.compatibilitySource, .manual)

    // The file on disk changes (grows) — a real rescan trigger.
    let newSize = try writeSafetensors(
      at: fileURL, keys: ["something.else.lora_A.weight": [4, 8]], paddingBytes: 4096)
    XCTAssertGreaterThan(UInt64(newSize), originalSize)

    let result = try library.scan()
    XCTAssertEqual(result.updated, 1)

    guard let afterRescan = library.entry(for: "manual-tag-test") else {
      return XCTFail("entry missing after rescan")
    }

    // The manual tag survives the rescan...
    XCTAssertEqual(afterRescan.modelCompatibility, ["ltx"], "manual compatibility tag must survive rescan")
    XCTAssertEqual(afterRescan.compatibilitySource, .manual)

    // ...while file-derived fields (size, hash invalidation) still refresh.
    XCTAssertEqual(afterRescan.sizeBytes, UInt64(newSize), "file-derived size must still update on rescan")
    XCTAssertNil(afterRescan.sha256, "hash must still be invalidated when the file changes")
  }

  func testRescanStillAutoRefreshesCompatibilityWhenNeverManuallySet() throws {
    let fileURL = tempDir.appendingPathComponent("auto-tag-test.safetensors")
    try writeSafetensors(at: fileURL, keys: ["something.else.lora_A.weight": [4, 8]])
    let library = try LoRALibrary(root: tempDir)

    _ = try library.scan()
    guard let firstPass = library.entry(for: "auto-tag-test") else {
      return XCTFail("entry not indexed after first scan")
    }
    XCTAssertEqual(firstPass.modelCompatibility, ["unknown"])
    XCTAssertEqual(firstPass.compatibilitySource, .auto)

    // File changes (now to keys that detect as z-image) without ever going
    // through update() — provenance stays "auto" the whole time.
    _ = try writeSafetensors(
      at: fileURL,
      keys: [
        "diffusion_model.layers.0.attention.to_q.lora_A.weight": [4, 8],
        "diffusion_model.context_refiner.0.attention.to_q.lora_A.weight": [4, 8],
      ],
      paddingBytes: 256)

    let result = try library.scan()
    XCTAssertEqual(result.updated, 1)

    guard let afterRescan = library.entry(for: "auto-tag-test") else {
      return XCTFail("entry missing after rescan")
    }
    XCTAssertEqual(afterRescan.modelCompatibility, ["z-image"], "auto-derived compatibility must still refresh on rescan")
    XCTAssertEqual(afterRescan.compatibilitySource, .auto)
  }
}
