import XCTest

@testable import ComfyBoxDesktop

/// LoRA import (spec 2026-08-10): the Models tab picker hands the planner a
/// mixed file/folder selection; the planner resolves it into the exact batch
/// the import sheet shows and submits.
final class LoRAImportPlannerTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
      .appendingPathComponent("lora-import-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  private func touch(_ relative: String) throws -> URL {
    let url = tempDir.appendingPathComponent(relative)
    try FileManager.default.createDirectory(
      at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data("stub".utf8).write(to: url)
    return url
  }

  // MARK: - expand

  func testExpandPassesThroughSafetensorsFiles() throws {
    let a = try touch("a.safetensors")
    let b = try touch("b.SafeTensors")  // extension check is case-insensitive
    let result = LoRAImportPlanner.expand(urls: [a, b])
    XCTAssertEqual(result.files.map(\.lastPathComponent), ["a.safetensors", "b.SafeTensors"])
    XCTAssertEqual(result.skipped, 0)
  }

  func testExpandRecursesFoldersAndSkipsNonSafetensors() throws {
    _ = try touch("batch/one.safetensors")
    _ = try touch("batch/nested/two.safetensors")
    _ = try touch("batch/readme.txt")
    _ = try touch("batch/preview.png")
    let result = LoRAImportPlanner.expand(urls: [tempDir.appendingPathComponent("batch")])
    XCTAssertEqual(result.files.map(\.lastPathComponent), ["one.safetensors", "two.safetensors"])
    XCTAssertEqual(result.skipped, 2)
  }

  func testExpandDedupesByFilenameAcrossSelections() throws {
    // The library keys entries by filename — importing two different files
    // with the same name would silently collide server-side. First wins.
    let direct = try touch("dup.safetensors")
    _ = try touch("folder/dup.safetensors")
    _ = try touch("folder/unique.safetensors")
    let result = LoRAImportPlanner.expand(
      urls: [direct, tempDir.appendingPathComponent("folder")])
    XCTAssertEqual(
      result.files.map(\.lastPathComponent), ["dup.safetensors", "unique.safetensors"])
    XCTAssertEqual(result.files.first, direct)
    XCTAssertEqual(result.skipped, 1)
  }

  func testExpandSkipsHiddenFiles() throws {
    _ = try touch("batch/.hidden.safetensors")
    _ = try touch("batch/visible.safetensors")
    let result = LoRAImportPlanner.expand(urls: [tempDir.appendingPathComponent("batch")])
    XCTAssertEqual(result.files.map(\.lastPathComponent), ["visible.safetensors"])
  }

  func testExpandOfNothingIsEmpty() {
    let result = LoRAImportPlanner.expand(urls: [])
    XCTAssertTrue(result.files.isEmpty)
    XCTAssertEqual(result.skipped, 0)
  }

  // MARK: - categories

  func testCategoriesPutVaultFirstThenSortedDistinct() {
    let loras = ["styles", "vault", "characters", "styles", ""].map { category in
      LoRAInfo(
        id: UUID().uuidString, filename: "x.safetensors", modelCompatibility: "krea2",
        format: "lora", rank: 32, sizeBytes: 1, quarantined: false, tags: [],
        category: category, triggerwords: [], recommendedScale: 1.0, isActive: false)
    }
    XCTAssertEqual(
      LoRAImportPlanner.categories(from: loras), ["vault", "characters", "styles"])
  }

  func testCategoriesAlwaysIncludeVaultEvenWhenAbsent() {
    XCTAssertEqual(LoRAImportPlanner.categories(from: []), ["vault"])
  }
}
