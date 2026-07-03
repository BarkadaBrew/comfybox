import XCTest
@testable import ZImage

final class NearlineLibraryTests: XCTestCase {

  private var tempRoot: URL!
  private var sourceDir: URL!
  private var loraCache: URL!
  private var modelCache: URL!
  private var library: NearlineLibrary!

  override func setUpWithError() throws {
    tempRoot = FileManager.default.temporaryDirectory
      .appendingPathComponent("nearline-tests-\(UUID().uuidString)", isDirectory: true)
    sourceDir = tempRoot.appendingPathComponent("archive", isDirectory: true)
    loraCache = tempRoot.appendingPathComponent("loras", isDirectory: true)
    modelCache = tempRoot.appendingPathComponent("models", isDirectory: true)
    try FileManager.default.createDirectory(at: sourceDir, withIntermediateDirectories: true)

    library = NearlineLibrary(
      statePath: tempRoot.appendingPathComponent("nearline.json"),
      loraCacheDir: loraCache,
      modelCacheDir: modelCache
    )
    library.updateConfiguration(.init(roots: [sourceDir.path], cacheLimitGB: 1))
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempRoot)
  }

  private func writeSource(_ name: String, megabytes: Int) throws {
    let url = sourceDir.appendingPathComponent(name)
    let data = Data(repeating: 0xAB, count: megabytes * 1_048_576)
    try data.write(to: url)
  }

  func testScanCatalogsAndClassifies() throws {
    try writeSource("small_lora.safetensors", megabytes: 2)
    try writeSource("nested/deep_lora.safetensors".replacingOccurrences(of: "nested/", with: ""), megabytes: 1)
    // A nested directory entry too.
    let nested = sourceDir.appendingPathComponent("checkpoints", isDirectory: true)
    try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
    try Data(count: 3 * 1_048_576).write(to: nested.appendingPathComponent("nested_lora.safetensors"))

    let count = library.scan()
    XCTAssertEqual(count, 3)
    let names = library.list().map(\.name)
    XCTAssertTrue(names.contains("small_lora.safetensors"))
    XCTAssertTrue(names.contains("nested_lora.safetensors"))
    XCTAssertTrue(library.list().allSatisfy { $0.kind == "lora" })  // all tiny
    XCTAssertTrue(library.list().allSatisfy { !$0.staged })
  }

  func testStageCopiesAndTouches() throws {
    try writeSource("a_lora.safetensors", megabytes: 2)
    library.scan()

    let staged = try library.stage(name: "a_lora.safetensors")
    XCTAssertTrue(FileManager.default.fileExists(atPath: staged))
    XCTAssertTrue(staged.hasPrefix(loraCache.path))
    XCTAssertEqual(library.item(named: "a_lora.safetensors")?.staged, true)

    // Original untouched.
    XCTAssertTrue(FileManager.default.fileExists(
      atPath: sourceDir.appendingPathComponent("a_lora.safetensors").path))

    // Staging again is a no-op returning the same path.
    let again = try library.stage(name: "a_lora.safetensors")
    XCTAssertEqual(again, staged)
  }

  func testLRUEvictionRespectsBudget() throws {
    // Budget ~5 MB; three 2 MB items → staging the third evicts the LRU.
    library.updateConfiguration(.init(roots: [sourceDir.path], cacheLimitGB: 5.0 / 1024.0))
    try writeSource("one.safetensors", megabytes: 2)
    try writeSource("two.safetensors", megabytes: 2)
    try writeSource("three.safetensors", megabytes: 2)
    library.scan()

    _ = try library.stage(name: "one.safetensors")
    Thread.sleep(forTimeInterval: 0.02)
    _ = try library.stage(name: "two.safetensors")
    Thread.sleep(forTimeInterval: 0.02)
    _ = try library.stage(name: "three.safetensors")

    let items = library.list()
    XCTAssertEqual(items.first { $0.name == "one.safetensors" }?.staged, false)  // LRU evicted
    XCTAssertEqual(items.first { $0.name == "two.safetensors" }?.staged, true)
    XCTAssertEqual(items.first { $0.name == "three.safetensors" }?.staged, true)
    XCTAssertLessThanOrEqual(library.stagedMB, 5.0)
  }

  func testEvictRemovesCopyKeepsOriginal() throws {
    try writeSource("gone.safetensors", megabytes: 1)
    library.scan()
    let staged = try library.stage(name: "gone.safetensors")

    XCTAssertTrue(library.evict(name: "gone.safetensors"))
    XCTAssertFalse(FileManager.default.fileExists(atPath: staged))
    XCTAssertTrue(FileManager.default.fileExists(
      atPath: sourceDir.appendingPathComponent("gone.safetensors").path))
    XCTAssertEqual(library.item(named: "gone.safetensors")?.staged, false)
    XCTAssertFalse(library.evict(name: "gone.safetensors"))  // already evicted
  }

  func testStagingSurvivesRescanAndReload() throws {
    try writeSource("keep.safetensors", megabytes: 1)
    library.scan()
    let staged = try library.stage(name: "keep.safetensors")

    library.scan()  // rescan keeps bookkeeping
    XCTAssertEqual(library.item(named: "keep.safetensors")?.stagedPath, staged)

    // A fresh instance reads the persisted state.
    let reloaded = NearlineLibrary(
      statePath: tempRoot.appendingPathComponent("nearline.json"),
      loraCacheDir: loraCache, modelCacheDir: modelCache)
    XCTAssertEqual(reloaded.item(named: "keep.safetensors")?.staged, true)
  }

  func testStageUnknownThrows() {
    XCTAssertThrowsError(try library.stage(name: "never-scanned.safetensors"))
  }
}
