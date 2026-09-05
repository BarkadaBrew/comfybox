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

  // MARK: - #273 anchoring

  func testAnchoredDefaultsFalseAndDecodesTolerantlyFromLegacyJSON() throws {
    // Legacy nearline.json entries (written before #273) have no "anchored"
    // key at all — must decode to false, not fail the whole file.
    let legacyJSON = """
      {"name":"legacy.safetensors","path":"/x/legacy.safetensors","sizeMB":2.0,"kind":"lora"}
      """
    let decoded = try JSONDecoder().decode(NearlineItem.self, from: Data(legacyJSON.utf8))
    XCTAssertEqual(decoded.anchored, false)
  }

  func testAnchoredRoundTripsThroughEncodeDecode() throws {
    let item = NearlineItem(name: "a.safetensors", path: "/x/a.safetensors", sizeMB: 1, kind: "lora", anchored: true)
    let encoder = JSONEncoder()
    let data = try encoder.encode(item)
    let decoded = try JSONDecoder().decode(NearlineItem.self, from: data)
    XCTAssertEqual(decoded.anchored, true)
  }

  func testSetAnchoredStagesAnUnstagedItemSynchronously() throws {
    try writeSource("anchor_me.safetensors", megabytes: 1)
    library.scan()
    XCTAssertEqual(library.item(named: "anchor_me.safetensors")?.staged, false)

    let result = try library.setAnchored(name: "anchor_me.safetensors", anchored: true)
    XCTAssertTrue(result.anchored)
    XCTAssertTrue(result.staged)
    XCTAssertEqual(library.item(named: "anchor_me.safetensors")?.anchored, true)
    XCTAssertEqual(library.item(named: "anchor_me.safetensors")?.staged, true)
  }

  func testSetAnchoredOnAlreadyStagedItemJustSetsFlag() throws {
    try writeSource("already.safetensors", megabytes: 1)
    library.scan()
    let staged = try library.stage(name: "already.safetensors")

    let result = try library.setAnchored(name: "already.safetensors", anchored: true)
    XCTAssertEqual(result.stagedPath, staged)
    XCTAssertTrue(result.anchored)
  }

  func testUnanchoringOnlyClearsFlagAndDoesNotEvict() throws {
    try writeSource("stay.safetensors", megabytes: 1)
    library.scan()
    _ = try library.setAnchored(name: "stay.safetensors", anchored: true)

    let result = try library.setAnchored(name: "stay.safetensors", anchored: false)
    XCTAssertFalse(result.anchored)
    XCTAssertTrue(result.staged, "un-anchoring must not itself evict")
  }

  func testSetAnchoredOnUnknownItemThrows() {
    XCTAssertThrowsError(try library.setAnchored(name: "never-scanned.safetensors", anchored: true)) { error in
      XCTAssertTrue(error is NearlineError)
    }
  }

  func testAnchoredSurvivesRescan() throws {
    try writeSource("pinned.safetensors", megabytes: 1)
    library.scan()
    _ = try library.setAnchored(name: "pinned.safetensors", anchored: true)

    library.scan()
    XCTAssertEqual(library.item(named: "pinned.safetensors")?.anchored, true)
  }

  func testPlanEvictionSkipsAnchoredItems() {
    let old = Date(timeIntervalSince1970: 0)
    let newer = Date(timeIntervalSince1970: 1000)
    let anchoredOld = NearlineItem(
      name: "anchored.safetensors", path: "/x/anchored.safetensors", sizeMB: 4, kind: "lora",
      stagedPath: "/cache/anchored.safetensors", lastUsedAt: old, anchored: true)
    let unanchoredNewer = NearlineItem(
      name: "free.safetensors", path: "/x/free.safetensors", sizeMB: 4, kind: "lora",
      stagedPath: "/cache/free.safetensors", lastUsedAt: newer, anchored: false)

    // Budget can't fit the incoming file without evicting something. Even
    // though `anchoredOld` is the LRU candidate, it must never be selected —
    // the non-anchored (newer-used) item is evicted instead.
    let toEvict = NearlineLibrary.planEviction(
      stagedItems: [anchoredOld, unanchoredNewer], limitMB: 5, incomingMB: 4)

    XCTAssertEqual(toEvict.map(\.name), ["free.safetensors"])
  }

  func testPlanEvictionReturnsEmptyWhenAllStagedAreAnchoredAndOverBudget() {
    let anchored = NearlineItem(
      name: "anchored.safetensors", path: "/x/anchored.safetensors", sizeMB: 10, kind: "lora",
      stagedPath: "/cache/anchored.safetensors", anchored: true)

    // Nothing evictable — anchored items are the only staged items, and an
    // over-budget incoming file must not evict them.
    let toEvict = NearlineLibrary.planEviction(stagedItems: [anchored], limitMB: 5, incomingMB: 4)
    XCTAssertEqual(toEvict, [])
  }

  // MARK: - #273 fix round 1 (C2): stage() throws instead of silently over-filling

  func testStageThrowsInsufficientCapacityWhenAllStagedItemsAreAnchored() throws {
    library.updateConfiguration(.init(roots: [sourceDir.path], cacheLimitGB: 2.0 / 1024.0))  // 2 MB budget
    try writeSource("pinned.safetensors", megabytes: 2)
    try writeSource("incoming.safetensors", megabytes: 2)
    library.scan()

    _ = try library.setAnchored(name: "pinned.safetensors", anchored: true)  // fills the entire budget

    XCTAssertThrowsError(try library.stage(name: "incoming.safetensors")) { error in
      guard case NearlineError.insufficientCapacity(let needMB, let freeMB, let anchoredMB) = error else {
        return XCTFail("expected insufficientCapacity, got \(error)")
      }
      XCTAssertEqual(needMB, 2, accuracy: 0.01)
      XCTAssertEqual(freeMB, 0, accuracy: 0.01)
      XCTAssertEqual(anchoredMB, 2, accuracy: 0.01)
    }
    // The failed stage must not have copied the file (no silent over-fill).
    XCTAssertEqual(library.item(named: "incoming.safetensors")?.staged, false)
    XCTAssertLessThanOrEqual(library.stagedMB, 2.0)
  }

  func testStageThrowsInsufficientCapacityWhenIncomingIsBiggerThanTheWholeBudget() throws {
    library.updateConfiguration(.init(roots: [sourceDir.path], cacheLimitGB: 1.0 / 1024.0))  // 1 MB budget
    try writeSource("huge.safetensors", megabytes: 5)
    library.scan()

    XCTAssertThrowsError(try library.stage(name: "huge.safetensors")) { error in
      guard case NearlineError.insufficientCapacity = error else {
        return XCTFail("expected insufficientCapacity, got \(error)")
      }
    }
  }

  // MARK: - #273 fix round 1 (M): setAnchored persists only after stage() succeeds

  func testSetAnchoredDoesNotPersistTheFlagWhenStagingFails() throws {
    library.updateConfiguration(.init(roots: [sourceDir.path], cacheLimitGB: 1.0 / 1024.0))
    try writeSource("wont_fit.safetensors", megabytes: 5)
    library.scan()

    XCTAssertThrowsError(try library.setAnchored(name: "wont_fit.safetensors", anchored: true))

    XCTAssertEqual(library.item(named: "wont_fit.safetensors")?.anchored, false,
                    "a failed anchor-and-stage must not leave nearline.json claiming the item is anchored")
    XCTAssertEqual(library.item(named: "wont_fit.safetensors")?.staged, false)
  }

  // MARK: - #273 fix round 1 (I3): lock released across the copy

  func testStageForADifferentNameIsNotBlockedByAnInFlightCopy() throws {
    try writeSource("slow.safetensors", megabytes: 1)
    try writeSource("fast.safetensors", megabytes: 1)
    library.scan()

    let copyStarted = XCTestExpectation(description: "slow copy started")
    let releaseCopy = DispatchSemaphore(value: 0)
    library.testCopyDelayHook = { item in
      guard item.name == "slow.safetensors" else { return }
      copyStarted.fulfill()
      releaseCopy.wait()
    }

    let slowStageDone = XCTestExpectation(description: "slow stage finished")
    DispatchQueue.global().async {
      _ = try? self.library.stage(name: "slow.safetensors")
      slowStageDone.fulfill()
    }

    wait(for: [copyStarted], timeout: 5.0)

    // While "slow" is mid-copy (blocked on releaseCopy), staging a
    // DIFFERENT item must complete promptly — the lock must not be held
    // across the copy.
    let start = Date()
    let fastPath = try library.stage(name: "fast.safetensors")
    let elapsed = Date().timeIntervalSince(start)
    XCTAssertTrue(FileManager.default.fileExists(atPath: fastPath))
    XCTAssertLessThan(elapsed, 2.0, "stage() for an unrelated item must not block on another item's in-flight copy")

    releaseCopy.signal()
    wait(for: [slowStageDone], timeout: 5.0)
    XCTAssertEqual(library.item(named: "slow.safetensors")?.staged, true)
  }

  func testConcurrentStageOfTheSameNameWaitsForTheInFlightCopyInsteadOfRacingIt() throws {
    try writeSource("shared.safetensors", megabytes: 1)
    library.scan()

    let copyStarted = XCTestExpectation(description: "copy started")
    let releaseCopy = DispatchSemaphore(value: 0)
    var copyCount = 0
    let copyCountLock = NSLock()
    library.testCopyDelayHook = { _ in
      copyCountLock.lock(); copyCount += 1; copyCountLock.unlock()
      copyStarted.fulfill()
      releaseCopy.wait()
    }

    var firstResult: String?
    var secondResult: String?
    let firstDone = XCTestExpectation(description: "first stage finished")
    let secondDone = XCTestExpectation(description: "second stage finished")

    DispatchQueue.global().async {
      firstResult = try? self.library.stage(name: "shared.safetensors")
      firstDone.fulfill()
    }
    wait(for: [copyStarted], timeout: 5.0)

    DispatchQueue.global().async {
      secondResult = try? self.library.stage(name: "shared.safetensors")
      secondDone.fulfill()
    }
    // Give the second caller a moment to reach the "wait for in-flight
    // group" branch before releasing the copy.
    Thread.sleep(forTimeInterval: 0.1)
    releaseCopy.signal()

    wait(for: [firstDone, secondDone], timeout: 5.0)
    XCTAssertEqual(copyCount, 1, "only one copy of the same file should ever run concurrently")
    XCTAssertNotNil(firstResult)
    XCTAssertEqual(firstResult, secondResult)
  }

  func testLRUEvictionNeverEvictsAnAnchoredItemEvenWhenOldest() throws {
    library.updateConfiguration(.init(roots: [sourceDir.path], cacheLimitGB: 5.0 / 1024.0))
    try writeSource("keep_anchored.safetensors", megabytes: 2)
    try writeSource("second.safetensors", megabytes: 2)
    try writeSource("third.safetensors", megabytes: 2)
    library.scan()

    _ = try library.setAnchored(name: "keep_anchored.safetensors", anchored: true)  // oldest use
    Thread.sleep(forTimeInterval: 0.02)
    _ = try library.stage(name: "second.safetensors")
    Thread.sleep(forTimeInterval: 0.02)
    _ = try library.stage(name: "third.safetensors")  // forces an eviction

    let items = library.list()
    XCTAssertEqual(items.first { $0.name == "keep_anchored.safetensors" }?.staged, true,
                    "anchored item must survive LRU pressure despite being the oldest use")
    XCTAssertEqual(items.first { $0.name == "second.safetensors" }?.staged, false, "the non-anchored LRU item is evicted instead")
    XCTAssertEqual(items.first { $0.name == "third.safetensors" }?.staged, true)
  }
}
