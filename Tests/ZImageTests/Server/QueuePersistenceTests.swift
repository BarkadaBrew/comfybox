import XCTest
@testable import ZImage

/// Verifies the on-disk queue snapshot round-trips correctly and that an
/// empty state removes the file — the two behaviors WarmServerCoordinator's
/// crash-recovery path depends on (see QueuePersistence.swift).
///
/// ISOLATED (K-FIX-1 follow-up): these tests used to run against the LIVE
/// `~/.comfybox/queue-state.json`, and `testSavingEmptyStateRemovesFile` /
/// `tearDown` DELETED it — so running the unit suite while the engine was
/// serving destroyed its persisted queue. Every path now resolves inside a
/// per-test temp directory via `isolateComfyBoxStateDirectory()`, which
/// asserts the redirection took effect before anything is written.
final class QueuePersistenceTests: XCTestCase {

  private var stateDirectory: URL!

  override func setUpWithError() throws {
    try super.setUpWithError()
    stateDirectory = try isolateComfyBoxStateDirectory()
  }

  func testSaveThenLoadRoundTrips() {
    let active = PersistedQueueJob(
      id: "active-1", kind: "generate", source: "api",
      enqueuedAt: Date(timeIntervalSince1970: 1_000), rawBody: Data("{\"prompt\":\"a\"}".utf8))
    let pending = [
      PersistedQueueJob(
        id: "pending-1", kind: "lora_swap", source: "desktop",
        enqueuedAt: Date(timeIntervalSince1970: 2_000), rawBody: Data("{\"loras\":[]}".utf8)),
    ]
    QueueStateStore.save(PersistedQueueState(active: active, pending: pending))

    let loaded = QueueStateStore.load()
    XCTAssertEqual(loaded?.active?.id, "active-1")
    XCTAssertEqual(loaded?.active?.kind, "generate")
    XCTAssertEqual(loaded?.active?.rawBody, Data("{\"prompt\":\"a\"}".utf8))
    XCTAssertEqual(loaded?.pending.map(\.id), ["pending-1"])
    XCTAssertEqual(loaded?.pending.first?.source, "desktop")
  }

  func testSavingEmptyStateRemovesFile() {
    QueueStateStore.save(PersistedQueueState(
      active: nil,
      pending: [PersistedQueueJob(id: "x", kind: "generate", source: "api", enqueuedAt: Date(), rawBody: Data())]))
    XCTAssertNotNil(QueueStateStore.load())

    QueueStateStore.save(PersistedQueueState(active: nil, pending: []))
    XCTAssertNil(QueueStateStore.load())
    XCTAssertFalse(FileManager.default.fileExists(atPath: QueueStateStore.path.path))
  }

  func testLoadWithNoFileReturnsNil() {
    try? FileManager.default.removeItem(at: QueueStateStore.path)
    XCTAssertNil(QueueStateStore.load())
  }

  /// The snapshot this suite writes and deletes lands in the temp directory,
  /// never in `~/.comfybox` — asserted on the file that was actually created,
  /// not just on the resolved path.
  func testTheSnapshotIsWrittenInsideTheTempDirectory() throws {
    QueueStateStore.save(PersistedQueueState(
      active: nil,
      pending: [PersistedQueueJob(
        id: "x", kind: "generate", source: "api", enqueuedAt: Date(), rawBody: Data("{}".utf8))]))

    XCTAssertTrue(FileManager.default.fileExists(atPath: QueueStateStore.path.path))
    XCTAssertEqual(
      QueueStateStore.path.deletingLastPathComponent().standardizedFileURL.path,
      stateDirectory.path)
    XCTAssertEqual(
      try FileManager.default.contentsOfDirectory(atPath: stateDirectory.path),
      ["queue-state.json"])
  }
}
