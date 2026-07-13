import XCTest
@testable import ZImage

/// Verifies the on-disk queue snapshot round-trips correctly and that an
/// empty state removes the file — the two behaviors WarmServerCoordinator's
/// crash-recovery path depends on (see QueuePersistence.swift).
final class QueuePersistenceTests: XCTestCase {

  override func tearDown() {
    try? FileManager.default.removeItem(at: QueueStateStore.path)
    super.tearDown()
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
}
