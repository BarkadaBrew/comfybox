import XCTest
@testable import ZImage

final class AuditLogTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-audit-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
  }

  private func makeLog(file: String = "audit-log.jsonl") -> (AuditLog, URL) {
    let path = tempDir.appendingPathComponent(file)
    return (AuditLog(path: path), path)
  }

  // MARK: - Basic append + read back

  func testAppendThenRecentReturnsEntry() {
    let (log, path) = makeLog()
    log.append(kind: .generationSubmitted, message: "job started", metadata: ["jobId": "abc"])

    XCTAssertTrue(FileManager.default.fileExists(atPath: path.path))
    let entries = log.recent()
    XCTAssertEqual(entries.count, 1)
    XCTAssertEqual(entries[0].kind, .generationSubmitted)
    XCTAssertEqual(entries[0].message, "job started")
    XCTAssertEqual(entries[0].metadata?["jobId"], "abc")
  }

  func testRecentOnMissingFileReturnsEmpty() {
    let (log, _) = makeLog(file: "does-not-exist.jsonl")
    XCTAssertTrue(log.recent().isEmpty)
  }

  // MARK: - Ordering (newest first)

  func testRecentReturnsNewestFirst() {
    let (log, _) = makeLog()
    log.append(kind: .generationSubmitted, message: "first")
    log.append(kind: .generationCompleted, message: "second")
    log.append(kind: .modelUnload, message: "third")

    let entries = log.recent()
    XCTAssertEqual(entries.map(\.message), ["third", "second", "first"])
    XCTAssertEqual(entries.first?.kind, .modelUnload)
  }

  // MARK: - Limit

  func testRecentRespectsLimit() {
    let (log, _) = makeLog()
    for i in 0..<10 {
      log.append(kind: .configChange, message: "event-\(i)")
    }

    let entries = log.recent(limit: 3)
    XCTAssertEqual(entries.count, 3)
    // Newest first: event-9, event-8, event-7
    XCTAssertEqual(entries.map(\.message), ["event-9", "event-8", "event-7"])
  }

  func testRecentLimitLargerThanCountReturnsAll() {
    let (log, _) = makeLog()
    log.append(kind: .modelLoad, message: "only")
    XCTAssertEqual(log.recent(limit: 100).count, 1)
  }

  func testRecentZeroLimitReturnsEmpty() {
    let (log, _) = makeLog()
    log.append(kind: .modelLoad, message: "x")
    XCTAssertTrue(log.recent(limit: 0).isEmpty)
  }

  // MARK: - Persistence is append-only across instances

  func testAppendIsPersistentAcrossInstances() {
    let path = tempDir.appendingPathComponent("shared.jsonl")
    let first = AuditLog(path: path)
    first.append(kind: .generationSubmitted, message: "a")

    let second = AuditLog(path: path)
    second.append(kind: .generationCompleted, message: "b")

    let entries = AuditLog(path: path).recent()
    XCTAssertEqual(entries.map(\.message), ["b", "a"])
  }

  // MARK: - Malformed lines are skipped

  func testMalformedLinesAreSkipped() throws {
    let path = tempDir.appendingPathComponent("mixed.jsonl")
    let log = AuditLog(path: path)
    log.append(kind: .generationSubmitted, message: "valid")
    // Corrupt the file by appending a junk line.
    let handle = try FileHandle(forWritingTo: path)
    try handle.seekToEnd()
    try handle.write(contentsOf: Data("not json\n".utf8))
    try handle.close()

    let entries = log.recent()
    XCTAssertEqual(entries.count, 1)
    XCTAssertEqual(entries[0].message, "valid")
  }

  // MARK: - Codable round-trip

  func testAuditEntryRoundTrip() throws {
    let entry = AuditEntry(
      timestamp: Date(timeIntervalSince1970: 1_700_000_000),
      kind: .modelLoad,
      message: "loaded z-image-turbo",
      metadata: ["path": "/models/z"]
    )
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(entry)

    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let decoded = try decoder.decode(AuditEntry.self, from: data)
    XCTAssertEqual(decoded, entry)
  }

  func testUnknownKindDecodesToRawValue() throws {
    let json = Data(#"{ "kind": "future.event", "message": "hi", "timestamp": "2023-11-14T22:13:20Z" }"#.utf8)
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    let decoded = try decoder.decode(AuditEntry.self, from: json)
    XCTAssertEqual(decoded.kind, AuditKind("future.event"))
    XCTAssertNil(decoded.metadata)
  }

  // MARK: - Concurrent appends serialize cleanly (no interleaved/partial lines)

  func testConcurrentAppendsProduceWellFormedLines() {
    let (log, _) = makeLog(file: "concurrent.jsonl")
    let count = 200
    DispatchQueue.concurrentPerform(iterations: count) { i in
      log.append(kind: .generationCompleted, message: "msg-\(i)")
    }
    let entries = log.recent(limit: count + 10)
    XCTAssertEqual(entries.count, count)
    // Every line decoded (no corruption) and messages are unique across the set.
    XCTAssertEqual(Set(entries.map(\.message)).count, count)
  }
}
