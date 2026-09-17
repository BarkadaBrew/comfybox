import Foundation
import XCTest

@testable import ZImage

/// In-process zip reading for library packs. `ditto` unpacks these fine from a
/// shell but hung when spawned inside the engine, so extraction is ours now.
final class LibraryZipTests: XCTestCase {

  private var dir: URL!

  override func setUpWithError() throws {
    dir = FileManager.default.temporaryDirectory.appendingPathComponent("zip-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: dir)
  }

  /// Build a real zip with the system tool, then read it with ours.
  private func makeZip(files: [String: String], compressed: Bool = true) throws -> URL {
    let source = dir.appendingPathComponent("src")
    for (name, contents) in files {
      let target = source.appendingPathComponent(name)
      try FileManager.default.createDirectory(
        at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
      try Data(contents.utf8).write(to: target)
    }
    let archive = dir.appendingPathComponent("test.zip")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
    process.arguments = ["-r", compressed ? "-9" : "-0", archive.path, "."]
    process.currentDirectoryURL = source
    process.standardOutput = FileHandle.nullDevice
    process.standardError = FileHandle.nullDevice
    try process.run()
    process.waitUntilExit()
    return archive
  }

  func testExtractsDeflatedAndStoredEntries() throws {
    let body = String(repeating: "the quick brown fox. ", count: 500)
    for compressed in [true, false] {
      let archive = try makeZip(
        files: ["manifest.json": #"{"format":"x"}"#, "library/items.json": body],
        compressed: compressed)
      let out = dir.appendingPathComponent("out-\(compressed)")
      let count = try LibraryZip.extract(archive, to: out)
      XCTAssertGreaterThanOrEqual(count, 2)
      XCTAssertEqual(
        try String(contentsOf: out.appendingPathComponent("manifest.json"), encoding: .utf8),
        #"{"format":"x"}"#)
      XCTAssertEqual(
        try String(contentsOf: out.appendingPathComponent("library/items.json"), encoding: .utf8),
        body, "compressed=\(compressed)")
    }
  }

  func testEmptyFileSurvives() throws {
    let archive = try makeZip(files: ["empty.txt": "", "a.txt": "a"])
    let out = dir.appendingPathComponent("out")
    try LibraryZip.extract(archive, to: out)
    XCTAssertEqual(
      try Data(contentsOf: out.appendingPathComponent("empty.txt")).count, 0)
  }

  func testNotAZipIsRefused() {
    let notZip = dir.appendingPathComponent("nope.zip")
    try? Data("hello".utf8).write(to: notZip)
    XCTAssertThrowsError(try LibraryZip.extract(notZip, to: dir.appendingPathComponent("out"))) {
      XCTAssertEqual($0 as? LibraryZipError, .notAZip)
    }
  }

  func testAnEntryCannotEscapeTheDestination() throws {
    // Hand-build a zip whose entry name walks up out of the destination.
    let escaping = "../escaped.txt"
    var data = Data()
    let contents = Data("pwned".utf8)
    let nameBytes = Data(escaping.utf8)
    func append32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    func append16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    let localOffset = 0
    append32(LibraryZip.localFileHeaderSignature)
    append16(20); append16(0); append16(0); append16(0); append16(0)
    append32(0); append32(UInt32(contents.count)); append32(UInt32(contents.count))
    append16(UInt16(nameBytes.count)); append16(0)
    data.append(nameBytes); data.append(contents)
    let centralOffset = data.count
    append32(LibraryZip.centralFileHeaderSignature)
    append16(20); append16(20); append16(0); append16(0); append16(0); append16(0)
    append32(0); append32(UInt32(contents.count)); append32(UInt32(contents.count))
    append16(UInt16(nameBytes.count)); append16(0); append16(0); append16(0); append16(0)
    append32(0); append32(UInt32(localOffset))
    data.append(nameBytes)
    let eocdOffset = data.count
    append32(LibraryZip.endOfCentralDirectorySignature)
    append16(0); append16(0); append16(1); append16(1)
    append32(UInt32(eocdOffset - centralOffset)); append32(UInt32(centralOffset)); append16(0)

    let archive = dir.appendingPathComponent("evil.zip")
    try data.write(to: archive)
    XCTAssertThrowsError(try LibraryZip.extract(archive, to: dir.appendingPathComponent("out"))) {
      XCTAssertEqual($0 as? LibraryZipError, .unsafePath(escaping))
    }
    XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("escaped.txt").path))
  }
}
