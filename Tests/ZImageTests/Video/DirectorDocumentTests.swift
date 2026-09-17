import XCTest
@testable import ZImage

/// WP1: `.cbdirector` read/write. The file is the wire timeline JSON,
/// pretty-printed with sorted keys; assets are referenced by path unless the
/// caller opts into embedding.
final class DirectorDocumentTests: XCTestCase {

  private var dir: URL!

  override func setUpWithError() throws {
    dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("director-doc-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: dir)
  }

  private func sample(imagePath: String) -> DirectorTimeline {
    DirectorTimeline(
      settings: .init(width: 576, height: 896, lengthFrames: 289, seed: 7),
      globalPrompt: "g",
      keyframes: [.init(id: "k1", imagePath: imagePath, frame: 0)],
      promptSegments: [.init(id: "p1", startFrame: 0, lengthFrames: 96, prompt: "s")])
  }

  func testFileExtension() {
    XCTAssertEqual(DirectorDocument.fileExtension, "cbdirector")
  }

  func testWriteReadRoundTrip() throws {
    let t = sample(imagePath: "/nonexistent/k1.png")
    let url = dir.appendingPathComponent("a.cbdirector")
    try DirectorDocument.write(t, to: url)
    let back = try DirectorDocument.read(from: url)
    XCTAssertEqual(back, t)
    // Bytes on disk == DirectorJSON.encoder() bytes (pretty, sorted keys).
    XCTAssertEqual(try Data(contentsOf: url), try DirectorJSON.encoder().encode(t))
    XCTAssertEqual(try DirectorDocument.data(t), try DirectorJSON.encoder().encode(t))
  }

  func testWriteNeverEmbedsBase64ByDefault() throws {
    let png = dir.appendingPathComponent("k1.png")
    try Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3]).write(to: png)
    var t = sample(imagePath: png.path)
    // Even an in-memory embedded payload is stripped unless asked.
    t.keyframes[0].imageBase64 = "AAAA"
    let url = dir.appendingPathComponent("b.cbdirector")
    try DirectorDocument.write(t, to: url)
    let text = try String(contentsOf: url, encoding: .utf8)
    XCTAssertFalse(text.contains("image_base64"))
    XCTAssertTrue(text.contains("\"image_path\""))
    let back = try DirectorDocument.read(from: url)
    XCTAssertNil(back.keyframes[0].imageBase64)
    XCTAssertEqual(back.keyframes[0].imagePath, png.path)
  }

  func testEmbedAssetsFillsBase64() throws {
    let bytes = Data((0..<64).map { UInt8($0) })
    let png = dir.appendingPathComponent("k1.png")
    try bytes.write(to: png)
    let t = sample(imagePath: png.path)
    let url = dir.appendingPathComponent("c.cbdirector")
    try DirectorDocument.write(t, to: url, embedAssets: true)
    let back = try DirectorDocument.read(from: url)
    XCTAssertEqual(back.keyframes[0].imagePath, png.path, "path is kept alongside the payload")
    let b64 = try XCTUnwrap(back.keyframes[0].imageBase64)
    XCTAssertEqual(Data(base64Encoded: b64), bytes)
    // A keyframe whose image is unreadable is written without a payload
    // rather than failing the whole save.
    var missing = t
    missing.keyframes.append(.init(id: "k2", imagePath: dir.appendingPathComponent("nope.png").path, frame: 96))
    let url2 = dir.appendingPathComponent("d.cbdirector")
    try DirectorDocument.write(missing, to: url2, embedAssets: true)
    let back2 = try DirectorDocument.read(from: url2)
    XCTAssertNotNil(back2.keyframes[0].imageBase64)
    XCTAssertNil(back2.keyframes[1].imageBase64)
  }

  func testPortableFileKeepsEmbeddedImageOnResave() throws {
    let bytes = Data((0..<32).map { UInt8($0) })
    var t = sample(imagePath: "/Users/someone-else/Pictures/k1.png")  // not on this machine
    t.keyframes[0].imageBase64 = bytes.base64EncodedString()
    for embed in [true, false] {
      let url = dir.appendingPathComponent("portable-\(embed).cbdirector")
      try DirectorDocument.write(t, to: url, embedAssets: embed)
      let back = try DirectorDocument.read(from: url)
      XCTAssertEqual(back.keyframes[0].imagePath, "/Users/someone-else/Pictures/k1.png")
      XCTAssertEqual(
        back.keyframes[0].imageBase64.flatMap { Data(base64Encoded: $0) }, bytes,
        "embed \(embed): the payload is the only copy of the image and must survive a re-save")
    }
  }

  func testUnreadableThrowsFileUnreadable() {
    let url = dir.appendingPathComponent("missing.cbdirector")
    XCTAssertThrowsError(try DirectorDocument.read(from: url)) { error in
      guard case DirectorError.fileUnreadable(let msg) = error else {
        return XCTFail("expected fileUnreadable, got \(error)")
      }
      XCTAssertTrue(msg.contains("missing.cbdirector"))
    }
    let garbage = dir.appendingPathComponent("garbage.cbdirector")
    try? Data("not json".utf8).write(to: garbage)
    XCTAssertThrowsError(try DirectorDocument.read(from: garbage)) { error in
      guard case DirectorError.fileUnreadable = error else {
        return XCTFail("expected fileUnreadable, got \(error)")
      }
    }
  }

  func testFutureVersionFileRejected() throws {
    let url = dir.appendingPathComponent("v9.cbdirector")
    try Data("""
    { "version": 9, "settings": { "width": 576, "height": 896, "length_frames": 289 }, "global_prompt": "p" }
    """.utf8).write(to: url)
    XCTAssertThrowsError(try DirectorDocument.read(from: url)) { error in
      guard case DirectorError.unsupportedVersion(9) = error else {
        return XCTFail("expected unsupportedVersion(9), got \(error)")
      }
    }
  }

  func testMissingVersionFileReadsAsOne() throws {
    let url = dir.appendingPathComponent("nover.cbdirector")
    try Data("""
    { "settings": { "width": 576, "height": 896, "length_frames": 289 }, "global_prompt": "p" }
    """.utf8).write(to: url)
    XCTAssertEqual(try DirectorDocument.read(from: url).version, 1)
  }
}
