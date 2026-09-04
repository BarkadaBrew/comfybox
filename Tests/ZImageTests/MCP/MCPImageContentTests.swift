import XCTest

@testable import ZImage

/// comfybox#294 — return a completed render as an MCP `image` content block
/// alongside the existing text/path result, gated by an additive
/// `return_image` parameter (default false).
///
/// Payload-size discipline is the whole reason for the gate: a 1024x1024 PNG
/// is typically 1.5-2.5 MB, and base64 inflates that by 4/3 into the JSON-RPC
/// line itself. Over the cap the block is dropped and the text result says so
/// — a truncated or refused response would lose the render entirely.
final class MCPImageContentTests: XCTestCase {

  private func tempFile(bytes: Int, ext: String = "png") throws -> String {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("mcp-image-\(UUID().uuidString).\(ext)")
    try Data(repeating: 0xAB, count: bytes).write(to: url)
    addTeardownBlock { try? FileManager.default.removeItem(at: url) }
    return url.path
  }

  // MARK: - Encoding

  func testSmallPNGIsAttachedAsBase64() throws {
    let path = try tempFile(bytes: 1024)
    let outcome = MCPImageAttachment.encode(path: path)
    guard case .attached(let base64, let mimeType, let fileBytes) = outcome else {
      return XCTFail("expected an attachment, got \(outcome)")
    }
    XCTAssertEqual(mimeType, "image/png")
    XCTAssertEqual(fileBytes, 1024)
    XCTAssertEqual(Data(base64Encoded: base64)?.count, 1024)
  }

  func testOversizedRenderIsSkippedNotTruncated() throws {
    let path = try tempFile(bytes: 4096)
    let outcome = MCPImageAttachment.encode(path: path, limitBytes: 1024)
    guard case .skippedTooLarge(let fileBytes, let limitBytes) = outcome else {
      return XCTFail("expected a size refusal, got \(outcome)")
    }
    XCTAssertEqual(fileBytes, 4096)
    XCTAssertEqual(limitBytes, 1024)
  }

  /// The cap is on the ENCODED size — base64 is 4/3 of the file — so a file
  /// just under the limit that inflates past it is still refused.
  func testCapIsMeasuredOnTheBase64EncodedSize() throws {
    let path = try tempFile(bytes: 900)
    guard case .skippedTooLarge = MCPImageAttachment.encode(path: path, limitBytes: 1000) else {
      return XCTFail("900 bytes encodes to 1200 — over a 1000-byte cap")
    }
  }

  func testMissingFileIsUnreadableNotACrash() {
    guard case .unreadable = MCPImageAttachment.encode(path: "/nope/missing.png") else {
      return XCTFail("expected .unreadable")
    }
  }

  func testMimeTypeFollowsTheExtension() {
    XCTAssertEqual(MCPImageAttachment.mimeType(forPath: "/a/b.png"), "image/png")
    XCTAssertEqual(MCPImageAttachment.mimeType(forPath: "/a/b.PNG"), "image/png")
    XCTAssertEqual(MCPImageAttachment.mimeType(forPath: "/a/b.jpg"), "image/jpeg")
    XCTAssertEqual(MCPImageAttachment.mimeType(forPath: "/a/b.jpeg"), "image/jpeg")
    // Unknown extensions default to PNG — the engine writes PNG.
    XCTAssertEqual(MCPImageAttachment.mimeType(forPath: "/a/b"), "image/png")
  }

  func testDefaultCapIsDocumentedAndSane() {
    // Big enough for a normal 1024-1536px PNG render, small enough that a
    // 4K upscale is refused rather than blowing up the client's context.
    XCTAssertEqual(MCPImageAttachment.defaultLimitBytes, 8 * 1024 * 1024)
  }

  /// One image per result. The engine renders exactly one image per request
  /// (there is no batch `count` on POST /v1/generate), so the cap is
  /// structural, not a policy that can drift.
  func testOneImageBlockPerResult() throws {
    let path = try tempFile(bytes: 64)
    let result = MCPToolResult(
      text: "{}", structuredJSON: nil,
      images: MCPImageAttachment.blocks(paths: [path, path, path]))
    let dict = result.toResponseDict()
    let content = try XCTUnwrap(dict["content"] as? [[String: Any]])
    XCTAssertEqual(content.filter { $0["type"] as? String == "image" }.count, 1)
  }

  // MARK: - Result wire shape

  func testImageBlockSerializesWithTypeDataAndMimeType() throws {
    let path = try tempFile(bytes: 32)
    let result = MCPToolResult(
      text: "{\"output_path\":\"\(path)\"}", structuredJSON: nil,
      images: MCPImageAttachment.blocks(paths: [path]))
    let dict = result.toResponseDict()
    let content = try XCTUnwrap(dict["content"] as? [[String: Any]])
    XCTAssertEqual(content.count, 2, "text block first (compat), then the image block")
    XCTAssertEqual(content[0]["type"] as? String, "text")
    XCTAssertEqual(content[1]["type"] as? String, "image")
    XCTAssertEqual(content[1]["mimeType"] as? String, "image/png")
    XCTAssertEqual(Data(base64Encoded: try XCTUnwrap(content[1]["data"] as? String))?.count, 32)
  }

  /// The default is OFF: an existing caller's result is byte-for-byte what it
  /// was before this change (intent.md — the MCP surface is production).
  func testResultWithoutImagesIsUnchanged() throws {
    let result = MCPToolResult(text: "{\"a\":1}", structuredJSON: nil, images: [])
    let content = try XCTUnwrap(result.toResponseDict()["content"] as? [[String: Any]])
    XCTAssertEqual(content.count, 1)
    XCTAssertEqual(content[0]["type"] as? String, "text")
  }

  // MARK: - get_job + return_image

  func testGetJobAttachesTheImageOnlyWhenAskedAndOnlyWhenCompleted() async throws {
    let path = try tempFile(bytes: 128)
    func run(returnImage: Bool, state: String) async throws -> MCPToolResult {
      try await MCPToolExecutor.runGetJob(jobId: "J-1", kind: .image, returnImage: returnImage) {
        _, p in
        if p == "/v1/queue" { return (200, Data("{}".utf8)) }
        return (
          200,
          try JSONSerialization.data(withJSONObject: [
            "job_id": "J-1", "status": state, "output_path": path,
          ] as [String: Any])
        )
      }
    }
    func imageBlocks(_ result: MCPToolResult) -> Int {
      result.content.filter { $0.type == "image" }.count
    }
    let offAndDone = imageBlocks(try await run(returnImage: false, state: "succeeded"))
    let onAndDone = imageBlocks(try await run(returnImage: true, state: "succeeded"))
    let onAndRunning = imageBlocks(try await run(returnImage: true, state: "processing"))
    XCTAssertEqual(offAndDone, 0, "return_image defaults off — no image block")
    XCTAssertEqual(onAndDone, 1)
    XCTAssertEqual(onAndRunning, 0, "nothing to attach until the render completes")
  }
}
