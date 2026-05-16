import XCTest
@testable import ZImage

/// Tests for the MCP `upscale` tool: registry, executor, payload decoding, and validation.
/// Phase 1 — Stories 1, 2, 4.
final class MCPUpscaleToolTests: XCTestCase {

  // MARK: - Tool Registration (Story 1)

  func testUpscaleToolRegistered() {
    let tool = MCPToolRegistry.tool(named: "upscale")
    XCTAssertNotNil(tool, "upscale tool should be registered in MCPToolRegistry")
  }

  func testUpscaleToolInToolsList() {
    let names = MCPToolRegistry.tools.map(\.name)
    XCTAssertTrue(names.contains("upscale"), "tools array should contain 'upscale'")
  }

  func testUpscaleToolHasRequiredImagePath() {
    let tool = MCPToolRegistry.tool(named: "upscale")!
    let required = tool.inputSchema["required"] as? [String]
    XCTAssertEqual(required, ["image_path"])
  }

  func testUpscaleToolSchemaHasAllProperties() {
    let tool = MCPToolRegistry.tool(named: "upscale")!
    let properties = tool.inputSchema["properties"] as? [String: Any]
    XCTAssertNotNil(properties)
    let expectedKeys = ["image_path", "target_resolution", "seed", "softness", "output_path", "model"]
    for key in expectedKeys {
      XCTAssertNotNil(properties?[key], "Schema should contain property '\(key)'")
    }
  }

  // MARK: - Executor: Missing image_path (Story 1)

  func testExecuteUpscaleMissingImagePathReturnsError() async {
    let client = WarmServerClient(host: "127.0.0.1", port: 19999)
    let executor = MCPToolExecutor(client: client)

    let result = await executor.execute(name: "upscale", arguments: nil)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("image_path") == true)
  }

  func testExecuteUpscaleEmptyImagePathReturnsError() async {
    let client = WarmServerClient(host: "127.0.0.1", port: 19999)
    let executor = MCPToolExecutor(client: client)

    let params = MCPParams(["image_path": AnyCodable("")])
    let result = await executor.execute(name: "upscale", arguments: params)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("image_path") == true)
  }

  // MARK: - UpscalePayload Decoding (Story 1)

  func testUpscalePayloadDecodesMinimalJSON() throws {
    let json = Data("""
    {"image_path": "/tmp/test.png"}
    """.utf8)

    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let payload = try decoder.decode(UpscalePayload.self, from: json)

    XCTAssertEqual(payload.imagePath, "/tmp/test.png")
    XCTAssertNil(payload.targetResolution)
    XCTAssertNil(payload.seed)
    XCTAssertNil(payload.softness)
    XCTAssertNil(payload.outputPath)
    XCTAssertNil(payload.model)
  }

  func testUpscalePayloadDecodesFullJSON() throws {
    let json = Data("""
    {
      "image_path": "/tmp/input.png",
      "target_resolution": 2048,
      "seed": 42,
      "softness": 0.5,
      "output_path": "/tmp/output.png",
      "model": "seedvr2-7b"
    }
    """.utf8)

    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let payload = try decoder.decode(UpscalePayload.self, from: json)

    XCTAssertEqual(payload.imagePath, "/tmp/input.png")
    XCTAssertEqual(payload.targetResolution, 2048)
    XCTAssertEqual(payload.seed, 42)
    XCTAssertEqual(payload.softness, 0.5)
    XCTAssertEqual(payload.outputPath, "/tmp/output.png")
    XCTAssertEqual(payload.model, "seedvr2-7b")
  }

  func testUpscalePayloadPreservesHighResValue() throws {
    let json = Data("""
    {"image_path": "/tmp/test.png", "target_resolution": 2048}
    """.utf8)

    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let payload = try decoder.decode(UpscalePayload.self, from: json)

    XCTAssertEqual(payload.targetResolution, 2048)
  }

  // MARK: - Resolution Validation (Story 4)

  func testResolutionRejectBelow256() {
    let error = UpscalePayload.validateResolution(128)
    XCTAssertNotNil(error, "Resolution 128 should be rejected")
    XCTAssertTrue(error!.contains("256"))
    XCTAssertTrue(error!.contains("2048"))
  }

  func testResolutionRejectAbove2048() {
    let error = UpscalePayload.validateResolution(4096)
    XCTAssertNotNil(error, "Resolution 4096 should be rejected")
  }

  func testResolutionAccept256() {
    let error = UpscalePayload.validateResolution(256)
    XCTAssertNil(error, "Resolution 256 should be accepted")
  }

  func testResolutionAccept1024() {
    let error = UpscalePayload.validateResolution(1024)
    XCTAssertNil(error, "Resolution 1024 should be accepted")
  }

  func testResolutionAccept2048() {
    let error = UpscalePayload.validateResolution(2048)
    XCTAssertNil(error, "Resolution 2048 should be accepted (experimental)")
  }

  func testResolutionAcceptMiddleRange() {
    let error = UpscalePayload.validateResolution(512)
    XCTAssertNil(error, "Resolution 512 should be accepted")
  }

  // MARK: - Softness Validation (Story 4)

  func testSoftnessRejectNegative() {
    let error = UpscalePayload.validateSoftness(-0.1)
    XCTAssertNotNil(error, "Softness -0.1 should be rejected")
  }

  func testSoftnessRejectAboveOne() {
    let error = UpscalePayload.validateSoftness(1.5)
    XCTAssertNotNil(error, "Softness 1.5 should be rejected")
  }

  func testSoftnessAcceptZero() {
    let error = UpscalePayload.validateSoftness(0.0)
    XCTAssertNil(error, "Softness 0.0 should be accepted")
  }

  func testSoftnessAcceptOne() {
    let error = UpscalePayload.validateSoftness(1.0)
    XCTAssertNil(error, "Softness 1.0 should be accepted")
  }

  func testSoftnessAcceptMiddle() {
    let error = UpscalePayload.validateSoftness(0.5)
    XCTAssertNil(error, "Softness 0.5 should be accepted")
  }

  // MARK: - Resolution Warning (Story 4)

  func testWarningForResolutionAbove1024() {
    let warning = UpscalePayload.resolutionWarning(for: 1025)
    XCTAssertNotNil(warning)
    XCTAssertTrue(warning!.contains("experimental"))
    XCTAssertTrue(warning!.contains("OOM"))
  }

  func testNoWarningForResolution1024() {
    let warning = UpscalePayload.resolutionWarning(for: 1024)
    XCTAssertNil(warning)
  }

  func testNoWarningForResolution512() {
    let warning = UpscalePayload.resolutionWarning(for: 512)
    XCTAssertNil(warning)
  }

  // MARK: - Model Variant Validation (Story 2)

  func testModelValidationAcceptsSeedvr23b() {
    let error = UpscalePayload.validateModel("seedvr2-3b")
    XCTAssertNil(error)
  }

  func testModelValidationAcceptsSeedvr27b() {
    let error = UpscalePayload.validateModel("seedvr2-7b")
    XCTAssertNil(error)
  }

  func testModelValidationAcceptsNil() {
    let error = UpscalePayload.validateModel(nil)
    XCTAssertNil(error)
  }

  func testModelValidationRejectsInvalid() {
    let error = UpscalePayload.validateModel("esrgan")
    XCTAssertNotNil(error)
    XCTAssertTrue(error!.contains("esrgan"))
    XCTAssertTrue(error!.contains("seedvr2-3b"))
    XCTAssertTrue(error!.contains("seedvr2-7b"))
  }

  // MARK: - UpscaleResponse Encoding

  func testUpscaleResponseEncodesCorrectly() throws {
    let response = UpscaleResponse(
      success: true,
      outputPath: "/tmp/output.png",
      durationMs: 5000,
      inputResolution: "512x512",
      outputResolution: "1024x1024",
      model: "seedvr2-3b",
      warning: nil
    )

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(response)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

    XCTAssertEqual(json["success"] as? Bool, true)
    XCTAssertEqual(json["output_path"] as? String, "/tmp/output.png")
    XCTAssertEqual(json["duration_ms"] as? Int, 5000)
    XCTAssertEqual(json["input_resolution"] as? String, "512x512")
    XCTAssertEqual(json["output_resolution"] as? String, "1024x1024")
    XCTAssertEqual(json["model"] as? String, "seedvr2-3b")
  }

  func testUpscaleResponseEncodesWarning() throws {
    let response = UpscaleResponse(
      success: true,
      outputPath: "/tmp/output.png",
      durationMs: 90000,
      inputResolution: "512x512",
      outputResolution: "2048x2048",
      model: "seedvr2-3b",
      warning: "target_resolution 2048 is experimental and may cause OOM errors. Safe maximum is 1024."
    )

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(response)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

    XCTAssertNotNil(json["warning"])
    XCTAssertEqual(json["warning"] as? String,
      "target_resolution 2048 is experimental and may cause OOM errors. Safe maximum is 1024.")
  }
}
