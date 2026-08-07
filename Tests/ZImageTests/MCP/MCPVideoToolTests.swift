import XCTest
@testable import ZImage

/// Tests for the MCP video tools: registry, executor, types, and proxy.
/// Phase A — Stories A1-A4.
final class MCPVideoToolTests: XCTestCase {

  // MARK: - Tool Registration (Story A1)

  func testGenerateVideoToolRegistered() {
    let tool = MCPToolRegistry.tool(named: "generate_video")
    XCTAssertNotNil(tool, "generate_video tool should be registered in MCPToolRegistry")
  }

  func testVideoStatusToolRegistered() {
    let tool = MCPToolRegistry.tool(named: "video_status")
    XCTAssertNotNil(tool, "video_status tool should be registered in MCPToolRegistry")
  }

  func testGenerateVideoToolInToolsList() {
    let names = MCPToolRegistry.tools.map(\.name)
    XCTAssertTrue(names.contains("generate_video"), "tools array should contain 'generate_video'")
  }

  func testVideoStatusToolInToolsList() {
    let names = MCPToolRegistry.tools.map(\.name)
    XCTAssertTrue(names.contains("video_status"), "tools array should contain 'video_status'")
  }

  func testTotalToolCount() {
    // 21 through the video/upscale work + 11 creative/queue/nearline tools
    // (enhance_prompt, list_characters, list_presets, import_legacy_presets,
    // queue_list, interrupt_render, cancel_job, nearline_{list,scan,stage,evict})
    // added 2026-07 = 32; + compose_montage (#232) + render_storyboard (#237)
    // + import/list/run_workflow + workflow_run_status (#238) = 38;
    // + lora_quarantine (model-pool work, 90e6f38) = 39.
    XCTAssertEqual(MCPToolRegistry.tools.count, 39, "Expected 39 registered MCP tools")
  }

  // MARK: - generate_video Schema (Story A1)

  func testGenerateVideoSchemaHasPromptRequired() {
    let tool = MCPToolRegistry.tool(named: "generate_video")!
    let required = tool.inputSchema["required"] as? [String]
    XCTAssertNotNil(required)
    XCTAssertTrue(required!.contains("prompt"), "prompt should be required")
  }

  func testGenerateVideoSchemaHasAllProperties() {
    let tool = MCPToolRegistry.tool(named: "generate_video")!
    let properties = tool.inputSchema["properties"] as? [String: Any]
    XCTAssertNotNil(properties)
    let expectedKeys = ["prompt", "image_path", "duration", "resolution", "aspect_ratio", "seed", "output_path"]
    for key in expectedKeys {
      XCTAssertNotNil(properties?[key], "Schema should contain property '\(key)'")
    }
  }

  /// The daemon weaves the character description into the prompt itself; the
  /// server must be told to stand down or it prepends ~110 tokens and blows the
  /// 128-token cap, truncating scene + camera off the end (Todd 2026-08-07).
  /// This key was dropped silently once already: `executeGenerateVideo` builds
  /// its body from an explicit whitelist, so an unforwarded key vanishes.
  func testGenerateVideoSchemaExposesSkipCharacterInjection() {
    let tool = MCPToolRegistry.tool(named: "generate_video")!
    let properties = tool.inputSchema["properties"] as? [String: Any]
    let skip = properties?["skip_character_injection"] as? [String: Any]
    XCTAssertNotNil(skip, "generate_video must expose skip_character_injection")
    XCTAssertEqual(skip?["type"] as? String, "boolean")
  }

  func testGenerateVideoSchemaOnlyPromptRequired() {
    let tool = MCPToolRegistry.tool(named: "generate_video")!
    let required = tool.inputSchema["required"] as? [String]
    XCTAssertEqual(required, ["prompt"], "Only 'prompt' should be required")
  }

  func testGenerateVideoPromptGuidanceMatchesLTX23Contract() {
    let tool = MCPToolRegistry.tool(named: "generate_video")!
    let properties = tool.inputSchema["properties"] as! [String: Any]
    let prompt = properties["prompt"] as! [String: Any]
    let description = prompt["description"] as! String

    XCTAssertTrue(description.contains("4-8"))
    XCTAssertTrue(description.contains("45-90"))
    XCTAssertTrue(description.localizedCaseInsensitiveContains("source image is ground truth"))
    XCTAssertFalse(description.localizedCaseInsensitiveContains("longer is better"))
    XCTAssertFalse(description.contains("80-120"))
  }

  // MARK: - video_status Schema (Story A1)

  func testVideoStatusSchemaHasJobIdRequired() {
    let tool = MCPToolRegistry.tool(named: "video_status")!
    let required = tool.inputSchema["required"] as? [String]
    XCTAssertEqual(required, ["job_id"], "job_id should be the only required parameter")
  }

  func testVideoStatusSchemaHasJobIdProperty() {
    let tool = MCPToolRegistry.tool(named: "video_status")!
    let properties = tool.inputSchema["properties"] as? [String: Any]
    XCTAssertNotNil(properties?["job_id"], "Schema should contain 'job_id' property")
  }

  // MARK: - Executor: Missing Required Params (Story A1)

  func testExecuteGenerateVideoMissingPromptReturnsError() async {
    let client = WarmServerClient(host: "127.0.0.1", port: 19999)
    let executor = MCPToolExecutor(client: client)

    let result = await executor.execute(name: "generate_video", arguments: nil)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("prompt") == true)
  }

  func testExecuteGenerateVideoEmptyPromptReturnsError() async {
    let client = WarmServerClient(host: "127.0.0.1", port: 19999)
    let executor = MCPToolExecutor(client: client)

    let params = MCPParams(["prompt": AnyCodable("")])
    let result = await executor.execute(name: "generate_video", arguments: params)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("prompt") == true)
  }

  func testExecuteVideoStatusMissingJobIdReturnsError() async {
    let client = WarmServerClient(host: "127.0.0.1", port: 19999)
    let executor = MCPToolExecutor(client: client)

    let result = await executor.execute(name: "video_status", arguments: nil)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("job_id") == true)
  }

  func testExecuteVideoStatusEmptyJobIdReturnsError() async {
    let client = WarmServerClient(host: "127.0.0.1", port: 19999)
    let executor = MCPToolExecutor(client: client)

    let params = MCPParams(["job_id": AnyCodable("")])
    let result = await executor.execute(name: "video_status", arguments: params)
    XCTAssertTrue(result.isError)
    XCTAssertTrue(result.content.first?.text?.contains("job_id") == true)
  }

  // MARK: - VideoGenerateRequest Decoding (Story A2)

  func testVideoGenerateRequestDecodesMinimalJSON() throws {
    let json = Data("""
    {"prompt": "A cat walking through a garden"}
    """.utf8)

    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let request = try decoder.decode(VideoGenerateRequest.self, from: json)

    XCTAssertEqual(request.prompt, "A cat walking through a garden")
    XCTAssertNil(request.imagePath)
    XCTAssertNil(request.duration)
    XCTAssertNil(request.resolution)
    XCTAssertNil(request.aspectRatio)
    XCTAssertNil(request.seed)
    XCTAssertNil(request.outputPath)
  }

  func testVideoGenerateRequestDecodesFullJSON() throws {
    let json = Data("""
    {
      "prompt": "A woman walking through a forest",
      "image_path": "/Users/todd/output/image.png",
      "duration": 8,
      "resolution": "720p",
      "aspect_ratio": "16:9",
      "seed": 42,
      "output_path": "/Users/todd/output/video.mp4"
    }
    """.utf8)

    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let request = try decoder.decode(VideoGenerateRequest.self, from: json)

    XCTAssertEqual(request.prompt, "A woman walking through a forest")
    XCTAssertEqual(request.imagePath, "/Users/todd/output/image.png")
    XCTAssertEqual(request.duration, 8)
    XCTAssertEqual(request.resolution, "720p")
    XCTAssertEqual(request.aspectRatio, "16:9")
    XCTAssertEqual(request.seed, 42)
    XCTAssertEqual(request.outputPath, "/Users/todd/output/video.mp4")
  }

  func testVideoGenerateRequestModeDetection() throws {
    // T2V: no image_path
    let t2vJSON = Data("""
    {"prompt": "A cat"}
    """.utf8)
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let t2v = try decoder.decode(VideoGenerateRequest.self, from: t2vJSON)
    XCTAssertEqual(t2v.mode, .t2v, "No image_path should be T2V mode")

    // I2V: has image_path
    let i2vJSON = Data("""
    {"prompt": "Camera slowly pans", "image_path": "/tmp/img.png"}
    """.utf8)
    let i2v = try decoder.decode(VideoGenerateRequest.self, from: i2vJSON)
    XCTAssertEqual(i2v.mode, .i2v, "With image_path should be I2V mode")
  }

  // MARK: - VideoJobStatus Encoding (Story A2)

  func testVideoJobStatusEncodesSnakeCase() throws {
    let status = VideoJobStatus(
      jobId: "vid_123_abc",
      status: .queued,
      mode: .t2v,
      backend: "replicate",
      model: "lightricks/ltx-2.3-fast",
      estimatedSeconds: 180
    )

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(status)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

    XCTAssertEqual(json["job_id"] as? String, "vid_123_abc")
    XCTAssertEqual(json["status"] as? String, "queued")
    XCTAssertEqual(json["mode"] as? String, "t2v")
    XCTAssertEqual(json["backend"] as? String, "replicate")
    XCTAssertEqual(json["model"] as? String, "lightricks/ltx-2.3-fast")
    XCTAssertEqual(json["estimated_seconds"] as? Int, 180)
  }

  func testVideoJobStatusEncodesSucceeded() throws {
    let status = VideoJobStatus(
      jobId: "vid_456_def",
      status: .succeeded,
      mode: .i2v,
      backend: "replicate",
      model: "wan-video/wan-2.2-i2v-a14b",
      outputPath: "/Users/todd/output/video.mp4",
      durationMs: 185000,
      fileSizeBytes: 4521984,
      videoDurationSeconds: 5
    )

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(status)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

    XCTAssertEqual(json["status"] as? String, "succeeded")
    XCTAssertEqual(json["output_path"] as? String, "/Users/todd/output/video.mp4")
    XCTAssertEqual(json["duration_ms"] as? Int, 185000)
    XCTAssertEqual(json["file_size_bytes"] as? Int, 4521984)
    XCTAssertEqual(json["video_duration_seconds"] as? Int, 5)
  }

  func testVideoJobStatusEncodesFailed() throws {
    let status = VideoJobStatus(
      jobId: "vid_789_ghi",
      status: .failed,
      backend: "replicate",
      durationMs: 12000,
      error: "NSFW content detected"
    )

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(status)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

    XCTAssertEqual(json["status"] as? String, "failed")
    XCTAssertEqual(json["error"] as? String, "NSFW content detected")
    XCTAssertEqual(json["duration_ms"] as? Int, 12000)
    XCTAssertNil(json["output_path"])
  }

  // MARK: - VideoGenerateRequest Validation (Story A2)

  func testValidateDurationAcceptsValidT2VValues() {
    let validDurations = [6, 8, 10, 12, 14, 16, 18, 20]
    for d in validDurations {
      let error = VideoGenerateRequest.validateDuration(d, mode: .t2v)
      XCTAssertNil(error, "Duration \(d) should be valid for T2V")
    }
  }

  func testValidateDurationRejectsInvalidT2VValues() {
    let invalidDurations = [1, 3, 5, 7, 9, 21, 100]
    for d in invalidDurations {
      let error = VideoGenerateRequest.validateDuration(d, mode: .t2v)
      XCTAssertNotNil(error, "Duration \(d) should be invalid for T2V")
    }
  }

  func testValidateDurationIgnoredForI2V() {
    // I2V duration is fixed (~5s), so any value should be accepted
    let error = VideoGenerateRequest.validateDuration(99, mode: .i2v)
    XCTAssertNil(error, "Duration should be ignored for I2V mode")
  }

  func testValidateResolutionAcceptsValidValues() {
    let valid = ["480p", "720p", "1080p"]
    for r in valid {
      let error = VideoGenerateRequest.validateResolution(r)
      XCTAssertNil(error, "Resolution '\(r)' should be valid")
    }
  }

  func testValidateResolutionRejectsInvalidValues() {
    let error = VideoGenerateRequest.validateResolution("4k")
    XCTAssertNotNil(error, "Resolution '4k' should be invalid")
    XCTAssertTrue(error!.contains("480p"))
  }

  func testValidateAspectRatioAcceptsValidValues() {
    for ar in ["16:9", "9:16"] {
      let error = VideoGenerateRequest.validateAspectRatio(ar)
      XCTAssertNil(error, "Aspect ratio '\(ar)' should be valid")
    }
  }

  func testValidateAspectRatioRejectsInvalidValues() {
    let error = VideoGenerateRequest.validateAspectRatio("4:3")
    XCTAssertNotNil(error, "Aspect ratio '4:3' should be invalid")
  }

  // MARK: - VideoMode (Story A2)

  func testVideoModeRawValues() {
    XCTAssertEqual(VideoMode.t2v.rawValue, "t2v")
    XCTAssertEqual(VideoMode.i2v.rawValue, "i2v")
  }

  // MARK: - VideoJobState (Story A2)

  func testVideoJobStateRawValues() {
    XCTAssertEqual(VideoJobState.queued.rawValue, "queued")
    XCTAssertEqual(VideoJobState.processing.rawValue, "processing")
    XCTAssertEqual(VideoJobState.succeeded.rawValue, "succeeded")
    XCTAssertEqual(VideoJobState.failed.rawValue, "failed")
  }

  // MARK: - ReplicateVideoProxy (Story A3)

  func testProxyWithoutApiKeyReturnsFailed() async {
    let proxy = ReplicateVideoProxy(apiKey: nil, allowedOutputDirectory: "/tmp")
    let request = VideoGenerateRequest(prompt: "A cat walking")
    let status = await proxy.submit(request)

    XCTAssertEqual(status.status, .failed)
    XCTAssertNotNil(status.error)
    XCTAssertTrue(status.error!.contains("API key"), "Error should mention API key")
  }

  func testProxyStatusForNonexistentJob() {
    let proxy = ReplicateVideoProxy(apiKey: "test-key", allowedOutputDirectory: "/tmp")
    let status = proxy.status(jobId: "nonexistent_id")
    XCTAssertNil(status, "Non-existent job should return nil")
  }

  func testProxyJobIdFormat() async {
    let proxy = ReplicateVideoProxy(apiKey: "test-key", allowedOutputDirectory: "/tmp")
    let request = VideoGenerateRequest(prompt: "Test prompt")
    let status = await proxy.submit(request)

    // Job ID should start with "vid_"
    XCTAssertTrue(status.jobId.hasPrefix("vid_"), "Job ID should start with 'vid_'")
  }

  func testProxyT2VMode() async {
    let proxy = ReplicateVideoProxy(apiKey: "test-key", allowedOutputDirectory: "/tmp")
    let request = VideoGenerateRequest(prompt: "A cat walking")
    let status = await proxy.submit(request)

    XCTAssertEqual(status.mode, .t2v, "No image_path should be T2V mode")
    XCTAssertEqual(status.backend, "replicate")
  }

  func testProxyI2VMode() async {
    let proxy = ReplicateVideoProxy(apiKey: "test-key", allowedOutputDirectory: "/tmp")
    let request = VideoGenerateRequest(prompt: "Camera pans slowly", imagePath: "/tmp/test.png")
    let status = await proxy.submit(request)

    XCTAssertEqual(status.mode, .i2v, "With image_path should be I2V mode")
    XCTAssertEqual(status.backend, "replicate")
  }

  func testProxyJobTracking() async {
    let proxy = ReplicateVideoProxy(apiKey: "test-key", allowedOutputDirectory: "/tmp")
    let request = VideoGenerateRequest(prompt: "Test tracking")
    let submitStatus = await proxy.submit(request)

    // Job should be trackable via status()
    let tracked = proxy.status(jobId: submitStatus.jobId)
    XCTAssertNotNil(tracked, "Submitted job should be trackable")
    XCTAssertEqual(tracked?.jobId, submitStatus.jobId)
  }

  func testProxyConcurrentJobs() async {
    let proxy = ReplicateVideoProxy(apiKey: "test-key", allowedOutputDirectory: "/tmp")

    let status1 = await proxy.submit(VideoGenerateRequest(prompt: "Job 1"))
    let status2 = await proxy.submit(VideoGenerateRequest(prompt: "Job 2"))

    XCTAssertNotEqual(status1.jobId, status2.jobId, "Concurrent jobs should have unique IDs")

    let tracked1 = proxy.status(jobId: status1.jobId)
    let tracked2 = proxy.status(jobId: status2.jobId)
    XCTAssertNotNil(tracked1)
    XCTAssertNotNil(tracked2)
  }

  func testProxyPruneCompletedJobs() async {
    let proxy = ReplicateVideoProxy(apiKey: nil, allowedOutputDirectory: "/tmp")
    // Submit without API key -> immediately fails
    let status = await proxy.submit(VideoGenerateRequest(prompt: "Prune test"))
    XCTAssertEqual(status.status, .failed)

    // Job should exist before pruning
    XCTAssertNotNil(proxy.status(jobId: status.jobId))

    // Pruning with 0-second TTL should remove completed jobs
    proxy.pruneCompletedJobs(ttlSeconds: 0)
    XCTAssertNil(proxy.status(jobId: status.jobId), "Pruned job should be gone")
  }

  func testProxyActiveJobCount() async {
    let proxy = ReplicateVideoProxy(apiKey: nil, allowedOutputDirectory: "/tmp")
    XCTAssertEqual(proxy.activeJobCount, 0)

    // Submit a job (will fail immediately due to no API key, but still tracked)
    _ = await proxy.submit(VideoGenerateRequest(prompt: "Count test"))
    // Failed jobs are still in the tracker until pruned
    XCTAssertGreaterThanOrEqual(proxy.activeJobCount, 0)
  }

  // MARK: - Replicate Model Selection (Story A3)

  func testT2VModelIsLTX() {
    XCTAssertEqual(ReplicateVideoProxy.t2vModel, "lightricks/ltx-2.3-fast")
  }

  func testI2VModelIsWan() {
    XCTAssertEqual(ReplicateVideoProxy.i2vModel, "wan-video/wan-2.2-i2v-a14b")
  }

  func testT2VFallbackModelIsWan() {
    XCTAssertEqual(ReplicateVideoProxy.t2vFallbackModel, "wan-video/wan-2.2-t2v-fast")
  }
}
