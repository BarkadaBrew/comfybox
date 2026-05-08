// MCPToolExecutor.swift — Executes tool calls by proxying to WarmServer
//
// Maps MCP tool names to WarmServer REST endpoints.
// Uses WarmServerClient for all HTTP calls except apply_style,
// which resolves locally using ComfyBoxStylePresets.

import Foundation

/// Executes MCP tool calls by dispatching to WarmServer HTTP endpoints.
public final class MCPToolExecutor: @unchecked Sendable {
  private let client: WarmServerClient

  public init(client: WarmServerClient) {
    self.client = client
  }

  /// Execute a tool call. Returns an MCPToolResult.
  public func execute(name: String, arguments: MCPParams?) async -> MCPToolResult {
    do {
      switch name {
      case "generate_image":
        return try await executeGenerateImage(arguments)
      case "swap_loras":
        return try await executeSwapLoras(arguments)
      case "list_models":
        return try await executeGet("/v1/models")
      case "list_styles":
        return try await executeGet("/v1/styles")
      case "server_health":
        return try await executeGet("/health")
      case "queue_status":
        return try await executeGet("/queue")
      case "clear_queue":
        return try await executeClearQueue()
      case "list_loras":
        return try await executeListLoras()
      case "shutdown_server":
        return try await executeShutdown(arguments)
      case "system_stats":
        return try await executeGet("/system_stats")
      case "apply_style":
        return executeApplyStyle(arguments)
      default:
        return MCPToolResult(error: "Unknown tool: \(name)")
      }
    } catch let error as WarmServerClientError {
      return MCPToolResult(error: "Error: \(error.errorDescription ?? String(describing: error))")
    } catch {
      return MCPToolResult(error: "Error: \(error.localizedDescription)")
    }
  }

  // MARK: - Tool Implementations

  /// generate_image -> POST /v1/generate
  private func executeGenerateImage(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let prompt = params?.string("prompt"), !prompt.isEmpty else {
      return MCPToolResult(error: "Error: 'prompt' is required")
    }

    var body: [String: Any] = ["prompt": prompt]

    if let negativePrompt = params?.string("negative_prompt") {
      body["negative_prompt"] = negativePrompt
    }
    if let width = params?.integer("width") {
      body["width"] = width
    }
    if let height = params?.integer("height") {
      body["height"] = height
    }
    if let steps = params?.integer("steps") {
      body["steps"] = steps
    }
    if let guidance = params?.number("guidance") {
      body["guidance"] = guidance
    }
    if let seed = params?.integer("seed") {
      body["seed"] = seed
    }
    if let outputPath = params?.string("output_path") {
      body["output_path"] = outputPath
    }
    if let scheduler = params?.string("scheduler") {
      body["scheduler"] = scheduler
    }
    if let sigmaSchedule = params?.string("sigma_schedule") {
      body["sigma_schedule"] = sigmaSchedule
    }
    if let imagePath = params?.string("image_path") {
      body["image_path"] = imagePath
    }
    if let imageStrength = params?.number("image_strength") {
      body["image_strength"] = imageStrength
    }

    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/generate", body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }

  /// swap_loras -> POST /v1/lora/swap
  private func executeSwapLoras(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let lorasArray = params?.array("loras") else {
      return MCPToolResult(error: "Error: 'loras' array is required")
    }

    var loraEntries: [[String: Any]] = []
    for item in lorasArray {
      guard let dict = item.dictValue else { continue }
      guard let path = dict["path"]?.stringValue else { continue }
      var entry: [String: Any] = ["path": path]
      if let scale = dict["scale"]?.doubleValue {
        entry["scale"] = scale
      }
      loraEntries.append(entry)
    }

    let body: [String: Any] = ["loras": loraEntries]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/lora/swap", body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }

  /// clear_queue -> POST /queue with {"clear": true}
  private func executeClearQueue() async throws -> MCPToolResult {
    let body: [String: Any] = ["clear": true]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/queue", body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }

  /// list_loras -> GET /object_info, extract LoraLoader lora_name options
  private func executeListLoras() async throws -> MCPToolResult {
    let (status, data) = try await client.get("/object_info")

    guard status == 200 else {
      return mapHTTPResponse(status: status, data: data)
    }

    // Parse /object_info JSON and extract LoraLoader.input.required.lora_name options
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let loraLoader = json["LoraLoader"] as? [String: Any],
          let input = loraLoader["input"] as? [String: Any],
          let required = input["required"] as? [String: Any],
          let loraName = required["lora_name"] as? [Any],
          let options = loraName.first as? [String] else {
      return MCPToolResult(error: "Error: Could not parse LoRA list from /object_info")
    }

    let result: [String: Any] = [
      "loras": options,
      "count": options.count,
      "directory": "~/bin/zimage/loras",
    ]
    let resultData = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
    return MCPToolResult(text: String(data: resultData, encoding: .utf8) ?? "{}")
  }

  /// shutdown_server -> POST /v1/shutdown (with confirm check)
  private func executeShutdown(_ params: MCPParams?) async throws -> MCPToolResult {
    guard params?.bool("confirm") == true else {
      return MCPToolResult(error: "Error: 'confirm' must be true to shut down the server. This is a safety guard.")
    }

    let (status, data) = try await client.post("/v1/shutdown", body: Data("{}".utf8))
    return mapHTTPResponse(status: status, data: data)
  }

  /// apply_style — local-only, no HTTP call.
  /// Looks up the style from ComfyBoxStylePresets and applies prompt transforms.
  private func executeApplyStyle(_ params: MCPParams?) -> MCPToolResult {
    guard let styleId = params?.string("style_id") else {
      return MCPToolResult(error: "Error: 'style_id' is required")
    }
    guard let prompt = params?.string("prompt") else {
      return MCPToolResult(error: "Error: 'prompt' is required")
    }

    guard let preset = ComfyBoxStylePresets.presets[styleId] else {
      let available = ComfyBoxStylePresets.allPresets.map { $0.id }.joined(separator: ", ")
      return MCPToolResult(error: "Error: Unknown style '\(styleId)'. Available: \(available)")
    }

    let negativePrompt = params?.string("negative_prompt")
    let (enhanced, combinedNegative) = preset.apply(prompt: prompt, negativePrompt: negativePrompt)

    var result: [String: Any] = [
      "enhanced_prompt": enhanced,
      "style_id": preset.id,
      "style_name": preset.name,
      "category": preset.category.rawValue,
      "recommended_model": preset.recommendedModel,
    ]

    if let neg = combinedNegative {
      result["negative_prompt"] = neg
    }

    // Include recommended parameters where they differ from defaults
    var recommended: [String: Any] = [:]
    if let steps = preset.steps { recommended["steps"] = steps }
    if let guidance = preset.guidance { recommended["guidance"] = guidance }
    if let w = preset.width { recommended["width"] = w }
    if let h = preset.height { recommended["height"] = h }
    if let strength = preset.img2imgStrength { recommended["img2img_strength"] = strength }
    if !recommended.isEmpty {
      result["recommended"] = recommended
    }

    guard let data = try? JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]),
          let text = String(data: data, encoding: .utf8) else {
      return MCPToolResult(error: "Error: Failed to serialize style result")
    }
    return MCPToolResult(text: text)
  }

  // MARK: - Helpers

  /// Generic GET endpoint handler.
  private func executeGet(_ path: String) async throws -> MCPToolResult {
    let (status, data) = try await client.get(path)
    return mapHTTPResponse(status: status, data: data)
  }

  /// Map WarmServer HTTP response to MCP tool result.
  /// 200 -> success text, any other -> error text.
  private func mapHTTPResponse(status: Int, data: Data) -> MCPToolResult {
    let text = String(data: data, encoding: .utf8) ?? "{}"
    if status == 200 {
      return MCPToolResult(text: text)
    } else {
      // Try to extract error message from JSON response
      if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
         let message = json["error"] as? String {
        return MCPToolResult(error: "Error (\(status)): \(message)")
      }
      return MCPToolResult(error: "Error (\(status)): \(text)")
    }
  }
}
