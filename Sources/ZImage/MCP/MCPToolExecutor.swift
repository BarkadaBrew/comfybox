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
      case "lora_library":
        return try await executeLoraLibrary(arguments)
      case "lora_scan":
        return try await executeLoraScan(arguments)
      case "lora_quarantine":
        return try await executeLoraQuarantine(arguments)
      case "load_model":
        return try await executeLoadModel(arguments)
      case "switch_model":
        return try await executeSwitchModel(arguments)
      case "model_pool":
        return try await executeGet("/v1/model/pool")
      case "unload_model":
        return try await executeUnloadModel(arguments)
      case "generate_video":
        return try await executeGenerateVideo(arguments)
      case "video_status":
        return try await executeVideoStatus(arguments)
      case "upscale":
        return try await executeUpscale(arguments)
      case "enhance_prompt":
        return try await executeEnhancePrompt(arguments)
      case "list_characters":
        return try await executeGet("/v1/characters")
      case "list_presets":
        return try await executeGet("/v1/presets")
      case "import_legacy_presets":
        return try await executePostEmpty("/v1/presets/import-legacy")
      case "queue_list":
        return try await executeGet("/v1/queue")
      case "interrupt_render":
        return try await executePostEmpty("/v1/queue/interrupt")
      case "cancel_job":
        return try await executeCancelJob(arguments)
      case "nearline_list":
        return try await executeGet("/v1/nearline")
      case "nearline_scan":
        return try await executePostEmpty("/v1/nearline/scan")
      case "nearline_stage":
        return try await executeNearlineAction("/v1/nearline/stage", arguments)
      case "nearline_evict":
        return try await executeNearlineAction("/v1/nearline/evict", arguments)
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
    if let maskPath = params?.string("mask_path") {
      body["mask_path"] = maskPath
    }
    if let mode = params?.string("content_mode") {
      body["content_mode"] = mode
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
      "directory": "~/Models/loras",
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


  /// lora_library -> GET /v1/loras (with optional query params)
  private func executeLoraLibrary(_ params: MCPParams?) async throws -> MCPToolResult {
    var path = "/v1/loras"
    var queryItems: [String] = []

    if let model = params?.string("model") {
      queryItems.append("model=\(model)")
    }
    if params?.bool("include_quarantined") == true {
      queryItems.append("include_quarantined=true")
    }
    if !queryItems.isEmpty {
      path += "?" + queryItems.joined(separator: "&")
    }

    let (status, data) = try await client.get(path)
    return mapHTTPResponse(status: status, data: data)
  }

  /// lora_scan -> POST /v1/loras/scan
  private func executeLoraScan(_ params: MCPParams?) async throws -> MCPToolResult {
    var body: [String: Any] = [:]
    if params?.bool("force") == true {
      body["force"] = true
    }
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/loras/scan", body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }

  /// lora_quarantine -> POST /v1/loras/{id}/quarantine or DELETE /v1/loras/{id}/quarantine
  private func executeLoraQuarantine(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let id = params?.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    guard let quarantine = params?.bool("quarantine") else {
      return MCPToolResult(error: "Error: 'quarantine' (boolean) is required")
    }

    let path = "/v1/loras/\(id)/quarantine"

    if quarantine {
      var body: [String: Any] = [:]
      if let reason = params?.string("reason") {
        body["reason"] = reason
      }
      let jsonData = try JSONSerialization.data(withJSONObject: body)
      let (status, data) = try await client.post(path, body: jsonData)
      return mapHTTPResponse(status: status, data: data)
    } else {
      let (status, data) = try await client.delete(path)
      return mapHTTPResponse(status: status, data: data)
    }
  }


  /// load_model -> POST /v1/model/load
  private func executeLoadModel(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let model = params?.string("model"), !model.isEmpty else {
      return MCPToolResult(error: "Error: 'model' is required")
    }
    var body: [String: Any] = ["model": model]
    if let quantization = params?.string("quantization") {
      body["quantization"] = quantization
    }
    if let activate = params?.bool("activate") {
      body["activate"] = activate
    }
    if let wait = params?.bool("wait") {
      body["wait"] = wait
    }
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/model/load", body: jsonData)
    // 202 is valid (async load in progress)
    if status == 200 || status == 202 {
      let text = String(data: data, encoding: .utf8) ?? "{}"
      return MCPToolResult(text: text)
    }
    return mapHTTPResponse(status: status, data: data)
  }

  /// switch_model -> POST /v1/model/activate
  private func executeSwitchModel(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let model = params?.string("model"), !model.isEmpty else {
      return MCPToolResult(error: "Error: 'model' is required")
    }
    let body: [String: Any] = ["model": model]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/model/activate", body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }


  /// unload_model -> POST /v1/model/unload
  private func executeUnloadModel(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let model = params?.string("model"), !model.isEmpty else {
      return MCPToolResult(error: "Error: 'model' is required")
    }
    let body: [String: Any] = ["model": model]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/model/unload", body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }


  /// generate_video -> POST /v1/video/generate/async (submit; poll video_status)
  private func executeGenerateVideo(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let prompt = params?.string("prompt"), !prompt.isEmpty else {
      return MCPToolResult(error: "Error: 'prompt' is required")
    }

    var body: [String: Any] = ["prompt": prompt]

    if let imagePath = params?.string("image_path") {
      body["image_path"] = imagePath
    }
    if let duration = params?.integer("duration") {
      body["duration"] = duration
    }
    if let resolution = params?.string("resolution") {
      body["resolution"] = resolution
    }
    if let aspectRatio = params?.string("aspect_ratio") {
      body["aspect_ratio"] = aspectRatio
    }
    if let seed = params?.integer("seed") {
      body["seed"] = seed
    }
    if let outputPath = params?.string("output_path") {
      body["output_path"] = outputPath
    }
    if let preset = params?.string("preset") {
      body["preset"] = preset
    }

    let jsonData = try JSONSerialization.data(withJSONObject: body)
    // Async route for BOTH backends: local renders return 202 + job_id
    // immediately (poll video_status), instead of blocking the MCP call for
    // the entire multi-minute render — which overran the daemon-side 300s
    // MCP tool timeout on every cold-start render (#219).
    let (status, data) = try await client.post("/v1/video/generate/async", body: jsonData)
    if status == 200 || status == 202 {
      let text = String(data: data, encoding: .utf8) ?? "{}"
      return MCPToolResult(text: text)
    }
    return mapHTTPResponse(status: status, data: data)
  }

  /// video_status -> GET /v1/video/status/{job_id}
  private func executeVideoStatus(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let jobId = params?.string("job_id"), !jobId.isEmpty else {
      return MCPToolResult(error: "Error: 'job_id' is required")
    }

    let (status, data) = try await client.get("/v1/video/status/\(jobId)")
    return mapHTTPResponse(status: status, data: data)
  }

  /// upscale -> POST /v1/upscale
  private func executeUpscale(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let imagePath = params?.string("image_path"), !imagePath.isEmpty else {
      return MCPToolResult(error: "Error: 'image_path' is required")
    }

    var body: [String: Any] = ["image_path": imagePath]

    if let targetResolution = params?.integer("target_resolution") {
      body["target_resolution"] = targetResolution
    }
    if let seed = params?.integer("seed") {
      body["seed"] = seed
    }
    if let softness = params?.number("softness") {
      body["softness"] = softness
    }
    if let outputPath = params?.string("output_path") {
      body["output_path"] = outputPath
    }
    if let model = params?.string("model") {
      body["model"] = model
    }

    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/upscale", body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }

  // MARK: - Helpers

  /// Generic GET endpoint handler.
  private func executeGet(_ path: String) async throws -> MCPToolResult {
    let (status, data) = try await client.get(path)
    return mapHTTPResponse(status: status, data: data)
  }

  /// POST an empty JSON body (for trigger-style endpoints).
  private func executePostEmpty(_ path: String) async throws -> MCPToolResult {
    let (status, data) = try await client.post(path, body: Data("{}".utf8))
    return mapHTTPResponse(status: status, data: data)
  }

  /// enhance_prompt -> POST /v1/enhance
  private func executeEnhancePrompt(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let prompt = params?.string("prompt"), !prompt.isEmpty else {
      return MCPToolResult(error: "Error: 'prompt' is required")
    }
    var body: [String: Any] = ["prompt": prompt]
    if let character = params?.string("character") { body["character"] = character }
    if let mode = params?.string("content_mode") { body["content_mode"] = mode }
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/enhance", body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }

  /// cancel_job -> DELETE /v1/queue/{id}
  private func executeCancelJob(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let id = params?.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
    let (status, data) = try await client.delete("/v1/queue/\(encoded)")
    return mapHTTPResponse(status: status, data: data)
  }

  /// nearline_stage / nearline_evict -> POST /v1/nearline/{action} { name }
  private func executeNearlineAction(_ path: String, _ params: MCPParams?) async throws -> MCPToolResult {
    guard let name = params?.string("name"), !name.isEmpty else {
      return MCPToolResult(error: "Error: 'name' is required")
    }
    let jsonData = try JSONSerialization.data(withJSONObject: ["name": name])
    let (status, data) = try await client.post(path, body: jsonData)
    return mapHTTPResponse(status: status, data: data)
  }

  /// Map WarmServer HTTP response to MCP tool result.
  /// 200 -> success (text + structured fields), any other -> error text.
  private func mapHTTPResponse(status: Int, data: Data) -> MCPToolResult {
    let text = String(data: data, encoding: .utf8) ?? "{}"
    if status == 200 {
      // Surface parsed fields as structuredContent (not a JSON string), and pin
      // a canonical status vocabulary: map "succeeded" -> "completed" so all
      // consumers see one done-state.
      if var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
        Self.normalizeStatus(&obj)
        let normalized = (try? JSONSerialization.data(withJSONObject: obj)) ?? data
        let normalizedText = String(data: normalized, encoding: .utf8) ?? text
        return MCPToolResult(text: normalizedText, structuredJSON: normalized)
      }
      return MCPToolResult(text: text, structuredJSON: data)
    } else {
      // Try to extract error message from JSON response
      if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
         let message = json["error"] as? String {
        return MCPToolResult(error: "Error (\(status)): \(message)")
      }
      return MCPToolResult(error: "Error (\(status)): \(text)")
    }
  }

  /// Canonical done-state: map any "succeeded" status/state field to "completed".
  private static func normalizeStatus(_ obj: inout [String: Any]) {
    for key in ["status", "state"] {
      if let s = obj[key] as? String, s == "succeeded" { obj[key] = "completed" }
    }
  }
}
