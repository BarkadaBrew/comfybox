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
      case "compose_montage":
        return try await executeComposeMontage(arguments)
      case "render_storyboard":
        return try await executeRenderStoryboard(arguments)
      case "import_workflow":
        return try await executeImportWorkflow(arguments)
      case "list_workflows":
        return try await executeGet("/v1/workflows")
      case "run_workflow":
        return try await executeRunWorkflow(arguments)
      case "workflow_run_status":
        guard let runId = arguments?.string("run_id"), !runId.isEmpty else {
          return MCPToolResult(error: "Error: 'run_id' is required")
        }
        return try await executeGet("/v1/workflows/runs/\(runId)")
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
    if let maskRegion = params?.string("mask_region") {
      body["mask_region"] = maskRegion
    }
    if let maskInvert = params?.bool("mask_invert") {
      body["mask_invert"] = maskInvert
    }
    if let maskGrow = params?.integer("mask_grow") {
      body["mask_grow"] = maskGrow
    }
    if let maskFeather = params?.integer("mask_feather") {
      body["mask_feather"] = maskFeather
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
    // T2V has no source frame, so translate aspect_ratio into explicit
    // width/height here: WarmServer's video request decodes width/height, NOT
    // aspect_ratio, so otherwise every t2v render fell back to the 704x448
    // landscape default regardless of the requested orientation. Portrait
    // ("9:16") swaps to 448x704 (both /64). I2V leaves dims unset so WarmServer
    // keeps matching the source image aspect.
    if params?.string("image_path") == nil {
      // Respect caller-supplied dims; otherwise orient the standard t2v dims by
      // aspect_ratio. WarmServer reads width/height (not aspect_ratio/resolution),
      // so without this every t2v fell back to the 704x448 landscape default.
      if let w = params?.integer("width"), let h = params?.integer("height") {
        body["width"] = w
        body["height"] = h
      } else {
        let portrait = (params?.string("aspect_ratio") ?? "16:9") == "9:16"
        body["width"] = portrait ? 448 : 704
        body["height"] = portrait ? 704 : 448
      }
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

  /// compose_montage -> POST /v1/montage/compose (sync — compositing is cheap).
  /// The params object IS the wire payload (segments/transitions/output/
  /// aspect_policy, already snake_case), so it forwards verbatim.
  private func executeComposeMontage(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let params, params.raw["segments"] != nil else {
      return MCPToolResult(error: "Error: 'segments' is required")
    }
    let jsonData = try JSONEncoder().encode(params.raw)
    let (status, data) = try await client.post("/v1/montage/compose", body: jsonData)
    if status == 200 {
      return MCPToolResult(text: String(data: data, encoding: .utf8) ?? "{}")
    }
    return mapHTTPResponse(status: status, data: data)
  }

  /// render_storyboard -> POST /v1/storyboard/render (202 + job id; poll
  /// video_status). Params forward verbatim — they ARE the wire spec.
  private func executeRenderStoryboard(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let params, params.raw["shots"] != nil else {
      return MCPToolResult(error: "Error: 'shots' is required")
    }
    let jsonData = try JSONEncoder().encode(params.raw)
    let (status, data) = try await client.post("/v1/storyboard/render", body: jsonData)
    if status == 200 || status == 202 {
      return MCPToolResult(text: String(data: data, encoding: .utf8) ?? "{}")
    }
    return mapHTTPResponse(status: status, data: data)
  }

  /// import_workflow -> POST /v1/workflows/import
  private func executeImportWorkflow(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let workflowJson = params?.string("workflow_json"), !workflowJson.isEmpty else {
      return MCPToolResult(error: "Error: 'workflow_json' is required")
    }
    var body: [String: Any] = ["workflow_json": workflowJson]
    if let name = params?.string("name") { body["name"] = name }
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/workflows/import", body: jsonData)
    if status == 200 {
      return MCPToolResult(text: String(data: data, encoding: .utf8) ?? "{}")
    }
    return mapHTTPResponse(status: status, data: data)
  }

  /// run_workflow -> POST /v1/workflows/{id}/run (202 + run_id), then poll the
  /// status route briefly so fast (turbo) renders return inline. Long renders
  /// return {status: running, run_id} — poll workflow_run_status.
  private func executeRunWorkflow(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let workflowId = params?.string("workflow_id"), !workflowId.isEmpty else {
      return MCPToolResult(error: "Error: 'workflow_id' is required")
    }
    var body: [String: Any] = [:]
    if let prompt = params?.string("prompt") { body["prompt"] = prompt }
    if let negative = params?.string("negative_prompt") { body["negative_prompt"] = negative }
    if let seed = params?.integer("seed") { body["seed"] = seed }
    if let outputPath = params?.string("output_path") { body["output_path"] = outputPath }
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/workflows/\(workflowId)/run", body: jsonData)
    guard status == 200 || status == 202 else {
      return mapHTTPResponse(status: status, data: data)
    }
    guard let submitted = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let runId = submitted["run_id"] as? String else {
      return MCPToolResult(text: String(data: data, encoding: .utf8) ?? "{}")
    }

    // Inline-wait window: covers turbo renders (~10-90s). Beyond it, hand the
    // run_id back for workflow_run_status polling.
    let deadline = Date().addingTimeInterval(120)
    while Date() < deadline {
      try await Task.sleep(nanoseconds: 2_000_000_000)
      let (pollStatus, pollData) = try await client.get("/v1/workflows/runs/\(runId)")
      guard pollStatus == 200,
            let state = try? JSONSerialization.jsonObject(with: pollData) as? [String: Any],
            let runState = state["status"] as? String else { continue }
      if runState == "succeeded" || runState == "failed" {
        return MCPToolResult(text: String(data: pollData, encoding: .utf8) ?? "{}")
      }
    }
    return MCPToolResult(text: String(
      data: try JSONSerialization.data(withJSONObject: [
        "run_id": runId, "status": "running",
        "note": "Render still in progress — poll workflow_run_status with this run_id.",
      ]), encoding: .utf8) ?? "{}")
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
