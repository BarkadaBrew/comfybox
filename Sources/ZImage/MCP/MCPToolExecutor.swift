// MCPToolExecutor.swift — Executes tool calls by proxying to WarmServer
//
// Maps MCP tool names to WarmServer REST endpoints.
// Uses WarmServerClient for all HTTP calls except apply_style,
// which resolves locally using ComfyBoxStylePresets.

import Foundation

/// Executes MCP tool calls by dispatching to WarmServer HTTP endpoints.
public final class MCPToolExecutor: @unchecked Sendable {
  private let client: WarmServerTransport

  public init(client: WarmServerTransport) {
    self.client = client
  }

  /// Execute a tool call. Returns an MCPToolResult.
  ///
  /// `progress` is non-nil only when the client supplied a `progressToken` in
  /// the request's `_meta` (comfybox#292). It is optional with a default so
  /// every existing call site is unchanged.
  public func execute(
    name: String, arguments: MCPParams?, progress: MCPProgressReporter? = nil
  ) async -> MCPToolResult {
    do {
      switch name {
      case "generate_image":
        return try await executeGenerateImage(arguments, progress: progress)
      case "get_job":
        return try await executeGetJob(arguments)
      case "repair_image":
        return try await executeRepairImage(arguments)
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
      case "pause_queue":
        return try await executeQueuePause(true)
      case "resume_queue":
        return try await executeQueuePause(false)
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
      case "rerender_video":
        return try await executeWinnerAction(arguments, route: "/v1/video/rerender")
      case "extend_video":
        return try await executeWinnerAction(arguments, route: "/v1/video/extend")
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
      case "nearline_anchor":
        return try await executeNearlineAnchor(arguments)
      case "nearline_evict":
        return try await executeNearlineAction("/v1/nearline/evict", arguments)
      case "civitai_search":
        return try await executeCivitAISearch(arguments)
      case "civitai_prompts":
        return try await executeCivitAIPrompts(arguments)
      case "move_queue_job":
        return try await executeMoveQueueJob(arguments)
      case "update_lora_triggerwords":
        return try await executeUpdateLoraTriggerwords(arguments)
      case "create_preset":
        return try await executeCreatePreset(arguments)
      case "delete_preset":
        return try await executeDeletePreset(arguments)
      case "set_warm_preset":
        return try await executeSetWarmPreset(arguments)
      case "create_character":
        return try await executeCreateCharacter(arguments)
      case "delete_character":
        return try await executeDeleteCharacter(arguments)
      case "get_config":
        return try await executeGet("/v1/config")
      case "patch_config":
        return try await executePatchConfig(arguments)
      case "update_config":
        return try await executeUpdateConfig(arguments)
      default:
        return MCPToolResult(error: "Unknown tool: \(name)")
      }
    } catch let error as WarmServerClientError {
      // comfybox#153 review round 2, point 1: the bridge no longer exits
      // when nothing is listening at startup — it warns once and serves
      // anyway. Every tool call made before the engine comes up must fail
      // with that SAME message (port + launchctl kickstart command), not
      // the generic "WarmServer not running" text, via the one shared
      // builder both call sites use.
      if case .connectionRefused(let host, let port) = error {
        return MCPToolResult(error: MCPBridgeStartupPolicy.nothingListeningMessage(host: host, port: port))
      }
      return MCPToolResult(error: "Error: \(error.errorDescription ?? String(describing: error))")
    } catch {
      return MCPToolResult(error: "Error: \(error.localizedDescription)")
    }
  }

  // MARK: - Tool Implementations

  /// generate_image -> POST /v1/generate
  /// Ask the local vision model (LM Studio / OpenAI-compatible /chat/completions
  /// on the Mac) to diagnose RENDER defects in an image. Fully on-device — no
  /// coffeeshop-server reliance. Returns a concise defect description, "CLEAN",
  /// or nil if vision is unreachable.
  private func diagnoseDefects(imagePath: String) async -> String? {
    let visionURL = ProcessInfo.processInfo.environment["COMFYBOX_VISION_URL"] ?? "http://127.0.0.1:1234/v1"
    var model = ProcessInfo.processInfo.environment["COMFYBOX_VISION_MODEL"] ?? ""
    if model.isEmpty, let mu = URL(string: visionURL + "/models"),
       let (md, _) = try? await URLSession.shared.data(from: mu),
       let mo = try? JSONSerialization.jsonObject(with: md) as? [String: Any],
       let arr = mo["data"] as? [[String: Any]] {
      let ids = arr.compactMap { $0["id"] as? String }
      model = ids.first(where: { $0.lowercased().contains("vl") || $0.lowercased().contains("vision") }) ?? ids.first ?? "local-model"
    }
    if model.isEmpty { model = "local-model" }
    let resolved = (imagePath as NSString).expandingTildeInPath
    guard let imgData = try? Data(contentsOf: URL(fileURLWithPath: resolved)),
          let url = URL(string: visionURL + "/chat/completions") else { return nil }
    let b64 = imgData.base64EncodedString()
    let question = "You are inspecting an AI-generated adult photo for RENDER defects only (not content or subject matter). List concise, concrete defects and WHERE each is located. Look for: mottled/blotchy/damaged skin, disfigured or deformed anatomy, extra/missing/fused fingers or limbs, warped or melted face, mesh/cross-hatch texture artifacts, color banding. Report locations using: face, hands, torso, legs, background. If there are NO render defects, reply with exactly the single word CLEAN."
    let payload: [String: Any] = [
      "model": model,
      "messages": [["role": "user", "content": [
        ["type": "text", "text": question],
        ["type": "image_url", "image_url": ["url": "data:image/png;base64,\(b64)"]],
      ] as [Any]] as [String: Any]],
      "max_tokens": 300, "temperature": 0.2,
    ]
    guard let body = try? JSONSerialization.data(withJSONObject: payload) else { return nil }
    var req = URLRequest(url: url); req.httpMethod = "POST"
    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
    req.httpBody = body; req.timeoutInterval = 90
    guard let (data, resp) = try? await URLSession.shared.data(for: req),
          let http = resp as? HTTPURLResponse, http.statusCode == 200,
          let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let choices = obj["choices"] as? [[String: Any]],
          let msg = choices.first?["message"] as? [String: Any],
          let content = msg["content"] as? String else { return nil }
    return content.trimmingCharacters(in: .whitespacesAndNewlines)
  }

  /// Repair a defective image on-device: local VLM diagnoses -> targeted img2img
  /// re-render (defect negatives + optional region inpaint), preserving composition
  /// and identity. No coffeeshop-server reliance.
  private func executeRepairImage(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let imagePath = params?.string("image_path"), !imagePath.isEmpty else {
      return MCPToolResult(error: "Error: 'image_path' is required")
    }
    // 1. Diagnose (local VLM) unless the caller supplied a note describing the defect.
    let userNote = params?.string("note")
    let diagnosis = await diagnoseDefects(imagePath: imagePath)
    FileHandle.standardError.write(Data("[repair_image] diagnosis: \(diagnosis ?? "(vision unreachable)")\n".utf8))

    // 2. Targeted negative = defect baseline + the VLM's findings + the user note.
    var negParts = ["mottled skin", "damaged skin", "blotchy skin", "discolored skin",
      "disfigured", "deformed anatomy", "distorted anatomy", "extra fingers", "fused fingers",
      "extra limbs", "missing limbs", "mutated", "malformed", "bad anatomy",
      "mesh artifacts", "cross-hatch texture", "blurry", "noise", "artifacts"]
    if let d = diagnosis, d.uppercased() != "CLEAN", !d.isEmpty { negParts.append(d) }
    if let n = userNote, !n.isEmpty { negParts.append(n) }
    let negative = negParts.joined(separator: ", ")

    // 3. Localize an inpaint region from the diagnosis/note keywords.
    let hay = ((diagnosis ?? "") + " " + (userNote ?? "")).lowercased()
    var maskRegion: String? = nil
    if hay.contains("face") || hay.contains("eye") || hay.contains("head") {
      maskRegion = "face"
    } else if hay.contains("hand") || hay.contains("finger") || hay.contains("arm")
              || hay.contains("chest") || hay.contains("torso") || hay.contains("breast") {
      maskRegion = "upper"
    } else if hay.contains("leg") || hay.contains("thigh") || hay.contains("foot")
              || hay.contains("hip") || hay.contains("genital") || hay.contains("vagina") {
      maskRegion = "lower"
    }

    // 4. img2img repair render (source-preserving strength keeps composition + identity).
    var body: [String: Any] = [
      "prompt": "smooth even skin, flawless skin, correct anatomy, natural proportions, photorealistic, sharp detail, Pinay",
      "negative_prompt": negative,
      "image_path": imagePath,
      "image_strength": params?.number("image_strength") ?? 0.6,
    ]
    if let mr = maskRegion { body["mask_region"] = mr }
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/generate", body: jsonData)
    guard status == 200, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
      return Self.mapHTTPResponse(status: status, data: data)
    }
    // Mark the rerender (Todd) + base64-encode so the caller delivers WITHOUT
    // server file access — fully on-device product.
    var outPath = (obj["image_path"] as? String) ?? (obj["local_path"] as? String) ?? (obj["output_path"] as? String)
    if let p = outPath {
      let ns = p as NSString
      let renamed = "\(ns.deletingPathExtension)-rerender.\(ns.pathExtension)"
      if (try? FileManager.default.moveItem(atPath: p, toPath: renamed)) != nil {
        try? FileManager.default.moveItem(atPath: "\((p as NSString).deletingPathExtension).json", toPath: "\((renamed as NSString).deletingPathExtension).json")
        outPath = renamed
      }
    }
    let b64 = outPath.flatMap { try? Data(contentsOf: URL(fileURLWithPath: $0)) }?.base64EncodedString() ?? ""
    let out: [String: Any] = ["image_path": outPath ?? "", "image_base64": b64, "diagnosis": diagnosis ?? "", "mask_region": maskRegion ?? ""]
    let outData = (try? JSONSerialization.data(withJSONObject: out)) ?? Data("{}".utf8)
    return MCPToolResult(text: String(data: outData, encoding: .utf8) ?? "{}")
  }

  private func executeGenerateImage(
    _ params: MCPParams?, progress: MCPProgressReporter? = nil
  ) async throws -> MCPToolResult {
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
    let returnImage = params?.bool(MCPImageAttachment.parameterName) ?? false
    let client = self.client

    // #288: additive async submit. `async: true` hands back a job_id from
    // POST /v1/generate/async immediately instead of holding the MCP call
    // open for the whole render — a slow render behind a busy queue used to
    // sit inside the client's 300s tool timeout and lose a result the engine
    // had actually produced. The DEFAULT is unchanged (synchronous).
    if params?.bool("async") == true {
      return try await Self.runSubmitImageJob(body: jsonData) { _, path, requestBody in
        try await client.post(path, body: requestBody)
      }
    }

    // #292: while the client supplied a progressToken, narrate the render
    // from GET /v1/queue (lock-based — it answers even while a render holds
    // the coordinator actor, comfybox#217). With no token this is a straight
    // pass-through and the engine is never polled.
    let (status, data) = try await Self.withProgressNotifications(
      reporter: progress,
      poll: {
        guard let (queueStatus, queueData) = try? await client.get("/v1/queue"),
          queueStatus == 200
        else { return nil }
        return MCPProgressScheduler.Snapshot(queuePayload: queueData)
      }
    ) {
      try await client.post("/v1/generate", body: jsonData)
    }
    return Self.mapImageRenderResponse(status: status, data: data, returnImage: returnImage)
  }

  /// get_job -> the one polling tool (#289). Routes by kind to the image or
  /// video tracker and returns the unified envelope; see `MCPJobModel`.
  private func executeGetJob(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let jobId = params?.string("job_id"), !jobId.isEmpty else {
      return MCPToolResult(error: "Error: 'job_id' is required")
    }
    var kind: MCPJobKind?
    if let raw = params?.string("kind"), !raw.isEmpty {
      // Only the SELECTABLE kinds: `swap` is not one, because a swap job id
      // only exists after queue replay and resolves through the image
      // tracker (review r1, item 3). Omitting `kind` finds it either way.
      guard let parsed = MCPJobKind(rawValue: raw),
        MCPJobKind.selectableCases.contains(parsed)
      else {
        return MCPToolResult(
          error: "Error: unknown 'kind' \(raw) — expected one of "
            + MCPJobKind.selectableCases.map(\.rawValue).joined(separator: ", ")
            + " (omit it to probe image then video; LoRA-swap ids resolve as 'image')")
      }
      kind = parsed
    }
    let returnImage = params?.bool(MCPImageAttachment.parameterName) ?? false
    let client = self.client
    return try await Self.runGetJob(jobId: jobId, kind: kind, returnImage: returnImage) {
      _, path in
      try await client.get(path)
    }
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
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// clear_queue -> POST /queue with {"clear": true}
  private func executeClearQueue() async throws -> MCPToolResult {
    let body: [String: Any] = ["clear": true]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/queue", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// pause_queue / resume_queue -> POST /v1/queue/pause | /v1/queue/resume
  /// (the same persistent gate the desktop toolbar and HTTP API use).
  private func executeQueuePause(_ pause: Bool) async throws -> MCPToolResult {
    let (status, data) = try await client.post(pause ? "/v1/queue/pause" : "/v1/queue/resume", body: Data())
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// list_loras -> GET /object_info, extract LoraLoader lora_name options
  private func executeListLoras() async throws -> MCPToolResult {
    let (status, data) = try await client.get("/object_info")

    guard status == 200 else {
      return Self.mapHTTPResponse(status: status, data: data)
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
    return Self.mapHTTPResponse(status: status, data: data)
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
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// lora_scan -> POST /v1/loras/scan
  private func executeLoraScan(_ params: MCPParams?) async throws -> MCPToolResult {
    var body: [String: Any] = [:]
    if params?.bool("force") == true {
      body["force"] = true
    }
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/loras/scan", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
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
      return Self.mapHTTPResponse(status: status, data: data)
    } else {
      let (status, data) = try await client.delete(path)
      return Self.mapHTTPResponse(status: status, data: data)
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
    // #339 review r2, item 2: `wait: false` returns 202 (async load in
    // progress) and is gated by the recovery gate the same as local video —
    // retry that specific refusal with backoff instead of surfacing it on
    // the first attempt. `wait: true` (200, synchronous, never gated) and
    // any OTHER error return immediately either way.
    return try await postWithQueueRecoveryRetry("/v1/model/load", body: jsonData)
  }

  /// switch_model -> POST /v1/model/activate
  private func executeSwitchModel(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let model = params?.string("model"), !model.isEmpty else {
      return MCPToolResult(error: "Error: 'model' is required")
    }
    let body: [String: Any] = ["model": model]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/model/activate", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }


  /// unload_model -> POST /v1/model/unload
  private func executeUnloadModel(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let model = params?.string("model"), !model.isEmpty else {
      return MCPToolResult(error: "Error: 'model' is required")
    }
    let body: [String: Any] = ["model": model]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/model/unload", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
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
    if let audio = params?.bool("audio") {
      body["audio"] = audio
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
    // Caller already wove the character description into the prompt, so tell
    // WarmServer not to prepend its own (Todd 2026-08-07). This body is an
    // explicit whitelist — an unforwarded key is silently dropped here, which
    // is exactly what happened on the first attempt at this fix.
    if let skipChar = params?.bool("skip_character_injection") {
      body["skip_character_injection"] = skipChar
    }
    // Motion vs fidelity per content type (#40): the daemon sends a high
    // img_compression + lower strength for partnered-action prompts (motion)
    // and low compression + strength 1.0 for solo/portrait (fidelity).
    if let strength = params?.number("strength") {
      body["strength"] = strength
    }
    if let comp = params?.integer("img_compression") {
      body["img_compression"] = comp
    }
    if let cfg = params?.number("guidance") {
      body["guidance"] = cfg
    }
    if let tuning = params?.dict("tuning") {
      body["tuning"] = tuning.mapValues { $0.value }
    }
    // Run overrides (coffeeshop-server#1751): request LoRAs replace the
    // preset/default stack; fps shifts the generation frame-rate basis. The
    // body is an explicit whitelist, so these must be forwarded by name.
    if let loras = params?.array("loras") {
      body["loras"] = loras.map { $0.value }
    }
    if let fps = params?.integer("fps") {
      body["fps"] = fps
    }
    if let attemptId = params?.string("optimization_attempt_id") {
      body["optimization_attempt_id"] = attemptId
    }
    if let frames = params?.integer("frames") {
      body["frames"] = frames
    }
    if let neg = params?.string("negative_prompt") {
      body["negative_prompt"] = neg
    }
    // Callers that ran their OWN optimizer send enhance:false — a second
    // server-side rewrite drifts the prompt off concrete staging (limb
    // placement, figure count) and double-injects the character description.
    if let enhance = params?.bool("enhance") {
      body["enhance"] = enhance
    }
    // comfybox#328: another explicit-whitelist key — an unforwarded
    // beat_schedule vanishes here exactly as skip_character_injection once
    // did, and the daemon (coffeeshop-server#1753) never even reaches
    // WarmServer with beats to lose to enhancement.
    if let beatSchedule = params?.array("beat_schedule") {
      body["beat_schedule"] = beatSchedule.map { $0.value }
    }

    let jsonData = try JSONSerialization.data(withJSONObject: body)
    // Async route for BOTH backends: local renders return 202 + job_id
    // immediately (poll video_status), instead of blocking the MCP call for
    // the entire multi-minute render — which overran the daemon-side 300s
    // MCP tool timeout on every cold-start render (#219).
    // #339 review r1, item 4: local video can also be refused with a
    // retryable 503 while the engine is replaying its persisted queue after
    // a restart — retry that specific refusal with backoff rather than
    // surfacing it as a bare tool error on the first attempt.
    return try await postWithQueueRecoveryRetry("/v1/video/generate/async", body: jsonData)
  }

  /// rerender_video / extend_video -> POST /v1/video/{rerender,extend}
  /// (202 + job id; poll video_status). Params forward verbatim — the route
  /// resolves render_id/path against the trace store server-side.
  private func executeWinnerAction(_ params: MCPParams?, route: String) async throws -> MCPToolResult {
    guard let params, params.raw["render_id"] != nil || params.raw["path"] != nil else {
      return MCPToolResult(error: "Error: 'render_id' or 'path' is required")
    }
    let jsonData = try JSONEncoder().encode(params.raw)
    // #339 review r1, item 4: same queue-recovery retry as generate_video —
    // both winner actions submit local video underneath.
    return try await postWithQueueRecoveryRetry(route, body: jsonData)
  }

  /// #339 review r1, item 4: POST a request that can be refused by the
  /// queue-recovery gate (`QueueRecoveryGate`: HTTP 503 + `error_code:
  /// "queue_recovery_in_progress"`). Retries using the server's own
  /// `retry_after_seconds` hint (`QueueRecoveryRetryPolicy`, pure and tested
  /// separately) for up to 15 minutes total elapsed, then surfaces a clear
  /// tool error naming why instead of retrying forever. Any OTHER response —
  /// success, or a DIFFERENT non-retryable error/status — is mapped through
  /// `mapHTTPResponse` immediately on the first attempt, unchanged.
  private func postWithQueueRecoveryRetry(
    _ path: String, body: Data, successStatuses: Set<Int> = [200, 202]
  ) async throws -> MCPToolResult {
    let start = Date()
    while true {
      let (status, data) = try await client.post(path, body: body)
      if successStatuses.contains(status) {
        return MCPToolResult(text: String(data: data, encoding: .utf8) ?? "{}")
      }
      guard status == 503, Self.isQueueRecoveryRefusal(data) else {
        return Self.mapHTTPResponse(status: status, data: data)
      }
      let elapsed = Date().timeIntervalSince(start)
      switch QueueRecoveryRetryPolicy.decide(elapsed: elapsed, retryAfterSeconds: Self.parseRetryAfterSeconds(from: data)) {
      case .retry(let wait):
        try await Task.sleep(nanoseconds: UInt64(max(0, wait) * 1_000_000_000))
      case .giveUp(_, _, let reason):
        // #339 review r2, item 3: `reason` already names `error_code` and
        // `retry_after_seconds` in prose so the daemon can reschedule this
        // call without re-parsing the earlier HTTP response.
        return MCPToolResult(error: reason)
      }
    }
  }

  /// Distinguishes the specific, retryable "engine is still replaying its
  /// persisted queue" 503 from any OTHER 503 (e.g. "LTX-2 not configured",
  /// which retrying can never fix).
  private static func isQueueRecoveryRefusal(_ data: Data) -> Bool {
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
    return (json["error_code"] as? String) == QueueRecoveryGate.errorCode
  }

  private static func parseRetryAfterSeconds(from data: Data) -> Int? {
    guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
    return json["retry_after_seconds"] as? Int
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
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// render_storyboard -> POST /v1/storyboard/render (202 + job id; poll
  /// video_status). Params forward verbatim — they ARE the wire spec.
  private func executeRenderStoryboard(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let params, params.raw["shots"] != nil else {
      return MCPToolResult(error: "Error: 'shots' is required")
    }
    let jsonData = try JSONEncoder().encode(params.raw)
    // #339 review r1, item 4: storyboards are gated the same as local video.
    return try await postWithQueueRecoveryRetry("/v1/storyboard/render", body: jsonData)
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
    return Self.mapHTTPResponse(status: status, data: data)
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
      return Self.mapHTTPResponse(status: status, data: data)
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
    return Self.mapHTTPResponse(status: status, data: data)
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
    return Self.mapHTTPResponse(status: status, data: data)
  }

  // MARK: - Headless parity Phase 1 (comfybox#300, FDD §4.2) — gap-set tools

  /// move_queue_job -> POST /v1/queue/{id}/move { direction }. `direction`
  /// is validated client-side against the declared enum: WarmServer's
  /// `movePending` (WarmServer.swift:6965) silently no-ops (200, moved:false)
  /// on an unrecognized direction rather than 400ing, so an unknown value
  /// has to be caught here to honor "unknown enum members produce a clean
  /// 400, not a trap" (FDD scope note).
  private func executeMoveQueueJob(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let id = params?.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    guard let direction = params?.string("direction"), !direction.isEmpty else {
      return MCPToolResult(error: "Error: 'direction' is required")
    }
    let allowedDirections: Set<String> = ["top", "up", "down"]
    guard allowedDirections.contains(direction) else {
      return MCPToolResult(error: "Error: 'direction' must be one of top, up, down (got '\(direction)')")
    }
    let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
    let body: [String: Any] = ["direction": direction]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/queue/\(encoded)/move", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// update_lora_triggerwords -> POST /v1/loras/{id}/update { triggerwords }
  private func executeUpdateLoraTriggerwords(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let id = params?.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    guard let triggerwordsArray = params?.array("triggerwords") else {
      return MCPToolResult(error: "Error: 'triggerwords' (array of strings) is required")
    }
    let triggerwords = triggerwordsArray.compactMap(\.stringValue)
    let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
    let body: [String: Any] = ["triggerwords": triggerwords]
    let jsonData = try JSONSerialization.data(withJSONObject: body)
    let (status, data) = try await client.post("/v1/loras/\(encoded)/update", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// create_preset -> POST /v1/presets. Params ARE the wire payload (decoded
  /// server-side with convertFromSnakeCase), forwarded verbatim — same
  /// pattern as compose_montage/render_storyboard — because ImagePreset has
  /// ~30 optional fields and a hand-maintained whitelist here would silently
  /// drop new ones (exactly the `skip_character_injection` failure mode
  /// noted on generate_video). Server requires non-empty id + name
  /// (PresetStore.swift:854-857); validated client-side too for a fast,
  /// clean error before the round trip.
  private func executeCreatePreset(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let params, let id = params.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    guard let name = params.string("name"), !name.isEmpty else {
      return MCPToolResult(error: "Error: 'name' is required")
    }
    let jsonData = try JSONEncoder().encode(params.raw)
    let (status, data) = try await client.post("/v1/presets", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// delete_preset -> DELETE /v1/presets/{id}
  private func executeDeletePreset(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let id = params?.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
    let (status, data) = try await client.delete("/v1/presets/\(encoded)")
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// set_warm_preset -> composite action mirroring the Desktop app's
  /// PresetView.setAsWarm (ComfyBoxDesktop/Views/PresetView.swift:207-219)
  /// exactly: activate the model (falling back to a load if it isn't in the
  /// pool yet — PresetView's do/catch), THEN fetch the server config,
  /// patch modelSpec, and save it back. The composite body lives in
  /// `runSetWarmPreset` below, parameterized over an injectable HTTP call so
  /// the order and abort-on-activation-failure contract are unit-testable
  /// without a real server.
  private func executeSetWarmPreset(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let model = params?.string("model"), !model.isEmpty else {
      return MCPToolResult(error: "Error: 'model' is required")
    }
    let client = self.client
    return try await Self.runSetWarmPreset(model: model) { method, path, body, headers in
      try await client.send(method: method, path: path, body: body, headers: headers)
    }
  }

  /// The set_warm_preset composite, isolated from `WarmServerClient` behind
  /// an injectable `call` closure. `internal` (not `private`) so tests can
  /// invoke it directly with a fake `call` and assert exact order without
  /// networking. Mirrors PresetView.setAsWarm:
  ///   1. POST /v1/model/activate { model }
  ///   2. IF (1) fails: POST /v1/model/load { model, activate: true, wait: true }
  ///      (PresetView's catch-and-load fallback). If this also fails, abort —
  ///      the config is never read or written.
  ///   3. GET /v1/config — capturing the response `ETag`
  ///   4. PUT /v1/config with modelSpec set to `model` (fetch-then-mutate-
  ///      then-save, because PUT is a whole-document replace — WarmServer's
  ///      `encode(to:)` only writes enumerated keys, so saving a document
  ///      that wasn't first fetched would drop unrelated config fields),
  ///      sending the captured ETag as `If-Match` (adversarial review F2,
  ///      2026-08-30): an unconditional PUT here silently CLOBBERS any
  ///      concurrent `PATCH /v1/config` that lands between the GET and the
  ///      PUT — the exact lost-update the store's advisory ETag exists to
  ///      catch. On `409`, re-GET (picking up the concurrent write), re-apply
  ///      only our modelSpec mutation on the fresh document, and retry ONCE —
  ///      bounded, so a pathological write storm degrades to a clean 409
  ///      error rather than an unbounded loop. Absent ETag (older server) →
  ///      unconditional PUT, exactly today's behavior.
  static func runSetWarmPreset(
    model: String,
    call: (_ method: String, _ path: String, _ body: Data, _ headers: [String: String]) async throws -> (Int, Data, [String: String])
  ) async throws -> MCPToolResult {
    let activateBody = try JSONSerialization.data(withJSONObject: ["model": model])
    let (activateStatus, activateData, _) = try await call("POST", "/v1/model/activate", activateBody, [:])
    if activateStatus != 200 {
      let loadBody = try JSONSerialization.data(withJSONObject: ["model": model, "activate": true, "wait": true])
      let (loadStatus, loadData, _) = try await call("POST", "/v1/model/load", loadBody, [:])
      guard loadStatus == 200 || loadStatus == 202 else {
        return Self.mapHTTPResponse(status: loadStatus, data: loadData)
      }
    }
    // GET → mutate → conditional PUT, with ONE bounded retry on 409.
    var lastStatus = 0
    var lastData = Data()
    for attempt in 0..<2 {
      let (getStatus, getData, getHeaders) = try await call("GET", "/v1/config", Data(), [:])
      guard getStatus == 200,
            var configObj = try? JSONSerialization.jsonObject(with: getData) as? [String: Any] else {
        return Self.mapHTTPResponse(status: getStatus, data: getData)
      }
      configObj["modelSpec"] = model
      let putBody = try JSONSerialization.data(withJSONObject: configObj)
      // Header names are case-insensitive per RFC 9110; fake closures in
      // tests and the real client may differ in casing.
      let etag = getHeaders.first { $0.key.caseInsensitiveCompare("ETag") == .orderedSame }?.value
      let putHeaders: [String: String] = etag.map { ["If-Match": $0] } ?? [:]
      let (putStatus, putData, _) = try await call("PUT", "/v1/config", putBody, putHeaders)
      if putStatus == 409, attempt == 0 {
        // A concurrent write landed between our GET and PUT. Loop: re-fetch
        // the (now newer) document — PRESERVING that write — and re-apply
        // only our own modelSpec mutation on top of it.
        lastStatus = putStatus
        lastData = putData
        continue
      }
      return Self.mapHTTPResponse(status: putStatus, data: putData)
    }
    return Self.mapHTTPResponse(status: lastStatus, data: lastData)
  }

  /// create_character -> POST /v1/characters. Params ARE the wire payload
  /// (CharacterEntry decodes convertFromSnakeCase server-side), forwarded
  /// verbatim like create_preset. 'name' is the only field the server
  /// actually requires — 'id' defaults to a slug of 'name'
  /// (CharacterStore.swift:60-62, WarmServer.swift:2967-2968).
  private func executeCreateCharacter(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let params, let name = params.string("name"), !name.isEmpty else {
      return MCPToolResult(error: "Error: 'name' is required")
    }
    let jsonData = try JSONEncoder().encode(params.raw)
    let (status, data) = try await client.post("/v1/characters", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// delete_character -> DELETE /v1/characters/{id}
  private func executeDeleteCharacter(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let id = params?.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
    let (status, data) = try await client.delete("/v1/characters/\(encoded)")
    return Self.mapHTTPResponse(status: status, data: data)
  }

  // MARK: - Headless parity Phase 3 (comfybox#300, FDD §3.3/§4.4) — config

  /// patch_config -> PATCH /v1/config { ...merge-patch document... }. `patch`
  /// IS the wire body verbatim (RFC 7386 JSON Merge Patch) — forwarded, not
  /// interpreted here, so nested-null deletes and nested-object merges survive
  /// exactly as the caller wrote them.
  private func executePatchConfig(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let patch = params?.dict("patch") else {
      return MCPToolResult(error: "Error: 'patch' (a JSON merge-patch object) is required")
    }
    let jsonData = try JSONEncoder().encode(patch)
    let (status, data) = try await client.patch("/v1/config", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// update_config -> PUT /v1/config { ...full document... }. `config` IS the
  /// wire payload verbatim — a full-document replace, same "forward the whole
  /// object" convention as create_preset/create_character. Prefer
  /// patch_config for changing one or a few fields.
  private func executeUpdateConfig(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let config = params?.dict("config") else {
      return MCPToolResult(error: "Error: 'config' (the full config document) is required")
    }
    let jsonData = try JSONEncoder().encode(config)
    let (status, data) = try await client.put("/v1/config", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  // MARK: - Helpers

  /// Generic GET endpoint handler.
  private func executeGet(_ path: String) async throws -> MCPToolResult {
    let (status, data) = try await client.get(path)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// POST an empty JSON body (for trigger-style endpoints).
  private func executePostEmpty(_ path: String) async throws -> MCPToolResult {
    let (status, data) = try await client.post(path, body: Data("{}".utf8))
    return Self.mapHTTPResponse(status: status, data: data)
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
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// cancel_job -> DELETE /v1/queue/{id}
  private func executeCancelJob(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let id = params?.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
    let (status, data) = try await client.delete("/v1/queue/\(encoded)")
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// nearline_stage / nearline_evict -> POST /v1/nearline/{action} { name }
  private func executeNearlineAction(_ path: String, _ params: MCPParams?) async throws -> MCPToolResult {
    guard let name = params?.string("name"), !name.isEmpty else {
      return MCPToolResult(error: "Error: 'name' is required")
    }
    let jsonData = try JSONSerialization.data(withJSONObject: ["name": name])
    let (status, data) = try await client.post(path, body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// nearline_anchor -> POST /v1/nearline/anchor { kind, id, anchored }
  private func executeNearlineAnchor(_ params: MCPParams?) async throws -> MCPToolResult {
    guard let kind = params?.string("kind"), !kind.isEmpty else {
      return MCPToolResult(error: "Error: 'kind' is required")
    }
    guard let id = params?.string("id"), !id.isEmpty else {
      return MCPToolResult(error: "Error: 'id' is required")
    }
    guard let anchored = params?.bool("anchored") else {
      return MCPToolResult(error: "Error: 'anchored' is required")
    }
    let jsonData = try JSONSerialization.data(withJSONObject: ["kind": kind, "id": id, "anchored": anchored])
    let (status, data) = try await client.post("/v1/nearline/anchor", body: jsonData)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  // MARK: - CivitAI conduit (#234)

  /// civitai_search -> GET /v1/civitai/search. Free-text params are
  /// percent-encoded — WarmServer's query-string split happens before any
  /// decoding, and a raw space can't survive an HTTP request line anyway.
  private func executeCivitAISearch(_ params: MCPParams?) async throws -> MCPToolResult {
    let path = "/v1/civitai/search" + Self.civitaiSearchQueryString(params)
    let (status, data) = try await client.get(path)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  /// civitai_prompts -> optional POST /v1/civitai/harvest, then always
  /// GET /v1/civitai/repo with the filter_* params.
  private func executeCivitAIPrompts(_ params: MCPParams?) async throws -> MCPToolResult {
    if params?.bool("harvest") == true {
      var body: [String: Any] = [:]
      if let q = params?.string("query") { body["query"] = q }
      if let types = params?.array("types") {
        body["types"] = types.compactMap(\.stringValue)
      }
      if let baseModel = params?.string("base_model") { body["base_model"] = baseModel }
      if let sort = params?.string("sort") { body["sort"] = sort }
      if let period = params?.string("period") { body["period"] = period }
      if let nsfw = params?.bool("nsfw") { body["nsfw"] = nsfw }
      if let limit = params?.integer("limit") { body["limit"] = limit }
      if let site = params?.string("site") { body["site"] = site }
      let jsonData = try JSONSerialization.data(withJSONObject: body)
      let (harvestStatus, harvestData) = try await client.post("/v1/civitai/harvest", body: jsonData)
      guard harvestStatus == 200 else {
        return Self.mapHTTPResponse(status: harvestStatus, data: harvestData)
      }
    }

    var queryItems: [String] = []
    if let v = params?.string("filter_base_model") {
      queryItems.append("base_model=\(Self.percentEncode(v))")
    }
    if let v = params?.string("filter_act") {
      queryItems.append("act=\(Self.percentEncode(v))")
    }
    if let v = params?.string("filter_tag") {
      queryItems.append("tag=\(Self.percentEncode(v))")
    }
    if let v = params?.string("keyword") {
      queryItems.append("keyword=\(Self.percentEncode(v))")
    }
    if let v = params?.integer("max_entries") {
      // Route-side this is `limit`: default 100, clamped to max 500.
      queryItems.append("limit=\(v)")
    }
    let path = "/v1/civitai/repo" + (queryItems.isEmpty ? "" : "?" + queryItems.joined(separator: "&"))
    let (status, data) = try await client.get(path)
    return Self.mapHTTPResponse(status: status, data: data)
  }

  private static func civitaiSearchQueryString(_ params: MCPParams?) -> String {
    var items: [String] = []
    if let q = params?.string("query") { items.append("query=\(percentEncode(q))") }
    if let types = params?.array("types") {
      let joined = types.compactMap(\.stringValue).joined(separator: ",")
      if !joined.isEmpty { items.append("types=\(percentEncode(joined))") }
    }
    if let baseModel = params?.string("base_model") { items.append("base_model=\(percentEncode(baseModel))") }
    if let sort = params?.string("sort") { items.append("sort=\(percentEncode(sort))") }
    if let period = params?.string("period") { items.append("period=\(percentEncode(period))") }
    if let nsfw = params?.bool("nsfw") { items.append("nsfw=\(nsfw)") }
    if let limit = params?.integer("limit") { items.append("limit=\(limit)") }
    if let site = params?.string("site") { items.append("site=\(percentEncode(site))") }
    return items.isEmpty ? "" : "?" + items.joined(separator: "&")
  }

  private static func percentEncode(_ s: String) -> String {
    s.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) ?? s
  }

  /// Map WarmServer HTTP response to MCP tool result.
  /// 200 -> success (text + structured fields), any other -> error text.
  /// `static` (no `self` use) so `runSetWarmPreset`'s injectable-call
  /// composite can share it without needing an executor instance.
  // MARK: - One job model (#288, #289, #292, #294)

  /// Submit an image render asynchronously and hand back the unified job
  /// envelope. `call` is injectable so the route, the accepted-status
  /// handling and the envelope are unit-testable without a server.
  static func runSubmitImageJob(
    body: Data,
    call: (_ method: String, _ path: String, _ body: Data) async throws -> (Int, Data)
  ) async throws -> MCPToolResult {
    let (status, data) = try await call("POST", "/v1/generate/async", body)
    guard status == 200 || status == 202 else {
      return Self.mapHTTPResponse(status: status, data: data)
    }
    guard let accepted = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let envelope = MCPJobModel.submitEnvelope(kind: .image, status: accepted)
    else {
      // The engine accepted it but named no job — hand back exactly what it
      // said rather than inventing an id the caller could never poll.
      return MCPToolResult(text: String(data: data, encoding: .utf8) ?? "{}")
    }
    return Self.jobResult(envelope, returnImage: false)
  }

  /// The one polling path (#289). Reads the status route for `kind` (probing
  /// image then video when the caller did not name one) and maps whatever
  /// shape comes back onto the single envelope.
  ///
  /// A RUNNING image job costs one extra read of `GET /v1/queue`: the image
  /// tracker carries no per-job percent, and the queue snapshot is the only
  /// live number — but it describes the ACTIVE render, so it is used only
  /// when that render is this job. Queued and finished jobs cost one call.
  static func runGetJob(
    jobId: String,
    kind: MCPJobKind?,
    returnImage: Bool = false,
    call: (_ method: String, _ path: String) async throws -> (Int, Data)
  ) async throws -> MCPToolResult {
    guard !jobId.isEmpty else { return MCPToolResult(error: "Error: 'job_id' is required") }
    let candidates = kind.map { [$0] } ?? MCPJobModel.probeOrder

    let encodedId = jobId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? jobId
    // A probe that fails for a reason OTHER than "not mine" must not end the
    // search — the job may still be in the next tracker. The failure is only
    // surfaced if every probe fails (PR #367 review r1, item 4).
    var lastFailure: (status: Int, data: Data)?

    for candidate in candidates {
      // Spelled out as literals rather than read off MCPJobKind so the §3.5
      // anti-drift parity check — which extracts a tool's real routes by
      // parsing THIS file — can see the paths get_job claims. The two agree
      // by test (MCPJobModelTests drives this function per kind and compares
      // the observed path against MCPJobKind.statusPathTemplate).
      let statusPath: String
      switch candidate {
      case .image, .swap:
        statusPath = "/v1/generate/status/\(encodedId)"
      case .video, .storyboard:
        statusPath = "/v1/video/status/\(encodedId)"
      }
      let (status, data) = try await call("GET", statusPath)
      switch status {
      case 200:
        guard let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
          return MCPToolResult(error: "Error: unreadable status payload for job \(jobId)")
        }
        var queuePercent: Int?
        let state = MCPJobModel.state(fromEngine: (payload["status"] as? String) ?? "")
        if !candidate.carriesOwnProgressPercent, state == .running {
          if let (queueStatus, queueData) = try? await call("GET", "/v1/queue"),
            queueStatus == 200,
            let snapshot = MCPProgressScheduler.Snapshot(queuePayload: queueData),
            snapshot.activeJobId == jobId
          {
            queuePercent = snapshot.progressPercent
          }
        }
        let envelope = MCPJobModel.unify(
          kind: candidate, jobId: jobId, status: payload, queueProgressPercent: queuePercent)
        return Self.jobResult(envelope, returnImage: returnImage)

      case 404:
        // Not this tracker's job — try the next candidate (or fall through
        // to the not-found error below).
        continue

      case 503 where Self.isQueueRecoveryRefusal(data):
        // DEFENSIVE. `QueueRecoveryGate` gates SUBMIT routes today, not the
        // status reads this function makes, so no current engine build
        // reaches here. It stays because the alternative — a future gate on
        // the status route surfacing as a bare error — would read to a
        // polling client as "your job died" when the job is merely queued
        // behind a restart replay. Cheap insurance, one branch
        // (PR #367 review r1, item 4).
        return Self.jobResult(
          MCPJobModel.recoveryEnvelope(
            kind: candidate, jobId: jobId,
            retryAfterSeconds: Self.parseRetryAfterSeconds(from: data)),
          returnImage: returnImage)

      default:
        // Any OTHER status (including a 503 that retrying can never fix,
        // e.g. "LTX-2 not configured") is a real error — but keep probing
        // the remaining trackers first.
        lastFailure = (status, data)
        continue
      }
    }

    if let lastFailure {
      return Self.mapHTTPResponse(status: lastFailure.status, data: lastFailure.data)
    }
    let probed = candidates.map(\.rawValue).joined(separator: ", ")
    return MCPToolResult(
      error: "Error: no job \(jobId) tracked as \(probed). Finished jobs are pruned from the "
        + "tracker an hour after they complete (both trackers, `pruneCompleted` TTL 3600s); "
        + "check the id, or pass 'kind' explicitly.")
  }

  /// Run `work` while emitting MCP progress notifications (#292).
  ///
  /// With no reporter (no `progressToken` in `_meta`) this is a straight
  /// pass-through — `poll` is never called. With one, a child task polls at
  /// `interval` and is cancelled AND awaited before this returns, on success
  /// and on error alike: nothing it starts outlives the request (intent.md).
  static func withProgressNotifications<T>(
    reporter: MCPProgressReporter?,
    interval: TimeInterval = MCPProgressScheduler.defaultInterval,
    maxDuration: TimeInterval = MCPProgressScheduler.maxDuration,
    poll: @escaping @Sendable () async -> MCPProgressScheduler.Snapshot?,
    work: () async throws -> T
  ) async throws -> T {
    guard let reporter else { return try await work() }

    let start = Date()
    let poller = Task<Void, Never> {
      var lastEmitted: Double?
      while !Task.isCancelled {
        do {
          try await Task.sleep(nanoseconds: UInt64(max(0, interval) * 1_000_000_000))
        } catch {
          return  // cancelled while sleeping
        }
        if Task.isCancelled { return }
        let snapshot = await poll()
        switch MCPProgressScheduler.decide(
          elapsed: Date().timeIntervalSince(start), maxDuration: maxDuration,
          lastEmitted: lastEmitted, snapshot: snapshot)
        {
        case .emit(let progress, let total, let message):
          lastEmitted = progress
          await reporter.report(progress: progress, total: total, message: message)
        case .skip:
          continue
        case .stop:
          return
        }
      }
    }

    do {
      let value = try await work()
      poller.cancel()
      await poller.value
      return value
    } catch {
      poller.cancel()
      await poller.value
      throw error
    }
  }

  /// Serialize a unified job envelope into a tool result, attaching the
  /// rendered image when the caller asked for it and the job finished.
  private static func jobResult(_ envelope: [String: Any], returnImage: Bool) -> MCPToolResult {
    let data = (try? JSONSerialization.data(withJSONObject: envelope)) ?? Data("{}".utf8)
    let base = MCPToolResult(
      text: String(data: data, encoding: .utf8) ?? "{}", structuredJSON: data)
    guard returnImage else { return base }

    guard (envelope["state"] as? String) == MCPJobState.completed.rawValue else {
      return Self.withNote(
        "return_image: nothing to attach yet — job state is "
          + "\((envelope["state"] as? String) ?? "unknown").",
        on: base)
    }
    guard let path = (envelope["result"] as? [String: Any])?["output_path"] as? String,
      !path.isEmpty
    else {
      return Self.withNote("return_image: this job reported no output path.", on: base)
    }
    return Self.attachImage(at: path, to: base)
  }

  /// `POST /v1/generate`'s response, plus the PNG as an MCP image block when
  /// `return_image` is set (#294).
  private static func mapImageRenderResponse(
    status: Int, data: Data, returnImage: Bool
  ) -> MCPToolResult {
    let base = Self.mapHTTPResponse(status: status, data: data)
    // A failed render has nothing to attach and stays an error untouched.
    guard returnImage, !base.isError else { return base }
    guard let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
      let path = payload["output_path"] as? String, !path.isEmpty
    else {
      return Self.withNote("return_image: this render reported no output path.", on: base)
    }
    return Self.attachImage(at: path, to: base)
  }

  /// Append the image content block, or a note saying why it was omitted —
  /// never a silent drop. The text block's JSON and structuredContent are
  /// unchanged; the note is appended after it.
  private static func attachImage(at path: String, to base: MCPToolResult) -> MCPToolResult {
    guard MCPImageAttachment.isAttachableImage(path: path) else {
      return Self.withNote(
        "return_image: \(path) is not an image — read output_path instead.", on: base)
    }
    // Goes through `attachment` so the per-result cap is genuinely enforced
    // here rather than merely declared (PR #367 review r1, item 4).
    let attachment = MCPImageAttachment.attachment(paths: [path])
    var text = base.content.first?.text ?? "{}"
    for note in attachment.notes { text += "\n" + note }
    return MCPToolResult(
      text: text, structuredJSON: base.structuredJSON, images: attachment.blocks)
  }

  private static func withNote(_ note: String, on base: MCPToolResult) -> MCPToolResult {
    MCPToolResult(
      text: (base.content.first?.text ?? "{}") + "\n" + note,
      structuredJSON: base.structuredJSON, images: [])
  }

  private static func mapHTTPResponse(status: Int, data: Data) -> MCPToolResult {
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
