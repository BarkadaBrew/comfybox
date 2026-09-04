// ComfyBridge.swift — ComfyUI protocol bridge for ZImageCLI
//
// Makes ZImageCLI's warm server speak ComfyUI's HTTP + WebSocket protocol
// so the Krita AI Diffusion plugin can connect directly.
//
// Phase 1: Discovery endpoints (static responses) + WebSocket skeleton + image cache.
// Phase 2: Workflow parsing + async generation + WebSocket progress events.

import Foundation
import Logging
import Metal
import Network

/// ComfyUI protocol bridge — translates ComfyUI API calls into ZImageCLI operations.
///
/// Phase 1 implements discovery endpoints that satisfy the Krita plugin's connection
/// handshake and node validation. Phase 2 adds workflow parsing and actual generation
/// routed through the warm server pipeline.
final class ComfyBridge {
  private let logger: Logger
  let wsManager: ComfyWebSocketManager
  let imageCache: ComfyImageCache
  let history: ComfyBridgeHistory

  /// The executor handles async generation and WebSocket event dispatch.
  /// Configured after init via `configureExecutor()`.
  private(set) var executor: ComfyBridgeExecutor?

  /// Queue status provider — set by WarmServer to expose coordinator queue state.
  var queueStatusProvider: (() async -> ComfyBridgeQueueStatus)?

  /// Queue clear handler — set by WarmServer to clear pending coordinator jobs.
  var queueClearHandler: (() async -> Void)?

  /// Model switch handler — set by WarmServer to auto-switch models when Krita
  /// selects a different checkpoint. Returns true if the model was switched successfully.
  /// The handler receives the detected model ID from the workflow's CheckpointLoaderSimple node.
  /// The WarmServer implementation runs the switch through the coordinator's FIFO
  /// render queue so the pool load/activate cannot race an in-flight render.
  var modelSwitchHandler: ((_ modelId: String) async throws -> Bool)?

  /// Interrupt handler — set by WarmServer to cancel the in-flight render task.
  /// Returns true if a render was actually cancelled.
  var interruptHandler: (() async -> Bool)?

  /// Debug workflow dumps/logging are gated behind an env flag — they write full
  /// prompts to a world-readable file in /tmp and to stdout.
  static let debugWorkflowDumpEnabled =
    ProcessInfo.processInfo.environment["COMFYBOX_DEBUG_WORKFLOW"] != nil

  /// LoRA Library reference — set by WarmServer for library-aware LoRA discovery.
  /// When set, invalidates the cached object_info so the next request picks up
  /// library-sourced LoRA listings filtered by model compatibility.
  var loraLibrary: LoRALibrary? {
    didSet {
      cacheLock.lock()
      cachedObjectInfo = nil
      cacheLock.unlock()
    }
  }

  // Lazily built and cached on first request. Guarded by cacheLock.
  private let cacheLock = NSLock()
  private var cachedSystemStats: Data?
  private var cachedObjectInfo: Data?

  init(logger: Logger) {
    self.logger = logger
    self.wsManager = ComfyWebSocketManager(logger: logger)
    self.imageCache = ComfyImageCache(logger: logger)
    self.history = ComfyBridgeHistory()
  }

  /// Configure the executor with generation and upscale handlers.
  /// Called by WarmServer after init to wire in the coordinator.
  func configureExecutor(
    generateHandler: @escaping ComfyBridgeGenerateHandler,
    upscaleHandler: ComfyBridgeUpscaleHandler? = nil
  ) {
    self.executor = ComfyBridgeExecutor(
      logger: logger,
      wsManager: wsManager,
      imageCache: imageCache,
      history: history,
      generateHandler: generateHandler,
      upscaleHandler: upscaleHandler
    )
    let upscaleStatus = upscaleHandler != nil ? "upscale enabled" : "upscale not configured"
    logger.info("ComfyBridge: executor configured — Phase 2 generation enabled, \(upscaleStatus)")
  }

  // MARK: - Route Dispatch

  /// Attempt to route a request through the ComfyUI bridge.
  /// Returns nil if this request is not a ComfyUI endpoint (falls through to WarmServer routes).
  /// Async so queue/interrupt handlers can await the coordinator directly instead
  /// of blocking a cooperative-pool thread on a semaphore.
  func route(_ request: HTTPRequest) async -> RoutedResponse? {
    let originalPath = request.path
    let path = Self.strippingAPIPrefix(from: originalPath)

    if request.method == "OPTIONS" {
      return .json(.empty(status: 204))
    }

    switch (request.method, path) {

    case ("GET", "/system_stats"):
      return handleSystemStats()

    case ("GET", "/object_info"):
      return handleObjectInfo()

    case ("GET", "/embeddings"):
      return rawJSON("[]")

    case ("GET", "/settings"):
      return rawJSON(#"{"Comfy.TutorialCompleted":true,"Comfy.Workflow.Persist":false}"#)

    case ("GET", "/extensions"):
      return rawJSON("[]")

    case ("GET", "/experiment/models"):
      return rawJSON("[]")

    case _ where request.method == "GET" && path.hasPrefix("/userdata"):
      return rawJSON("[]")

    case ("GET", "/users"):
      return rawJSON(#"{"storage":"server","migrated":true,"users":{"":"default"}}"#)

    case ("GET", "/queue"):
      return await handleGetQueue()

    case ("POST", "/queue"):
      return await handlePostQueue(request)

    case ("GET", "/prompt"):
      return handleGetPrompt()

    case ("POST", "/prompt"):
      return handlePrompt(request)

    case ("POST", "/interrupt"):
      return await handleInterrupt()

    case ("GET", "/view"):
      return handleView(request)

    case ("GET", "/history"):
      return handleHistory(request)

    case ("POST", "/upload/image"):
      return handleUploadImage(request)

    case ("GET", "/ws"):
      // WebSocket upgrade is handled separately in ConnectionHandler.
      // Return websocketUpgrade to signal the router that this path is claimed.
      return .websocketUpgrade

    default:
      // Image cache endpoints use path-prefix matching.
      if request.method == "GET", path.hasPrefix("/history/") {
        let promptId = String(path.dropFirst("/history/".count))
        return handleHistory(promptId: promptId)
      }

      if path.hasPrefix("/etn/image/") {
        let id = String(path.dropFirst("/etn/image/".count))
        guard !id.isEmpty else {
          return .error(.error(status: 400, message: "Missing image ID"))
        }
        switch request.method {
        case "PUT":
          return handleImagePut(id: id, body: request.body)
        case "GET":
          return handleImageGet(id: id)
        default:
          return .error(.error(status: 405, message: "Method not allowed"))
        }
      }

      // Model info endpoint with pagination.
      if path.hasPrefix("/etn/model_info/") {
        let folder = String(path.dropFirst("/etn/model_info/".count))
        return handleModelInfo(folder: folder, queryString: request.queryString)
      }

      // Experiment models sub-path (e.g. /experiment/models/checkpoints).
      if path.hasPrefix("/experiment/models/") {
        return rawJSON("[]")
      }

      // No-op translation passthrough. The plugin expects the translated
      // string as the JSON response body; without this route the /api
      // catch-all returns "{}" and Krita uses "{}" as the prompt text.
      if request.method == "GET", path.hasPrefix("/etn/translate/") {
        return handleTranslate(path: path)
      }

      // LoRA upload: PUT /api/etn/upload/loras/{id} with raw file bytes.
      if path.hasPrefix("/etn/upload/loras/") {
        guard request.method == "PUT" || request.method == "POST" else {
          return .error(.error(status: 405, message: "Method not allowed"))
        }
        let id = String(path.dropFirst("/etn/upload/loras/".count))
        return handleLoRAUpload(id: id, body: request.body)
      }

      // Languages endpoint.
      if path == "/etn/languages" {
        if let data = try? JSONSerialization.data(withJSONObject: [] as [Any]) {
          return .json(.rawJSON(status: 200, data: data))
        }
        return .json(status: 200, payload: EmptyObject())
      }

      if originalPath == "/api" || originalPath.hasPrefix("/api/") {
        let warning = "ComfyBridge: unknown frontend API route \(request.method) \(originalPath) — returning empty object"
        logger.warning("\(warning)")
        FileHandle.standardError.write(Data("warning: \(warning)\n".utf8))
        return rawJSON("{}")
      }

      return nil
    }
  }

  private static func strippingAPIPrefix(from path: String) -> String {
    if path == "/api" {
      return "/"
    }
    if path.hasPrefix("/api/") {
      return String(path.dropFirst("/api".count))
    }
    return path
  }

  private func rawJSON(_ json: String, status: Int = 200) -> RoutedResponse {
    .json(.rawJSON(status: status, data: Data(json.utf8)))
  }

  // MARK: - GET /system_stats

  private func handleSystemStats() -> RoutedResponse {
    cacheLock.lock()
    let cached = cachedSystemStats
    cacheLock.unlock()

    if let cached {
      return .json(.rawJSON(status: 200, data: cached))
    }

    let deviceName = queryMetalDeviceName()
    let totalMemory = ProcessInfo.processInfo.physicalMemory

    let stats: [String: Any] = [
      "devices": [
        [
          "name": deviceName,
          "type": "mps",
          "vram_total": totalMemory
        ]
      ]
    ]

    if let data = try? JSONSerialization.data(withJSONObject: stats) {
      cacheLock.lock()
      cachedSystemStats = data
      cacheLock.unlock()
      logger.info("ComfyBridge: /system_stats — \(deviceName), \(totalMemory / (1024*1024*1024))GB")
      return .json(.rawJSON(status: 200, data: data))
    }

    return .error(.error(status: 500, message: "Failed to serialize system_stats"))
  }

  private func queryMetalDeviceName() -> String {
    if let device = MTLCreateSystemDefaultDevice() {
      return device.name
    }
    return "Apple Silicon"
  }

  // MARK: - GET /object_info

  private func handleObjectInfo() -> RoutedResponse {
    cacheLock.lock()
    let cached = cachedObjectInfo
    cacheLock.unlock()

    if let cached {
      return .json(.rawJSON(status: 200, data: cached))
    }

    let info = ComfyBridgeObjectInfo.build(loraLibrary: loraLibrary)

    // Use orderedJSONData instead of JSONSerialization to preserve
    // input key order. ComfyUI frontend maps widgets_values positionally
    // based on the key order in /object_info — scrambled keys cause
    // width/height/batch_size to swap, resulting in wrong dimensions
    // and runaway batch renders.
    if let data = orderedJSONData(info) {
      cacheLock.lock()
      cachedObjectInfo = data
      cacheLock.unlock()
      logger.info("ComfyBridge: /object_info — \(info.count) nodes declared")
      return .json(.rawJSON(status: 200, data: data))
    }

    return .error(.error(status: 500, message: "Failed to serialize object_info"))
  }

  // MARK: - GET /queue

  private func handleGetQueue() async -> RoutedResponse {
    // Fetch rich queue status from the coordinator if available.
    if let provider = queueStatusProvider {
      let s = await provider()
      let queue: [String: Any] = [
        "queue_running": s.isRendering ? [["active_job"]] as [Any] : [] as [Any],
        "queue_pending": Array(repeating: ["pending_job"] as [Any], count: s.pendingCount),
        "queue_status": [
          "pending_count": s.pendingCount,
          "max_pending": s.maxPending,
          "is_rendering": s.isRendering,
          "current_job_id": s.currentJobId as Any,
          "progress_percent": s.progressPercent as Any,
          "render_count": s.renderCount,
          "failed_count": s.failedCount,
        ] as [String: Any],
      ]
      if let data = try? JSONSerialization.data(withJSONObject: queue) {
        return .json(.rawJSON(status: 200, data: data))
      }
    }

    // Fallback: basic status from executor.
    let isRunning = executor?.isExecuting ?? false
    let queue: [String: Any] = [
      "queue_running": isRunning ? [["placeholder"]] as [Any] : [] as [Any],
      "queue_pending": [] as [Any]
    ]
    if let data = try? JSONSerialization.data(withJSONObject: queue) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize queue"))
  }

  // MARK: - GET /prompt

  private func handleGetPrompt() -> RoutedResponse {
    let response: [String: Any] = [
      "exec_info": [
        "queue_remaining": 0
      ]
    ]
    if let data = try? JSONSerialization.data(withJSONObject: response) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize prompt status"))
  }

  // MARK: - POST /queue (delete queued jobs)

  private func handlePostQueue(_ request: HTTPRequest) async -> RoutedResponse {
    let body = (try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any]

    // {"delete": [prompt_ids]} — cancel only the listed queued jobs.
    // The Krita plugin uses this to cancel a single queued generation;
    // clearing everything would silently kill its other queued jobs.
    if let deleteList = body?["delete"] as? [Any] {
      let promptIds = deleteList.compactMap { $0 as? String }
      let cancelled = executor?.cancelQueued(promptIds: promptIds) ?? 0
      logger.info("ComfyBridge: POST /queue delete — cancelled \(cancelled)/\(promptIds.count) queued job(s)")
      let response: [String: Any] = [
        "success": true,
        "cancelled_count": cancelled,
      ]
      if let data = try? JSONSerialization.data(withJSONObject: response) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .json(status: 200, payload: EmptyObject())
    }

    // {"clear": true} (or a legacy/empty body) — cancel all pending jobs:
    // the bridge's serialized queue plus the coordinator's pending queue.
    let clearedBridge = executor?.cancelAllQueued() ?? 0
    var clearedCoordinator = 0
    if let provider = queueStatusProvider {
      clearedCoordinator = await provider().pendingCount
      // Signal the coordinator to clear pending jobs.
      if let clearFn = queueClearHandler {
        await clearFn()
      }
    }
    logger.info("ComfyBridge: POST /queue clear — cancelled \(clearedBridge) bridge job(s), \(clearedCoordinator) coordinator job(s)")

    let response: [String: Any] = [
      "success": true,
      "cleared_count": clearedBridge + clearedCoordinator,
    ]
    if let data = try? JSONSerialization.data(withJSONObject: response) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .json(status: 200, payload: EmptyObject())
  }

  // MARK: - POST /prompt

  private func handlePrompt(_ request: HTTPRequest) -> RoutedResponse {
    guard let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
      return .error(.error(status: 400, message: "Invalid JSON body"))
    }

    // ComfyUI frontend doesn't send prompt_id — generate one server-side.
    let promptId = (json["prompt_id"] as? String) ?? UUID().uuidString

    guard json["prompt"] is [String: Any] else {
      return .error(.error(status: 400, message: "Missing or invalid 'prompt' object"))
    }

    let clientId = json["client_id"] as? String ?? "unknown"

    // Phase 2: parse the workflow and dispatch async generation.
    guard let executor = executor else {
      // No executor configured — acknowledge but don't generate (Phase 1 behavior).
      logger.info("ComfyBridge: /prompt received (no executor) — prompt_id=\(promptId), client_id=\(clientId)")
      let response: [String: Any] = ["prompt_id": promptId]
      if let data = try? JSONSerialization.data(withJSONObject: response) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize prompt response"))
    }

    // Debug: dump raw workflow JSON. Gated behind COMFYBOX_DEBUG_WORKFLOW —
    // the dump contains full prompts and lands in world-readable /tmp.
    if Self.debugWorkflowDumpEnabled,
       let dumpData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]),
       let dumpStr = String(data: dumpData, encoding: .utf8) {
      try? dumpStr.write(toFile: "/tmp/zimage-debug-workflow.json", atomically: true, encoding: .utf8)
      logger.info("ComfyBridge: dumped workflow to /tmp/zimage-debug-workflow.json (\(dumpData.count) bytes)")
      // Also log the node class types
      if let prompt = json["prompt"] as? [String: [String: Any]] {
        let nodeTypes = prompt.values.compactMap { ($0["class_type"] as? String) }.sorted()
        logger.info("ComfyBridge: workflow node types: \(nodeTypes.joined(separator: ", "))")
      }
    }

    // Parse the workflow — detects generate vs upscale from node types.
    let parsedWorkflow: ComfyBridgeParsedWorkflow
    do {
      parsedWorkflow = try ComfyBridgeWorkflowParser.parseWorkflow(json)
    } catch {
      logger.error("ComfyBridge: workflow parse failed — \(error)")
      return .error(.error(status: 400, message: "Workflow parse error: \(error)"))
    }

    // Route to the correct executor based on workflow type.
    switch parsedWorkflow {
    case .generate(let generateRequest):
      // #22: same resolution/memory preflight `decodedGeneratePayload` runs
      // for `/v1/generate` — BEFORE enqueue, let alone any model load. The
      // bridge's `ComfyBridgeGenerateRequest` has no `dype` field of its own
      // (Krita never sends one), so `dype` is derived with the identical
      // auto-enable threshold `resolvedDyPEConfig` uses.
      do {
        let family = ImageMemoryPreflight.resolvedFamily(model: generateRequest.detectedModel)
        let dype = ImageMemoryPreflight.autoDyPEEnabled(width: generateRequest.width, height: generateRequest.height)
        try ImageMemoryPreflight.validate(
          width: generateRequest.width, height: generateRequest.height, family: family, dype: dype,
          caps: ServerConfigStore.shared.imageMemoryCaps(),
          availableBytes: MemoryProbe.systemAvailableMemoryBytes())
      } catch {
        logger.warning("ComfyBridge: /prompt [generate] refused by image-memory preflight — \(error.localizedDescription)")
        return .error(WarmServer.errorResponse(for: error))
      }

      logger.info("ComfyBridge: /prompt [generate] — \(generateRequest.width)x\(generateRequest.height), \(generateRequest.steps) steps, cfg=\(generateRequest.guidance), seed=\(generateRequest.seed.map(String.init) ?? "random")")

      // Acknowledge immediately, but run the job through the executor's
      // serialized queue: execution_start must only be emitted when the
      // generation actually begins. The Krita plugin frees its one-job send
      // slot on execution_start — emitting it while a previous job is still
      // rendering makes the plugin mark that job interrupted and drop its
      // results. The model switch (if detected) runs inside the serialized
      // job so it cannot land in the coordinator queue ahead of a previously
      // accepted generation.
      let detectedModel = generateRequest.detectedModel
      let switchHandler = modelSwitchHandler
      executor.enqueue(promptId: generateRequest.promptId) { [logger, executor] in
        // Auto-switch model if the workflow specifies a different checkpoint.
        if let modelId = detectedModel, let handler = switchHandler {
          do {
            let switched = try await handler(modelId)
            if switched {
              logger.info("ComfyBridge: auto-switched to model '\(modelId)' for this render")
            }
          } catch {
            // #339 review r2, item 1: `ModelSwitchFailurePolicy` (pure,
            // tested directly) decides whether this must FAIL the prompt —
            // a queue-recovery refusal — or, as before, just log and
            // continue rendering on the current model for any other
            // transient switch failure. Continuing on a recovery refusal
            // used to silently render under a stale checkpoint with no
            // error the caller could see.
            switch ModelSwitchFailurePolicy.decide(error) {
            case .failPrompt(let message):
              logger.warning("ComfyBridge: model switch to '\(modelId)' refused — failing prompt \(generateRequest.promptId) rather than risk a wrong-checkpoint render (#339 r2)")
              executor.failPrompt(promptId: generateRequest.promptId, clientId: generateRequest.clientId, message: message)
              return
            case .continueOnCurrentModel:
              logger.error("ComfyBridge: model switch to '\(modelId)' failed — \(error.localizedDescription)")
              // Continue with current model rather than failing the entire render.
            }
          }
        }
        await executor.execute(generateRequest)
      }

    case .upscale(let upscaleRequest):
      logger.info("ComfyBridge: /prompt [upscale] — model=\(upscaleRequest.upscaleModelName), input=\(upscaleRequest.inputImageNodeId)")
      executor.enqueue(promptId: upscaleRequest.promptId) {
        await executor.executeUpscale(upscaleRequest)
      }
    }

    let response: [String: Any] = ["prompt_id": promptId]
    if let data = try? JSONSerialization.data(withJSONObject: response) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize prompt response"))
  }

  /// Submit an already-normalized ComfyUI API graph for execution outside the
  /// Krita wire protocol (workflow run API, #238). Parses and enqueues exactly
  /// like POST /prompt; results land in the image cache + history under
  /// `promptId`, and optional overrides are applied to the parsed request
  /// before enqueue. Throws on parse failure or missing executor.
  func submitWorkflowGraph(
    _ graph: [String: Any],
    promptId: String,
    promptOverride: String? = nil,
    negativePromptOverride: String? = nil,
    seedOverride: UInt64? = nil
  ) throws {
    guard let executor = executor else {
      throw WorkflowError.storeFailed("ComfyBridge executor not configured")
    }
    let body: [String: Any] = [
      "prompt": graph,
      "prompt_id": promptId,
      "client_id": "workflow-api",
    ]
    let parsed = try ComfyBridgeWorkflowParser.parseWorkflow(body)
    switch parsed {
    case .generate(var request):
      if let promptOverride { request.prompt = promptOverride }
      if let negativePromptOverride { request.negativePrompt = negativePromptOverride }
      if let seedOverride { request.seed = seedOverride }
      let detectedModel = request.detectedModel
      let switchHandler = modelSwitchHandler
      let generateRequest = request
      // #22: same preflight as POST /prompt — `submitWorkflowGraph` is
      // `throws`, so a refusal here propagates to this call's caller before
      // any enqueue, let alone any model load.
      try ImageMemoryPreflight.validate(
        width: generateRequest.width, height: generateRequest.height,
        family: ImageMemoryPreflight.resolvedFamily(model: detectedModel),
        dype: ImageMemoryPreflight.autoDyPEEnabled(width: generateRequest.width, height: generateRequest.height),
        caps: ServerConfigStore.shared.imageMemoryCaps(),
        availableBytes: MemoryProbe.systemAvailableMemoryBytes())
      logger.info("ComfyBridge: workflow-api [generate] — \(generateRequest.width)x\(generateRequest.height), \(generateRequest.steps) steps, seed=\(generateRequest.seed.map(String.init) ?? "random")")
      executor.enqueue(promptId: generateRequest.promptId) { [logger, executor] in
        if let modelId = detectedModel, let handler = switchHandler {
          do {
            let switched = try await handler(modelId)
            if switched {
              logger.info("ComfyBridge: auto-switched to model '\(modelId)' for workflow run")
            }
          } catch {
            // #339 review r2, item 1 — same fix as `handlePrompt` above.
            switch ModelSwitchFailurePolicy.decide(error) {
            case .failPrompt(let message):
              logger.warning("ComfyBridge: model switch to '\(modelId)' refused — failing workflow run \(generateRequest.promptId) rather than risk a wrong-checkpoint render (#339 r2)")
              executor.failPrompt(promptId: generateRequest.promptId, clientId: generateRequest.clientId, message: message)
              return
            case .continueOnCurrentModel:
              logger.error("ComfyBridge: model switch to '\(modelId)' failed — \(error.localizedDescription)")
            }
          }
        }
        await executor.execute(generateRequest)
      }
    case .upscale(let upscaleRequest):
      logger.info("ComfyBridge: workflow-api [upscale] — model=\(upscaleRequest.upscaleModelName)")
      executor.enqueue(promptId: upscaleRequest.promptId) {
        await executor.executeUpscale(upscaleRequest)
      }
    }
  }

  // MARK: - POST /interrupt

  private func handleInterrupt() async -> RoutedResponse {
    // Broadcast execution_interrupted for the active prompt(s), then cancel the
    // in-flight render task so the pipeline's denoise loop actually stops.
    let hadActive = executor?.interrupt() ?? false
    let cancelled = await interruptHandler?() ?? false
    if hadActive || cancelled {
      logger.info("ComfyBridge: /interrupt — active generation interrupted (render cancelled: \(cancelled))")
    } else {
      logger.info("ComfyBridge: /interrupt — no active generation")
    }
    return .json(status: 200, payload: EmptyObject())
  }

  // MARK: - Image Cache

  private func handleImagePut(id: String, body: Data) -> RoutedResponse {
    guard !body.isEmpty else {
      return .error(.error(status: 400, message: "Empty image body"))
    }

    guard imageCache.store(id: id, data: body) else {
      return .error(.error(status: 500, message: "Failed to persist image \(id)"))
    }
    logger.info("ComfyBridge: stored image \(id) (\(body.count) bytes)")
    return .json(status: 200, payload: EmptyObject())
  }

  private func handleImageGet(id: String) -> RoutedResponse {
    guard let data = imageCache.retrieve(id: id) else {
      return .error(.error(status: 404, message: "Image not found: \(id)"))
    }

    return .json(.binary(status: 200, contentType: "image/png", data: data))
  }

  private func handleView(_ request: HTTPRequest) -> RoutedResponse {
    guard var filename = decodedQueryParameter("filename", in: request), !filename.isEmpty else {
      return .error(.error(status: 400, message: "Missing filename"))
    }

    filename = URL(fileURLWithPath: filename).lastPathComponent
    if filename.lowercased().hasSuffix(".png") {
      filename.removeLast(4)
    }

    guard let data = imageCache.retrieve(id: filename) else {
      return .error(.error(status: 404, message: "Image not found: \(filename)"))
    }

    return .json(.binary(status: 200, contentType: "image/png", data: data))
  }

  // MARK: - History

  private func handleHistory(_ request: HTTPRequest) -> RoutedResponse {
    let maxItems = intQueryParameter("max_items", in: request) ?? intQueryParameter("maxItems", in: request) ?? 200
    let offset = intQueryParameter("offset", in: request) ?? 0
    let payload = history.toJSON(maxItems: maxItems, offset: offset)
    if let data = try? JSONSerialization.data(withJSONObject: payload) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize history"))
  }

  private func handleHistory(promptId: String) -> RoutedResponse {
    let decodedPromptId = promptId.removingPercentEncoding ?? promptId
    let payload = history.entry(for: decodedPromptId) ?? [:]
    if let data = try? JSONSerialization.data(withJSONObject: payload) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize history item"))
  }

  // MARK: - POST /upload/image

  private func handleUploadImage(_ request: HTTPRequest) -> RoutedResponse {
    do {
      let fields = try ComfyBridgeMultipart.parse(
        body: request.body,
        contentType: request.headers["content-type"]
      )
      guard let imageData = fields["image"] else {
        return .error(.error(status: 400, message: "Missing image upload field"))
      }

      let imageId = UUID().uuidString
      let filename = "\(imageId).png"
      guard imageCache.store(id: imageId, data: imageData) else {
        return .error(.error(status: 500, message: "Failed to cache uploaded image"))
      }

      let response: [String: Any] = [
        "name": filename,
        "subfolder": "",
        "type": "input"
      ]
      if let data = try? JSONSerialization.data(withJSONObject: response) {
        logger.info("ComfyBridge: uploaded image \(filename) as cache id \(imageId) (\(imageData.count) bytes)")
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize upload response"))
    } catch {
      return .error(.error(status: 400, message: "Invalid multipart upload: \(error)"))
    }
  }

  // MARK: - Model Info

  private func handleModelInfo(folder: String, queryString: String?) -> RoutedResponse {
    var offset = 0
    var limit = 8

    if let qs = queryString {
      for pair in qs.split(separator: "&") {
        let parts = pair.split(separator: "=", maxSplits: 1)
        guard parts.count == 2 else { continue }
        let key = String(parts[0])
        let val = String(parts[1])
        switch key {
        case "offset": offset = max(0, Int(val) ?? 0)
        case "limit": limit = max(1, min(100, Int(val) ?? 8))
        default: break
        }
      }
    }

    let models = ComfyBridgeModelInfo.models(for: folder)
    let total = models.count
    let keys = Array(models.keys).sorted()
    let pageKeys = Array(keys.dropFirst(offset).prefix(limit))

    var page: [String: Any] = [:]
    for key in pageKeys {
      page[key] = models[key]
    }
    page["_meta"] = ["total": total]

    if let data = try? JSONSerialization.data(withJSONObject: page) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize model_info"))
  }

  // MARK: - Translation (no-op passthrough)

  /// GET /api/etn/translate/{lang}/{text} — ComfyBox has no translation
  /// backend (GET /etn/languages returns []), so echo the text back as a
  /// JSON string. Prompts pass through unchanged when translation is enabled.
  private func handleTranslate(path: String) -> RoutedResponse {
    let remainder = String(path.dropFirst("/etn/translate/".count))
    guard let separator = remainder.firstIndex(of: "/") else {
      return .error(.error(status: 400, message: "Expected /etn/translate/{lang}/{text}"))
    }
    let rawText = String(remainder[remainder.index(after: separator)...])
    let text = rawText.removingPercentEncoding ?? rawText
    if let data = try? JSONSerialization.data(withJSONObject: text, options: [.fragmentsAllowed]) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize translation"))
  }

  // MARK: - LoRA Upload

  /// LoRA upload destination — matches the directory the workflow parser and
  /// WarmServer resolve bare LoraLoader names against.
  private static let loraUploadDirectoryPath = ("~/bin/zimage/loras" as NSString).expandingTildeInPath

  /// PUT /api/etn/upload/loras/{id} — persist the uploaded LoRA file so it
  /// appears in LoraLoader options and resolves at generation time. Without
  /// this, the catch-all returns 200 and the upload silently vanishes.
  private func handleLoRAUpload(id: String, body: Data) -> RoutedResponse {
    guard !body.isEmpty else {
      return .error(.error(status: 400, message: "Empty LoRA upload body"))
    }

    let decoded = id.removingPercentEncoding ?? id
    // Sanitize: allow subdirectories but reject empty ids, absolute paths,
    // and path traversal.
    let components = decoded.split(separator: "/").map(String.init)
    guard !decoded.hasPrefix("/"), !components.isEmpty, !components.contains("..") else {
      return .error(.error(status: 400, message: "Invalid LoRA id"))
    }

    var fileURL = URL(fileURLWithPath: Self.loraUploadDirectoryPath)
    for component in components {
      fileURL.appendPathComponent(component)
    }

    do {
      try FileManager.default.createDirectory(
        at: fileURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
      )
      try body.write(to: fileURL, options: .atomic)
    } catch {
      logger.error("ComfyBridge: failed to store uploaded LoRA \(decoded) — \(error)")
      return .error(.error(status: 500, message: "Failed to store LoRA: \(error)"))
    }

    // Invalidate the cached object_info so the next /object_info request
    // lists the new LoRA in LoraLoader options.
    cacheLock.lock()
    cachedObjectInfo = nil
    cacheLock.unlock()

    logger.info("ComfyBridge: stored uploaded LoRA \(decoded) (\(body.count) bytes)")
    return .json(status: 200, payload: EmptyObject())
  }

  private func decodedQueryParameter(_ name: String, in request: HTTPRequest) -> String? {
    guard let queryString = request.queryString else { return nil }
    for pair in queryString.split(separator: "&", omittingEmptySubsequences: false) {
      let parts = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
      guard let rawKey = parts.first else { continue }
      let key = String(rawKey).replacingOccurrences(of: "+", with: " ").removingPercentEncoding ?? String(rawKey)
      guard key == name else { continue }
      if parts.count == 1 { return "" }
      let rawValue = String(parts[1]).replacingOccurrences(of: "+", with: " ")
      return rawValue.removingPercentEncoding ?? rawValue
    }
    return nil
  }

  private func intQueryParameter(_ name: String, in request: HTTPRequest) -> Int? {
    guard let value = decodedQueryParameter(name, in: request) else { return nil }
    return Int(value)
  }

  // MARK: - WebSocket Upgrade

  /// Build the HTTP 101 WebSocket upgrade response for a validated request.
  func handleWebSocketUpgrade(request: HTTPRequest, connection: NWConnection, queue: DispatchQueue) -> Data? {
    guard request.headers["upgrade"]?.lowercased() == "websocket" else {
      return nil
    }

    guard let connectionHeader = request.headers["connection"],
          connectionHeader.lowercased().contains("upgrade") else {
      return nil
    }

    guard request.headers["sec-websocket-version"] == "13" else {
      return nil
    }

    guard let wsKey = request.headers["sec-websocket-key"],
          let decoded = Data(base64Encoded: wsKey),
          decoded.count == 16 else {
      return nil
    }

    let acceptKey = ComfyWebSocketManager.computeAcceptKey(from: wsKey)
    let response = [
      "HTTP/1.1 101 Switching Protocols",
      "Upgrade: websocket",
      "Connection: Upgrade",
      "Sec-WebSocket-Accept: \(acceptKey)",
      "Access-Control-Allow-Origin: *",
      "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS",
      "Access-Control-Allow-Headers: Content-Type",
      "",
      ""
    ].joined(separator: "\r\n")

    logger.info("ComfyBridge: WebSocket upgrade for clientId=\(request.queryParameters["clientId"] ?? "unknown")")
    return Data(response.utf8)
  }
}

// MARK: - Queue Status

/// Queue status returned by the coordinator, bridged to ComfyUI /queue format.
struct ComfyBridgeQueueStatus: Sendable {
  let pendingCount: Int
  let maxPending: Int
  let isRendering: Bool
  let currentJobId: String?
  let progressPercent: Int?
  let renderCount: Int
  let failedCount: Int
}

// MARK: - Helpers

/// Empty JSON object for responses that need `{}`.
private struct EmptyObject: Encodable {}
