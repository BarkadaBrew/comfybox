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

  /// The executor handles async generation and WebSocket event dispatch.
  /// Configured after init via `configureExecutor()`.
  private(set) var executor: ComfyBridgeExecutor?

  // Lazily built and cached on first request. Guarded by cacheLock.
  private let cacheLock = NSLock()
  private var cachedSystemStats: Data?
  private var cachedObjectInfo: Data?

  init(logger: Logger) {
    self.logger = logger
    self.wsManager = ComfyWebSocketManager(logger: logger)
    self.imageCache = ComfyImageCache(logger: logger)
  }

  /// Configure the executor with a generation handler.
  /// Called by WarmServer after init to wire in the coordinator.
  func configureExecutor(generateHandler: @escaping ComfyBridgeGenerateHandler) {
    self.executor = ComfyBridgeExecutor(
      logger: logger,
      wsManager: wsManager,
      imageCache: imageCache,
      generateHandler: generateHandler
    )
    logger.info("ComfyBridge: executor configured — Phase 2 generation enabled")
  }

  // MARK: - Route Dispatch

  /// Attempt to route a request through the ComfyUI bridge.
  /// Returns nil if this request is not a ComfyUI endpoint (falls through to WarmServer routes).
  func route(_ request: HTTPRequest) -> RoutedResponse? {
    switch (request.method, request.path) {

    case ("GET", "/system_stats"):
      return handleSystemStats()

    case ("GET", "/object_info"):
      return handleObjectInfo()

    case ("GET", "/queue"):
      return handleGetQueue()

    case ("POST", "/queue"):
      return handlePostQueue()

    case ("POST", "/prompt"):
      return handlePrompt(request)

    case ("POST", "/interrupt"):
      return handleInterrupt()

    case ("GET", "/ws"):
      // WebSocket upgrade is handled separately in ConnectionHandler.
      // Return websocketUpgrade to signal the router that this path is claimed.
      return .websocketUpgrade

    default:
      // Image cache endpoints use path-prefix matching.
      if request.path.hasPrefix("/api/etn/image/") {
        let id = String(request.path.dropFirst("/api/etn/image/".count))
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
      if request.path.hasPrefix("/api/etn/model_info/") {
        let folder = String(request.path.dropFirst("/api/etn/model_info/".count))
        return handleModelInfo(folder: folder, queryString: request.queryString)
      }

      // Languages endpoint.
      if request.path == "/api/etn/languages" {
        if let data = try? JSONSerialization.data(withJSONObject: [] as [Any]) {
          return .json(.rawJSON(status: 200, data: data))
        }
        return .json(status: 200, payload: EmptyObject())
      }

      return nil
    }
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

    let info = ComfyBridgeObjectInfo.build()

    if let data = try? JSONSerialization.data(withJSONObject: info) {
      cacheLock.lock()
      cachedObjectInfo = data
      cacheLock.unlock()
      logger.info("ComfyBridge: /object_info — \(info.count) nodes declared")
      return .json(.rawJSON(status: 200, data: data))
    }

    return .error(.error(status: 500, message: "Failed to serialize object_info"))
  }

  // MARK: - GET /queue

  private func handleGetQueue() -> RoutedResponse {
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

  // MARK: - POST /queue (delete queued jobs)

  private func handlePostQueue() -> RoutedResponse {
    return .json(status: 200, payload: EmptyObject())
  }

  // MARK: - POST /prompt

  private func handlePrompt(_ request: HTTPRequest) -> RoutedResponse {
    guard let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
      return .error(.error(status: 400, message: "Invalid JSON body"))
    }

    guard let promptId = json["prompt_id"] as? String else {
      return .error(.error(status: 400, message: "Missing prompt_id"))
    }

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

    // Debug: dump raw workflow JSON
    if let dumpData = try? JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys]),
       let dumpStr = String(data: dumpData, encoding: .utf8) {
      try? dumpStr.write(toFile: "/tmp/zimage-debug-workflow.json", atomically: true, encoding: .utf8)
      logger.info("ComfyBridge: dumped workflow to /tmp/zimage-debug-workflow.json (\(dumpData.count) bytes)")
      // Also log the node class types
      if let prompt = json["prompt"] as? [String: [String: Any]] {
        let nodeTypes = prompt.values.compactMap { ($0["class_type"] as? String) }.sorted()
        logger.info("ComfyBridge: workflow node types: \(nodeTypes.joined(separator: ", "))")
      }
    }

    // Parse the workflow into generation parameters.
    let generateRequest: ComfyBridgeGenerateRequest
    do {
      generateRequest = try ComfyBridgeWorkflowParser.parse(json)
    } catch {
      logger.error("ComfyBridge: workflow parse failed — \(error)")
      return .error(.error(status: 400, message: "Workflow parse error: \(error)"))
    }

    logger.info("ComfyBridge: /prompt — \(generateRequest.width)x\(generateRequest.height), \(generateRequest.steps) steps, cfg=\(generateRequest.guidance), seed=\(generateRequest.seed.map(String.init) ?? "random")")

    // Acknowledge immediately — generation runs async via WebSocket events.
    Task {
      await executor.execute(generateRequest)
    }

    let response: [String: Any] = ["prompt_id": promptId]
    if let data = try? JSONSerialization.data(withJSONObject: response) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize prompt response"))
  }

  // MARK: - POST /interrupt

  private func handleInterrupt() -> RoutedResponse {
    if let executor, executor.interrupt() {
      logger.info("ComfyBridge: /interrupt — active generation interrupted")
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
      "",
      ""
    ].joined(separator: "\r\n")

    logger.info("ComfyBridge: WebSocket upgrade for clientId=\(request.queryParameters["clientId"] ?? "unknown")")
    return Data(response.utf8)
  }
}

// MARK: - Helpers

/// Empty JSON object for responses that need `{}`.
private struct EmptyObject: Encodable {}
