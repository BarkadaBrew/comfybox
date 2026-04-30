// ComfyBridge.swift — ComfyUI protocol bridge for ZImageCLI
//
// Makes ZImageCLI's warm server speak ComfyUI's HTTP + WebSocket protocol
// so the Krita AI Diffusion plugin can connect directly.
//
// Phase 1: Discovery endpoints (static responses) + WebSocket skeleton + image cache.

import Foundation
import Logging
import Metal
import Network

/// ComfyUI protocol bridge — translates ComfyUI API calls into ZImageCLI operations.
///
/// Phase 1 implements discovery endpoints that satisfy the Krita plugin's connection
/// handshake and node validation. The bridge is designed as a clean layer that can be
/// enabled/disabled without affecting the existing `/v1/generate` API.
final class ComfyBridge {
  private let logger: Logger
  let wsManager: ComfyWebSocketManager
  private let imageCache: ComfyImageCache

  // Lazily built and cached on first request. Guarded by cacheLock.
  private let cacheLock = NSLock()
  private var cachedSystemStats: Data?
  private var cachedObjectInfo: Data?

  init(logger: Logger) {
    self.logger = logger
    self.wsManager = ComfyWebSocketManager(logger: logger)
    self.imageCache = ComfyImageCache(logger: logger)
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

      // Model info endpoint — returns metadata for our available models.
      if request.path.hasPrefix("/api/etn/model_info/") {
        let folder = String(request.path.dropFirst("/api/etn/model_info/".count))
        return handleModelInfo(folder: folder, queryString: request.queryString)
      }

      // Translation languages endpoint.
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

    // Build the response manually to avoid snake_case encoding of the fields
    // that ComfyUI expects in their exact casing.
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

  /// Query the Metal device name. Falls back to a sensible default.
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
    // Phase 1: always report empty queues.
    let queue: [String: Any] = [
      "queue_running": [] as [Any],
      "queue_pending": [] as [Any]
    ]
    if let data = try? JSONSerialization.data(withJSONObject: queue) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize queue"))
  }

  // MARK: - POST /queue (delete queued jobs)

  private func handlePostQueue() -> RoutedResponse {
    // Phase 1: acknowledge the delete request, nothing to cancel.
    return .json(status: 200, payload: EmptyObject())
  }

  // MARK: - POST /prompt

  private func handlePrompt(_ request: HTTPRequest) -> RoutedResponse {
    // Phase 1: validate the shape, echo back prompt_id, don't execute.
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
    logger.info("ComfyBridge: /prompt received — prompt_id=\(promptId), client_id=\(clientId)")

    // Phase 2 will extract workflow parameters and route to the pipeline.
    // For now, just acknowledge.
    let response: [String: Any] = [
      "prompt_id": promptId
    ]

    if let data = try? JSONSerialization.data(withJSONObject: response) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize prompt response"))
  }

  // MARK: - POST /interrupt

  private func handleInterrupt() -> RoutedResponse {
    // Phase 1: acknowledge, nothing to cancel yet.
    logger.info("ComfyBridge: /interrupt received (no active job)")
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


  // MARK: - GET /api/etn/model_info/{folder}

  private func handleModelInfo(folder: String, queryString: String?) -> RoutedResponse {
    // Parse offset/limit from query string for pagination.
    var offset = 0
    var limit = 8
    if let qs = queryString {
      for param in qs.split(separator: "&") {
        let parts = param.split(separator: "=", maxSplits: 1)
        if parts.count == 2 {
          if parts[0] == "offset", let v = Int(parts[1]) { offset = v }
          if parts[0] == "limit", let v = Int(parts[1]) { limit = v }
        }
      }
    }

    let models = ComfyBridgeModelInfo.models(for: folder)
    let total = models.count

    // Apply pagination.
    let keys = Array(models.keys).sorted()
    let pageKeys = Array(keys.dropFirst(offset).prefix(limit))
    var page: [String: Any] = [:]
    for key in pageKeys {
      page[key] = models[key]
    }
    page["_meta"] = ["total": total]

    if let data = try? JSONSerialization.data(withJSONObject: page) {
      logger.info("ComfyBridge: /api/etn/model_info/\(folder) — \(pageKeys.count)/\(total) models (offset=\(offset))")
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize model_info"))
  }

  // MARK: - WebSocket Upgrade

  /// Build the HTTP 101 WebSocket upgrade response for a validated request.
  /// Returns the raw response bytes to send, or nil if the request is not a valid upgrade.
  ///
  /// Validates RFC 6455 Section 4.2.1 requirements:
  /// - Upgrade: websocket header present
  /// - Connection header contains "Upgrade"
  /// - Sec-WebSocket-Key is a valid 16-byte base64 value
  /// - Sec-WebSocket-Version is 13
  func handleWebSocketUpgrade(request: HTTPRequest, connection: NWConnection, queue: DispatchQueue) -> Data? {
    // Validate Upgrade header.
    guard request.headers["upgrade"]?.lowercased() == "websocket" else {
      return nil
    }

    // Validate Connection header contains "Upgrade" (case-insensitive, may be comma-separated).
    guard let connectionHeader = request.headers["connection"],
          connectionHeader.lowercased().contains("upgrade") else {
      return nil
    }

    // Validate Sec-WebSocket-Version is 13.
    guard request.headers["sec-websocket-version"] == "13" else {
      return nil
    }

    // Validate Sec-WebSocket-Key is present and is a valid 16-byte base64 value.
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
