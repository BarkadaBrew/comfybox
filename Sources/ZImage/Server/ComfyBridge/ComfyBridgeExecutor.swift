// ComfyBridgeExecutor.swift — Async generation lifecycle for ComfyUI bridge
//
// Manages the full lifecycle of a generation request:
// parse workflow → enqueue via warm server → send WebSocket progress events → store output.
//
// Phase 2: txt2img execution with WebSocket event dispatch.

import Foundation
import Logging

/// Callback type for generating images via the warm server pipeline.
/// Accepts a ComfyBridgeGenerateRequest and returns the output path + timing.
typealias ComfyBridgeGenerateHandler = @Sendable (ComfyBridgeGenerateRequest) async throws -> ComfyBridgeGenerateResult

/// Manages async generation execution and WebSocket event dispatch.
final class ComfyBridgeExecutor {
  private let logger: Logger
  private let wsManager: ComfyWebSocketManager
  private let imageCache: ComfyImageCache
  let generateHandler: ComfyBridgeGenerateHandler?

  /// Active prompt being executed, if any.
  private let lock = NSLock()
  private var activePromptId: String?

  init(
    logger: Logger,
    wsManager: ComfyWebSocketManager,
    imageCache: ComfyImageCache,
    generateHandler: ComfyBridgeGenerateHandler?
  ) {
    self.logger = logger
    self.wsManager = wsManager
    self.imageCache = imageCache
    self.generateHandler = generateHandler
  }

  /// Whether a generation is currently in progress.
  var isExecuting: Bool {
    lock.lock()
    defer { lock.unlock() }
    return activePromptId != nil
  }

  /// Execute a parsed generation request asynchronously.
  /// Sends the full sequence of ComfyUI WebSocket events during execution.
  func execute(_ request: ComfyBridgeGenerateRequest) async {
    guard let handler = generateHandler else {
      logger.error("ComfyBridge: no generate handler configured — generation disabled")
      sendError(promptId: request.promptId, clientId: request.clientId,
                message: "Generation not configured on this server")
      return
    }

    lock.lock()
    activePromptId = request.promptId
    lock.unlock()

    defer {
      lock.lock()
      activePromptId = nil
      lock.unlock()
    }

    let promptPreview = request.prompt.count > 80
      ? String(request.prompt.prefix(77)) + "..."
      : request.prompt
    logger.info("ComfyBridge: executing prompt_id=\(request.promptId) — \(request.width)x\(request.height), \(request.steps) steps, cfg=\(request.guidance)")
    logger.info("ComfyBridge: prompt — \"\(promptPreview)\"")

    // --- Phase 1: execution_start ---
    sendEvent(to: request.clientId, type: "execution_start", data: [
      "prompt_id": request.promptId
    ])

    // --- Phase 2: simulate node progression for loader nodes ---
    // Send executing events for the early nodes (model loading, text encoding).
    // These complete instantly since the model is already warm.
    let loaderNodes = ["1", "2", "3", "4", "5", "6"]
    for nodeId in loaderNodes {
      sendEvent(to: request.clientId, type: "execution_cached", data: [
        "prompt_id": request.promptId,
        "nodes": [nodeId]
      ])
    }

    // The sampler node is where actual work happens.
    sendEvent(to: request.clientId, type: "executing", data: [
      "prompt_id": request.promptId,
      "node": "11"  // SamplerCustomAdvanced
    ])

    // --- Phase 3: run actual generation ---
    do {
      let result = try await handler(request)

      // Read the output image.
      let outputURL = URL(fileURLWithPath: result.outputPath)
      guard let imageData = try? Data(contentsOf: outputURL) else {
        logger.error("ComfyBridge: failed to read output at \(result.outputPath)")
        sendError(promptId: request.promptId, clientId: request.clientId,
                  message: "Failed to read generated image")
        return
      }

      // Store in the image cache.
      let imageId = UUID().uuidString
      guard imageCache.store(id: imageId, data: imageData) else {
        logger.error("ComfyBridge: failed to cache output image \(imageId)")
        sendError(promptId: request.promptId, clientId: request.clientId,
                  message: "Failed to cache generated image")
        return
      }

      // --- Phase 4: send output events ---

      // Mark the output node as executing.
      sendEvent(to: request.clientId, type: "executing", data: [
        "prompt_id": request.promptId,
        "node": request.outputNodeId
      ])

      // Send the executed event with the image reference.
      sendExecutedEvent(
        to: request.clientId,
        promptId: request.promptId,
        nodeId: request.outputNodeId,
        imageId: imageId
      )

      // Workflow complete — node=null signals done.
      sendExecutingDone(to: request.clientId, promptId: request.promptId)

      logger.info("ComfyBridge: generation complete — prompt_id=\(request.promptId), \(result.durationMs)ms, image=\(imageId) (\(imageData.count) bytes)")

      // Clean up the temp file — the image is now in the cache.
      try? FileManager.default.removeItem(at: outputURL)

    } catch {
      logger.error("ComfyBridge: generation failed — prompt_id=\(request.promptId): \(error)")
      sendError(promptId: request.promptId, clientId: request.clientId,
                message: error.localizedDescription)
    }
  }

  /// Interrupt the active generation.
  /// Returns true if there was an active prompt to interrupt.
  func interrupt() -> Bool {
    lock.lock()
    let active = activePromptId
    lock.unlock()

    guard let promptId = active else { return false }

    wsManager.broadcast(text: jsonString([
      "type": "execution_interrupted",
      "data": ["prompt_id": promptId]
    ]))

    logger.info("ComfyBridge: interrupted prompt_id=\(promptId)")
    return true
  }

  // MARK: - WebSocket Event Helpers

  private func sendEvent(to clientId: String, type: String, data: [String: Any]) {
    let event: [String: Any] = ["type": type, "data": data]
    let text = jsonString(event)
    wsManager.send(to: clientId, text: text)
  }

  private func sendExecutedEvent(to clientId: String, promptId: String, nodeId: String, imageId: String) {
    let event: [String: Any] = [
      "type": "executed",
      "data": [
        "prompt_id": promptId,
        "node": nodeId,
        "output": [
          "images": [
            ["source": "http", "id": imageId]
          ]
        ]
      ] as [String: Any]
    ]
    wsManager.send(to: clientId, text: jsonString(event))
  }

  private func sendExecutingDone(to clientId: String, promptId: String) {
    // node: null signals workflow completion.
    // JSONSerialization renders NSNull() as JSON null.
    let event: [String: Any] = [
      "type": "executing",
      "data": [
        "prompt_id": promptId,
        "node": NSNull()
      ] as [String: Any]
    ]
    wsManager.send(to: clientId, text: jsonString(event))
  }

  private func sendError(promptId: String, clientId: String, message: String) {
    let event: [String: Any] = [
      "type": "execution_error",
      "data": [
        "prompt_id": promptId,
        "exception_message": message,
        "traceback": [] as [String]
      ] as [String: Any]
    ]
    wsManager.send(to: clientId, text: jsonString(event))

    // Also send executing-done so the plugin cleans up the prompt state.
    sendExecutingDone(to: clientId, promptId: promptId)
  }

  private func jsonString(_ dict: [String: Any]) -> String {
    guard let data = try? JSONSerialization.data(withJSONObject: dict),
          let str = String(data: data, encoding: .utf8) else {
      return "{}"
    }
    return str
  }
}
