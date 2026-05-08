// ComfyBridgeExecutor.swift — Async generation lifecycle for ComfyUI bridge
//
// Manages the full lifecycle of a generation or upscale request:
// parse workflow → enqueue via warm server → send WebSocket progress events → store output.
//
// Phase 2: txt2img execution with WebSocket event dispatch.
// Phase 5: SeedVR2 upscale execution with WebSocket progress.

import Foundation
import Logging

/// Callback type for generating images via the warm server pipeline.
/// Accepts a ComfyBridgeGenerateRequest and returns the output path + timing.
/// Progress callback sent during generation — maps to WebSocket progress events.
typealias ComfyBridgeProgressHandler = @Sendable (Int, Int) -> Void  // (stepIndex, totalSteps)

/// Callback type for generating images via the warm server pipeline.
/// Accepts a request and optional progress callback, returns output path + timing.
typealias ComfyBridgeGenerateHandler = @Sendable (ComfyBridgeGenerateRequest, ComfyBridgeProgressHandler?) async throws -> ComfyBridgeGenerateResult

/// Callback type for upscaling images via the SeedVR2 pipeline.
/// Accepts input image data, upscale model name, and optional progress callback.
/// Returns the output path + timing.
typealias ComfyBridgeUpscaleHandler = @Sendable (Data, String, ComfyBridgeProgressHandler?) async throws -> ComfyBridgeGenerateResult

/// Manages async generation execution and WebSocket event dispatch.
final class ComfyBridgeExecutor {
  private let logger: Logger
  private let wsManager: ComfyWebSocketManager
  private let imageCache: ComfyImageCache
  let generateHandler: ComfyBridgeGenerateHandler?
  let upscaleHandler: ComfyBridgeUpscaleHandler?

  /// Active prompt being executed, if any.
  private let lock = NSLock()
  private var activePromptId: String?

  init(
    logger: Logger,
    wsManager: ComfyWebSocketManager,
    imageCache: ComfyImageCache,
    generateHandler: ComfyBridgeGenerateHandler?,
    upscaleHandler: ComfyBridgeUpscaleHandler? = nil
  ) {
    self.logger = logger
    self.wsManager = wsManager
    self.imageCache = imageCache
    self.generateHandler = generateHandler
    self.upscaleHandler = upscaleHandler
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

    // --- Load inpaint images from cache if this is an inpaint request ---
    var mutableRequest = request
    if let imageId = request.inpaintImageId {
      guard let imageData = imageCache.retrieve(id: imageId) else {
        logger.error("ComfyBridge: inpaint image not found in cache: \(imageId)")
        sendError(promptId: request.promptId, clientId: request.clientId,
                  message: "Inpaint image not found: \(imageId)")
        return
      }
      mutableRequest.inpaintImageData = imageData
      logger.info("ComfyBridge: loaded inpaint image \(imageId) (\(imageData.count) bytes)")
    }
    if let maskId = request.maskImageId {
      guard let maskData = imageCache.retrieve(id: maskId) else {
        logger.error("ComfyBridge: mask image not found in cache: \(maskId)")
        sendError(promptId: request.promptId, clientId: request.clientId,
                  message: "Mask image not found: \(maskId)")
        return
      }
      mutableRequest.maskImageData = maskData
      logger.info("ComfyBridge: loaded mask image \(maskId) (\(maskData.count) bytes)")
    }
    if let controlImageId = request.controlImageId {
      guard let controlData = imageCache.retrieve(id: controlImageId) else {
        logger.error("ComfyBridge: control image not found in cache: \(controlImageId)")
        sendError(promptId: request.promptId, clientId: request.clientId,
                  message: "Control image not found: \(controlImageId)")
        return
      }
      mutableRequest.controlImageData = controlData
      logger.info("ComfyBridge: loaded control image \(controlImageId) (\(controlData.count) bytes)")
    }

    let promptPreview = mutableRequest.prompt.count > 80
      ? String(mutableRequest.prompt.prefix(77)) + "..."
      : mutableRequest.prompt
    let modeLabel = mutableRequest.isControlNet ? "controlnet" : (mutableRequest.isInpaint ? "inpaint" : "txt2img")
    logger.info("ComfyBridge: executing \(modeLabel) prompt_id=\(mutableRequest.promptId) — \(mutableRequest.width)x\(mutableRequest.height), \(mutableRequest.steps) steps, denoise=\(mutableRequest.denoise)")
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
      // Create a progress callback that sends WebSocket events.
      let progressCallback: ComfyBridgeProgressHandler = { [wsManager, clientId = request.clientId, promptId = request.promptId] step, total in
        let event: [String: Any] = [
          "type": "progress",
          "data": [
            "prompt_id": promptId,
            "value": step,
            "max": total
          ] as [String: Any]
        ]
        if let data = try? JSONSerialization.data(withJSONObject: event),
           let text = String(data: data, encoding: .utf8) {
          wsManager.send(to: clientId, text: text)
        }
      }

      let result = try await handler(mutableRequest, progressCallback)

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

  /// Execute a parsed upscale request asynchronously.
  /// Sends the full sequence of ComfyUI WebSocket events during upscale.
  func executeUpscale(_ request: ComfyBridgeUpscaleRequest) async {
    guard let handler = upscaleHandler else {
      logger.error("ComfyBridge: no upscale handler configured — upscale disabled")
      sendError(promptId: request.promptId, clientId: request.clientId,
                message: "Upscale not configured on this server")
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

    // --- Load input image from cache ---
    var mutableRequest = request
    guard let inputImageData = imageCache.retrieve(id: request.inputImageNodeId) else {
      logger.error("ComfyBridge: upscale input image not found in cache: \(request.inputImageNodeId)")
      sendError(promptId: request.promptId, clientId: request.clientId,
                message: "Upscale input image not found: \(request.inputImageNodeId)")
      return
    }
    mutableRequest.inputImageData = inputImageData

    logger.info("ComfyBridge: executing upscale prompt_id=\(mutableRequest.promptId) — model=\(mutableRequest.upscaleModelName), input=\(inputImageData.count) bytes")

    // --- Phase 1: execution_start ---
    sendEvent(to: request.clientId, type: "execution_start", data: [
      "prompt_id": request.promptId
    ])

    // --- Phase 2: simulate cached loader nodes ---
    // Upscale workflows have fewer nodes — just mark the upscale model loader as cached.
    sendEvent(to: request.clientId, type: "execution_cached", data: [
      "prompt_id": request.promptId,
      "nodes": ["1", "2"]
    ])

    // The upscale node is where actual work happens.
    sendEvent(to: request.clientId, type: "executing", data: [
      "prompt_id": request.promptId,
      "node": "3"  // Upscale processing node
    ])

    // --- Phase 3: run actual upscale ---
    do {
      // Create a progress callback that sends WebSocket events.
      let progressCallback: ComfyBridgeProgressHandler = { [wsManager, clientId = request.clientId, promptId = request.promptId] step, total in
        let event: [String: Any] = [
          "type": "progress",
          "data": [
            "prompt_id": promptId,
            "value": step,
            "max": total
          ] as [String: Any]
        ]
        if let data = try? JSONSerialization.data(withJSONObject: event),
           let text = String(data: data, encoding: .utf8) {
          wsManager.send(to: clientId, text: text)
        }
      }

      let result = try await handler(inputImageData, mutableRequest.upscaleModelName, progressCallback)

      // Read the output image.
      let outputURL = URL(fileURLWithPath: result.outputPath)
      guard let imageData = try? Data(contentsOf: outputURL) else {
        logger.error("ComfyBridge: failed to read upscale output at \(result.outputPath)")
        sendError(promptId: request.promptId, clientId: request.clientId,
                  message: "Failed to read upscaled image")
        return
      }

      // Store in the image cache.
      let imageId = UUID().uuidString
      guard imageCache.store(id: imageId, data: imageData) else {
        logger.error("ComfyBridge: failed to cache upscale output image \(imageId)")
        sendError(promptId: request.promptId, clientId: request.clientId,
                  message: "Failed to cache upscaled image")
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

      logger.info("ComfyBridge: upscale complete — prompt_id=\(request.promptId), \(result.durationMs)ms, image=\(imageId) (\(imageData.count) bytes)")

      // Clean up the temp file — the image is now in the cache.
      try? FileManager.default.removeItem(at: outputURL)

    } catch {
      logger.error("ComfyBridge: upscale failed — prompt_id=\(request.promptId): \(error)")
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
