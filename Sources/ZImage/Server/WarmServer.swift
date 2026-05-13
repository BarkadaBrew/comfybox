import Foundation
import Dispatch
import Logging
import Network
import Darwin
import MLX

public struct WarmServerConfiguration: Sendable {
  public var port: UInt16
  public var modelSpec: String?
  public var textEncoderPath: String?
  public var initialLoRAs: [LoRAConfiguration]
  public var forceTransformerOverrideOnly: Bool
  public var maxSequenceLength: Int
  public var maxPendingRequests: Int
  public var allowedOutputDirectory: String
  /// Path to SeedVR2 upscale model weights directory.
  /// When set, enables upscale via the ComfyUI bridge. The pipeline is lazy-loaded
  /// on first upscale request to avoid the ~6GB memory cost until needed.
  public var seedvr2WeightsPath: String?

  public init(
    port: UInt16 = 7862,
    modelSpec: String? = nil,
    textEncoderPath: String? = nil,
    initialLoRAs: [LoRAConfiguration] = [],
    forceTransformerOverrideOnly: Bool = false,
    maxSequenceLength: Int = 512,
    maxPendingRequests: Int = 10,
    allowedOutputDirectory: String = FileManager.default.currentDirectoryPath,
    seedvr2WeightsPath: String? = nil
  ) {
    self.port = port
    self.modelSpec = modelSpec
    self.textEncoderPath = textEncoderPath
    self.initialLoRAs = initialLoRAs
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
    self.maxSequenceLength = maxSequenceLength
    self.maxPendingRequests = max(1, maxPendingRequests)
    self.allowedOutputDirectory = allowedOutputDirectory
    self.seedvr2WeightsPath = seedvr2WeightsPath
  }
}

/// Model family used by the warm server to route generation to the correct pipeline.
enum WarmModelFamily: String, Sendable {
  case flux1
  case flux2
  case fibo
  case chroma
}

enum WarmServerOutputPathValidator {
  static func resolveOutputPath(_ outputPath: String, allowedOutputDirectory: String) throws -> URL {
    let allowedURL = canonicalFileURL(for: allowedOutputDirectory)
    let outputURL = canonicalFileURL(for: outputPath)

    guard outputURL.isContained(in: allowedURL) else {
      throw WarmServerError.invalidOutputPath(path: outputURL.path, allowedDirectory: allowedURL.path)
    }

    return outputURL
  }

  private static func canonicalFileURL(for path: String) -> URL {
    let expandedPath = (path as NSString).expandingTildeInPath
    let absolutePath: String
    if expandedPath.hasPrefix("/") {
      absolutePath = expandedPath
    } else {
      absolutePath = (FileManager.default.currentDirectoryPath as NSString)
        .appendingPathComponent(expandedPath)
    }

    return resolvePathComponents(in: absolutePath)
  }

  private static func resolvePathComponents(in path: String, symlinkDepth: Int = 0) -> URL {
    let fileManager = FileManager.default
    var currentURL = URL(fileURLWithPath: "/")

    for component in (path as NSString).pathComponents.dropFirst() {
      switch component {
      case "", ".":
        continue
      case "..":
        currentURL = currentURL.deletingLastPathComponent()
      default:
        let nextURL = currentURL.appendingPathComponent(component)
        if let destination = try? fileManager.destinationOfSymbolicLink(atPath: nextURL.path),
           symlinkDepth < 32 {
          let destinationPath: String
          if destination.hasPrefix("/") {
            destinationPath = destination
          } else {
            destinationPath = (currentURL.path as NSString).appendingPathComponent(destination)
          }
          currentURL = resolvePathComponents(in: destinationPath, symlinkDepth: symlinkDepth + 1)
        } else if fileManager.fileExists(atPath: nextURL.path) {
          currentURL = nextURL.resolvingSymlinksInPath()
        } else {
          currentURL = nextURL
        }
      }
    }

    return currentURL
  }
}

private extension URL {
  func isContained(in directory: URL) -> Bool {
    let pathComponents = standardizedFileURL.pathComponents
    let directoryComponents = directory.standardizedFileURL.pathComponents
    guard pathComponents.count >= directoryComponents.count else { return false }
    return Array(pathComponents.prefix(directoryComponents.count)) == directoryComponents
  }
}

public final class WarmServer {
  private static let pngSignature: [UInt8] = [137, 80, 78, 71, 13, 10, 26, 10]

  private let configuration: WarmServerConfiguration
  private let host: String
  private let logger: Logger
  private let coordinator: WarmServerCoordinator
  let comfyBridge: ComfyBridge
  private let listenerQueue = DispatchQueue(label: "z-image.warm-server.listener")
  private let lifecycleLock = NSLock()
  private var listener: NWListener?
  private var shutdownSignalled = false

  /// Lazy-loaded SeedVR2 upscale pipeline. Created on first upscale request
  /// to avoid the ~6GB memory cost until actually needed.
  private var seedvr2Pipeline: SeedVR2Pipeline?
  /// Resolved path to SeedVR2 weights directory.
  private let seedvr2WeightsPath: String?

  /// Lazy-loaded ESRGAN upscale pipeline. Created on first ESRGAN upscale request.
  private var esrganPipeline: ESRGANPipeline?

  /// LoRA Library — indexes, queries, and manages LoRA adapter files.
  /// Initialized at startup; auto-scans if no library.json exists.
  private var loraLibrary: LoRALibrary?

  /// Default upscale models directory path — ESRGAN weights are stored here.
  private static let upscaleModelsDirectoryPath = ("~/bin/zimage/upscale_models" as NSString).expandingTildeInPath

  public init(
    configuration: WarmServerConfiguration,
    host: String = "127.0.0.1",
    logger: Logger = Logger(label: "z-image.warm-server")
  ) {
    self.configuration = configuration
    self.host = host
    self.logger = logger
    self.coordinator = WarmServerCoordinator(configuration: configuration, logger: logger)
    self.seedvr2WeightsPath = configuration.seedvr2WeightsPath

    self.comfyBridge = ComfyBridge(logger: logger)

    // Initialize the LoRA Library. The library root defaults to ~/Models/loras/
    // (via COMFYBOX_MODELS env or LoRALibrary default). If no library.json exists,
    // the first API call to /v1/loras/scan will create it.
    do {
      let library = try LoRALibrary(logger: logger)
      self.loraLibrary = library

      // Auto-scan if no library.json exists yet (first run).
      if library.count == 0 {
        logger.info("LoRA Library: no index found, running initial scan...")
        let result = try library.scan()
        logger.info("LoRA Library: initial scan complete — \(result.added) LoRAs indexed")
      } else {
        logger.info("LoRA Library: loaded \(library.count) entries from index")
      }

      // Wire the library into the ComfyBridge for LoRA discovery.
      comfyBridge.loraLibrary = library
    } catch {
      logger.warning("LoRA Library: failed to initialize — \(error.localizedDescription). LoRA API endpoints will return 503.")
    }

    // Wire up the upscale handler. ESRGAN models are always available (lazy-loaded from
    // ~/bin/zimage/upscale_models/); SeedVR2 additionally requires a configured weights path.
    let upscaleHandler: ComfyBridgeUpscaleHandler? = { [unowned self] (imageData: Data, modelName: String, progressCallback: ComfyBridgeProgressHandler?) async throws -> ComfyBridgeGenerateResult in
      try await self.bridgeUpscale(imageData: imageData, modelName: modelName, progressCallback: progressCallback)
    }

    self.comfyBridge.configureExecutor(
      generateHandler: { [unowned self] request, progressCallback in
        try await self.bridgeGenerate(request, progressCallback: progressCallback)
      },
      upscaleHandler: upscaleHandler
    )

    // Wire queue status provider and clear handler for ComfyUI /queue endpoint.
    self.comfyBridge.queueStatusProvider = { [unowned self] in
      await self.coordinator.queueStatus()
    }
    self.comfyBridge.queueClearHandler = { [unowned self] in
      let cleared = await self.coordinator.clearPending()
      self.logger.info("ComfyBridge: cleared \(cleared) pending job(s) from queue")
    }

    // Wire model switch handler for Krita checkpoint auto-detection.
    // When Krita sends a workflow with a different checkpoint, this handler
    // checks if the model is already in the pool (activate) or needs loading.
    self.comfyBridge.modelSwitchHandler = { [unowned self] (modelId: String) async throws -> Bool in
      // Check if this model is already active — no switch needed.
      let currentActive = await self.coordinator.modelPool.activeModelId()
      let requestedKey = ModelPool.poolKey(for: modelId)
      if currentActive == requestedKey {
        return false
      }

      // Check if the model is already in the pool — just activate it (instant).
      if let existing = await self.coordinator.modelPool.findEntry(for: modelId) {
        try await self.coordinator.poolActivate(modelId: existing.id)
        self.logger.info("ComfyBridge: activated pool model '\(existing.id)' for Krita checkpoint switch")
        return true
      }

      // Model not in pool — load and activate it.
      let quantization = Self.parseQuantization(from: modelId)
      let modelSpec = Self.parseModelSpec(from: modelId)
      let result = try await self.coordinator.poolLoad(modelSpec: modelSpec, quantization: quantization, activate: true)
      self.logger.info("ComfyBridge: loaded + activated '\(result.model)' (\(result.loadTimeMs)ms) for Krita checkpoint switch")
      return true
    }
  }

  public func run() throws {
    // Ignore SIGHUP before model loading — prevents SSH disconnect from
    // killing the daemon during the ~40s pipeline initialization phase.
    signal(SIGHUP, SIG_IGN)

    try preparePipeline()

    guard let port = NWEndpoint.Port(rawValue: configuration.port) else {
      throw WarmServerError.invalidPort(configuration.port)
    }

    // Handle SIGTERM for clean launchd stop/restart.
    let sigTermSource = DispatchSource.makeSignalSource(signal: SIGTERM, queue: listenerQueue)
    signal(SIGTERM, SIG_IGN)
    sigTermSource.setEventHandler { [weak self] in
      self?.logger.info("Received SIGTERM, shutting down gracefully...")
      self?.initiateShutdown()
    }
    sigTermSource.resume()

    // Handle SIGINT for clean Ctrl-C during development.
    let sigIntSource = DispatchSource.makeSignalSource(signal: SIGINT, queue: listenerQueue)
    signal(SIGINT, SIG_IGN)
    sigIntSource.setEventHandler { [weak self] in
      self?.logger.info("Received SIGINT, shutting down...")
      self?.initiateShutdown()
    }
    sigIntSource.resume()

    let endpoint = NWEndpoint.hostPort(host: NWEndpoint.Host(host), port: port)
    let parameters = NWParameters.tcp
    parameters.requiredLocalEndpoint = endpoint
    let listener = try NWListener(using: parameters)
    self.listener = listener

    listener.stateUpdateHandler = { [weak self] state in
      self?.handleListenerState(state)
    }
    listener.newConnectionHandler = { [weak self] connection in
      self?.accept(connection: connection)
    }

    listener.start(queue: listenerQueue)

    // Use dispatchMain() instead of semaphore.wait() for daemon reliability.
    // DispatchSemaphore.wait() blocks without processing GCD events, which
    // causes NWListener to enter .cancelled state when the process loses its
    // controlling terminal (SSH disconnect, launchd restart, nohup).
    // dispatchMain() keeps the main dispatch loop alive properly.
    dispatchMain()
  }

  private func preparePipeline() throws {
    let result = SyncResult<Void>()
    Task {
      do {
        try await coordinator.prepare()
        result.succeed(())
      } catch {
        result.fail(error)
      }
    }
    try result.wait()
  }

  private func handleListenerState(_ state: NWListener.State) {
    switch state {
    case .ready:
      logger.info("Warm server listening on http://\(self.host):\(self.configuration.port)")
    case .failed(let error):
      logger.error("Warm server listener failed: \(error.localizedDescription)")
      initiateShutdown(exitCode: 1)
    case .cancelled:
      // Only exit if we intentionally cancelled (via /v1/shutdown or signal).
      // NWListener can be cancelled by macOS when the process loses its
      // controlling terminal — we must NOT treat that as a shutdown request.
      lifecycleLock.lock()
      let wasIntentional = shutdownSignalled
      lifecycleLock.unlock()

      if wasIntentional {
        logger.info("Listener cancelled (intentional shutdown)")
        exit(0)
      } else {
        logger.warning("Listener cancelled unexpectedly — ignoring (daemon will continue)")
      }
    default:
      break
    }
  }

  private func accept(connection: NWConnection) {
    let handler = ConnectionHandler(
      connection: connection,
      queue: DispatchQueue(label: "z-image.warm-server.connection.\(UUID().uuidString)"),
      server: self
    )
    handler.start()
  }

  fileprivate func respond(to request: HTTPRequest) async -> RoutedResponse {
    // Try ComfyUI bridge routes first.
    if let bridgeResponse = comfyBridge.route(request) {
      return bridgeResponse
    }

    switch (request.method, request.path) {
    case ("GET", "/health"):
      let memoryBytes = Self.currentMemoryFootprintBytes()
      let health = await coordinator.health(memoryBytes: memoryBytes)
      return .json(status: 200, payload: health)

    case ("POST", "/v1/generate"):
      do {
        let payload = try decode(GeneratePayload.self, from: request.body)
        try payload.validateOutputPath(configuration: configuration)
        let result = try await coordinator.enqueueGenerate(payload)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/lora/swap"):
      do {
        let payload = try decode(LoRASwapPayload.self, from: request.body)
        let result = try await coordinator.enqueueSwap(payload)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/shutdown"):
      do {
        let result = try await coordinator.enqueueShutdown()
        return .shutdown(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("GET", "/v1/models"):
      let models = ComfyBoxModelRegistry.allModels.map { model -> [String: Any] in
        [
          "id": model.id,
          "family": model.family.rawValue,
          "variant": model.variant.rawValue,
          "quantization": model.quantization.rawValue,
          "display_name": model.displayName,
          "description": model.description,
          "parameters_b": model.parametersBillions,
          "default_steps": model.defaultSteps,
          "default_guidance": model.defaultGuidance,
          "supports_guidance": model.supportsGuidance,
          "supports_lora": model.supportsLoRA,
          "supports_controlnet": model.supportsControlNet,
          "supports_img2img": model.supportsImg2Img,
          "default_resolution": "\(model.defaultWidth)x\(model.defaultHeight)",
          "estimated_vram_gb": model.estimatedVRAM_GB,
          "huggingface_id": model.huggingFaceId,
        ] as [String: Any]
      }
      if let data = try? JSONSerialization.data(
        withJSONObject: ["models": models, "count": models.count]
      ) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize models"))

    case ("GET", "/v1/styles"):
      let styles = ComfyBoxStylePresets.toJSON()
      if let data = try? JSONSerialization.data(
        withJSONObject: ["styles": styles, "count": styles.count]
      ) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize styles"))

    // MARK: - Model Pool Endpoints

    case ("POST", "/v1/model/load"):
      do {
        let payload = try decode(ModelLoadRequest.self, from: request.body)
        let shouldActivate = payload.activate ?? true
        let shouldWait = payload.wait ?? true

        if shouldWait {
          let result = try await coordinator.poolLoad(
            modelSpec: payload.model,
            quantization: payload.quantization,
            activate: shouldActivate
          )
          return .json(status: 200, payload: result)
        } else {
          // Fire-and-forget: start loading in background, return immediately.
          Task {
            do {
              try await coordinator.poolLoad(
                modelSpec: payload.model,
                quantization: payload.quantization,
                activate: shouldActivate
              )
            } catch {
              logger.error("ModelPool: background load failed for '\(payload.model)': \(error.localizedDescription)")
            }
          }
          let ack = ModelLoadResponse(
            status: "loading",
            model: payload.model,
            family: "pending",
            loadTimeMs: 0,
            vramEstimateMB: 0,
            poolSize: await coordinator.modelPool.count(),
            poolBudgetMB: await coordinator.modelPool.budget()
          )
          return .json(status: 202, payload: ack)
        }
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/model/activate"):
      do {
        let payload = try decode(ModelActivateRequest.self, from: request.body)
        let result = try await coordinator.poolActivate(modelId: payload.model)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("GET", "/v1/model/pool"):
      let result = await coordinator.poolList()
      return .json(status: 200, payload: result)

    case ("POST", "/v1/model/unload"):
      do {
        let payload = try decode(ModelUnloadRequest.self, from: request.body)
        let result = try await coordinator.poolUnload(modelId: payload.model)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    // MARK: - LoRA Library Endpoints

    case ("GET", "/v1/loras"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let allEntries = library.list(includeQuarantined: true)
      let activeLoRANames = await coordinator.activeLoRAIdentifiers
      let quarantinedCount = allEntries.filter { $0.quarantined }.count

      var loraList: [[String: Any]] = []
      for entry in allEntries {
        var dict: [String: Any] = [
          "id": entry.id,
          "filename": entry.filename,
          "model_compatibility": entry.modelCompatibility,
          "format": entry.format.rawValue,
          "rank": entry.rank,
          "size_bytes": entry.sizeBytes,
          "quarantined": entry.quarantined,
          "tags": entry.tags,
          "category": entry.category,
          "triggerwords": entry.triggerwords,
          "recommended_scale": entry.recommendedScale,
          "date_added": entry.dateAdded,
        ]
        if let reason = entry.quarantineReason { dict["quarantine_reason"] = reason }
        if !entry.notes.isEmpty { dict["notes"] = entry.notes }
        loraList.append(dict)
      }

      let responseDict: [String: Any] = [
        "loras": loraList,
        "active_loras": activeLoRANames,
        "total": allEntries.count,
        "quarantined": quarantinedCount,
      ]
      if let data = try? JSONSerialization.data(withJSONObject: responseDict) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize LoRA list"))

    case ("GET", _) where request.path.hasPrefix("/v1/loras/"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let id = String(request.path.dropFirst("/v1/loras/".count))
      guard !id.isEmpty, !id.contains("/") else {
        return .error(.error(status: 400, message: "Invalid LoRA ID"))
      }
      guard let entry = library.entry(for: id) else {
        return .error(.error(status: 404, message: "LoRA not found: \(id)"))
      }

      var dict: [String: Any] = [
        "id": entry.id,
        "filename": entry.filename,
        "relative_path": entry.relativePath,
        "size_bytes": entry.sizeBytes,
        "size_formatted": entry.sizeFormatted,
        "model_compatibility": entry.modelCompatibility,
        "format": entry.format.rawValue,
        "rank": entry.rank,
        "key_count": entry.keyCount,
        "layer_targets": entry.layerTargets,
        "triggerwords": entry.triggerwords,
        "recommended_scale": entry.recommendedScale,
        "scale_range": entry.scaleRange,
        "tags": entry.tags,
        "category": entry.category,
        "notes": entry.notes,
        "date_added": entry.dateAdded,
        "quarantined": entry.quarantined,
      ]
      if let sha = entry.sha256 { dict["sha256"] = sha }
      if let alpha = entry.alpha { dict["alpha"] = alpha }
      if let reason = entry.quarantineReason { dict["quarantine_reason"] = reason }
      if let url = entry.sourceURL { dict["source_url"] = url }
      if let civitaiId = entry.civitaiModelId { dict["civitai_model_id"] = civitaiId }
      if let meta = entry.safetensorsMetadata { dict["safetensors_metadata"] = meta }

      if let data = try? JSONSerialization.data(withJSONObject: dict) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize LoRA entry"))

    case ("POST", "/v1/loras/scan"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      do {
        let force: Bool
        if !request.body.isEmpty,
           let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
           let f = json["force"] as? Bool {
          force = f
        } else {
          force = false
        }
        let result = try library.scan(force: force)
        let responseDict: [String: Any] = [
          "added": result.added,
          "updated": result.updated,
          "removed": result.removed,
          "unchanged": result.unchanged,
          "total": result.total,
          "errors": result.errors.map { ["file": $0.0, "error": $0.1] },
        ]
        if let data = try? JSONSerialization.data(withJSONObject: responseDict) {
          return .json(.rawJSON(status: 200, data: data))
        }
        return .error(.error(status: 500, message: "Failed to serialize scan result"))
      } catch {
        return .error(.error(status: 500, message: "Scan failed: \(error.localizedDescription)"))
      }

    case ("POST", _) where request.path.hasSuffix("/quarantine") && request.path.hasPrefix("/v1/loras/"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let pathBody = String(request.path.dropFirst("/v1/loras/".count).dropLast("/quarantine".count))
      guard !pathBody.isEmpty, !pathBody.contains("/") else {
        return .error(.error(status: 400, message: "Invalid LoRA ID"))
      }
      do {
        let reason: String
        if !request.body.isEmpty,
           let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
           let r = json["reason"] as? String {
          reason = r
        } else {
          reason = "Quarantined via API"
        }
        try library.quarantine(pathBody, reason: reason)
        return .json(.rawJSON(status: 200, data: Data("{\"success\":true,\"id\":\"\(pathBody)\",\"quarantined\":true}".utf8)))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: error.localizedDescription))
      }

    case ("DELETE", _) where request.path.hasSuffix("/quarantine") && request.path.hasPrefix("/v1/loras/"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let pathBody = String(request.path.dropFirst("/v1/loras/".count).dropLast("/quarantine".count))
      guard !pathBody.isEmpty, !pathBody.contains("/") else {
        return .error(.error(status: 400, message: "Invalid LoRA ID"))
      }
      do {
        try library.unquarantine(pathBody)
        return .json(.rawJSON(status: 200, data: Data("{\"success\":true,\"id\":\"\(pathBody)\",\"quarantined\":false}".utf8)))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: error.localizedDescription))
      }

    default:
      if ["/v1/generate", "/v1/lora/swap", "/v1/shutdown", "/health",
          "/v1/model/load", "/v1/model/activate", "/v1/model/pool", "/v1/model/unload",
          "/v1/loras", "/v1/loras/scan"
      ].contains(request.path) || request.path.hasPrefix("/v1/loras/") {
        return .error(.error(status: 405, message: "Method not allowed"))
      }
      return .error(.error(status: 404, message: "Not found"))
    }
  }


  /// Bridge a ComfyUI workflow request to the internal generate pipeline.
  /// Called by ComfyBridgeExecutor via the closure set in init.
  /// Read PNG dimensions from IHDR chunk (bytes 16-23 of a valid PNG).
  private func pngDimensions(from data: Data) -> (width: Int, height: Int)? {
    guard data.count >= 24, data.prefix(Self.pngSignature.count).elementsEqual(Self.pngSignature) else { return nil }
    let w = Int(data[16]) << 24 | Int(data[17]) << 16 | Int(data[18]) << 8 | Int(data[19])
    let h = Int(data[20]) << 24 | Int(data[21]) << 16 | Int(data[22]) << 8 | Int(data[23])
    return (w, h)
  }

  /// Round up to nearest multiple of 16 (for latent alignment).
  private func roundTo16(_ n: Int) -> Int {
    return ((n + 15) / 16) * 16
  }

  /// Default LoRA directory path — matches ComfyBridgeObjectInfo discovery path.
  private static let loraDirectoryPath = ("~/bin/zimage/loras" as NSString).expandingTildeInPath

  /// Default ControlNet directory path — matches ComfyBridgeObjectInfo discovery path.
  private static let controlnetDirectoryPath = ("~/bin/zimage/controlnet" as NSString).expandingTildeInPath

  private func bridgeGenerate(_ request: ComfyBridgeGenerateRequest, progressCallback: ComfyBridgeProgressHandler?) async throws -> ComfyBridgeGenerateResult {
    // --- Phase 4: Dynamic LoRA swap ---
    // If the workflow contains LoraLoader nodes, swap LoRAs before generating.
    // The coordinator serializes operations, so swap completes before generate starts.
    if !request.loras.isEmpty {
      let loraEntries = request.loras.map { lora -> LoRAEntry in
        // Resolve bare filenames to full paths in the LoRA directory.
        let resolvedPath: String
        if lora.name.contains("/") || lora.name.hasPrefix("~") {
          resolvedPath = lora.name
        } else {
          resolvedPath = Self.loraDirectoryPath + "/" + lora.name
        }
        return LoRAEntry(path: resolvedPath, scale: lora.scale)
      }
      let swapPayload = LoRASwapPayload(loras: loraEntries)
      let swapResult = try await coordinator.enqueueSwap(swapPayload)
      logger.info("WarmServer: bridge LoRA swap complete — \(swapResult.loraCount) LoRA(s) active")
    }

    // Derive dimensions from inpaint image if parser returned 0x0
    // (happens when workflow has no ImageCrop or EmptyLatentImage nodes)
    var genWidth = request.width
    var genHeight = request.height
    if genWidth == 0 || genHeight == 0, let imgData = request.inpaintImageData {
      if let dims = pngDimensions(from: imgData) {
        genWidth = roundTo16(dims.width)
        genHeight = roundTo16(dims.height)
        logger.info("WarmServer: derived dimensions from inpaint image: \(dims.width)x\(dims.height) -> \(genWidth)x\(genHeight)")
      } else {
        genWidth = 1024
        genHeight = 1024
        logger.warning("WarmServer: could not read inpaint image dimensions, falling back to 1024x1024")
      }
    }

    // --- Phase 4: ControlNet routing ---
    // If the workflow contains ControlNet nodes, route to ZImageControlPipeline
    // instead of the standard ZImagePipeline.
    // ControlNet is not supported for Flux 2 models.
    if request.isControlNet, let controlnetModel = request.controlnetModel {
      if await coordinator.modelFamily == .flux2 {
        throw WarmServerError.controlNetNotSupported
      }
      logger.info("WarmServer: routing to ControlNet pipeline — model=\(controlnetModel), strength=\(request.controlnetStrength)")

      // Resolve controlnet model name to a path or HuggingFace ID
      let resolvedControlnetWeights: String
      if controlnetModel.contains("/") || controlnetModel.hasPrefix("~") || controlnetModel.hasPrefix(".") {
        // Already a path or HuggingFace ID — use as-is
        resolvedControlnetWeights = controlnetModel
      } else {
        // Bare name — check if it's a local directory/file in the controlnet dir
        let localPath = Self.controlnetDirectoryPath + "/" + controlnetModel
        if FileManager.default.fileExists(atPath: localPath) {
          resolvedControlnetWeights = localPath
        } else {
          // Treat as HuggingFace ID
          resolvedControlnetWeights = controlnetModel
        }
      }

      // Write control image data to a temp file if we have it
      var controlImageURL: URL? = nil
      if let controlData = request.controlImageData {
        let tempPath = NSTemporaryDirectory() + "zimage-control-\(UUID().uuidString).png"
        try controlData.write(to: URL(fileURLWithPath: tempPath))
        controlImageURL = URL(fileURLWithPath: tempPath)
        logger.info("WarmServer: wrote control image to \(tempPath) (\(controlData.count) bytes)")
      }

      // Write inpaint image to temp file if present
      var inpaintImageURL: URL? = nil
      if let inpaintData = request.inpaintImageData {
        let tempPath = NSTemporaryDirectory() + "zimage-inpaint-\(UUID().uuidString).png"
        try inpaintData.write(to: URL(fileURLWithPath: tempPath))
        inpaintImageURL = URL(fileURLWithPath: tempPath)
      }

      // Write mask to temp file if present
      var maskImageURL: URL? = nil
      if let maskData = request.maskImageData {
        let tempPath = NSTemporaryDirectory() + "zimage-mask-\(UUID().uuidString).png"
        try maskData.write(to: URL(fileURLWithPath: tempPath))
        maskImageURL = URL(fileURLWithPath: tempPath)
      }

      let outputURL = URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent("zimage-\(UUID().uuidString).png")

      // Build LoRA configurations for the control pipeline
      let controlLoRAs: [LoRAConfiguration] = request.loras.map { lora in
        let resolvedPath: String
        if lora.name.contains("/") || lora.name.hasPrefix("~") {
          resolvedPath = lora.name
        } else {
          resolvedPath = Self.loraDirectoryPath + "/" + lora.name
        }
        return .local(resolvedPath, scale: lora.scale)
      }

      let controlRequest = ZImageControlGenerationRequest(
        prompt: request.prompt,
        negativePrompt: nil,
        controlImage: controlImageURL,
        inpaintImage: inpaintImageURL,
        maskImage: maskImageURL,
        controlContextScale: request.controlnetStrength,
        width: genWidth,
        height: genHeight,
        steps: request.steps,
        guidanceScale: 0.0,
        seed: request.seed,
        outputPath: outputURL,
        model: nil,
        textEncoderPath: configuration.textEncoderPath,
        controlnetWeights: resolvedControlnetWeights,
        // For HuggingFace repos with multiple safetensors, specify the 8-step variant
        controlnetWeightsFile: resolvedControlnetWeights.contains("alibaba-pai")
          ? "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors" : nil,
        maxSequenceLength: configuration.maxSequenceLength,
        loras: controlLoRAs,
        progressCallback: progressCallback.map { callback in
          return { progress in
            if progress.stage == "Denoising" {
              callback(progress.stepIndex, progress.totalSteps)
            }
          }
        },
        enhancePrompt: false,
        enhanceMaxTokens: 512
      )

      let start = Date()
      let result = try await coordinator.enqueueControlGenerate(controlRequest)
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)

      // Clean up temp files
      if let url = controlImageURL { try? FileManager.default.removeItem(at: url) }
      if let url = inpaintImageURL { try? FileManager.default.removeItem(at: url) }
      if let url = maskImageURL { try? FileManager.default.removeItem(at: url) }

      return ComfyBridgeGenerateResult(
        outputPath: result.outputPath,
        durationMs: result.durationMs
      )
    }

    // Family-aware defaults for step clamping, guidance, and negative prompts.
    let family = await coordinator.modelFamily
    let resolvedSteps: Int
    let resolvedGuidance: Float
    let resolvedNegativePrompt: String?

    switch family {
    case .fibo:
      // FIBO: use model defaults, no step clamping
      resolvedSteps = request.steps
      resolvedGuidance = request.guidance > 0 ? request.guidance : 4.0
      resolvedNegativePrompt = request.negativePrompt
    case .chroma:
      // Chroma: 28 steps default, guidance 0.0 (unconditioned)
      resolvedSteps = request.steps > 0 ? request.steps : 28
      resolvedGuidance = request.guidance
      resolvedNegativePrompt = nil
    case .flux1:
      let zimageVariant = await coordinator.currentZImageVariant
      if zimageVariant == .base {
        // Z-Image Base: non-distilled, supports CFG guidance and negative prompts
        resolvedSteps = request.steps
        resolvedGuidance = request.guidance > 0 ? request.guidance : ZImageModelMetadata.Base.recommendedGuidanceScale
        resolvedNegativePrompt = request.negativePrompt
      } else {
        // Z-Image Turbo: distilled, optimal at 9 steps, no CFG, no negative prompts
        resolvedSteps = min(request.steps, 9)
        resolvedGuidance = 0.0
        resolvedNegativePrompt = nil
      }
    case .flux2:
      // Base (non-distilled) models support guidance > 1.0 and default to 50 steps;
      // distilled models default to 4 steps and guidance 1.0.
      let isBaseModel = await coordinator.isFlux2BaseModel
      resolvedSteps = request.steps                 // Klein: no step clamp
      resolvedGuidance = isBaseModel ? request.guidance : 1.0
      resolvedNegativePrompt = nil                  // Klein: CFG only when guidance > 1.0
    }

    let payload = GeneratePayload(
      prompt: request.prompt,
      negativePrompt: resolvedNegativePrompt,
      width: genWidth,
      height: genHeight,
      steps: resolvedSteps,
      guidance: resolvedGuidance,
      seed: request.seed,
      outputPath: nil,
      levelsMin: request.levelsMin,
      levelsMax: request.levelsMax,
      scheduler: request.sampler,
      sigmaSchedule: request.sigmaSchedule,
      inpaintImageData: request.inpaintImageData,
      maskData: request.maskImageData,
      denoise: request.denoise,
      maskGrow: request.maskGrow,
      maskFeather: request.maskFeather,
      maskCropX: request.maskCropX,
      maskCropY: request.maskCropY
    )

    // Convert bridge progress callback to pipeline progress handler.
    let pipelineProgress: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = progressCallback.map { callback in
      return { progress in
        if progress.stage == .denoising {
          callback(progress.stepIndex, progress.totalSteps)
        }
      }
    }

    // Batch generation: if batchSize > 1 (from RepeatLatentBatch), loop and return last result.
    if request.batchSize > 1 {
      logger.info("WarmServer: batch generation — \(request.batchSize) images")
      var lastResult: ComfyBridgeGenerateResult?
      var totalDurationMs = 0
      for i in 0..<request.batchSize {
        // Vary seed per batch item for unique outputs.
        var batchPayload = payload
        if let baseSeed = request.seed {
          batchPayload = GeneratePayload(
            prompt: payload.prompt,
            negativePrompt: payload.negativePrompt,
            width: payload.width,
            height: payload.height,
            steps: payload.steps,
            guidance: payload.guidance,
            seed: baseSeed + UInt64(i),
            outputPath: payload.outputPath,
            levelsMin: payload.levelsMin,
            levelsMax: payload.levelsMax,
            scheduler: payload.scheduler,
            sigmaSchedule: payload.sigmaSchedule,
            inpaintImageData: payload.inpaintImageData,
            maskData: payload.maskData,
            denoise: payload.denoise,
            maskGrow: payload.maskGrow,
            maskFeather: payload.maskFeather,
            maskCropX: payload.maskCropX,
            maskCropY: payload.maskCropY
          )
        }
        let result = try await coordinator.enqueueGenerate(batchPayload, progressHandler: pipelineProgress)
        totalDurationMs += result.durationMs
        lastResult = ComfyBridgeGenerateResult(outputPath: result.outputPath, durationMs: totalDurationMs)
      }
      return lastResult!
    }

    let result = try await coordinator.enqueueGenerate(payload, progressHandler: pipelineProgress)
    return ComfyBridgeGenerateResult(
      outputPath: result.outputPath,
      durationMs: result.durationMs
    )
  }

  /// Known ESRGAN model name patterns.
  /// If the upscale model name matches any of these, route to ESRGANPipeline.
  private static let esrganModelPatterns: [String] = [
    "RealESRGAN_x4",
    "4x-UltraSharp",
    "4xNomos8k",
    "4x_NMKD-Superscale",
    "OmniSR_",
  ]

  /// Whether the given upscale model name should be routed to ESRGAN.
  private static func isESRGANModel(_ modelName: String) -> Bool {
    esrganModelPatterns.contains { modelName.hasPrefix($0) || modelName.contains($0) }
  }

  /// Bridge a ComfyUI upscale workflow request to the appropriate upscale pipeline.
  /// Routes to ESRGANPipeline for ESRGAN-family models, SeedVR2Pipeline for SeedVR2.
  /// Both pipelines are lazy-loaded on first use to avoid startup memory costs.
  private func bridgeUpscale(
    imageData: Data,
    modelName: String,
    progressCallback: ComfyBridgeProgressHandler?
  ) async throws -> ComfyBridgeGenerateResult {
    if Self.isESRGANModel(modelName) {
      return try await bridgeUpscaleESRGAN(imageData: imageData, modelName: modelName)
    } else {
      return try await bridgeUpscaleSeedVR2(imageData: imageData, modelName: modelName, progressCallback: progressCallback)
    }
  }

  /// ESRGAN upscale path. Lazy-loads the ESRGANPipeline on first use.
  /// Weights are resolved from ~/bin/zimage/upscale_models/<modelName>/.
  private func bridgeUpscaleESRGAN(
    imageData: Data,
    modelName: String
  ) async throws -> ComfyBridgeGenerateResult {
    // Resolve weights directory: ~/bin/zimage/upscale_models/<modelName>/
    // Strip file extension if present (e.g. "4x-UltraSharp.pth" -> "4x-UltraSharp")
    let baseName: String
    if let dotIndex = modelName.lastIndex(of: ".") {
      baseName = String(modelName[modelName.startIndex..<dotIndex])
    } else {
      baseName = modelName
    }
    let weightsDir = URL(fileURLWithPath: Self.upscaleModelsDirectoryPath)
      .appendingPathComponent(baseName)

    // Lazy-load ESRGAN pipeline (re-create if model changed)
    if esrganPipeline == nil || esrganPipeline!.weightsDirectory.path != weightsDir.path {
      logger.info("WarmServer: lazy-loading ESRGAN pipeline from \(weightsDir.path)...")
      let pipeline = try ESRGANPipeline(weightsDirectory: weightsDir, logger: logger)
      esrganPipeline = pipeline
      logger.info("WarmServer: ESRGAN pipeline ready (scale=\(pipeline.config.scale)x, blocks=\(pipeline.config.numBlock))")
    }

    guard let pipeline = esrganPipeline else {
      throw ESRGANPipeline.PipelineError.weightsDirectoryNotFound(weightsDir.path)
    }

    // Write input image data to a temp file.
    let inputTempPath = NSTemporaryDirectory() + "zimage-esrgan-input-\(UUID().uuidString).png"
    try imageData.write(to: URL(fileURLWithPath: inputTempPath))
    logger.info("WarmServer: wrote ESRGAN input to \(inputTempPath) (\(imageData.count) bytes)")

    let outputTempPath = NSTemporaryDirectory() + "zimage-esrgan-output-\(UUID().uuidString).png"

    let start = Date()
    do {
      let outputPath = try pipeline.upscaleAndSave(
        imagePath: inputTempPath,
        outputPath: outputTempPath
      )
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      try? FileManager.default.removeItem(atPath: inputTempPath)
      logger.info("WarmServer: ESRGAN upscale complete — \(durationMs)ms, output=\(outputPath)")
      return ComfyBridgeGenerateResult(outputPath: outputPath, durationMs: durationMs)
    } catch {
      try? FileManager.default.removeItem(atPath: inputTempPath)
      try? FileManager.default.removeItem(atPath: outputTempPath)
      throw error
    }
  }

  /// SeedVR2 upscale path. Lazy-loads on first use.
  private func bridgeUpscaleSeedVR2(
    imageData: Data,
    modelName: String,
    progressCallback: ComfyBridgeProgressHandler?
  ) async throws -> ComfyBridgeGenerateResult {
    guard let weightsPath = seedvr2WeightsPath else {
      throw SeedVR2Pipeline.PipelineError.weightsDirectoryNotFound("No SeedVR2 weights path configured")
    }

    // Lazy-load the SeedVR2 pipeline on first upscale request.
    if seedvr2Pipeline == nil {
      logger.info("WarmServer: lazy-loading SeedVR2 pipeline from \(weightsPath)...")
      let pipeline = try SeedVR2Pipeline(weightsPath: weightsPath, logger: logger)
      seedvr2Pipeline = pipeline
      logger.info("WarmServer: SeedVR2 pipeline ready (\(pipeline.modelConfig == .preset7B ? "7B" : "3B"))")
    }

    guard let pipeline = seedvr2Pipeline else {
      throw SeedVR2Pipeline.PipelineError.weightsDirectoryNotFound(weightsPath)
    }

    // Write input image data to a temp file.
    let inputTempPath = NSTemporaryDirectory() + "zimage-upscale-input-\(UUID().uuidString).png"
    try imageData.write(to: URL(fileURLWithPath: inputTempPath))
    logger.info("WarmServer: wrote upscale input to \(inputTempPath) (\(imageData.count) bytes)")

    let outputTempPath = NSTemporaryDirectory() + "zimage-upscale-output-\(UUID().uuidString).png"

    let start = Date()
    do {
      let outputPath = try pipeline.upscaleAndSave(
        imagePath: inputTempPath,
        outputPath: outputTempPath,
        progressHandler: progressCallback
      )
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      try? FileManager.default.removeItem(atPath: inputTempPath)
      logger.info("WarmServer: upscale complete — \(durationMs)ms, output=\(outputPath)")
      return ComfyBridgeGenerateResult(outputPath: outputPath, durationMs: durationMs)
    } catch {
      try? FileManager.default.removeItem(atPath: inputTempPath)
      try? FileManager.default.removeItem(atPath: outputTempPath)
      throw error
    }
  }

  fileprivate func requestShutdownAfterResponse() {
    initiateShutdown()
  }

  /// Initiate a clean shutdown. Cancels the listener and exits.
  /// Safe to call from any thread — idempotent via shutdownSignalled flag.
  private func initiateShutdown(exitCode: Int32 = 0) {
    lifecycleLock.lock()
    defer { lifecycleLock.unlock() }

    guard !shutdownSignalled else { return }
    shutdownSignalled = true

    logger.info("Server shutting down (exit code \(exitCode))...")
    listener?.cancel()

    // Give in-flight connections 1 second to drain, then exit.
    DispatchQueue.global().asyncAfter(deadline: .now() + 1.0) {
      exit(exitCode)
    }
  }

  private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(type, from: data)
  }

  private func response(for error: Error) -> HTTPResponse {
    switch error {
    case let error as WarmServerCoordinator.ServerError:
      switch error {
      case .queueFull(let maxPending):
        return .error(status: 429, message: "Queue full (\(maxPending) pending max)")
      case .shuttingDown:
        return .error(status: 503, message: "Server is shutting down")
      }

    case let error as ZImagePipeline.PipelineError:
      switch error {
      case .invalidDimensions(let message):
        return .error(status: 400, message: message)
      case .loraError(let loraError):
        return .error(status: 400, message: loraError.localizedDescription)
      default:
        return .error(status: 500, message: error.localizedDescription)
      }

    case let error as LoRAError:
      return .error(status: 400, message: error.localizedDescription)

    case let error as WarmServerError:
      switch error {
      case .loraSwapNotSupported, .controlNetNotSupported:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      case .invalidOutputPath:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      case .flux2NotLoaded, .flux2DetectionFailed, .fiboNotLoaded, .fiboDetectionFailed,
           .chromaNotLoaded, .chromaDetectionFailed:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      case .invalidPort:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      }

    case let error as Flux2Pipeline.Flux2PipelineError:
      switch error {
      case .invalidDimensions:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      default:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      }

    case let error as FiboPipeline.FiboPipelineError:
      switch error {
      case .invalidDimensions:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      default:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      }

    case let error as ModelPoolError:
      switch error {
      case .modelNotInPool, .cannotUnloadActive:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      case .budgetExceeded:
        return .error(status: 507, message: error.localizedDescription ?? error.localizedDescription)
      case .alreadyLoaded:
        return .error(status: 409, message: error.localizedDescription ?? error.localizedDescription)
      case .loadFailed, .modelDetectionFailed:
        return .error(status: 500, message: error.localizedDescription ?? error.localizedDescription)
      }

    case let error as DecodingError:
      return .error(status: 400, message: "Invalid JSON body: \(describe(decodingError: error))")

    default:
      return .error(status: 500, message: error.localizedDescription)
    }
  }

  private func describe(decodingError: DecodingError) -> String {
    switch decodingError {
    case .dataCorrupted(let context):
      return context.debugDescription
    case .keyNotFound(let key, let context):
      return "Missing key '\(key.stringValue)' (\(context.debugDescription))"
    case .typeMismatch(_, let context):
      return context.debugDescription
    case .valueNotFound(_, let context):
      return context.debugDescription
    @unknown default:
      return decodingError.localizedDescription
    }
  }

  private static func currentMemoryFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info) / MemoryLayout<natural_t>.size)
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
      }
    }
    guard result == KERN_SUCCESS else { return 0 }
    return info.phys_footprint
  }

  // MARK: - Krita Model Detection Helpers

  /// Parse quantization suffix from a model ID string.
  /// e.g. "z-image-turbo-q8" -> "q8", "klein-4b-q8" -> "q8", "briaai/FIBO" -> nil
  static func parseQuantization(from modelId: String) -> String? {
    let lowered = modelId.lowercased()
    if lowered.hasSuffix("-q4") { return "q4" }
    if lowered.hasSuffix("-q8") { return "q8" }
    if lowered.hasSuffix("-bf16") { return "bf16" }
    return nil
  }

  /// Parse the model spec from a pool-style model ID.
  /// Strips quantization suffixes since poolLoad takes them separately.
  /// e.g. "z-image-turbo-q8" -> "z-image-turbo", "briaai/FIBO" -> "briaai/FIBO"
  static func parseModelSpec(from modelId: String) -> String {
    let knownSpecs = [
      "briaai/FIBO",
      "chroma-8.9b",
      "z-image-turbo",
      "z-image-turbo-bf16",
      "klein-4b",
      "klein-9b",
    ]
    if knownSpecs.contains(modelId) { return modelId }
    let suffixes = ["-q4", "-q8", "-bf16"]
    for suffix in suffixes {
      if modelId.lowercased().hasSuffix(suffix) {
        return String(modelId.dropLast(suffix.count))
      }
    }
    return modelId
  }
}

private actor WarmServerCoordinator {
  enum ServerError: Error {
    case queueFull(maxPending: Int)
    case shuttingDown
  }

  private let configuration: WarmServerConfiguration
  private let logger: Logger
  private var pipeline: ZImagePipeline
  /// Flux 2 pipeline — created when the model is detected as Flux 2 Klein.
  private var flux2Pipeline: Flux2Pipeline?
  /// FIBO pipeline — created when the model is detected as FIBO.
  private var fiboPipeline: FiboPipeline?
  /// Chroma pipeline — created when the model is detected as Chroma.
  private var chromaPipeline: ChromaPipeline?
  /// Chroma tokenizer — loaded alongside the Chroma pipeline.
  private var chromaTokenizer: ChromaTokenizer?
  /// Which model family is loaded — determines generation routing.
  private var currentModelFamily: WarmModelFamily = .flux1
  /// Detected Flux 2 model info (variant, configs) — nil when running Flux 1.
  private var detectedFlux2Model: Flux2DetectedModel?
  /// Detected FIBO model info — nil when running Flux 1/2.
  private var detectedFiboModel: FiboDetectedModel?
  /// Detected Z-Image variant (Base vs Turbo) — only set when running Flux 1 (Z-Image).
  private var zimageVariant: ZImageVariant = .turbo
  /// Lazy-initialized ControlNet pipeline — only created when first ControlNet request arrives.
  private var controlPipeline: ZImageControlPipeline?
  private let startTime = Date()
  private var activeLoRAs: [LoRAConfiguration]
  private var pending: [QueuedOperation] = []
  private var isProcessing = false
  private var shuttingDown = false
  private var successfulRenderCount = 0
  private var failedRenderCount = 0
  private var lastRenderDurationMs: Int?
  private var lastError: String?
  private var activeRenderStartedAt: Date?
  private var pipelinePrepared = false

  /// Model hot-swap pool — holds loaded pipelines with LRU eviction.
  let modelPool: ModelPool

  init(configuration: WarmServerConfiguration, logger: Logger) {
    self.configuration = configuration
    self.logger = logger
    self.pipeline = ZImagePipeline(logger: logger, retentionPolicy: .keepLoaded)
    self.activeLoRAs = configuration.initialLoRAs
    self.modelPool = ModelPool(
      textEncoderPath: configuration.textEncoderPath,
      maxSequenceLength: configuration.maxSequenceLength,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly,
      logger: logger
    )
  }

  func prepare() async throws {
    // Resolve model snapshot path for family detection
    let modelSpec = configuration.modelSpec
    var isFlux2 = false
    var snapshotURL: URL?

    var isFibo = false
    var isChroma = false

    if let spec = modelSpec {
      // Check by known model ID first
      if ChromaModelDetection.isKnownChromaModel(spec) {
        isChroma = true
      } else if FiboModelDetection.isKnownFiboModel(spec) {
        isFibo = true
      } else if Flux2ModelDetection.isKnownFlux2Model(spec) {
        isFlux2 = true
      }

      // Resolve snapshot — needed for both detection and loading
      let resolved = try await ModelResolution.resolveOrDefault(
        modelSpec: spec,
        filePatterns: ["*.safetensors", "*.json", "tokenizer/*"]
      )
      snapshotURL = resolved

      // If not already detected by name, check the snapshot directory
      if !isFibo && !isFlux2 && !isChroma {
        if ChromaModelDetection.detect(at: resolved) != nil {
          isChroma = true
        } else if FiboModelDetection.detect(at: resolved) != nil {
          isFibo = true
        } else if Flux2ModelDetection.detectFamily(at: resolved) == .flux2 {
          isFlux2 = true
        }
      }
    }

    if isChroma, let snapshot = snapshotURL {
      // --- Chroma path ---
      currentModelFamily = .chroma

      guard let detected = ChromaModelDetection.detect(at: snapshot) else {
        throw WarmServerError.chromaDetectionFailed(modelSpec ?? "unknown")
      }

      logger.info("Detected Chroma model — estimated GPU memory: ~17GB")

      let components = try ChromaInitializer.load(
        from: snapshot,
        paths: detected.componentPaths,
        config: detected.config,
        dtype: .bfloat16,
        logger: logger
      )

      // Load tokenizer
      let tokenizer = try ChromaTokenizer.load(from: detected.componentPaths.tokenizerPath)

      chromaPipeline = ChromaPipeline(
        transformer: components.transformer,
        t5: components.t5,
        vae: components.vae,
        config: detected.config
      )
      chromaTokenizer = tokenizer
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (Chroma)")
    } else if isFibo, let snapshot = snapshotURL {
      // --- FIBO path ---
      currentModelFamily = .fibo

      guard let detected = FiboModelDetection.detect(at: snapshot) else {
        throw WarmServerError.fiboDetectionFailed(modelSpec ?? "unknown")
      }
      detectedFiboModel = detected
      logger.info("Detected FIBO model — estimated GPU memory: ~16GB")

      let fp = FiboPipeline(logger: logger)
      try fp.loadModel(
        from: snapshot,
        transformerConfig: detected.transformerConfig,
        vaeConfig: detected.vaeConfig,
        textEncoderConfig: detected.textEncoderConfig
      )
      fiboPipeline = fp
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (FIBO)")
    } else if isFlux2, let snapshot = snapshotURL {
      // --- Flux 2 Klein path ---
      currentModelFamily = .flux2

      guard let detected = Flux2ModelDetection.detect(at: snapshot) else {
        throw WarmServerError.flux2DetectionFailed(modelSpec ?? "unknown")
      }
      detectedFlux2Model = detected

      // Log memory estimate
      let estimatedGB: String
      switch detected.variant {
      case "klein-4b", "klein-base-4b": estimatedGB = "~15GB"
      case "klein-9b", "klein-base-9b": estimatedGB = "~25GB"
      default: estimatedGB = "unknown"
      }
      let modelType = detected.isBaseModel ? "base (non-distilled)" : "distilled"
      logger.info("Detected Flux 2 Klein \(detected.variant) [\(modelType)] — estimated GPU memory: \(estimatedGB)")

      let f2 = Flux2Pipeline(logger: logger)
      try f2.loadModel(
        from: snapshot,
        config: detected.transformerConfig,
        textEncoderConfig: detected.textEncoderConfig,
        isBase: detected.isBaseModel
      )
      flux2Pipeline = f2
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (Flux 2 Klein \(detected.variant))")
    } else {
      // --- Flux 1 / Z-Image path ---
      currentModelFamily = .flux1

      // Detect Z-Image variant (Base vs Turbo)
      if let spec = modelSpec, let variant = ZImageVariant.fromModelSpec(spec) {
        zimageVariant = variant
      } else if let spec = modelSpec, spec.hasSuffix(".safetensors") {
        // Detect from CivitAI checkpoint inspection
        let localURL = URL(fileURLWithPath: spec)
        if FileManager.default.fileExists(atPath: localURL.path) {
          let inspection = CivitAICheckpoint.inspect(fileURL: localURL)
          if let variant = inspection.variant {
            zimageVariant = variant
          }
        }
      } else if let resolvedSnapshot = snapshotURL {
        zimageVariant = ZImageVariant.fromSnapshot(at: resolvedSnapshot)
      } else if let spec = modelSpec {
        // Resolve and detect from snapshot if not already resolved
        if let resolved = try? await ModelResolution.resolveOrDefault(
          modelSpec: spec,
          filePatterns: ["*.safetensors", "*.json", "tokenizer/*"]
        ) {
          zimageVariant = ZImageVariant.fromSnapshot(at: resolved)
        }
      }
      let variantLabel = zimageVariant == .base ? "Base (non-distilled)" : "Turbo (distilled)"
      logger.info("Preloading warm server pipeline (Flux 1 / Z-Image \(variantLabel))")
      try await pipeline.prepare(
        modelSpec: modelSpec,
        textEncoderPath: configuration.textEncoderPath,
        loras: activeLoRAs,
        forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly
      )
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (Flux 1 / Z-Image \(zimageVariant.rawValue))")

      // Pre-load the full VAE encoder for img2img support.
      // Without this, the first img2img request triggers synchronous weight
      // loading inside the actor-isolated render path, which can deadlock
      // the cooperative thread pool (issue #141).
      do {
        try pipeline.prepareFullVAE()
        logger.info("Full VAE encoder pre-loaded for img2img")
      } catch {
        logger.warning("Failed to pre-load full VAE encoder: \(error). First img2img request will attempt lazy load.")
      }
    }

    // Register the initial model in the pool so it appears in pool listings
    // and can be managed alongside hot-swapped models.
    // We register the already-loaded pipeline to avoid double-loading.
    if let spec = modelSpec {
      let box: PipelineBox
      let detectedInfo: Any?
      let vramMB: Int
      switch currentModelFamily {
      case .chroma:
        box = PipelineBox(pipeline: chromaPipeline! as AnyObject)
        if let tok = chromaTokenizer { box.context["tokenizer"] = tok as AnyObject }
        detectedInfo = nil
        vramMB = 17408
      case .fibo:
        box = PipelineBox(pipeline: fiboPipeline! as AnyObject)
        detectedInfo = detectedFiboModel
        vramMB = 22528
      case .flux2:
        box = PipelineBox(pipeline: flux2Pipeline! as AnyObject)
        detectedInfo = detectedFlux2Model
        vramMB = (detectedFlux2Model?.variant.contains("9b") ?? false) ? 18432 : 8704
      case .flux1:
        box = PipelineBox(pipeline: pipeline as AnyObject)
        detectedInfo = zimageVariant
        vramMB = 12288
      }
      let poolKey = ModelPool.poolKey(for: spec)
      await modelPool.registerExisting(
        poolKey: poolKey,
        modelSpec: spec,
        family: currentModelFamily,
        box: box,
        vramEstimateMB: vramMB,
        detectedInfo: detectedInfo
      )
      logger.info("ModelPool: initial model '\(poolKey)' registered and activated")
    }
  }

  /// Expose the current model family for routing decisions outside the actor.
  var modelFamily: WarmModelFamily {
    currentModelFamily
  }

  /// Active LoRA identifiers (bare filenames without path or extension) for the library API.
  var activeLoRAIdentifiers: [String] {
    activeLoRAs.map { config in
      switch config.source {
      case .local(let url):
        return (url.lastPathComponent as NSString).deletingPathExtension
      case .huggingFace(let modelId, let filename):
        if let filename {
          return (filename as NSString).deletingPathExtension
        }
        return modelId.components(separatedBy: "/").last ?? modelId
      }
    }
  }

  /// Whether the loaded Flux 2 model is a base (non-distilled) variant.
  var isFlux2BaseModel: Bool {
    detectedFlux2Model?.isBaseModel ?? false
  }

  /// The detected Z-Image variant (Base vs Turbo) for Flux 1 models.
  var currentZImageVariant: ZImageVariant {
    zimageVariant
  }

  // MARK: - Model Pool Operations

  /// Load a model into the pool, optionally activating it.
  func poolLoad(modelSpec: String, quantization: String?, activate: Bool) async throws -> ModelLoadResponse {
    let start = Date()
    let entry = try await modelPool.load(
      modelSpec: modelSpec,
      quantization: quantization,
      initialLoRAs: activeLoRAs
    )
    let loadTimeMs = Int(Date().timeIntervalSince(start) * 1000.0)

    if activate {
      try await poolActivate(modelId: entry.id)
    }

    return ModelLoadResponse(
      status: "loaded",
      model: entry.modelSpec,
      family: entry.family.rawValue,
      loadTimeMs: loadTimeMs,
      vramEstimateMB: entry.vramEstimateMB,
      poolSize: await modelPool.count(),
      poolBudgetMB: await modelPool.budget()
    )
  }

  /// Activate a model that is already in the pool.
  @discardableResult
  func poolActivate(modelId: String) async throws -> ModelActivateResponse {
    // Try by pool key first, then by model spec.
    let entry: PoolEntry
    if let e = await modelPool.findEntry(for: modelId) {
      entry = try await modelPool.activate(modelId: e.id)
    } else {
      throw ModelPoolError.modelNotInPool(modelId)
    }

    // Sync coordinator state from pool entry.
    currentModelFamily = entry.family
    switch entry.family {
    case .chroma:
      chromaPipeline = entry.box.pipeline as? ChromaPipeline
      chromaTokenizer = entry.box.context["tokenizer"] as? ChromaTokenizer
    case .fibo:
      fiboPipeline = entry.box.pipeline as? FiboPipeline
      detectedFiboModel = entry.detectedInfo as? FiboDetectedModel
    case .flux2:
      flux2Pipeline = entry.box.pipeline as? Flux2Pipeline
      detectedFlux2Model = entry.detectedInfo as? Flux2DetectedModel
    case .flux1:
      // Reassign the pipeline so that runSwap and runFlux1Generate
      // operate on the pool-loaded pipeline, not the original one (#138).
      if let poolZImage = entry.box.pipeline as? ZImagePipeline {
        pipeline = poolZImage
      }
      zimageVariant = (entry.detectedInfo as? ZImageVariant) ?? .turbo
    }
    pipelinePrepared = true

    return ModelActivateResponse(
      status: "activated",
      model: entry.modelSpec,
      family: entry.family.rawValue
    )
  }

  /// Unload a model from the pool.
  func poolUnload(modelId: String) async throws -> ModelUnloadResponse {
    // Find the entry to get the model spec before unloading.
    guard let entry = await modelPool.findEntry(for: modelId) else {
      throw ModelPoolError.modelNotInPool(modelId)
    }
    let freedMB = try await modelPool.unload(modelId: entry.id)
    return ModelUnloadResponse(
      status: "unloaded",
      model: entry.modelSpec,
      freedMB: freedMB,
      poolSize: await modelPool.count()
    )
  }

  /// List all models in the pool.
  func poolList() async -> ModelPoolListResponse {
    let entries = await modelPool.listPool()
    let activeId = await modelPool.activeModelId()
    let activeSpec: String?
    if let aid = activeId, let entry = await modelPool.findEntry(for: aid) {
      activeSpec = entry.modelSpec
    } else {
      activeSpec = nil
    }
    return ModelPoolListResponse(
      active: activeSpec,
      pool: entries,
      totalVramMB: await modelPool.totalVramMB(),
      budgetMB: await modelPool.budget()
    )
  }

  func enqueueGenerate(
    _ payload: GeneratePayload,
    progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil
  ) async throws -> GenerateResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(.generate(payload, ContinuationBox(continuation), progressHandler))
      startProcessingIfNeeded()
    }
  }

  func enqueueSwap(_ payload: LoRASwapPayload) async throws -> LoRASwapResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(.swap(payload, ContinuationBox(continuation)))
      startProcessingIfNeeded()
    }
  }

  func enqueueControlGenerate(_ request: ZImageControlGenerationRequest) async throws -> GenerateResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(.controlGenerate(request, ContinuationBox(continuation)))
      startProcessingIfNeeded()
    }
  }

  func enqueueShutdown() async throws -> ShutdownResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }

    shuttingDown = true
    return try await withCheckedThrowingContinuation { continuation in
      pending.append(.shutdown(ContinuationBox(continuation)))
      startProcessingIfNeeded()
    }
  }

  /// Maximum render age before the health endpoint reports the render as stale.
  /// After this threshold, the health status changes to "render_stale" to signal
  /// that the render is likely deadlocked (issue #141).
  private static let renderStaleThresholdMs = 300_000 // 5 minutes

  func health(memoryBytes: UInt64) -> HealthResponse {
    let uptimeSeconds = Int(Date().timeIntervalSince(startTime))
    let activeAgeMs = activeRenderStartedAt.map { Int(Date().timeIntervalSince($0) * 1000.0) }

    return HealthResponse(
      status: shuttingDown ? "shutting_down" : (activeAgeMs.map { $0 > Self.renderStaleThresholdMs } ?? false ? "render_stale" : "ok"),
      model: configuration.modelSpec ?? ZImageRepository.id,
      modelFamily: currentModelFamily.rawValue,
      modelVariant: currentModelFamily == .fibo ? "fibo" : (currentModelFamily == .flux1 ? zimageVariant.rawValue : detectedFlux2Model?.variant),
      textEncoderPath: configuration.textEncoderPath,
      loaded: pipelinePrepared,
      loras: activeLoRAs.map(LoRAState.init),
      uptimeSeconds: uptimeSeconds,
      renderCount: successfulRenderCount,
      failedRenderCount: failedRenderCount,
      pendingCount: pending.count,
      maxPending: configuration.maxPendingRequests,
      isRendering: activeRenderStartedAt != nil,
      activeRequestAgeMs: activeAgeMs,
      memoryUsageBytes: memoryBytes,
      memoryUsageMB: memoryBytes / (1024 * 1024),
      lastRenderDurationMs: lastRenderDurationMs,
      lastError: lastError
    )
  }

  /// Queue status for the ComfyUI bridge /queue endpoint.
  func queueStatus() -> ComfyBridgeQueueStatus {
    return ComfyBridgeQueueStatus(
      pendingCount: pending.count,
      maxPending: configuration.maxPendingRequests,
      isRendering: activeRenderStartedAt != nil,
      currentJobId: nil,
      progressPercent: nil,
      renderCount: successfulRenderCount,
      failedCount: failedRenderCount
    )
  }

  /// Clear all pending jobs from the queue. Active job continues.
  func clearPending() -> Int {
    let count = pending.count
    // Cancel all pending continuations with an error.
    for op in pending {
      switch op {
      case .generate(_, let cont, _):
        cont.resume(throwing: ServerError.shuttingDown)
      case .controlGenerate(_, let cont):
        cont.resume(throwing: ServerError.shuttingDown)
      case .swap(_, let cont):
        cont.resume(throwing: ServerError.shuttingDown)
      case .shutdown(let cont):
        cont.resume(throwing: ServerError.shuttingDown)
      }
    }
    pending.removeAll()
    return count
  }

  private func startProcessingIfNeeded() {
    guard !isProcessing else { return }
    isProcessing = true
    Task {
      await processLoop()
    }
  }

  private func processLoop() async {
    while true {
      guard !pending.isEmpty else {
        isProcessing = false
        return
      }

      let operation = pending.removeFirst()
      switch operation {
      case .generate(let payload, let continuation, let progressHandler):
        await runGenerate(payload, continuation: continuation, progressHandler: progressHandler)
      case .controlGenerate(let request, let continuation):
        await runControlGenerate(request, continuation: continuation)
      case .swap(let payload, let continuation):
        await runSwap(payload, continuation: continuation)
      case .shutdown(let continuation):
        continuation.resume(
          returning: ShutdownResponse(
            success: true,
            message: "Server shutdown requested"
          )
        )
      }
    }
  }

  private func runGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>, progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil) async {
    switch currentModelFamily {
    case .chroma:
      await runChromaGenerate(payload, continuation: continuation)
    case .fibo:
      await runFiboGenerate(payload, continuation: continuation)
    case .flux2:
      await runFlux2Generate(payload, continuation: continuation)
    case .flux1:
      await runFlux1Generate(payload, continuation: continuation, progressHandler: progressHandler)
    }
  }

  private func runFlux1Generate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>, progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil) async {
    activeRenderStartedAt = Date()
    let start = Date()

    do {
      let outputURL: URL
      if payload.imagePath != nil {
        let img2imgRequest = try payload.makeImg2ImgRequest(
          configuration: configuration,
          activeLoRAs: activeLoRAs
        )
        outputURL = try await pipeline.generateImg2Img(img2imgRequest, progressHandler: progressHandler)
      } else {
        let request = try payload.makePipelineRequest(
          configuration: configuration,
          activeLoRAs: activeLoRAs
        )
        outputURL = try await pipeline.generateFromRequest(request, progressHandler: progressHandler)
      }
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: outputURL.path,
          durationMs: durationMs
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      continuation.resume(throwing: error)
    }
  }

  private func runFlux2Generate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    let start = Date()

    do {
      guard let f2 = flux2Pipeline else {
        throw WarmServerError.flux2NotLoaded
      }

      let outputURL: URL
      outputURL = try payload.resolvedOutputURL(
        configuration: configuration,
        defaultFilename: "zimage-flux2-\(UUID().uuidString).png"
      )

      // Map GeneratePayload fields to Flux2GenerationRequest.
      // Base models: 50 steps, guidance configurable.
      // Distilled models: 4 steps, guidance 1.0.
      let defaultSteps = f2.defaultSteps
      let defaultGuidance: Float = f2.isDistilled ? 1.0 : 3.5
      // Resolve img2img parameters from payload.
      // imagePath takes priority; denoise defaults to 1.0 (txt2img).
      // imageStrength maps to denoise as (1.0 - strength), creativity maps directly.
      let inputImageURL: URL? = payload.imagePath.map { URL(fileURLWithPath: $0) }
      let resolvedDenoise: Float
      if inputImageURL != nil {
        if let creativity = payload.creativity {
          resolvedDenoise = max(0.01, min(1.0, creativity))
        } else if let strength = payload.imageStrength {
          resolvedDenoise = max(0.01, min(1.0, 1.0 - strength))
        } else if let d = payload.denoise {
          resolvedDenoise = max(0.01, min(1.0, d))
        } else {
          resolvedDenoise = 0.7  // sensible default for img2img
        }
      } else {
        resolvedDenoise = 1.0
      }

      let flux2Request = Flux2GenerationRequest(
        prompt: payload.prompt,
        negativePrompt: payload.negativePrompt,
        width: payload.width ?? 1024,
        height: payload.height ?? 1024,
        steps: payload.steps ?? defaultSteps,
        guidanceScale: payload.guidance ?? defaultGuidance,
        seed: payload.seed,
        outputPath: outputURL,
        levelsMin: payload.levelsMin ?? 0.0,
        levelsMax: payload.levelsMax ?? 1.0,
        maxSequenceLength: configuration.maxSequenceLength,
        inputImagePath: inputImageURL,
        denoise: resolvedDenoise
      )

      let result = try await f2.generate(flux2Request, progressHandler: { progress in
        // Flux2Pipeline progress — not routed to ZImagePipeline progress handler
        // since the types differ. Logged internally by the pipeline.
      })

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: result.path,
          durationMs: durationMs
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      continuation.resume(throwing: error)
    }
  }

  private func runFiboGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    let start = Date()

    do {
      guard let fp = fiboPipeline else {
        throw WarmServerError.fiboNotLoaded
      }

      let outputURL: URL
      outputURL = try payload.resolvedOutputURL(
        configuration: configuration,
        defaultFilename: "zimage-fibo-\(UUID().uuidString).png"
      )

      let fiboRequest = FiboGenerationRequest(
        prompt: payload.prompt,
        negativePrompt: payload.negativePrompt,
        width: payload.width ?? 1024,
        height: payload.height ?? 1024,
        steps: payload.steps ?? 30,
        guidanceScale: payload.guidance ?? 4.0,
        seed: payload.seed,
        outputPath: outputURL,
        levelsMin: payload.levelsMin ?? 0.0,
        levelsMax: payload.levelsMax ?? 1.0
      )

      let result = try await fp.generate(fiboRequest, progressHandler: nil)

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: result.path,
          durationMs: durationMs
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      continuation.resume(throwing: error)
    }
  }

  private func runChromaGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    let start = Date()

    do {
      guard let pipeline = chromaPipeline else {
        throw WarmServerError.chromaNotLoaded
      }
      guard let tokenizer = chromaTokenizer else {
        throw WarmServerError.chromaNotLoaded
      }

      let outputURL: URL
      if let outputPath = payload.outputPath, !outputPath.isEmpty {
        outputURL = URL(fileURLWithPath: outputPath)
      } else {
        outputURL = URL(fileURLWithPath: NSTemporaryDirectory())
          .appendingPathComponent("zimage-chroma-\(UUID().uuidString).png")
      }

      let width = payload.width ?? 1024
      let height = payload.height ?? 1024
      let steps = payload.steps ?? 28
      let guidance = payload.guidance ?? 0.0
      let seed = payload.seed ?? UInt64.random(in: 0...UInt64.max)

      // Tokenize prompt (unpadded — matches Python behavior)
      let tokenIds = tokenizer.encodeUnpadded(prompt: payload.prompt)

      // Tokenize negative prompt for CFG (empty string = unconditional)
      let negTokenIds = tokenizer.encodeUnpadded(prompt: payload.negativePrompt ?? "")

      // CFG parameters (default: cfg=4.0, no warmup steps)
      let cfgScale = payload.cfg ?? 4.0
      let cfgWarmup = payload.firstNStepsWithoutCFG ?? 0

      // Generate — returns MLXArray in [B, H, W, C] (NHWC, values [0,1])
      let result = pipeline.generate(
        tokenIds: tokenIds,
        negativeTokenIds: negTokenIds,
        width: width,
        height: height,
        numSteps: steps,
        guidance: guidance,
        cfg: cfgScale,
        firstNStepsWithoutCFG: cfgWarmup,
        seed: seed,
        progressCallback: { step, total in
          // Progress logging
        }
      )

      // Transpose from NHWC [1, H, W, 3] to CHW [3, H, W] for QwenImageIO
      let imageArray = result.squeezed(axis: 0).transposed(2, 0, 1)

      // Save image
      try QwenImageIO.saveImage(array: imageArray, to: outputURL)

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: outputURL.path,
          durationMs: durationMs
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      continuation.resume(throwing: error)
    }
  }

  private func runControlGenerate(_ request: ZImageControlGenerationRequest, continuation: ContinuationBox<GenerateResponse>) async {
    if currentModelFamily == .flux2 || currentModelFamily == .fibo || currentModelFamily == .chroma {
      continuation.resume(throwing: WarmServerError.controlNetNotSupported)
      return
    }

    activeRenderStartedAt = Date()
    let start = Date()

    do {
      // Lazy-init the control pipeline on first ControlNet request
      if controlPipeline == nil {
        logger.info("Initializing ControlNet pipeline (first use)...")
        controlPipeline = ZImageControlPipeline(logger: logger)
      }

      let outputURL = try await controlPipeline!.generate(request)
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      continuation.resume(
        returning: GenerateResponse(
          success: true,
          outputPath: outputURL.path,
          durationMs: durationMs
        )
      )
    } catch {
      failedRenderCount += 1
      lastError = error.localizedDescription
      activeRenderStartedAt = nil
      continuation.resume(throwing: error)
    }
  }

  private func runSwap(_ payload: LoRASwapPayload, continuation: ContinuationBox<LoRASwapResponse>) async {
    if currentModelFamily == .fibo || currentModelFamily == .chroma {
      continuation.resume(throwing: WarmServerError.loraSwapNotSupported)
      return
    }

    do {
      let newLoRAs = try payload.makeConfigurations()

      if currentModelFamily == .flux2 {
        // Flux 2 LoRA swap via Flux2Pipeline.loadLoRAs()
        guard let f2 = flux2Pipeline else {
          continuation.resume(throwing: WarmServerError.flux2NotLoaded)
          return
        }
        try await f2.loadLoRAs(newLoRAs)
        activeLoRAs = newLoRAs
      } else {
        // Flux 1 LoRA swap via ZImagePipeline.swapLoRAs()
        try await pipeline.swapLoRAs(newLoRAs)
        activeLoRAs = newLoRAs
      }

      lastError = nil
      continuation.resume(
        returning: LoRASwapResponse(
          success: true,
          loraCount: activeLoRAs.count,
          loras: activeLoRAs.map(LoRAState.init)
        )
      )
    } catch {
      if currentModelFamily == .flux2 {
        activeLoRAs = flux2Pipeline?.loadedLoRAConfigs ?? []
      } else {
        activeLoRAs = pipeline.loadedLoRAConfigs
      }
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
    }
  }
}

private final class ConnectionHandler {
  private static let headerDelimiter = Data("\r\n\r\n".utf8)
  /// 10 MB — raised from 1 MB to support ComfyUI image uploads via PUT /api/etn/image/.
  private static let maximumRequestBytes = 10_485_760

  private let connection: NWConnection
  private let queue: DispatchQueue
  private weak var server: WarmServer?
  private var buffer = Data()
  private var responseSent = false
  private var retainSelf: ConnectionHandler?

  init(connection: NWConnection, queue: DispatchQueue, server: WarmServer) {
    self.connection = connection
    self.queue = queue
    self.server = server
  }

  func start() {
    retainSelf = self
    connection.start(queue: queue)
    receiveNextChunk()
  }

  private func receiveNextChunk() {
    connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, isComplete, error in
      guard let self else { return }

      if let data, !data.isEmpty {
        self.buffer.append(data)
      }

      if self.buffer.count > Self.maximumRequestBytes {
        self.finish(with: .error(status: 413, message: "Request too large"))
        return
      }

      switch self.parseRequest() {
      case .request(let request):
        self.handle(request: request)
        return
      case .error(let response):
        self.finish(with: response)
        return
      case .incomplete:
        break
      }

      if let error {
        self.finish(with: .error(status: 400, message: error.localizedDescription))
        return
      }

      if isComplete {
        self.finish(with: .error(status: 400, message: "Unexpected end of request"))
        return
      }

      self.receiveNextChunk()
    }
  }

  private func parseRequest() -> HTTPParseResult {
    guard let headerRange = buffer.range(of: Self.headerDelimiter) else {
      return .incomplete
    }

    let headerData = buffer.subdata(in: 0..<headerRange.lowerBound)
    guard let headerString = String(data: headerData, encoding: .utf8) else {
      return .error(.error(status: 400, message: "Invalid request headers"))
    }

    let lines = headerString.components(separatedBy: "\r\n")
    guard let requestLine = lines.first, !requestLine.isEmpty else {
      return .error(.error(status: 400, message: "Missing request line"))
    }

    let requestParts = requestLine.split(separator: " ", omittingEmptySubsequences: true)
    guard requestParts.count >= 2 else {
      return .error(.error(status: 400, message: "Malformed request line"))
    }

    var headers: [String: String] = [:]
    for line in lines.dropFirst() where !line.isEmpty {
      guard let separator = line.firstIndex(of: ":") else {
        return .error(.error(status: 400, message: "Malformed header"))
      }
      let name = line[..<separator].trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
      let value = line[line.index(after: separator)...].trimmingCharacters(in: .whitespacesAndNewlines)
      headers[name] = value
    }

    let contentLength = Int(headers["content-length"] ?? "0") ?? 0
    if contentLength < 0 || contentLength > Self.maximumRequestBytes {
      return .error(.error(status: 413, message: "Request body too large"))
    }

    let bodyStart = headerRange.upperBound
    let totalLength = bodyStart + contentLength
    guard buffer.count >= totalLength else {
      return .incomplete
    }

    let body = buffer.subdata(in: bodyStart..<totalLength)
    let rawPath = String(requestParts[1])
    let pathAndQuery = rawPath.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
    let path = pathAndQuery.first.map(String.init) ?? rawPath
    let queryString: String? = pathAndQuery.count > 1 ? String(pathAndQuery[1]) : nil

    return .request(
      HTTPRequest(
        method: String(requestParts[0]).uppercased(),
        path: path,
        queryString: queryString,
        headers: headers,
        body: body
      )
    )
  }

  private func handle(request: HTTPRequest) {
    guard let server else {
      finish(with: .error(status: 500, message: "Server unavailable"))
      return
    }

    // Check for WebSocket upgrade before entering the async router.
    if request.path == "/ws", request.method == "GET" {
      if let wsResponse = server.comfyBridge.handleWebSocketUpgrade(request: request, connection: connection, queue: queue) {
        // Send the upgrade response, then keep the connection alive for WebSocket framing.
        guard !responseSent else { return }
        responseSent = true
        connection.send(content: wsResponse, completion: .contentProcessed { [weak self] _ in
          guard let self, let server = self.server else { return }
          let clientId = request.queryParameters["clientId"] ?? UUID().uuidString
          server.comfyBridge.wsManager.registerConnection(
            clientId: clientId,
            connection: self.connection,
            queue: self.queue
          )
          // Release the ConnectionHandler — the WS manager now owns the NWConnection.
          // Do NOT cancel the connection; only release our retain cycle.
          self.retainSelf = nil
        })
      } else {
        // Invalid WebSocket upgrade request — send 400 and close.
        finish(with: .error(status: 400, message: "Invalid WebSocket upgrade request"))
      }
      return
    }

    Task {
      let routed = await server.respond(to: request)
      switch routed {
      case .error(let response):
        self.finish(with: response)
      case .json(let response):
        self.finish(with: response)
      case .shutdown(let response):
        self.finish(with: response, shutdownAfterSend: true)
      case .websocketUpgrade:
        // Should not reach here — /ws is handled above before async dispatch.
        self.finish(with: .error(status: 400, message: "Invalid WebSocket upgrade request"))
      }
    }
  }

  private func finish(with response: HTTPResponse, shutdownAfterSend: Bool = false) {
    guard !responseSent else { return }
    responseSent = true

    connection.send(content: response.serialize(), completion: .contentProcessed { [weak self] _ in
      guard let self else { return }
      self.connection.cancel()
      if shutdownAfterSend {
        self.server?.requestShutdownAfterResponse()
      }
      self.retainSelf = nil
    })
  }
}

struct HTTPRequest {
  let method: String
  let path: String
  let queryString: String?
  let headers: [String: String]
  let body: Data

  /// Parse query parameters from the query string.
  var queryParameters: [String: String] {
    guard let qs = queryString, !qs.isEmpty else { return [:] }
    var params: [String: String] = [:]
    for pair in qs.split(separator: "&") {
      let parts = pair.split(separator: "=", maxSplits: 1)
      if parts.count == 2 {
        params[String(parts[0])] = String(parts[1])
      } else if parts.count == 1 {
        params[String(parts[0])] = ""
      }
    }
    return params
  }
}

enum HTTPParseResult {
  case incomplete
  case request(HTTPRequest)
  case error(HTTPResponse)
}

struct HTTPResponse {
  let status: Int
  let reasonPhrase: String
  let contentType: String
  let body: Data

  static func json<T: Encodable>(status: Int, payload: T) -> HTTPResponse {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let body = (try? encoder.encode(payload)) ?? Data("{\"success\":false,\"error\":\"encoding failure\"}".utf8)
    return HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), contentType: "application/json", body: body)
  }

  /// Create a JSON response from pre-encoded Data (no snake_case conversion).
  static func rawJSON(status: Int, data: Data) -> HTTPResponse {
    HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), contentType: "application/json", body: data)
  }

  /// Create a binary response with a specified content type.
  static func binary(status: Int, contentType: String, data: Data) -> HTTPResponse {
    HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), contentType: contentType, body: data)
  }

  static func error(status: Int, message: String) -> HTTPResponse {
    json(status: status, payload: ErrorPayload(success: false, error: message))
  }

  func serialize() -> Data {
    var data = Data()
    let header = [
      "HTTP/1.1 \(status) \(reasonPhrase)",
      "Content-Type: \(contentType)",
      "Content-Length: \(body.count)",
      "Connection: close",
      "",
      ""
    ].joined(separator: "\r\n")
    data.append(Data(header.utf8))
    data.append(body)
    return data
  }

  static func reasonPhrase(for status: Int) -> String {
    switch status {
    case 200: return "OK"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 413: return "Payload Too Large"
    case 429: return "Too Many Requests"
    case 500: return "Internal Server Error"
    case 503: return "Service Unavailable"
    default: return "OK"
    }
  }
}

enum RoutedResponse {
  case json(HTTPResponse)
  case shutdown(HTTPResponse)
  case error(HTTPResponse)
  /// WebSocket upgrade — the bridge takes ownership of the connection.
  case websocketUpgrade

  static func json<T: Encodable>(status: Int, payload: T) -> RoutedResponse {
    .json(.json(status: status, payload: payload))
  }

  static func shutdown<T: Encodable>(status: Int, payload: T) -> RoutedResponse {
    .shutdown(.json(status: status, payload: payload))
  }
}

private struct GeneratePayload: Sendable {
  let prompt: String
  let negativePrompt: String?
  let width: Int?
  let height: Int?
  let steps: Int?
  let guidance: Float?
  let seed: UInt64?
  let outputPath: String?
  let levelsMin: Float?
  let levelsMax: Float?
  let scheduler: String?
  let sigmaSchedule: String?
  let eta: Float?
  let dype: String?
  // Phase 3: Inpainting data (set by bridge, not by HTTP API)
  let inpaintImageData: Data?
  let maskData: Data?
  let denoise: Float?
  let maskGrow: Int?
  let maskFeather: Int?
  let maskCropX: Int?
  let maskCropY: Int?

  // Chroma CFG parameters
  let cfg: Float?
  let firstNStepsWithoutCFG: Int?

  // Phase 4: Img2img (set via HTTP API)
  let imagePath: String?
  let imageStrength: Float?
  let creativity: Float?

  /// Default memberwise init for bridge-created payloads.
  init(
    prompt: String, negativePrompt: String? = nil,
    width: Int? = nil, height: Int? = nil, steps: Int? = nil,
    guidance: Float? = nil, seed: UInt64? = nil, outputPath: String? = nil,
    levelsMin: Float? = nil, levelsMax: Float? = nil,
    scheduler: String? = nil, sigmaSchedule: String? = nil, eta: Float? = nil,
    dype: String? = nil, inpaintImageData: Data? = nil, maskData: Data? = nil,
    denoise: Float? = nil, maskGrow: Int? = nil, maskFeather: Int? = nil,
    maskCropX: Int? = nil, maskCropY: Int? = nil,
    cfg: Float? = nil, firstNStepsWithoutCFG: Int? = nil,
    imagePath: String? = nil, imageStrength: Float? = nil, creativity: Float? = nil
  ) {
    self.prompt = prompt; self.negativePrompt = negativePrompt
    self.width = width; self.height = height; self.steps = steps
    self.guidance = guidance; self.seed = seed; self.outputPath = outputPath
    self.levelsMin = levelsMin; self.levelsMax = levelsMax
    self.scheduler = scheduler; self.sigmaSchedule = sigmaSchedule
    self.eta = eta; self.dype = dype
    self.inpaintImageData = inpaintImageData; self.maskData = maskData
    self.denoise = denoise; self.maskGrow = maskGrow; self.maskFeather = maskFeather
    self.maskCropX = maskCropX; self.maskCropY = maskCropY
    self.cfg = cfg; self.firstNStepsWithoutCFG = firstNStepsWithoutCFG
    self.imagePath = imagePath; self.imageStrength = imageStrength; self.creativity = creativity
  }
}

extension GeneratePayload: Decodable {
  private enum CodingKeys: String, CodingKey {
    case prompt, negativePrompt, width, height, steps, guidance, seed
    case outputPath, levelsMin, levelsMax, scheduler, sigmaSchedule, eta, dype
    case cfg, firstNStepsWithoutCFG
    case imagePath, imageStrength, creativity
  }

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    prompt = try c.decode(String.self, forKey: .prompt)
    negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
    width = try c.decodeIfPresent(Int.self, forKey: .width)
    height = try c.decodeIfPresent(Int.self, forKey: .height)
    steps = try c.decodeIfPresent(Int.self, forKey: .steps)
    guidance = try c.decodeIfPresent(Float.self, forKey: .guidance)
    seed = try c.decodeIfPresent(UInt64.self, forKey: .seed)
    outputPath = try c.decodeIfPresent(String.self, forKey: .outputPath)
    levelsMin = try c.decodeIfPresent(Float.self, forKey: .levelsMin)
    levelsMax = try c.decodeIfPresent(Float.self, forKey: .levelsMax)
    scheduler = try c.decodeIfPresent(String.self, forKey: .scheduler)
    sigmaSchedule = try c.decodeIfPresent(String.self, forKey: .sigmaSchedule)
    eta = try c.decodeIfPresent(Float.self, forKey: .eta)
    dype = try c.decodeIfPresent(String.self, forKey: .dype)
    inpaintImageData = nil
    maskData = nil
    denoise = nil
    maskGrow = nil
    maskFeather = nil
    maskCropX = nil
    maskCropY = nil
    cfg = try c.decodeIfPresent(Float.self, forKey: .cfg)
    firstNStepsWithoutCFG = try c.decodeIfPresent(Int.self, forKey: .firstNStepsWithoutCFG)
    imagePath = try c.decodeIfPresent(String.self, forKey: .imagePath)
    imageStrength = try c.decodeIfPresent(Float.self, forKey: .imageStrength)
    creativity = try c.decodeIfPresent(Float.self, forKey: .creativity)
  }

  func makePipelineRequest(
    configuration: WarmServerConfiguration,
    activeLoRAs: [LoRAConfiguration]
  ) throws -> ZImageGenerationRequest {
    let outputURL = try resolvedOutputURL(
      configuration: configuration,
      defaultFilename: "zimage-\(UUID().uuidString).png"
    )

    let schedulerKind = Self.parseSchedulerKind(scheduler)
    let sigmaScheduleKind = Self.parseSigmaScheduleKind(sigmaSchedule)

    // Build DyPE config — auto-enable for high-res requests
    let resolvedWidth = width ?? ZImageModelMetadata.recommendedWidth
    let resolvedHeight = height ?? ZImageModelMetadata.recommendedHeight
    let dyPEConfig: DyPEConfig
    if let dypeRaw = dype?.lowercased() {
      switch dypeRaw {
      case "ntk": dyPEConfig = .ntk
      case "yarn": dyPEConfig = .yarn
      case "none", "off": dyPEConfig = .disabled
      default: dyPEConfig = .disabled
      }
    } else if max(resolvedWidth, resolvedHeight) > 1024 {
      dyPEConfig = .ntk  // Auto-enable for high-res
    } else {
      dyPEConfig = .disabled
    }

    return ZImageGenerationRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: resolvedWidth,
      height: resolvedHeight,
      steps: steps ?? ZImageModelMetadata.recommendedInferenceSteps,
      guidanceScale: guidance ?? ZImageModelMetadata.recommendedGuidanceScale,
      seed: seed,
      outputPath: outputURL,
      levelsMin: levelsMin ?? 0.0,
      levelsMax: levelsMax ?? 1.0,
      model: configuration.modelSpec,
      textEncoderPath: configuration.textEncoderPath,
      maxSequenceLength: configuration.maxSequenceLength,
      loras: activeLoRAs,
      enhancePrompt: false,
      enhanceMaxTokens: 512,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly,
      schedulerKind: schedulerKind,
      sigmaSchedule: sigmaScheduleKind,
      eta: eta,
      dyPE: dyPEConfig,
      inpaintImageData: inpaintImageData,
      maskData: maskData,
      denoise: denoise ?? 1.0,
      maskGrow: maskGrow ?? 0,
      maskFeather: maskFeather ?? 0,
      maskCropX: maskCropX ?? 0,
      maskCropY: maskCropY ?? 0
    )
  }

  func makeImg2ImgRequest(
    configuration: WarmServerConfiguration,
    activeLoRAs: [LoRAConfiguration]
  ) throws -> Img2ImgRequest {
    guard let imagePath else {
      fatalError("makeImg2ImgRequest called without imagePath")
    }

    if imageStrength != nil && creativity != nil {
      throw Img2ImgValidationError.mutuallyExclusive("imageStrength and creativity cannot both be specified")
    }

    let resolvedStrength: Float
    let specifiedAs: Img2ImgRequest.Img2ImgSpecifier
    if let creativity {
      resolvedStrength = 1.0 - max(0.01, min(0.99, creativity))
      specifiedAs = .creativity
    } else {
      resolvedStrength = imageStrength ?? 0.3
      specifiedAs = .strength
    }

    let schedulerKind = Self.parseSchedulerKind(scheduler)
    let sigmaScheduleKind = Self.parseSigmaScheduleKind(sigmaSchedule)

    let resolvedWidth = width ?? ZImageModelMetadata.recommendedWidth
    let resolvedHeight = height ?? ZImageModelMetadata.recommendedHeight
    let dyPEConfig: DyPEConfig
    if let dypeRaw = dype?.lowercased() {
      switch dypeRaw {
      case "ntk": dyPEConfig = .ntk
      case "yarn": dyPEConfig = .yarn
      default: dyPEConfig = .disabled
      }
    } else if max(resolvedWidth, resolvedHeight) > 1024 {
      dyPEConfig = .ntk
    } else {
      dyPEConfig = .disabled
    }

    let outputURL = try resolvedOutputURL(
      configuration: configuration,
      defaultFilename: "zimage-img2img-\(UUID().uuidString).png"
    )

    return Img2ImgRequest(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: width,
      height: height,
      steps: steps ?? ZImageModelMetadata.recommendedInferenceSteps,
      guidanceScale: guidance ?? ZImageModelMetadata.recommendedGuidanceScale,
      seed: seed,
      outputPath: outputURL,
      levelsMin: levelsMin ?? 0.0,
      levelsMax: levelsMax ?? 1.0,
      model: configuration.modelSpec,
      textEncoderPath: configuration.textEncoderPath,
      maxSequenceLength: configuration.maxSequenceLength,
      loras: activeLoRAs,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly,
      schedulerKind: schedulerKind,
      sigmaSchedule: sigmaScheduleKind,
      eta: eta,
      dyPE: dyPEConfig,
      sourceImagePath: imagePath,
      strength: resolvedStrength,
      specifiedAs: specifiedAs
    )
  }

  enum Img2ImgValidationError: Error, LocalizedError {
    case mutuallyExclusive(String)
    var errorDescription: String? {
      switch self {
      case .mutuallyExclusive(let msg): return msg
      }
    }
  }

  private static func parseSchedulerKind(_ rawValue: String?) -> SchedulerKind {
    guard let rawValue else { return .euler }
    switch rawValue {
    case "res_2s":
      return .res2s
    default:
      return SchedulerKind(rawValue: rawValue) ?? .euler
    }
  }

  private static func parseSigmaScheduleKind(_ rawValue: String?) -> SigmaScheduleKind {
    guard let rawValue else { return .flow }
    switch rawValue {
    case "beta57":
      return .beta57
    default:
      return SigmaScheduleKind(rawValue: rawValue) ?? .flow
    }
  }

  func validateOutputPath(configuration: WarmServerConfiguration) throws {
    guard let outputPath, !outputPath.isEmpty else { return }
    _ = try WarmServerOutputPathValidator.resolveOutputPath(
      outputPath,
      allowedOutputDirectory: configuration.allowedOutputDirectory
    )
  }

  func resolvedOutputURL(
    configuration: WarmServerConfiguration,
    defaultFilename: String
  ) throws -> URL {
    guard let outputPath, !outputPath.isEmpty else {
      return URL(fileURLWithPath: NSTemporaryDirectory())
        .appendingPathComponent(defaultFilename)
    }

    return try WarmServerOutputPathValidator.resolveOutputPath(
      outputPath,
      allowedOutputDirectory: configuration.allowedOutputDirectory
    )
  }
}

private struct GenerateResponse: Encodable, Sendable {
  let success: Bool
  let outputPath: String
  let durationMs: Int
}

private struct LoRASwapPayload: Decodable, Sendable {
  let loras: [LoRAEntry]

  func makeConfigurations() throws -> [LoRAConfiguration] {
    try loras.map { try $0.makeConfiguration() }
  }
}

private struct LoRASwapResponse: Encodable, Sendable {
  let success: Bool
  let loraCount: Int
  let loras: [LoRAState]
}

private struct LoRAEntry: Codable, Sendable {
  let path: String
  let scale: Float?

  func makeConfiguration() throws -> LoRAConfiguration {
    let expanded = (path as NSString).expandingTildeInPath
    if path.hasPrefix("/") || path.hasPrefix("./") || path.hasPrefix("../") || path.hasPrefix("~") || FileManager.default.fileExists(atPath: expanded) {
      return .local(expanded, scale: scale ?? 1.0)
    }
    return .huggingFace(path, scale: scale ?? 1.0)
  }
}

private struct ShutdownResponse: Encodable, Sendable {
  let success: Bool
  let message: String
}

private struct HealthResponse: Encodable, Sendable {
  let status: String
  let model: String
  let modelFamily: String
  let modelVariant: String?
  let textEncoderPath: String?
  let loaded: Bool
  let loras: [LoRAState]
  let uptimeSeconds: Int
  let renderCount: Int
  let failedRenderCount: Int
  let pendingCount: Int
  let maxPending: Int
  let isRendering: Bool
  let activeRequestAgeMs: Int?
  let memoryUsageBytes: UInt64
  let memoryUsageMB: UInt64
  let lastRenderDurationMs: Int?
  let lastError: String?
}

private struct LoRAState: Encodable, Sendable {
  let source: String
  let scale: Float

  init(_ configuration: LoRAConfiguration) {
    switch configuration.source {
    case .local(let url):
      self.source = url.path
    case .huggingFace(let modelId, let filename):
      self.source = filename.map { "\(modelId)/\($0)" } ?? modelId
    }
    self.scale = configuration.scale
  }
}

struct ErrorPayload: Encodable {
  let success: Bool
  let error: String
}

private enum QueuedOperation: Sendable {
  case generate(GeneratePayload, ContinuationBox<GenerateResponse>, (@Sendable (ZImagePipeline.GenerationProgress) -> Void)?)
  case controlGenerate(ZImageControlGenerationRequest, ContinuationBox<GenerateResponse>)
  case swap(LoRASwapPayload, ContinuationBox<LoRASwapResponse>)
  case shutdown(ContinuationBox<ShutdownResponse>)
}

private final class ContinuationBox<Value>: @unchecked Sendable {
  private let continuation: CheckedContinuation<Value, Error>

  init(_ continuation: CheckedContinuation<Value, Error>) {
    self.continuation = continuation
  }

  func resume(returning value: Value) {
    continuation.resume(returning: value)
  }

  func resume(throwing error: Error) {
    continuation.resume(throwing: error)
  }
}

private final class SyncResult<Value> {
  private let semaphore = DispatchSemaphore(value: 0)
  private let lock = NSLock()
  private var result: Result<Value, Error>?

  func succeed(_ value: Value) {
    store(.success(value))
  }

  func fail(_ error: Error) {
    store(.failure(error))
  }

  func wait() throws -> Value {
    semaphore.wait()
    lock.lock()
    defer { lock.unlock() }
    return try result!.get()
  }

  private func store(_ result: Result<Value, Error>) {
    lock.lock()
    defer { lock.unlock() }
    guard self.result == nil else { return }
    self.result = result
    semaphore.signal()
  }
}

public enum WarmServerError: Error, LocalizedError {
  case invalidPort(UInt16)
  case invalidOutputPath(path: String, allowedDirectory: String)
  case flux2DetectionFailed(String)
  case flux2NotLoaded
  case fiboDetectionFailed(String)
  case fiboNotLoaded
  case chromaDetectionFailed(String)
  case chromaNotLoaded
  case loraSwapNotSupported
  case controlNetNotSupported

  public var errorDescription: String? {
    switch self {
    case .invalidPort(let port):
      return "Invalid server port: \(port)"
    case .invalidOutputPath(let path, let allowedDirectory):
      return "Output path '\(path)' must be under allowed output directory '\(allowedDirectory)'"
    case .flux2DetectionFailed(let model):
      return "Model '\(model)' was identified as Flux 2 but detection failed at the snapshot directory"
    case .flux2NotLoaded:
      return "Flux 2 pipeline is not loaded"
    case .fiboDetectionFailed(let model):
      return "Model '\(model)' was identified as FIBO but detection failed at the snapshot directory"
    case .fiboNotLoaded:
      return "FIBO pipeline is not loaded"
    case .chromaDetectionFailed(let model):
      return "Model '\(model)' was identified as Chroma but detection failed at the snapshot directory"
    case .chromaNotLoaded:
      return "Chroma pipeline is not loaded"
    case .loraSwapNotSupported:
      return "LoRA swap is not supported for this model family"
    case .controlNetNotSupported:
      return "ControlNet is not supported for this model family"
    }
  }
}
