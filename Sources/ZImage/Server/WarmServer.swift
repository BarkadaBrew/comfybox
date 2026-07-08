import Foundation
import Dispatch
import Logging
import Network
import Darwin
import MLX
import CoreGraphics
import ImageIO

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
  /// Path to the LTX-2 weights directory (transformer / VAE / connector).
  /// When set (with `ltx2GemmaPath`), enables LOCAL video generation on
  /// /v1/video/generate. Lazy-loaded on first request (~38GB), so it's off
  /// until a video is requested.
  public var ltx2WeightsPath: String?
  /// Gemma-3 tokenizer + text-encoder snapshot dir for LTX-2.
  public var ltx2GemmaPath: String?

  public init(
    port: UInt16 = ComfyBoxServerConfig.canonicalPort,
    modelSpec: String? = nil,
    textEncoderPath: String? = nil,
    initialLoRAs: [LoRAConfiguration] = [],
    forceTransformerOverrideOnly: Bool = false,
    maxSequenceLength: Int = 512,
    maxPendingRequests: Int = 10,
    allowedOutputDirectory: String = FileManager.default.currentDirectoryPath,
    seedvr2WeightsPath: String? = nil,
    ltx2WeightsPath: String? = nil,
    ltx2GemmaPath: String? = nil
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
    self.ltx2WeightsPath = ltx2WeightsPath
    self.ltx2GemmaPath = ltx2GemmaPath
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

  /// Serializes lazy initialization of the upscale pipelines. WarmServer is a
  /// plain class reached from concurrent request tasks — without this lock,
  /// simultaneous first-use requests could double-load multi-GB pipelines.
  private let upscalePipelineLock = NSLock()

  /// Replicate video proxy — handles video generation via Replicate API.
  /// Initialized at startup if REPLICATE_API_TOKEN is available; nil otherwise.
  private var replicateVideoProxy: ReplicateVideoProxy?

  /// LoRA Library — indexes, queries, and manages LoRA adapter files.
  /// Initialized at startup; auto-scans if no library.json exists.
  private var loraLibrary: LoRALibrary?

  /// Default upscale models directory path — ESRGAN weights are stored here.
  private static let upscaleModelsDirectoryPath = ("~/bin/zimage/upscale_models" as NSString).expandingTildeInPath

  // MARK: - Creative-layer stores
  //
  // Feature parity with the Coffee Shop image service's creative subsystems. Each persists
  // to a JSON file under ~/.comfybox/ (characters.json, presets.json, content-modes.json,
  // audit-log.jsonl). Constructed eagerly so the first request has warm data; they are cheap
  // (small JSON loads) and thread-safe internally (CharacterStore is an actor; PresetStore /
  // AuditLog guard with a lock / serial queue; ContentModeStore is a value type).

  /// Character registry (~/.comfybox/characters.json).
  let characterStore = CharacterStore()
  /// Nearline model/LoRA catalog (attached storage staged on demand).
  let nearlineLibrary = NearlineLibrary()
  /// Local LTX-2 video generator, built lazily when the weights are configured.
  private var ltx2Generator: LTX2VideoGenerator?
  /// Generation presets (~/.comfybox/presets.json). Seeds defaults on first run.
  let presetStore = PresetStore()
  /// Content-mode definitions (~/.comfybox/content-modes.json). Built-ins ship in-code.
  let contentModeStore = ContentModeStore.loadOrCreate()
  /// Append-only audit trail (~/.comfybox/audit-log.jsonl).
  let auditLog = AuditLog()
  /// Server stats + memory-pressure sampler (pure logic; live probes isolated).
  private let statsProvider = StatsProvider()
  /// Server start time, for the /v1/stats uptime figure.
  private let serverStartTime = Date()

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

    // Initialize Replicate video proxy if API key is available.
    if let replicateKey = ProcessInfo.processInfo.environment["REPLICATE_API_TOKEN"], !replicateKey.isEmpty {
      self.replicateVideoProxy = ReplicateVideoProxy(
        apiKey: replicateKey,
        allowedOutputDirectory: configuration.allowedOutputDirectory,
        logger: logger
      )
      logger.info("Video proxy: enabled (Replicate)")
    } else {
      self.replicateVideoProxy = nil
      logger.info("Video proxy: disabled (no API key)")
    }

    // Wire up the upscale handler. ESRGAN models are always available (lazy-loaded from
    // ~/bin/zimage/upscale_models/); SeedVR2 additionally requires a configured weights path.
    let upscaleHandler: ComfyBridgeUpscaleHandler? = { [unowned self] (imageData: Data, modelName: String, progressCallback: ComfyBridgeProgressHandler?) async throws -> ComfyBridgeGenerateResult in
      try await self.bridgeUpscale(imageData: imageData, modelName: modelName, progressCallback: progressCallback)
    }

    self.comfyBridge.configureExecutor(
      generateHandler: { [unowned self] request, progressCallback, latentPreviewCallback in
        try await self.bridgeGenerate(request, progressCallback: progressCallback, latentPreviewCallback: latentPreviewCallback)
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
    // The switch runs through the coordinator's FIFO render queue so the pool
    // load/activate cannot mutate the active pipeline while a queued render
    // is mid-flight.
    self.comfyBridge.modelSwitchHandler = { [unowned self] (modelId: String) async throws -> Bool in
      try await self.coordinator.enqueueModelSwitch { [unowned self] in
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

    // Wire interrupt handler so ComfyUI /interrupt cancels the in-flight render
    // task — the pipelines observe cancellation in their denoise loops.
    self.comfyBridge.interruptHandler = { [unowned self] in
      await self.coordinator.cancelActiveRender()
    }
  }

  public func run() throws {
    // Ignore SIGHUP before model loading — prevents SSH disconnect from
    // killing the daemon during the ~40s pipeline initialization phase.
    signal(SIGHUP, SIG_IGN)

    // Merge the legacy Coffee Shop image-service character registry (source of
    // truth for hand-written character text) before serving. Idempotent: only
    // missing or never-edited entries change, so user edits are never clawed back.
    let store = characterStore
    let migrationLogger = logger
    Task {
      let migrated = await store.importLegacyRegistry()
      if migrated > 0 {
        migrationLogger.info("Characters: merged \(migrated) entries from legacy image-service registry")
      }
    }

    // Same idempotent one-time merge for the old image-service presets.
    let importedPresets = presetStore.importLegacyImageService()
    if importedPresets > 0 {
      logger.info("Presets: imported \(importedPresets) from legacy image-service")
    }

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

    // Video job pruning timer — clean up completed jobs older than 1 hour.
    if replicateVideoProxy != nil {
      let pruneTimer = DispatchSource.makeTimerSource(queue: listenerQueue)
      pruneTimer.schedule(deadline: .now() + 600, repeating: 600)  // Every 10 minutes
      pruneTimer.setEventHandler { [weak self] in
        self?.replicateVideoProxy?.pruneCompletedJobs()
      }
      pruneTimer.resume()
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
    if let bridgeResponse = await comfyBridge.route(request) {
      return bridgeResponse
    }

    switch (request.method, request.path) {
    case ("GET", "/health"):
      let memoryBytes = Self.currentMemoryFootprintBytes()
      let health = await coordinator.health(memoryBytes: memoryBytes)
      // Encode base health, then inject video section
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      if var healthJSON = try? JSONSerialization.jsonObject(
        with: encoder.encode(health)
      ) as? [String: Any] {
        // Telemetry contract: always emit these keys (JSON null when idle) so
        // clients can decode current_job_id / progress_percent unconditionally.
        healthJSON["current_job_id"] = (health.currentJobId as Any?) ?? NSNull()
        healthJSON["progress_percent"] = (health.progressPercent as Any?) ?? NSNull()
        let videoAvailable = replicateVideoProxy != nil
        healthJSON["video"] = [
          "available": videoAvailable,
          "backend": videoAvailable ? "replicate" : "none",
          "active_jobs": replicateVideoProxy?.activeJobCount ?? 0,
        ] as [String: Any]
        if let data = try? JSONSerialization.data(withJSONObject: healthJSON, options: [.sortedKeys]) {
          return .json(.rawJSON(status: 200, data: data))
        }
      }
      return .json(status: 200, payload: health)

    case ("POST", "/v1/generate"):
      do {
        var payload = try decode(GeneratePayload.self, from: request.body)
        // Bytes-uploaded img2img init image (init_image_base64) — write it to a
        // temp file so remote clients don't need a pre-existing server path.
        if let initData = payload.initImageData, payload.imagePath == nil {
          let tempPath = NSTemporaryDirectory() + "zimage-init-\(UUID().uuidString).png"
          try initData.write(to: URL(fileURLWithPath: tempPath))
          payload.imagePath = tempPath
        }
        try payload.validateOutputPath(configuration: configuration)
        let result = try await coordinator.enqueueGenerate(payload, source: payload.source ?? "api")
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/lora/swap"):
      do {
        var payload = try decode(LoRASwapPayload.self, from: request.body)
        payload = stageNearlineLoras(in: payload)
        let result = try await coordinator.enqueueSwap(payload)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    // MARK: - Nearline storage

    case ("GET", "/v1/nearline"):
      return nearlineListResponse()

    case ("POST", "/v1/nearline/scan"):
      let count = nearlineLibrary.scan()
      auditLog.append(kind: "nearline.scan", message: "Nearline scan found \(count) items")
      return nearlineListResponse()

    case ("POST", "/v1/nearline/stage"):
      struct NameBody: Decodable { let name: String }
      do {
        let body = try decode(NameBody.self, from: request.body)
        let staged = try nearlineLibrary.stage(name: body.name)
        auditLog.append(kind: "nearline.stage", message: "Staged \(body.name)", metadata: ["path": staged])
        return nearlineListResponse()
      } catch let error as NearlineError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: "Stage failed: \(error.localizedDescription)"))
      }

    case ("POST", "/v1/nearline/evict"):
      struct NameBody: Decodable { let name: String }
      do {
        let body = try decode(NameBody.self, from: request.body)
        let evicted = nearlineLibrary.evict(name: body.name)
        if evicted {
          auditLog.append(kind: "nearline.evict", message: "Evicted \(body.name)")
        }
        return evicted
          ? nearlineListResponse()
          : .error(.error(status: 404, message: "Not staged: \(body.name)"))
      } catch {
        return .error(.error(status: 400, message: "Invalid evict payload"))
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

    // MARK: - Config
    // The config document is served/accepted in its canonical camelCase shape (matching
    // ~/.comfybox/config.json and the desktop's plain Codable) — not the snake_case DTO
    // convention used by the render/status routes.

    case ("GET", "/v1/config"):
      let config = ComfyBoxServerConfig.loadOrMigrate()
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      if let data = try? encoder.encode(config) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize config"))

    case ("PUT", "/v1/config"):
      do {
        let updated = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: request.body)
        try updated.save()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(updated)
        // Port/host changes take effect on next server start; the running listener is unchanged.
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(.error(status: 400, message: "Invalid config: \(error.localizedDescription)"))
      }

    case ("GET", "/v1/providers/status"):
      let config = ComfyBoxServerConfig.loadOrMigrate()
      func status(_ endpoint: AIProviderEndpoint?) -> [String: Any] {
        guard let endpoint else { return ["configured": false] }
        return [
          "configured": true,
          "model": endpoint.model,
          "base_url": endpoint.baseUrl,
          "has_api_key": !(endpoint.apiKey ?? "").isEmpty,
        ]
      }
      let payload: [String: Any] = [
        "prompt_optimization": status(config.providers.promptOptimization),
        "vision": status(config.providers.vision),
        "captioning": status(config.providers.captioning),
        "replicate": ["configured": !(config.replicate?.apiKey ?? "").isEmpty],
      ]
      if let data = try? JSONSerialization.data(withJSONObject: payload) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize provider status"))

    // MARK: - Model Pool Endpoints

    case ("POST", "/v1/model/load"):
      do {
        let payload = try decode(ModelLoadRequest.self, from: request.body)
        let shouldActivate = payload.activate ?? true
        let shouldWait = payload.wait ?? true

        // Resolve CivitAI model IDs (e.g. 'cyberrealistic-v5') to file paths
        let resolvedSpec = Self.parseModelSpec(from: payload.model)
        let resolvedQuantization = payload.quantization ?? Self.parseQuantization(from: payload.model)

        if shouldWait {
          let result = try await coordinator.poolLoad(
            modelSpec: resolvedSpec,
            quantization: resolvedQuantization,
            activate: shouldActivate
          )
          return .json(status: 200, payload: result)
        } else {
          // Fire-and-forget: start loading in background, return immediately.
          Task {
            do {
              try await coordinator.poolLoad(
                modelSpec: resolvedSpec,
                quantization: resolvedQuantization,
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

    // MARK: - Video Endpoints

    case ("POST", "/v1/video/generate"):
      // Determine the caller's backend intent (local / cloud / unspecified).
      let videoIntent = (try? decode(VideoGenerateRequest.self, from: request.body))?.backendIntent ?? .unspecified

      // Explicit cloud is the ONLY way to reach paid Replicate. Otherwise prefer
      // local, and never silently fall back to cloud for an explicit-local request.
      if videoIntent != .cloud {
        if let localResponse = await localVideoResponseIfConfigured(body: request.body) {
          logger.info("video: routing to local LTX-2")
          return localResponse
        }
        if videoIntent == .local {
          return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights). Pass backend: \"replicate\" to explicitly use paid cloud."))
        }
        // Unspecified + local unavailable: fall back to cloud, but LOUDLY — this
        // spends money and leaves the device. Callers wanting zero-cloud should
        // pass backend: "local".
        logger.warning("video: local LTX-2 not configured; falling back to PAID Replicate cloud (\(ReplicateVideoProxy.i2vModel)). Pass backend:\"local\" to forbid, backend:\"replicate\" to silence this warning.")
      }
      guard let proxy = replicateVideoProxy else {
        return .error(.error(status: 503, message: "Video generation not available: configure LTX-2 (--ltx2-weights) for local video, or a Replicate API key for cloud"))
      }
      logger.info("video: routing to Replicate cloud (\(ReplicateVideoProxy.i2vModel))")
      do {
        var videoRequest = try decode(VideoGenerateRequest.self, from: request.body)
        // Accept a bytes-uploaded init image (image_base64) when no path is given.
        if videoRequest.imagePath == nil, let tempPath = Self.writeTempImage(base64: videoRequest.imageBase64) {
          videoRequest.imagePath = tempPath
        }
        if let validationError = videoRequest.validate() {
          return .error(.error(status: 400, message: validationError))
        }
        // Enforce output path containment within the allowed output directory
        // (throws WarmServerError.invalidOutputPath -> 400 via response(for:)).
        if let outputPath = videoRequest.outputPath, !outputPath.isEmpty {
          _ = try WarmServerOutputPathValidator.resolveOutputPath(
            outputPath,
            allowedOutputDirectory: configuration.allowedOutputDirectory
          )
        }
        // I2V: verify image_path exists, is a regular file, and has PNG/JPEG
        // magic bytes before it gets base64-uploaded to Replicate.
        if let imagePath = videoRequest.imagePath {
          if let imageError = ReplicateVideoProxy.validateSourceImage(atPath: imagePath) {
            return .error(.error(status: 400, message: imageError))
          }
        }
        let jobStatus = await proxy.submit(videoRequest)
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(jobStatus)
        return .json(.rawJSON(status: 202, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("GET", _) where request.path.hasPrefix("/v1/video/status/"):
      let jobId = String(request.path.dropFirst("/v1/video/status/".count))
      guard !jobId.isEmpty else {
        return .error(.error(status: 400, message: "Missing job_id in path"))
      }
      guard let proxy = replicateVideoProxy else {
        return .error(.error(status: 503, message: "Video generation not available: Replicate API key not configured"))
      }
      guard let jobStatus = proxy.status(jobId: jobId) else {
        return .error(.error(status: 404, message: "Video job not found: \(jobId)"))
      }
      do {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(jobStatus)
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(.error(status: 500, message: "Failed to encode job status"))
      }

    case ("GET", "/v1/video/output"):
      // Download a rendered video's bytes so remote clients don't need SCP.
      // ?path=<server output path>, validated to be within the allowed dir.
      guard let raw = request.queryParameters["path"], !raw.isEmpty,
            let path = raw.removingPercentEncoding else {
        return .error(.error(status: 400, message: "Missing ?path= for video output"))
      }
      do {
        let resolved = try WarmServerOutputPathValidator.resolveOutputPath(
          path, allowedOutputDirectory: configuration.allowedOutputDirectory).path
        guard FileManager.default.fileExists(atPath: resolved),
              let data = FileManager.default.contents(atPath: resolved) else {
          return .error(.error(status: 404, message: "Video output not found (still rendering?): \(path)"))
        }
        return .json(.binary(status: 200, contentType: "video/mp4", data: data))
      } catch {
        return .error(response(for: error))
      }

    // MARK: - Remote gallery (browse the server's output folder)

    case ("GET", "/v1/gallery/list"):
      // List media in the gallery output folder for remote desktop browsing.
      let limit = request.queryParameters["limit"].flatMap { Int($0) } ?? 500
      let dir = (configuration.allowedOutputDirectory as NSString).expandingTildeInPath
      let fm = FileManager.default
      let exts: Set<String> = ["png", "jpg", "jpeg", "webp", "tiff", "heic", "mp4", "mov", "m4v"]
      var items: [[String: Any]] = []
      if let en = fm.enumerator(at: URL(fileURLWithPath: dir), includingPropertiesForKeys: [.contentModificationDateKey, .fileSizeKey], options: [.skipsHiddenFiles]) {
        for case let url as URL in en {
          let ext = url.pathExtension.lowercased()
          guard exts.contains(ext) else { continue }
          let vals = try? url.resourceValues(forKeys: [.contentModificationDateKey, .fileSizeKey])
          let isVideo = ["mp4", "mov", "m4v"].contains(ext)
          items.append([
            "path": url.path,
            "filename": url.lastPathComponent,
            "kind": isVideo ? "video" : "image",
            "size": vals?.fileSize ?? 0,
            "modified": (vals?.contentModificationDate.map { ISO8601DateFormatter().string(from: $0) }) ?? "",
          ])
        }
      }
      items.sort { (($0["modified"] as? String) ?? "") > (($1["modified"] as? String) ?? "") }
      if items.count > limit { items = Array(items.prefix(limit)) }
      guard let data = try? JSONSerialization.data(withJSONObject: ["items": items]) else {
        return .error(.error(status: 500, message: "Failed to serialize gallery list"))
      }
      return .json(.rawJSON(status: 200, data: data))

    case ("GET", "/v1/gallery/file"):
      // Serve a gallery file's bytes (validated within the allowed dir).
      guard let raw = request.queryParameters["path"], !raw.isEmpty,
            let path = raw.removingPercentEncoding else {
        return .error(.error(status: 400, message: "Missing ?path="))
      }
      do {
        let resolved = try WarmServerOutputPathValidator.resolveOutputPath(
          path, allowedOutputDirectory: configuration.allowedOutputDirectory).path
        guard FileManager.default.fileExists(atPath: resolved),
              let data = FileManager.default.contents(atPath: resolved) else {
          return .error(.error(status: 404, message: "File not found: \(path)"))
        }
        let ct: String
        switch (resolved as NSString).pathExtension.lowercased() {
        case "png": ct = "image/png"
        case "jpg", "jpeg": ct = "image/jpeg"
        case "webp": ct = "image/webp"
        case "tiff": ct = "image/tiff"
        case "heic": ct = "image/heic"
        case "mp4", "m4v": ct = "video/mp4"
        case "mov": ct = "video/quicktime"
        default: ct = "application/octet-stream"
        }
        return .json(.binary(status: 200, contentType: ct, data: data))
      } catch {
        return .error(response(for: error))
      }

    // MARK: - Upscale Endpoint

    case ("POST", "/v1/upscale"):
      do {
        let payload = try decode(UpscalePayload.self, from: request.body)
        let result = try await handleUpscale(payload)
        return .json(status: 200, payload: result)
      } catch {
        return .error(response(for: error))
      }

    // MARK: - Creative Layer: Characters
    // Character registry parity with the image service. Path-parameter routes follow the
    // /v1/loras/ hasPrefix pattern.

    case ("POST", "/v1/enhance"):
      return await enhancePromptResponse(body: request.body)

    // MARK: - Queue management

    case ("GET", "/v1/queue"):
      return await queueListResponse()

    case ("POST", "/v1/queue/interrupt"):
      struct InterruptResult: Encodable { let success: Bool; let interrupted: Bool }
      let cancelled = await coordinator.cancelActiveRender()
      auditLog.append(kind: "queue.interrupt", message: cancelled ? "Interrupted active render" : "No active render")
      return .json(status: 200, payload: InterruptResult(success: true, interrupted: cancelled))

    case ("POST", "/v1/queue/clear"):
      struct ClearResult: Encodable { let success: Bool; let cleared: Int }
      let cleared = await coordinator.clearPending()
      auditLog.append(kind: "queue.clear", message: "Cleared \(cleared) pending job(s)")
      return .json(status: 200, payload: ClearResult(success: true, cleared: cleared))

    case ("POST", "/v1/queue/pause"), ("POST", "/v1/queue/resume"):
      struct PauseResult: Encodable { let success: Bool; let paused: Bool }
      let paused = request.path.hasSuffix("/pause")
      await coordinator.setPaused(paused)
      auditLog.append(kind: "queue.pause", message: paused ? "Queue paused" : "Queue resumed")
      return .json(status: 200, payload: PauseResult(success: true, paused: paused))

    case ("POST", _) where request.path.hasPrefix("/v1/queue/") && request.path.hasSuffix("/move"):
      let mid = request.path.dropFirst("/v1/queue/".count).dropLast("/move".count)
      guard let id = Self.pathIdComponent(String(mid)) else {
        return .error(.error(status: 400, message: "Invalid job id"))
      }
      struct MoveBody: Decodable { let direction: String }
      let direction = (try? JSONDecoder().decode(MoveBody.self, from: request.body))?.direction ?? "up"
      struct MoveResult: Encodable { let success: Bool; let moved: Bool }
      let moved = await coordinator.movePending(id: id, direction: direction)
      if moved { auditLog.append(kind: "queue.move", message: "Moved job \(id) \(direction)", metadata: ["id": id, "direction": direction]) }
      return .json(status: 200, payload: MoveResult(success: true, moved: moved))

    case ("DELETE", _) where request.path.hasPrefix("/v1/queue/"):
      guard let id = Self.pathIdComponent(String(request.path.dropFirst("/v1/queue/".count))) else {
        return .error(.error(status: 400, message: "Invalid job id"))
      }
      let removed = await coordinator.cancelPending(id: id)
      if removed {
        auditLog.append(kind: "queue.cancel", message: "Cancelled pending job \(id)", metadata: ["id": id])
      }
      return removed
        ? .json(status: 200, payload: DeleteResult(success: true, id: id, deleted: true))
        : .error(.error(status: 404, message: "Job not pending: \(id)"))

    case ("GET", "/v1/characters"):
      return await listCharactersResponse()

    case ("POST", "/v1/characters"), ("PUT", "/v1/characters"):
      return await upsertCharacterResponse(body: request.body)

    case ("GET", _) where request.path.hasPrefix("/v1/characters/"):
      return await getCharacterResponse(rawId: String(request.path.dropFirst("/v1/characters/".count)))

    case ("DELETE", _) where request.path.hasPrefix("/v1/characters/"):
      return await deleteCharacterResponse(rawId: String(request.path.dropFirst("/v1/characters/".count)))

    // MARK: - Creative Layer: Presets

    case ("GET", "/v1/presets"):
      return presetsListResponse()

    case ("POST", "/v1/presets/resolve"):
      // Match before the generic /v1/presets/ prefix routes below.
      return resolvePresetResponse(body: request.body)

    case ("POST", "/v1/presets/import-legacy"):
      struct ImportResult: Encodable { let success: Bool; let imported: Int }
      let count = presetStore.importLegacyImageService()
      if count > 0 {
        auditLog.append(kind: "preset.import", message: "Imported \(count) legacy image-service preset(s)")
      }
      return .json(status: 200, payload: ImportResult(success: true, imported: count))

    case ("POST", "/v1/presets"), ("PUT", "/v1/presets"):
      return upsertPresetResponse(body: request.body)

    case ("GET", _) where request.path.hasPrefix("/v1/presets/"):
      return getPresetResponse(rawId: String(request.path.dropFirst("/v1/presets/".count)))

    case ("DELETE", _) where request.path.hasPrefix("/v1/presets/"):
      return deletePresetResponse(rawId: String(request.path.dropFirst("/v1/presets/".count)))

    // MARK: - Creative Layer: Content modes

    case ("GET", "/v1/content-modes"):
      return contentModesResponse()

    // MARK: - Creative Layer: Stats / memory

    case ("GET", "/v1/stats"):
      return await statsResponse()

    case ("GET", "/v1/memory"):
      return memoryResponse()

    // MARK: - Creative Layer: Audit log

    case ("GET", "/v1/audit-log"):
      return auditLogResponse(query: request.queryParameters)

    default:
      if ["/v1/generate", "/v1/lora/swap", "/v1/shutdown", "/health",
          "/v1/model/load", "/v1/model/activate", "/v1/model/pool", "/v1/model/unload",
          "/v1/loras", "/v1/loras/scan", "/v1/video/generate", "/v1/upscale",
          "/v1/characters", "/v1/presets", "/v1/presets/resolve",
          "/v1/content-modes", "/v1/stats", "/v1/memory", "/v1/audit-log"
      ].contains(request.path) || request.path.hasPrefix("/v1/loras/")
         || request.path.hasPrefix("/v1/video/status/")
         || request.path.hasPrefix("/v1/characters/")
         || request.path.hasPrefix("/v1/presets/") {
        return .error(.error(status: 405, message: "Method not allowed"))
      }
      return .error(.error(status: 404, message: "Not found"))
    }
  }

  // MARK: - Creative-layer route handlers
  //
  // These back the /v1/characters, /v1/presets, /v1/content-modes, /v1/stats, /v1/memory,
  // and /v1/audit-log routes above. Kept as small private methods so the main route switch
  // stays readable. Responses use the same helpers as the rest of the server:
  // `RoutedResponse.json(status:payload:)` (snake_case JSON) and `.error(.error(...))`.

  /// Small `{ success, id, deleted }` payload for DELETE responses.
  private struct DeleteResult: Encodable {
    let success: Bool
    let id: String
    let deleted: Bool
  }

  /// Validate + percent-decode a single path-parameter id (rejects empty / nested paths),
  /// matching the guard the /v1/loras/{id} routes use.
  private static func pathIdComponent(_ raw: String) -> String? {
    let decoded = raw.removingPercentEncoding ?? raw
    guard !decoded.isEmpty, !decoded.contains("/") else { return nil }
    return decoded
  }

  // Nearline -------------------------------------------------------------------

  /// GET /v1/nearline payload: config + full catalog with staging state.
  private func nearlineListResponse() -> RoutedResponse {
    let iso = ISO8601DateFormatter()
    let config = nearlineLibrary.configuration
    let payload: [String: Any] = [
      "roots": config.roots,
      "cache_limit_gb": config.cacheLimitGB,
      "staged_mb": nearlineLibrary.stagedMB,
      "items": nearlineLibrary.list().map { item -> [String: Any] in
        var dict: [String: Any] = [
          "name": item.name,
          "path": item.path,
          "size_mb": item.sizeMB,
          "kind": item.kind,
          "staged": item.staged,
        ]
        if let stagedPath = item.stagedPath { dict["staged_path"] = stagedPath }
        if let lastUsed = item.lastUsedAt { dict["last_used_at"] = iso.string(from: lastUsed) }
        return dict
      },
    ]
    guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
      return .error(.error(status: 500, message: "Failed to serialize nearline catalog"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }

  /// Auto-stage: rewrite bare LoRA filenames that only exist on nearline
  /// storage to their freshly staged local paths, so a preset (or any swap
  /// request) can reference archived LoRAs and they appear on demand.
  private func stageNearlineLoras(in payload: LoRASwapPayload) -> LoRASwapPayload {
    let entries = payload.loras.map { entry -> LoRAEntry in
      // Only bare safetensors filenames are candidates — absolute/relative
      // paths and HF ids resolve through the normal machinery.
      guard !entry.path.hasPrefix("/"), !entry.path.hasPrefix("~"), !entry.path.hasPrefix("."),
            entry.path.hasSuffix(".safetensors"),
            !FileManager.default.fileExists(atPath: (entry.path as NSString).expandingTildeInPath),
            nearlineLibrary.item(named: entry.path) != nil
      else { return entry }
      guard let staged = try? nearlineLibrary.stage(name: entry.path) else { return entry }
      logger.info("Nearline: auto-staged \(entry.path) for LoRA swap")
      return LoRAEntry(path: staged, scale: entry.scale)
    }
    return LoRASwapPayload(loras: entries)
  }

  // Local video (LTX-2) ---------------------------------------------------------

  /// Body for the local LTX-2 video route (snake_case over the wire).
  /// Decode a base64 image (image_base64) to a temp PNG and return its path, so
  /// remote clients can send an init image without a pre-existing server file.
  /// Returns nil when the string is absent/undecodable.
  private static func writeTempImage(base64: String?) -> String? {
    guard let base64, let data = Data(base64Encoded: base64) else { return nil }
    let path = NSTemporaryDirectory() + "zimage-vidinit-\(UUID().uuidString).png"
    return (try? data.write(to: URL(fileURLWithPath: path))) != nil ? path : nil
  }

  private struct LocalVideoRequest: Decodable {
    let prompt: String
    let negativePrompt: String?
    let imagePath: String?
    /// I2V init image sent as base64 (image_base64) for remote clients.
    let imageBase64: String?
    let width: Int?
    let height: Int?
    let frames: Int?
    let steps: Int?
    let seed: UInt64?
    let strength: Float?
    let extendToSeconds: Float?
    let fps: Int?
    let loraPath: String?
    let loraStrength: Float?
    let outputPath: String?
  }

  private struct LocalVideoResponse: Encodable {
    let success: Bool
    let outputPath: String
    let frameCount: Int
    let durationSeconds: Float
    let elapsedSeconds: Double
    let backend: String
  }

  /// If LTX-2 is configured, generate the video locally and return the result;
  /// otherwise nil so the caller falls through to the Replicate proxy.
  private func localVideoResponseIfConfigured(body: Data) async -> RoutedResponse? {
    guard let weights = configuration.ltx2WeightsPath, let gemma = configuration.ltx2GemmaPath else {
      return nil
    }
    do {
      let req = try decode(LocalVideoRequest.self, from: body)

      // Contain the output within the allowed directory (default alongside models).
      let requestedOutput = req.outputPath ?? "ltx2-\(UUID().uuidString).mp4"
      let resolvedOutput = try WarmServerOutputPathValidator.resolveOutputPath(
        requestedOutput, allowedOutputDirectory: configuration.allowedOutputDirectory).path

      let generator = ltx2Generator ?? LTX2VideoGenerator(
        config: .init(weightsDir: weights, gemmaPath: gemma), logger: logger)
      ltx2Generator = generator

      // Accept an init image as bytes (image_base64) when no server path is given.
      let effectiveInitImage = req.imagePath ?? Self.writeTempImage(base64: req.imageBase64)

      let videoRequest = LTX2VideoRequest(
        prompt: req.prompt,
        negativePrompt: req.negativePrompt,
        initImagePath: effectiveInitImage,
        width: req.width ?? 704,
        height: req.height ?? 448,
        framesPerChunk: req.frames ?? 97,
        steps: req.steps ?? 8,
        seed: req.seed ?? 42,
        strength: req.strength ?? 1.0,
        extendToSeconds: req.extendToSeconds ?? 0,
        fps: req.fps ?? 24,
        loraPath: req.loraPath,
        loraStrength: req.loraStrength ?? 1.0,
        outputPath: resolvedOutput
      )
      // Validate before enqueuing so bad frames/dims fail fast.
      try generator.validate(videoRequest)

      logger.info("LTX-2: local video request queued (\(videoRequest.width)x\(videoRequest.height), \(videoRequest.framesPerChunk)f)")
      let result = try await coordinator.enqueueLocalVideo {
        try generator.generate(videoRequest)
      }
      auditLog.append(kind: "video.local", message: "LTX-2 video \(result.frameCount)f -> \(result.outputPath)")
      return .json(status: 200, payload: LocalVideoResponse(
        success: true,
        outputPath: result.outputPath,
        frameCount: result.frameCount,
        durationSeconds: result.durationSeconds,
        elapsedSeconds: result.elapsedSeconds,
        backend: "ltx2-local"
      ))
    } catch let error as LTX2VideoError {
      return .error(.error(status: 400, message: error.localizedDescription))
    } catch {
      return .error(.error(status: 500, message: "LTX-2 video failed: \(error.localizedDescription)"))
    }
  }

  // Queue ----------------------------------------------------------------------

  /// GET /v1/queue: the active operation + every pending job (cancellable by id).
  private func queueListResponse() async -> RoutedResponse {
    let snapshot = await coordinator.queueSnapshot()
    let iso = ISO8601DateFormatter()
    var payload: [String: Any] = [
      "is_rendering": snapshot.isRendering,
      "is_paused": snapshot.isPaused,
      "max_pending": snapshot.maxPending,
      "render_count": snapshot.renderCount,
      "failed_count": snapshot.failedCount,
      "pending": snapshot.pending.map { job in
        [
          "id": job.id,
          "kind": job.kind,
          "summary": job.summary,
          "source": job.source,
          "enqueued_at": iso.string(from: job.enqueuedAt),
        ] as [String: Any]
      },
    ]
    if let id = snapshot.activeJobId { payload["active_job_id"] = id }
    if let summary = snapshot.activeSummary { payload["active_summary"] = summary }
    if let source = snapshot.activeSource { payload["active_source"] = source }
    if let started = snapshot.activeStartedAt { payload["active_started_at"] = iso.string(from: started) }
    if let pct = snapshot.progressPercent { payload["progress_percent"] = pct }
    guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
      return .error(.error(status: 500, message: "Failed to serialize queue snapshot"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }

  // Prompt enhancement --------------------------------------------------------

  /// POST /v1/enhance body (snake_case over the wire).
  private struct EnhanceRequest: Decodable {
    let prompt: String
    let character: String?
    let characterDescription: String?
    let contentMode: String?
  }

  /// Enhance a prompt through the configured prompt-optimization provider
  /// (Settings → AI Providers; e.g. Dan's heresy model on LM Studio). Falls
  /// back to the raw prompt when the provider is unreachable — the optimizer
  /// never blocks a render.
  private func enhancePromptResponse(body: Data) async -> RoutedResponse {
    guard let req = try? decode(EnhanceRequest.self, from: body),
          !req.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else {
      return .error(.error(status: 400, message: "'prompt' is required"))
    }

    let config = ComfyBoxServerConfig.loadOrMigrate()
    guard let endpoint = config.providers.promptOptimization else {
      return .error(.error(
        status: 503,
        message: "No prompt-optimization provider configured (Settings → AI Providers)"))
    }

    // PromptOptimizer appends /v1/chat/completions itself; the configured
    // baseUrl is an OpenAI-style root that usually already ends in /v1.
    var base = endpoint.baseUrl
    while base.hasSuffix("/") { base.removeLast() }
    if base.hasSuffix("/v1") { base = String(base.dropLast(3)) }
    while base.hasSuffix("/") { base.removeLast() }

    let optimizer = PromptOptimizer(
      configuration: PromptOptimizer.Configuration(
        ollamaBaseURL: base,
        lmStudioBaseURL: nil,
        model: endpoint.model,
        timeoutSeconds: 90,
        enabled: true
      ),
      logger: logger
    )

    // Resolve a named character to its mode-gated description when the
    // caller didn't supply one.
    let mode = req.contentMode ?? ContentModeManager.Mode.neutral.rawValue
    var characterDescription = req.characterDescription
    if characterDescription == nil, let name = req.character,
       let entry = await characterStore.get(CharacterEntry.slug(name)) {
      characterDescription = entry.resolvedDescription(
        for: ContentModeManager.Mode(rawValue: mode) ?? .neutral)
    }

    let result = await optimizer.optimize(
      prompt: req.prompt,
      character: req.character,
      characterDescription: characterDescription,
      contentMode: mode
    )

    var payload: [String: Any] = [
      "success": true,
      "prompt": result.prompt,
      "enhanced": result.enhanced,
    ]
    if let note = result.note { payload["note"] = note }
    guard let data = try? JSONSerialization.data(withJSONObject: payload) else {
      return .error(.error(status: 500, message: "Failed to serialize enhance response"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }

  // Characters ---------------------------------------------------------------

  private func listCharactersResponse() async -> RoutedResponse {
    .json(status: 200, payload: await characterStore.list())
  }

  private func getCharacterResponse(rawId: String) async -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid character id"))
    }
    guard let character = await characterStore.get(id) else {
      return .error(.error(status: 404, message: "Character not found: \(id)"))
    }
    return .json(status: 200, payload: character)
  }

  private func upsertCharacterResponse(body: Data) async -> RoutedResponse {
    do {
      let character = try decode(CharacterEntry.self, from: body)
      guard !character.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
        return .error(.error(status: 400, message: "Character 'name' is required"))
      }
      let saved = await characterStore.upsert(character)
      auditLog.append(
        kind: "character.upsert",
        message: "Upserted character \(saved.id)",
        metadata: ["id": saved.id, "name": saved.name]
      )
      return .json(status: 200, payload: saved)
    } catch {
      return .error(.error(status: 400, message: "Invalid character payload: \(error.localizedDescription)"))
    }
  }

  private func deleteCharacterResponse(rawId: String) async -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid character id"))
    }
    let deleted = await characterStore.delete(id)
    if deleted {
      auditLog.append(kind: "character.delete", message: "Deleted character \(id)", metadata: ["id": id])
    }
    return .json(status: deleted ? 200 : 404, payload: DeleteResult(success: deleted, id: id, deleted: deleted))
  }

  // Presets ------------------------------------------------------------------

  private func presetsListResponse() -> RoutedResponse {
    .json(status: 200, payload: presetStore.list())
  }

  private func getPresetResponse(rawId: String) -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid preset id"))
    }
    guard let preset = presetStore.get(id) else {
      return .error(.error(status: 404, message: "Preset not found: \(id)"))
    }
    return .json(status: 200, payload: preset)
  }

  private func upsertPresetResponse(body: Data) -> RoutedResponse {
    do {
      let preset = try decode(ImagePreset.self, from: body)
      let saved = try presetStore.upsert(preset)
      auditLog.append(kind: "preset.upsert", message: "Upserted preset \(saved.id)", metadata: ["id": saved.id])
      return .json(status: 200, payload: saved)
    } catch let error as PresetStoreError {
      return presetErrorResponse(error)
    } catch {
      return .error(.error(status: 400, message: "Invalid preset payload: \(error.localizedDescription)"))
    }
  }

  private func deletePresetResponse(rawId: String) -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid preset id"))
    }
    do {
      let deleted = try presetStore.delete(id)
      if deleted {
        auditLog.append(kind: "preset.delete", message: "Deleted preset \(id)", metadata: ["id": id])
      }
      return .json(status: deleted ? 200 : 404, payload: DeleteResult(success: deleted, id: id, deleted: deleted))
    } catch {
      return .error(.error(status: 500, message: "Failed to delete preset: \(error.localizedDescription)"))
    }
  }

  private func resolvePresetResponse(body: Data) -> RoutedResponse {
    struct ResolveRequest: Decodable { let id: String }
    do {
      let request = try decode(ResolveRequest.self, from: body)
      let resolved = try presetStore.resolve(request.id)
      return .json(status: 200, payload: resolved)
    } catch let error as PresetStoreError {
      return presetErrorResponse(error)
    } catch {
      return .error(.error(status: 400, message: #"Invalid resolve request (expected {"id": ...}): \#(error.localizedDescription)"#))
    }
  }

  /// Map a ``PresetStoreError`` to the right HTTP status: validation -> 400, notFound -> 404.
  private func presetErrorResponse(_ error: PresetStoreError) -> RoutedResponse {
    switch error {
    case .validation(let message):
      return .error(.error(status: 400, message: message))
    case .notFound(let id):
      return .error(.error(status: 404, message: "Preset not found: \(id)"))
    }
  }

  // Content modes ------------------------------------------------------------

  private func contentModesResponse() -> RoutedResponse {
    .json(status: 200, payload: contentModeStore.listModes())
  }

  // Stats / memory -----------------------------------------------------------

  private func statsResponse() async -> RoutedResponse {
    let queue = await coordinator.queueStatus()
    let config = ComfyBoxServerConfig.loadOrMigrate()
    let snapshot = statsProvider.snapshot(
      memory: statsProvider.sampleMemoryStatus(),
      uptimeSeconds: StatsProvider.uptimeSeconds(startTime: serverStartTime),
      renderCount: queue.renderCount,
      failedRenderCount: queue.failedCount,
      pendingCount: queue.pendingCount,
      config: config
    )
    return .json(status: 200, payload: snapshot)
  }

  private func memoryResponse() -> RoutedResponse {
    .json(status: 200, payload: statsProvider.sampleMemoryStatus())
  }

  // Audit log ----------------------------------------------------------------

  private func auditLogResponse(query: [String: String]) -> RoutedResponse {
    let limit = query["limit"].flatMap { Int($0) } ?? 100
    let entries = auditLog.recent(limit: max(0, limit))
    // Custom encoder: ISO8601 timestamps (matching the on-disk JSONL). No snake_case
    // conversion — AuditEntry keys are already flat single words, and converting would
    // also mangle arbitrary `metadata` dictionary keys.
    let encoder = JSONEncoder()
    encoder.dateEncodingStrategy = .iso8601
    encoder.outputFormatting = [.sortedKeys]
    guard let data = try? encoder.encode(entries) else {
      return .error(.error(status: 500, message: "Failed to serialize audit log"))
    }
    return .json(.rawJSON(status: 200, data: data))
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

  private func bridgeGenerate(_ request: ComfyBridgeGenerateRequest, progressCallback: ComfyBridgeProgressHandler?, latentPreviewCallback: ComfyBridgeLatentPreviewHandler? = nil) async throws -> ComfyBridgeGenerateResult {
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
    let resolvedSampler: String?

    switch family {
    case .fibo:
      // FIBO: use model defaults, no step clamping
      resolvedSteps = request.steps
      resolvedGuidance = request.guidance > 0 ? request.guidance : 4.0
      resolvedNegativePrompt = request.negativePrompt
      resolvedSampler = request.sampler
    case .chroma:
      // Chroma: 28 steps default, guidance 0.0 (unconditioned)
      resolvedSteps = request.steps > 0 ? request.steps : 28
      resolvedGuidance = request.guidance
      resolvedNegativePrompt = nil
      resolvedSampler = request.sampler
    case .flux1:
      let zimageVariant = await coordinator.currentZImageVariant
      if zimageVariant == .base {
        // Z-Image Base / undistilled checkpoints (Moody, etc.): the ComfyUI/Krita
        // KSampler defaults are tuned for Turbo (9 steps, euler) and produce noise on
        // undistilled models. When the request still carries those turbo defaults, apply
        // the undistilled recommendations (40 steps, dpmpp_2m). If the user changed a
        // value, respect it. (Model-aware defaults from PR #164, @bree.)
        resolvedSteps = request.steps <= 9 ? 40 : request.steps
        resolvedGuidance = request.guidance > 0 ? request.guidance : ZImageModelMetadata.Base.recommendedGuidanceScale
        resolvedNegativePrompt = request.negativePrompt
        let sampler = request.sampler ?? "euler"
        resolvedSampler = sampler == "euler" ? "dpmpp_2m" : sampler
        if resolvedSteps != request.steps || resolvedSampler != request.sampler {
          logger.info("[WarmServer] Z-Image Base override: steps=\(resolvedSteps) (was \(request.steps)), sampler=\(resolvedSampler ?? "nil") (was \(request.sampler ?? "nil"))")
        }
      } else {
        // Z-Image Turbo: distilled, optimal at 9 steps. Honor the requested
        // guidance rather than hardcoding 0 — merged/finetuned "turbo"
        // checkpoints do respond to CFG, so forcing 0 removed real user control
        // (0 is the recommended default, passed through when the client sends it).
        resolvedSteps = min(request.steps, 9)
        resolvedGuidance = request.guidance
        resolvedNegativePrompt = nil
        resolvedSampler = request.sampler
      }
    case .flux2:
      // Base (non-distilled) models support guidance > 1.0 and default to 50 steps;
      // distilled models default to 4 steps and guidance 1.0.
      let isBaseModel = await coordinator.isFlux2BaseModel
      resolvedSteps = request.steps                 // Klein: no step clamp
      resolvedGuidance = isBaseModel ? request.guidance : 1.0
      resolvedNegativePrompt = nil                  // Klein: CFG only when guidance > 1.0
      resolvedSampler = request.sampler
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
      scheduler: resolvedSampler,
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

    // Forward the latent preview callback directly — it uses the same
    // (MLXArray, Int, Int, Int, Int) signature as the pipeline handler.
    let pipelineLatentPreview: ZImagePipeline.LatentPreviewHandler? = latentPreviewCallback

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
        let result = try await coordinator.enqueueGenerate(batchPayload, progressHandler: pipelineProgress, latentPreviewHandler: pipelineLatentPreview, source: "comfyui")
        totalDurationMs += result.durationMs
        lastResult = ComfyBridgeGenerateResult(outputPath: result.outputPath, durationMs: totalDurationMs)
      }
      return lastResult!
    }

    let result = try await coordinator.enqueueGenerate(payload, progressHandler: pipelineProgress, latentPreviewHandler: pipelineLatentPreview, source: "comfyui")
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
    let pipeline = try loadESRGANPipelineIfNeeded(weightsDirectory: weightsDir)

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
    let pipeline = try loadSeedVR2PipelineIfNeeded(weightsPath: weightsPath)

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

  /// Get or lazily create the SeedVR2 pipeline. Double-checked under
  /// `upscalePipelineLock` so concurrent first-use requests cannot
  /// double-load the ~6GB weights.
  private func loadSeedVR2PipelineIfNeeded(weightsPath: String) throws -> SeedVR2Pipeline {
    upscalePipelineLock.lock()
    defer { upscalePipelineLock.unlock() }

    if let pipeline = seedvr2Pipeline {
      return pipeline
    }

    logger.info("WarmServer: lazy-loading SeedVR2 pipeline from \(weightsPath)...")
    let pipeline = try SeedVR2Pipeline(weightsPath: weightsPath, logger: logger)
    seedvr2Pipeline = pipeline
    logger.info("WarmServer: SeedVR2 pipeline ready (\(pipeline.modelConfig == .preset7B ? "7B" : "3B"))")
    return pipeline
  }

  /// Get or lazily create the ESRGAN pipeline for the given weights directory,
  /// re-creating it when the requested model changes. Serialized under
  /// `upscalePipelineLock` like SeedVR2 to prevent concurrent double-loads.
  private func loadESRGANPipelineIfNeeded(weightsDirectory weightsDir: URL) throws -> ESRGANPipeline {
    upscalePipelineLock.lock()
    defer { upscalePipelineLock.unlock() }

    if let pipeline = esrganPipeline, pipeline.weightsDirectory.path == weightsDir.path {
      return pipeline
    }

    logger.info("WarmServer: lazy-loading ESRGAN pipeline from \(weightsDir.path)...")
    let pipeline = try ESRGANPipeline(weightsDirectory: weightsDir, logger: logger)
    esrganPipeline = pipeline
    logger.info("WarmServer: ESRGAN pipeline ready (scale=\(pipeline.config.scale)x, blocks=\(pipeline.config.numBlock))")
    return pipeline
  }

  // MARK: - Upscale Handler

  /// Handle a direct upscale request via the REST API.
  /// Lazy-loads the SeedVR2 pipeline on first call.
  private func handleUpscale(_ payload: UpscalePayload) async throws -> UpscaleResponse {
    guard let weightsPath = seedvr2WeightsPath else {
      throw WarmServerError.invalidRequest(
        message: "SeedVR2 upscale not available: no weights path configured"
      )
    }

    // Validate input file exists
    guard FileManager.default.fileExists(atPath: payload.imagePath) else {
      throw WarmServerError.invalidRequest(
        message: "Input image not found: \(payload.imagePath)"
      )
    }

    let targetResolution = payload.targetResolution ?? 1024
    let softness = payload.softness ?? 0.0

    // Resolution guard
    if let error = UpscalePayload.validateResolution(targetResolution) {
      throw WarmServerError.invalidRequest(message: error)
    }

    // Validate softness range
    if let error = UpscalePayload.validateSoftness(softness) {
      throw WarmServerError.invalidRequest(message: error)
    }

    // Model variant validation
    if let error = UpscalePayload.validateModel(payload.model) {
      throw WarmServerError.invalidRequest(message: error)
    }

    // Lazy-load pipeline
    let pipeline = try loadSeedVR2PipelineIfNeeded(weightsPath: weightsPath)

    // Check model variant matches request
    if let requestedModel = payload.model {
      let is7B = pipeline.modelConfig == .preset7B
      let requested7B = requestedModel == "seedvr2-7b"
      if is7B != requested7B {
        let loaded = is7B ? "seedvr2-7b" : "seedvr2-3b"
        throw WarmServerError.invalidRequest(
          message: "Requested \(requestedModel) but loaded weights are \(loaded)"
        )
      }
    }

    // Build warning for experimental resolutions
    let warning = UpscalePayload.resolutionWarning(for: targetResolution)

    let start = Date()

    // Resolve output path
    let resolvedOutputPath: String?
    if let op = payload.outputPath {
      resolvedOutputPath = try WarmServerOutputPathValidator
        .resolveOutputPath(op, allowedOutputDirectory: configuration.allowedOutputDirectory)
        .path
    } else {
      resolvedOutputPath = nil
    }

    let outputPath = try pipeline.upscaleAndSave(
      imagePath: payload.imagePath,
      outputPath: resolvedOutputPath,
      targetResolution: targetResolution,
      seed: payload.seed,
      softness: softness
    )

    let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
    let modelName = pipeline.modelConfig == .preset7B ? "seedvr2-7b" : "seedvr2-3b"

    // Read output image dimensions for the response
    let outputResolution = readImageDimensions(at: outputPath)
    let inputResolution = readImageDimensions(at: payload.imagePath)

    return UpscaleResponse(
      success: true,
      outputPath: outputPath,
      durationMs: durationMs,
      inputResolution: inputResolution,
      outputResolution: outputResolution,
      model: modelName,
      warning: warning
    )
  }

  /// Read image dimensions as "WxH" string. Returns "unknown" on failure.
  private func readImageDimensions(at path: String) -> String {
    guard let source = CGImageSourceCreateWithURL(
      URL(fileURLWithPath: path) as CFURL, nil
    ),
    let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
    let width = properties[kCGImagePropertyPixelWidth] as? Int,
    let height = properties[kCGImagePropertyPixelHeight] as? Int else {
      return "unknown"
    }
    return "\(width)x\(height)"
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
      case .cancelled:
        return .error(status: 409, message: "Request cancelled (queue cleared)")
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
      case .invalidOutputPath, .invalidRequest:
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

    // CivitAI checkpoint path mappings (Moody family)
    let civitaiPaths: [String: String] = [
      "moody-wild-v4": "~/Models-working/moody-wild-mix/moody-wild-v4-fp16-full.safetensors",
      "moody-wild-v4-distilled": "~/Models-working/moody-wild-mix/moody-wild-v4-distilled-10step-fp16.safetensors",
      "moody-wild-v4-fp8": "~/Models-working/moody-wild-mix/moody-wild-v4-fp8.safetensors",
      "moody-real-v6": "~/Models-working/moody-real-v6/moody-real-v6.safetensors",
      "cyberrealistic-v5": "~/Models-working/cyberrealistic-z-image/cyberrealisticZImage_v50.safetensors",
    ]
    if let path = civitaiPaths[modelId] {
      return NSString(string: path).expandingTildeInPath
    }

    let suffixes = ["-q4", "-q8", "-bf16"]
    for suffix in suffixes {
      if modelId.lowercased().hasSuffix(suffix) {
        return String(modelId.dropLast(suffix.count))
      }
    }
    return modelId
  }
}

/// Thread-safe holder for the active render's progress percent. Written from
/// the (off-actor, `@Sendable`) pipeline progress callback and read by the
/// actor's `queueStatus()` — lock-protected so it can cross the actor boundary
/// safely without an actor hop on every denoising step.
private final class RenderProgressTracker: @unchecked Sendable {
  private let lock = NSLock()
  private var percent: Int?
  func set(_ value: Int?) { lock.lock(); percent = value; lock.unlock() }
  func get() -> Int? { lock.lock(); defer { lock.unlock() }; return percent }
}

private actor WarmServerCoordinator {
  enum ServerError: Error {
    case queueFull(maxPending: Int)
    case shuttingDown
    /// The pending request was removed by a queue clear (not a server shutdown).
    case cancelled
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
  /// A queued operation tagged with identity + arrival time so the queue can
  /// be listed and individual pending jobs cancelled.
  private struct PendingJob {
    let id = UUID().uuidString
    let enqueuedAt = Date()
    /// Which client/app submitted this job (desktop, comfyui/krita, bree, api…).
    var source: String = "api"
    let operation: QueuedOperation
  }

  private var pending: [PendingJob] = []
  /// Human-readable summary of the operation the loop is currently running.
  private var activeJobSummary: String?
  /// Source/app of the currently-running job.
  private var activeJobSource: String?
  private var isProcessing = false
  /// When paused, the process loop finishes the current job (if any) but does
  /// not start pending ones until resumed.
  private var isPaused = false
  private var shuttingDown = false
  private var successfulRenderCount = 0
  private var failedRenderCount = 0
  private var lastRenderDurationMs: Int?
  private var lastError: String?
  private var activeRenderStartedAt: Date?
  /// Synthetic id for the currently-rendering job — surfaced as `current_job_id`.
  private var activeJobId: String?
  /// Handle for the in-flight render — retained so /interrupt can cancel it.
  /// The pipelines observe cancellation via Task.checkCancellation() in their
  /// denoise loops; the render's continuation then resumes with CancellationError.
  private var activeRenderTask: Task<Void, Never>?
  /// Live progress (0-100) of the active render; nil when idle. Updated from the
  /// pipeline denoising callback, read by `queueStatus()`.
  private let progressTracker = RenderProgressTracker()
  private var pipelinePrepared = false
  /// When a pool model is activated, this holds its modelSpec so that
  /// generation requests use the pool model instead of the startup
  /// configuration.modelSpec. Reset to nil when the startup model is
  /// re-activated or the pool model is unloaded.
  private var activePoolModelSpec: String?

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
    // Track the activated pool model's spec so generation requests use
    // the correct model instead of the startup configuration.modelSpec.
    activePoolModelSpec = entry.modelSpec
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
        // Pre-load full VAE for the pool-activated pipeline to avoid
        // deadlock on first img2img request (same issue as #141).
        do {
          try poolZImage.prepareFullVAE()
        } catch {
          logger.warning("Failed to pre-load full VAE for pool model '\(entry.modelSpec)': \(error)")
        }
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
    progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil,
    latentPreviewHandler: ZImagePipeline.LatentPreviewHandler? = nil,
    source: String = "api"
  ) async throws -> GenerateResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(PendingJob(source: source, operation: .generate(payload, ContinuationBox(continuation), progressHandler, latentPreviewHandler)))
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
      pending.append(PendingJob(operation: .swap(payload, ContinuationBox(continuation))))
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
      pending.append(PendingJob(operation: .controlGenerate(request, ContinuationBox(continuation))))
      startProcessingIfNeeded()
    }
  }

  /// Run a Krita model auto-switch through the FIFO render queue so the pool
  /// load/activate executes after any in-flight render finishes instead of
  /// mutating the active pipeline underneath it. The body performs the actual
  /// pool operations and returns whether a switch occurred.
  func enqueueModelSwitch(_ body: @escaping @Sendable () async throws -> Bool) async throws -> Bool {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(PendingJob(operation: .modelSwitch(body, ContinuationBox(continuation))))
      startProcessingIfNeeded()
    }
  }

  /// Enqueue a local LTX-2 video generation through the FIFO render queue so
  /// it never runs the GPU concurrently with an image render.
  func enqueueLocalVideo(_ body: @escaping @Sendable () throws -> LTX2VideoResult) async throws -> LTX2VideoResult {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(PendingJob(operation: .localVideo(body, ContinuationBox(continuation))))
      startProcessingIfNeeded()
    }
  }

  func enqueueShutdown() async throws -> ShutdownResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }

    shuttingDown = true
    return try await withCheckedThrowingContinuation { continuation in
      pending.append(PendingJob(operation: .shutdown(ContinuationBox(continuation))))
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
      model: activePoolModelSpec ?? configuration.modelSpec ?? ZImageRepository.id,
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
      currentJobId: activeJobId,
      progressPercent: progressTracker.get(),
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
      currentJobId: activeJobId,
      progressPercent: progressTracker.get(),
      renderCount: successfulRenderCount,
      failedCount: failedRenderCount
    )
  }

  /// Cancel the in-flight render, if any (ComfyUI /interrupt).
  /// Returns true if a render task was cancelled. Pending jobs are unaffected.
  func cancelActiveRender() -> Bool {
    guard let task = activeRenderTask else { return false }
    task.cancel()
    return true
  }

  /// Clear all pending jobs from the queue. Active job continues.
  func clearPending() -> Int {
    let count = pending.count
    // Cancel all pending continuations with a queue-clear error (distinct
    // from shuttingDown — the server keeps running after a queue clear).
    for job in pending {
      Self.cancel(job.operation)
    }
    pending.removeAll()
    return count
  }

  /// Cancel one pending job by id. Returns false when the id isn't queued
  /// (already running or already finished).
  func cancelPending(id: String) -> Bool {
    guard let index = pending.firstIndex(where: { $0.id == id }) else { return false }
    Self.cancel(pending[index].operation)
    pending.remove(at: index)
    return true
  }

  private static func cancel(_ operation: QueuedOperation) {
    switch operation {
    case .generate(_, let cont, _, _):
      cont.resume(throwing: ServerError.cancelled)
    case .controlGenerate(_, let cont):
      cont.resume(throwing: ServerError.cancelled)
    case .swap(_, let cont):
      cont.resume(throwing: ServerError.cancelled)
    case .modelSwitch(_, let cont):
      cont.resume(throwing: ServerError.cancelled)
    case .localVideo(_, let cont):
      cont.resume(throwing: ServerError.cancelled)
    case .shutdown(let cont):
      cont.resume(throwing: ServerError.cancelled)
    }
  }

  /// One line describing an operation for queue listings.
  private static func describe(_ operation: QueuedOperation) -> String {
    switch operation {
    case .generate(let payload, _, _, _):
      return "Render: \(payload.prompt.prefix(100))"
    case .controlGenerate(let request, _):
      return "ControlNet render: \(request.prompt.prefix(100))"
    case .swap(let payload, _):
      return "LoRA swap (\(payload.loras.count))"
    case .modelSwitch:
      return "Model switch"
    case .localVideo:
      return "LTX-2 video"
    case .shutdown:
      return "Shutdown"
    }
  }

  private static func kind(of operation: QueuedOperation) -> String {
    switch operation {
    case .generate: return "generate"
    case .controlGenerate: return "controlnet"
    case .swap: return "lora_swap"
    case .modelSwitch: return "model_switch"
    case .localVideo: return "video"
    case .shutdown: return "shutdown"
    }
  }

  /// One pending entry in a /v1/queue listing.
  struct QueueJobInfo: Sendable {
    let id: String
    let kind: String
    let summary: String
    let source: String
    let enqueuedAt: Date
  }

  /// Full queue listing for /v1/queue: the running operation plus every
  /// pending job with enough identity to cancel it.
  struct QueueSnapshot: Sendable {
    let isRendering: Bool
    let isPaused: Bool
    let activeJobId: String?
    let activeSummary: String?
    let activeSource: String?
    let activeStartedAt: Date?
    let progressPercent: Int?
    let pending: [QueueJobInfo]
    let maxPending: Int
    let renderCount: Int
    let failedCount: Int
  }

  func queueSnapshot() -> QueueSnapshot {
    QueueSnapshot(
      isRendering: activeRenderStartedAt != nil,
      isPaused: isPaused,
      activeJobId: activeJobId,
      activeSummary: activeJobSummary,
      activeSource: activeJobSource,
      activeStartedAt: activeRenderStartedAt,
      progressPercent: progressTracker.get(),
      pending: pending.map { job in
        QueueJobInfo(
          id: job.id,
          kind: Self.kind(of: job.operation),
          summary: Self.describe(job.operation),
          source: job.source,
          enqueuedAt: job.enqueuedAt
        )
      },
      maxPending: configuration.maxPendingRequests,
      renderCount: successfulRenderCount,
      failedCount: failedRenderCount
    )
  }

  // MARK: - Queue controls (pause / resume / reorder)

  func setPaused(_ paused: Bool) {
    isPaused = paused
    if !paused { startProcessingIfNeeded() }
  }

  /// Move a pending job within the queue. direction: up | down | top | bottom.
  /// Returns true if the job was found and moved.
  func movePending(id: String, direction: String) -> Bool {
    guard let idx = pending.firstIndex(where: { $0.id == id }) else { return false }
    let job = pending.remove(at: idx)
    let target: Int
    switch direction {
    case "top": target = 0
    case "bottom": target = pending.count
    case "up": target = max(0, idx - 1)
    case "down": target = min(pending.count, idx + 1)
    default: pending.insert(job, at: idx); return false
    }
    pending.insert(job, at: target)
    return true
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
      // Paused: stop pulling new jobs until resumed (setPaused restarts the loop).
      if isPaused {
        isProcessing = false
        return
      }
      guard !pending.isEmpty else {
        isProcessing = false
        return
      }

      let job = pending.removeFirst()
      activeJobSummary = Self.describe(job.operation)
      activeJobSource = job.source
      // Keep the same id the job had while pending, so clients can correlate.
      activeJobId = job.id
      defer { activeJobSummary = nil; activeJobSource = nil; activeJobId = nil }
      switch job.operation {
      case .generate(let payload, let continuation, let progressHandler, let latentPreviewHandler):
        // Run the render in a retained child task so /interrupt can cancel it
        // without cancelling the queue's processing loop.
        let renderTask = Task {
          await self.runGenerate(payload, continuation: continuation, progressHandler: progressHandler, latentPreviewHandler: latentPreviewHandler)
        }
        activeRenderTask = renderTask
        await renderTask.value
        activeRenderTask = nil
      case .controlGenerate(let request, let continuation):
        let renderTask = Task {
          await self.runControlGenerate(request, continuation: continuation)
        }
        activeRenderTask = renderTask
        await renderTask.value
        activeRenderTask = nil
      case .swap(let payload, let continuation):
        await runSwap(payload, continuation: continuation)
      case .modelSwitch(let body, let continuation):
        do {
          continuation.resume(returning: try await body())
        } catch {
          continuation.resume(throwing: error)
        }
      case .localVideo(let body, let continuation):
        // Runs on the serial queue so LTX-2 never shares the GPU with a render.
        activeRenderStartedAt = Date()
        // activeJobId is set from job.id at the top of the loop.
        defer { activeRenderStartedAt = nil; activeJobId = nil }
        do {
          continuation.resume(returning: try body())
        } catch {
          continuation.resume(throwing: error)
        }
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

  private func runGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>, progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil, latentPreviewHandler: ZImagePipeline.LatentPreviewHandler? = nil) async {
    // Queue telemetry: tag this render with a job id and stream denoising
    // progress into the tracker that queueStatus() reads. Cleared on return
    // (success or failure) via defer. flux1 forwards the wrapped handler so the
    // pipeline's per-step callback updates progress; other families currently
    // have no per-step callback, so they report only is_rendering + job id.
    // (activeJobId is set from job.id at the top of the process loop.)
    progressTracker.set(0)
    let tracker = progressTracker
    let trackedHandler: @Sendable (ZImagePipeline.GenerationProgress) -> Void = { progress in
      if progress.stage == .denoising {
        tracker.set(Int(progress.fractionCompleted * 100))
      }
      progressHandler?(progress)
    }
    defer { activeJobId = nil; progressTracker.set(nil) }

    switch currentModelFamily {
    case .chroma:
      await runChromaGenerate(payload, continuation: continuation)
    case .fibo:
      await runFiboGenerate(payload, continuation: continuation)
    case .flux2:
      await runFlux2Generate(payload, continuation: continuation)
    case .flux1:
      await runFlux1Generate(payload, continuation: continuation, progressHandler: trackedHandler, latentPreviewHandler: latentPreviewHandler)
    }
  }

  private func runFlux1Generate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>, progressHandler: (@Sendable (ZImagePipeline.GenerationProgress) -> Void)? = nil, latentPreviewHandler: ZImagePipeline.LatentPreviewHandler? = nil) async {
    activeRenderStartedAt = Date()
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runFlux1Generate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "Flux1 generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "Flux1 generation failed unexpectedly"))
      }
    }

    // When a pool model is active, override configuration.modelSpec so
    // that generateCore loads/validates the pool model, not the startup model.
    let effectiveConfig: WarmServerConfiguration
    if let poolSpec = activePoolModelSpec, poolSpec != configuration.modelSpec {
      var cfg = configuration
      cfg.modelSpec = poolSpec
      effectiveConfig = cfg
    } else {
      effectiveConfig = configuration
    }

    do {
      let outputURL: URL
      if payload.imagePath != nil {
        let img2imgRequest = try payload.makeImg2ImgRequest(
          configuration: effectiveConfig,
          activeLoRAs: activeLoRAs
        )
        outputURL = try await pipeline.generateImg2Img(img2imgRequest, progressHandler: progressHandler)
      } else {
        let request = try payload.makePipelineRequest(
          configuration: effectiveConfig,
          activeLoRAs: activeLoRAs
        )
        outputURL = try await pipeline.generateFromRequest(request, progressHandler: progressHandler, latentPreviewHandler: latentPreviewHandler)
      }
      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      resumed = true
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
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  private func runFlux2Generate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runFlux2Generate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "Flux2 generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "Flux2 generation failed unexpectedly"))
      }
    }

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
        denoise: resolvedDenoise,
        contentMode: payload.contentMode
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

      resumed = true
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
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  private func runFiboGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runFiboGenerate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "FIBO generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "FIBO generation failed unexpectedly"))
      }
    }

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
        levelsMax: payload.levelsMax ?? 1.0,
        contentMode: payload.contentMode
      )

      let result = try await fp.generate(fiboRequest, progressHandler: nil)

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      resumed = true
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
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  private func runChromaGenerate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runChromaGenerate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "Chroma generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "Chroma generation failed unexpectedly"))
      }
    }

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

      // Run the synchronous Chroma render off the actor (the static helper is
      // nonisolated, so it executes on the global concurrent executor). This
      // mirrors the flux2/fibo paths, which await pipeline work without
      // blocking the actor — keeping /health, /queue, and progress telemetry
      // responsive for the duration of the render.
      try await Self.renderChroma(
        pipeline: pipeline,
        tokenizer: tokenizer,
        payload: payload,
        outputURL: outputURL,
        loras: activeLoRAs
      )

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      activeRenderStartedAt = nil

      resumed = true
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
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  /// Perform the synchronous Chroma pipeline render. Static (hence nonisolated)
  /// and async, so it runs on the global concurrent executor rather than on
  /// the coordinator actor — a Chroma render would otherwise block /health,
  /// /queue, and progress telemetry for its full duration.
  private static func renderChroma(
    pipeline: ChromaPipeline,
    tokenizer: ChromaTokenizer,
    payload: GeneratePayload,
    outputURL: URL,
    loras: [LoRAConfiguration]
  ) async throws {
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

    // Save image (with embedded, Finder-readable generation metadata)
    try QwenImageIO.saveImage(array: imageArray, to: outputURL,
      metadata: .generation(prompt: payload.prompt, negativePrompt: payload.negativePrompt,
        seed: seed, steps: steps, guidance: guidance, width: width, height: height,
        generatedBy: payload.source, contentMode: payload.contentMode, loras: loras))
  }

  private func runControlGenerate(_ request: ZImageControlGenerationRequest, continuation: ContinuationBox<GenerateResponse>) async {
    if currentModelFamily == .flux2 || currentModelFamily == .fibo || currentModelFamily == .chroma {
      continuation.resume(throwing: WarmServerError.controlNetNotSupported)
      return
    }

    activeRenderStartedAt = Date()
    let start = Date()

    var resumed = false

    defer {
      if !resumed {
        logger.error("runControlGenerate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "ControlNet generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "ControlNet generation failed unexpectedly"))
      }
    }

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

      resumed = true
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
      resumed = true
      continuation.resume(throwing: error)
    }
  }

  private func runSwap(_ payload: LoRASwapPayload, continuation: ContinuationBox<LoRASwapResponse>) async {
    if currentModelFamily == .fibo || currentModelFamily == .chroma {
      continuation.resume(throwing: WarmServerError.loraSwapNotSupported)
      return
    }

    var resumed = false

    defer {
      if !resumed {
        logger.error("runSwap: continuation was not resumed — likely a crash in LoRA application. Resuming with error.")
        if currentModelFamily == .flux2 {
          activeLoRAs = flux2Pipeline?.loadedLoRAConfigs ?? []
        } else {
          activeLoRAs = pipeline.loadedLoRAConfigs
        }
        lastError = "LoRA swap failed unexpectedly (continuation not resumed)"
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "LoRA swap failed unexpectedly"))
      }
    }

    do {
      let newLoRAs = try payload.makeConfigurations()

      if currentModelFamily == .flux2 {
        // Flux 2 LoRA swap via Flux2Pipeline.loadLoRAs()
        guard let f2 = flux2Pipeline else {
          resumed = true
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
      resumed = true
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
      resumed = true
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
    if (request.path == "/ws" || request.path == "/api/ws"), request.method == "GET" {
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

  static func empty(status: Int) -> HTTPResponse {
    HTTPResponse(status: status, reasonPhrase: reasonPhrase(for: status), contentType: "application/json", body: Data())
  }

  static func error(status: Int, message: String) -> HTTPResponse {
    json(status: status, payload: ErrorPayload(success: false, error: message))
  }

  func serialize() -> Data {
    var data = Data()
    // No CORS headers: all known clients (desktop app, Krita plugin, Telegram
    // bot, MCP) are native, so browser cross-origin access is intentionally
    // not enabled.
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
    case 204: return "No Content"
    case 200: return "OK"
    case 400: return "Bad Request"
    case 404: return "Not Found"
    case 405: return "Method Not Allowed"
    case 409: return "Conflict"
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

struct GeneratePayload: Sendable {
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
  var imagePath: String?   // var: may be filled in from initImageData (bytes upload)
  /// Img2img init image sent as base64 (init_image_base64) — for remote clients
  /// that can't put a file on the server's filesystem. Decoded to a temp file.
  let initImageData: Data?
  let imageStrength: Float?
  let creativity: Float?

  /// Submitting client/app (desktop, bree, api…) — for queue attribution.
  let source: String?

  /// Fruit mode (neutral | banana | avocado) — stamped into render metadata.
  let contentMode: String?

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
    imagePath: String? = nil, imageStrength: Float? = nil, creativity: Float? = nil,
    source: String? = nil, contentMode: String? = nil, initImageData: Data? = nil
  ) {
    self.source = source
    self.contentMode = contentMode
    self.initImageData = initImageData
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
    case denoise, maskGrow, maskFeather
    // NOTE: the /v1/generate decoder uses .convertFromSnakeCase, which rewrites
    // incoming keys to camelCase BEFORE matching CodingKey stringValues. So the
    // wire keys inpaint_image_base64 / mask_base64 arrive as these camelCase
    // forms — the rawValues MUST be the post-conversion spelling, not snake_case.
    case inpaintImageData = "inpaintImageBase64"
    case maskImageData = "maskBase64"
    case cfg, firstNStepsWithoutCFG
    case imagePath, imageStrength, creativity
    case source
    case contentMode
    // Wire key init_image_base64 arrives as this camelCase form after
    // .convertFromSnakeCase (same gotcha as the inpaint keys).
    case initImageData = "initImageBase64"
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
    // Inpaint image + mask arrive as base64 strings from the HTTP API.
    inpaintImageData = (try c.decodeIfPresent(String.self, forKey: .inpaintImageData))
        .flatMap { Data(base64Encoded: $0) }
    maskData = (try c.decodeIfPresent(String.self, forKey: .maskImageData))
        .flatMap { Data(base64Encoded: $0) }
    denoise = try c.decodeIfPresent(Float.self, forKey: .denoise)
    maskGrow = try c.decodeIfPresent(Int.self, forKey: .maskGrow)
    maskFeather = try c.decodeIfPresent(Int.self, forKey: .maskFeather)
    maskCropX = nil
    maskCropY = nil
    cfg = try c.decodeIfPresent(Float.self, forKey: .cfg)
    firstNStepsWithoutCFG = try c.decodeIfPresent(Int.self, forKey: .firstNStepsWithoutCFG)
    imagePath = try c.decodeIfPresent(String.self, forKey: .imagePath)
    initImageData = (try c.decodeIfPresent(String.self, forKey: .initImageData))
        .flatMap { Data(base64Encoded: $0) }
    imageStrength = try c.decodeIfPresent(Float.self, forKey: .imageStrength)
    creativity = try c.decodeIfPresent(Float.self, forKey: .creativity)
    source = try c.decodeIfPresent(String.self, forKey: .source)
    contentMode = try c.decodeIfPresent(String.self, forKey: .contentMode)
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
      source: source,
      contentMode: contentMode,
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
    } else if let imageStrength {
      resolvedStrength = imageStrength
      specifiedAs = .strength
    } else if let denoise {
      resolvedStrength = 1.0 - max(0.01, min(0.99, denoise))
      specifiedAs = .denoise
    } else {
      resolvedStrength = 0.3
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
      specifiedAs: specifiedAs,
      contentMode: contentMode,
      source: source
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
    case "dpmpp_2m":
      return .dpmplusplus2m
    case "dpmpp_2s_ancestral":
      return .dpmplusplus2sa
    default:
      return SchedulerKind(rawValue: rawValue) ?? .euler
    }
  }

  private static func parseSigmaScheduleKind(_ rawValue: String?) -> SigmaScheduleKind {
    guard let rawValue else { return .flow }
    switch rawValue {
    case "beta57":
      return .beta57
    case "normal", "simple", "sgm_uniform", "ddim_uniform":
      // ComfyUI schedule names that map to the model's native flow-matching schedule.
      return .flow
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
      // Default to the gallery folder, NOT temp — otherwise renders from clients
      // that omit outputPath (e.g. HTTP/MCP pipelines) land in /var/folders/T and
      // are silently purged by macOS. Fall back to temp only if the gallery dir
      // can't be created.
      let dir = (configuration.allowedOutputDirectory as NSString).expandingTildeInPath
      let created = (try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)) != nil
      if created || FileManager.default.fileExists(atPath: dir) {
        return URL(fileURLWithPath: dir).appendingPathComponent(defaultFilename)
      }
      return URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(defaultFilename)
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

// MARK: - Upscale Payload & Response

struct UpscalePayload: Decodable, Sendable {
  let imagePath: String
  let targetResolution: Int?
  let seed: Int?
  let softness: Float?
  let outputPath: String?
  let model: String?   // "seedvr2-3b" or "seedvr2-7b"

  /// Validate target resolution. Returns an error message if invalid, nil if valid.
  static func validateResolution(_ resolution: Int) -> String? {
    guard resolution >= 256 && resolution <= 2048 else {
      return "target_resolution must be between 256 and 2048"
    }
    return nil
  }

  /// Validate softness. Returns an error message if invalid, nil if valid.
  static func validateSoftness(_ softness: Float) -> String? {
    guard softness >= 0.0 && softness <= 1.0 else {
      return "softness must be between 0.0 and 1.0"
    }
    return nil
  }

  /// Validate model variant. Returns an error message if invalid, nil if valid.
  static func validateModel(_ model: String?) -> String? {
    guard let model = model else { return nil }
    guard model == "seedvr2-3b" || model == "seedvr2-7b" else {
      return "Invalid model: '\(model)'. Must be 'seedvr2-3b' or 'seedvr2-7b'."
    }
    return nil
  }

  /// Return a warning string if resolution is experimental (>1024), nil otherwise.
  static func resolutionWarning(for resolution: Int) -> String? {
    resolution > 1024
      ? "target_resolution \(resolution) is experimental and may cause OOM errors. Safe maximum is 1024."
      : nil
  }
}

struct UpscaleResponse: Encodable, Sendable {
  let success: Bool
  let outputPath: String
  let durationMs: Int
  let inputResolution: String     // e.g. "512x512"
  let outputResolution: String    // e.g. "1024x1024"
  let model: String               // "seedvr2-3b" or "seedvr2-7b"
  let warning: String?            // non-nil if target_resolution > 1024
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

  /// Allowed range for LoRA scales — finite values outside are clamped.
  private static let scaleRange: ClosedRange<Float> = -10.0...10.0

  /// Validate the requested scale: reject non-finite values, clamp finite
  /// values to `scaleRange`. Defaults to 1.0 when absent.
  private func resolvedScale() throws -> Float {
    guard let scale else { return 1.0 }
    guard scale.isFinite else {
      throw WarmServerError.invalidRequest(
        message: "Invalid LoRA scale for '\(path)': must be a finite number"
      )
    }
    return min(max(scale, Self.scaleRange.lowerBound), Self.scaleRange.upperBound)
  }

  func makeConfiguration() throws -> LoRAConfiguration {
    let clampedScale = try resolvedScale()
    let expanded = (path as NSString).expandingTildeInPath

    // Direct path (absolute, relative, tilde-expanded)
    if path.hasPrefix("/") || path.hasPrefix("./") || path.hasPrefix("../") || path.hasPrefix("~")
       || FileManager.default.fileExists(atPath: expanded) {
      return .local(expanded, scale: clampedScale)
    }

    // Library resolution: search the LoRA library root for the filename
    let libraryRoot = ("~/Models/loras" as NSString).expandingTildeInPath
    let fm = FileManager.default
    if let enumerator = fm.enumerator(at: URL(fileURLWithPath: libraryRoot),
                                       includingPropertiesForKeys: [.isRegularFileKey]) {
      for case let fileURL as URL in enumerator {
        if fileURL.lastPathComponent == path {
          return .local(fileURL.path, scale: clampedScale)
        }
      }
    }

    // HuggingFace fallback
    return .huggingFace(path, scale: clampedScale)
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
  /// Synthetic id of the currently-rendering job — `current_job_id` on the wire.
  let currentJobId: String?
  /// Live progress (0-100) of the active render — `progress_percent` on the wire.
  let progressPercent: Int?
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
  case generate(GeneratePayload, ContinuationBox<GenerateResponse>, (@Sendable (ZImagePipeline.GenerationProgress) -> Void)?, ZImagePipeline.LatentPreviewHandler?)
  case controlGenerate(ZImageControlGenerationRequest, ContinuationBox<GenerateResponse>)
  case swap(LoRASwapPayload, ContinuationBox<LoRASwapResponse>)
  case modelSwitch(@Sendable () async throws -> Bool, ContinuationBox<Bool>)
  /// Local LTX-2 video generation, run through the queue so it serializes with
  /// image renders on the shared GPU. The closure captures the generator+request.
  case localVideo(@Sendable () throws -> LTX2VideoResult, ContinuationBox<LTX2VideoResult>)
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
  case invalidRequest(message: String)
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
    case .invalidRequest(let message):
      return message
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
