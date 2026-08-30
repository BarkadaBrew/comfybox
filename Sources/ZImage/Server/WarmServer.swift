import Foundation
import Dispatch
import Logging
import Network
import Darwin
import MLX
import CoreGraphics
import ImageIO

/// #1479: refuse a preemption when finishing beats preempting. INERT until
/// telemetry has samples for both `meanStepSec` AND `evictReloadRoundTripSec`
/// — it never refuses on a guess (spec: "until telemetry has samples for a
/// family, the guard is inert rather than refusing"). `remainingPhaseMeansSec`
/// is the observed mean duration of each phase still ahead of the render's
/// current position (e.g. decode/vocoder/postProcess when mid-denoise) —
/// phases with no samples yet are simply omitted by the caller, not zeroed.
///
/// Returns nil to allow the preemption; a non-nil value is the projected
/// remaining seconds (the ETA a refused caller can report).
func preemptionRefusalETA(
  stepsRemaining: Int, meanStepSec: Double?,
  remainingPhaseMeansSec: [Double],
  evictReloadRoundTripSec: Double?
) -> Double? {
  guard let stepSec = meanStepSec, let roundTrip = evictReloadRoundTripSec else { return nil }
  let projected = Double(stepsRemaining) * stepSec + remainingPhaseMeansSec.reduce(0, +)
  return projected < roundTrip ? projected : nil
}

public struct WarmServerConfiguration: Sendable {
  public var port: UInt16
  public var modelSpec: String?
  public var textEncoderPath: String?
  public var initialLoRAs: [LoRAConfiguration]
  public var forceTransformerOverrideOnly: Bool
  public var maxSequenceLength: Int
  public var maxPendingRequests: Int
  /// Separate cap for MUTATING pool operations waiting in the FIFO
  /// (`/v1/model/load|activate|unload`), counted independently of
  /// `maxPendingRequests` (WP-E8 review, finding 1).
  ///
  /// Model ops deliberately do not sit under the render capacity gate — a
  /// paused engine full of parked renders must still accept the operator
  /// action that frees the GPU. "Bounded in number" was an assumption about
  /// the caller, though, and these routes are unauthenticated: repeated
  /// `wait: false` loads could grow the queue without limit. This is the
  /// bound that makes the sentence true.
  public var maxPendingModelOps: Int
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
  /// Default LoRA ("path" or "path@scale") merged into every LOCAL video
  /// render when the request carries none — lets preset-only callers (daemon
  /// MCP) get e.g. a distill LoRA required by a non-distilled checkpoint.
  public var ltx2DefaultLoRA: String?
  /// Explicit CivitAI API key — the top tier of `CivitAISecrets.resolve`'s
  /// resolution order (--civitai-key flag > CIVITAI_API_KEY env > Keychain).
  /// nil here just means "no explicit override"; the /v1/civitai/* routes
  /// still fall through to env/Keychain before giving up (#234).
  public var civitaiApiKey: String?

  public init(
    port: UInt16 = ComfyBoxServerConfig.canonicalPort,
    modelSpec: String? = nil,
    textEncoderPath: String? = nil,
    initialLoRAs: [LoRAConfiguration] = [],
    forceTransformerOverrideOnly: Bool = false,
    maxSequenceLength: Int = 512,
    maxPendingRequests: Int = 10,
    maxPendingModelOps: Int = 8,
    allowedOutputDirectory: String = FileManager.default.currentDirectoryPath,
    seedvr2WeightsPath: String? = nil,
    ltx2WeightsPath: String? = nil,
    ltx2GemmaPath: String? = nil,
    ltx2DefaultLoRA: String? = nil,
    civitaiApiKey: String? = nil
  ) {
    self.port = port
    self.modelSpec = modelSpec
    self.textEncoderPath = textEncoderPath
    self.initialLoRAs = initialLoRAs
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
    self.maxSequenceLength = maxSequenceLength
    self.maxPendingRequests = max(1, maxPendingRequests)
    self.maxPendingModelOps = max(1, maxPendingModelOps)
    self.allowedOutputDirectory = allowedOutputDirectory
    self.seedvr2WeightsPath = seedvr2WeightsPath
    self.ltx2WeightsPath = ltx2WeightsPath
    self.ltx2GemmaPath = ltx2GemmaPath
    self.ltx2DefaultLoRA = ltx2DefaultLoRA
    self.civitaiApiKey = civitaiApiKey
  }
}

/// Model family used by the warm server to route generation to the correct pipeline.
enum WarmModelFamily: String, Sendable, CaseIterable {
  case flux1
  case flux2
  case fibo
  case chroma
  case krea2
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
  /// Submit/poll tracker for async image generation (GH: queue-submit —
  /// see ImageJobTracker's doc comment for the incident that motivated it).
  private let imageJobTracker = ImageJobTracker()
  /// Tracks async-submitted LOCAL LTX-2 video jobs (submit → 202 + jobId, poll
  /// GET /v1/video/status/{id}). Mirrors `imageJobTracker` so a multi-minute
  /// local render never holds an HTTP connection open. The Replicate cloud path
  /// keeps its own tracker inside `replicateVideoProxy`.
  private let videoJobTracker = VideoJobTracker()
  private let renderTraceStore = RenderTraceStore()
  let comfyBridge: ComfyBridge

  /// Imported ComfyUI workflows (#238), file-backed at ~/.comfybox/workflows/.
  let workflowStore = WorkflowStore()
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
  /// Held in a shared, lock-based box so the coordinator can evict it before an
  /// image load — image + video cannot co-reside in unified memory (#218).
  let videoHolder = VideoGeneratorHolder()

  // MARK: - #1479 preemption support
  //
  // Lock-based (see the block comment above `RollingMeanSec`), installed on
  // every LTX-2 generator instance (fresh or reused) in `prepareLocalVideo`,
  // so a video render is ALWAYS preemptible-capable but never actually
  // preempted unless a job raises `ltx2PreemptionSignal`.

  /// Per-phase render timings feeding the refusal guard and `/v1/queue`.
  let ltx2Telemetry = LTX2PhaseTelemetry()
  /// Raised by an image job's route handler to checkpoint the in-flight video
  /// render; read inside the render loop with no actor hop.
  let ltx2PreemptionSignal = PreemptionSignal()
  /// Live steps-remaining of the in-flight video render (fed from its
  /// progress callback) — the refusal guard's `stepsRemaining` input.
  let ltx2StepPosition = LTX2StepPosition()
  /// Observed evict/reload durations from past preemption episodes — the
  /// refusal guard's `evictReloadRoundTripSec` input (nil until both have a
  /// sample).
  let ltx2EvictMean = RollingMeanSec()
  let ltx2ReloadMean = RollingMeanSec()
  /// Exactly one preemption in flight at a time — a preemptor cannot itself
  /// be preempted (spec).
  let preemptionInFlight = LockedFlag()
  /// Single-slot mailbox bridging the route handler (raises the signal, then
  /// awaits a continuation) to the coordinator's `.localVideo` case (observes
  /// the yield, runs the image job, resumes the video) — see the mechanism
  /// note above `RollingMeanSec`.
  let pendingPreemptorBox = PendingPreemptorBox()

  /// Unified-memory pressure monitor (#218). On warning/critical it sheds the
  /// MLX buffer cache and any idle heavy model to stay clear of jetsam.
  private var memoryPressureSource: DispatchSourceMemoryPressure?
  /// Auto-rescans the LoRA library on any external filesystem change (CivitAI
  /// browser download, curl, cp, an MCP/Bree fetch) so new LoRAs are indexed
  /// without a manual `lora scan`. Started in `run()` once `loraLibrary` exists.
  private var loraLibraryWatcher: LoRALibraryWatcher?
  /// Lock-based health snapshot the coordinator publishes to, so GET /health is
  /// served without hopping onto the actor — stays responsive during a render (#217).
  private let liveHealth = LiveHealthState()
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
    self.coordinator = WarmServerCoordinator(
      configuration: configuration, logger: logger, videoHolder: self.videoHolder, liveHealth: self.liveHealth,
      videoJobTracker: self.videoJobTracker, ltx2Telemetry: self.ltx2Telemetry,
      ltx2PreemptionSignal: self.ltx2PreemptionSignal, ltx2StepPosition: self.ltx2StepPosition,
      ltx2EvictMean: self.ltx2EvictMean, ltx2ReloadMean: self.ltx2ReloadMean,
      preemptionInFlight: self.preemptionInFlight, pendingPreemptorBox: self.pendingPreemptorBox)
    self.seedvr2WeightsPath = configuration.seedvr2WeightsPath

    self.comfyBridge = ComfyBridge(logger: logger)

    // Initialize the LoRA Library. The library root defaults to ~/Models/loras/
    // (via COMFYBOX_MODELS env or LoRALibrary default). If no library.json exists,
    // the first API call to /v1/loras/scan will create it.
    do {
      let library = try LoRALibrary(logger: logger)
      self.loraLibrary = library
      Task { await coordinator.setLoraLibrary(library) }

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

    // Task #19: lifecycle traces. Recovery first — any trace left open by a
    // crash/kill is marked abandoned so failed renders are never invisible.
    videoJobTracker.traceStore = renderTraceStore
    renderTraceStore.markAbandonedOpenTraces()

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

    recoverPersistedQueue()

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

    // Same cleanup for async image generation jobs (queue-submit).
    let imageJobPruneTimer = DispatchSource.makeTimerSource(queue: listenerQueue)
    imageJobPruneTimer.schedule(deadline: .now() + 600, repeating: 600)
    imageJobPruneTimer.setEventHandler { [weak self] in
      self?.imageJobTracker.pruneCompleted()
      self?.videoJobTracker.pruneCompleted()
    }
    imageJobPruneTimer.resume()

    // Unified-memory pressure guard (#218). Loading LTX-2 outside the pool used
    // to co-reside with an image model and trip OS_REASON_JETSAM. As a
    // last-resort backstop to the single-heavy-model residency logic, when the
    // kernel signals memory pressure we drop the MLX buffer cache and release
    // any idle heavy model (a resident-but-not-rendering LTX-2 stack first,
    // then the LRU inactive image model) — never disturbing an in-flight render.
    let pressureSource = DispatchSource.makeMemoryPressureSource(
      eventMask: [.warning, .critical], queue: listenerQueue)
    pressureSource.setEventHandler { [weak self] in
      guard let self else { return }
      let event = pressureSource.data
      let level = event.contains(.critical) ? "critical" : "warning"
      self.logger.warning("Memory pressure: \(level) — shedding caches/idle heavy models (#218)")
      // Always drop the MLX buffer cache; cheap and often enough.
      GPU.clearCache()
      // Release an idle (not mid-render) LTX-2 stack — the single biggest chunk.
      if self.videoHolder.releaseIfIdle() {
        self.logger.warning("Memory pressure: released idle LTX-2 video stack")
      }
      // On critical, also shed the LRU inactive image model from the pool.
      if event.contains(.critical) {
        Task { [weak self] in
          guard let self else { return }
          let freed = await self.coordinator.shedInactivePoolModelUnderPressure()
          if freed > 0 {
            self.logger.warning("Memory pressure: evicted LRU inactive image model (~\(freed)MB)")
          }
        }
      }
    }
    pressureSource.resume()
    self.memoryPressureSource = pressureSource

    if let library = loraLibrary {
      self.loraLibraryWatcher = LoRALibraryWatcher(library: library, queue: listenerQueue, logger: logger)
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
      if case .posix(.EADDRINUSE) = error {
        // The most common cause on this machine: com.barkadabrew.comfybox is a
        // KeepAlive=true launchd agent, so killing a manually-started `serve`
        // process's port-holder — or even just a stray manual `serve` left
        // running — gets silently re-occupied within ~5s (ThrottleInterval).
        // Point at the actual fix instead of a bare "address in use" (GH #153).
        fputs("""
        Port \(self.configuration.port) is already in use.

        If you're trying to run a manual/dev server, com.barkadabrew.comfybox \
        (a launchd agent with KeepAlive) may have respawned onto this port. \
        Stop it first:
          launchctl bootout gui/\(getuid())/com.barkadabrew.comfybox
        Then restart it later with:
          launchctl bootstrap gui/\(getuid()) ~/Library/LaunchAgents/com.barkadabrew.comfybox.plist

        To restart the managed server normally (rebuild + relaunch in place),
        use scripts/deploy-server.sh instead of killing the process directly.

        """, stderr)
      }
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
      // #300: pin QoS explicitly. Without it the queue runs at whatever QoS
      // is donated by whoever schedules onto it, which during a render is
      // the coordinator's `.utility` render work — demoting control/HTTP
      // responses on this connection right when they need to stay responsive.
      queue: DispatchQueue(label: "z-image.warm-server.connection.\(UUID().uuidString)", qos: .userInitiated),
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
      // #217: read from the lock-based snapshot instead of `await
      // coordinator.health()`. The coordinator actor is blocked for a whole
      // synchronous render, so awaiting it made /health hang (HTTP 000) for the
      // render's duration. The snapshot is published on every state transition.
      let health = liveHealthResponse(memoryBytes: memoryBytes)
      if let data = Self.healthJSON(
        health, videoAvailable: replicateVideoProxy != nil,
        activeVideoJobs: replicateVideoProxy?.activeJobCount ?? 0) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .json(status: 200, payload: health)

    case ("POST", "/v1/generate"):
      do {
        let payload = try decodedGeneratePayload(from: request.body)
        let source = payload.source ?? "api"
        // #1479: absent/false `preempt` (or no video rendering, or a nested
        // attempt) is `.notApplicable` — same call as before this feature.
        switch await attemptPreemption(payload, source: source, rawBody: request.body) {
        case .notApplicable:
          let result = try await coordinator.enqueueGenerate(payload, source: source, rawBody: request.body)
          return .json(status: 200, payload: result)
        case .ran(let result):
          return .json(status: 200, payload: result)
        case .ranFailed(let error):
          return .error(response(for: error))
        case .refused(let eta):
          let result = try await coordinator.enqueueGenerate(payload, source: source, rawBody: request.body)
          let stamped = GenerateResponse(
            success: result.success, outputPath: result.outputPath, durationMs: result.durationMs,
            preemptRefused: true, etaSec: eta, applied: result.applied)
          return .json(status: 200, payload: stamped)
        }
      } catch {
        return .error(response(for: error))
      }

    // Queue-submit: returns a job id immediately instead of blocking the HTTP
    // connection for the whole render. Poll GET /v1/generate/status/{id} for
    // completion — same convention as /v1/video/generate + /v1/video/status.
    // Built after a render's Telegram delivery was orphaned by a blocking
    // /v1/generate call outliving the caller's own turn timeout.
    case ("POST", "/v1/generate/async"):
      do {
        let payload = try decodedGeneratePayload(from: request.body)
        let source = payload.source ?? "api"
        // #1479: `submitPreempting` runs the SAME `attemptPreemption` check
        // inside the job's own detached Task, so a `preempt`-absent/false
        // submit takes the exact same `coordinator.enqueueGenerate` path as
        // before this feature.
        let status = imageJobTracker.submitPreempting(
          payload, source: source, coordinator: coordinator, rawBody: request.body,
          preemptor: { [weak self] jobId in
            await self?.attemptPreemption(payload, source: source, rawBody: request.body, jobId: jobId) ?? .notApplicable
          })
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(status)
        return .json(.rawJSON(status: 202, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("GET", _) where request.path.hasPrefix("/v1/generate/status/"):
      let jobId = String(request.path.dropFirst("/v1/generate/status/".count))
      guard !jobId.isEmpty else {
        return .error(.error(status: 400, message: "Missing job_id in path"))
      }
      guard let status = imageJobTracker.status(jobId: jobId) else {
        return .error(.error(status: 404, message: "Image job not found: \(jobId)"))
      }
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      let data = try? encoder.encode(status)
      return .json(.rawJSON(status: 200, data: data ?? Data()))

    case ("GET", "/v1/generate/preview"):
      guard let frame = await coordinator.latestPreviewFrame() else {
        return .json(.empty(status: 204))
      }
      return .json(.binary(status: 200, contentType: "image/jpeg", data: frame))

    case ("POST", "/v1/lora/swap"):
      do {
        var payload = try decode(LoRASwapPayload.self, from: request.body)
        payload = stageNearlineLoras(in: payload)
        let result = try await coordinator.enqueueSwap(payload, rawBody: request.body)
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

        // K-FIX-1 / Codex C2: every MUTATING pool operation goes through the
        // same FIFO as renders, LoRA swaps and the ComfyBridge model switch.
        // Calling `coordinator.poolLoad` here was not serialized: actor
        // isolation does not hold across an await, so the load's eviction and
        // `GPU.clearCache()` could run under an in-flight render.
        let operation = ModelOperation.load(
          modelSpec: resolvedSpec, quantization: resolvedQuantization, activate: shouldActivate)
        if shouldWait {
          guard case .load(let result) = try await coordinator.enqueueModelOperation(operation) else {
            return .error(.error(status: 500, message: "Model load returned the wrong result kind"))
          }
          return .json(status: 200, payload: result)
        } else {
          // Fire-and-forget is now a TRACKED queue job, not a detached Task:
          // it is listed in /v1/queue under the id returned here, it can be
          // cancelled, and it still cannot begin under a render.
          let jobId = try await coordinator.enqueueModelOperationDetached(operation)
          let ack = ModelLoadResponse(
            // "loading" is kept verbatim: an out-of-repo client (the daemons)
            // may branch on it, and C2's wire change is the ADDED `job_id`.
            status: "loading",
            model: payload.model,
            family: "pending",
            loadTimeMs: 0,
            vramEstimateMB: 0,
            poolSize: await coordinator.modelPool.count(),
            poolBudgetMB: await coordinator.modelPool.budget(),
            jobId: jobId
          )
          return .json(status: 202, payload: ack)
        }
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/model/activate"):
      do {
        let payload = try decode(ModelActivateRequest.self, from: request.body)
        // C2: activation swaps the resident pipeline out from under whatever
        // is rendering unless it is queued behind it.
        guard case .activate(let result) =
          try await coordinator.enqueueModelOperation(.activate(modelId: payload.model))
        else {
          return .error(.error(status: 500, message: "Model activate returned the wrong result kind"))
        }
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
        // C2: an unload releases the pipeline's weights — the same
        // use-after-release hazard as an eviction.
        guard case .unload(let result) =
          try await coordinator.enqueueModelOperation(.unload(modelId: payload.model))
        else {
          return .error(.error(status: 500, message: "Model unload returned the wrong result kind"))
        }
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
      // #217 pattern applied here too: `await coordinator.activeLoRAIdentifiers`
      // hopped onto the coordinator ACTOR, which stays occupied for the whole of
      // a synchronous render — so listing the LoRA library hung with HTTP 000
      // until the render finished, and any UI listing from this route rendered
      // an EMPTY list (observed 2026-08-10). The catalog is static data and must
      // never depend on GPU state; only the "currently loaded" decoration did.
      // Read that from the same lock-based snapshot /health uses.
      let activeLoRANames = liveHealth.read().0.loras.map { st in
        ((st.source as NSString).lastPathComponent as NSString).deletingPathExtension
      }
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

    case ("POST", "/v1/loras/import"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      guard let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any],
            let sourcePath = json["path"] as? String, !sourcePath.isEmpty
      else {
        return .error(.error(status: 400, message: "Missing 'path'"))
      }
      let category = (json["category"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "vault"
      do {
        let entry = try library.importFile(from: sourcePath, category: category)
        let responseDict: [String: Any] = [
          "success": true,
          "id": entry.id,
          "filename": entry.filename,
          "model_compatibility": entry.modelCompatibility,
          "triggerwords": entry.triggerwords,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: responseDict) {
          return .json(.rawJSON(status: 200, data: data))
        }
        return .error(.error(status: 500, message: "Failed to serialize imported entry"))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: "Import failed: \(error.localizedDescription)"))
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

    case ("POST", _) where request.path.hasSuffix("/update") && request.path.hasPrefix("/v1/loras/"):
      guard let library = loraLibrary else {
        return .error(.error(status: 503, message: "LoRA Library not initialized"))
      }
      let id = String(request.path.dropFirst("/v1/loras/".count).dropLast("/update".count))
      guard !id.isEmpty, !id.contains("/") else {
        return .error(.error(status: 400, message: "Invalid LoRA ID"))
      }
      guard let json = try? JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
        return .error(.error(status: 400, message: "Invalid request body"))
      }
      // WP-E6: `krea2_relative` is declared by the user — an unknown value
      // is a 400, never silently dropped.
      var krea2Relative: Krea2Variant?
      if let raw = json["krea2_relative"] {
        guard let str = raw as? String, let parsed = Krea2Variant(rawValue: str) else {
          return .error(.error(
            status: 400,
            message: "Invalid krea2_relative '\(raw)': expected one of \(Krea2Variant.allCases.map(\.rawValue))"))
        }
        krea2Relative = parsed
      }
      let patch = LoRAEntryPatch(
        triggerwords: json["triggerwords"] as? [String],
        recommendedScale: (json["recommended_scale"] as? NSNumber)?.floatValue,
        scaleRange: (json["scale_range"] as? [NSNumber])?.map { $0.floatValue },
        tags: json["tags"] as? [String],
        notes: json["notes"] as? String,
        sourceURL: json["source_url"] as? String,
        civitaiModelId: json["civitai_model_id"] as? Int,
        krea2Relative: krea2Relative
      )
      do {
        try library.update(id, patch: patch)
        guard let entry = library.entry(for: id) else {
          return .error(.error(status: 404, message: "LoRA not found: \(id)"))
        }
        let responseDict: [String: Any] = [
          "success": true,
          "id": entry.id,
          "triggerwords": entry.triggerwords,
          "recommended_scale": entry.recommendedScale,
          "tags": entry.tags,
          "notes": entry.notes,
        ]
        if let data = try? JSONSerialization.data(withJSONObject: responseDict) {
          return .json(.rawJSON(status: 200, data: data))
        }
        return .error(.error(status: 500, message: "Failed to serialize updated entry"))
      } catch let error as LoRALibraryError {
        return .error(.error(status: 404, message: error.localizedDescription))
      } catch {
        return .error(.error(status: 500, message: error.localizedDescription))
      }

    // MARK: - Video Endpoints

    // Montage compositor (#232): assemble images (ken-burns) + clips into one
    // MP4 with transitions. Memory-light editorial motion — no LTX-2, no heavy
    // -model admission gate, runs alongside a resident video model. Sync by
    // design: compositing a <30s montage takes seconds.
    case ("POST", "/v1/montage/compose"):
      do {
        let payload = try decode(MontagePayload.self, from: request.body)
        let result = try await composeMontage(payload)
        return .json(status: 200, payload: MontageResponse(
          outputPath: result.outputPath,
          durationS: result.durationS,
          width: result.width,
          height: result.height,
          segmentCount: result.segmentCount,
          frameCount: result.frameCount))
      } catch {
        return .error(response(for: error))
      }

    // Workflow import/run (#238): ComfyUI API-format workflows as first-class
    // stored objects, executed through the existing ComfyBridge parser/executor.
    case ("POST", "/v1/workflows/import"):
      do {
        return try await handleWorkflowImport(body: request.body)
      } catch {
        return .error(response(for: error))
      }

    case ("GET", "/v1/workflows"):
      let items = workflowStore.list().map { $0.summaryJSON() }
      if let data = try? JSONSerialization.data(withJSONObject: ["workflows": items]) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize workflow list"))

    case ("DELETE", _) where request.path.hasPrefix("/v1/workflows/"):
      let id = String(request.path.dropFirst("/v1/workflows/".count))
      guard workflowStore.delete(id) else {
        return .error(.error(status: 404, message: "Workflow not found: \(id)"))
      }
      return .json(status: 200, payload: ["deleted": id])

    case ("POST", _) where request.path.hasPrefix("/v1/workflows/") && request.path.hasSuffix("/run"):
      let id = String(request.path.dropFirst("/v1/workflows/".count).dropLast("/run".count))
      do {
        return try await handleWorkflowRun(id: id, body: request.body)
      } catch {
        return .error(response(for: error))
      }

    case ("GET", _) where request.path.hasPrefix("/v1/workflows/runs/"):
      let runId = String(request.path.dropFirst("/v1/workflows/runs/".count))
      return handleWorkflowRunStatus(runId: runId)

    case ("GET", _) where request.path.hasPrefix("/v1/workflows/"):
      let id = String(request.path.dropFirst("/v1/workflows/".count))
      guard let workflow = workflowStore.get(id) else {
        return .error(.error(status: 404, message: "Workflow not found: \(id)"))
      }
      var record = workflow.summaryJSON()
      record["graph"] = workflow.graph
      if let data = try? JSONSerialization.data(withJSONObject: record) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize workflow"))

    // Storyboard renderer (#237): ordered shot list → chained i2v renders
    // (each anchored on the previous shot's extracted last frame), optional
    // i2i inserts, final assembly. Long-running → 202 + job id; poll
    // GET /v1/video/status/{id} like any local video job.
    case ("POST", "/v1/storyboard/render"):
      do {
        guard configuration.ltx2WeightsPath != nil, configuration.ltx2GemmaPath != nil else {
          return .error(.error(status: 503, message: "Storyboard rendering needs local LTX-2 (--ltx2-weights/--ltx2-gemma)"))
        }
        let payload = try decode(StoryboardPayload.self, from: request.body)
        let spec = try storyboardSpec(from: payload)
        try spec.validate()
        let source = payload.source ?? "api"
        let status = videoJobTracker.submitOrchestrated(source: source, mode: .storyboard) { [weak self] report in
          guard let self else {
            throw StoryboardError.shotFailed(shot: 0, stage: "server", message: "server shutting down")
          }
          return try await self.runStoryboard(spec: spec, source: source, report: report)
        }
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(status)
        return .json(.rawJSON(status: 202, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/video/config/effective"):
      // Finding #16: a HYPOTHETICAL resolution — request-shaped context in,
      // requested_config + derived render_plan out. GET (below) stays as the
      // no-context readout.
      do {
        struct EffectiveQuery: Decodable {
          let width: Int?
          let height: Int?
          let frames: Int?
          let duration: Float?
          let fps: Int?
          let tuning: LTX2VideoTuning?
          let preset: String?
        }
        let q = (try? decode(EffectiveQuery.self, from: request.body)) ?? EffectiveQuery(
          width: nil, height: nil, frames: nil, duration: nil, fps: nil, tuning: nil, preset: nil)
        let videoPreset: ImagePreset? = q.preset.flatMap { presetStore.get($0) }
        let resolvedTyped = LTX2ConfigResolver.resolveTyped(
          request: q.tuning, preset: videoPreset?.videoTuning)

        // Derived plan, mirroring prepareLocalVideo's math step by step.
        var plan: [[String: String]] = []
        var w = q.width ?? videoPreset?.width ?? 704
        var h = q.height ?? videoPreset?.height ?? 448
        let snappedW = Self.snapDim64(w), snappedH = Self.snapDim64(h)
        if snappedW != w || snappedH != h {
          plan.append(["step": "snap_64", "note": "\(w)x\(h) -> \(snappedW)x\(snappedH)"])
          w = snappedW; h = snappedH
        }
        if resolvedTyped.twoStage {
          let s1 = Self.stageOneDims(finalWidth: w, finalHeight: h)
          if s1.halved {
            plan.append(["step": "two_stage_halving",
                         "note": "request dims are FINAL; stage 1 paints \(s1.width)x\(s1.height), refine doubles back"])
          } else {
            plan.append(["step": "stage1_floor",
                         "note": "final \(w)x\(h) too small to halve (floor 512x320) — single-scale render"])
          }
        }
        let fps = q.fps ?? 24
        var framesPerChunk = q.frames ?? 97
        var extendSeconds = Self.extendSecondsFromDuration(q.duration, framesPerChunk: framesPerChunk, fps: fps)
        if extendSeconds > 0 {
          let targetFrames = Int((extendSeconds * Float(fps)).rounded())
          if targetFrames <= 289 {
            let singleFrames = min(289, ((max(targetFrames, 9) - 2) / 8) * 8 + 9)
            framesPerChunk = max(framesPerChunk, singleFrames)
            extendSeconds = 0
            plan.append(["step": "single_pass_fold",
                         "note": "\(q.duration ?? 0)s folds into one \(framesPerChunk)f chunk (continuation chunks degenerate)"])
          } else {
            plan.append(["step": "chunked_continuation",
                         "note": "\(q.duration ?? 0)s exceeds the 289f window — continuation chunks with identity anchor"])
          }
        }
        plan.append(["step": "final", "note": "\(w)x\(h) @ \(framesPerChunk)f, fps \(fps)"])
        if q.width == nil && q.height == nil {
          plan.append(["step": "caveat",
                       "note": "i2v aspect-matching to a source image is not simulated here (no image supplied)"])
        }

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        struct Response: Encodable {
          let requestedConfig: [LTX2ResolvedParam]
          let renderPlan: [[String: String]]
          let presetId: String?
        }
        let data = try encoder.encode(Response(
          requestedConfig: resolvedTyped.params
            .map { p -> LTX2ResolvedParam in
              // Overlay request/preset provenance onto the readout rows.
              guard let src = resolvedTyped.provenance[p.name], src != p.source else { return p }
              return LTX2ResolvedParam(
                name: p.name, envKey: p.envKey, tier: p.tier,
                value: resolvedTyped.valueString(for: p.name) ?? p.value,
                source: src, valid: p.valid, note: p.note)
            },
          renderPlan: plan,
          presetId: q.preset))
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("GET", "/v1/video/traces"):
      // Task #19: the Prompt Lab feed — newest-first render traces.
      do {
        let limit = Int(request.queryParameters["limit"] ?? "") ?? 50
        let summaries = renderTraceStore.recentSummaries(limit: min(200, max(1, limit)))
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(["traces": summaries])
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("POST", _) where request.path.hasPrefix("/v1/video/traces/") && request.path.hasSuffix("/promote"):
      // Task #19 finding #5: promote a rated render's optimization pair into
      // the exemplar set. Intent comes from the bound attempt record; falls
      // back to the render prompt when the render skipped optimization.
      let id = String(request.path.dropFirst("/v1/video/traces/".count).dropLast("/promote".count))
      guard !id.isEmpty else { return .error(.error(status: 400, message: "Missing render_id")) }
      let events = renderTraceStore.events(renderId: id)
      guard let submitted = events.first(where: { $0.event == .submitted }) else {
        return .error(.error(status: 404, message: "No trace for render \(id)"))
      }
      let finalPrompt = submitted.payload["prompt"] ?? ""
      var intent = finalPrompt
      var contentMode = ContentModeManager.Mode.neutral.rawValue
      if let attemptId = submitted.payload["optimization_attempt_id"],
         let attempt = renderTraceStore.events(renderId: attemptId).last {
        intent = attempt.payload["intent"] ?? intent
        contentMode = attempt.payload["content_mode"] ?? contentMode
      }
      guard !finalPrompt.isEmpty else {
        return .error(.error(status: 422, message: "Trace has no prompt to promote"))
      }
      ExemplarStore.shared.add(PromptExemplar(
        intent: intent, final: finalPrompt, mediaKind: "video",
        contentMode: contentMode, sourceRenderId: id))
      return .json(status: 200, payload: ["success": true])

    case ("POST", _) where request.path.hasPrefix("/v1/video/traces/") && request.path.hasSuffix("/rating"):
      // Task #19: post-hoc human verdict, appended as a `rated` event.
      let id = String(request.path.dropFirst("/v1/video/traces/".count).dropLast("/rating".count))
      guard !id.isEmpty else { return .error(.error(status: 400, message: "Missing render_id")) }
      struct RatingBody: Decodable { let vote: String; let axis: String?; let note: String? }
      guard let body = try? decode(RatingBody.self, from: request.body) else {
        return .error(.error(status: 400, message: "'vote' is required (up/down or 1-5)"))
      }
      var payload = ["vote": body.vote, "axis": body.axis ?? "overall"]
      if let note = body.note { payload["note"] = note }
      renderTraceStore.append(RenderTraceEvent(
        renderId: id, event: .rated, taskKind: .videoRender, payload: payload))
      renderTraceStore.flush()
      return .json(status: 200, payload: ["success": true])

    case ("GET", "/v1/video/config/effective"):
      // Task #9 Phase 1: the effective Tier A/B video config with provenance
      // per parameter (configFile > env > builtin). The missing-rescale
      // detector: anything the caller expects to be set shows `builtin`.
      do {
        let params = LTX2ConfigResolver.resolveEffective()
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(["params": params])
        return .json(.rawJSON(status: 200, data: data))
      } catch {
        return .error(response(for: error))
      }

    case ("POST", "/v1/video/generate"):
      // Backward-compatible route. LOCAL renders here still block the HTTP
      // connection for the whole (synchronous) render — kept working for
      // existing callers. New / long renders should POST /v1/video/generate/async
      // and poll GET /v1/video/status/{id}. The Replicate cloud path was always
      // submit-and-poll (202), and stays so.
      let videoIntent = (try? decode(VideoGenerateRequest.self, from: request.body))?.backendIntent ?? .unspecified
      if videoIntent != .cloud {
        if let localResponse = await localVideoResponseIfConfigured(body: request.body) {
          logger.info("video: routing to local LTX-2 (synchronous)")
          return localResponse
        }
        if videoIntent == .local {
          return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights). Pass backend: \"replicate\" to explicitly use paid cloud."))
        }
        logger.warning("video: local LTX-2 not configured; falling back to PAID Replicate cloud (\(ReplicateVideoProxy.i2vModel)). Pass backend:\"local\" to forbid, backend:\"replicate\" to silence this warning.")
      }
      return await submitReplicateVideo(body: request.body)

    // Async LOCAL video: submit → 202 + job id immediately; poll
    // GET /v1/video/status/{id} for completion. This is the path a multi-minute /
    // multi-chunk render must take — the HTTP connection is never held open for
    // the whole denoise, and /health stays live (#217) so progress can be polled.
    // Cloud requests are already async via the Replicate proxy; this route
    // delegates to it for the cloud/fallback case, so one endpoint covers both.
    case ("POST", "/v1/video/generate/async"):
      let videoIntent = (try? decode(VideoGenerateRequest.self, from: request.body))?.backendIntent ?? .unspecified
      if videoIntent != .cloud {
        if let localResponse = await localVideoAsyncResponseIfConfigured(body: request.body) {
          logger.info("video: async-submitting local LTX-2 job")
          return localResponse
        }
        if videoIntent == .local {
          return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights). Pass backend: \"replicate\" to explicitly use paid cloud."))
        }
        logger.warning("video: local LTX-2 not configured; async-submitting to PAID Replicate cloud (\(ReplicateVideoProxy.i2vModel)). Pass backend:\"local\" to forbid, backend:\"replicate\" to silence this warning.")
      }
      return await submitReplicateVideo(body: request.body)

    case ("POST", "/v1/video/rerender"):
      // Winner action: replay a rendered clip's exact request at a higher
      // resolution budget (default 720p). Same seed + stored effective prompt
      // = the same clip, larger. Async job like /v1/video/generate/async.
      return await videoRerenderResponse(body: request.body)

    case ("POST", "/v1/video/extend"):
      // Winner action: chain a fresh continuation from a clip's last frame at
      // the 4s/480p standard (storyboard-style anchoring, new seed).
      return await videoExtendResponse(body: request.body)

    case ("GET", _) where request.path.hasPrefix("/v1/video/status/"):
      let jobId = String(request.path.dropFirst("/v1/video/status/".count))
      guard !jobId.isEmpty else {
        return .error(.error(status: 400, message: "Missing job_id in path"))
      }
      // Local jobs first (this box owns them), then the Replicate proxy. Both
      // report the same `VideoJobStatus` shape, so a single poll loop covers
      // whichever backend produced the job.
      let jobStatus: VideoJobStatus
      if let local = videoJobTracker.status(jobId: jobId) {
        jobStatus = local
      } else if let proxy = replicateVideoProxy, let cloud = proxy.status(jobId: jobId) {
        jobStatus = cloud
      } else {
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

    // MARK: - CivitAI conduit + prompt repository (#234)

    case ("GET", "/v1/civitai/search"):
      return await civitaiSearchRoute(request: request)

    case ("POST", "/v1/civitai/harvest"):
      return await civitaiHarvestRoute(request: request)

    case ("GET", "/v1/civitai/repo"):
      return civitaiRepoRoute(request: request)

    default:
      if ["/v1/generate", "/v1/lora/swap", "/v1/shutdown", "/health",
          "/v1/model/load", "/v1/model/activate", "/v1/model/pool", "/v1/model/unload",
          "/v1/loras", "/v1/loras/scan", "/v1/video/generate", "/v1/video/generate/async", "/v1/upscale",
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
      // Staging changes only the storage path. The semantic slot is part of
      // the requested stack and must survive (notably `role: "accel"` for
      // Krea-2 distill adapters whose names do not contain `turbo_lora`).
      return LoRAEntry(path: staged, scale: entry.scale, role: entry.role)
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

  struct LocalVideoRequest: Decodable {
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
    /// Conditioning compression (libx264 CRF) override for THIS render — the
    /// daemon sends a higher value (more motion) for partnered-action prompts
    /// and a low value (fidelity) for solo/portrait. nil = env default.
    let imgCompression: Int?
    /// CFG guidance override (>1 amplifies action; motion recipe sends 2.0 for
    /// partnered-action, omits for solo=fidelity). nil = env/config default.
    let guidance: Float?
    let extendToSeconds: Float?
    /// Target duration in seconds — the daemon/MCP vocabulary. For local
    /// renders this maps onto `extendToSeconds` (chunked continuation, each
    /// chunk re-anchored on the previous chunk\u{27}s last frame) when it
    /// exceeds one chunk. `extend_to_seconds` still wins when both are set.
    let duration: Float?
    /// Identity re-anchor strength for continuation chunks (0 disables).
    /// Default 0.5 for extended renders \u{2014} counters per-chunk subject drift.
    let identityAnchorStrength: Float?
    let fps: Int?
    let loraPath: String?
    let loraStrength: Float?
    /// Multiple LoRAs, applied in order — same {path, scale} shape as image
    /// LoRA requests. `loraPath`/`loraStrength` still work for a single LoRA.
    let loras: [LoRAEntry]?
    let outputPath: String?
    /// Which client/app submitted this job (desktop, bree, api…) — surfaced in
    /// the async job status and /health, same as image `GeneratePayload.source`.
    let source: String?
    /// Tier A tuning overrides (snake_case JSON via decoder strategy).
    let tuning: LTX2VideoTuning?
    /// Server-minted id from /v1/enhance binding this render to its
    /// optimization lineage (task #19, finding #6).
    let optimizationAttemptId: String?
    /// Optional preset id resolved from the shared PresetStore (mediaKind
    /// "video"): LoRAs, prompt prefix/suffix, negative prompt, dims budget,
    /// steps, seed. Explicit request fields always override preset values.
    let preset: String?
    /// Character whose canonical description is prepended to the prompt so the
    /// subject renders on-model. For T2V (no init image) this is the ONLY
    /// identity source; defaults to "kira" when unset. For I2V the init image
    /// already carries identity, so it's injected only when explicitly named.
    let character: String?
    /// Content mode (neutral/apple/banana/avocado) gating the character's
    /// mode-specific description addendum. Defaults to the server's current mode.
    let contentMode: String?
    /// Auto-enhance the prompt through the configured prompt-optimization
    /// provider (Dan's-PE via LM Studio) before encoding. Default on when a
    /// provider is configured; set false to send the raw prompt.
    let enhance: Bool?
    /// Named resolution budget: "480p" | "720p" | "1080p". Maps to a
    /// width x height pixel budget when explicit width/height are absent
    /// (previously this key was silently DROPPED on the local path).
    let resolution: String?
    /// Aspect ratio for the resolution budget: "16:9" (default) or "9:16".
    /// For I2V the source image's aspect still wins (budget only).
    let aspectRatio: String?
    /// Generate synchronized audio (task #21). T2V single-chunk only in v1;
    /// first audio render reloads the transformer with the audio branch.
    let audio: Bool?
    /// Suppress the manual character prepend when the CALLER has already woven
    /// the description into the prompt (Todd 2026-08-07). Mirrors the image
    /// path's `skip_character_injection`, which the video path never had.
    ///
    /// Without this the description is injected twice — once by the caller,
    /// once here — and at ~110 tokens each that alone overruns the 128-token
    /// tokenizer cap, truncating the scene and the camera direction off the
    /// end of the prompt. The idempotency check below cannot be relied on:
    /// it compares the first four words of THIS host's description against a
    /// prompt composed from a DIFFERENT character record on the daemon host,
    /// at a different framing, so it silently misses.
    ///
    /// `character` still applies — it drives preset resolution, the output
    /// directory and gallery attribution. Only the prompt prepend is skipped.
    let skipCharacterInjection: Bool?
    /// Suppress the preset promptPrefix/promptSuffix wrap when the caller's
    /// prompt is a stored EFFECTIVE prompt that already carries them (winner
    /// re-render/extend replay a trace's composed prompt — re-wrapping would
    /// condition on "prefix, prefix, …"). The preset's LoRAs, negatives, dims
    /// and steps still apply; only the prompt wrap is skipped.
    let skipPresetPrompt: Bool?
  }

  /// Map a named resolution + aspect to a width x height budget. Dims are
  /// budgets, not finals — the existing /64 snapping and I2V source-aspect
  /// derivation still apply downstream.
  private static func videoDims(resolution: String?, aspectRatio: String?) -> (width: Int, height: Int)? {
    guard let res = resolution?.lowercased() else { return nil }
    let landscape: (Int, Int)
    switch res {
    case "480p": landscape = (832, 480)
    case "720p": landscape = (1280, 720)
    case "1080p": landscape = (1920, 1080)
    default: return nil
    }
    let portrait = (aspectRatio ?? "16:9") == "9:16"
    return portrait ? (landscape.1, landscape.0) : landscape
  }

  private struct LocalVideoResponse: Encodable {
    let success: Bool
    let outputPath: String
    let frameCount: Int
    let durationSeconds: Float
    let elapsedSeconds: Double
    let backend: String
  }

  /// Submit a video render to the paid Replicate cloud proxy and return its 202
  /// job status (already submit-and-poll). Shared by the sync and async video
  /// routes for the cloud / unspecified-fallback case.
  private func submitReplicateVideo(body: Data) async -> RoutedResponse {
    guard let proxy = replicateVideoProxy else {
      return .error(.error(status: 503, message: "Video generation not available: configure LTX-2 (--ltx2-weights) for local video, or a Replicate API key for cloud"))
    }
    logger.info("video: routing to Replicate cloud (\(ReplicateVideoProxy.i2vModel))")
    do {
      var videoRequest = try decode(VideoGenerateRequest.self, from: body)
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
  }

  /// A LTX-2 generator + validated request, ready to enqueue. Shared by the
  /// synchronous (`/v1/video/generate`) and async (`/v1/video/generate/async`)
  /// local video paths so they build the render identically.
  private struct PreparedLocalVideo {
    let generator: LTX2VideoGenerator
    let request: LTX2VideoRequest
    /// t2v when there's no init image, i2v otherwise.
    let mode: VideoMode
    let source: String
    /// Lineage reference from /v1/enhance, if the caller optimized first.
    let optimizationAttemptId: String?
  }

  /// Map the daemon/MCP `duration` field onto chunked continuation: 0 when
  /// the request fits one chunk (single-chunk render, no continuation cost),
  /// else the requested seconds. Pure for unit testing.
  static func extendSecondsFromDuration(_ duration: Float?, framesPerChunk: Int, fps: Int) -> Float {
    guard let seconds = duration, seconds > 0, fps > 0 else { return 0 }
    let singleChunkSeconds = Float(framesPerChunk) / Float(fps)
    return seconds > singleChunkSeconds ? seconds : 0
  }

  /// Whether a video request renders more than one chunk (continuation path).
  /// Mirrors the extendToSeconds resolution above — the identity anchor
  /// defaults on only for these (#231).
  static func isExtendedRender(
    extendToSeconds: Float?, duration: Float?, framesPerChunk: Int, fps: Int
  ) -> Bool {
    if let explicit = extendToSeconds { return explicit > 0 }
    return extendSecondsFromDuration(
      duration, framesPerChunk: framesPerChunk, fps: fps) > 0
  }

  /// Snap a render dimension to the nearest multiple of 64 (floor 256).
  /// LTX-2 renders at dims that are 32-multiples but NOT 64-multiples (e.g.
  /// 480) exhibit progressive haze (#219) — every clean render in the 07-13
  /// bisect used /64 dims, every hazy one used 480.
  /// Resolve the stage-1 (painted) dims for a two-stage render whose request
  /// dims mean the FINAL output size. Pure, so it can be tested at the sizes
  /// callers actually send rather than only the ones convenient to validate.
  ///
  /// The refine SHARPENS what stage 1 painted; it cannot invent detail that was
  /// never generated. Halving a request sized for the old single-pass
  /// convention therefore degrades output silently — Kira's 704x448 became a
  /// 384x256 base pass (a third of her previous pixels) and went visibly
  /// diffuse, while every render validated that day asked for 960x576 and
  /// halved comfortably to 512x320 (2026-08-02).
  ///
  /// Below the floor the request is treated as STAGE-1 dims (pre-halving
  /// behaviour): the clip finishes at 2x the requested size. A sharp surprise
  /// beats a soft silent degradation.
  ///
  /// - Returns: the dims to paint at, and whether halving was applied.
  static func stageOneDims(
    finalWidth: Int, finalHeight: Int, floorPixels: Int = 512 * 320
  ) -> (width: Int, height: Int, halved: Bool) {
    let w = snapDim64(finalWidth / 2)
    let h = snapDim64(finalHeight / 2)
    guard w * h >= floorPixels else {
      return (finalWidth, finalHeight, false)
    }
    return (w, h, true)
  }

  static func snapDim64(_ value: Int) -> Int {
    max(256, Int((Double(value) / 64.0).rounded()) * 64)
  }

  /// Derive I2V render dims matching the source image aspect within the
  /// requested pixel-area budget, both dims /64. Pure for unit testing.
  ///
  /// Rounding each axis to /64 independently compounds error in opposite
  /// directions: a 1664x896 source (aspect 1.857) at a 448x704 budget produced
  /// 768x384 (aspect 2.000) — the height's ideal 412.1 sat almost exactly on a
  /// 64-boundary midpoint and rounded DOWN while the width rounded up, a 7.7%
  /// distortion that visibly squashes the subject (2026-08-01). Search the /64
  /// neighbourhood instead and keep the pair whose aspect is closest to the
  /// source, breaking ties toward the pixel budget.
  static func deriveVideoDims(
    sourceWidth: Int, sourceHeight: Int, budgetWidth: Int, budgetHeight: Int
  ) -> (width: Int, height: Int) {
    guard sourceWidth > 0, sourceHeight > 0 else {
      return (snapDim64(budgetWidth), snapDim64(budgetHeight))
    }
    let aspect = Double(sourceWidth) / Double(sourceHeight)
    let budget = Double(max(budgetWidth, 64) * max(budgetHeight, 64))
    let idealW = (budget * aspect).squareRoot()
    let idealH = idealW / aspect

    let baseW = Int((idealW / 64.0).rounded())
    let baseH = Int((idealH / 64.0).rounded())

    func search(areaCap: Double) -> (w: Int, h: Int, aspectErr: Double, areaErr: Double)? {
      var best: (w: Int, h: Int, aspectErr: Double, areaErr: Double)?
      for dw in -1...1 {
        for dh in -1...1 {
          let w = max(256, (baseW + dw) * 64)
          let h = max(256, (baseH + dh) * 64)
          let area = Double(w * h)
          guard area <= budget * areaCap else { continue }
          let aspectErr = abs(Double(w) / Double(h) - aspect) / aspect
          let areaErr = abs(area - budget) / budget
          if let b = best {
            let better = aspectErr < b.aspectErr - 1e-9
              || (abs(aspectErr - b.aspectErr) <= 1e-9 && areaErr < b.areaErr)
            if better { best = (w, h, aspectErr, areaErr) }
          } else {
            best = (w, h, aspectErr, areaErr)
          }
        }
      }
      return best
    }

    // Prefer staying near the budget; but at small budgets the 256 floor pins one
    // axis and the tight cap can force a badly stretched pair (a halved two-stage
    // budget hit 19% that way), so allow a larger clip rather than distort.
    var pick = search(areaCap: 1.25)
    if pick == nil || pick!.aspectErr > 0.03, let relaxed = search(areaCap: 1.6),
       relaxed.aspectErr < (pick?.aspectErr ?? .infinity) - 1e-9 {
      pick = relaxed
    }
    guard let chosen = pick else {
      return (snapDim64(Int(idealW.rounded())), snapDim64(Int(idealH.rounded())))
    }
    return (chosen.w, chosen.h)
  }

  /// Pixel dimensions of an image file without decoding the bitmap.
  static func imagePixelSize(atPath path: String) -> (width: Int, height: Int)? {
    let url = URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
          let width = props[kCGImagePropertyPixelWidth] as? Int,
          let height = props[kCGImagePropertyPixelHeight] as? Int
    else { return nil }
    return (width, height)
  }

  /// Map an LTX-2 (chunk, step) progress tick to an overall 0-100 percent across
  /// all chunks. Pure so it can be unit-tested and reused by both local paths.
  static func localVideoProgressPercent(chunk: Int, totalChunks: Int, step: Int, totalSteps: Int) -> Int {
    let chunks = max(1, totalChunks)
    let steps = max(1, totalSteps)
    let done = max(0, chunk) * steps + max(0, step)
    let total = chunks * steps
    return min(100, max(0, Int((Double(done) / Double(total)) * 100.0)))
  }

  /// Resolve LTX-2 weights, build + validate the render request. Returns nil when
  /// local LTX-2 isn't configured (caller falls through to Replicate); throws for
  /// a malformed request or invalid output path.
  private func prepareLocalVideo(body: Data) async throws -> PreparedLocalVideo? {
    guard let weights = configuration.ltx2WeightsPath, let gemma = configuration.ltx2GemmaPath else {
      return nil
    }
    let req = try decode(LocalVideoRequest.self, from: body)

    // Video presets — same PresetStore as images (mediaKind "video"). A
    // preset is a named bundle: LoRAs (bare filenames resolve through the
    // LoRA library search roots), prompt shaping, negative prompt, dims
    // budget, steps, seed. Explicit request fields win over the preset.
    var videoPreset: ImagePreset? = nil
    if let presetId = req.preset, !presetId.isEmpty {
      guard let found = presetStore.get(presetId) else {
        throw WarmServerError.invalidRequest(message: "Unknown preset \u{27}\(presetId)\u{27} — see /v1/presets")
      }
      videoPreset = found
      logger.info("LTX-2: applying video preset \u{27}\(presetId)\u{27} (\(found.loras.count) LoRA(s))")
    }

    // Contain the output within the allowed directory. A relative (or absent)
    // output path is resolved against the ALLOWED directory, not the process
    // CWD — under launchd the CWD is the repo checkout, so the old bare-name
    // default failed its own containment check on every submit (#219).
    let requestedOutputRaw = req.outputPath ?? "ltx2-\(UUID().uuidString).mp4"
    let requestedOutput: String
    if requestedOutputRaw.hasPrefix("/") || requestedOutputRaw.hasPrefix("~") {
      requestedOutput = requestedOutputRaw
    } else {
      requestedOutput = (configuration.allowedOutputDirectory as NSString)
        .appendingPathComponent(requestedOutputRaw)
    }
    let resolvedOutput = try WarmServerOutputPathValidator.resolveOutputPath(
      requestedOutput, allowedOutputDirectory: configuration.allowedOutputDirectory).path

    let generator: LTX2VideoGenerator
    if let existing = videoHolder.get() {
      generator = existing
    } else {
      logger.info("LTX-2: resolving weights/text-encoder (downloads on first use if not cached)…")
      let weightsURL = try await ModelResolution.resolve(
        modelSpec: weights,
        filePatterns: ["transformer-distilled.safetensors", "connector.safetensors",
                        "vae_decoder.safetensors", "vae_encoder.safetensors", "config.json"]
      )
      let gemmaURL = try await ModelResolution.resolve(
        modelSpec: gemma,
        filePatterns: ["*.safetensors", "*.json", "tokenizer/*", "*.model"]
      )
      generator = LTX2VideoGenerator(
        config: .init(weightsDir: weightsURL.path, gemmaPath: gemmaURL.path), logger: logger)
    }
    // Publish into the shared holder so the coordinator can evict it before an
    // image load, and so the render queue evicts image models before this one
    // actually loads its ~65GB of weights inside generate() (#218).
    videoHolder.set(generator)
    // #1479: wire telemetry + the (normally never-raised) preemption signal
    // on EVERY generator instance — fresh or reused — so the video path is
    // always preemptible-capable while staying byte-identical unless a job
    // actually raises the signal. Also covers the fresh instance a preemption
    // eviction produces (`VideoGeneratorHolder.release()` deallocates the old
    // one), since this is the same function a post-eviction cold reload runs.
    generator.setTelemetry(ltx2Telemetry)
    generator.setPreemptionSignal(ltx2PreemptionSignal)

    // Accept an init image as bytes (image_base64) when no server path is given.
    let effectiveInitImage = req.imagePath ?? Self.writeTempImage(base64: req.imageBase64)

    var loraEntries: [LoRAEntry] = req.loras ?? []
    if loraEntries.isEmpty, req.loraPath == nil, let preset = videoPreset, !preset.loras.isEmpty {
      loraEntries = preset.loras.map {
        LoRAEntry(path: $0.filename, scale: Float($0.scale), role: $0.role)
      }
    }
    if loraEntries.isEmpty, req.loraPath == nil,
       let defaultLoRA = configuration.ltx2DefaultLoRA, !defaultLoRA.isEmpty {
      // "path" or "path@scale"
      let parts = defaultLoRA.split(separator: "@", maxSplits: 1).map(String.init)
      let scale = parts.count == 2 ? Float(parts[1]) ?? 1.0 : 1.0
      loraEntries = [LoRAEntry(path: parts[0], scale: scale)]
      logger.info("LTX-2: applying default video LoRA \(parts[0]) @ \(scale) (--ltx2-lora)")
    }
    let resolvedLoRAs: [LTX2LoRAReference] = try loraEntries.map { entry in
      let config = try entry.makeConfiguration()
      guard case .local(let url) = config.source else {
        throw WarmServerError.invalidRequest(
          message: "LTX-2 video LoRAs must be local files (got a HuggingFace reference for '\(entry.path)')")
      }
      return LTX2LoRAReference(path: url.path, scale: config.scale)
    }

    // LTX-2 render dims must be divisible by 64: 32-multiples that are not
    // 64-multiples (e.g. 480) produce progressive haze/ghosting (#219). I2V
    // output must additionally match the SOURCE image aspect ratio — a fixed
    // preset like 704x448 applied to a portrait source distorts the
    // conditioning frame and the render drifts off the image. The requested
    // width x height is kept only as a pixel-area budget for I2V.
    // Priority: explicit width/height > named resolution ("720p" etc., FIXED:
    // previously silently dropped) > preset dims > 704x448 default.
    let namedDims = Self.videoDims(resolution: req.resolution, aspectRatio: req.aspectRatio)
    var renderWidth = req.width ?? namedDims?.width ?? videoPreset?.width ?? 704
    var renderHeight = req.height ?? namedDims?.height ?? videoPreset?.height ?? 448
    if req.width == nil, let nd = namedDims {
      logger.info("LTX-2: resolution '\(req.resolution ?? "")' -> \(nd.width)x\(nd.height) budget")
    }
    if let initPath = effectiveInitImage,
       let sourceSize = Self.imagePixelSize(atPath: initPath) {
      let derived = Self.deriveVideoDims(
        sourceWidth: sourceSize.width, sourceHeight: sourceSize.height,
        budgetWidth: renderWidth, budgetHeight: renderHeight)
      if derived.width != renderWidth || derived.height != renderHeight {
        logger.info(
          "LTX-2 I2V: adjusted \(renderWidth)x\(renderHeight) -> \(derived.width)x\(derived.height) (source \(sourceSize.width)x\(sourceSize.height), aspect-matched, /64)")
        renderWidth = derived.width
        renderHeight = derived.height
      }
    } else {
      let snappedW = Self.snapDim64(renderWidth)
      let snappedH = Self.snapDim64(renderHeight)
      if snappedW != renderWidth || snappedH != renderHeight {
        logger.info("LTX-2: snapped \(renderWidth)x\(renderHeight) -> \(snappedW)x\(snappedH) (dims must be /64, #219)")
        renderWidth = snappedW
        renderHeight = snappedH
      }
    }
    // Two-stage dims convention (2026-08-02): with LTX2_TWO_STAGE=1 the request
    // dims are the FINAL output size (matching ComfyUI and every caller's
    // intuition). ComfyBox's pipeline doubles stage-1 dims through the refine,
    // so hand it the /64-snapped HALF. Without this, enabling two-stage doubled
    // every clip's output size and the refine-volume gate silently skipped the
    // refine — the worst of both worlds. Stage-1 /64 keeps the final /128-ish
    // and matches all validated two-stage renders (stage 1 at 448x256 etc.).
    // Typed resolution honors request/preset tuning overrides (finding #18):
    // a request can enable two-stage without the plist knowing.
    if LTX2ConfigResolver.resolveTyped(request: req.tuning, preset: videoPreset?.videoTuning).twoStage {
      let s1 = Self.stageOneDims(finalWidth: renderWidth, finalHeight: renderHeight)
      if s1.halved {
        logger.info(
          "LTX-2 two-stage: request dims \(renderWidth)x\(renderHeight) = FINAL; stage 1 paints \(s1.width)x\(s1.height), refine doubles to \(s1.width * 2)x\(s1.height * 2)")
        renderWidth = s1.width
        renderHeight = s1.height
      } else {
        logger.warning("""
          LTX-2 two-stage: request \(renderWidth)x\(renderHeight) would paint stage 1 at \
          \(Self.snapDim64(renderWidth / 2))x\(Self.snapDim64(renderHeight / 2)) — below the \
          stage-1 floor, which renders SOFT (the refine sharpens, it cannot invent detail). \
          Treating the request as stage-1 dims instead; output will be \
          \(renderWidth * 2)x\(renderHeight * 2). Send ~2x larger dims for the intended size.
          """)
      }
    }
    if let requestedSteps = req.steps, requestedSteps != 8 {
      logger.warning(
        "LTX-2: steps=\(requestedSteps) requested, but the distilled pipeline uses a fixed 8-step sigma schedule — the value is currently ignored (#219)")
    }

    var effectivePrompt = req.prompt

    // Character identity + optional prompt enhancement. For T2V (no init image)
    // there is no other identity source, so default to "kira" when the caller
    // names no character. For I2V the init image already carries identity.
    let isT2V = (effectiveInitImage == nil)
    let characterName = req.character ?? (isT2V ? "kira" : nil)
    let charMode = req.contentMode.flatMap { ContentModeManager.Mode(rawValue: $0) } ?? .neutral
    var characterDesc: String? = nil
    if let name = characterName,
       let entry = await characterStore.get(CharacterEntry.slug(name)) {
      characterDesc = entry.resolvedDescription(for: charMode)
    }

    // Auto-enhance the video prompt through the configured prompt-optimization
    // provider (Dan's-PE via LM Studio). The optimizer weaves in the character
    // description AND enriches the scene, so it replaces the manual character
    // prepend. Opt out per request with enhance:false; falls back to the manual
    // prepend when no provider is configured or enhancement fails.
    var enhancedApplied = false
    let aiProviderConfig = ComfyBoxServerConfig.loadOrMigrate()
    if req.enhance != false, let endpoint = aiProviderConfig.providers.promptOptimization {
      var base = endpoint.baseUrl
      while base.hasSuffix("/") { base.removeLast() }
      if base.hasSuffix("/v1") { base = String(base.dropLast(3)) }
      while base.hasSuffix("/") { base.removeLast() }
      let optimizer = PromptOptimizer(
        configuration: PromptOptimizer.Configuration(
          ollamaBaseURL: base, lmStudioBaseURL: nil, model: endpoint.model,
          timeoutSeconds: 90, enabled: true),
        logger: logger)
      // i2v: motion-only enhancement (the init image fixes subject/scene); t2v: full scene.
      let result = await optimizer.optimize(
        prompt: req.prompt, character: characterName,
        characterDescription: characterDesc, contentMode: charMode.rawValue,
        mediaKind: isT2V ? "video" : "video-i2v")
      if result.enhanced {
        effectivePrompt = result.prompt
        enhancedApplied = true
        logger.info("Video: enhanced prompt via \(endpoint.model)\(characterName.map { " (character \($0))" } ?? "").")
      }
    }

    // Fallback: manual character prepend when enhancement didn't run/apply.
    // Skipped outright when the caller says it already wove the description in
    // — see `skipCharacterInjection` on the request for why the idempotency
    // check below is not sufficient on its own.
    if req.skipCharacterInjection == true, characterName != nil {
      logger.info("Video: character injection skipped — caller composed identity.")
    }
    if req.skipCharacterInjection != true,
       !enhancedApplied, let name = characterName, let desc = characterDesc, !desc.isEmpty {
      // Idempotency: skip if the caller already wrote the description in.
      let alreadyPresent = desc.split(separator: " ").prefix(4).allSatisfy {
        effectivePrompt.localizedCaseInsensitiveContains($0)
      }
      if !alreadyPresent {
        effectivePrompt = desc + " " + effectivePrompt
        logger.info("Video: prepended character '\(name)' (mode \(charMode.rawValue)) to prompt.")
      }
    }

    if let preset = videoPreset, req.skipPresetPrompt != true {
      if let prefix = preset.promptPrefix, !prefix.isEmpty { effectivePrompt = prefix + ", " + effectivePrompt }
      if let suffix = preset.promptSuffix, !suffix.isEmpty { effectivePrompt = effectivePrompt + ", " + suffix }
    }

    // Single-pass fold (2026-08-02): continuation chunks DEGENERATE — chunk 2
    // collapses into fragments (long-known for i2v, which is why the daemon
    // sends explicit single-pass frames there; observed for T2V today the
    // moment two-stage went live: chunk 1 clean, chunk 2 psychedelic). The
    // daemon's rule only covers i2v, so fold ANY duration that fits the
    // trained window (289f = 12s) into ONE chunk here, t2v included.
    var foldedFramesPerChunk = req.frames ?? 97
    var foldedExtendSeconds = req.extendToSeconds
      ?? Self.extendSecondsFromDuration(req.duration, framesPerChunk: foldedFramesPerChunk, fps: req.fps ?? 24)
    if foldedExtendSeconds > 0 {
      let fps = req.fps ?? 24
      let targetFrames = Int((foldedExtendSeconds * Float(fps)).rounded())
      if targetFrames <= 289 {
        let singleFrames = min(289, ((max(targetFrames, 9) - 2) / 8) * 8 + 9)  // 1+8k covering target
        foldedFramesPerChunk = max(foldedFramesPerChunk, singleFrames)
        logger.info(
          "LTX-2: folded \(foldedExtendSeconds)s request into a single \(foldedFramesPerChunk)f chunk (continuation chunks degenerate; ≤289f renders single-pass)")
        foldedExtendSeconds = 0  // 0 = no continuation chunks
      }
    }

    let videoRequest = LTX2VideoRequest(
      prompt: effectivePrompt,
      negativePrompt: req.negativePrompt ?? videoPreset?.negativePrompt,
      initImagePath: effectiveInitImage,
      width: renderWidth,
      height: renderHeight,
      framesPerChunk: foldedFramesPerChunk,
      steps: req.steps ?? videoPreset?.steps ?? 8,
      seed: req.seed ?? videoPreset?.seed.map(UInt64.init) ?? 42,
      strength: req.strength ?? 1.0,
      imgCompression: req.imgCompression,
      guidance: req.guidance,
      // Re-enabled by default for EXTENDED renders (#231, 2026-07-16): the
      // 2026-07-13 MLX mutex crash on this path was memory pressure — with
      // the int8 stack (#230) a 12s/3-chunk anchored render completed clean
      // (289f, no crash). Single-chunk renders don't anchor (nothing to
      // drift); callers can still pass 0 to disable.
      // Mid-pass identity re-anchor is OPT-IN and default OFF — it was superseded
      // by the face-region anchor (LTX2_FACE_ANCHOR_STRENGTH), which holds partner
      // faces without the multi-keyframe gap-collapse. Enable explicitly via
      // LTX2_REANCHOR_INTERVAL>0 (+ _STRENGTH); a standard 97f/4s render NEVER takes
      // it unless the interval is set below the frame count. Only the pre-existing
      // extended/chunked anchor stays on by default (unchanged behavior).
      identityAnchorStrength: req.identityAnchorStrength
        ?? (Self.isExtendedRender(
              extendToSeconds: req.extendToSeconds, duration: req.duration,
              framesPerChunk: req.frames ?? 97, fps: req.fps ?? 24)
            ? 0.5
            : ((effectiveInitImage != nil
                && (Int(ProcessInfo.processInfo.environment["LTX2_REANCHOR_INTERVAL"] ?? "") ?? 0) > 0
                && (req.frames ?? 97) > (Int(ProcessInfo.processInfo.environment["LTX2_REANCHOR_INTERVAL"] ?? "") ?? 0))
               ? (Float(ProcessInfo.processInfo.environment["LTX2_REANCHOR_STRENGTH"] ?? "") ?? 0.4) : 0)),
      identityReAnchorInterval: (Int(ProcessInfo.processInfo.environment["LTX2_REANCHOR_INTERVAL"] ?? "") ?? 0),
      extendToSeconds: foldedExtendSeconds,
      fps: req.fps ?? 24,
      loraPath: req.loraPath,
      loraStrength: req.loraStrength ?? 1.0,
      loras: resolvedLoRAs,
      outputPath: resolvedOutput
,
      tuning: req.tuning,
      presetTuning: videoPreset?.videoTuning,
      audio: req.audio ?? false
    )
    // Validate before enqueuing so bad frames/dims fail fast.
    try generator.validate(videoRequest)

    return PreparedLocalVideo(
      generator: generator,
      request: videoRequest,
      mode: (effectiveInitImage?.isEmpty == false) ? .i2v : .t2v,
      source: req.source ?? "api",
      optimizationAttemptId: req.optimizationAttemptId)
  }

  /// If LTX-2 is configured, ASYNC-submit the local render and return 202 + a
  /// job status immediately; otherwise nil so the caller falls through to the
  /// Replicate proxy. Poll GET /v1/video/status/{id} for completion. This is the
  /// path a long (multi-minute / multi-chunk) render must take — it never holds
  /// the HTTP connection open for the whole denoise.
  private func localVideoAsyncResponseIfConfigured(body: Data) async -> RoutedResponse? {
    do {
      guard let prep = try await prepareLocalVideo(body: body) else { return nil }
      logger.info("LTX-2: local video job submitted (\(prep.request.width)x\(prep.request.height), \(prep.request.framesPerChunk)f)")
      var tracePayload: [String: String] = ["prompt": prep.request.prompt]
      if prep.request.audio {
        tracePayload["has_audio"] = "true"
        tracePayload["audio_seconds"] = String(format: "%.2f",
          Float(prep.request.framesPerChunk) / Float(prep.request.fps))
      }
      if let attemptId = prep.optimizationAttemptId {
        tracePayload["optimization_attempt_id"] = attemptId
      }
      // Winner actions (2026-08-10): store the sanitized request + the
      // resolved seed/dims so this render_id is replayable — /v1/video/rerender
      // replays it at 720p, /v1/video/extend chains a continuation.
      if let requestJSON = VideoWinnerActions.sanitizedRequestJSON(fromBody: body) {
        tracePayload["request_json"] = requestJSON
      }
      tracePayload["seed"] = String(prep.request.seed)
      tracePayload["width"] = String(prep.request.width)
      tracePayload["height"] = String(prep.request.height)
      tracePayload["frames"] = String(prep.request.framesPerChunk)
      tracePayload["fps"] = String(prep.request.fps)
      if let initImage = prep.request.initImagePath {
        tracePayload["image_path"] = initImage
      }
      let status = videoJobTracker.submit(
        source: prep.source, mode: prep.mode, coordinator: coordinator,
        // Snapshot at SUBMIT time (finding #15): the authoritative resolution
        // this render will use, durable on the job status.
        resolvedConfig: LTX2ConfigResolver.resolveTyped(
          request: prep.request.tuning, preset: prep.request.presetTuning).params,
        tracePayload: tracePayload,
        wantsAudio: prep.request.audio
      ) { report in
        // #1479: the preemptible entry — a no-op unless a `preempt: true`
        // image job raises `ltx2PreemptionSignal` while this render is
        // in-flight; otherwise behaves exactly like `.generate`.
        try prep.generator.generatePreemptible(prep.request) { chunk, totalChunks, step, totalSteps in
          report(Self.localVideoProgressPercent(
            chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps))
          self.ltx2StepPosition.update(chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps)
        }
      }
      let encoder = JSONEncoder()
      encoder.keyEncodingStrategy = .convertToSnakeCase
      let data = try encoder.encode(status)
      return .json(.rawJSON(status: 202, data: data))
    } catch let error as LTX2VideoError {
      return .error(.error(status: 400, message: error.localizedDescription))
    } catch {
      return .error(response(for: error))
    }
  }

  // MARK: - Winner actions (2026-08-10: 480p/4s standard, improve the keepers)

  private struct VideoRerenderBody: Decodable {
    let renderId: String?
    let path: String?
    let resolution: String?
  }

  private struct VideoExtendBody: Decodable {
    let renderId: String?
    let path: String?
    let seconds: Int?
    let prompt: String?
  }

  /// "1786475197556_ltx2-ABC.mp4" → "ltx2-ABC.mp4" (daemon temp prefix).
  static func stripTimestampPrefix(_ filename: String) -> String {
    guard let underscore = filename.firstIndex(of: "_"),
      filename[filename.startIndex..<underscore].allSatisfy({ $0.isNumber }),
      filename.distance(from: filename.startIndex, to: underscore) >= 10
    else { return filename }
    return String(filename[filename.index(after: underscore)...])
  }

  /// Locate the trace behind a winner action: by render_id directly, or by
  /// matching a gallery clip's path/filename against recent terminal outputs.
  private func findVideoTrace(
    renderId: String?, path: String?
  ) -> (renderId: String, submitted: [String: String], outputPath: String?)? {
    if let id = renderId, !id.isEmpty {
      let events = renderTraceStore.events(renderId: id)
      guard let submitted = events.first(where: { $0.event == .submitted }) else { return nil }
      let terminal = events.last { $0.event == .terminal }
      return (id, submitted.payload, terminal?.payload["output_path"])
    }
    if let path, !path.isEmpty {
      let summaries = renderTraceStore.recentSummaries(limit: 500)
      let filename = (path as NSString).lastPathComponent
      // Daemon-side copies carry a `<epoch-ms>_` temp prefix on the engine's
      // original basename (fetch-before-save path, 2026-08-11) — the sidecar
      // join key recorded that prefixed name for existing clips, so strip a
      // leading digit run before comparing.
      let normalized = Self.stripTimestampPrefix(filename)
      // Exact full-path match first — a newer clip that merely shares a
      // basename must not shadow the clip the caller actually named.
      let match =
        summaries.first { $0.outputPath == path }
        ?? summaries.first {
          guard let base = $0.outputPath.map({ ($0 as NSString).lastPathComponent }) else { return false }
          return base == filename || base == normalized
        }
      if let match, let out = match.outputPath {
        let events = renderTraceStore.events(renderId: match.renderId)
        let submitted = events.first { $0.event == .submitted }
        return (match.renderId, submitted?.payload ?? [:], out)
      }
    }
    return nil
  }

  private func videoRerenderResponse(body: Data) async -> RoutedResponse {
    guard let req = try? decode(VideoRerenderBody.self, from: body),
      req.renderId != nil || req.path != nil
    else {
      return .error(.error(status: 400, message: "Body must include 'render_id' or 'path'"))
    }
    let resolution = req.resolution ?? "720p"
    if let validationError = VideoGenerateRequest.validateResolution(resolution) {
      return .error(.error(status: 400, message: validationError))
    }
    guard let trace = findVideoTrace(renderId: req.renderId, path: req.path) else {
      return .error(.error(status: 404, message: "No render trace matches that render_id/path"))
    }
    guard let requestJSON = trace.submitted["request_json"] else {
      return .error(.error(
        status: 422,
        message:
          "Trace \(trace.renderId) predates replay support (no stored request) — re-render works for clips rendered after this deploy"))
    }
    do {
      let newBody = try VideoWinnerActions.rerenderBody(
        requestJSON: requestJSON,
        resolvedSeed: trace.submitted["seed"],
        effectivePrompt: trace.submitted["prompt"],
        resolution: resolution,
        initImagePath: trace.submitted["image_path"])
      if let routed = await localVideoAsyncResponseIfConfigured(body: newBody) {
        logger.info("video: winner re-render of \(trace.renderId) at \(resolution)")
        return routed
      }
      return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights)"))
    } catch {
      return .error(.error(status: 422, message: "\(error)"))
    }
  }

  private func videoExtendResponse(body: Data) async -> RoutedResponse {
    guard let req = try? decode(VideoExtendBody.self, from: body),
      req.renderId != nil || req.path != nil
    else {
      return .error(.error(status: 400, message: "Body must include 'render_id' or 'path'"))
    }
    let trace = findVideoTrace(renderId: req.renderId, path: req.path)
    // Source clip: the caller's path when it exists, else the trace's output.
    let clipPath = [req.path, trace?.outputPath].compactMap { $0 }
      .first { FileManager.default.fileExists(atPath: $0) }
    guard let clipPath else {
      return .error(.error(
        status: 404,
        message: "Source clip not found on disk — pass 'path' or a 'render_id' whose output still exists"))
    }
    var framePath: String?
    do {
      // Same containment rule as /v1/video/output: only gallery clips.
      _ = try WarmServerOutputPathValidator.resolveOutputPath(
        clipPath, allowedOutputDirectory: configuration.allowedOutputDirectory)
      let extracted = NSTemporaryDirectory() + "winner-extend-\(UUID().uuidString).png"
      try LastFrameExtractor.extractLastFrame(from: clipPath, to: extracted)
      framePath = extracted
      let newBody = try VideoWinnerActions.extendBody(
        requestJSON: trace?.submitted["request_json"],
        framePath: extracted,
        seconds: req.seconds ?? 4,
        prompt: req.prompt,
        effectivePrompt: trace?.submitted["prompt"])
      if let routed = await localVideoAsyncResponseIfConfigured(body: newBody) {
        // The frame must OUTLIVE this request — the queued render reads it
        // when its GPU turn comes. tmp is system-cleaned between boots.
        logger.info("video: winner extend of \((clipPath as NSString).lastPathComponent) (+\(req.seconds ?? 4)s)")
        return routed
      }
      if let framePath { try? FileManager.default.removeItem(atPath: framePath) }
      return .error(.error(status: 503, message: "Local LTX-2 video not configured (--ltx2-weights)"))
    } catch {
      // No render was queued — the extracted frame would be orphaned.
      if let framePath { try? FileManager.default.removeItem(atPath: framePath) }
      return .error(response(for: error))
    }
  }

  /// If LTX-2 is configured, generate the video locally and return the result
  /// SYNCHRONOUSLY (blocks the HTTP connection for the whole render); otherwise
  /// nil so the caller falls through to the Replicate proxy. Kept for backward
  /// compatibility — new/long renders should use the async path above.
  private func localVideoResponseIfConfigured(body: Data) async -> RoutedResponse? {
    guard configuration.ltx2WeightsPath != nil, configuration.ltx2GemmaPath != nil else {
      return nil
    }
    do {
      guard let prep = try await prepareLocalVideo(body: body) else { return nil }
      let generator = prep.generator
      let videoRequest = prep.request

      logger.info("LTX-2: local video request queued (\(videoRequest.width)x\(videoRequest.height), \(videoRequest.framesPerChunk)f)")
      let result = try await coordinator.enqueueLocalVideo(wantsAudio: videoRequest.audio) { report in
        // #1479: preemptible entry — see the async path's doc comment above.
        try generator.generatePreemptible(videoRequest) { chunk, totalChunks, step, totalSteps in
          report(Self.localVideoProgressPercent(
            chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps))
          self.ltx2StepPosition.update(chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps)
        }
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

  // MARK: - #1479 preemption entry point
  //
  // Called from BOTH `/v1/generate` and `/v1/generate/async` route handlers,
  // BEFORE either calls into `coordinator` — see the block comment above
  // `RollingMeanSec` for why this cannot itself be an actor method. Returns
  // `.notApplicable` when the flag is absent/false, no video is rendering, or
  // a preemption is already in flight (nested preemption refused, spec) — the
  // caller then falls through to its normal enqueue, unmodified, which is
  // exactly pre-#1479 behaviour.

  /// Result of an `attemptPreemption` call. `fileprivate` (not `private`)
  /// because `ImageJobTracker.submitPreempting`, a sibling top-level type in
  /// this same file, needs to name it in its own parameter type.
  enum PreemptionOutcome {
    /// Nothing to do — caller enqueues normally, unmodified.
    case notApplicable
    /// The refusal guard declined (finishing beats preempting). `eta` is the
    /// projected remaining seconds — callers stamp it onto the normal queued
    /// response/status as `preempt_refused`/`eta_sec`.
    case refused(eta: Double)
    /// The preemption ran end-to-end and the image job completed.
    case ran(GenerateResponse)
    /// The preemption ran end-to-end and the image job itself failed (the
    /// video still resumed — that failure is independent of this one).
    case ranFailed(Error)
  }

  /// `evictReloadRoundTripSec` for the refusal guard: sum of the two observed
  /// rolling means, nil until BOTH have at least one sample (spec: never
  /// refuse on a guess).
  private func combinedEvictReloadRoundTripSec() -> Double? {
    guard let evictMean = ltx2EvictMean.mean(), let reloadMean = ltx2ReloadMean.mean() else { return nil }
    return evictMean + reloadMean
  }

  private func attemptPreemption(
    _ payload: GeneratePayload, source: String, rawBody: Data?, jobId: String? = nil
  ) async -> PreemptionOutcome {
    guard payload.preempt == true, videoHolder.isRendering() else { return .notApplicable }
    guard preemptionInFlight.trySet() else {
      // Nested preemption refused (spec) — a preemptor cannot itself be
      // preempted. Falls through as a normal enqueue.
      return .notApplicable
    }

    let tv = ltx2Telemetry.view()
    let stepsRemaining = ltx2StepPosition.read() ?? 0
    var remainingPhaseMeans: [Double] = []
    if let currentPhase = tv.currentPhase.flatMap(LTX2Phase.init(rawValue:)) {
      for phase in LTX2Phase.allCases where LTX2UnwindGuard.rank(phase) > LTX2UnwindGuard.rank(currentPhase) {
        if let m = tv.phases[phase.rawValue]?.meanSec { remainingPhaseMeans.append(m) }
      }
    }
    if let eta = preemptionRefusalETA(
      stepsRemaining: stepsRemaining, meanStepSec: tv.meanStepSec,
      remainingPhaseMeansSec: remainingPhaseMeans, evictReloadRoundTripSec: combinedEvictReloadRoundTripSec()
    ) {
      preemptionInFlight.clear()
      return .refused(eta: eta)
    }

    // #1479 (review C1, second half): re-check right before raising — the
    // render may have finished on its own in the window between the
    // isRendering()/guard checks above and here (telemetry/guard evaluation
    // takes real, if small, time). Raising into nothing would strand the
    // signal for a FUTURE, unrelated render to observe at its very first
    // unwind point (`resume == nil && isRaised` — instant, near-zero-cost
    // false checkpoint; see the `.localVideo` defer fix for the other half
    // of this bug).
    guard videoHolder.isRendering() else {
      preemptionInFlight.clear()
      return .notApplicable
    }

    logger.info("#1479: preempting in-flight video render for image job (source=\(source))")

    // Checkpoint-failure fallback window (spec, Error handling: "Checkpoint
    // fails -> refuse the preemption, keep rendering"): if the render doesn't
    // observe the signal and yield within this window — it may be deep in an
    // uninterruptible phase, or may simply finish on its own first — clear
    // the signal and run the image job WITHOUT preemption rather than lose it
    // or hang forever waiting for a yield that may never come.
    let windowSec = tv.meanStepSec.map { $0 * 2 + 30 } ?? 120

    do {
      let response: GenerateResponse = try await withCheckedThrowingContinuation { (cont: CheckedContinuation<GenerateResponse, Error>) in
        // #1479 (review I1): the token minted here is what makes this
        // episode's watchdog harmless once the episode is over. Without it a
        // watchdog that wakes late — after its own episode completed and a
        // LATER image job armed the box and raised a fresh signal — would
        // claim the NEXT preemptor, clear its raise and its
        // `preemptionInFlight` flag, and quietly run it unpreempted while its
        // continuation was still outstanding. `claim(matching:)` below only
        // succeeds against the entry this very call parked.
        let episodeToken = pendingPreemptorBox.set(
          .init(payload: payload, source: source, rawBody: rawBody, jobId: jobId, continuation: ContinuationBox(cont)))
        ltx2PreemptionSignal.raise()
        Task { [weak self] in
          try? await Task.sleep(nanoseconds: UInt64(max(0, windowSec) * 1_000_000_000))
          guard let self, let claimed = self.pendingPreemptorBox.claim(matching: episodeToken) else { return }
          self.logger.error("#1479: checkpoint fallback — video render did not yield within \(windowSec)s; running image job without preemption")
          self.ltx2PreemptionSignal.clear()
          self.preemptionInFlight.clear()
          do {
            let result = try await self.coordinator.enqueueGenerate(claimed.payload, source: claimed.source, rawBody: claimed.rawBody)
            claimed.continuation.resume(returning: result)
          } catch {
            claimed.continuation.resume(throwing: error)
          }
        }
      }
      return .ran(response)
    } catch {
      return .ranFailed(error)
    }
  }

  // Queue ----------------------------------------------------------------------

  /// GET /v1/queue: the active operation + every pending job (cancellable by id).
  /// GET /v1/queue — served from the lock-based ``LiveHealthState`` snapshot
  /// instead of `await coordinator.queueSnapshot()` so the Queue tab stays
  /// responsive during a render, matching the /health fix (#217). Hopping
  /// onto the actor here queued this request behind the whole render and
  /// also read `isRendering` off a stale field, so the tab showed an empty,
  /// not-rendering queue while a job was actually active.
  private func queueListResponse() async -> RoutedResponse {
    let (snap, progress) = liveHealth.read()
    let iso = ISO8601DateFormatter()
    var payload: [String: Any] = [
      "is_rendering": snap.isRendering,
      "is_paused": snap.isPaused,
      "max_pending": snap.maxPending,
      "render_count": snap.renderCount,
      "failed_count": snap.failedRenderCount,
      "pending": snap.pending.map { job in
        [
          "id": job.id,
          "kind": job.kind,
          "summary": job.summary,
          "source": job.source,
          "enqueued_at": iso.string(from: job.enqueuedAt),
        ] as [String: Any]
      },
    ]
    if let id = snap.activeJobId { payload["active_job_id"] = id }
    if let summary = snap.activeSummary { payload["active_summary"] = summary }
    if let source = snap.activeSource { payload["active_source"] = source }
    if let started = snap.activeRenderStartedAt { payload["active_started_at"] = iso.string(from: started) }
    if let pct = progress { payload["progress_percent"] = pct }
    // #1479: LTX-2 phase telemetry — additive, lock-based (no actor hop), so
    // this stays as responsive during a render as the rest of this route.
    let tv = ltx2Telemetry.view()
    if let phase = tv.currentPhase { payload["phase"] = phase }
    if let m = tv.maxUninterruptibleSec { payload["max_uninterruptible_sec"] = m }
    payload["phase_timings"] = tv.phases.mapValues { ["mean_sec": $0.meanSec, "samples": $0.samples] }
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
    /// Target model family: "image" (Z-Image, default) or "video" (LTX). Selects
    /// the prompt FORMAT — image uses YOUR CONTEXT/YOUR PHOTO, video uses LTX-2.3
    /// cinematic prose.
    let mediaKind: String?
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
      contentMode: mode,
      mediaKind: req.mediaKind ?? "image"
    )

    // Task #19 (Codex finding #6): a server-minted attempt id bound to
    // input, result, template and outcome — render submissions reference
    // this instead of shipping client-echoed strings. Persisted as a trace
    // event so the lineage survives the 1h job prune.
    let attemptId = "opt-" + UUID().uuidString
    renderTraceStore.append(RenderTraceEvent(
      renderId: attemptId, event: .terminal, taskKind: .videoRender,
      payload: [
        "kind": "optimization_attempt",
        "intent": req.prompt,
        "optimized": result.prompt,
        "outcome": result.outcome,
        "template_id": result.templateId ?? "",
        "template_hash": result.templateHash ?? "",
        "template_source": result.templateSource ?? "",
        "media_kind": req.mediaKind ?? "image",
        "content_mode": mode,
      ]))

    var payload: [String: Any] = [
      "success": true,
      "prompt": result.prompt,
      "enhanced": result.enhanced,
      "optimization_attempt_id": attemptId,
      "optimizer_outcome": result.outcome,
    ]
    if let tid = result.templateId { payload["template_id"] = tid }
    if let th = result.templateHash { payload["template_hash"] = th }
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
    Self.presetsList(store: presetStore)
  }

  /// `GET /v1/presets` (WP-E20, AC-44c): every preset, flat, with
  /// `invalid` / `invalid_reason` so a flagged preset is visible and
  /// un-selectable by the desktop app, the bridge and MCP alike.
  static func presetsList(store: PresetStore) -> RoutedResponse {
    .json(status: 200, payload: store.listing())
  }

  private func getPresetResponse(rawId: String) -> RoutedResponse {
    guard let id = Self.pathIdComponent(rawId) else {
      return .error(.error(status: 400, message: "Invalid preset id"))
    }
    // WP-E20: the single-preset read carries the same validity flag as the list.
    guard let entry = presetStore.listing().first(where: { $0.preset.id == id }) else {
      return .error(.error(status: 404, message: "Preset not found: \(id)"))
    }
    return .json(status: 200, payload: entry)
  }

  private func upsertPresetResponse(body: Data) -> RoutedResponse {
    let (response, saved) = Self.upsertPreset(store: presetStore, body: body)
    if let saved {
      auditLog.append(kind: "preset.upsert", message: "Upserted preset \(saved.id)", metadata: ["id": saved.id])
    }
    return response
  }

  /// `POST`/`PUT /v1/presets` (WP-E20, AC-44b): decode, validate through
  /// `PresetStore.upsert` (O4a kroma rule, recipe-name resolution, ranges)
  /// and persist. A refused preset is a 400 naming the preset and the field;
  /// nothing is stored. Returns the saved preset for the audit log.
  static func upsertPreset(store: PresetStore, body: Data) -> (RoutedResponse, saved: ImagePreset?) {
    do {
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      let preset = try decoder.decode(ImagePreset.self, from: body)
      let saved = try store.upsert(preset)
      return (.json(status: 200, payload: saved), saved)
    } catch let error as PresetStoreError {
      return (presetErrorResponse(error), nil)
    } catch {
      return (.error(.error(status: 400, message: "Invalid preset payload: \(error.localizedDescription)")), nil)
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
    Self.resolvePreset(store: presetStore, body: body)
  }

  /// `POST /v1/presets/resolve`. A preset flagged invalid at load (WP-E20,
  /// AC-44c) is a 400 naming it and the reason — it can never be selected.
  static func resolvePreset(store: PresetStore, body: Data) -> RoutedResponse {
    struct ResolveRequest: Decodable { let id: String }
    do {
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      let request = try decoder.decode(ResolveRequest.self, from: body)
      let resolved = try store.resolve(request.id)
      return .json(status: 200, payload: resolved)
    } catch let error as PresetStoreError {
      return presetErrorResponse(error)
    } catch {
      return .error(.error(status: 400, message: #"Invalid resolve request (expected {"id": ...}): \#(error.localizedDescription)"#))
    }
  }

  /// Map a ``PresetStoreError`` to the right HTTP status: validation -> 400,
  /// invalid (flagged on disk) -> 400, notFound -> 404.
  private static func presetErrorResponse(_ error: PresetStoreError) -> RoutedResponse {
    switch error {
    case .validation(let message):
      return .error(.error(status: 400, message: message))
    case .invalid:
      return .error(.error(status: 400, message: error.description))
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
  private static let krea2ControlLoRAPath = "/Volumes/Bolt/Models/krea2-controlnet/depth-control-lora.safetensors"

  private func bridgeGenerate(_ request: ComfyBridgeGenerateRequest, progressCallback: ComfyBridgeProgressHandler?, latentPreviewCallback: ComfyBridgeLatentPreviewHandler? = nil) async throws -> ComfyBridgeGenerateResult {
    // --- Phase 4: Dynamic LoRA swap ---
    // If the workflow contains LoraLoader nodes, swap LoRAs before generating.
    // The coordinator serializes operations, so swap completes before generate starts.
    if !request.loras.isEmpty {
      let loraEntries = request.loras.map { lora -> LoRAEntry in
        // Resolve the LoRA name to a path. Prefer an uploaded LoRA in the bridge
        // dir; otherwise pass the BARE filename so the applicator resolves it
        // against the LoRA library — the same resolution /v1/lora/swap uses.
        // (Blindly prepending the upload dir turned resolvable library LoRAs like
        // "Anneliese_Zbase3.safetensors" into non-existent paths → fileNotFound.)
        let resolvedPath: String
        if lora.name.contains("/") || lora.name.hasPrefix("~") {
          resolvedPath = (lora.name as NSString).expandingTildeInPath
        } else {
          let uploadPath = Self.loraDirectoryPath + "/" + lora.name
          resolvedPath = FileManager.default.fileExists(atPath: uploadPath) ? uploadPath : lora.name
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
    // ControlNet is not supported for Flux 2 or Krea-2 models.
    if request.isControlNet, let controlnetModel = request.controlnetModel {
      let family = await coordinator.modelFamily
      if family == .flux2 || family == .krea2 {
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
    case .krea2:
      // WP-E19 (FDD §3.5, D13, AC-5a): split by the PHYSICAL variant the
      // engine loaded. .turbo is byte-identical to today (clamp 12, guidance
      // 0, negative dropped); .raw takes what Krita sent — no clamp, CFG and
      // the negative prompt live, variant defaults only when absent. A krea2
      // family with no known variant is a fault, never "turbo".
      let resolution = try BridgeKrea2Arm.resolve(request, variant: await coordinator.currentKrea2Variant)
      resolvedSteps = resolution.steps
      resolvedGuidance = resolution.guidance
      resolvedNegativePrompt = resolution.negativePrompt
      resolvedSampler = resolution.sampler
      let clampNote: String = resolution.stepsClamped ? " (clamped from \(request.steps))" : ""
      let negativeNote: String = resolution.negativePrompt.map { "\($0.count) chars" } ?? "none"
      let samplerNote: String = resolution.sampler ?? "nil"
      let scheduleNote: String = request.sigmaSchedule ?? "nil"
      let armLine = "WarmServer: bridge krea2 arm (\(resolution.variant.rawValue)): steps=\(resolution.steps)\(clampNote) "
        + "guidance=\(resolution.guidance) (requested \(request.guidance)) negative=\(negativeNote) "
        + "sampler=\(samplerNote) sigma_schedule=\(scheduleNote)"
      logger.info("\(armLine)")
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

    // One constructor for every family arm (BridgeKrea2Arm.swift) so the
    // field set is asserted once, field-for-field (AC-5a).
    let payload = request.makeGeneratePayload(
      width: genWidth, height: genHeight,
      steps: resolvedSteps, guidance: resolvedGuidance,
      negativePrompt: resolvedNegativePrompt, sampler: resolvedSampler)
    // WP-E4 (§3.4): the bridge builds its payload directly, so it runs the
    // same fail-loud name resolution the /v1/generate decoder does — a Krita
    // style whose sampler we do not implement is refused by name, never
    // rendered as euler. The krea2 tier gates run here too (D18).
    let recipeNames = try payload.validateRecipeNames()
    // I5: the bridge builds its payload directly and forwards the request's
    // sampler on every family arm, so it runs the same family capability gate
    // the REST dispatch does — a Krita style naming a sampler this family
    // cannot drive is refused, never rendered as euler under that name.
    if let error = GeneratePayload.validateFamilyRecipe(recipeNames, family: family) {
      throw error
    }
    // WP-E17: the bridge builds its payload directly too, so it runs the same
    // stage-2 gate. It never SETS `stage2` today; the gate is here so that
    // adding it later cannot skip the family check.
    if let error = GeneratePayload.stage2Gate(payload, family: family) {
      throw error
    }
    if family == .krea2 {
      try payload.validateKrea2TierGates(recipeNames)
    }

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

  // MARK: - Workflows (#238)

  private struct WorkflowImportPayload: Decodable {
    let name: String?
    /// The ComfyUI workflow as a JSON string. Objects also accepted via the
    /// raw-body fallback in the handler.
    let workflowJson: String?
  }

  private func handleWorkflowImport(body: Data) async throws -> RoutedResponse {
    // Accept {name, workflow_json: "<string>"} or {name, workflow: {...}} or a
    // bare graph as the whole body.
    var name = "imported-workflow"
    var graphData: Data = body
    if let envelope = try? JSONSerialization.jsonObject(with: body) as? [String: Any] {
      if let n = envelope["name"] as? String, !n.isEmpty { name = n }
      if let s = envelope["workflow_json"] as? String, let d = s.data(using: .utf8) {
        graphData = d
      } else if let obj = envelope["workflow"] as? [String: Any] {
        graphData = try JSONSerialization.data(withJSONObject: obj)
      }
    }

    let graph = try WorkflowStore.apiGraph(fromJSON: graphData)
    // Dry-run normalization + parse so the compat report reflects
    // runnability without touching input files.
    var parses = true
    var parseError: String? = nil
    do {
      let normalized = try WorkflowStore.normalizeGenericNodes(graph, stageImage: nil)
      _ = try ComfyBridgeWorkflowParser.parseWorkflow(
        ["prompt": normalized, "prompt_id": "import-validate", "client_id": "workflow-api"])
    } catch {
      parses = false
      parseError = "\(error)"
    }

    let workflow = StoredWorkflow(
      id: UUID().uuidString.lowercased(),
      name: name,
      importedAt: Date(),
      graph: graph,
      compat: WorkflowStore.compatReport(for: graph, parses: parses, parseError: parseError))
    try workflowStore.save(workflow)
    logger.info("Workflows: imported '\(name)' (\(workflow.compat.nodeCount) nodes, parses=\(parses)) as \(workflow.id)")
    auditLog.append(kind: "workflow.import", message: "\(name) -> \(workflow.id)", metadata: [:])
    if let data = try? JSONSerialization.data(withJSONObject: workflow.summaryJSON()) {
      return .json(.rawJSON(status: 200, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize import result"))
  }

  private struct WorkflowRunPayload: Decodable {
    let prompt: String?
    let negativePrompt: String?
    let seed: UInt64?
    let outputPath: String?
    let timeoutS: Double?
  }

  /// Pending workflow runs: run id → (workflow id, requested output path).
  /// The status route consumes this to place the finished image on disk.
  private let workflowRunLock = NSLock()
  private var workflowRuns: [String: (workflowId: String, outputPath: String?, resolved: String?)] = [:]

  /// Submit a workflow run. Long renders (base models run 40 steps — several
  /// minutes) outlive the HTTP connection, so this is async by design: 202 +
  /// run_id, polled via GET /v1/workflows/runs/{run_id} — the same convention
  /// as /v1/generate/async and /v1/video/generate/async.
  private func handleWorkflowRun(id: String, body: Data) async throws -> RoutedResponse {
    guard let workflow = workflowStore.get(id) else {
      return .error(.error(status: 404, message: "Workflow not found: \(id)"))
    }
    let payload: WorkflowRunPayload = body.isEmpty
      ? WorkflowRunPayload(prompt: nil, negativePrompt: nil, seed: nil, outputPath: nil, timeoutS: nil)
      : try decode(WorkflowRunPayload.self, from: body)

    // Normalize with REAL staging: LoadImage files go into the bridge cache.
    let normalized = try WorkflowStore.normalizeGenericNodes(workflow.graph) { data in
      let cacheId = "wf-\(UUID().uuidString.lowercased())"
      guard self.comfyBridge.imageCache.store(id: cacheId, data: data) else {
        throw WorkflowError.storeFailed("image cache store failed")
      }
      return cacheId
    }

    let promptId = "wfrun-\(UUID().uuidString.lowercased())"
    try comfyBridge.submitWorkflowGraph(
      normalized,
      promptId: promptId,
      promptOverride: payload.prompt,
      negativePromptOverride: payload.negativePrompt,
      seedOverride: payload.seed)
    workflowRunLock.lock()
    workflowRuns[promptId] = (workflow.id, payload.outputPath, nil)
    workflowRunLock.unlock()
    logger.info("Workflows: run \(promptId) of '\(workflow.name)' submitted")

    let response: [String: Any] = [
      "run_id": promptId,
      "workflow_id": workflow.id,
      "status": "queued",
    ]
    if let data = try? JSONSerialization.data(withJSONObject: response) {
      return .json(.rawJSON(status: 202, data: data))
    }
    return .error(.error(status: 500, message: "Failed to serialize run submission"))
  }

  private func handleWorkflowRunStatus(runId: String) -> RoutedResponse {
    workflowRunLock.lock()
    let run = workflowRuns[runId]
    workflowRunLock.unlock()
    guard let run else {
      return .error(.error(status: 404, message: "Workflow run not found: \(runId) (runs are tracked in memory — a server restart loses them)"))
    }

    func respond(_ dict: [String: Any]) -> RoutedResponse {
      if let data = try? JSONSerialization.data(withJSONObject: dict) {
        return .json(.rawJSON(status: 200, data: data))
      }
      return .error(.error(status: 500, message: "Failed to serialize run status"))
    }

    // Already finalized on a previous poll.
    if let resolved = run.resolved {
      return respond(["run_id": runId, "workflow_id": run.workflowId,
                      "status": "succeeded", "output_path": resolved])
    }
    guard let entry = comfyBridge.history.entry(for: runId) else {
      return respond(["run_id": runId, "workflow_id": run.workflowId, "status": "running"])
    }
    // History filenames carry a .png suffix for ComfyUI /view compatibility;
    // the cache is keyed on the bare id — try both.
    let imageData: Data? = Self.firstImageId(inHistoryEntry: entry).flatMap { imageId in
      comfyBridge.imageCache.retrieve(id: imageId)
        ?? comfyBridge.imageCache.retrieve(id: (imageId as NSString).deletingPathExtension)
    }
    guard let imageData else {
      var detail = "no output image recorded"
      if let status = entry["status"],
         let statusData = try? JSONSerialization.data(withJSONObject: status),
         let statusText = String(data: statusData, encoding: .utf8) {
        detail = String(statusText.prefix(400))
      }
      return respond(["run_id": runId, "workflow_id": run.workflowId,
                      "status": "failed", "error": detail])
    }

    // Success: place the image at the requested (contained) path exactly once.
    do {
      let requestedRaw = run.outputPath ?? "workflow-\(run.workflowId.prefix(8))-\(runId.suffix(8)).png"
      let requested = requestedRaw.hasPrefix("/") || requestedRaw.hasPrefix("~")
        ? requestedRaw
        : (configuration.allowedOutputDirectory as NSString).appendingPathComponent(requestedRaw)
      let resolved = try WarmServerOutputPathValidator.resolveOutputPath(
        requested, allowedOutputDirectory: configuration.allowedOutputDirectory)
      try imageData.write(to: resolved)
      workflowRunLock.lock()
      workflowRuns[runId] = (run.workflowId, run.outputPath, resolved.path)
      workflowRunLock.unlock()
      logger.info("Workflows: run \(runId) -> \(resolved.path)")
      auditLog.append(kind: "workflow.run", message: "\(run.workflowId) -> \(resolved.path)", metadata: [:])
      return respond(["run_id": runId, "workflow_id": run.workflowId,
                      "status": "succeeded", "output_path": resolved.path])
    } catch {
      return respond(["run_id": runId, "workflow_id": run.workflowId,
                      "status": "failed", "error": "output write failed: \(error.localizedDescription)"])
    }
  }

  /// Dig the first output image id out of a bridge history entry
  /// ({outputs: {<nodeId>: {images: [{filename,...}]}}} — the bridge stores
  /// cache ids in `filename`).
  static func firstImageId(inHistoryEntry entry: [String: Any]) -> String? {
    guard let outputs = entry["outputs"] as? [String: Any] else { return nil }
    for value in outputs.values {
      guard let node = value as? [String: Any],
            let images = node["images"] as? [[String: Any]] else { continue }
      for image in images {
        if let filename = image["filename"] as? String, !filename.isEmpty {
          return filename
        }
      }
    }
    return nil
  }

  // MARK: - Storyboard (#237)

  /// Wire payload for POST /v1/storyboard/render (snake_case).
  struct StoryboardPayload: Decodable {
    struct InsertSpec: Decodable {
      let prompt: String
      let creativity: Double?
      let negativePrompt: String?
      let maskPath: String?
      let maskRegion: String?
      let maskInvert: Bool?
      let maskGrow: Int?
      let maskFeather: Int?
      let seed: UInt64?
    }
    struct ShotSpec: Decodable {
      let prompt: String
      let durationS: Double?
      let anchorImage: String?
      let insert: InsertSpec?
      let negativePrompt: String?
      let seed: UInt64?
    }
    struct TransitionSpec: Decodable {
      let type: String
      let durationS: Double?
    }
    struct OutputSpec: Decodable {
      let width: Int?
      let height: Int?
      let fps: Int?
      let path: String?
    }
    let shots: [ShotSpec]
    let transitions: [TransitionSpec]?
    let output: OutputSpec?
    let loras: [LoRAEntry]?
    let source: String?
  }

  private func storyboardSpec(from payload: StoryboardPayload) throws -> StoryboardSpec {
    var transitions: [MontageTransition] = []
    for t in payload.transitions ?? [] {
      guard let kind = MontageTransition.Kind(rawValue: t.type) else {
        throw WarmServerError.invalidRequest(
          message: "transitions[].type must be cut|fade|dissolve (got '\(t.type)')")
      }
      transitions.append(MontageTransition(kind: kind, durationS: t.durationS ?? 0.5))
    }
    let shots = payload.shots.map { s in
      StoryboardSpec.Shot(
        prompt: s.prompt,
        durationS: s.durationS,
        anchorImage: s.anchorImage,
        insert: s.insert.map { ins in
          StoryboardSpec.Insert(
            prompt: ins.prompt,
            creativity: ins.creativity ?? 0.35,
            negativePrompt: ins.negativePrompt,
            maskPath: ins.maskPath,
            maskRegion: ins.maskRegion,
            maskInvert: ins.maskInvert ?? false,
            maskGrow: ins.maskGrow ?? 0,
            maskFeather: ins.maskFeather ?? 0,
            seed: ins.seed)
        },
        negativePrompt: s.negativePrompt,
        seed: s.seed)
    }
    return StoryboardSpec(
      shots: shots,
      transitions: transitions,
      output: StoryboardSpec.Output(
        width: payload.output?.width ?? 640,
        height: payload.output?.height ?? 640,
        fps: payload.output?.fps ?? 24,
        path: payload.output?.path),
      loras: (payload.loras ?? []).map { ($0.path, $0.scale ?? 1.0) })
  }

  /// Execute a storyboard: per-shot [i2i insert →] i2v render, chained on the
  /// previous shot's extracted last frame, then montage assembly. Runs OUTSIDE
  /// the GPU queue (via submitOrchestrated) and takes a normal queue turn for
  /// each render, so other jobs interleave between shots. Intermediates are
  /// named storyboard-* (NOT ltx2-*) so the daemon's orphan reconciler ignores
  /// them.
  private func runStoryboard(
    spec: StoryboardSpec,
    source: String,
    report: @escaping @Sendable (Int) -> Void
  ) async throws -> LTX2VideoResult {
    let started = Date()
    let session = UUID().uuidString.prefix(8)
    let dir = configuration.allowedOutputDirectory
    var anchor: String? = nil
    var clips: [String] = []
    // Assembly gets the final ~4%; shots split the rest evenly.
    let shotWeight = 96.0 / Double(spec.shots.count)

    for (i, shot) in spec.shots.enumerated() {
      var shotAnchor: String
      if let explicit = shot.anchorImage, !explicit.isEmpty {
        guard FileManager.default.fileExists(atPath: explicit) else {
          throw StoryboardError.anchorNotFound(shot: i, path: explicit)
        }
        shotAnchor = explicit
      } else if let chained = anchor {
        shotAnchor = chained
      } else {
        throw StoryboardError.shotFailed(shot: i, stage: "anchor", message: "no previous last frame to chain from")
      }

      // Optional i2i insert on the anchor (e.g. add an element i2v can't
      // invent) — uses the #239 selective-inpainting path.
      if let insert = shot.insert {
        logger.info("Storyboard[\(session)] shot \(i): i2i insert (creativity \(insert.creativity))")
        let insertOut = (dir as NSString)
          .appendingPathComponent("storyboard-\(session)-shot\(i)-insert.png")
        let payload = GeneratePayload(
          prompt: insert.prompt,
          negativePrompt: insert.negativePrompt,
          width: spec.output.width,
          height: spec.output.height,
          seed: insert.seed,
          outputPath: insertOut,
          maskGrow: insert.maskGrow,
          maskFeather: insert.maskFeather,
          imagePath: shotAnchor,
          creativity: Float(insert.creativity),
          maskPath: insert.maskPath,
          maskRegion: insert.maskRegion,
          maskInvert: insert.maskInvert,
          source: source)
        do {
          let generated = try await coordinator.enqueueGenerate(payload, source: source)
          guard FileManager.default.fileExists(atPath: generated.outputPath) else {
            throw StoryboardError.shotFailed(shot: i, stage: "insert", message: "no output produced")
          }
          shotAnchor = generated.outputPath
        } catch let error as StoryboardError {
          throw error
        } catch {
          throw StoryboardError.shotFailed(shot: i, stage: "insert", message: error.localizedDescription)
        }
      }

      // i2v render for this shot, through the same preparation the video
      // routes use (dims snapping, presets, LoRA resolution, validation).
      logger.info("Storyboard[\(session)] shot \(i)/\(spec.shots.count): i2v from \((shotAnchor as NSString).lastPathComponent)")
      var body: [String: Any] = [
        "prompt": shot.prompt,
        "image_path": shotAnchor,
        "width": spec.output.width,
        "height": spec.output.height,
        "fps": spec.output.fps,
        "output_path": "storyboard-\(session)-shot\(i).mp4",
        "source": source,
      ]
      if let d = shot.durationS { body["duration"] = d }
      if let n = shot.negativePrompt { body["negative_prompt"] = n }
      if let s = shot.seed { body["seed"] = s }
      if !spec.loras.isEmpty {
        body["loras"] = spec.loras.map { ["path": $0.path, "scale": $0.scale] as [String: Any] }
      }
      let shotResult: LTX2VideoResult
      do {
        let bodyData = try JSONSerialization.data(withJSONObject: body)
        guard let prep = try await prepareLocalVideo(body: bodyData) else {
          throw StoryboardError.shotFailed(shot: i, stage: "i2v", message: "local LTX-2 not configured")
        }
        let base = Double(i) * shotWeight
        shotResult = try await coordinator.enqueueLocalVideo { coordReport in
          // #1479: storyboard shots stay on the legacy, non-preemptible
          // entry — the checkpoint's generator-level context travels with a
          // single request, and a multi-shot orchestration issues its own
          // per-shot coordinator enqueues (see `submitOrchestrated`'s doc
          // comment), which the preemption episode does not model. `.generate`
          // never yields, so this is always `.completed`.
          .completed(try prep.generator.generate(prep.request) { chunk, totalChunks, step, totalSteps in
            let pct = Self.localVideoProgressPercent(
              chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps)
            coordReport(pct)
            report(Int(base + Double(pct) * shotWeight / 100.0))
          })
        }
      } catch let error as StoryboardError {
        throw error
      } catch {
        throw StoryboardError.shotFailed(shot: i, stage: "i2v", message: error.localizedDescription)
      }
      clips.append(shotResult.outputPath)

      // Chain: the next shot's default anchor is this shot's last frame.
      if i < spec.shots.count - 1 {
        let framePath = (dir as NSString)
          .appendingPathComponent("storyboard-\(session)-shot\(i)-lastframe.png")
        do {
          anchor = try LastFrameExtractor.extractLastFrame(from: shotResult.outputPath, to: framePath)
        } catch {
          throw StoryboardError.shotFailed(shot: i, stage: "last-frame", message: error.localizedDescription)
        }
      }
    }

    // Assembly (hard cuts unless the spec provided transitions).
    let requestedOutput = spec.output.path ?? "storyboard-\(session).mp4"
    let requested = requestedOutput.hasPrefix("/") || requestedOutput.hasPrefix("~")
      ? requestedOutput
      : (dir as NSString).appendingPathComponent(requestedOutput)
    let resolvedOutput = try WarmServerOutputPathValidator.resolveOutputPath(
      requested, allowedOutputDirectory: dir).path
    logger.info("Storyboard[\(session)]: assembling \(clips.count) shot(s) -> \(resolvedOutput)")
    let montage = try MontageComposer.compose(
      segments: clips.map { MontageSegment(kind: .clip, path: $0) },
      transitions: spec.transitions,
      width: spec.output.width,
      height: spec.output.height,
      fps: spec.output.fps,
      outputPath: resolvedOutput)
    report(100)
    auditLog.append(
      kind: "video.storyboard",
      message: "\(spec.shots.count) shots -> \(resolvedOutput)", metadata: [:])
    return LTX2VideoResult(
      outputPath: montage.outputPath,
      frameCount: montage.frameCount,
      durationSeconds: Float(montage.durationS),
      elapsedSeconds: Date().timeIntervalSince(started))
  }

  // MARK: - Montage (#232)

  /// Wire response for POST /v1/montage/compose.
  struct MontageResponse: Encodable {
    let outputPath: String
    let durationS: Double
    let width: Int
    let height: Int
    let segmentCount: Int
    let frameCount: Int

    enum CodingKeys: String, CodingKey {
      case outputPath = "output_path"
      case durationS = "duration_s"
      case width, height
      case segmentCount = "segment_count"
      case frameCount = "frame_count"
    }
  }

  /// Wire payload for POST /v1/montage/compose (snake_case; see the FDD).
  struct MontagePayload: Decodable {
    struct Segment: Decodable {
      struct KenBurnsSpec: Decodable {
        /// [startScale, endScale]
        let zoom: [Double]?
        /// [[x0,y0],[x1,y1]] normalized output-unit offsets
        let pan: [[Double]]?
      }
      let type: String
      let path: String
      let durationS: Double?
      let kenburns: KenBurnsSpec?
    }
    struct Transition: Decodable {
      let type: String
      let durationS: Double?
    }
    struct Output: Decodable {
      let width: Int?
      let height: Int?
      let fps: Int?
      let path: String?
    }
    let segments: [Segment]
    let transitions: [Transition]?
    let output: Output?
    let aspectPolicy: String?
  }

  private func composeMontage(_ payload: MontagePayload) async throws -> MontageResult {
    var segments: [MontageSegment] = []
    for (i, s) in payload.segments.enumerated() {
      guard let kind = MontageSegment.Kind(rawValue: s.type) else {
        throw WarmServerError.invalidRequest(
          message: "segments[\(i)].type must be image|clip (got '\(s.type)')")
      }
      var kenBurns: MontageSegment.KenBurns? = nil
      if let spec = s.kenburns {
        var kb = MontageSegment.KenBurns()
        if let zoom = spec.zoom {
          guard zoom.count == 2 else {
            throw WarmServerError.invalidRequest(
              message: "segments[\(i)].kenburns.zoom must be [start, end]")
          }
          kb.zoomStart = zoom[0]
          kb.zoomEnd = zoom[1]
        }
        if let pan = spec.pan {
          guard pan.count == 2, pan[0].count == 2, pan[1].count == 2 else {
            throw WarmServerError.invalidRequest(
              message: "segments[\(i)].kenburns.pan must be [[x0,y0],[x1,y1]]")
          }
          kb.panStart = (pan[0][0], pan[0][1])
          kb.panEnd = (pan[1][0], pan[1][1])
        }
        kenBurns = kb
      }
      segments.append(MontageSegment(kind: kind, path: s.path, durationS: s.durationS, kenBurns: kenBurns))
    }

    var transitions: [MontageTransition] = []
    for t in payload.transitions ?? [] {
      guard let kind = MontageTransition.Kind(rawValue: t.type) else {
        throw WarmServerError.invalidRequest(
          message: "transitions[].type must be cut|fade|dissolve (got '\(t.type)')")
      }
      transitions.append(MontageTransition(kind: kind, durationS: t.durationS ?? 0.5))
    }

    let aspectPolicy: MontageAspectPolicy
    if let raw = payload.aspectPolicy {
      guard let parsed = MontageAspectPolicy(rawValue: raw) else {
        throw WarmServerError.invalidRequest(
          message: "aspect_policy must be fill_crop|fit_pad (got '\(raw)')")
      }
      aspectPolicy = parsed
    } else {
      aspectPolicy = .fillCrop
    }

    // Output containment — same convention as the video routes (#219).
    let requestedRaw = payload.output?.path ?? "montage-\(UUID().uuidString).mp4"
    let requested: String
    if requestedRaw.hasPrefix("/") || requestedRaw.hasPrefix("~") {
      requested = requestedRaw
    } else {
      requested = (configuration.allowedOutputDirectory as NSString)
        .appendingPathComponent(requestedRaw)
    }
    let resolvedOutput = try WarmServerOutputPathValidator.resolveOutputPath(
      requested, allowedOutputDirectory: configuration.allowedOutputDirectory).path

    let width = payload.output?.width ?? 448
    let height = payload.output?.height ?? 768
    let fps = payload.output?.fps ?? 30

    logger.info("Montage: \(segments.count) segment(s), \(transitions.count) transition(s) -> \(width)x\(height)@\(fps)")
    let start = Date()
    // Compositing is CPU-bound and synchronous — run it off the request task's
    // executor so a long montage can't starve other routes.
    let result = try await Task.detached(priority: .userInitiated) {
      try MontageComposer.compose(
        segments: segments,
        transitions: transitions,
        width: width, height: height, fps: fps,
        aspectPolicy: aspectPolicy,
        outputPath: resolvedOutput)
    }.value
    let elapsed = Int(Date().timeIntervalSince(start) * 1000)
    logger.info("Montage: wrote \(result.outputPath) (\(String(format: "%.2f", result.durationS))s, \(result.frameCount) frames) in \(elapsed)ms")
    auditLog.append(kind: "montage.compose", message: "\(segments.count) segments -> \(result.outputPath)", metadata: [:])
    return result
  }

  /// Shared decode + validation for both the synchronous and queue-submit
  /// generate routes, so output-path containment can't drift between them.
  private func decodedGeneratePayload(from body: Data) throws -> GeneratePayload {
    var payload = try decode(GeneratePayload.self, from: body)
    // Bytes-uploaded img2img init image (init_image_base64) — write it to a
    // temp file so remote clients don't need a pre-existing server path.
    if let initData = payload.initImageData, payload.imagePath == nil {
      let tempPath = NSTemporaryDirectory() + "zimage-init-\(UUID().uuidString).png"
      try initData.write(to: URL(fileURLWithPath: tempPath))
      payload.imagePath = tempPath
    }
    try payload.validateOutputPath(configuration: configuration)
    // WP-E4 (D22): name resolution + structural checks happen ONCE, here, for
    // /v1/generate, /v1/generate/async and persisted-queue replay alike. An
    // unknown sampler/schedule name is a 400 before anything is enqueued —
    // never euler/flow by coercion. Family-specific tier gates (eta on krea2,
    // …) live in runKrea2Generate, NOT here: the family is unknown at this
    // point and Z-Image `eta` is a shipped parameter (D18, AC-28).
    _ = try payload.validateRecipeNames()
    return payload
  }

  /// Replay any queue jobs left over from before a crash (see
  /// QueuePersistence.swift) — the "active" slot (if any) first, since it
  /// was originally at the front, then everything still pending, in order.
  /// Each job is decoded through the exact same path its live route handler
  /// uses, then re-enqueued with no caller to respond to (fire-and-forget —
  /// the original HTTP connection is long gone by the time a crashed process
  /// restarts). Runs as a detached background task so a large recovered
  /// queue never delays the listener from coming up.
  private func recoverPersistedQueue() {
    guard let state = QueueStateStore.load() else { return }
    let jobs = (state.active.map { [$0] } ?? []) + state.pending
    guard !jobs.isEmpty else { return }
    logger.info("Queue recovery: replaying \(jobs.count) job(s) left over from before a restart")

    Task {
      for job in jobs {
        do {
          switch job.kind {
          case "generate":
            let payload = try decodedGeneratePayload(from: job.rawBody)
            // AC-18: replay under the job's OWN id (the client-visible one for
            // an async job), so a second restart persists the same name.
            _ = try await coordinator.enqueueGenerate(payload, source: job.source, rawBody: job.rawBody, jobId: job.id)
            logger.info("Queue recovery: completed generate job \(job.id)")
          case "lora_swap":
            let payload = stageNearlineLoras(in: try decode(LoRASwapPayload.self, from: job.rawBody))
            _ = try await coordinator.enqueueSwap(payload, rawBody: job.rawBody)
            logger.info("Queue recovery: completed lora_swap job \(job.id)")
          default:
            logger.warning("Queue recovery: unknown job kind '\(job.kind)' for \(job.id), skipping")
          }
        } catch {
          // WP-E4 (D22, AC-18): a persisted job that fails replay is marked
          // FAILED with the reason on its own id (GET /v1/generate/status/{id})
          // and in the audit log — never rendered, never silently dropped.
          logger.error("Queue recovery: job \(job.id) (\(job.kind)) failed — \(error.localizedDescription)")
          if job.kind == "generate" {
            imageJobTracker.recordFailedReplay(jobId: job.id, source: job.source, error: error)
          }
          auditLog.append(
            kind: "queue.recovery_failed",
            message: "job \(job.id) (\(job.kind)) failed replay: \(error.localizedDescription)",
            metadata: ["job_id": job.id, "kind": job.kind, "source": job.source])
        }
      }
    }
  }

  private func response(for error: Error) -> HTTPResponse {
    Self.errorResponse(for: error)
  }

  // MARK: - CivitAI conduit (#234)

  /// Message returned on every /v1/civitai/* route when no API key resolves
  /// via CivitAISecrets — never crash/trap on a missing key.
  private static let civitaiKeyMissingMessage =
    "CivitAI API key not configured. Set --civitai-key on `serve`, export CIVITAI_API_KEY, " +
    "or save a key in the Desktop app's CivitAI settings (shared Keychain entry, " +
    "service com.barkadabrew.comfybox.desktop / account civitai)."

  private func civitaiSearchRoute(request: HTTPRequest) async -> RoutedResponse {
    guard let apiKey = CivitAISecrets.resolve(explicit: configuration.civitaiApiKey) else {
      return .error(.error(status: 503, message: Self.civitaiKeyMissingMessage))
    }
    let q = CivitAISearchQuery(queryParameters: request.queryParameters)
    // P1-1: the site allowlist (CivitAIHostAllowlist, shared with the
    // harvest route) must pass BEFORE a client carrying the Bearer key is
    // ever constructed — an unlisted host would receive the CivitAI key.
    guard let baseURL = q.validatedBaseURL else {
      return .error(.error(status: 400, message: CivitAIHostAllowlist.rejectionMessage(forSite: q.site)))
    }
    let client = CivitAIClient(baseURL: baseURL, apiKey: apiKey)
    do {
      let page = try await client.searchModels(
        query: q.query, types: q.types, baseModel: q.baseModel,
        sort: q.sort, period: q.period, nsfw: q.nsfw, cursor: q.cursor, limit: q.limit)
      let payload = CivitAISearchResponse(
        models: page.items.map(CivitAISearchResultModel.init),
        count: page.items.count,
        nextCursor: page.nextCursor)
      return .json(status: 200, payload: payload)
    } catch {
      return .error(.error(status: 502, message: "CivitAI search failed: \(error.localizedDescription)"))
    }
  }

  private func civitaiHarvestRoute(request: HTTPRequest) async -> RoutedResponse {
    guard let apiKey = CivitAISecrets.resolve(explicit: configuration.civitaiApiKey) else {
      return .error(.error(status: 503, message: Self.civitaiKeyMissingMessage))
    }
    let body: CivitAIHarvestRequestBody
    do {
      body = try request.body.isEmpty
        ? CivitAIHarvestRequestBody()
        : decode(CivitAIHarvestRequestBody.self, from: request.body)
    } catch {
      return .error(.error(status: 400, message: "Invalid harvest request: \(error.localizedDescription)"))
    }
    // P1-1: same shared allowlist as the search route, same reason.
    guard let baseURL = body.validatedBaseURL else {
      return .error(.error(status: 400, message: CivitAIHostAllowlist.rejectionMessage(forSite: body.site)))
    }
    let client = CivitAIClient(baseURL: baseURL, apiKey: apiKey)
    do {
      // Behavior notes (P1-2): the runner clamps body.limit to
      // CivitAIHarvestRunner.maxModelsPerHarvest (200) models per call,
      // upserts page-by-page, and stops after ~60s with truncated: true in
      // the summary — partial results are already persisted.
      let summary = try await CivitAIHarvestRunner.run(client: client, request: body)
      return .json(status: 200, payload: summary)
    } catch {
      return .error(.error(status: 502, message: "CivitAI harvest failed: \(error.localizedDescription)"))
    }
  }

  private func civitaiRepoRoute(request: HTTPRequest) -> RoutedResponse {
    let q = CivitAIRepoQuery(queryParameters: request.queryParameters)
    // Result cap (P2): default 100 entries, raisable via ?limit= up to 500 —
    // never the whole store per request.
    let entries = PromptRepositoryStore.query(
      baseModel: q.baseModel, act: q.act, tag: q.tag, keyword: q.keyword, limit: q.limit)
    let payload = CivitAIRepoResponse(entries: entries, count: entries.count)
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.dateEncodingStrategy = .iso8601
    guard let data = try? encoder.encode(payload) else {
      return .error(.error(status: 500, message: "Failed to serialize prompt repository"))
    }
    return .json(.rawJSON(status: 200, data: data))
  }

  /// Error → HTTP mapping, static so the 400/500 split is unit-testable
  /// without a listening server (WP-E4 `WarmServerRejectionTests`).
  static func errorResponse(for error: Error) -> HTTPResponse {
    switch error {
    case let error as WarmServerCoordinator.ServerError:
      switch error {
      case .queueFull(let maxPending):
        return .error(status: 429, message: "Queue full (\(maxPending) pending max)")
      case .modelOperationQueueFull(let maxPending):
        return .error(
          status: 429,
          message: "Model operation queue full (\(maxPending) pending model operations max)")
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

    // WP-E9 (AC-56): a VAE the caller named that is not on disk, or a file in
    // a key layout the engine cannot name, is the caller's error — named in
    // full, never substituted.
    case let error as Krea2VAESelectionError:
      return .error(status: 400, message: error.localizedDescription)
    case let error as Krea2VAEKeyMapError:
      return .error(status: 400, message: error.localizedDescription)

    case let error as WarmServerError:
      switch error {
      case .loraSwapNotSupported, .controlNetNotSupported:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      case .invalidOutputPath, .invalidRequest:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      // WP-E4: a bad recipe name / key conflict / unimplemented tier is the
      // caller's error, named in full (AC-15, AC-28).
      case .unknownSampler, .unknownSigmaSchedule, .mutuallyExclusive, .unsupportedRecipeField,
           .unsupportedSampler, .orphanField:
        return .error(status: 400, message: error.localizedDescription ?? error.localizedDescription)
      case .flux2NotLoaded, .flux2DetectionFailed, .fiboNotLoaded, .fiboDetectionFailed,
           .chromaNotLoaded, .chromaDetectionFailed, .krea2NotLoaded, .krea2VariantUnknown:
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

    // WP-E4 / WP-E17: the pipeline's own fail-loud refusals are the CALLER's
    // error, not the server's. Every one of these is pre-empted by a wire gate
    // on the paths that have one; a non-server caller — and any path a gate
    // does not cover — must still get a 400 naming the field rather than a 500
    // naming nothing.
    case let error as Krea2ScheduleError:
      return .error(status: 400, message: error.description)
    case let error as Krea2StageError:
      return .error(status: 400, message: error.description)
    case let error as SchedulerFactoryError:
      switch error {
      // "this schedule cannot be built at that step count" is the request's
      // problem. `missingMu` is the engine failing to hand the factory a shift
      // it owns, which is ours.
      case .stepCountBelowMinimum:
        return .error(status: 400, message: error.description)
      case .missingMu:
        return .error(status: 500, message: error.description)
      }

    case let error as DecodingError:
      return .error(status: 400, message: "Invalid JSON body: \(describe(decodingError: error))")

    default:
      return .error(status: 500, message: error.localizedDescription)
    }
  }

  private static func describe(decodingError: DecodingError) -> String {
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

  /// The /health body: the snake_case encoding of `health` plus the video
  /// section, with the telemetry-contract keys ALWAYS present (JSON null when
  /// idle) so clients decode them unconditionally — `current_job_id`,
  /// `progress_percent`, and (WP-E10) `last_recipe`, `model_alias`,
  /// `model_variant`. Static so the contract is unit-testable (`HealthSinkTests`).
  static func healthJSON(_ health: HealthResponse, videoAvailable: Bool, activeVideoJobs: Int) -> Data? {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    guard var healthJSON = try? JSONSerialization.jsonObject(with: encoder.encode(health)) as? [String: Any] else {
      return nil
    }
    healthJSON["current_job_id"] = (health.currentJobId as Any?) ?? NSNull()
    healthJSON["progress_percent"] = (health.progressPercent as Any?) ?? NSNull()
    healthJSON["model_variant"] = (health.modelVariant as Any?) ?? NSNull()
    healthJSON["model_alias"] = (health.modelAlias as Any?) ?? NSNull()
    if healthJSON["last_recipe"] == nil { healthJSON["last_recipe"] = NSNull() }
    healthJSON["video"] = [
      "available": videoAvailable,
      "backend": videoAvailable ? "replicate" : "none",
      "active_jobs": activeVideoJobs,
    ] as [String: Any]
    return try? JSONSerialization.data(withJSONObject: healthJSON, options: [.sortedKeys])
  }

  /// Max render age (ms) before /health flags the render as likely deadlocked.
  private static let healthRenderStaleThresholdMs = 300_000  // 5 minutes (#141)

  /// Assemble the /health payload from the lock-based ``LiveHealthState``
  /// snapshot — NO actor hop — so /health stays responsive during a render (#217).
  private func liveHealthResponse(memoryBytes: UInt64) -> HealthResponse {
    let (snap, progress) = liveHealth.read()
    let uptimeSeconds = Int(Date().timeIntervalSince(serverStartTime))
    let activeAgeMs = snap.activeRenderStartedAt.map { Int(Date().timeIntervalSince($0) * 1000.0) }
    let status = WarmServerHealthStatus.derive(
      shuttingDown: snap.shuttingDown,
      activeRenderAgeMs: activeAgeMs,
      staleThresholdMs: Self.healthRenderStaleThresholdMs)
    return HealthResponse(
      status: status,
      model: snap.model.isEmpty ? (configuration.modelSpec ?? ZImageRepository.id) : snap.model,
      modelFamily: snap.modelFamily,
      modelVariant: snap.modelVariant,
      modelAlias: snap.modelAlias,
      buildSha: BuildInfo.gitSHA,
      textEncoderPath: configuration.textEncoderPath,
      loaded: snap.loaded,
      loras: snap.loras,
      uptimeSeconds: uptimeSeconds,
      renderCount: snap.renderCount,
      failedRenderCount: snap.failedRenderCount,
      pendingCount: snap.pendingCount,
      maxPending: configuration.maxPendingRequests,
      isRendering: snap.isRendering,
      isPaused: snap.isPaused,
      activeRequestAgeMs: activeAgeMs,
      currentJobId: snap.activeJobId,
      progressPercent: progress,
      memoryUsageBytes: memoryBytes,
      memoryUsageMB: memoryBytes / (1024 * 1024),
      lastRenderDurationMs: snap.lastRenderDurationMs,
      lastError: snap.lastError,
      lastRecipe: snap.lastRecipe
    )
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
    // Krea-2 family installs (kroma-v0.2-turbo, krea2-raw, …) live in ONE
    // declared spec→directory table (WP-E5) — seeded from config.json
    // `krea2Models` over the built-in defaults — so this function never grows
    // a second one. The directory is then detected fail-closed.
    if let dir = Krea2ModelDetection.specDirectory(modelId) {
      return dir.path
    }

    let suffixes = ["-q4", "-q8", "-bf16"]
    for suffix in suffixes {
      if modelId.lowercased().hasSuffix(suffix) {
        return parseModelSpec(from: String(modelId.dropLast(suffix.count)))
      }
    }
    return modelId
  }
}

/// Snapshot of the coordinator's health-relevant state, published to the
/// lock-based ``LiveHealthState`` so GET /health can be served WITHOUT hopping
/// onto the ``WarmServerCoordinator`` actor.
///
/// The actor is blocked for the full duration of a synchronous GPU render
/// (seconds to minutes). Routing /health through `await coordinator.health()`
/// made the endpoint queue behind the render and return nothing (HTTP 000) for
/// the render's whole duration, then respond instantly once it finished — the
/// Desktop queue/progress UI and external monitors went stale mid-render (#217).
/// The coordinator publishes this snapshot at each state transition instead.
private struct HealthSnapshot: Sendable {
  var shuttingDown: Bool
  var model: String
  var modelFamily: String
  var modelVariant: String?
  /// WP-E10 "E9b" (AC-34b): the declared alias `model` resolved from
  /// (`krea2-raw`), when the active krea2 spec is one — `model` itself carries
  /// the resolved directory path once `parseModelSpec` has expanded it.
  var modelAlias: String? = nil
  /// WP-E10 sink 3: the record of the last successful Krea 2 render.
  var lastRecipe: AppliedRecordSlot? = nil
  var loaded: Bool
  var loras: [LoRAState]
  var renderCount: Int
  var failedRenderCount: Int
  var pendingCount: Int
  var isRendering: Bool
  var activeRenderStartedAt: Date?
  var activeJobId: String?
  var lastRenderDurationMs: Int?
  var lastError: String?
  /// Queue-specific fields (#217 follow-up: GET /v1/queue also used to hop
  /// onto the actor via `coordinator.queueSnapshot()`, so the Queue tab went
  /// stale during a render exactly like /health used to before this snapshot
  /// existed). Populated alongside everything else in `publishHealth()`.
  var isPaused: Bool = false
  var activeSummary: String?
  var activeSource: String?
  var pending: [WarmServerCoordinator.QueueJobInfo] = []
  var maxPending: Int = 0

  static let initial = HealthSnapshot(
    shuttingDown: false, model: "", modelFamily: WarmModelFamily.flux1.rawValue,
    modelVariant: nil, loaded: false, loras: [], renderCount: 0, failedRenderCount: 0,
    pendingCount: 0, isRendering: false, activeRenderStartedAt: nil, activeJobId: nil,
    lastRenderDurationMs: nil, lastError: nil)
}

/// Lock-based publisher for ``HealthSnapshot`` + live progress. Written on the
/// actor at each state transition and from the off-actor progress callback;
/// read by the /health route with no actor hop, so /health stays responsive
/// throughout a render (#217).
private final class LiveHealthState: @unchecked Sendable {
  private let lock = NSLock()
  private var snapshot = HealthSnapshot.initial
  private var progressPercent: Int?

  func publish(_ s: HealthSnapshot) { lock.lock(); snapshot = s; lock.unlock() }
  func setProgress(_ p: Int?) { lock.lock(); progressPercent = p; lock.unlock() }
  func read() -> (HealthSnapshot, Int?) {
    lock.lock(); defer { lock.unlock() }
    return (snapshot, progressPercent)
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

// MARK: - #1479 preemption support (lock-based, no actor hop)
//
// The coordinator actor is synchronously blocked for the whole duration of an
// in-flight LTX-2 render (`WarmServerCoordinator.processLoop`'s `.localVideo`
// case calls `body(report)` with no internal `await`), so a preempt decision
// that needs to run WHILE that render is in flight cannot itself be an actor
// call — it would simply queue behind the render and only run once the render
// already finished on its own, defeating the whole feature (see the #217
// PreemptionSignal precedent this mirrors). Everything below is therefore
// lock-protected, exactly like `PreemptionSignal`/`LTX2PhaseTelemetry`, and
// lives on `WarmServer` (not the actor) so the `/v1/generate` route handler
// can read/raise it before ever calling into the coordinator.

/// Rolling mean of an observed duration (evict, reload). Recorded by the
/// coordinator when it actually performs the operation during a preemption
/// episode; read with no actor hop by the pre-raise refusal guard.
final class RollingMeanSec: @unchecked Sendable {
  private let lock = NSLock()
  private var sumSec: Double = 0
  private var samples: Int = 0
  func record(_ seconds: Double) { lock.lock(); sumSec += seconds; samples += 1; lock.unlock() }
  func mean() -> Double? { lock.lock(); defer { lock.unlock() }; return samples > 0 ? sumSec / Double(samples) : nil }
}

/// Live "steps remaining" of the in-flight video render, fed from the same
/// per-chunk/per-step progress callback the video path already reports
/// through. Read (no actor hop) by the pre-raise refusal guard; cleared when
/// the render ends (completed or failed).
final class LTX2StepPosition: @unchecked Sendable {
  private let lock = NSLock()
  private var remaining: Int?
  /// #1479 (review I3): a multi-chunk render's true remaining step count is
  /// (steps left in THIS chunk) + (steps per chunk) x (whole chunks left) —
  /// counting only the current chunk's remainder under-projects a multi-
  /// chunk render's remaining time, causing spurious refusals.
  func update(chunk: Int, totalChunks: Int, step: Int, totalSteps: Int) {
    lock.lock()
    let inCurrentChunk = max(0, totalSteps - step)
    let fullChunksLeft = max(0, totalChunks - chunk - 1)
    remaining = inCurrentChunk + fullChunksLeft * max(0, totalSteps)
    lock.unlock()
  }
  func clear() { lock.lock(); remaining = nil; lock.unlock() }
  func read() -> Int? { lock.lock(); defer { lock.unlock() }; return remaining }
}

/// Single-owner lock flag: `trySet()` succeeds for exactly one caller until
/// `clear()`. Used to refuse nested preemption (spec: "a preemptor cannot
/// itself be preempted") without an actor hop.
final class LockedFlag: @unchecked Sendable {
  private let lock = NSLock()
  private var value = false
  @discardableResult
  func trySet() -> Bool {
    lock.lock(); defer { lock.unlock() }
    if value { return false }
    value = true
    return true
  }
  func clear() { lock.lock(); value = false; lock.unlock() }
  func get() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
}

/// A lock-protected, exactly-once single-slot handoff whose occupant carries
/// an *episode token* — a monotonically increasing stamp minted by `set`.
///
/// The token exists because a slot can be re-armed by a LATER episode while an
/// EARLIER episode's timeout watchdog is still asleep (#1479 review I1): the
/// stale watchdog would otherwise wake up, take an unqualified `claim()`, and
/// hijack the *next* episode's occupant — silently degrading a preemption that
/// was about to be honoured and releasing its in-flight flag out from under a
/// live continuation. `claim(matching:)` gives a holder a way to claim only the
/// occupant it armed itself; `claim()` (unqualified) is for the authoritative
/// consumer that always wants whatever is currently parked.
///
/// Generic purely so the claim semantics are unit-testable without building a
/// live `GeneratePayload`/continuation pair (see `PendingPreemptorBoxTests`).
final class TokenedSlot<Value>: @unchecked Sendable {
  private let lock = NSLock()
  private var value: Value?
  private var token: UInt64 = 0

  /// Park `v`, evicting anything already there, and return the episode token
  /// that identifies THIS occupant. Tokens never repeat within a process.
  @discardableResult
  func set(_ v: Value) -> UInt64 {
    lock.lock(); defer { lock.unlock() }
    token &+= 1
    value = v
    return token
  }

  /// Unconditional, exactly-once claim: whoever calls first gets the occupant,
  /// everyone else gets nil.
  func claim() -> Value? {
    lock.lock(); defer { lock.unlock() }
    let v = value
    value = nil
    return v
  }

  /// Token-qualified claim: succeeds only if the slot still holds the exact
  /// occupant `t` was minted for. Returns nil if it was already claimed, or if
  /// the slot has since been re-armed by a later episode.
  func claim(matching t: UInt64) -> Value? {
    lock.lock(); defer { lock.unlock() }
    guard token == t, let v = value else { return nil }
    value = nil
    return v
  }
}

/// The preempting image job, parked between the HTTP route handler raising
/// `PreemptionSignal` and whichever side observes the render yield first — the
/// coordinator's `.localVideo` case (the normal path, which takes the
/// unqualified `claim()`) or the checkpoint-failure watchdog Task (the render
/// never yielded in time, or finished on its own before the signal was
/// observed; it takes `claim(matching:)` with its own episode's token, so a
/// stale watchdog can never hijack a later preemptor — #1479 review I1).
struct PendingPreemptor {
  fileprivate let payload: GeneratePayload
  fileprivate let source: String
  fileprivate let rawBody: Data?
  /// The client-visible async job id, when the preemptor came through
  /// `/v1/generate/async` (AC-18) — the preempting render takes it as its
  /// active-job identity so the persisted snapshot names the same job.
  fileprivate let jobId: String?
  fileprivate let continuation: ContinuationBox<GenerateResponse>

  fileprivate init(payload: GeneratePayload, source: String, rawBody: Data?, jobId: String?, continuation: ContinuationBox<GenerateResponse>) {
    self.payload = payload
    self.source = source
    self.rawBody = rawBody
    self.jobId = jobId
    self.continuation = continuation
  }
}

typealias PendingPreemptorBox = TokenedSlot<PendingPreemptor>

/// Holds the latest live-denoising preview JPEG for polling clients (the
/// Desktop app, which already polls /health for progress_percent — see
/// GH #216). Krita/ComfyUI get previews pushed over their own WebSocket;
/// this is the same frame made available to REST/polling clients instead.
private final class RenderPreviewTracker: @unchecked Sendable {
  private let lock = NSLock()
  private var frame: Data?
  func set(_ value: Data?) { lock.lock(); frame = value; lock.unlock() }
  func get() -> Data? { lock.lock(); defer { lock.unlock() }; return frame }
}

/// State of an async-submitted image generation job — mirrors `VideoJobState`
/// so image and video generation share one submit/poll convention.
public enum ImageJobState: String, Codable, Sendable {
  case queued
  case processing
  case succeeded
  case failed
}

/// Wire status for `POST /v1/generate/async` / `GET /v1/generate/status/{id}`.
public struct ImageJobStatus: Codable, Sendable {
  public let jobId: String
  public let status: ImageJobState
  public let source: String
  public let outputPath: String?
  public let durationMs: Int?
  public let error: String?
  public let elapsedMs: Int
  /// #1479: set when this job asked to preempt an in-flight video render but
  /// the refusal guard declined — the job still ran normally (just queued),
  /// `etaSec` is the guard's projected remaining seconds. Both Optional (not
  /// merely defaulted) so `Codable`'s synthesized decode tolerates JSON that
  /// predates this field.
  public let preemptRefused: Bool?
  public let etaSec: Double?
  /// WP-E10 sink 4 (FDD §3.10): the provenance record of a succeeded Krea 2
  /// job — the same `applied` the sync response carries. Optional so
  /// persisted pre-upgrade JSON still decodes (AC-64); null for other
  /// families (D12) and until the job succeeds.
  public let applied: AppliedRecordSlot?

  /// The record itself; see ``AppliedRecordSlot`` for absent-vs-null.
  public var appliedRecord: RenderRecipe? { applied?.record }

  public init(
    jobId: String, status: ImageJobState, source: String, outputPath: String?, durationMs: Int?,
    error: String?, elapsedMs: Int, preemptRefused: Bool?, etaSec: Double?,
    applied: AppliedRecordSlot? = nil
  ) {
    self.jobId = jobId
    self.status = status
    self.source = source
    self.outputPath = outputPath
    self.durationMs = durationMs
    self.error = error
    self.elapsedMs = elapsedMs
    self.preemptRefused = preemptRefused
    self.etaSec = etaSec
    self.applied = applied
  }
}

/// Internal mutable state for a tracked async image generation job.
private final class ImageJob: @unchecked Sendable {
  let id: String
  let source: String
  let startTime = Date()
  var state: ImageJobState = .queued
  var outputPath: String?
  var durationMs: Int?
  var error: String?
  var completedAt: Date?
  /// #1479: set once, before the job is enqueued, if `attemptPreemption`
  /// refused it.
  var preemptRefused: Bool?
  var etaSec: Double?
  /// WP-E10: set with the result on success (tri-state, see AppliedRecordSlot).
  var applied: AppliedRecordSlot?

  init(id: String, source: String) {
    self.id = id
    self.source = source
  }

  var elapsedMs: Int {
    let end = completedAt ?? Date()
    return Int(end.timeIntervalSince(startTime) * 1000)
  }

  func toStatus() -> ImageJobStatus {
    ImageJobStatus(
      jobId: id, status: state, source: source, outputPath: outputPath,
      durationMs: durationMs, error: error, elapsedMs: elapsedMs,
      preemptRefused: preemptRefused, etaSec: etaSec, applied: applied
    )
  }
}

/// Submit-and-poll wrapper around `WarmServerCoordinator.enqueueGenerate` so
/// callers (Bree's async envelope, the Telegram bot, MCP tools) can fire a
/// render without holding a connection open for the whole denoising run.
/// A blocking `POST /v1/generate` that takes minutes is what orphaned a
/// Telegram delivery in production once: the caller's own turn timeout
/// (180s) expired before the render finished, and there was no live turn
/// left to deliver through. Queue-submit decouples render time from the
/// caller's timeout — submit returns a job id immediately, and the caller
/// polls `GET /v1/generate/status/{id}` (or `/v1/video/status/{id}`'s twin)
/// until it sees `succeeded`/`failed`, exactly like the video path already
/// does via `ReplicateVideoProxy`.
final class ImageJobTracker: @unchecked Sendable {
  private let lock = NSLock()
  private var jobs: [String: ImageJob] = [:]

  /// Submit a job. Returns immediately with `queued` status; the render
  /// itself runs in a detached Task against the existing FIFO render queue,
  /// so submitting async doesn't skip the line ahead of synchronous callers.
  fileprivate func submit(_ payload: GeneratePayload, source: String, coordinator: WarmServerCoordinator, rawBody: Data? = nil) -> ImageJobStatus {
    submit(payload, source: source, rawBody: rawBody) { jobId in
      try await coordinator.enqueueGenerate(payload, source: source, rawBody: rawBody, jobId: jobId)
    }
  }

  /// The enqueue seam (WP-E10 "E9b", AC-18): the tracker's OWN id — the one
  /// the `/v1/generate/async` caller receives — is handed to the queue, so
  /// the persisted job, a failed replay after a restart, and the status
  /// route all name the same job. `enqueue` receives that id.
  func submit(
    _ payload: GeneratePayload, source: String, rawBody: Data?,
    enqueue: @escaping @Sendable (String) async throws -> GenerateResponse
  ) -> ImageJobStatus {
    let jobId = UUID().uuidString
    let job = ImageJob(id: jobId, source: source)
    lock.lock(); jobs[jobId] = job; lock.unlock()

    Task { [weak self] in
      guard let self else { return }
      self.markProcessing(jobId)
      do {
        let result = try await enqueue(jobId)
        self.markSucceeded(jobId, result: result)
      } catch {
        self.markFailed(jobId, error: error)
      }
    }
    return job.toStatus()
  }

  /// #1479: submit-and-poll with a preemption attempt tried FIRST, inside
  /// this job's own detached Task. `preemptor` is `WarmServer.attemptPreemption`
  /// bound to this payload — kept as a closure (rather than a direct
  /// dependency on `WarmServer`) so `ImageJobTracker`'s state machine stays
  /// unit-testable in isolation, matching the file's existing convention.
  /// `.notApplicable` (flag absent/false, no video rendering, nested
  /// preemption) takes the EXACT SAME `coordinator.enqueueGenerate` path
  /// `submit` above does.
  fileprivate func submitPreempting(
    _ payload: GeneratePayload, source: String, coordinator: WarmServerCoordinator, rawBody: Data? = nil,
    preemptor: @escaping @Sendable (String) async -> WarmServer.PreemptionOutcome
  ) -> ImageJobStatus {
    submitPreempting(payload, source: source, rawBody: rawBody, preemptor: preemptor) { jobId in
      try await coordinator.enqueueGenerate(payload, source: source, rawBody: rawBody, jobId: jobId)
    }
  }

  /// Closure-seam twin of `submit(_:source:rawBody:enqueue:)` for the
  /// preempting path; both the preemptor and the enqueue receive the
  /// client-visible id (AC-18).
  func submitPreempting(
    _ payload: GeneratePayload, source: String, rawBody: Data?,
    preemptor: @escaping @Sendable (String) async -> WarmServer.PreemptionOutcome,
    enqueue: @escaping @Sendable (String) async throws -> GenerateResponse
  ) -> ImageJobStatus {
    let jobId = UUID().uuidString
    let job = ImageJob(id: jobId, source: source)
    lock.lock(); jobs[jobId] = job; lock.unlock()

    Task { [weak self] in
      guard let self else { return }
      switch await preemptor(jobId) {
      case .ran(let result):
        self.markSucceeded(jobId, result: result)
        return
      case .ranFailed(let error):
        self.markFailed(jobId, error: error)
        return
      case .refused(let eta):
        self.markPreemptRefused(jobId, eta: eta)
      case .notApplicable:
        break
      }
      self.markProcessing(jobId)
      do {
        let result = try await enqueue(jobId)
        self.markSucceeded(jobId, result: result)
      } catch {
        self.markFailed(jobId, error: error)
      }
    }
    return job.toStatus()
  }

  func status(jobId: String) -> ImageJobStatus? {
    lock.lock(); defer { lock.unlock() }
    return jobs[jobId]?.toStatus()
  }

  /// WP-E4 (D22, AC-18): register a persisted-queue job that failed replay
  /// validation as FAILED under its original id, with the reason, so a client
  /// polling `GET /v1/generate/status/{id}` across a restart sees why it never
  /// rendered instead of a 404.
  func recordFailedReplay(jobId: String, source: String, error: Error) {
    let job = ImageJob(id: jobId, source: source)
    job.state = .failed
    job.error = error.localizedDescription
    job.completedAt = Date()
    lock.lock(); jobs[jobId] = job; lock.unlock()
  }

  private func markProcessing(_ jobId: String) {
    lock.lock(); jobs[jobId]?.state = .processing; lock.unlock()
  }

  /// #1479: the refusal guard declined — this job still queues normally
  /// right after (see `submitPreempting`), just annotated.
  private func markPreemptRefused(_ jobId: String, eta: Double) {
    lock.lock()
    jobs[jobId]?.preemptRefused = true
    jobs[jobId]?.etaSec = eta
    lock.unlock()
  }

  private func markSucceeded(_ jobId: String, result: GenerateResponse) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .succeeded
      job.outputPath = result.outputPath
      job.durationMs = result.durationMs
      job.applied = result.applied
      job.completedAt = Date()
    }
    lock.unlock()
  }

  private func markFailed(_ jobId: String, error: Error) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .failed
      job.error = error.localizedDescription
      job.completedAt = Date()
    }
    lock.unlock()
  }

  /// Drop completed/failed jobs older than `ttl` so this doesn't grow
  /// unboundedly on a long-running server. Mirrors `ReplicateVideoProxy`'s
  /// prune convention.
  func pruneCompleted(olderThan ttl: TimeInterval = 3600) {
    lock.lock(); defer { lock.unlock() }
    let cutoff = Date().addingTimeInterval(-ttl)
    jobs = jobs.filter { _, job in
      guard let completedAt = job.completedAt else { return true }
      return completedAt > cutoff
    }
  }
}

/// Internal mutable state for a tracked async LOCAL LTX-2 video job. Mirrors
/// `ImageJob`, with the extra video fields (`mode`, `frameCount`,
/// `videoDurationSeconds`, live `progressPercent`) the wire `VideoJobStatus`
/// carries.
private final class LocalVideoJob: @unchecked Sendable {
  let id: String
  let source: String
  let mode: VideoMode
  let startTime = Date()
  var state: VideoJobState = .queued
  var outputPath: String?
  var frameCount: Int?
  var videoDurationSeconds: Int?
  var durationMs: Int?
  var error: String?
  var progressPercent: Int?
  var completedAt: Date?
  /// Authoritative config snapshot, set at submit (finding #15).
  var resolvedConfig: [LTX2ResolvedParam]?

  init(id: String, source: String, mode: VideoMode) {
    self.id = id
    self.source = source
    self.mode = mode
  }

  var elapsedMs: Int {
    let end = completedAt ?? Date()
    return Int(end.timeIntervalSince(startTime) * 1000)
  }

  func toStatus() -> VideoJobStatus {
    VideoJobStatus(
      jobId: id,
      status: state,
      mode: mode,
      backend: "ltx2-local",
      outputPath: outputPath,
      durationMs: durationMs,
      videoDurationSeconds: videoDurationSeconds,
      error: error,
      elapsedMs: elapsedMs,
      progressPercent: progressPercent,
      resolvedConfig: resolvedConfig,
      frameCount: frameCount
    )
  }
}

/// Submit-and-poll tracker for LOCAL LTX-2 video renders — the video twin of
/// ``ImageJobTracker``. A local render can run for minutes across multiple
/// chunks; holding the HTTP connection open for the whole thing is what the
/// async `POST /v1/video/generate/async` + `GET /v1/video/status/{id}` pair
/// avoids. Submit returns a job id immediately; the render itself runs on the
/// coordinator's serial GPU queue (so it never shares the GPU with an image
/// render), streaming progress into the job as it goes.
///
/// The state-transition surface (`register`/`markProcessing`/`markSucceeded`/
/// `markFailed`/`setProgress`/`status`/`pruneCompleted`) is deliberately kept
/// free of any coordinator dependency so the state machine is unit-testable in
/// isolation; `submit` is the thin production wrapper that drives it against the
/// real render queue.
final class VideoJobTracker: @unchecked Sendable {
  /// Task #19: append-only lifecycle trace (submitted/started/terminal).
  /// nil in unit tests that only exercise the state machine.
  var traceStore: RenderTraceStore?
  private let lock = NSLock()
  private var jobs: [String: LocalVideoJob] = [:]

  /// Create a tracked job in `.queued` and return (jobId, its status). Testable
  /// without a coordinator.
  @discardableResult
  func register(
    source: String, mode: VideoMode,
    resolvedConfig: [LTX2ResolvedParam]? = nil,
    tracePayload: [String: String] = [:]
  ) -> (jobId: String, status: VideoJobStatus) {
    let jobId = UUID().uuidString
    let job = LocalVideoJob(id: jobId, source: source, mode: mode)
    job.resolvedConfig = resolvedConfig
    lock.lock(); jobs[jobId] = job; lock.unlock()
    var payload = tracePayload
    payload["source"] = source
    payload["mode"] = mode.rawValue
    if let rc = resolvedConfig {
      payload["config"] = rc.map { "\($0.name)=\($0.value)(\($0.source.rawValue))" }.joined(separator: " ")
    }
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .submitted, taskKind: .videoRender, payload: payload))
    return (jobId, job.toStatus())
  }

  /// Submit a local render. Returns immediately with a `.queued` status; the
  /// render runs in a detached Task against the coordinator's FIFO GPU queue.
  /// `render` receives a `report(percent)` callback to stream progress; the
  /// tracker fans that out to both this job's status and (via the coordinator's
  /// own report wired in `enqueueLocalVideo`) the /health + /queue trackers.
  fileprivate func submit(
    source: String,
    mode: VideoMode,
    coordinator: WarmServerCoordinator,
    resolvedConfig: [LTX2ResolvedParam]? = nil,
    tracePayload: [String: String] = [:],
    wantsAudio: Bool = false,
    render: @escaping @Sendable (@escaping @Sendable (Int) -> Void) throws -> LTX2RenderOutcome
  ) -> VideoJobStatus {
    let (jobId, queued) = register(
      source: source, mode: mode, resolvedConfig: resolvedConfig,
      tracePayload: tracePayload)
    Task { [weak self] in
      guard let self else { return }
      self.markProcessing(jobId)
      do {
        // #1479: pass this job's tracker id so the coordinator can mark it
        // paused-for-preemption / resumed by id (see `enqueueLocalVideo`).
        let result = try await coordinator.enqueueLocalVideo(wantsAudio: wantsAudio, videoJobId: jobId) { coordReport in
          try render { pct in
            // Fan progress to both the coordinator's health/queue trackers and
            // this job's own status.
            coordReport(pct)
            self.setProgress(jobId, pct)
          }
        }
        self.markSucceeded(jobId, result: result)
      } catch {
        self.markFailed(jobId, error: error)
      }
    }
    return queued
  }

  /// Submit a multi-step orchestration (storyboard, #237). Unlike `submit`,
  /// the work closure is NOT wrapped in a single `enqueueLocalVideo` — it
  /// issues its OWN coordinator enqueues (one per shot render / i2i insert),
  /// so each step takes a normal turn on the FIFO GPU queue and other jobs
  /// can interleave between shots. Wrapping the whole storyboard in one queue
  /// entry would deadlock: the closure would enqueue from inside the queue.
  fileprivate func submitOrchestrated(
    source: String,
    mode: VideoMode,
    work: @escaping @Sendable (@escaping @Sendable (Int) -> Void) async throws -> LTX2VideoResult
  ) -> VideoJobStatus {
    let (jobId, queued) = register(source: source, mode: mode)
    Task { [weak self] in
      guard let self else { return }
      self.markProcessing(jobId)
      do {
        let result = try await work { pct in
          self.setProgress(jobId, pct)
        }
        self.markSucceeded(jobId, result: result)
      } catch {
        self.markFailed(jobId, error: error)
      }
    }
    return queued
  }

  func status(jobId: String) -> VideoJobStatus? {
    lock.lock(); defer { lock.unlock() }
    return jobs[jobId]?.toStatus()
  }

  func markProcessing(_ jobId: String) {
    lock.lock(); jobs[jobId]?.state = .processing; lock.unlock()
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .started, taskKind: .videoRender, payload: [:]))
  }

  /// #1479: checkpointed for an in-flight preemption — not terminal, no trace
  /// event (the trace store's `.terminal` kind means "done"; this isn't).
  func markPausedForPreemption(_ jobId: String) {
    lock.lock(); jobs[jobId]?.state = .pausedForPreemption; lock.unlock()
  }

  /// #1479: the preempting image job finished (success or failure) and the
  /// video render resumed. Reuses `.processing` — this job was never
  /// "finished" from the tracker's point of view, so there is no distinct
  /// "resumed" state to model.
  func markResumedFromPreemption(_ jobId: String) {
    lock.lock(); jobs[jobId]?.state = .processing; lock.unlock()
  }

  func setProgress(_ jobId: String, _ percent: Int) {
    lock.lock(); jobs[jobId]?.progressPercent = min(100, max(0, percent)); lock.unlock()
  }

  func markSucceeded(_ jobId: String, result: LTX2VideoResult) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .succeeded
      job.outputPath = result.outputPath
      job.frameCount = result.frameCount
      job.videoDurationSeconds = Int(result.durationSeconds.rounded())
      job.durationMs = Int(result.elapsedSeconds * 1000)
      job.progressPercent = 100
      job.completedAt = Date()
    }
    lock.unlock()
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .terminal, taskKind: .videoRender,
      payload: [
        "status": "succeeded",
        "output_path": result.outputPath,
        "frames": String(result.frameCount),
        "elapsed_ms": String(Int(result.elapsedSeconds * 1000)),
      ]))
  }

  func markFailed(_ jobId: String, error: Error) {
    lock.lock()
    if let job = jobs[jobId] {
      job.state = .failed
      job.error = error.localizedDescription
      job.completedAt = Date()
    }
    lock.unlock()
    traceStore?.append(RenderTraceEvent(
      renderId: jobId, event: .terminal, taskKind: .videoRender,
      payload: ["status": "failed", "error": error.localizedDescription]))
  }

  /// Drop completed/failed jobs older than `ttl`. Mirrors `ImageJobTracker`.
  func pruneCompleted(olderThan ttl: TimeInterval = 3600) {
    lock.lock(); defer { lock.unlock() }
    let cutoff = Date().addingTimeInterval(-ttl)
    jobs = jobs.filter { _, job in
      guard let completedAt = job.completedAt else { return true }
      return completedAt > cutoff
    }
  }
}

private actor WarmServerCoordinator {
  enum ServerError: Error {
    case queueFull(maxPending: Int)
    /// The model-operation cap (`maxPendingModelOps`), which is counted and
    /// reported separately from the render queue so the message names the
    /// limit the caller actually hit (WP-E8 review, finding 1).
    case modelOperationQueueFull(maxPending: Int)
    case shuttingDown
    /// The pending request was removed by a queue clear (not a server shutdown).
    case cancelled
  }

  /// #300: this actor's isolated work (including the synchronous render call)
  /// otherwise runs on the Swift cooperative thread pool
  /// (`com.apple.root.utility-qos.cooperative`), which is width-capped at
  /// ~core count and does NOT grow when a worker blocks. A `sample` of the
  /// live process during a render showed 2964/2972 samples parked in
  /// `__psynch_cvwait` on that pool — starving every other actor hop,
  /// including `Task { await respond(...) }` for async HTTP routes (HTTP 000
  /// at 120s while sync routes stayed fine). Giving the coordinator its own
  /// serial executor (SE-0392) moves its work off the shared cooperative pool
  /// entirely; actor serialization semantics are unchanged.
  private let executorQueue = DispatchSerialQueue(label: "z-image.warm-server.coordinator", qos: .userInitiated)
  nonisolated var unownedExecutor: UnownedSerialExecutor {
    executorQueue.asUnownedSerialExecutor()
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

  /// Krea-2 pipeline (native port), loaded when the model spec is Krea-2.
  private var krea2Pipeline: Krea2Pipeline?
  /// The physical Krea-2 variant the resident pipeline loaded (WP-E5, D7) —
  /// beside `zimageVariant`. nil when no Krea-2 model is resident.
  private var krea2Variant: Krea2Variant?
  /// Trigger lookups for the rewriter-proof guard (set by WarmServer.run()).
  var loraLibrary: LoRALibrary?
  func setLoraLibrary(_ library: LoRALibrary) { loraLibrary = library }
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
    /// The job's id everywhere it is named: /v1/queue, the persisted
    /// snapshot, a failed replay. Supplied by the caller when a client already
    /// holds one (AC-18); defaults to a fresh UUID.
    var id: String = UUID().uuidString
    let enqueuedAt = Date()
    /// Which client/app submitted this job (desktop, comfyui/krita, bree, api…).
    var source: String = "api"
    let operation: QueuedOperation
    /// The original raw JSON request body, kept only for kinds that can be
    /// replayed after a crash (see QueuePersistence.swift). nil for kinds
    /// that can't be recovered (modelSwitch/localVideo close over live
    /// in-memory state; controlGenerate closes over resolved temp files;
    /// shutdown never needs recovery).
    var rawBody: Data? = nil
  }

  private var pending: [PendingJob] = []
  /// Human-readable summary of the operation the loop is currently running.
  private var activeJobSummary: String?
  /// Source/app of the currently-running job.
  private var activeJobSource: String?
  private var isProcessing = false
  /// When paused, the process loop finishes the current job (if any) but does
  /// not start pending ones until resumed.
  ///
  /// PERSISTED across restarts via a sentinel file (2026-08-10): the flag was
  /// in-memory only, so any engine restart — watchdog kickstart, crash,
  /// deploy — silently resumed creation. "Paused" that un-pauses itself is
  /// how the July mystery-GPU-usage class of incident happens.
  private var isPaused = FileManager.default.fileExists(atPath: WarmServerCoordinator.pauseSentinelPath)

  /// Sentinel marking the queue paused; survives engine restarts. Computed
  /// from `QueueStateStore.stateDirectory` so it follows `COMFYBOX_STATE_DIR`
  /// (K-FIX-1: a test driving a real coordinator must not read — or clear —
  /// the LIVE engine's pause flag).
  static var pauseSentinelPath: String {
    QueueStateStore.stateDirectory.appendingPathComponent("queue-paused").path
  }
  private var shuttingDown = false
  private var successfulRenderCount = 0
  private var failedRenderCount = 0
  private var lastRenderDurationMs: Int?
  private var lastError: String?
  /// WP-E10 sink 3: the record of the last successful Krea 2 render,
  /// published into /health as `last_recipe`. Set only from a completed
  /// render (a failed one writes no record), never from a request.
  private var lastRecipe: AppliedRecordSlot?

  /// Re-decide whether `lastRecipe` may still be published, from what is
  /// resident RIGHT NOW (WP-E10 sink 3).
  ///
  /// `/health` prints `model`, `loaded`, `model_variant` and `last_recipe`
  /// side by side. A record that outlived its checkpoint reads as provenance
  /// and is not — most starkly during an LTX-2 render, where the whole image
  /// stack is evicted (#218) and `/health` would otherwise show a full Krea 2
  /// provenance block beside `loaded: false` for tens of minutes.
  ///
  /// EVERY writer of `krea2Pipeline` calls this immediately after: pool
  /// activation, `prepare()`, and `releaseImageModelsForVideo()`. Adding a
  /// fourth writer without this call is the bug this comment exists to catch.
  /// A record that no longer describes what is resident is dropped ENTIRELY
  /// (the key goes absent), not turned into a `null` — `null` means "a Krea 2
  /// render just refused its record", which is a different statement.
  private func revalidateLastRecipe() {
    guard let slot = lastRecipe else { return }
    guard let record = slot.record else {
      // A refused record describes a render, not a checkpoint; once the
      // resident model may have changed there is nothing left to say.
      lastRecipe = currentModelFamily == .krea2 ? slot : nil
      return
    }
    lastRecipe = RenderRecipe.retained(
      record,
      family: currentModelFamily,
      krea2TransformerFile: krea2Pipeline?.paths.transformerFile.path
    ).map(AppliedRecordSlot.init(record:))
  }
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
  private let previewTracker = RenderPreviewTracker()
  private var pipelinePrepared = false
  /// When a pool model is activated, this holds its modelSpec so that
  /// generation requests use the pool model instead of the startup
  /// configuration.modelSpec. Reset to nil when the startup model is
  /// re-activated or the pool model is unloaded.
  private var activePoolModelSpec: String?

  /// Model hot-swap pool — holds loaded pipelines with LRU eviction.
  let modelPool: ModelPool

  /// Shared, lock-based owner of the LTX-2 video generator (#218). The video
  /// stack lives outside the pool; this lets the coordinator evict it before an
  /// image load, keeping a single heavy model resident across image + video.
  private let videoHolder: VideoGeneratorHolder

  /// Pure single-heavy-model residency accounting (#218).
  private let heavyAdmission = HeavyModelAdmission()

  /// Lock-based health snapshot the /health route reads without an actor hop (#217).
  private let liveHealth: LiveHealthState
  /// When the current queue job started processing — the /health start time even
  /// before a render method sets `activeRenderStartedAt` past its first await.
  private var currentJobStartedAt: Date?
  /// Raw request body + kind of the currently-active job, kept only long
  /// enough to persist it as the "active" slot in QueueStateStore — cleared
  /// alongside the other activeJob* fields once the job finishes (see
  /// QueuePersistence.swift for why only these two kinds are recoverable).
  private var activeJobRawBody: Data?
  private var activeJobKindForPersistence: String?

  /// True after the image models were released to make room for LTX-2 video —
  /// the next image render must reload before it can run (#218).
  private var imageModelsEvicted = false
  /// The image model that was active when it was evicted for video, so the next
  /// image render can restore exactly that model.
  private var lastActiveImageSpec: String?

  // MARK: - #1479 preemption

  /// Async-video job tracker, so the coordinator can mark the render
  /// paused-for-preemption / resumed by job id (nil for the synchronous
  /// `/v1/video/generate` path, which has no tracker entry).
  private let videoJobTracker: VideoJobTracker
  /// Installed on every LTX-2 generator instance (`prepareLocalVideo` wires
  /// the initial one; re-wired here on a fresh post-eviction instance).
  private let ltx2Telemetry: LTX2PhaseTelemetry
  private let ltx2PreemptionSignal: PreemptionSignal
  private let ltx2StepPosition: LTX2StepPosition
  private let ltx2EvictMean: RollingMeanSec
  private let ltx2ReloadMean: RollingMeanSec
  /// Cleared unconditionally once a preemption episode is fully resolved
  /// (resumed or handed to the checkpoint-fallback watchdog) — see
  /// `runPreemptionEpisode`'s defer.
  private let preemptionInFlight: LockedFlag
  private let pendingPreemptorBox: PendingPreemptorBox
  /// The one checkpoint this coordinator may hold at a time (spec: hold
  /// exactly one checkpoint; nested preemption refused separately via
  /// `preemptionInFlight`). Non-nil only for the span between the video
  /// render yielding and the video being resumed.
  private var checkpointedVideo: LTX2ResumeState?

  init(
    configuration: WarmServerConfiguration, logger: Logger, videoHolder: VideoGeneratorHolder, liveHealth: LiveHealthState,
    videoJobTracker: VideoJobTracker, ltx2Telemetry: LTX2PhaseTelemetry, ltx2PreemptionSignal: PreemptionSignal,
    ltx2StepPosition: LTX2StepPosition, ltx2EvictMean: RollingMeanSec, ltx2ReloadMean: RollingMeanSec,
    preemptionInFlight: LockedFlag, pendingPreemptorBox: PendingPreemptorBox
  ) {
    self.configuration = configuration
    self.logger = logger
    self.videoHolder = videoHolder
    self.liveHealth = liveHealth
    self.videoJobTracker = videoJobTracker
    self.ltx2Telemetry = ltx2Telemetry
    self.ltx2PreemptionSignal = ltx2PreemptionSignal
    self.ltx2StepPosition = ltx2StepPosition
    self.ltx2EvictMean = ltx2EvictMean
    self.ltx2ReloadMean = ltx2ReloadMean
    self.preemptionInFlight = preemptionInFlight
    self.pendingPreemptorBox = pendingPreemptorBox
    self.pipeline = ZImagePipeline(logger: logger, retentionPolicy: .keepLoaded)
    self.activeLoRAs = configuration.initialLoRAs
    self.modelPool = ModelPool(
      textEncoderPath: configuration.textEncoderPath,
      maxSequenceLength: configuration.maxSequenceLength,
      forceTransformerOverrideOnly: configuration.forceTransformerOverrideOnly,
      logger: logger
    )
  }

  // MARK: - Single-heavy-model residency (#218)

  /// Release EVERY resident image model — the pool (including the active model)
  /// and the coordinator's own per-family pipelines — to vacate unified memory
  /// for the ~65GB LTX-2 video stack. Records what was active so the next image
  /// render can restore it. Returns the estimated MB freed.
  @discardableResult
  func releaseImageModelsForVideo() async -> Int {
    lastActiveImageSpec = activePoolModelSpec ?? configuration.modelSpec
    let freedMB = await modelPool.releaseAll()
    pipeline.unloadModel()
    flux2Pipeline = nil
    fiboPipeline = nil
    chromaPipeline = nil
    krea2Pipeline = nil
    revalidateLastRecipe()  // WP-E10: no checkpoint, no record (#218)
    chromaTokenizer = nil
    controlPipeline = nil
    pipelinePrepared = false
    activePoolModelSpec = nil
    imageModelsEvicted = true
    GPU.clearCache()
    logger.info("Released image models for LTX-2 video (~\(freedMB)MB pool est; base pipeline + per-family pipelines unloaded) (#218)")
    publishHealth()
    return freedMB
  }

  /// If image models were evicted for a video render, reload the previously
  /// active image model (or the one this request explicitly asks for) before
  /// rendering. Throws if the reload fails. No-op when nothing was evicted.
  private func reloadImageModelIfEvicted(requestedModel: String?) async throws {
    guard imageModelsEvicted else { return }
    let spec = requestedModel.map { WarmServer.parseModelSpec(from: $0) }
      ?? lastActiveImageSpec
      ?? configuration.modelSpec
    guard let reloadSpec = spec else { imageModelsEvicted = false; return }
    let quant = requestedModel.flatMap { WarmServer.parseQuantization(from: $0) }
    logger.info("Reloading image model '\(reloadSpec)' after video eviction (#218)")
    _ = try await poolLoad(modelSpec: reloadSpec, quantization: quant, activate: true)
    imageModelsEvicted = false
  }

  /// Shed the least-recently-used *inactive* pool model under memory pressure.
  /// Never touches the active model or an in-flight render. Returns MB freed.
  @discardableResult
  func shedInactivePoolModelUnderPressure() async -> Int {
    await modelPool.releaseLRUInactive()
  }

  // MARK: - #1479 preemption orchestration
  //
  // Entered from the `.localVideo` case in `processLoop` when
  // `body(report)` (or a prior `resume(from:)`) returns `.yielded` — i.e.
  // `ltx2PreemptionSignal` was raised by an image job's route handler
  // (`WarmServer.attemptPreemption`, which runs BEFORE any actor call — see
  // the block comment above `RollingMeanSec`) and the render loop observed
  // it at a step boundary. Everything from here on runs on the actor, in the
  // SAME Task that was blocked inside the video's synchronous render, so no
  // further actor hop is needed to reach `pipeline`/`runGenerate`/etc.

  /// #1479/#230/#218/#34: the LTX-2 stack's memory need assuming it stays
  /// WARM (resident) — the warm-stack-discount / audio-mode-mismatch logic
  /// `vacateImageModelsAndAdmitVideo`'s admission gate uses. (Until the final
  /// review this was also consulted by a no-eviction "fast path" decision in
  /// `runPreemptionEpisode`; that path is gone — see `runPreemptionEpisode`'s
  /// doc comment — so this now has exactly one caller.)
  private func ltx2WarmNeedBytes(wantsAudio: Bool) -> UInt64 {
    let gen = videoHolder.get()
    let audioModeMatches = (gen?.isAudioLoaded ?? false) == wantsAudio
    let videoStackWarm = gen?.isLoaded == true && audioModeMatches
    let audioDelta: UInt64 = wantsAudio ? 13 * 1024 * 1024 * 1024 : 0
    return videoStackWarm
      ? 24 * 1024 * 1024 * 1024
      : HeavyModelAdmission.ltx2EstimateBytes(forWeightsPath: configuration.ltx2WeightsPath) + audioDelta
  }

  /// Rebuild the LTX-2 generator after an eviction. After `videoHolder
  /// .release()` the previous instance (and anything it had wired) is gone —
  /// this is a cold start by the exact same resolution `prepareLocalVideo`
  /// uses, publishing the fresh instance into `videoHolder` so the rest of
  /// the codebase (admission, `/health`, later renders) sees it immediately.
  /// Only resolves paths and constructs the object — the actual multi-GB
  /// weight load is lazy, inside `resume(from:)` -> `render` -> `load(...)`
  /// (review I1: do not time THIS call as "the reload").
  private func reloadVideoGeneratorAfterEviction() async throws -> LTX2VideoGenerator {
    guard let weights = configuration.ltx2WeightsPath, let gemma = configuration.ltx2GemmaPath else {
      throw WarmServerError.invalidRequest(
        message: "#1479: LTX-2 not configured — cannot reload after a preemption eviction")
    }
    let weightsURL = try await ModelResolution.resolve(
      modelSpec: weights,
      filePatterns: ["transformer-distilled.safetensors", "connector.safetensors",
                      "vae_decoder.safetensors", "vae_encoder.safetensors", "config.json"]
    )
    let gemmaURL = try await ModelResolution.resolve(
      modelSpec: gemma,
      filePatterns: ["*.safetensors", "*.json", "tokenizer/*", "*.model"]
    )
    let gen = LTX2VideoGenerator(
      config: .init(weightsDir: weightsURL.path, gemmaPath: gemmaURL.path), logger: logger)
    videoHolder.set(gen)
    return gen
  }

  /// #1479 (review C2): the SAME admission gate a cold video start runs
  /// (`.localVideo`'s admission block, now just a thin wrapper around this),
  /// extracted so a preemption resume can run it too. The preempting image
  /// job loaded ITS OWN weights while the video was away; resuming into
  /// whatever memory is left over without re-vacating image models and
  /// re-checking the drain/admission gate is the documented #218/#34 SIGKILL
  /// condition — "a failed tap must never cost a video" includes never
  /// costing it an OOM kill.
  private func vacateImageModelsAndAdmitVideo(
    wantsAudio: Bool
  ) async -> (admitted: Bool, availableMB: Int, neededMB: Int) {
    let freedForVideoMB = await releaseImageModelsForVideo()
    var availableForVideo = MemoryProbe.systemAvailableMemoryBytes()
    // Precision-keyed (#230), warm-stack discount, audio-mode-mismatch cold
    // fallback (task #21, Codex #2) — factored into `ltx2WarmNeedBytes`, which
    // is now this gate's only consumer, so no second, independent estimate can
    // disagree with it (review round 2, finding 3).
    let genForLog = videoHolder.get()
    let videoStackWarm = genForLog?.isLoaded == true && (genForLog?.isAudioLoaded ?? false) == wantsAudio
    let ltx2Need = ltx2WarmNeedBytes(wantsAudio: wantsAudio)
    // Drain-until-settled (#34): back-to-back renders (e.g. Kira's i2v →
    // multi-keyframe in the same second) start while the previous job's
    // MLX buffer pool + lazy macOS reclaim still hold tens of GB. Admission
    // then either refuses spuriously OR passes on memory that isn't really
    // reclaimed yet — and the render dies ~60s in on a Metal allocation
    // abort (SIGKILL, no app error; 3x reproduced 2026-07-25). Actively
    // drain and re-probe until free ≥ need + margin, up to ~18s, before
    // deciding. clearCache() returns pooled buffers; the settle sleep gives
    // the OS time to actually reclaim them.
    let drainMargin: UInt64 = 6 * 1024 * 1024 * 1024
    var drainAttempts = 0
    while availableForVideo < ltx2Need + drainMargin && drainAttempts < 6 {
      GPU.clearCache()
      try? await Task.sleep(nanoseconds: 3_000_000_000)
      availableForVideo = MemoryProbe.systemAvailableMemoryBytes()
      drainAttempts += 1
      logger.info("LTX-2 admission drain #\(drainAttempts): \(availableForVideo >> 20)MB free (target \((ltx2Need + drainMargin) >> 20)MB)")
    }
    let admitVideo = heavyAdmission.admitsAfterEvict(
      needBytes: ltx2Need, freeBytes: availableForVideo)
    logger.info("LTX-2 admission: freed ~\(freedForVideoMB)MB image, \(availableForVideo >> 20)MB free after \(drainAttempts) drain(s), need ~\(ltx2Need >> 20)MB\(videoStackWarm ? " (warm stack)" : "") → admit=\(admitVideo) (#218/#34)")
    return (admitVideo, Int(availableForVideo >> 20), Int(ltx2Need >> 20))
  }

  /// Reload (if needed) and resume the checkpointed video render.
  ///
  /// Runs the full #218/#34 admission gate (`vacateImageModelsAndAdmitVideo`)
  /// before touching the generator (review C2).
  ///
  /// **Residency is the authority** (final review, C1): there is no `evicted`
  /// parameter any more. The caller's belief about whether it evicted the
  /// video is not trustworthy — `#218` makes ANY image load release the video
  /// stack underneath it (`poolLoad`'s first statement is
  /// `videoHolder.release()`), and the memory-pressure guard can release it
  /// too — so this asks `videoHolder` what is actually resident and rebuilds
  /// when nothing is. Absence is recoverable (a cold rebuild costs a reload,
  /// which is exactly what the always-evict episode already budgets for);
  /// throwing on it, as the previous `evicted: false` branch did, destroyed
  /// the checkpoint and failed the video job outright.
  ///
  /// Binding constraint from Task 4's reviews, enforced HERE so no call site
  /// can get it wrong: a fresh post-eviction instance has no signal/telemetry
  /// wired (`VideoGeneratorHolder.release()` deallocates the old one) —
  /// re-wired unconditionally below. The OTHER Task 4 constraint — the
  /// signal must be cleared before `resume(from:)` — is enforced by
  /// `runPreemptionEpisode`'s caller instead (review round 2, finding 2):
  /// clearing it HERE, after this function's own admission gate can `await`
  /// for up to ~18s in the drain loop, left a window where a SECOND
  /// preemptor could raise a NEW signal during that wait, which this clear
  /// would then erase — stranding the second preemptor behind its own 120s
  /// watchdog. `runPreemptionEpisode` clears while `preemptionInFlight` is
  /// still held, before anything here can run, so no such window exists.
  ///
  /// Throws on failure (config drift, missing generator, admission refused)
  /// — the caller's `continuation` then resumes with the error, i.e. the
  /// video job fails loudly. No silent restart from step 0 (spec, Error
  /// handling).
  private func resumeCheckpointedVideo(
    state: LTX2ResumeState, wantsAudio: Bool, report: @escaping @Sendable (Int) -> Void
  ) async throws -> LTX2RenderOutcome {
    let admission = await vacateImageModelsAndAdmitVideo(wantsAudio: wantsAudio)
    guard admission.admitted else {
      throw WarmServerError.invalidRequest(
        message: "#1479: insufficient memory to resume LTX-2 video after preemption: only \(admission.availableMB)MB free (need ~\(admission.neededMB)MB)")
    }
    let gen: LTX2VideoGenerator
    if let existing = videoHolder.get() {
      // Still resident — nothing released it while the preemptor ran. Reuse
      // it; the checkpoint resumes with no weight reload at all.
      logger.info("#1479: resuming LTX-2 video — generator still resident, no reload needed")
      gen = existing
    } else {
      // Nothing resident: the expected state after an episode's eviction, and
      // also the recovery path if something ELSE released it out from under
      // the checkpoint (#218's release-before-image-load, the memory-pressure
      // guard). Either way this is recoverable — rebuild cold and resume from
      // the checkpoint rather than throwing the render away.
      logger.info("#1479: resuming LTX-2 video — no generator resident, rebuilding cold before resume")
      gen = try await reloadVideoGeneratorAfterEviction()
    }
    gen.setPreemptionSignal(ltx2PreemptionSignal)
    gen.setTelemetry(ltx2Telemetry)
    return try gen.resume(from: state) { chunk, totalChunks, step, totalSteps in
      report(WarmServer.localVideoProgressPercent(chunk: chunk, totalChunks: totalChunks, step: step, totalSteps: totalSteps))
    }
  }

  /// Sum of observed `modelLoad` phase durations (`meanSec x samples`) —
  /// the real instrument for "how long does a weight reload take" (review
  /// round 2, finding 1). The load itself is bracketed by
  /// `telemetry?.begin(.modelLoad)`/`.end(.modelLoad)` inside
  /// `LTX2VideoGenerator`'s `render()`, around the actual `load(...)` call —
  /// snapshotting this before/after a resume and recording the DELTA (only
  /// when it grew, i.e. a load actually happened) is accurate regardless of
  /// how long the REST of the resumed render takes, unlike wall-clocking the
  /// resume call itself (which the previous fix for I1 got wrong — see the
  /// call site below).
  private func ltx2ModelLoadTotalSec() -> Double {
    guard let p = ltx2Telemetry.view().phases["modelLoad"] else { return 0 }
    return p.meanSec * Double(p.samples)
  }

  /// The full preemption episode: checkpoint received -> evict the video
  /// weights -> run the preempting image job -> ALWAYS resume the video.
  ///
  /// **v1 always evicts** (final review C1 + controller ruling, 2026-08-15;
  /// the spec's Decision 1 has been amended to match). The original design had
  /// a "fits alongside" fast path that kept the video weights resident and
  /// skipped the evict/reload round trip. That path was not merely an
  /// optimisation that sometimes failed to pay off — it was actively
  /// destructive: `runGenerate` begins with `reloadImageModelIfEvicted` ->
  /// `poolLoad`, whose FIRST statement is `videoHolder.release()` (#218's
  /// invariant that video must vacate before any image load, and
  /// `imageModelsEvicted` is always true while a video is resident, so that
  /// reload always runs). The layer below therefore released the very weights
  /// the fast path was preserving, and the resume then found nothing resident
  /// and threw — losing the checkpoint and failing the video job. On a 128GB
  /// box the fast path was the LIKELY branch. Honouring pause-in-place would
  /// require a preemption-aware carve-out in `poolLoad` that lets both heavy
  /// stacks co-reside — the exact 2x22GB pool-budget hazard the spec's
  /// Decision 1 exists to avoid — so v1 simply always evicts, and
  /// `resumeCheckpointedVideo` treats live residency (not a caller's belief)
  /// as the authority on whether a rebuild is needed.
  ///
  /// Always-resume is enforced two ways: structurally, the image-job call
  /// sits in its own `do`/`catch` (review I4) so that even if `runGenerate`
  /// ever grows a `throws` in the future, control still reaches the resume
  /// below instead of silently skipping it; today `runGenerate` is
  /// non-throwing by construction (every internal failure already resolves
  /// `claimed.continuation` via `continuation.resume(throwing:)`), so that
  /// `catch` is presently unreachable — it exists as a guardrail, not because
  /// it fires. A failed preemptor therefore costs the video nothing but the
  /// wall-clock time it ran (spec, Error handling).
  ///
  /// Checkpoint/paused/in-flight bookkeeping is cleared BEFORE the (possibly
  /// long, up to the rest of the render) resume call, not after (review I2):
  /// clearing after would hold `preemptionInFlight` for the render's entire
  /// remaining duration, making a SECOND preemption of the resumed render
  /// permanently unreachable — exactly the double-preempt path Task 4's
  /// forward-only unwind guard exists to handle correctly. `resolved` guards
  /// the `defer` so the two code paths (claimed vs. not-claimed) each clear
  /// exactly once, and a throw before either path reaches its own clearing
  /// (e.g. `pendingPreemptorBox.claim()` itself can't throw, but future edits
  /// might add one) still cleans up via the `defer`.
  ///
  /// `ltx2PreemptionSignal.clear()` happens FIRST, before anything else
  /// (review round 2, finding 2): this runs while `preemptionInFlight` is
  /// STILL held (it isn't released until `clearEpisodeState()` below), so no
  /// other caller can have raised a competing signal in between — clearing
  /// it any later (previously done inside `resumeCheckpointedVideo`, after
  /// its own admission gate could `await` up to ~18s in the drain loop) left
  /// a window where a second preemptor could raise, get its raise erased by
  /// this render's stale clear, and be stranded behind its own 120s
  /// watchdog. `checkpointedVideo` is read once, right after, as a
  /// single-checkpoint sanity check (spec: hold exactly one at a time) —
  /// this function must never be reentered while a checkpoint is already
  /// held.
  private func runPreemptionEpisode(
    state: LTX2ResumeState, videoJobId: String?, wantsAudio: Bool, report: @escaping @Sendable (Int) -> Void
  ) async throws -> LTX2RenderOutcome {
    ltx2PreemptionSignal.clear()
    assert(checkpointedVideo == nil, "#1479: hold exactly one checkpoint at a time (spec) — runPreemptionEpisode entered with one already held")
    checkpointedVideo = state
    if let videoJobId { videoJobTracker.markPausedForPreemption(videoJobId) }
    publishHealth()

    var resolved = false
    func clearEpisodeState() {
      resolved = true
      checkpointedVideo = nil
      if let videoJobId { videoJobTracker.markResumedFromPreemption(videoJobId) }
      // Idempotent: the checkpoint-fallback watchdog may have already
      // cleared this if it raced ahead of the yield.
      preemptionInFlight.clear()
    }
    defer { if !resolved { clearEpisodeState() } }

    guard let claimed = pendingPreemptorBox.claim() else {
      // The checkpoint-fallback watchdog already handled this preemptor (it
      // raced ahead of the yield, or the render finished on its own before
      // the signal was observed) — nothing left to run. Resume immediately.
      logger.warning("#1479: video yielded but no preemptor was waiting (checkpoint-fallback watchdog already handled it) — resuming immediately")
      clearEpisodeState()
      return try await resumeCheckpointedVideo(state: state, wantsAudio: wantsAudio, report: report)
    }

    logger.info("#1479: video checkpointed at chunk \(state.chunkIndex), phase \(state.phase.rawValue), step \(state.stepIndex) — running preempting image job (source=\(claimed.source))")

    // Always evict (see doc comment): the image load below would release these
    // weights anyway, one layer down and without timing it. Doing it here
    // explicitly keeps `ltx2EvictMean` — half of the refusal guard's
    // round-trip estimate — measuring the real thing.
    let t0 = Date()
    videoHolder.release()
    ltx2EvictMean.record(Date().timeIntervalSince(t0))
    logger.info("#1479: evicted LTX-2 video weights to admit the preempting image job")

    // #1479 (final review I2/M13): the preemptor is a REAL render — it just
    // didn't arrive through `pending`. Swap the entire active-job identity
    // over to it (id, summary, source, start time) and, critically, the
    // persistence pair (rawBody + kind) so the durable queue snapshot names
    // the image job while the image job is what's actually running. Without
    // this, a crash during the preemptor lost it outright (the persisted
    // "active" slot still described the paused video, whose own rawBody is
    // nil and therefore unrecoverable), and /health + /v1/queue reported the
    // video as active for the whole image render. Persist AFTER each swap so
    // the on-disk snapshot is never behind the in-memory truth.
    let videoIdentity = (
      id: activeJobId, summary: activeJobSummary, source: activeJobSource,
      rawBody: activeJobRawBody, kind: activeJobKindForPersistence,
      startedAt: currentJobStartedAt, renderStartedAt: activeRenderStartedAt
    )
    activeJobId = claimed.jobId ?? UUID().uuidString  // AC-18: the async caller's id when it has one
    activeJobSummary = "Render (preempting): \(claimed.payload.prompt.prefix(100))"
    activeJobSource = claimed.source
    activeJobRawBody = claimed.rawBody
    activeJobKindForPersistence = "generate"
    currentJobStartedAt = Date()
    activeRenderStartedAt = nil
    publishHealth()
    persistQueueState()

    // Run the image job exactly as a normal render — the same actor method
    // every non-preempting `.generate` job runs, including its own
    // model/LoRA application and (since `imageModelsEvicted` is already true
    // whenever a video is resident, #218) its own image-model reload. See
    // this function's doc comment (review I4) for why this is its own
    // do/catch rather than a bare call.
    do {
      await runGenerate(claimed.payload, continuation: claimed.continuation)
    } catch {
      logger.error("#1479: preempting image job threw past runGenerate's own boundary (\(error)) — resuming the video anyway")
    }

    // Restore the video's identity — symmetric with the swap above, and done
    // BEFORE its (possibly long, synchronous) resume so /health, /v1/queue and
    // the persisted snapshot all go back to describing the video job. The
    // preemptor is finished by now, so leaving it in the persisted active slot
    // would replay a completed image job on the next restart.
    activeJobId = videoIdentity.id
    activeJobSummary = videoIdentity.summary
    activeJobSource = videoIdentity.source
    activeJobRawBody = videoIdentity.rawBody
    activeJobKindForPersistence = videoIdentity.kind
    currentJobStartedAt = videoIdentity.startedAt
    activeRenderStartedAt = videoIdentity.renderStartedAt
    publishHealth()
    persistQueueState()

    // Clear BEFORE the resume, not in the defer (review I2) — see doc comment.
    clearEpisodeState()

    // Review round 2, finding 1: snapshot the REAL instrument (modelLoad
    // phase telemetry), not a wall clock around the whole resume call — the
    // resume call's return time includes the rest of the render (15-60min),
    // which would poison a cumulative rolling mean into a de facto refuse-
    // everything kill switch. Record only the delta, and gate it on a real
    // reload having happened (review round 3, finding 1): `load()`'s
    // idempotent early-return still sits INSIDE the telemetry bracket
    // (`LTX2VideoGenerator.swift`), so a resume that reuses a resident
    // generator emits its own microseconds-scale `modelLoad` sample —
    // `modelLoadDelta > 0` alone is true there too, and a near-zero sample
    // decays the rolling mean toward zero, degrading the refusal guard to
    // never-refuse (the opposite failure this metric exists to avoid). The
    // gate is therefore "was the generator absent going in", read from the
    // same authority `resumeCheckpointedVideo` uses and evaluated on the
    // actor with nothing able to change residency in between; the delta is
    // only for accuracy of the reload's DURATION, not for detecting whether
    // it occurred. This also swallows the ~1e-13 float round-trip noise the
    // mean*samples sum reconstruction can produce. Under the always-evict
    // rule this is normally true — it stays a check rather than a constant so
    // a future preemption-aware carve-out cannot silently poison the metric.
    //
    // Single-flight assumption: this snapshot/delta is only correct because
    // exactly one video render is ever in flight at a time (the coordinator
    // serializes video on the same queue as image renders); a future
    // concurrent-video path would let a second render's modelLoad samples
    // land inside this window and silently corrupt the delta.
    let willReload = videoHolder.get() == nil
    let modelLoadBefore = ltx2ModelLoadTotalSec()
    let outcome = try await resumeCheckpointedVideo(state: state, wantsAudio: wantsAudio, report: report)
    let modelLoadDelta = ltx2ModelLoadTotalSec() - modelLoadBefore
    if willReload, modelLoadDelta > 0 {
      ltx2ReloadMean.record(modelLoadDelta)
    }
    return outcome
  }

  func prepare() async throws {
    // Resolve model snapshot path for family detection
    let modelSpec = configuration.modelSpec
    var isFlux2 = false
    var snapshotURL: URL?

    var isFibo = false
    var isChroma = false
    var isKrea2 = false

    if let spec = modelSpec {
      // Check by known model ID first
      if Krea2ModelDetection.isKnownKrea2Model(spec) {
        isKrea2 = true
      } else if ChromaModelDetection.isKnownChromaModel(spec) {
        isChroma = true
      } else if FiboModelDetection.isKnownFiboModel(spec) {
        isFibo = true
      } else if Flux2ModelDetection.isKnownFlux2Model(spec) {
        isFlux2 = true
      }

      if !isKrea2 {
        // Resolve snapshot — needed for both detection and loading
        let resolved = try await ModelResolution.resolveOrDefault(
          modelSpec: spec,
          filePatterns: ["*.safetensors", "*.json", "tokenizer/*"]
        )
        snapshotURL = resolved

        // If not already detected by name, check the snapshot directory
        if !isFibo && !isFlux2 && !isChroma {
          if Krea2ModelDetection.isKrea2ModelDirectory(resolved) {
            isKrea2 = true
          } else if ChromaModelDetection.detect(at: resolved) != nil {
            isChroma = true
          } else if FiboModelDetection.detect(at: resolved) != nil {
            isFibo = true
          } else if Flux2ModelDetection.detectFamily(at: resolved) == .flux2 {
            isFlux2 = true
          }
        }
      }
    }

    if isKrea2, let spec = modelSpec {
      // --- Krea-2 path (native port) — variant read off disk, fail-closed (WP-E5) ---
      currentModelFamily = .krea2
      let paths = try Krea2ModelDetection.resolve(spec: spec)
      logger.info(
        "Detected Krea-2 \(paths.variant.rawValue) (\(paths.transformerFile.path)) — 8-bit transformer, estimated GPU memory: ~22GB")
      krea2Pipeline = try Krea2Pipeline(paths: paths, quantizeTransformer: 8)
      krea2Variant = paths.variant
      pipelinePrepared = true
      logger.info("Warm server pipeline ready (Krea-2 \(paths.variant.rawValue))")
    } else if isChroma, let snapshot = snapshotURL {
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
    // WP-E10 sink 3: `prepare()` is the third writer of `krea2Pipeline`, and
    // its non-Krea-2 arms replace the ACTIVE family out from under a record.
    // Placed after the whole chain so every arm is covered by the one rule.
    revalidateLastRecipe()

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
      case .krea2:
        box = PipelineBox(pipeline: krea2Pipeline! as AnyObject)
        detectedInfo = krea2Pipeline!.variant
        vramMB = 22528
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
    // Seed the lock-based health snapshot now that the model is loaded, so
    // GET /health returns real data before the first render (#217).
    publishHealth()
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

  /// The physical Krea-2 variant of the resident pipeline (WP-E5). nil when
  /// the active family is not krea2 — callers on the krea2 arm must treat nil
  /// as a fault, never as "turbo".
  var currentKrea2Variant: Krea2Variant? {
    currentModelFamily == .krea2 ? krea2Variant : nil
  }

  // MARK: - Model Pool Operations

  /// Load a model into the pool, optionally activating it.
  /// QUEUE-INTERNAL (K-FIX-1 / Codex C2). Call this ONLY from inside the
  /// process loop — a queued `.modelOperation`, the ComfyBridge switch's
  /// `enqueueModelSwitch` body, or a render's own #218 reload. A route or
  /// handler that reaches it directly reintroduces the race: actor isolation
  /// does not hold across an await, so the pool's eviction and
  /// `GPU.clearCache()` would be free to run under an active render. From
  /// outside the loop use `enqueueModelOperation` / `enqueueModelOperationDetached`.
  func poolLoad(modelSpec: String, quantization: String?, activate: Bool) async throws -> ModelLoadResponse {
    // #218: an image load must vacate a resident LTX-2 video stack first — the
    // two heavy models can't co-reside on a 128GB box. Safe here because
    // poolLoad and the video render are serialized on the same actor/queue, so
    // no video render is ever in flight at this point.
    if videoHolder.release() {
      logger.info("Released resident LTX-2 video stack before image load (#218)")
    }
    // D17 (AC-59a): every ACTUAL base handoff that touches the krea2 family
    // logs outgoing and incoming spec/variant, so a slow A/B is attributable,
    // not mysterious. `Krea2Handoff.logLine` is nil for a no-op re-activation
    // of the resident base and for a cold start with nothing resident.
    let outgoing: Krea2Handoff.Side? = {
      guard pipelinePrepared, let spec = activePoolModelSpec ?? configuration.modelSpec else { return nil }
      return Krea2Handoff.Side(spec: spec, family: currentModelFamily, krea2Variant: currentKrea2Variant)
    }()
    let start = Date()
    let entry = try await modelPool.load(
      modelSpec: modelSpec,
      quantization: quantization,
      initialLoRAs: activeLoRAs,
      // A load that intends to activate is a HANDOFF — the pool may evict the
      // current active model to make room (two ~22GB krea2-family models
      // cannot co-reside; without this a switch 507s, 2026-08-11).
      allowActiveEviction: activate
    )
    let loadTimeMs = Int(Date().timeIntervalSince(start) * 1000.0)

    if activate {
      try await poolActivate(modelId: entry.id)
      let incoming = Krea2Handoff.Side(spec: entry.modelSpec, family: entry.family, krea2Variant: currentKrea2Variant)
      if let line = Krea2Handoff.logLine(outgoing: outgoing, incoming: incoming, loadTimeMs: loadTimeMs) {
        logger.info("\(line)")
      }
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
  /// QUEUE-INTERNAL (K-FIX-1 / Codex C2). Call this ONLY from inside the
  /// process loop — a queued `.modelOperation`, the ComfyBridge switch's
  /// `enqueueModelSwitch` body, or a render's own #218 reload. A route or
  /// handler that reaches it directly reintroduces the race: actor isolation
  /// does not hold across an await, so the pool's eviction and
  /// `GPU.clearCache()` would be free to run under an active render. From
  /// outside the loop use `enqueueModelOperation` / `enqueueModelOperationDetached`.
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
    // An image model is now resident and active — clear the video-eviction flag
    // so a later render doesn't redundantly reload (#218).
    imageModelsEvicted = false
    // Track the activated pool model's spec so generation requests use
    // the correct model instead of the startup configuration.modelSpec.
    activePoolModelSpec = entry.modelSpec
    switch entry.family {
    case .krea2:
      krea2Pipeline = entry.box.pipeline as? Krea2Pipeline
      // The pipeline is the physical fact; the pool entry carries the same
      // value back from loadPipeline (WP-E5).
      krea2Variant = krea2Pipeline?.variant ?? (entry.detectedInfo as? Krea2Variant)
      if krea2Pipeline == nil {
        logger.warning(
          "ModelPool: activated krea2 entry '\(entry.id)' but its pipeline could not be read back — publishing an EMPTY LoRA stack (I1)")
      }
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
    // K-FIX-1 / Codex I1: the activated pipeline is the authority on what is
    // applied. Reconciling here — BEFORE `publishHealth()` — is what stops
    // `/health.loras` (and the next render's default stack) from describing
    // the model that just left.
    let reconciledLoRAs = PoolAdapterState.reconciled(
      family: entry.family, activated: krea2Pipeline, coordinator: activeLoRAs)
    if reconciledLoRAs.map(\.source.displayName) != activeLoRAs.map(\.source.displayName) {
      let names = reconciledLoRAs.map { $0.source.displayName }.joined(separator: ", ")
      let line = "ModelPool: activation reconciled the LoRA stack from the pipeline read-back — "
        + "\(activeLoRAs.count) advertised → \(reconciledLoRAs.count) applied [\(names)]"
      logger.info("\(line)")
    }
    activeLoRAs = reconciledLoRAs

    revalidateLastRecipe()
    pipelinePrepared = true
    // Model/family/variant just changed — refresh the health snapshot (#217).
    publishHealth()

    return ModelActivateResponse(
      status: "activated",
      model: entry.modelSpec,
      family: entry.family.rawValue
    )
  }

  /// Unload a model from the pool.
  /// QUEUE-INTERNAL (K-FIX-1 / Codex C2). Call this ONLY from inside the
  /// process loop — a queued `.modelOperation`, the ComfyBridge switch's
  /// `enqueueModelSwitch` body, or a render's own #218 reload. A route or
  /// handler that reaches it directly reintroduces the race: actor isolation
  /// does not hold across an await, so the pool's eviction and
  /// `GPU.clearCache()` would be free to run under an active render. From
  /// outside the loop use `enqueueModelOperation` / `enqueueModelOperationDetached`.
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
    source: String = "api",
    rawBody: Data? = nil,
    jobId: String? = nil
  ) async throws -> GenerateResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      // AC-18 (WP-E10 "E9b"): `jobId` is the CLIENT-VISIBLE id when the caller
      // has one (`/v1/generate/async`'s tracker id, or a persisted job's own
      // id on replay), so the queue, its on-disk snapshot, a failed replay
      // and the status route all name the same job. nil → a fresh id (the
      // synchronous route, which never exposes one).
      pending.append(PendingJob(
        id: jobId ?? UUID().uuidString, source: source,
        operation: .generate(payload, ContinuationBox(continuation), progressHandler, latentPreviewHandler),
        rawBody: rawBody))
      startProcessingIfNeeded()
    }
  }

  func enqueueSwap(_ payload: LoRASwapPayload, rawBody: Data? = nil) async throws -> LoRASwapResponse {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(PendingJob(operation: .swap(payload, ContinuationBox(continuation)), rawBody: rawBody))
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

  /// Execute one ``ModelOperation`` — called ONLY from the process loop, so
  /// the whole mutation (including `ModelPool.load`'s eviction and
  /// `GPU.clearCache()`) is serialized against renders (K-FIX-1 / C2).
  ///
  /// The `poolLoad`/`poolActivate`/`poolUnload` methods stay as the internal
  /// implementation because the loop's OWN jobs call them (a render that must
  /// restore an image model after a video eviction, #218): re-entering the
  /// queue from inside the queue would deadlock. What changed is that no
  /// route reaches them except through here.
  private func runModelOperation(_ op: ModelOperation) async throws -> ModelOperationResult {
    switch op {
    case .load(let modelSpec, let quantization, let activate):
      return .load(try await poolLoad(
        modelSpec: modelSpec, quantization: quantization, activate: activate))
    case .activate(let modelId):
      return .activate(try await poolActivate(modelId: modelId))
    case .unload(let modelId):
      return .unload(try await poolUnload(modelId: modelId))
    }
  }

  /// How many MUTATING pool operations are waiting. Counts only
  /// `.modelOperation` jobs, so parked renders never consume the model-op
  /// budget and vice versa.
  private var pendingModelOperationCount: Int {
    pending.reduce(into: 0) { count, job in
      if case .modelOperation = job.operation { count += 1 }
    }
  }

  /// The model-operation bound (review finding 1). Independent of
  /// `maxPendingRequests` in both directions.
  private func checkModelOperationCapacity() throws {
    if pendingModelOperationCount >= configuration.maxPendingModelOps {
      throw ServerError.modelOperationQueueFull(maxPending: configuration.maxPendingModelOps)
    }
  }

  /// Enqueue a mutating pool operation and wait for it (K-FIX-1 / C2).
  ///
  /// MUST NOT be called from inside the process loop — see `runModelOperation`.
  func enqueueModelOperation(_ op: ModelOperation) async throws -> ModelOperationResult {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    // Not the RENDER capacity gate (WP-E8 hygiene): that gate exists to bound
    // render backlog, and applied here it reproduced the pause wedge one
    // layer up — a paused queue holding `maxPendingRequests` parked renders
    // answered `/v1/model/unload` with `queueFull`, so the operator could not
    // free the GPU precisely when they needed to. Model operations get their
    // OWN cap instead (review finding 1): parked renders never crowd them
    // out, and an unauthenticated client cannot grow the FIFO without limit.
    try checkModelOperationCapacity()

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(PendingJob(operation: .modelOperation(op, ContinuationBox(continuation))))
      startProcessingIfNeeded()
    }
  }

  /// Enqueue a mutating pool operation WITHOUT waiting, returning its FIFO job
  /// id (K-FIX-1 / C2).
  ///
  /// This replaces `/v1/model/load`'s `wait: false` arm, which used to start a
  /// detached `Task` that ran `poolLoad` outside the queue entirely — the
  /// worst version of the race, because nothing in the system knew it was
  /// running. Now it is an ordinary queue job: it appears in `/v1/queue`, it
  /// can be cancelled, and it cannot begin under a render.
  @discardableResult
  func enqueueModelOperationDetached(_ op: ModelOperation) throws -> String {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    // Same reasoning as `enqueueModelOperation` — the model-op cap, not the
    // render capacity gate.
    try checkModelOperationCapacity()
    let job = PendingJob(operation: .modelOperation(op, nil))
    pending.append(job)
    startProcessingIfNeeded()
    return job.id
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
  /// Enqueue a local LTX-2 render on the serial GPU queue. `body` receives a
  /// `report(percent)` callback (0-100) it should call from the generator's
  /// per-chunk/per-step progress hook; the coordinator wires it into the
  /// lock-based progress + health trackers so /health and /v1/queue reflect the
  /// live render without an actor hop (mirrors the image render path, #217).
  func enqueueLocalVideo(
    wantsAudio: Bool = false,
    videoJobId: String? = nil,
    _ body: @escaping @Sendable (@escaping @Sendable (Int) -> Void) throws -> LTX2RenderOutcome
  ) async throws -> LTX2VideoResult {
    if shuttingDown {
      throw ServerError.shuttingDown
    }
    if pending.count >= configuration.maxPendingRequests {
      throw ServerError.queueFull(maxPending: configuration.maxPendingRequests)
    }

    return try await withCheckedThrowingContinuation { continuation in
      pending.append(PendingJob(operation: .localVideo(body, ContinuationBox(continuation), wantsAudio: wantsAudio, videoJobId: videoJobId)))
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

  /// Publish the current health-relevant state into the lock-based
  /// ``LiveHealthState`` so GET /health reads it without hopping onto this
  /// actor (which blocks for a whole render). Call at every state transition:
  /// job start/end, enqueue, model/LoRA change, pause, shutdown, startup (#217).
  ///
  /// `isRendering` keys off `activeJobId` (set the instant a job is dequeued,
  /// before the render method's first await sets `activeRenderStartedAt`), and
  /// the render start time falls back to `currentJobStartedAt` so the age/stale
  /// signal is correct throughout the synchronous GPU section.
  private func publishHealth() {
    let snap = HealthSnapshot(
      shuttingDown: shuttingDown,
      model: activePoolModelSpec ?? configuration.modelSpec ?? ZImageRepository.id,
      modelFamily: currentModelFamily.rawValue,
      modelVariant: {
        switch currentModelFamily {
        case .fibo: return "fibo"
        case .flux1: return zimageVariant.rawValue
        case .flux2: return detectedFlux2Model?.variant
        case .krea2: return krea2Variant?.rawValue  // "turbo" | "raw" (WP-E5, AC-34b)
        case .chroma: return nil
        }
      }(),
      // WP-E10 "E9b": the declared alias beside the resolved path (AC-34b).
      modelAlias: currentModelFamily == .krea2
        ? (activePoolModelSpec ?? configuration.modelSpec).flatMap { Krea2ModelDetection.alias(forSpec: $0) }
        : nil,
      lastRecipe: lastRecipe,
      loaded: pipelinePrepared,
      loras: activeLoRAs.map(LoRAState.init),
      renderCount: successfulRenderCount,
      failedRenderCount: failedRenderCount,
      pendingCount: pending.count,
      isRendering: activeJobId != nil,
      activeRenderStartedAt: activeRenderStartedAt ?? currentJobStartedAt,
      activeJobId: activeJobId,
      lastRenderDurationMs: lastRenderDurationMs,
      lastError: lastError,
      isPaused: isPaused,
      activeSummary: activeJobSummary,
      activeSource: activeJobSource,
      pending: pending.map { job in
        QueueJobInfo(
          id: job.id,
          kind: Self.kind(of: job.operation),
          summary: Self.describe(job.operation),
          source: job.source,
          enqueuedAt: job.enqueuedAt
        )
      },
      maxPending: configuration.maxPendingRequests
    )
    liveHealth.publish(snap)
  }

  /// Mirror the recoverable slice of the queue (see QueuePersistence.swift)
  /// to disk so it survives a crash. Called at every mutation: enqueue,
  /// dequeue-into-active, job completion, cancel, reorder, clear. Cheap
  /// (small JSON, atomic write) relative to how rarely the queue actually
  /// changes compared to render duration.
  private func persistQueueState() {
    let active: PersistedQueueJob? = {
      guard let rawBody = activeJobRawBody,
            let kind = activeJobKindForPersistence,
            let id = activeJobId else { return nil }
      return PersistedQueueJob(
        id: id, kind: kind, source: activeJobSource ?? "api",
        enqueuedAt: currentJobStartedAt ?? Date(), rawBody: rawBody)
    }()
    let pendingJobs: [PersistedQueueJob] = pending.compactMap { job in
      guard let rawBody = job.rawBody else { return nil }
      return PersistedQueueJob(
        id: job.id, kind: Self.kind(of: job.operation), source: job.source,
        enqueuedAt: job.enqueuedAt, rawBody: rawBody)
    }
    QueueStateStore.save(PersistedQueueState(active: active, pending: pendingJobs))
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
    publishHealth()
    persistQueueState()
    return count
  }

  /// Cancel one pending job by id. Returns false when the id isn't queued
  /// (already running or already finished).
  func cancelPending(id: String) -> Bool {
    guard let index = pending.firstIndex(where: { $0.id == id }) else { return false }
    Self.cancel(pending[index].operation)
    pending.remove(at: index)
    publishHealth()
    persistQueueState()
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
    case .modelOperation(_, let cont):
      // A `wait: false` load has no waiting caller — cancelling it is simply
      // dropping the job (C2).
      cont?.resume(throwing: ServerError.cancelled)
    case .localVideo(_, let cont, _, _):
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
    case .modelOperation(let op, _):
      return op.summary
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
    case .modelOperation(let op, _): return op.kind
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

  // MARK: - Queue controls (pause / resume / reorder)

  func setPaused(_ paused: Bool) {
    isPaused = paused
    // Persist so a watchdog kickstart / crash / deploy cannot silently
    // resume creation the user paused (see isPaused declaration).
    if paused {
      FileManager.default.createFile(atPath: Self.pauseSentinelPath, contents: Data("paused \(Date())\n".utf8))
    } else {
      try? FileManager.default.removeItem(atPath: Self.pauseSentinelPath)
    }
    if !paused { startProcessingIfNeeded() }
    publishHealth()
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
    publishHealth()
    persistQueueState()
    return true
  }

  private func startProcessingIfNeeded() {
    // Every enqueue routes through here, so this is the one spot that reflects a
    // just-changed pending count into the lock-based health snapshot (#217).
    publishHealth()
    persistQueueState()
    guard !isProcessing else { return }
    isProcessing = true
    Task {
      await processLoop()
    }
  }

  /// Whether a queued operation may run while the queue is PAUSED
  /// (K-FIX-1 round 2, New-1).
  ///
  /// "Pause" means *no renders* — it is what Todd uses to free the GPU for a
  /// deploy, and it persists across restarts via the sentinel. Mutating model
  /// operations are exactly what an operator needs during that window
  /// (swap the resident checkpoint, unload to free memory), and they were
  /// available before this wave because the routes called the pool directly.
  /// Routing them through the FIFO (C2) must not take that away: the FIFO's
  /// job is to serialise them against an in-flight render, which it still
  /// does — the loop runs one job at a time either way.
  ///
  /// Exhaustive on purpose (no `default`): a new queue kind must decide.
  ///
  /// `.shutdown` joins them (WP-E8 hygiene). `enqueueShutdown` sets
  /// `shuttingDown = true` BEFORE appending, so a shutdown parked behind the
  /// pause gate wedged the engine permanently: the caller's continuation
  /// never resumed and every later enqueue threw `.shuttingDown`, leaving
  /// SIGKILL as the only recovery — on exactly the paused engine an operator
  /// is trying to shut down. Its handler is a bare continuation resume with
  /// no GPU work, so nothing about "pause means no RENDERS" argues for
  /// parking it.
  private static func runsWhilePaused(_ operation: QueuedOperation) -> Bool {
    switch operation {
    case .modelOperation, .shutdown:
      return true
    case .generate, .controlGenerate, .swap, .modelSwitch, .localVideo:
      return false
    }
  }

  private func processLoop() async {
    while true {
      // Paused: renders stay parked, but a model operation still runs (New-1).
      // Picking it out of the middle does not reorder anything that runs: the
      // jobs it passes are parked until resume, and they keep their relative
      // order for when it comes.
      let index: Int
      if isPaused {
        guard let next = pending.firstIndex(where: { Self.runsWhilePaused($0.operation) }) else {
          isProcessing = false
          return
        }
        index = next
      } else {
        guard !pending.isEmpty else {
          isProcessing = false
          return
        }
        index = 0
      }

      let job = pending.remove(at: index)
      activeJobSummary = Self.describe(job.operation)
      activeJobSource = job.source
      // Keep the same id the job had while pending, so clients can correlate.
      activeJobId = job.id
      currentJobStartedAt = Date()
      // Move the job's recoverable data (if any) from "pending" to "active" in
      // the persisted queue snapshot, so a crash mid-render still recovers it
      // (replayed from scratch on restart — there's no way to resume a
      // partial diffusion render, so "at least once" is the correctness goal).
      activeJobRawBody = job.rawBody
      activeJobKindForPersistence = Self.kind(of: job.operation)
      // Publish is_rendering + job id BEFORE the synchronous GPU section begins,
      // so /health reflects the render for its whole (actor-blocking) duration (#217).
      publishHealth()
      persistQueueState()
      defer {
        activeJobSummary = nil; activeJobSource = nil; activeJobId = nil; currentJobStartedAt = nil
        activeJobRawBody = nil; activeJobKindForPersistence = nil
        publishHealth()
        persistQueueState()
      }
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
      case .modelOperation(let op, let continuation):
        // C2: the pool mutation runs HERE, with the loop holding the queue —
        // no render can be dequeued until it returns.
        do {
          let result = try await runModelOperation(op)
          continuation?.resume(returning: result)
        } catch {
          if let continuation {
            continuation.resume(throwing: error)
          } else {
            // A `wait: false` load has no caller left to throw to; it is
            // recorded the same way a failed render is, so /health's
            // last_error names it instead of it vanishing into a log line.
            failedRenderCount += 1
            lastError = "\(op.summary) failed: \(error.localizedDescription)"
            logger.error("Queued model operation failed — \(op.summary): \(error.localizedDescription)")
          }
        }
      case .localVideo(let body, let continuation, let wantsAudio, let videoJobId):
        // Runs on the serial queue so LTX-2 never shares the GPU with a render.
        activeRenderStartedAt = Date()
        // activeJobId is set from job.id at the top of the loop.
        defer { activeRenderStartedAt = nil; activeJobId = nil }
        // #218: single-heavy-model residency. Right before the ~65GB LTX-2
        // stack loads inside body(), vacate ALL image models (pool + per-family
        // pipelines), then verify there is enough physical RAM to proceed —
        // refuse cleanly instead of OOM-killing the whole process. Doing this
        // on the serial render queue guarantees no image render can re-load
        // between the eviction and the video load. Extracted into
        // `vacateImageModelsAndAdmitVideo` (#1479, review C2) so a preemption
        // resume can run the EXACT SAME gate before resuming — the preempting
        // image job loaded its own weights while the video was evicted, and
        // resuming into whatever memory is left without re-checking is the
        // documented SIGKILL condition this gate exists to prevent.
        let admission = await vacateImageModelsAndAdmitVideo(wantsAudio: wantsAudio)
        if !admission.admitted {
          continuation.resume(throwing: WarmServerError.invalidRequest(
            message: "Insufficient memory for LTX-2 video: only \(admission.availableMB)MB free after evicting image models (need ~\(admission.neededMB)MB)"))
        } else {
          videoHolder.beginRender()
          // Stream render progress into the lock-based trackers /health + /queue
          // read, exactly like the image path. Both trackers are Sendable, so the
          // off-actor @Sendable report closure can update them without an actor
          // hop. Cleared on completion via defer.
          let progress = self.progressTracker
          let health = self.liveHealth
          progress.set(0)
          health.setProgress(0)
          let report: @Sendable (Int) -> Void = { pct in
            progress.set(pct)
            health.setProgress(pct)
          }
          defer {
            videoHolder.endRender()
            progress.set(nil)
            health.setProgress(nil)
            ltx2StepPosition.clear()
            // #1479 (review C1): covers EVERY non-yielding exit from this
            // render's whole execution (initial body() completing/throwing,
            // a storyboard `.generate()` shot that never checks the signal
            // at all, or the `.yielded` loop below ending in `.completed`/
            // throw) — if a preempt raised the signal but the render never
            // actually checkpointed on it, a stranded raise would otherwise
            // hit the NEXT render's pre-load unwind point instantly (near-
            // zero-cost false checkpoint, bogus evict/reload sample). Clear
            // unconditionally; clearing an unraised signal is a no-op.
            ltx2PreemptionSignal.clear()
          }
          do {
            // #1479: body() may hand back a checkpoint instead of a finished
            // clip (a `preempt: true` image job raised the signal). Loop
            // rather than recurse: a resume can itself yield again (a second,
            // later preemption), and each iteration is handled identically.
            var outcome = try body(report)
            while case .yielded(let state) = outcome {
              outcome = try await runPreemptionEpisode(state: state, videoJobId: videoJobId, wantsAudio: wantsAudio, report: report)
            }
            if case .completed(let result) = outcome {
              continuation.resume(returning: result)
            }
          } catch {
            continuation.resume(throwing: error)
          }
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
    liveHealth.setProgress(0)
    let tracker = progressTracker
    // Feed the lock-based health snapshot too, so /health's progress_percent
    // advances live during the render without an actor hop (#217).
    let health = liveHealth
    let trackedHandler: @Sendable (ZImagePipeline.GenerationProgress) -> Void = { progress in
      if progress.stage == .denoising {
        let pct = Int(progress.fractionCompleted * 100)
        tracker.set(pct)
        health.setProgress(pct)
      }
      progressHandler?(progress)
    }

    // Live denoising preview (GH #216): approximate each step's latents as a
    // small JPEG and stash it for REST/polling clients (Desktop already
    // polls /health for progress_percent — this rides the same cadence).
    // Krita/ComfyUI still get their own frame pushed via the bridge
    // WebSocket, forwarded first so that behavior is unchanged.
    let preview = previewTracker
    let trackedPreviewHandler: ZImagePipeline.LatentPreviewHandler = { latents, step, total, latentH, latentW in
      latentPreviewHandler?(latents, step, total, latentH, latentW)
      #if canImport(CoreGraphics)
      guard step > 0, step % Self.previewStepInterval == 0, step < total else { return }
      guard let approx = LatentPreviewApproximator.latentsToRGBA(latents, latentHeight: latentH, latentWidth: latentW) else { return }
      guard let framed = ComfyBridgePreviewEncoder.encodePreviewFrame(
        fromRGBA: approx.data, width: approx.width, height: approx.height,
        maxDimension: Self.previewMaxDimension, jpegQuality: Self.previewJPEGQuality
      ) else { return }
      // Strip the 8-byte ComfyUI WebSocket binary-frame header — REST
      // clients just want the raw JPEG bytes.
      preview.set(framed.dropFirst(8))
      #endif
    }
    defer { activeJobId = nil; progressTracker.set(nil); liveHealth.setProgress(nil); previewTracker.set(nil) }

    // #218: if a prior LTX-2 video render evicted the image models, restore the
    // previously-active image model before this render can run. poolLoad also
    // releases any resident video stack, so image and video stay mutually
    // exclusive. Runs before the per-job model/LoRA application below.
    do {
      try await reloadImageModelIfEvicted(requestedModel: payload.model)
    } catch {
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
      return
    }

    // Per-job model/LoRA application (queue-submit race fix): a job's own
    // model+loras are applied right before it runs, instead of trusting
    // whatever the shared pool's "currently active" model/LoRAs happen to
    // be by the time it dequeues — a plain synchronous /v1/generate caller
    // activates the model right before calling generate, but async
    // queue-submit (POST /v1/generate/async) can dequeue well after a
    // different request has changed global state. nil model/loras preserve
    // the old caller-activates-first behavior exactly.
    if let modelSpec = payload.model, !modelSpec.isEmpty {
      let resolvedSpec = WarmServer.parseModelSpec(from: modelSpec)
      let currentSpec = activePoolModelSpec ?? configuration.modelSpec
      if resolvedSpec != currentSpec {
        let resolvedQuant = WarmServer.parseQuantization(from: modelSpec)
        do {
          _ = try await poolLoad(modelSpec: resolvedSpec, quantization: resolvedQuant, activate: true)
        } catch {
          lastError = error.localizedDescription
          continuation.resume(throwing: error)
          return
        }
      }
    }
    if let loraEntries = payload.loras {
      do {
        let newLoRAs = try loraEntries.map { try $0.makeConfiguration() }
        try await applyActiveLoRAs(newLoRAs)
      } catch {
        lastError = error.localizedDescription
        continuation.resume(throwing: error)
        return
      }
    }

    // D3 `shift`: refuse before dispatch so no family can silently ignore it.
    if let message = GeneratePayload.validateShift(payload.shift, family: currentModelFamily) {
      lastError = message
      continuation.resume(throwing: WarmServerError.invalidRequest(message: message))
      return
    }
    // WP-E10 "E9b": `vae` on a non-krea2 family is refused, never ignored.
    if let error = GeneratePayload.vaeGate(payload.vae, family: currentModelFamily) {
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
      return
    }
    // WP-E17 (§3.14, D4): `stage2` on a family with no second-stage seam, an
    // out-of-range `stage2.denoise`, and the tool schema's `detail_pass` /
    // `detail_denoise` spelling — all 400 here, never a silently single-stage
    // render.
    if let error = GeneratePayload.stage2Gate(payload, family: currentModelFamily) {
      lastError = error.localizedDescription
      continuation.resume(throwing: error)
      return
    }

    // I5: the family capability gate. The name resolved at decode (unknown
    // names are already 400 by then); whether THIS family's loop can honour
    // it is decided here, from the one matrix (D18: family gates at dispatch).
    if let names = try? payload.validateRecipeNames(),
       let error = GeneratePayload.validateFamilyRecipe(names, family: currentModelFamily) {
      lastError = error.localizedDescription ?? "unsupported sampler"
      continuation.resume(throwing: error)
      return
    }

    switch currentModelFamily {
    case .chroma:
      await runChromaGenerate(payload, continuation: continuation)
    case .fibo:
      await runFiboGenerate(payload, continuation: continuation)
    case .krea2:
      await runKrea2Generate(payload, continuation: continuation)
    case .flux2:
      await runFlux2Generate(payload, continuation: continuation)
    case .flux1:
      await runFlux1Generate(payload, continuation: continuation, progressHandler: trackedHandler, latentPreviewHandler: trackedPreviewHandler)
    }
  }

  /// How often (in denoising steps) to refresh the live preview frame.
  private static let previewStepInterval = 2
  private static let previewMaxDimension = 256
  private static let previewJPEGQuality: CGFloat = 0.6

  /// The latest live-denoising preview JPEG, if a render is active and has
  /// produced at least one frame. Served by GET /v1/generate/preview.
  func latestPreviewFrame() -> Data? {
    previewTracker.get()
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
        defaultFilename: ComfyBoxOutputNaming.defaultFilename(
          modelSpec: activePoolModelSpec ?? "flux2", presetId: payload.preset,
          contentMode: payload.contentMode, source: payload.source)
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

  private func runKrea2Generate(_ payload: GeneratePayload, continuation: ContinuationBox<GenerateResponse>) async {
    activeRenderStartedAt = Date()
    let start = Date()
    var resumed = false
    defer {
      if !resumed {
        logger.error("runKrea2Generate: continuation was not resumed — resuming with error.")
        failedRenderCount += 1
        lastError = "Krea2 generation failed unexpectedly (continuation not resumed)"
        activeRenderStartedAt = nil
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "Krea2 generation failed unexpectedly"))
      }
    }
    do {
      guard let k2 = krea2Pipeline else {
        throw WarmServerError.krea2NotLoaded
      }
      // WP-E4 (D18): krea2-only tier gates — a 400, never a silent downgrade.
      try payload.validateKrea2TierGates(try payload.validateRecipeNames())
      // WP-E3 (§3.3, D11, D25): the sampler, the sigma schedule and the shift
      // the caller asked for, forwarded into the request the loop dispatches on.
      let recipe = try payload.krea2RecipeFields()
      // WP-E17 (§3.14, D4): the second stage, resolved fail-loud — an unknown
      // sampler/schedule name on the STAGE throws here, before any model work,
      // exactly as it does for the render's own recipe.
      let stage2 = try payload.krea2Stage2Fields()
      let samplerAsked: String = recipe.samplerRequested ?? "-"
      let scheduleAsked: String = recipe.sigmaScheduleRequested ?? "-"
      let shiftLabel: String = recipe.shift.map { "\($0)" } ?? "dynamic"
      let recipeLine: String =
        "Krea2 recipe: sampler=\(recipe.sampler.rawValue) (requested \(samplerAsked)) "
        + "sigma_schedule=\(recipe.sigmaSchedule.rawValue) (requested \(scheduleAsked)) "
        + "shift=\(shiftLabel) eta=\(recipe.eta) bongmath=\(recipe.bongmath)"
      logger.info("\(recipeLine)")
      if let stage2 {
        let stage2Line: String =
          "Krea2 stage 2: sampler=\(stage2.sampler?.rawValue ?? recipe.sampler.rawValue) "
          + "sigma_schedule=\(stage2.sigmaSchedule?.rawValue ?? recipe.sigmaSchedule.rawValue) "
          + "steps=\(stage2.steps) denoise=\(stage2.denoise)"
        logger.info("\(stage2Line)")
      }
      // WP-E9 (§3.9, D16, D17): VAE selection — payload.vae → model dir. A
      // named file that is not on disk fails the render here (AC-56); a
      // different file than the resident one reloads the decoder IN PLACE on
      // the one Krea2VAE (never a pool eviction), and the selection is
      // recorded on the pipeline for the response record (WP-E10).
      let vaeChoice = try Krea2VAESelector.resolve(requested: payload.vae, paths: k2.paths)
      try k2.ensureVAE(path: vaeChoice.file, source: vaeChoice.source)
      logger.info("Krea2: VAE \(k2.currentVAE.layout.rawValue) \(k2.currentVAE.file.path) (source=\(k2.currentVAE.source.rawValue), reloads=\(k2.vaeReloadCount))")
      let outputURL = try payload.resolvedOutputURL(
        configuration: configuration,
        defaultFilename: ComfyBoxOutputNaming.defaultFilename(
          modelSpec: activePoolModelSpec ?? configuration.modelSpec ?? "krea2", presetId: payload.preset,
          contentMode: payload.contentMode, source: payload.source)
      )

      let seed = payload.seed ?? UInt64.random(in: 1..<UInt64(UInt32.max))
      // Variant defaults (WP-E5, AC-5b): turbo 9 / 1.0, raw 30 / 1.0 — never 3.5.
      let variant = k2.variant
      let steps = variant.resolvedSteps(payload.steps)
      let guidance = variant.resolvedGuidance(payload.guidance)
      let width = payload.width ?? 1024
      let height = payload.height ?? 1024
      // Krea-2 builds its requests straight from the payload rather than going
      // through makePipelineRequest, so resolve DyPE explicitly here.
      let krea2DyPE = payload.resolvedDyPEConfig(width: width, height: height)

      // img2img fix (2026-07-19): Krea2Pipeline.generateImg2Img already
      // implements VAE-encode + partial-denoise; runKrea2Generate simply never
      // wired it, so an init image was silently ignored (txt2img). When an init
      // image is present, load+normalize it to NHWC [-1,1] and run img2img.
      // strength = 1 - denoise, matching flux1 makeImg2ImgRequest's convention.
      // Depth Control-LoRA: load control weights + encode control tokens when a control image is supplied.
      var controlPixels: MLXArray? = nil
      let resolvedControlData: Data? = payload.controlImageData ?? payload.controlImage.flatMap { try? Data(contentsOf: URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)) }
      if let controlData = resolvedControlData {
        let ccg = try InpaintUtilities.loadCGImage(from: controlData)
        let cpix = try QwenImageIO.resizedPixelArray(from: ccg, width: width, height: height)
        controlPixels = QwenImageIO.normalizeForEncoder(cpix).transposed(0, 2, 3, 1)
        let loraURL = URL(fileURLWithPath: "/Volumes/Bolt/Models/krea2-controlnet/depth-control-lora.safetensors")
        try await k2.setControlLoRA(loraURL, scale: payload.controlnetStrength ?? 1.0)
        logger.info("Krea2: depth Control-LoRA active (strength=\(payload.controlnetStrength ?? 1.0))")
      } else if k2.controlLoRAActive {
        try await k2.setControlLoRA(nil)
      }
      // Rewriter-proof trigger guarantee (Todd 2026-08-11): re-assert every
      // applied LoRA's library trigger on the FINAL prompt — the sealed
      // rewrite happens upstream and can drop activation tokens.
      let loraTriggers: [String] = (payload.loras ?? []).compactMap { entry in
        let filename = (entry.path as NSString).lastPathComponent
        return loraLibrary?.entry(for: filename)?.triggerwords.first
      }
      let guardedPrompt = LoRATriggerGuard.ensure(prompt: payload.prompt, triggers: loraTriggers)

      // WP-E10: publish per-step progress the way the Z-Image path does.
      // `/health.progress_percent` used to stay 0 for the whole of a Krea 2
      // render because this arm's callback only logged.
      let tracker = progressTracker
      let health = liveHealth
      let publishProgress: @Sendable (Int, Int) -> Void = { [logger] step, total in
        let pct = RenderProgressPercent.of(step: step, total: total)
        tracker.set(pct)
        health.setProgress(pct)
        logger.info("Krea2: step \(step)/\(total)")
      }

      let image: MLXArray
      // WP-E17: one trace per stage that ran, in order. `traces[0]` is the
      // render's own — the geometry, seed and schedule shift every sink reads.
      let traces: [Krea2RunTrace]
      if let initPath = payload.imagePath {
        let imageData = try Data(contentsOf: URL(fileURLWithPath: initPath))
        let cg = try InpaintUtilities.loadCGImage(from: imageData)
        let pixNCHW = try QwenImageIO.resizedPixelArray(from: cg, width: width, height: height)
        let sourceNHWC = QwenImageIO.normalizeForEncoder(pixNCHW).transposed(0, 2, 3, 1)
        let strength: Float
        if let c = payload.creativity {
          strength = 1.0 - max(0.01, min(0.99, c))
        } else if let sVal = payload.imageStrength {
          strength = max(0.01, min(0.99, sVal))
        } else if let d = payload.denoise {
          strength = 1.0 - max(0.01, min(0.99, d))
        } else {
          strength = 0.3
        }
        logger.info("Krea2: img2img init=\(initPath) strength=\(strength)")
        // §3.14 leaves img2img alone on purpose: its `strength → startIndex`
        // rule is the established contract on this path, and stage 2 is a
        // different, differently-specified mechanism (AC-30 keeps them
        // distinguishable). Composing the two is unspecified, so it is refused
        // rather than silently resolved one way.
        guard stage2 == nil else {
          throw WarmServerError.mutuallyExclusive(
            "stage2 and image_path cannot be combined: the second stage re-noises the LATENT to "
              + "the stretched tail's first sigma (WP-E17), while img2img starts partway down the "
              + "grid from `strength` — two different mechanisms with no defined composition. "
              + "Send one of them")
        }
        let trace1: Krea2RunTrace
        (image, trace1) = try k2.generateImg2ImgWithRecipe(
          .init(prompt: guardedPrompt, negativePrompt: payload.negativePrompt,
                guidance: guidance,
                sourceImage: sourceNHWC, width: width, height: height,
                steps: steps, seed: seed, strength: strength, dyPE: krea2DyPE,
                shift: recipe.shift,
                sampler: recipe.sampler, sigmaSchedule: recipe.sigmaSchedule,
                sigmaScheduleRequested: recipe.sigmaScheduleRequested,
                eta: recipe.eta, bongmath: recipe.bongmath),
          progress: publishProgress)
        traces = [trace1]
      } else {
        // WP-E17: `generateStaged` IS `generateWithRecipe`'s body. Without
        // `stage2` it executes the same statements it did before this WP —
        // which is what makes the byte-identity gates hold by construction —
        // and with it, the second stage runs before the one `vae.decode`.
        (image, traces) = try k2.generateStaged(
          .init(prompt: guardedPrompt, negativePrompt: payload.negativePrompt,
                guidance: guidance,
                width: width, height: height, steps: steps, seed: seed,
                controlImagePixels: controlPixels, dyPE: krea2DyPE,
                shift: recipe.shift,
                sampler: recipe.sampler, sigmaSchedule: recipe.sigmaSchedule,
                sigmaScheduleRequested: recipe.sigmaScheduleRequested,
                eta: recipe.eta, bongmath: recipe.bongmath, stage2: stage2),
          progress: publishProgress)
      }
      let trace = traces[0]
      // WP-E10 (FDD §3.10, AC-60): the provenance record is READ BACK from
      // the pipeline — the variant and transformer file it loaded, the
      // quantization it applied, the VAE resident in its slot, its loaded
      // LoRA configs joined with their bind reports, and the run trace the
      // loop just counted. `steps`/`guidance` above are NOT consulted here.
      // Fail CLOSED on an incomplete read-back: a record naming two of three
      // adapters is worse than no record, because it reads as complete.
      let loraReadBacks = RenderRecipe.loRAReadBacks(
        configs: k2.loadedLoRAConfigs, reports: k2.loadedLoRAReports,
        // I6: the relativity the guard ENFORCED, so `relative_to` names what
        // was applied rather than what the request happened to declare.
        relativities: k2.loadedLoRARelativities)
      if loraReadBacks == nil {
        let mismatch = "Krea2 provenance: \(k2.loadedLoRAConfigs.count) loaded LoRA configs but "
          + "\(k2.loadedLoRAReports.count) bind reports — refusing to emit a partial `applied` for this render"
        logger.error("\(mismatch)")
      }
      // `base_model` is the declared alias when the active spec is one (or
      // resolves to one's directory — AC-34b), else the spec as loaded.
      let activeSpec = activePoolModelSpec ?? configuration.modelSpec ?? "krea2"
      let record: RenderRecipe? = loraReadBacks.map { readBacks in
        RenderRecipe.krea2(.init(
          baseModel: Krea2ModelDetection.alias(forSpec: activeSpec) ?? activeSpec,
          variant: k2.variant,
          transformerFile: k2.paths.transformerFile,
          quantizationBits: k2.transformerQuantBits,
          vae: k2.currentVAE,
          textEncoderFile: k2.paths.textEncoderFile,
          loras: readBacks,
          control: k2.controlLoRAActive ? k2.controlLoRAApplied : nil,
          // D4 / WP-E17: every stage that ran, so `applied.stages[]` and
          // `model_evals_total` describe the whole render rather than its
          // first half.
          traces: traces))
      }
      // Sink 2 — the PNG. The negative comes from the TRACE (K-FIX-1 / I4):
      // absent when CFG never ran (AC-61), and an applied `""` is written as
      // `""` rather than dropped, so the file records the second model pass
      // an omitted negative still paid for.
      let metadata = QwenImageIO.ImageMetadata.generation(
        prompt: guardedPrompt,
        negativePrompt: trace.negativePromptApplied,
        seed: trace.seed,
        // WP-E17: the flat scalar a human reads names the WHOLE render, not its
        // first stage — `applied.stages[]` is where the split lives. Identical
        // to `trace.stepsRequested` for a single-stage render.
        steps: traces.reduce(0) { $0 + $1.stepsRequested },
        guidance: trace.guidance,
        width: trace.width,
        height: trace.height,
        model: ComfyBoxOutputNaming.shortModelName(activePoolModelSpec ?? configuration.modelSpec),
        generatedBy: payload.source,
        contentMode: payload.contentMode,
        loras: k2.loadedLoRAConfigs,
        // The SLOT, so a refused record writes `"applied": null` in the file
        // rather than looking like a family that has no record (round 2, C4).
        appliedSlot: AppliedRecordSlot(record: record)
      )
      try QwenImageIO.saveImage(array: image.transposed(2, 0, 1), to: outputURL, metadata: metadata)

      let durationMs = Int(Date().timeIntervalSince(start) * 1000.0)
      successfulRenderCount += 1
      lastRenderDurationMs = durationMs
      lastError = nil
      // Round 2 (C4): THIS render is Krea 2, so the slot is always present —
      // it carries the record, or a literal `null` when `loraReadBacks` refused
      // it above. An absent key must never be able to mean "engine-incomplete".
      let applied = AppliedRecordSlot(record: record)
      lastRecipe = applied  // sink 3 — /health.last_recipe
      activeRenderStartedAt = nil
      resumed = true
      // sink 1 — the response; sink 4 reads `applied` off this same value.
      continuation.resume(returning: GenerateResponse(
        success: true, outputPath: outputURL.path, durationMs: durationMs, applied: applied))
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
        defaultFilename: ComfyBoxOutputNaming.defaultFilename(
          modelSpec: activePoolModelSpec ?? "fibo", presetId: payload.preset,
          contentMode: payload.contentMode, source: payload.source)
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

    // K-FIX-1 / Codex I5: Chroma HAS native heun and beta and this call used
    // to pass neither, so `scheduler: "heun"` rendered Euler pixels under the
    // name "heun". The pair is mapped through the one family matrix — which
    // has already refused (400) any pair it cannot map, so the fallback here
    // is the unreachable default, not a silent substitution.
    let names = try payload.validateRecipeNames()
    let chromaScheduler = FamilyRecipeMatrix.chromaSchedulerType(
      sampler: names.scheduler, schedule: names.sigmaSchedule) ?? .euler

    // Generate — returns MLXArray in [B, H, W, C] (NHWC, values [0,1])
    let result = try pipeline.generate(
      tokenIds: tokenIds,
      negativeTokenIds: negTokenIds,
      width: width,
      height: height,
      numSteps: steps,
      guidance: guidance,
      cfg: cfgScale,
      firstNStepsWithoutCFG: cfgWarmup,
      schedulerType: chromaScheduler,
      seed: seed,
      progressCallback: { step, total in
        // Progress logging
      }
    )

    // Transpose from NHWC [1, H, W, 3] to CHW [3, H, W] for QwenImageIO
    let imageArray = result.squeezed(axis: 0).transposed(2, 0, 1)

    // Save image (with embedded, Finder-readable generation metadata)
    try QwenImageIO.saveImage(array: imageArray, to: outputURL,
      // Non-Krea 2 families hand `ImageMetadata` a RAW payload value, where
      // `""` means "not given" — normalised through the one shared helper so
      // the I4 change to `ImageMetadata.generation` (an explicit `""` is
      // written) cannot add a `negative_prompt: ""` key to any other family's
      // PNG. Krea 2's APPLIED value deliberately does not go through it.
      metadata: .generation(prompt: payload.prompt,
        negativePrompt: QwenImageIO.ImageMetadata.requestNegative(payload.negativePrompt),
        seed: seed, steps: steps, guidance: guidance, width: width, height: height,
        generatedBy: payload.source, contentMode: payload.contentMode, loras: loras))
  }

  private func runControlGenerate(_ request: ZImageControlGenerationRequest, continuation: ContinuationBox<GenerateResponse>) async {
    if currentModelFamily == .flux2 || currentModelFamily == .fibo || currentModelFamily == .chroma || currentModelFamily == .krea2 {
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

  /// Currently-loaded LoRA configs for whichever pipeline is active — used to
  /// resync `activeLoRAs` after a failed or crashed swap.
  private func loadedLoRAConfigs(for family: WarmModelFamily?) -> [LoRAConfiguration] {
    switch family {
    case .flux2: return flux2Pipeline?.loadedLoRAConfigs ?? []
    case .krea2: return krea2Pipeline?.loadedLoRAConfigs ?? []
    default: return pipeline.loadedLoRAConfigs
    }
  }

  /// Apply LoRAs to whichever pipeline is active for `currentModelFamily`.
  /// Shared by POST /v1/lora/swap and per-job LoRA application at generate
  /// dequeue time (queue-submit race fix — see GeneratePayload.loras).
  private func applyActiveLoRAs(_ newLoRAs: [LoRAConfiguration]) async throws {
    if currentModelFamily == .flux2 {
      guard let f2 = flux2Pipeline else { throw WarmServerError.flux2NotLoaded }
      try await f2.loadLoRAs(newLoRAs)
      activeLoRAs = newLoRAs
    } else if currentModelFamily == .krea2 {
      guard let k2 = krea2Pipeline else { throw WarmServerError.krea2NotLoaded }
      // WP-E6: fold the library's declared relativity into any config that
      // did not declare one itself (request > library > seed — never inferred).
      let declared = newLoRAs.map { cfg -> LoRAConfiguration in
        guard cfg.requiresBase == nil, case .local(let url) = cfg.source,
              let entry = loraLibrary?.entry(for: url.lastPathComponent),
              let relative = entry.krea2Relative
        else { return cfg }
        var out = cfg
        out.requiresBase = relative
        return out
      }
      try await k2.loadLoRAs(declared)
      activeLoRAs = declared
    } else {
      try await pipeline.swapLoRAs(newLoRAs)
      activeLoRAs = newLoRAs
    }
    // Active LoRA set changed — refresh the health snapshot (#217).
    publishHealth()
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
        activeLoRAs = loadedLoRAConfigs(for: currentModelFamily)
        lastError = "LoRA swap failed unexpectedly (continuation not resumed)"
        continuation.resume(throwing: WarmServerError.invalidRequest(message: "LoRA swap failed unexpectedly"))
      }
    }

    do {
      // A swap-first client (kira-daemon: swap → generate) must not fail
      // because the image pipeline is not resident — video eviction (#218)
      // or a fresh boot leaves it nil, and the only restore path lived in
      // runGenerate, so a failed swap meant generate was never called and
      // image creation deadlocked until an out-of-band generate landed.
      let familyPipelineMissing =
        (currentModelFamily == .krea2 && krea2Pipeline == nil)
        || (currentModelFamily == .flux2 && flux2Pipeline == nil)
      let restoreSpec = activePoolModelSpec ?? lastActiveImageSpec ?? configuration.modelSpec
      switch SwapResidencyRestore.decide(
        imageModelsEvicted: imageModelsEvicted,
        familyPipelineMissing: familyPipelineMissing,
        restoreSpec: restoreSpec
      ) {
      case .none:
        break
      case .reloadEvicted:
        try await reloadImageModelIfEvicted(requestedModel: nil)
      case .load(let modelSpec):
        logger.info("Swap arrived with no resident image pipeline — loading '\(modelSpec)' before applying LoRAs")
        _ = try await poolLoad(
          modelSpec: WarmServer.parseModelSpec(from: modelSpec),
          quantization: WarmServer.parseQuantization(from: modelSpec),
          activate: true)
      }

      let newLoRAs = try payload.makeConfigurations()
      try await applyActiveLoRAs(newLoRAs)

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
      activeLoRAs = loadedLoRAConfigs(for: currentModelFamily)
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
    case 502: return "Bad Gateway"
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

/// WP-E17 (FDD-krea2-raw-recipe §3.14, D4, D25): the `stage2` object on
/// `/v1/generate` — the detail pass, inside ONE render.
///
/// Additive: a payload without it is byte-identical to today. `steps` and
/// `denoise` are REQUIRED because they are the two fields that decide the
/// stretched grid, and a default for either would be an engine-invented recipe;
/// everything else absent means "the render's own value" (see
/// ``Krea2Pipeline/Stage2``). The family's published pairing lives in the
/// client's policy table (WP-C8) and arrives here spelled out.
struct Stage2Payload: Sendable, Decodable, Equatable {
  let steps: Int
  /// **`Double`** — every other float on `GeneratePayload` decodes as `Float`
  /// and this one must not: `total = int(steps/denoise)` is sensitive to which
  /// side of the integer the division lands on (§3.14, AC-31).
  let denoise: Double
  /// D25: the wire key is `scheduler`, with `sampler` as an accepted alias;
  /// both present and different is a 400, exactly as at the top level.
  let scheduler: String?
  let sigmaSchedule: String?
  let guidance: Float?
  let eta: Float?
  let bongmath: Bool?
  /// `null`/absent → the stage-1 seed `&+ 1`, recorded either way.
  let seed: UInt64?

  private enum CodingKeys: String, CodingKey {
    case steps, denoise, scheduler, sampler, sigmaSchedule, guidance, eta, bongmath, seed
  }

  init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    guard let steps = try c.decodeIfPresent(Int.self, forKey: .steps) else {
      throw WarmServerError.invalidRequest(
        message: "stage2.steps is required — it decides the stretched grid and has no default")
    }
    guard let denoise = try c.decodeIfPresent(Double.self, forKey: .denoise) else {
      throw WarmServerError.invalidRequest(
        message: "stage2.denoise is required — it decides the stretched grid and has no default")
    }
    self.steps = steps
    self.denoise = denoise
    let schedulerRaw = try c.decodeIfPresent(String.self, forKey: .scheduler)
    let samplerRaw = try c.decodeIfPresent(String.self, forKey: .sampler)
    if let schedulerRaw, let samplerRaw, schedulerRaw != samplerRaw {
      throw WarmServerError.mutuallyExclusive(
        "stage2.scheduler='\(schedulerRaw)' and stage2.sampler='\(samplerRaw)' disagree — "
          + "'sampler' is an alias of 'scheduler'; send one, or the same value in both")
    }
    self.scheduler = schedulerRaw ?? samplerRaw
    self.sigmaSchedule = try c.decodeIfPresent(String.self, forKey: .sigmaSchedule)
    self.guidance = try c.decodeIfPresent(Float.self, forKey: .guidance)
    self.eta = try c.decodeIfPresent(Float.self, forKey: .eta)
    self.bongmath = try c.decodeIfPresent(Bool.self, forKey: .bongmath)
    self.seed = try c.decodeIfPresent(UInt64.self, forKey: .seed)
  }

  /// For bridge/test construction; the wire always goes through `init(from:)`.
  init(
    steps: Int, denoise: Double, scheduler: String? = nil, sigmaSchedule: String? = nil,
    guidance: Float? = nil, eta: Float? = nil, bongmath: Bool? = nil, seed: UInt64? = nil
  ) {
    self.steps = steps
    self.denoise = denoise
    self.scheduler = scheduler
    self.sigmaSchedule = sigmaSchedule
    self.guidance = guidance
    self.eta = eta
    self.bongmath = bongmath
    self.seed = seed
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
  /// RES4LYF `bongmath` (parity tier T3, WP-E16). Krea 2 + the RES4LYF
  /// samplers only; asked for with any other sampler it is a 400 naming the
  /// sampler (`validateKrea2TierGates`), never a silent drop. Absent/false is
  /// byte-identical to today.
  let bongmath: Bool?
  /// Explicit schedule shift (FDD-krea2-raw-recipe D3, Addendum A.1). Krea 2
  /// only: nil = the resolution-dependent default; a value IS `mu` (ComfyUI's
  /// `ModelSamplingFlux(shift=…)` parameterisation — `1.15` reproduces the
  /// published grid). Validated by `validateShift(_:family:)` → 400, never clamped.
  let shift: Float?
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

  /// Optional mask PNG file path for selective inpainting on the img2img path.
  /// White = inpaint region, black = keep. nil → standard full-frame img2img.
  let maskPath: String?

  /// Auto-generated mask region ("face" | "upper" | "lower") for img2img —
  /// mutually exclusive with maskPath. The named region is regenerated.
  let maskRegion: String?

  /// Flip the img2img mask (white ⇄ black): e.g. mask_region "face" +
  /// mask_invert = lock the face, regenerate everything else.
  let maskInvert: Bool?

  /// Submitting client/app (desktop, bree, api…) — for queue attribution.
  let source: String?

  /// Preset id that produced this request, when one was resolved — a LABEL
  /// only (image presets resolve DAEMON-side; the engine never expands it).
  /// Carried into the gallery filename + PNG metadata (Todd 2026-08-11).
  let preset: String?

  /// Fruit mode (neutral | banana | avocado) — stamped into render metadata.
  let contentMode: String?

  /// Per-job model override (spec/CivitAI id/pool key). When set, the job's
  /// own model is loaded/activated at dequeue time instead of trusting
  /// whatever the shared pool's "currently active" model happens to be —
  /// required for queue-submit (POST /v1/generate/async) to be race-free,
  /// since a job's dequeue can happen well after another request changed
  /// the active model. nil preserves the old "caller activates first"
  /// behavior for direct /v1/generate callers.
  let model: String?
  /// Per-job LoRA override, applied the same way as `model` at dequeue time.
  let loras: [LoRAEntry]?
  // Depth Control-LoRA (docs/FDD-krea2-depth-controlnet.md)
  let controlImageData: Data?
  let controlnetStrength: Float?
  let controlImage: String?  // Mac-side control map path (e.g. depth), read in place

  /// #1479: request an in-flight LTX-2 video render be checkpointed so this
  /// image job can run immediately, resuming the video afterward. Additive,
  /// default absent/false — omitting it is byte-identical to today.
  let preempt: Bool?

  /// WP-E9 (FDD §3.9, D16): path of the VAE file to decode (and encode)
  /// through — e.g. `Wan2_1_VAE_fp32.safetensors`. Tilde allowed. Absent →
  /// the model directory's VAE. A path that is not on disk FAILS the render
  /// (AC-56); the layout is sniffed from the file's keys, never its name.
  /// Krea 2 only today (the other families ignore it).
  let vae: String?

  /// WP-E17 (§3.14, D4): the second stage of this render. Krea 2 only —
  /// refused, never ignored, on any other family (``stage2Gate(_:family:)``).
  let stage2: Stage2Payload?

  /// The MCP tool schema's spelling of a detail pass (§3.17, AC-68a): the
  /// CLIENT expands `detail_pass` into `stage2` from its family policy table.
  /// Decoded here so the engine can REFUSE them by name — it has no policy
  /// table to expand a bare boolean into a sampler/schedule/step recipe, and
  /// inventing one is the silent substitution this programme exists to kill.
  let detailPass: Bool?
  /// `detail_denoise` without `detail_pass` is an orphan (Addendum A.2 → C3),
  /// NaN included. `Double` for the same reason `stage2.denoise` is.
  let detailDenoise: Double?

  /// Default memberwise init for bridge-created payloads.
  init(
    prompt: String, negativePrompt: String? = nil,
    width: Int? = nil, height: Int? = nil, steps: Int? = nil,
    guidance: Float? = nil, seed: UInt64? = nil, outputPath: String? = nil,
    levelsMin: Float? = nil, levelsMax: Float? = nil,
    scheduler: String? = nil, sigmaSchedule: String? = nil, eta: Float? = nil,
    bongmath: Bool? = nil,
    shift: Float? = nil,
    dype: String? = nil, inpaintImageData: Data? = nil, maskData: Data? = nil,
    denoise: Float? = nil, maskGrow: Int? = nil, maskFeather: Int? = nil,
    maskCropX: Int? = nil, maskCropY: Int? = nil,
    cfg: Float? = nil, firstNStepsWithoutCFG: Int? = nil,
    imagePath: String? = nil, imageStrength: Float? = nil, creativity: Float? = nil,
    maskPath: String? = nil, maskRegion: String? = nil, maskInvert: Bool? = nil,
    source: String? = nil, contentMode: String? = nil, initImageData: Data? = nil,
    model: String? = nil, loras: [LoRAEntry]? = nil,
    controlImageData: Data? = nil, controlnetStrength: Float? = nil, controlImage: String? = nil,
    preempt: Bool? = nil, vae: String? = nil,
    stage2: Stage2Payload? = nil, detailPass: Bool? = nil, detailDenoise: Double? = nil
  ) {
    self.preempt = preempt
    self.vae = vae
    self.stage2 = stage2
    self.detailPass = detailPass
    self.detailDenoise = detailDenoise
    self.source = source
    self.preset = nil
    self.contentMode = contentMode
    self.initImageData = initImageData
    self.model = model
    self.loras = loras
    self.controlImageData = controlImageData; self.controlnetStrength = controlnetStrength; self.controlImage = controlImage
    self.prompt = prompt; self.negativePrompt = negativePrompt
    self.width = width; self.height = height; self.steps = steps
    self.guidance = guidance; self.seed = seed; self.outputPath = outputPath
    self.levelsMin = levelsMin; self.levelsMax = levelsMax
    self.scheduler = scheduler; self.sigmaSchedule = sigmaSchedule
    self.eta = eta; self.bongmath = bongmath; self.shift = shift; self.dype = dype
    self.inpaintImageData = inpaintImageData; self.maskData = maskData
    self.denoise = denoise; self.maskGrow = maskGrow; self.maskFeather = maskFeather
    self.maskCropX = maskCropX; self.maskCropY = maskCropY
    self.cfg = cfg; self.firstNStepsWithoutCFG = firstNStepsWithoutCFG
    self.imagePath = imagePath; self.imageStrength = imageStrength; self.creativity = creativity
    self.maskPath = maskPath
    self.maskRegion = maskRegion
    self.maskInvert = maskInvert
  }
}

extension GeneratePayload: Decodable {
  private enum CodingKeys: String, CodingKey {
    case prompt, negativePrompt, width, height, steps, guidance, seed
    case outputPath, levelsMin, levelsMax, scheduler, sigmaSchedule, eta, bongmath, shift, dype
    /// D25: `sampler` is an accepted ALIAS of the wire key `scheduler` (so a
    /// value pasted out of a ComfyUI/RES4LYF UI works). The request key stays
    /// `scheduler` — two live senders post that spelling — while the response
    /// record reports `applied.stages[].sampler` (WP-E10); the asymmetry is
    /// deliberate. Both present and different → `mutuallyExclusive` (400).
    case sampler
    case preset
    case denoise, maskGrow, maskFeather
    // NOTE: the /v1/generate decoder uses .convertFromSnakeCase, which rewrites
    // incoming keys to camelCase BEFORE matching CodingKey stringValues. So the
    // wire keys inpaint_image_base64 / mask_base64 arrive as these camelCase
    // forms — the rawValues MUST be the post-conversion spelling, not snake_case.
    case inpaintImageData = "inpaintImageBase64"
    case maskImageData = "maskBase64"
    case cfg, firstNStepsWithoutCFG
    case imagePath, imageStrength, creativity
    case maskPath
    case maskRegion
    case maskInvert
    case source
    case contentMode
    // Wire key init_image_base64 arrives as this camelCase form after
    // .convertFromSnakeCase (same gotcha as the inpaint keys).
    case initImageData = "initImageBase64"
    case controlImageData
    case controlnetStrength
    case controlImage
    case model, loras
    case preempt
    case vae
    // WP-E17. `stage2` has no underscore, so `.convertFromSnakeCase` leaves it
    // alone; `detail_pass` / `detail_denoise` arrive as these camelCase forms.
    case stage2
    case detailPass
    case detailDenoise
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
    let schedulerRaw = try c.decodeIfPresent(String.self, forKey: .scheduler)
    let samplerRaw = try c.decodeIfPresent(String.self, forKey: .sampler)
    if let schedulerRaw, let samplerRaw, schedulerRaw != samplerRaw {
      throw WarmServerError.mutuallyExclusive(
        "scheduler='\(schedulerRaw)' and sampler='\(samplerRaw)' disagree — 'sampler' is an alias of 'scheduler'; send one, or the same value in both")
    }
    scheduler = schedulerRaw ?? samplerRaw
    sigmaSchedule = try c.decodeIfPresent(String.self, forKey: .sigmaSchedule)
    eta = try c.decodeIfPresent(Float.self, forKey: .eta)
    bongmath = try c.decodeIfPresent(Bool.self, forKey: .bongmath)
    shift = try c.decodeIfPresent(Float.self, forKey: .shift)
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
    maskPath = try c.decodeIfPresent(String.self, forKey: .maskPath)
    maskRegion = try c.decodeIfPresent(String.self, forKey: .maskRegion)
    maskInvert = try c.decodeIfPresent(Bool.self, forKey: .maskInvert)
    source = try c.decodeIfPresent(String.self, forKey: .source)
    preset = try c.decodeIfPresent(String.self, forKey: .preset)
    contentMode = try c.decodeIfPresent(String.self, forKey: .contentMode)
    model = try c.decodeIfPresent(String.self, forKey: .model)
    loras = try c.decodeIfPresent([LoRAEntry].self, forKey: .loras)
    controlImageData = (try c.decodeIfPresent(String.self, forKey: .controlImageData)).flatMap { Data(base64Encoded: $0) }
    controlnetStrength = try c.decodeIfPresent(Float.self, forKey: .controlnetStrength)
    controlImage = try c.decodeIfPresent(String.self, forKey: .controlImage)
    preempt = try c.decodeIfPresent(Bool.self, forKey: .preempt)
    vae = try c.decodeIfPresent(String.self, forKey: .vae)
    stage2 = try c.decodeIfPresent(Stage2Payload.self, forKey: .stage2)
    detailPass = try c.decodeIfPresent(Bool.self, forKey: .detailPass)
    detailDenoise = try c.decodeIfPresent(Double.self, forKey: .detailDenoise)
  }

  /// Validate the D3 `shift` field for the family that will render it.
  ///
  /// Returns a 400 message, or nil when the request is acceptable. `shift` is
  /// a Krea 2 schedule field: it must be a positive finite number, and it is
  /// refused — not ignored — on any other family, whose schedules never read
  /// it (FDD-krea2-raw-recipe D3; "fail loud, never silently substitute").
  static func validateShift(_ shift: Float?, family: WarmModelFamily) -> String? {
    guard let shift else { return nil }
    guard shift.isFinite, shift > 0 else {
      return "shift must be a positive number (got \(shift)); omit it for the resolution-dependent default"
    }
    guard family == .krea2 else {
      return "shift is a Krea 2 request field and is not honoured by model family '\(family.rawValue)'; remove it"
    }
    return nil
  }

  /// WP-E10 "E9b" (Addendum A.2, E9 review MAJOR): `vae` is a Krea 2 request
  /// field. On any other family it used to be silently ignored — the caller
  /// named a decoder and got the family's default with no error. Refused
  /// with the D18 shape (`unsupportedRecipeField` → 400) naming the field and
  /// the family. nil when the request carries no `vae` or the family honours it.
  static func vaeGate(_ vae: String?, family: WarmModelFamily) -> WarmServerError? {
    guard let vae, family != .krea2 else { return nil }
    return .unsupportedRecipeField(
      field: "vae", value: vae, family: family.rawValue,
      reason: "VAE selection is a Krea 2 request field (WP-E9); this family decodes through its own VAE and does not honour it — remove it")
  }

  /// K-FIX-1 / Codex I5: refuse a sampler / sigma schedule the ACTIVE FAMILY
  /// cannot honour, from the one family capability matrix
  /// (``FamilyRecipeMatrix``).
  ///
  /// Supersedes WP-E13's `validateTableauSampler`, which is now one row of
  /// that table: N-row tableaus (`ralston_2s/3s/4s`, `res_3s`) are dispatched
  /// only by `Krea2DenoiseLoop`. The table also closes the silences E13's
  /// single row left open — Chroma ignoring its own native `heun`/`beta`, and
  /// Flux 2 / FIBO accepting any name into a fixed Euler loop.
  ///
  /// Names stay accepted and advertised globally (E4: advertised == accepted
  /// as a NAME); what is family-scoped is whether the render can honour them.
  /// Returns the 400 to throw, or nil. Runs at the one dispatch point in
  /// `generate` and in the bridge's family arm, beside
  /// `validateShift(_:family:)` (D18: family gates live at dispatch, not at
  /// the decoder).
  static func validateFamilyRecipe(
    _ names: ResolvedRecipeNames, family: WarmModelFamily
  ) -> WarmServerError? {
    FamilyRecipeMatrix.validate(names, family: family)
  }

  /// The DyPE configuration this payload implies at the given resolution.
  ///
  /// An explicit `dype` always wins, including "none". Otherwise DyPE
  /// auto-enables above the model's base resolution — the branch that matters
  /// most, since the callers that need it (Kira's HQ 2K rerender, the Krita
  /// bridge) send no `dype` at all.
  ///
  /// `.ntk` is deliberately the ceiling: `.yarn` is an unimplemented stub that
  /// warns and falls back to NTK, so selecting it would only add log noise.
  func resolvedDyPEConfig(width resolvedWidth: Int, height resolvedHeight: Int) -> DyPEConfig {
    if let raw = dype?.lowercased() {
      switch raw {
      case "ntk": return .ntk
      case "yarn": return .yarn
      default: return .disabled
      }
    }
    return max(resolvedWidth, resolvedHeight) > 1024 ? .ntk : .disabled
  }

  func makePipelineRequest(
    configuration: WarmServerConfiguration,
    activeLoRAs: [LoRAConfiguration]
  ) throws -> ZImageGenerationRequest {
    let outputURL = try resolvedOutputURL(
      configuration: configuration,
      defaultFilename: ComfyBoxOutputNaming.defaultFilename(
        modelSpec: configuration.modelSpec ?? "z-image", presetId: preset,
        contentMode: contentMode, source: source)
    )

    let names = try validateRecipeNames()
    let schedulerKind = names.scheduler ?? .euler
    let sigmaScheduleKind = names.sigmaSchedule ?? .flow

    // Build DyPE config — auto-enable for high-res requests
    let resolvedWidth = width ?? ZImageModelMetadata.recommendedWidth
    let resolvedHeight = height ?? ZImageModelMetadata.recommendedHeight
    let dyPEConfig = resolvedDyPEConfig(width: resolvedWidth, height: resolvedHeight)

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

    let names = try validateRecipeNames()
    let schedulerKind = names.scheduler ?? .euler
    let sigmaScheduleKind = names.sigmaSchedule ?? .flow

    let resolvedWidth = width ?? ZImageModelMetadata.recommendedWidth
    let resolvedHeight = height ?? ZImageModelMetadata.recommendedHeight
    let dyPEConfig = resolvedDyPEConfig(width: resolvedWidth, height: resolvedHeight)

    let outputURL = try resolvedOutputURL(
      configuration: configuration,
      defaultFilename: ComfyBoxOutputNaming.defaultFilename(
        modelSpec: configuration.modelSpec ?? "z-image", presetId: preset,
        contentMode: contentMode, source: source)
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
      source: source,
      maskPath: maskPath,
      maskRegion: maskRegion,
      maskInvert: maskInvert ?? false,
      maskGrow: maskGrow ?? 0,
      maskFeather: maskFeather ?? 0
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

  /// WP-E4 (FDD-krea2-raw-recipe §3.4, D22, D25): resolve the sampler and
  /// sigma-schedule names fail-loud. Returns the resolved kinds AND the raw
  /// strings so the record can carry `sigma_schedule_requested`. Absent names
  /// come back nil — the request builders apply euler / flow as today.
  /// Family-agnostic; safe before the family is known.
  func validateRecipeNames() throws -> ResolvedRecipeNames {
    try RecipeNameResolver.resolve(scheduler: scheduler, sigmaSchedule: sigmaSchedule)
  }

  /// WP-E4 (D18, §3.4): Krea 2 tier / capability gates. Runs inside the
  /// Krea 2 generate path and the bridge's `.krea2` arm ONLY — `eta` on the
  /// Z-Image path is a different, shipped parameter (DDIM η / DPM++ 2S-A η)
  /// and keeps working (AC-28).
  ///
  /// The sampler and the sigma-schedule arms went with WP-E3: the Krea 2 loop
  /// dispatches on both, so every name `RecipeNameResolver` accepts is honoured
  /// rather than refused. **The `eta` arm went with WP-E15**: tier T2 has
  /// landed, so a non-zero `eta` is now either applied (the RES4LYF samplers)
  /// or refused BY SAMPLER at `Krea2Pipeline.makeSDEInjector` — it is never
  /// ignored, and the refusal names the sampler rather than the tier.
  ///
  /// `names` is still taken — the unknown-name failure happens in
  /// `validateRecipeNames()`, which the caller runs to produce it, and the
  /// parameter keeps that ordering explicit at every call site.
  ///
  /// **The `bongmath` arm arrived with WP-E16**, and it is the eta arm's twin:
  /// tier T3 has landed, `bongmath` now has a wire key, and it is applied on
  /// the RES4LYF samplers or refused BY SAMPLER — never ignored. Mirrors
  /// `Krea2Pipeline.makeBongMath(bongmath:sampler:sigmaSchedule:shift:)`,
  /// which refuses the same request for a non-server caller.
  ///
  /// What remains is the SAMPLER boundary the SDE has: RES4LYF's `eta` splits
  /// a step against RES4LYF's own prepared grid and re-noises the non-final
  /// rows of its tableau, so it is defined for the RES4LYF ports and for
  /// nothing else. Asked for with `euler` — the Krea 2 default — or with
  /// `ddim` / `dpmpp-2s-a`, where the same wire key already means a different
  /// stochasticity parameter on the Z-Image path, it is a 400 naming the
  /// sampler, never a silent drop. Mirrors
  /// `Krea2Pipeline.makeSDEInjector(eta:sampler:stageSeed:layout:)`, which
  /// refuses the same request for a non-server caller.
  func validateKrea2TierGates(_ names: ResolvedRecipeNames) throws {
    let sampler = names.scheduler ?? .euler
    let res4lyfList =
      "res_2s / res_3s / ralston_2s / ralston_3s / ralston_4s / deis_2m / deis_3m / deis_4m"
    if let eta, eta != 0, !sampler.isRES4LYFFamily {
      throw WarmServerError.unsupportedRecipeField(
        field: "eta", value: "\(eta)", family: "krea2",
        reason: "eta is RES4LYF's SDE (parity tier T2) and applies to the RES4LYF samplers only; "
          + "'\(sampler.rawValue)' is not one of them. Send eta 0, or a sampler from "
          + res4lyfList)
    }
    if bongmath == true, !sampler.isRES4LYFFamily {
      throw WarmServerError.unsupportedRecipeField(
        field: "bongmath", value: "true", family: "krea2",
        reason: "bongmath is RES4LYF's fixed point (parity tier T3) over its own tableau rows "
          + "and applies to the RES4LYF samplers only; '\(sampler.rawValue)' is not one of "
          + "them. Send bongmath false, or a sampler from " + res4lyfList)
    }
  }

  /// WP-E3 (§3.3, D11, D22, D25): the recipe fields a Krea 2 request carries,
  /// resolved. A pure function of the payload, so the forwarding is asserted
  /// without a server or weights (`Krea2RecipeForwardingTests`).
  ///
  /// The defaults ARE today's render: euler over the family's native `krea2`
  /// warp, no explicit shift, no SDE. An unknown name throws (it does not
  /// become euler) because `validateRecipeNames()` throws.
  func krea2RecipeFields() throws -> Krea2RecipeFields {
    let names = try validateRecipeNames()
    return Krea2RecipeFields(
      sampler: names.scheduler ?? .euler,
      sigmaSchedule: names.sigmaSchedule ?? .krea2,
      shift: shift,
      eta: eta ?? 0,
      bongmath: bongmath ?? false,
      samplerRequested: names.schedulerRequested,
      sigmaScheduleRequested: names.sigmaScheduleRequested)
  }

  /// WP-E17 (§3.14, D4, D22, D25): the second stage, resolved into what the
  /// pipeline runs — or `nil` when the request has no second stage.
  ///
  /// Pure, like `krea2RecipeFields()`, so the forwarding is asserted without a
  /// server or weights. An unknown sampler / schedule name THROWS here (it does
  /// not become euler), and an unstated field stays `nil` so
  /// `Krea2Pipeline.Stage2.resolved(against:)` fills it from the render's own
  /// recipe rather than from an engine default.
  func krea2Stage2Fields() throws -> Krea2Pipeline.Stage2? {
    guard let stage2 else { return nil }
    let names = try RecipeNameResolver.resolve(
      scheduler: stage2.scheduler, sigmaSchedule: stage2.sigmaSchedule)
    return Krea2Pipeline.Stage2(
      steps: stage2.steps,
      denoise: stage2.denoise,
      sampler: names.scheduler,
      sigmaSchedule: names.sigmaSchedule,
      sigmaScheduleRequested: names.sigmaScheduleRequested,
      guidance: stage2.guidance,
      eta: stage2.eta,
      bongmath: stage2.bongmath,
      seed: stage2.seed)
  }

  /// WP-E17 (§3.14, D18; Addendum A.2 → C3): the `stage2` family + range gate,
  /// and the refusal of the tool schema's `detail_pass` / `detail_denoise`
  /// spelling. Returns the 400 to throw, or `nil`.
  ///
  /// Runs at the same dispatch point as `vaeGate` and `validateFamilyRecipe`
  /// (D18: family gates live at dispatch, not at the decoder), for EVERY
  /// family — the detail-pass keys are wrong everywhere, and `stage2` is a
  /// Krea 2 field that no other family's loop could honour.
  static func stage2Gate(_ payload: GeneratePayload, family: WarmModelFamily) -> WarmServerError? {
    // The tool-schema keys first: they are wrong on every family, and a request
    // carrying both them and `stage2` should be told about the spelling rather
    // than about the family.
    if payload.detailPass != nil {
      return .unsupportedRecipeField(
        field: "detail_pass", value: "\(payload.detailPass ?? false)", family: family.rawValue,
        reason: "`detail_pass` is the MCP tool schema's spelling; the client expands it into the "
          + "engine's `stage2` object from its family policy table (AC-68a). The engine holds no "
          + "such table and will not invent a sampler, schedule or step count — send "
          + "`stage2: {steps, denoise, scheduler, sigma_schedule, …}`")
    }
    if let detailDenoise = payload.detailDenoise {
      return .orphanField(
        field: "detail_denoise", requires: "detail_pass",
        reason: "`detail_denoise` = \(detailDenoise) names the denoise of a detail pass that was "
          + "never requested. It used to be dropped silently; send `stage2.denoise` instead")
    }

    guard let stage2 = payload.stage2 else { return nil }

    guard family == .krea2 else {
      return .unsupportedRecipeField(
        field: "stage2", value: "{steps: \(stage2.steps), denoise: \(stage2.denoise)}",
        family: family.rawValue,
        reason: "a second stage inside one render is a Krea 2 mechanism (WP-E17): it re-noises the "
          + "LATENT to the stretched tail's first sigma and solves again with no VAE round-trip. "
          + "This family's loop has no such seam and would have rendered one stage under a "
          + "two-stage record — load a krea2 model, or remove `stage2`")
    }

    guard stage2.steps > 0 else {
      return .unsupportedRecipeField(
        field: "stage2.steps", value: "\(stage2.steps)", family: family.rawValue,
        reason: "stage2.steps must be positive")
    }
    guard stage2.denoise.isFinite, stage2.denoise > 0, stage2.denoise <= 1 else {
      return .unsupportedRecipeField(
        field: "stage2.denoise", value: "\(stage2.denoise)", family: family.rawValue,
        reason: "stage2.denoise is the fraction of the schedule the stage runs and must be in "
          + "(0, 1]; `denoise <= 0` has no schedule to stretch and there is nothing to "
          + "substitute (§3.14)")
    }

    // The stage's own sampler / schedule against the family's capability
    // matrix — the same gate the render's own recipe goes through. Unknown
    // NAMES already threw at `krea2Stage2Fields()`; this is whether the family
    // can honour a known one.
    guard let names = try? RecipeNameResolver.resolve(
      scheduler: stage2.scheduler, sigmaSchedule: stage2.sigmaSchedule),
      let renderNames = try? RecipeNameResolver.resolve(
        scheduler: payload.scheduler, sigmaSchedule: payload.sigmaSchedule)
    else { return nil }
    if let error = FamilyRecipeMatrix.validate(names, family: family) { return error }

    // The stage's TIER gates, evaluated on the values the stage will actually
    // run with — an unstated field inherits the render's, so the pairing that
    // matters is the resolved one (a stage that names `euler` inherits the
    // render's `eta: 0.5` and would be an SDE on a sampler RES4LYF's SDE is not
    // defined against). `Krea2StagedRender.preflight` refuses these too, before
    // any model work; refusing them HERE is what makes them a 400 rather than
    // the 500 an unmapped pipeline error would become.
    if stage2.bongmath ?? false {
      return .unsupportedRecipeField(
        field: "stage2.bongmath", value: "true", family: family.rawValue,
        reason: "bongmath is parity tier T3 (WP-E16) and is not implemented yet; omit it or send false")
    }
    let effectiveEta = stage2.eta ?? payload.eta ?? 0
    let effectiveSampler = names.scheduler ?? renderNames.scheduler ?? .euler
    if effectiveEta != 0, !effectiveSampler.isRES4LYFFamily {
      return .unsupportedRecipeField(
        field: "stage2.eta", value: "\(effectiveEta)", family: family.rawValue,
        reason: "eta is RES4LYF's SDE (parity tier T2) and applies to the RES4LYF samplers only; "
          + "stage 2 runs '\(effectiveSampler.rawValue)', which is not one of them. Send "
          + "stage2.eta 0, or a stage2 sampler from res_2s / res_3s / ralston_2s / ralston_3s / "
          + "ralston_4s / deis_2m / deis_3m / deis_4m")
    }
    return nil
  }

  /// What `krea2RecipeFields()` resolved: the kinds the pipeline runs, plus
  /// the raw names the caller sent so an alias is visible in the record and
  /// the log rather than silently applied (D22).
  struct Krea2RecipeFields: Sendable, Equatable {
    let sampler: SchedulerKind
    let sigmaSchedule: SigmaScheduleKind
    /// `nil` = the resolution-dependent mu (D3/A.1); a value IS mu.
    let shift: Float?
    /// RES4LYF SDE eta (T2, WP-E15). Forwarded whatever the sampler;
    /// `validateKrea2TierGates` is what decides whether the render may have
    /// it, and it decides by SAMPLER.
    let eta: Float
    /// RES4LYF bongmath (T3, WP-E16). Forwarded on the same terms as `eta`,
    /// and gated the same way — by sampler, in `validateKrea2TierGates`.
    let bongmath: Bool
    let samplerRequested: String?
    let sigmaScheduleRequested: String?
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

/// Internal (not private) so the async-job seam is unit-testable
/// (`AsyncJobIdTests`); still `Encodable` only — nothing decodes into it (AC-64).
struct GenerateResponse: Encodable, Sendable {
  let success: Bool
  let outputPath: String
  let durationMs: Int
  /// #1479: set on the SYNC `/v1/generate` path when this job asked to
  /// preempt an in-flight video but the refusal guard declined (finishing
  /// beats preempting) — the job still ran normally, just not preempting.
  /// Additive: defaulted in this explicit init (a property-level default
  /// would DROP the parameter from the synthesized memberwise init entirely,
  /// making it impossible to ever construct a `true`/non-nil one), so every
  /// pre-#1479 construction site is unaffected.
  let preemptRefused: Bool
  let etaSec: Double?
  /// WP-E10 sink 1 (FDD §3.10, D8): the provenance record — what APPLIED,
  /// read back from the pipeline. Krea 2 only (D12).
  ///
  /// Tri-state via ``AppliedRecordSlot`` (round 2, C4): key ABSENT for another
  /// family, literal `null` for a Krea 2 render whose record was refused
  /// (engine-incomplete), the object otherwise. Defaulted in this explicit init
  /// for the same reason as `preemptRefused`: a property-level default would
  /// drop it from the synthesized memberwise init.
  let applied: AppliedRecordSlot?

  /// The record itself, for Swift readers that do not care about the
  /// absent-vs-null distinction.
  var appliedRecord: RenderRecipe? { applied?.record }

  init(success: Bool, outputPath: String, durationMs: Int, preemptRefused: Bool = false, etaSec: Double? = nil,
       applied: AppliedRecordSlot? = nil) {
    self.success = success
    self.outputPath = outputPath
    self.durationMs = durationMs
    self.preemptRefused = preemptRefused
    self.etaSec = etaSec
    self.applied = applied
  }
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

struct LoRAEntry: Codable, Sendable {
  let path: String
  let scale: Float?
  /// WP-E10 (FDD §3.10 `Applied.role`): the configuration SLOT this adapter
  /// fills — `kroma` | `accel` | `bypass` | `control`. Declared by the sender
  /// that expanded the preset (the engine never expands image presets, so it
  /// cannot infer the slot from a flat list); stored on the
  /// `LoRAConfiguration` the pipeline applies and READ BACK from there into
  /// `applied.loras[].role`. An unknown label is a 400, never stored.
  let role: String?

  /// The declared roles, in one place.
  static let roles: [String] = ["kroma", "accel", "bypass", "control"]

  init(path: String, scale: Float?, role: String? = nil) {
    self.path = path
    self.scale = scale
    self.role = role
  }

  private enum CodingKeys: String, CodingKey { case path, scale, role }

  /// Allowed range for LoRA scales — finite values outside are clamped.
  private static let scaleRange: ClosedRange<Float> = -10.0...10.0

  /// The declared role, validated. nil when absent.
  private func resolvedRole() throws -> String? {
    guard let role else { return nil }
    guard Self.roles.contains(role) else {
      throw WarmServerError.invalidRequest(
        message: "Invalid LoRA role '\(role)' for '\(path)': expected one of \(Self.roles.joined(separator: ", "))")
    }
    return role
  }

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

  /// Directories searched, in order, when a LoRA is named by bare filename
  /// (no path — typically reconstructed from embedded PNG metadata, which
  /// only ever stores a display name, never the original absolute path).
  /// COMFYBOX_MODELS matches the same env var LoRALibrary itself resolves
  /// against — this used to hardcode a stale, unrelated "~/Models/loras"
  /// path that nothing actually writes to, so bare-filename resolution
  /// against the real library silently never worked.
  private static var bareFilenameSearchRoots: [String] {
    var roots: [String] = []
    if let envRoot = ProcessInfo.processInfo.environment["COMFYBOX_MODELS"], !envRoot.isEmpty {
      roots.append((envRoot as NSString).expandingTildeInPath)
    }
    roots.append(("~/.comfybox/loras" as NSString).expandingTildeInPath)
    roots.append("/Volumes/Bolt/Models/loras")
    // Ad-hoc/test LoRAs commonly land in Downloads before being filed into
    // the library proper — worth checking before giving up.
    roots.append(("~/Downloads" as NSString).expandingTildeInPath)
    var seen = Set<String>()
    return roots.filter { seen.insert($0).inserted }
  }

  func makeConfiguration() throws -> LoRAConfiguration {
    var configuration = try resolveSource()
    configuration.role = try resolvedRole()
    return configuration
  }

  private func resolveSource() throws -> LoRAConfiguration {
    let clampedScale = try resolvedScale()
    let expanded = (path as NSString).expandingTildeInPath

    // Direct path (absolute, relative, tilde-expanded)
    if path.hasPrefix("/") || path.hasPrefix("./") || path.hasPrefix("../") || path.hasPrefix("~")
       || FileManager.default.fileExists(atPath: expanded) {
      return .local(expanded, scale: clampedScale)
    }

    // Library resolution: search known local LoRA locations for the bare
    // filename before assuming it's a remote reference.
    let fm = FileManager.default
    for root in Self.bareFilenameSearchRoots {
      guard fm.fileExists(atPath: root) else { continue }
      guard let enumerator = fm.enumerator(
        at: URL(fileURLWithPath: root), includingPropertiesForKeys: [.isRegularFileKey]
      ) else { continue }
      for case let fileURL as URL in enumerator {
        if fileURL.lastPathComponent == path {
          return .local(fileURL.path, scale: clampedScale)
        }
      }
    }

    // A string shaped like a local filename (ends in a known weight
    // extension, no "/") is never a valid HuggingFace repo id — don't
    // attempt a network download that's certain to fail with a confusing
    // "Model not found" error. This is almost always a stale reference
    // (e.g. reconstructed from a PNG's embedded metadata, which only ever
    // stores a display name, not the original path) — say so plainly.
    let looksLikeLocalFilename = !path.contains("/") &&
      [".safetensors", ".ckpt", ".pt", ".bin"].contains { path.hasSuffix($0) }
    if looksLikeLocalFilename {
      throw WarmServerError.invalidRequest(
        message: "LoRA '\(path)' not found. Searched: \(Self.bareFilenameSearchRoots.joined(separator: ", "))."
      )
    }

    // HuggingFace fallback — only for strings actually shaped like a repo id.
    return .huggingFace(path, scale: clampedScale)
  }
}

private struct ShutdownResponse: Encodable, Sendable {
  let success: Bool
  let message: String
}

/// Internal (not private) so the sink shape is unit-testable (`HealthSinkTests`).
struct HealthResponse: Encodable, Sendable {
  let status: String
  let model: String
  let modelFamily: String
  let modelVariant: String?
  /// WP-E10 "E9b" (AC-34b): `model_alias` — the declared alias beside the
  /// resolved `model` path for the krea2 family; null otherwise.
  let modelAlias: String?
  /// WP-E10 (FDD §7.3 smoke e): git short sha stamped at build time, or
  /// `"unknown"` for a build that did not run `scripts/gen-build-info.sh`.
  let buildSha: String
  let textEncoderPath: String?
  let loaded: Bool
  let loras: [LoRAState]
  let uptimeSeconds: Int
  let renderCount: Int
  let failedRenderCount: Int
  let pendingCount: Int
  let maxPending: Int
  let isRendering: Bool
  /// Queue pause gate (`is_paused` on the wire) — surfaced in /health so every
  /// client (desktop toolbar, daemons, MCP) sees the same creation state
  /// without an extra request.
  let isPaused: Bool
  let activeRequestAgeMs: Int?
  /// Synthetic id of the currently-rendering job — `current_job_id` on the wire.
  let currentJobId: String?
  /// Live progress (0-100) of the active render — `progress_percent` on the wire.
  let progressPercent: Int?
  let memoryUsageBytes: UInt64
  let memoryUsageMB: UInt64
  let lastRenderDurationMs: Int?
  let lastError: String?
  /// WP-E10 sink 3: `last_recipe` — the record of the last successful Krea 2
  /// render, identical to that render's `applied` (AC-62). Tri-state via
  /// ``AppliedRecordSlot`` (round 2, C4): ABSENT until a Krea 2 render has run
  /// (and for other families, D12), literal `null` when the last Krea 2
  /// render's record was refused, the object otherwise.
  let lastRecipe: AppliedRecordSlot?
}

struct LoRAState: Encodable, Sendable {
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
  /// K-FIX-1 / Codex C2: a MUTATING pool operation (`/v1/model/load`,
  /// `/v1/model/activate`, `/v1/model/unload`) run on the SAME FIFO as
  /// renders, LoRA swaps and the ComfyBridge model switch — so a load,
  /// eviction or `GPU.clearCache()` can never begin under an active render.
  /// The continuation is nil for a `wait: false` load, which is tracked by
  /// its queue job id instead of by a waiting caller (never a detached Task).
  case modelOperation(ModelOperation, ContinuationBox<ModelOperationResult>?)
  /// Local LTX-2 video generation, run through the queue so it serializes with
  /// image renders on the shared GPU. The closure captures the generator+request.
  /// #1479: the closure returns `LTX2RenderOutcome` (not the bare result) so
  /// the process loop can observe a `.yielded` checkpoint and run a
  /// preemption episode before finally resolving `continuation` with the
  /// completed `LTX2VideoResult`. `videoJobId` is the async tracker's id
  /// (nil for the synchronous `/v1/video/generate` path), used to mark the
  /// job paused-for-preemption / resumed in `VideoJobTracker`.
  case localVideo(@Sendable (@escaping @Sendable (Int) -> Void) throws -> LTX2RenderOutcome, ContinuationBox<LTX2VideoResult>, wantsAudio: Bool, videoJobId: String?)
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
  case krea2NotLoaded
  /// WP-E19: the krea2 family is resident but the coordinator holds no
  /// `Krea2Variant` for it. The bridge arm refuses rather than assuming turbo.
  case krea2VariantUnknown
  case loraSwapNotSupported
  case controlNetNotSupported
  // WP-E4 (FDD-krea2-raw-recipe §3.4, D22, D25, D18): fail-loud recipe names.
  /// A sampler name that is neither a `SchedulerKind` raw value nor a declared
  /// alias. `valid` is the full accepted set, listed in the message.
  case unknownSampler(name: String, valid: [String])
  /// A sigma-schedule name that is neither a `SigmaScheduleKind` raw value nor
  /// a declared alias (`normal`/`simple`/`sgm_uniform`/`ddim_uniform`/`beta57`).
  case unknownSigmaSchedule(name: String, valid: [String])
  /// Two request keys that name the same thing carry different values
  /// (`scheduler` vs its `sampler` alias, D25).
  case mutuallyExclusive(String)
  /// A recipe field the named family cannot honour yet — an unimplemented
  /// tier is a 400, never a downgrade (D18).
  case unsupportedRecipeField(field: String, value: String, family: String, reason: String)
  /// WP-E13: an N-row tableau sampler (`ralston_2s/3s/4s`, `res_3s`) asked for
  /// on a family whose denoise loop takes one model evaluation per step. It
  /// would render first-order Euler under the sampler's name, so it is a 400.
  case unsupportedSampler(name: String, family: String, reason: String)
  /// WP-E17 / Addendum A.2 → C3: a field that only means something beside
  /// another field, sent without it. Silently dropping it made a request that
  /// asked for something render as if it had not.
  case orphanField(field: String, requires: String, reason: String)

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
    case .krea2NotLoaded:
      return "Krea-2 pipeline is not loaded"
    case .krea2VariantUnknown:
      return "Krea-2 pipeline is resident but its variant (turbo|raw) is unknown — refusing to assume turbo"
    case .loraSwapNotSupported:
      return "LoRA swap is not supported for this model family"
    case .controlNetNotSupported:
      return "ControlNet is not supported for this model family"
    case .unknownSampler(let name, let valid):
      return "Unknown sampler '\(name)'. Valid samplers: \(valid.joined(separator: ", "))"
    case .unknownSigmaSchedule(let name, let valid):
      return "Unknown sigma schedule '\(name)'. Valid schedules: \(valid.joined(separator: ", "))"
    case .mutuallyExclusive(let message):
      return message
    case .unsupportedRecipeField(let field, let value, let family, let reason):
      return "'\(field)' = '\(value)' is not supported on the \(family) family: \(reason)"
    case .unsupportedSampler(let name, let family, let reason):
      return "sampler '\(name)' is not supported on the \(family) family: \(reason)"
    case .orphanField(let field, let requires, let reason):
      return "'\(field)' has no meaning without '\(requires)': \(reason)"
    }
  }
}

#if DEBUG
/// K-FIX-1 / Codex C2 test seam — drives the coordinator's FIFO directly.
///
/// `WarmServerCoordinator` and the queue types are file-private (deliberately:
/// nothing outside this file should hold the queue), so the barrier test that
/// proves "no pool load or eviction begins until an active render exits" gets
/// this narrow probe instead of a widened actor. It exposes exactly three
/// things: a way to occupy the queue with a controllable job, the two model
/// operation enqueue seams the REST routes now use, and the pending count.
///
/// Construct it with `COMFYBOX_STATE_DIR` pointed at a temp directory — the
/// coordinator persists its queue snapshot and reads its pause sentinel from
/// that directory, and the LIVE engine's are not the test's to touch.
final class WarmServerQueueProbe: @unchecked Sendable {
  private let coordinator: WarmServerCoordinator
  private let liveHealth = LiveHealthState()
  private var liveHealthSnapshot: HealthSnapshot { liveHealth.read().0 }

  init(maxPendingRequests: Int = 10, maxPendingModelOps: Int = 8) {
    self.coordinator = WarmServerCoordinator(
      configuration: WarmServerConfiguration(
        maxPendingRequests: maxPendingRequests, maxPendingModelOps: maxPendingModelOps),
      logger: Logger(label: "z-image.queue-probe"),
      videoHolder: VideoGeneratorHolder(),
      liveHealth: liveHealth,
      videoJobTracker: VideoJobTracker(),
      ltx2Telemetry: LTX2PhaseTelemetry(),
      ltx2PreemptionSignal: PreemptionSignal(),
      ltx2StepPosition: LTX2StepPosition(),
      ltx2EvictMean: RollingMeanSec(),
      ltx2ReloadMean: RollingMeanSec(),
      preemptionInFlight: LockedFlag(),
      pendingPreemptorBox: PendingPreemptorBox())
  }

  /// Occupy the queue with an arbitrary body — the stand-in for a render.
  /// `.modelSwitch` is used because it is the one existing queue kind whose
  /// work is a caller-supplied closure; the loop treats it exactly like any
  /// other job, one at a time.
  func enqueueFakeRender(_ body: @escaping @Sendable () async throws -> Bool) async throws -> Bool {
    try await coordinator.enqueueModelSwitch(body)
  }

  /// The seam `/v1/model/load` (wait: true), `/v1/model/activate` and
  /// `/v1/model/unload` use.
  func enqueueModelOperation(_ op: ModelOperation) async throws -> ModelOperationResult {
    try await coordinator.enqueueModelOperation(op)
  }

  /// The seam `/v1/model/load` (wait: false) uses — returns the queue job id.
  func enqueueModelOperationDetached(_ op: ModelOperation) async throws -> String {
    try await coordinator.enqueueModelOperationDetached(op)
  }

  /// `ServerError` is nested in the file-private coordinator, so a test
  /// cannot pattern-match it. This is the one predicate the WP-E8 hygiene
  /// tests need: was this refusal the capacity gate?
  static func isQueueFull(_ error: Error) -> Bool {
    if case WarmServerCoordinator.ServerError.queueFull = error { return true }
    return false
  }

  /// The model-operation cap, distinct from the render queue's.
  static func isModelOperationQueueFull(_ error: Error) -> Bool {
    if case WarmServerCoordinator.ServerError.modelOperationQueueFull = error { return true }
    return false
  }

  /// The seam `/v1/shutdown` uses. Returns the response's `success` flag —
  /// the response type is file-private, and the only thing a test needs to
  /// know is that the call RETURNED rather than parking forever under a
  /// pause (WP-E8 hygiene).
  func enqueueShutdown() async throws -> Bool {
    try await coordinator.enqueueShutdown().success
  }

  /// The kinds of the jobs still waiting, read through the same
  /// lock-based snapshot `/health` and `/v1/queue` publish.
  func pendingJobKinds() -> [String] {
    liveHealthSnapshot.pending.map { $0.kind }
  }

  /// The pause sentinel the coordinator would read, exposed so a test can
  /// assert it is redirected away from the LIVE `~/.comfybox/queue-paused`
  /// before any coordinator is constructed.
  static var pauseSentinelPath: String { WarmServerCoordinator.pauseSentinelPath }

  /// Pause / resume the queue, exactly as `/v1/queue/pause` does.
  func setPaused(_ paused: Bool) async {
    await coordinator.setPaused(paused)
  }

  /// The summary of the job the loop is running, or nil when idle.
  var activeJobSummary: String? { liveHealthSnapshot.activeSummary }

  /// Nothing running and nothing waiting.
  ///
  /// The drain guard reads this: a test must not tear down its isolated state
  /// directory while the coordinator still has work, because the loop's
  /// per-job `persistQueueState()` would then resolve — and `removeItem` —
  /// the LIVE snapshot path (K-FIX-1 round 2, New-2).
  var isDrained: Bool {
    let snapshot = liveHealthSnapshot
    return snapshot.activeJobId == nil && snapshot.pending.isEmpty
  }

  var pendingCount: Int { liveHealthSnapshot.pending.count }

  var isPaused: Bool { liveHealthSnapshot.isPaused }
}
#endif
