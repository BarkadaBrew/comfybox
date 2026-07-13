// ModelPool.swift — Multi-model pool with LRU eviction for WarmServer
//
// Manages loaded generation pipelines with a configurable VRAM budget.
// When the budget is exceeded, the least-recently-used non-active model
// is evicted to make room. Thread-safe via Swift actor isolation.
//
// Issue: #88

import Foundation
import Logging
import MLX

// MARK: - Pool Types

/// Status of a single model in the pool.
public struct ModelPoolStatus: Encodable, Sendable {
  public let model: String
  public let family: String
  public let vramMB: Int
  public let active: Bool
  public let lastUsed: String

  enum CodingKeys: String, CodingKey {
    case model, family, vramMB = "vram_mb", active, lastUsed = "last_used"
  }
}

/// Response for pool listing endpoint.
struct ModelPoolListResponse: Encodable, Sendable {
  let active: String?
  let pool: [ModelPoolStatus]
  let totalVramMB: Int
  let budgetMB: Int

  enum CodingKeys: String, CodingKey {
    case active, pool, totalVramMB = "total_vram_mb", budgetMB = "budget_mb"
  }
}

/// Response for model load endpoint.
struct ModelLoadResponse: Encodable, Sendable {
  let status: String
  let model: String
  let family: String
  let loadTimeMs: Int
  let vramEstimateMB: Int
  let poolSize: Int
  let poolBudgetMB: Int

  enum CodingKeys: String, CodingKey {
    case status, model, family
    case loadTimeMs = "load_time_ms"
    case vramEstimateMB = "vram_estimate_mb"
    case poolSize = "pool_size"
    case poolBudgetMB = "pool_budget_mb"
  }
}

/// Response for model activate endpoint.
struct ModelActivateResponse: Encodable, Sendable {
  let status: String
  let model: String
  let family: String
}

/// Response for model unload endpoint.
struct ModelUnloadResponse: Encodable, Sendable {
  let status: String
  let model: String
  let freedMB: Int
  let poolSize: Int

  enum CodingKeys: String, CodingKey {
    case status, model, freedMB = "freed_mb", poolSize = "pool_size"
  }
}

/// Request payload for model load.
struct ModelLoadRequest: Decodable, Sendable {
  let model: String
  let activate: Bool?
  let quantization: String?
  let wait: Bool?
}

/// Request payload for model activate.
struct ModelActivateRequest: Decodable, Sendable {
  let model: String
}

/// Request payload for model unload.
struct ModelUnloadRequest: Decodable, Sendable {
  let model: String
}

// MARK: - Pool Errors

enum ModelPoolError: Error, LocalizedError {
  case modelNotInPool(String)
  case cannotUnloadActive(String)
  case budgetExceeded(needed: Int, available: Int)
  case loadFailed(String, Error)
  case alreadyLoaded(String)
  case modelDetectionFailed(String)

  var errorDescription: String? {
    switch self {
    case .modelNotInPool(let id):
      return "Model '\(id)' is not loaded in the pool"
    case .cannotUnloadActive(let id):
      return "Cannot unload active model '\(id)' — switch to another model first"
    case .budgetExceeded(let needed, let available):
      return "VRAM budget exceeded: need \(needed) MB, only \(available) MB available after eviction"
    case .loadFailed(let id, let error):
      return "Failed to load model '\(id)': \(error.localizedDescription)"
    case .alreadyLoaded(let id):
      return "Model '\(id)' is already loaded in the pool"
    case .modelDetectionFailed(let id):
      return "Could not detect model family for '\(id)'"
    }
  }
}

// MARK: - VRAM Estimates

/// Approximate VRAM usage per model (in MB).
/// These are hardcoded estimates based on observed memory usage.
private enum VRAMEstimates {
  static func estimate(for modelSpec: String, family: WarmModelFamily, quantization: String?) -> Int {
    let normalized = modelSpec.lowercased()
    let quant = quantization?.lowercased()

    switch family {
    case .flux1:
      // Z-Image family
      if quant == "q4" { return 4096 }
      if quant == "q8" { return 7168 }
      if normalized.contains("base") {
        return 12288  // Base BF16
      }
      return 12288    // Turbo BF16
    case .flux2:
      // Klein family
      if normalized.contains("9b") { return 18432 }
      if normalized.contains("4b") { return 8704 }
      return 10240    // Default Klein
    case .fibo:
      if quant == "4bit" || quant == "q4" { return 8192 }
      return 22528    // BF16
    case .krea2:
      // 8-bit transformer (~13.5GB) + bf16 Qwen3-VL-4B encoder (~8GB) + VAE
      return 22528
    case .chroma:
      return 17408    // Always BF16
    }
  }
}

// MARK: - Pipeline Box

/// Type-erased container for a loaded pipeline.
/// Pipelines are not Sendable, so we use nonisolated(unsafe) within the actor.
final class PipelineBox {
  nonisolated(unsafe) var pipeline: AnyObject?

  /// Additional context needed for generation (e.g., tokenizers, detected model info).
  nonisolated(unsafe) var context: [String: AnyObject] = [:]

  init(pipeline: AnyObject) {
    self.pipeline = pipeline
  }

  /// Release the pipeline and its context to free GPU memory.
  func release() {
    pipeline = nil
    context.removeAll()
    GPU.clearCache()
  }
}

// MARK: - Pool Entry

/// A single model loaded in the pool.
struct PoolEntry {
  let id: String
  let modelSpec: String
  let family: WarmModelFamily
  let quantization: String?
  let box: PipelineBox
  var vramEstimateMB: Int
  var lastUsed: Date
  /// Detected model info, stored per-family for generation routing.
  /// For flux2: Flux2DetectedModel. For fibo: FiboDetectedModel.
  /// For flux1: ZImageVariant. For chroma: ChromaConfig.
  var detectedInfo: Any?

  func toStatus(isActive: Bool) -> ModelPoolStatus {
    let formatter = ISO8601DateFormatter()
    return ModelPoolStatus(
      model: modelSpec,
      family: family.rawValue,
      vramMB: vramEstimateMB,
      active: isActive,
      lastUsed: formatter.string(from: lastUsed)
    )
  }
}

// MARK: - ModelPool Actor

/// Multi-model pool with LRU eviction.
///
/// Manages loaded generation pipelines. Each model is identified by its
/// model spec string (HuggingFace ID or local path). The pool tracks VRAM
/// usage estimates and evicts least-recently-used models when the budget
/// is exceeded.
actor ModelPool {
  private var pool: [String: PoolEntry] = [:]
  private var activeId: String?
  private let budgetMB: Int
  private let logger: Logger

  /// In-flight load tasks keyed by pool key. `load` suspends across the
  /// pipeline load, so the actor is reentrant — without this map, concurrent
  /// requests for the same model would each pass the duplicate check and
  /// load the multi-GB pipeline twice, overshooting the VRAM budget.
  private var inFlightLoads: [String: Task<Void, Error>] = [:]

  /// Configuration snapshot from WarmServerConfiguration, needed for pipeline loading.
  private let textEncoderPath: String?
  private let maxSequenceLength: Int
  private let forceTransformerOverrideOnly: Bool

  init(
    budgetMB: Int? = nil,
    textEncoderPath: String?,
    maxSequenceLength: Int,
    forceTransformerOverrideOnly: Bool,
    logger: Logger
  ) {
    // Default 40 GB budget, overridden by env var.
    if let envBudget = ProcessInfo.processInfo.environment["COMFYBOX_POOL_BUDGET_MB"],
       let parsed = Int(envBudget) {
      self.budgetMB = parsed
    } else {
      self.budgetMB = budgetMB ?? 40960
    }
    self.textEncoderPath = textEncoderPath
    self.maxSequenceLength = maxSequenceLength
    self.forceTransformerOverrideOnly = forceTransformerOverrideOnly
    self.logger = logger
  }

  // MARK: - Public API

  /// Load a model into the pool. If already loaded, returns the existing entry.
  ///
  /// - Parameters:
  ///   - modelSpec: HuggingFace model ID or local path.
  ///   - quantization: Optional quantization override (q4, q8).
  ///   - initialLoRAs: LoRA configurations to apply during loading (flux1 only).
  /// - Returns: The pool entry for the loaded model.
  func load(
    modelSpec: String,
    quantization: String? = nil,
    initialLoRAs: [LoRAConfiguration] = []
  ) async throws -> PoolEntry {
    let poolKey = Self.poolKey(for: modelSpec, quantization: quantization)

    // Already loaded — just update lastUsed.
    if var existing = pool[poolKey] {
      existing.lastUsed = Date()
      pool[poolKey] = existing
      logger.info("ModelPool: '\(poolKey)' already loaded, updated lastUsed")
      return existing
    }

    // A load for this key is already in flight (actor reentrancy across the
    // awaits in performLoad) — await it instead of duplicating the load.
    if let inFlight = inFlightLoads[poolKey] {
      logger.info("ModelPool: load of '\(poolKey)' already in flight, awaiting existing task")
      try await inFlight.value
    } else {
      let task = Task {
        try await self.performLoad(
          poolKey: poolKey,
          modelSpec: modelSpec,
          quantization: quantization,
          initialLoRAs: initialLoRAs
        )
      }
      inFlightLoads[poolKey] = task
      defer { inFlightLoads[poolKey] = nil }
      try await task.value
    }

    guard var entry = pool[poolKey] else {
      // Entry vanished between load completion and this resume (e.g. evicted
      // or unloaded by a concurrent operation) — report as not in pool.
      throw ModelPoolError.modelNotInPool(poolKey)
    }
    entry.lastUsed = Date()
    pool[poolKey] = entry
    return entry
  }

  /// Detect the family, evict if needed, load the pipeline, and insert the
  /// entry into the pool. Factored out of `load` so concurrent callers can
  /// await a single in-flight task instead of loading twice.
  private func performLoad(
    poolKey: String,
    modelSpec: String,
    quantization: String?,
    initialLoRAs: [LoRAConfiguration]
  ) async throws {
    // Detect model family.
    let family = try await detectFamily(modelSpec: modelSpec)
    let vramEstimate = VRAMEstimates.estimate(for: modelSpec, family: family, quantization: quantization)

    // Evict if needed to stay within budget.
    try evictIfNeeded(neededMB: vramEstimate)

    // Load the pipeline.
    logger.info("ModelPool: loading '\(poolKey)' (family=\(family.rawValue), vram=\(vramEstimate)MB)...")
    let start = Date()
    let (box, detectedInfo) = try await loadPipeline(
      modelSpec: modelSpec,
      family: family,
      quantization: quantization,
      initialLoRAs: initialLoRAs
    )
    let loadTimeMs = Int(Date().timeIntervalSince(start) * 1000.0)
    logger.info("ModelPool: '\(poolKey)' loaded in \(loadTimeMs)ms")

    let entry = PoolEntry(
      id: poolKey,
      modelSpec: modelSpec,
      family: family,
      quantization: quantization,
      box: box,
      vramEstimateMB: vramEstimate,
      lastUsed: Date(),
      detectedInfo: detectedInfo
    )
    pool[poolKey] = entry
  }

  /// Activate a model for generation. Must already be loaded.
  func activate(modelId: String) throws -> PoolEntry {
    guard var entry = pool[modelId] else {
      throw ModelPoolError.modelNotInPool(modelId)
    }
    entry.lastUsed = Date()
    pool[modelId] = entry
    activeId = modelId
    logger.info("ModelPool: activated '\(modelId)'")
    return entry
  }

  /// Unload a model from the pool. Cannot unload the active model.
  func unload(modelId: String) throws -> Int {
    guard let entry = pool[modelId] else {
      throw ModelPoolError.modelNotInPool(modelId)
    }
    if activeId == modelId {
      throw ModelPoolError.cannotUnloadActive(modelId)
    }

    let freedMB = entry.vramEstimateMB
    entry.box.release()
    pool.removeValue(forKey: modelId)
    GPU.clearCache()
    logger.info("ModelPool: unloaded '\(modelId)' (~\(freedMB)MB freed)")
    return freedMB
  }

  /// Release EVERY loaded model (including the active one) and clear the pool.
  ///
  /// Used when a heavy model of a *different class* — the LTX-2 video stack —
  /// must take over unified memory. Image and video cannot co-reside on a
  /// 128GB machine (#218); the pool's VRAM budget is blind to LTX-2, so the
  /// only safe handoff is to fully vacate the image side first. Unlike
  /// `unload`, this deliberately releases the active model too. Returns the
  /// estimated MB freed.
  @discardableResult
  func releaseAll() -> Int {
    let freed = totalVramMB()
    for (_, entry) in pool { entry.box.release() }
    pool.removeAll()
    activeId = nil
    GPU.clearCache()
    logger.info("ModelPool: released ALL loaded models (~\(freed)MB) — vacating for heavy-model handoff")
    return freed
  }

  /// Release the least-recently-used *non-active* model, if any. Used by the
  /// memory-pressure guard to shed memory without disturbing an in-flight
  /// render. Returns the MB freed (0 if nothing was evictable).
  @discardableResult
  func releaseLRUInactive() -> Int {
    guard let lru = pool.values
      .filter({ $0.id != activeId })
      .min(by: { $0.lastUsed < $1.lastUsed })
    else { return 0 }
    let freed = lru.vramEstimateMB
    lru.box.release()
    pool.removeValue(forKey: lru.id)
    GPU.clearCache()
    logger.warning("ModelPool: memory-pressure eviction of '\(lru.id)' (~\(freed)MB)")
    return freed
  }

  /// Get the currently active pool entry.
  func activeEntry() -> PoolEntry? {
    guard let id = activeId else { return nil }
    return pool[id]
  }

  /// Get the active model ID.
  func activeModelId() -> String? {
    activeId
  }

  /// Mark the active model as recently used (call on every generation).
  func touchActive() {
    guard let id = activeId, var entry = pool[id] else { return }
    entry.lastUsed = Date()
    pool[id] = entry
  }

  /// Register an already-loaded pipeline in the pool (used for the initial startup model).
  /// This avoids double-loading by accepting a pre-created PipelineBox.
  func registerExisting(
    poolKey: String,
    modelSpec: String,
    family: WarmModelFamily,
    box: PipelineBox,
    vramEstimateMB: Int,
    detectedInfo: Any?
  ) {
    let entry = PoolEntry(
      id: poolKey,
      modelSpec: modelSpec,
      family: family,
      quantization: nil,
      box: box,
      vramEstimateMB: vramEstimateMB,
      lastUsed: Date(),
      detectedInfo: detectedInfo
    )
    pool[poolKey] = entry
    activeId = poolKey
  }

  /// List all models in the pool.
  func listPool() -> [ModelPoolStatus] {
    pool.values.map { $0.toStatus(isActive: $0.id == activeId) }
      .sorted { $0.model < $1.model }
  }

  /// Total estimated VRAM across all loaded models.
  func totalVramMB() -> Int {
    pool.values.reduce(0) { $0 + $1.vramEstimateMB }
  }

  /// Number of models in the pool.
  func count() -> Int {
    pool.count
  }

  /// The pool VRAM budget.
  func budget() -> Int {
    budgetMB
  }

  /// Find a pool entry by model spec (tries exact match, then pool key variants).
  func findEntry(for modelSpec: String, quantization: String? = nil) -> PoolEntry? {
    let key = Self.poolKey(for: modelSpec, quantization: quantization)
    if let entry = pool[key] { return entry }
    // Fallback: search by modelSpec.
    return pool.values.first { $0.modelSpec == modelSpec }
  }

  // MARK: - Internal Helpers

  /// Generate a stable pool key from model spec and quantization.
  static func poolKey(for modelSpec: String, quantization: String? = nil) -> String {
    var key = modelSpec
      .replacingOccurrences(of: "/", with: "-")
      .lowercased()
    if let q = quantization?.lowercased(), !q.isEmpty, q != "bf16", q != "none" {
      key += "-\(q)"
    }
    return key
  }

  /// Detect the model family for a given model spec.
  private func detectFamily(modelSpec: String) async throws -> WarmModelFamily {
    if Krea2ModelDetection.isKnownKrea2Model(modelSpec) {
      return .krea2
    } else if ChromaModelDetection.isKnownChromaModel(modelSpec) {
      return .chroma
    } else if FiboModelDetection.isKnownFiboModel(modelSpec) {
      return .fibo
    } else if Flux2ModelDetection.isKnownFlux2Model(modelSpec) {
      return .flux2
    }

    // Check if modelSpec points to a single safetensors file
    let localURL = URL(fileURLWithPath: modelSpec)
    if localURL.pathExtension == "safetensors" && FileManager.default.fileExists(atPath: localURL.path) {
      // Single-file checkpoint — check if CivitAI or AIO
      let aio = ZImageAIOCheckpoint.inspect(fileURL: localURL)
      if aio.isAIO { return .flux1 }

      let civitai = CivitAICheckpoint.inspect(fileURL: localURL)
      if civitai.isCivitAI { return .flux1 }
      if let variant = civitai.variant {
        logger.info("ModelPool: checkpoint inspection detected Z-Image variant=\(variant.rawValue) for \(localURL.lastPathComponent)")
        return .flux1
      }

      // Could also be a single-file transformer override for flux1
      return .flux1
    }

    // Not detected by name — try resolving and inspecting the snapshot.
    let resolved = try await ModelResolution.resolveOrDefault(
      modelSpec: modelSpec,
      filePatterns: ["*.safetensors", "*.json", "tokenizer/*"]
    )

    if Krea2ModelDetection.detect(at: resolved) != nil {
      return .krea2
    } else if ChromaModelDetection.detect(at: resolved) != nil {
      return .chroma
    } else if FiboModelDetection.detect(at: resolved) != nil {
      return .fibo
    } else if Flux2ModelDetection.detectFamily(at: resolved) == .flux2 {
      return .flux2
    }

    // Default to flux1 (Z-Image).
    return .flux1
  }

  /// Load a pipeline for the given model family.
  /// Returns a PipelineBox and optional detection info.
  private func loadPipeline(
    modelSpec: String,
    family: WarmModelFamily,
    quantization: String?,
    initialLoRAs: [LoRAConfiguration]
  ) async throws -> (PipelineBox, Any?) {
    if family == .krea2 {
      // Krea-2 resolves its own weights (HF cache snapshot or explicit dir) —
      // skip the generic snapshot resolution.
      let paths = try Krea2ModelDetection.resolve(spec: modelSpec)
      let bits: Int? = (quantization?.lowercased() == "bf16") ? nil : 8
      let pipeline = try Krea2Pipeline(paths: paths, quantizeTransformer: bits)
      return (PipelineBox(pipeline: pipeline as AnyObject), nil)
    }
    let resolved = try await ModelResolution.resolveOrDefault(
      modelSpec: modelSpec,
      filePatterns: ["*.safetensors", "*.json", "tokenizer/*"]
    )

    switch family {
    case .krea2:
      // Handled by the early return above — unreachable.
      throw ModelPoolError.modelDetectionFailed(modelSpec)

    case .chroma:
      guard let detected = ChromaModelDetection.detect(at: resolved) else {
        throw ModelPoolError.modelDetectionFailed(modelSpec)
      }
      let components = try ChromaInitializer.load(
        from: resolved,
        paths: detected.componentPaths,
        config: detected.config,
        dtype: .bfloat16,
        logger: logger
      )
      let tokenizer = try ChromaTokenizer.load(from: detected.componentPaths.tokenizerPath)
      let pipeline = ChromaPipeline(
        transformer: components.transformer,
        t5: components.t5,
        vae: components.vae,
        config: detected.config
      )
      let box = PipelineBox(pipeline: pipeline)
      box.context["tokenizer"] = tokenizer
      return (box, detected.config)

    case .fibo:
      guard let detected = FiboModelDetection.detect(at: resolved) else {
        throw ModelPoolError.modelDetectionFailed(modelSpec)
      }
      let pipeline = FiboPipeline(logger: logger)
      try pipeline.loadModel(
        from: resolved,
        transformerConfig: detected.transformerConfig,
        vaeConfig: detected.vaeConfig,
        textEncoderConfig: detected.textEncoderConfig
      )
      let box = PipelineBox(pipeline: pipeline)
      return (box, detected)

    case .flux2:
      guard let detected = Flux2ModelDetection.detect(at: resolved) else {
        throw ModelPoolError.modelDetectionFailed(modelSpec)
      }
      let pipeline = Flux2Pipeline(logger: logger)
      try pipeline.loadModel(
        from: resolved,
        config: detected.transformerConfig,
        textEncoderConfig: detected.textEncoderConfig,
        isBase: detected.isBaseModel
      )
      let box = PipelineBox(pipeline: pipeline)
      return (box, detected)

    case .flux1:
      let pipeline = ZImagePipeline(logger: logger, retentionPolicy: .keepLoaded)
      // Detect variant (base vs turbo).
      // Priority: filename heuristic > checkpoint key inspection > snapshot heuristic.
      let variant: ZImageVariant
      if let v = ZImageVariant.fromModelSpec(modelSpec) {
        variant = v
      } else if modelSpec.hasSuffix(".safetensors"),
                let fileURL = URL(string: modelSpec) ?? (FileManager.default.fileExists(atPath: modelSpec) ? URL(fileURLWithPath: modelSpec) : nil) {
        // For local .safetensors files, use CivitAI inspection which reads
        // actual key signatures instead of the less reliable snapshot heuristic.
        let inspection = CivitAICheckpoint.inspect(fileURL: fileURL)
        if let v = inspection.variant {
          variant = v
          logger.info("ModelPool: checkpoint inspection detected variant=\(v.rawValue) for \(fileURL.lastPathComponent)")
        } else {
          variant = ZImageVariant.fromSnapshot(at: resolved)
        }
      } else {
        variant = ZImageVariant.fromSnapshot(at: resolved)
      }
      try await pipeline.prepare(
        modelSpec: modelSpec,
        textEncoderPath: textEncoderPath,
        loras: initialLoRAs,
        forceTransformerOverrideOnly: forceTransformerOverrideOnly
      )
      let box = PipelineBox(pipeline: pipeline)
      return (box, variant)
    }
  }

  /// Evict least-recently-used non-active models until there is room for `neededMB`.
  private func evictIfNeeded(neededMB: Int) throws {
    var currentTotal = totalVramMB()
    while currentTotal + neededMB > budgetMB {
      // Find LRU non-active entry.
      guard let lruEntry = pool.values
        .filter({ $0.id != activeId })
        .min(by: { $0.lastUsed < $1.lastUsed })
      else {
        // No more evictable entries. Allow single-model exceeding budget.
        if pool.isEmpty {
          logger.warning("ModelPool: budget exceeded but pool is empty — allowing load anyway")
          return
        }
        throw ModelPoolError.budgetExceeded(needed: neededMB, available: budgetMB - currentTotal)
      }

      logger.info("ModelPool: evicting '\(lruEntry.id)' (~\(lruEntry.vramEstimateMB)MB) — LRU eviction")
      lruEntry.box.release()
      pool.removeValue(forKey: lruEntry.id)
      GPU.clearCache()
      currentTotal = totalVramMB()
    }
  }
}
