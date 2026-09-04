// ComfyBoxServerConfig.swift — Unified ComfyBox configuration (~/.comfybox/config.json).
//
// P0 of the image-service consolidation: one config model, one canonical port, one
// AI-provider registry. On first launch (config absent) this non-destructively migrates
// settings from the retiring Coffee Shop image service under ~/.coffeeshop/.

import Foundation

/// Endpoint config for one AI capability, served by an OpenAI-compatible local server
/// (e.g. LM Studio). Adding a capability is a new field on ``AIProviderRegistry`` — no fork.
public struct AIProviderEndpoint: Codable, Equatable, Sendable {
  public var baseUrl: String
  public var model: String
  public var apiKey: String?

  public init(baseUrl: String, model: String, apiKey: String? = nil) {
    self.baseUrl = baseUrl
    self.model = model
    self.apiKey = apiKey
  }
}

/// Named-capability → endpoint registry. Optional fields so unconfigured capabilities are absent.
public struct AIProviderRegistry: Codable, Equatable, Sendable {
  public var promptOptimization: AIProviderEndpoint?
  public var vision: AIProviderEndpoint?
  public var captioning: AIProviderEndpoint?

  public init(
    promptOptimization: AIProviderEndpoint? = nil,
    vision: AIProviderEndpoint? = nil,
    captioning: AIProviderEndpoint? = nil
  ) {
    self.promptOptimization = promptOptimization
    self.vision = vision
    self.captioning = captioning
  }

  /// Default prompt-optimization endpoint: LM Studio serving Todd's Dan's Personality Engine model.
  public static let lmStudioPromptDefault = AIProviderEndpoint(
    baseUrl: "http://localhost:1234/v1",
    model: "dans-pe-v1.3.0-24b-heresy@8bit"
  )

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    promptOptimization = try c.decodeIfPresent(AIProviderEndpoint.self, forKey: .promptOptimization)
    vision = try c.decodeIfPresent(AIProviderEndpoint.self, forKey: .vision)
    captioning = try c.decodeIfPresent(AIProviderEndpoint.self, forKey: .captioning)
  }
}

/// Replicate (remote video / fallback) configuration.
public struct ReplicateProviderConfig: Codable, Equatable, Sendable {
  public var apiKey: String?
  public var baseUrl: String?
  public var model: String?
  public var imageModel: String?
  public var videoModel: String?

  public init(
    apiKey: String? = nil,
    baseUrl: String? = nil,
    model: String? = nil,
    imageModel: String? = nil,
    videoModel: String? = nil
  ) {
    self.apiKey = apiKey
    self.baseUrl = baseUrl
    self.model = model
    self.imageModel = imageModel
    self.videoModel = videoModel
  }
}

/// One family's (or the cross-family default's) render-parameter overrides (FDD §3.3, D3).
/// Every field is optional: an absent field means "no override here" and resolution falls
/// through to the next layer. Width/height are pixels, steps is an integer step count,
/// guidance is the CFG/guidance scale — kept `Double` for JSON portability across the families
/// that model it as `Float` internally (converted at each call site).
public struct RenderDefaultValues: Codable, Equatable, Sendable {
  public var width: Int?
  public var height: Int?
  public var steps: Int?
  public var guidance: Double?

  public init(width: Int? = nil, height: Int? = nil, steps: Int? = nil, guidance: Double? = nil) {
    self.width = width
    self.height = height
    self.steps = steps
    self.guidance = guidance
  }

  public var isEmpty: Bool { width == nil && height == nil && steps == nil && guidance == nil }
}

/// `renderDefaults` document shape: a cross-family `default` plus per-family overrides,
/// because the engine's own fallbacks are family-dependent (FDD §2.5) — a flat block would
/// flatten real behavior. Resolution order (FDD §3.3): `request → preset →
/// config.byFamily[family] → config.default → engine constant`. With every field absent
/// (freshly-initialized or explicitly emptied), ``resolved(family:)`` returns an
/// all-nil ``RenderDefaultValues``, so callers' existing `?? <engine constant>` fallback is
/// unchanged — this is what makes an empty config bit-identical to today's behavior.
public struct RenderDefaultsConfig: Codable, Equatable, Sendable {
  public var `default`: RenderDefaultValues
  public var byFamily: [String: RenderDefaultValues]

  public init(default: RenderDefaultValues = RenderDefaultValues(), byFamily: [String: RenderDefaultValues] = [:]) {
    self.default = `default`
    self.byFamily = byFamily
  }

  private enum CodingKeys: String, CodingKey { case `default`, byFamily }

  /// Tolerant decode: EITHER key absent falls back to empty, not a decode
  /// failure. This is not just forward-compat hygiene — it is load-bearing
  /// for `PATCH /v1/config` (RFC 7386 merge-patch, FDD §3.3): a patch that
  /// introduces `renderDefaults.byFamily.<x>` on a document that never had a
  /// `renderDefaults` block at all produces a merged fragment of exactly
  /// `{"byFamily": {...}}`, with no `default` key — a REQUIRED-key
  /// (synthesized) decode would reject that as malformed, when it is exactly
  /// the shape a from-scratch single-family patch is supposed to produce.
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    self.default = try c.decodeIfPresent(RenderDefaultValues.self, forKey: .default) ?? RenderDefaultValues()
    self.byFamily = try c.decodeIfPresent([String: RenderDefaultValues].self, forKey: .byFamily) ?? [:]
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(self.default, forKey: .default)
    try c.encode(byFamily, forKey: .byFamily)
  }

  public var isEmpty: Bool { self.default.isEmpty && byFamily.isEmpty }

  /// Config-layer resolution for one family: a byFamily field wins over the cross-family
  /// default, per field independently (a family can override just `steps` and still inherit
  /// `width`/`height` from `default`). Callers apply this ABOVE the engine's hardcoded
  /// constant and BELOW whatever the request/preset already supplied.
  public func resolved(family: String) -> RenderDefaultValues {
    let fam = byFamily[family] ?? RenderDefaultValues()
    return RenderDefaultValues(
      width: fam.width ?? self.default.width,
      height: fam.height ?? self.default.height,
      steps: fam.steps ?? self.default.steps,
      guidance: fam.guidance ?? self.default.guidance
    )
  }
}

/// The Motion tab's migrated video defaults (FDD §3.3): only `videoWidth/Height/Frames`
/// genuinely migrate from the desktop's `desktop-config.json` (`MotionView.swift:393–398` is
/// the only real reader) — NOT steps/backend, which either don't exist on `DesktopSettings`
/// or have no client-side reader to preserve.
public struct VideoDefaultValues: Codable, Equatable, Sendable {
  public var width: Int?
  public var height: Int?
  public var frames: Int?

  public init(width: Int? = nil, height: Int? = nil, frames: Int? = nil) {
    self.width = width
    self.height = height
    self.frames = frames
  }

  public var isEmpty: Bool { width == nil && height == nil && frames == nil }
}

/// `videoDefaults` document shape — mirrors ``RenderDefaultsConfig``'s `{default, byFamily}`
/// shape for API/discovery consistency (FDD §4's `ControlDescriptor` treats both uniformly),
/// though today there is exactly one video engine (LTX-2) so `byFamily` is normally unused;
/// the migrated Motion-tab values land in `default`.
public struct VideoDefaultsConfig: Codable, Equatable, Sendable {
  public var `default`: VideoDefaultValues
  public var byFamily: [String: VideoDefaultValues]

  public init(default: VideoDefaultValues = VideoDefaultValues(), byFamily: [String: VideoDefaultValues] = [:]) {
    self.default = `default`
    self.byFamily = byFamily
  }

  private enum CodingKeys: String, CodingKey { case `default`, byFamily }

  /// Tolerant decode — see ``RenderDefaultsConfig/init(from:)``; the same
  /// merge-patch partial-fragment hazard applies here.
  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    self.default = try c.decodeIfPresent(VideoDefaultValues.self, forKey: .default) ?? VideoDefaultValues()
    self.byFamily = try c.decodeIfPresent([String: VideoDefaultValues].self, forKey: .byFamily) ?? [:]
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(self.default, forKey: .default)
    try c.encode(byFamily, forKey: .byFamily)
  }

  public var isEmpty: Bool { self.default.isEmpty && byFamily.isEmpty }

  public func resolved(family: String) -> VideoDefaultValues {
    let fam = byFamily[family] ?? VideoDefaultValues()
    return VideoDefaultValues(
      width: fam.width ?? self.default.width,
      height: fam.height ?? self.default.height,
      frames: fam.frames ?? self.default.frames
    )
  }
}

/// Persistent ComfyBox configuration, stored as a single `~/.comfybox/config.json`.
///
/// Decoding tolerates missing keys (partial/older files load with defaults), so the
/// schema can grow across phases without breaking existing installs.
public struct ComfyBoxServerConfig: Codable, Equatable, Sendable {
  public var port: UInt16
  public var host: String
  public var modelSpec: String?
  public var allowedOutputDirectory: String?
  public var seedvr2WeightsPath: String?
  public var providers: AIProviderRegistry
  public var replicate: ReplicateProviderConfig?
  /// Content mode (rawValue, e.g. "neutral"/"banana"/"avocado") → default
  /// preset id, applied automatically when Generate's content mode changes.
  public var contentModeDefaultPresets: [String: String]
  /// Declared Krea-2 spec → model directory (e.g. `"krea2-raw": "~/LocalModels/krea2-raw"`).
  /// Merged over `Krea2ModelDetection.defaultSpecDirectories` at server start
  /// (WP-E5). An alias that is in neither table fails the load loudly rather
  /// than falling back to the Krea-2-Turbo snapshot.
  public var krea2Models: [String: String]
  /// Server-side engine render defaults (FDD §3.3, D3): family-aware width/height/steps/
  /// guidance overrides, writable via `PATCH`/`PUT /v1/config`. Empty by default — an empty
  /// document resolves identically to the engine's own hardcoded fallbacks.
  public var renderDefaults: RenderDefaultsConfig
  /// Server-side video (Motion tab) defaults migrated from the desktop's local config
  /// (FDD §3.3): only `width`/`height`/`frames`.
  public var videoDefaults: VideoDefaultsConfig

  /// The one true ComfyBox HTTP port.
  public static let canonicalPort: UInt16 = 7870
  /// Legacy port accepted for one release with a deprecation warning (old WarmServer default,
  /// image-service warm-worker target). Retires with the Node service.
  public static let deprecatedAliasPort: UInt16 = 7862

  public init(
    port: UInt16 = ComfyBoxServerConfig.canonicalPort,
    host: String = "127.0.0.1",
    modelSpec: String? = nil,
    allowedOutputDirectory: String? = nil,
    seedvr2WeightsPath: String? = nil,
    providers: AIProviderRegistry = AIProviderRegistry(promptOptimization: AIProviderRegistry.lmStudioPromptDefault),
    replicate: ReplicateProviderConfig? = nil,
    contentModeDefaultPresets: [String: String] = [:],
    krea2Models: [String: String] = [:],
    renderDefaults: RenderDefaultsConfig = RenderDefaultsConfig(),
    videoDefaults: VideoDefaultsConfig = VideoDefaultsConfig()
  ) {
    self.port = port
    self.host = host
    self.modelSpec = modelSpec
    self.allowedOutputDirectory = allowedOutputDirectory
    self.seedvr2WeightsPath = seedvr2WeightsPath
    self.providers = providers
    self.replicate = replicate
    self.contentModeDefaultPresets = contentModeDefaultPresets
    self.krea2Models = krea2Models
    self.renderDefaults = renderDefaults
    self.videoDefaults = videoDefaults
  }

  private enum CodingKeys: String, CodingKey {
    case port, host, modelSpec, allowedOutputDirectory, seedvr2WeightsPath, providers, replicate
    case contentModeDefaultPresets
    case krea2Models
    case renderDefaults
    case videoDefaults
    // Legacy keys written by the desktop AppConfig (read-only, for smooth upgrade).
    case serverPort, serverHost, outputDirectory
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    port = try c.decodeIfPresent(UInt16.self, forKey: .port)
      ?? c.decodeIfPresent(UInt16.self, forKey: .serverPort)
      ?? ComfyBoxServerConfig.canonicalPort
    host = try c.decodeIfPresent(String.self, forKey: .host)
      ?? c.decodeIfPresent(String.self, forKey: .serverHost)
      ?? "127.0.0.1"
    modelSpec = try c.decodeIfPresent(String.self, forKey: .modelSpec)
    allowedOutputDirectory = try c.decodeIfPresent(String.self, forKey: .allowedOutputDirectory)
      ?? c.decodeIfPresent(String.self, forKey: .outputDirectory)
    seedvr2WeightsPath = try c.decodeIfPresent(String.self, forKey: .seedvr2WeightsPath)
    // Absent `providers` key → seed the default (e.g. upgrading from an old desktop-only
    // config). An explicit `providers: {}` is respected as intentionally empty.
    providers = try c.decodeIfPresent(AIProviderRegistry.self, forKey: .providers)
      ?? AIProviderRegistry(promptOptimization: AIProviderRegistry.lmStudioPromptDefault)
    replicate = try c.decodeIfPresent(ReplicateProviderConfig.self, forKey: .replicate)
    contentModeDefaultPresets = try c.decodeIfPresent([String: String].self, forKey: .contentModeDefaultPresets) ?? [:]
    krea2Models = try c.decodeIfPresent([String: String].self, forKey: .krea2Models) ?? [:]
    renderDefaults = try c.decodeIfPresent(RenderDefaultsConfig.self, forKey: .renderDefaults) ?? RenderDefaultsConfig()
    videoDefaults = try c.decodeIfPresent(VideoDefaultsConfig.self, forKey: .videoDefaults) ?? VideoDefaultsConfig()
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(port, forKey: .port)
    try c.encode(host, forKey: .host)
    try c.encodeIfPresent(modelSpec, forKey: .modelSpec)
    try c.encodeIfPresent(allowedOutputDirectory, forKey: .allowedOutputDirectory)
    try c.encodeIfPresent(seedvr2WeightsPath, forKey: .seedvr2WeightsPath)
    try c.encode(providers, forKey: .providers)
    try c.encodeIfPresent(replicate, forKey: .replicate)
    if !contentModeDefaultPresets.isEmpty {
      try c.encode(contentModeDefaultPresets, forKey: .contentModeDefaultPresets)
    }
    if !krea2Models.isEmpty {
      try c.encode(krea2Models, forKey: .krea2Models)
    }
    if !renderDefaults.isEmpty {
      try c.encode(renderDefaults, forKey: .renderDefaults)
    }
    if !videoDefaults.isEmpty {
      try c.encode(videoDefaults, forKey: .videoDefaults)
    }
  }

  // MARK: - Paths

  public static func homeDirectory() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
  }

  /// The engine's state directory — `~/.comfybox`, or `COMFYBOX_STATE_DIR`
  /// when set. Mirrors `QueueStateStore.stateDirectory` exactly (K-FIX-1: a
  /// test that touches this path unguarded reads/writes/DELETES the LIVE
  /// engine's config — see `ComfyBoxStateDirectoryIsolation.swift`). COMPUTED,
  /// not a cached `static let`, so the override is honored even if it's set
  /// after this type has already been touched once in the process.
  public static func stateDirectory() -> URL {
    if let override = ProcessInfo.processInfo.environment["COMFYBOX_STATE_DIR"], !override.isEmpty {
      let dir = URL(fileURLWithPath: (override as NSString).expandingTildeInPath, isDirectory: true)
      try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
      return dir
    }
    let dir = homeDirectory().appendingPathComponent(".comfybox", isDirectory: true)
    try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  /// `~/.comfybox/config.json`, or `$COMFYBOX_STATE_DIR/config.json`.
  public static func defaultPath() -> URL {
    stateDirectory().appendingPathComponent("config.json")
  }

  /// `~/.coffeeshop/providers.json` (retiring image service).
  public static func coffeeShopProvidersPath() -> URL {
    homeDirectory().appendingPathComponent(".coffeeshop/providers.json")
  }

  /// `~/.coffeeshop/image-service/config.json` (retiring image service).
  public static func coffeeShopConfigPath() -> URL {
    homeDirectory().appendingPathComponent(".coffeeshop/image-service/config.json")
  }

  // MARK: - Load / Save / Migrate

  /// Load `~/.comfybox/config.json`. If absent, auto-migrate from `~/.coffeeshop/`
  /// (non-destructive), persist the result, and return it.
  @discardableResult
  public static func loadOrMigrate(
    at path: URL = ComfyBoxServerConfig.defaultPath(),
    coffeeShopProviders: URL = ComfyBoxServerConfig.coffeeShopProvidersPath(),
    coffeeShopConfig: URL = ComfyBoxServerConfig.coffeeShopConfigPath(),
    fileManager: FileManager = .default
  ) -> ComfyBoxServerConfig {
    if fileManager.fileExists(atPath: path.path),
       let data = try? Data(contentsOf: path),
       let config = try? JSONDecoder().decode(ComfyBoxServerConfig.self, from: data) {
      return config
    }

    let migrated = migrate(
      coffeeShopProviders: coffeeShopProviders,
      coffeeShopConfig: coffeeShopConfig,
      fileManager: fileManager
    )
    try? migrated.save(to: path, fileManager: fileManager)
    return migrated
  }

  public func save(to path: URL = ComfyBoxServerConfig.defaultPath(), fileManager: FileManager = .default) throws {
    let dir = path.deletingLastPathComponent()
    try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(self)
    try data.write(to: path, options: .atomic)
  }

  /// Build a fresh config, folding in whatever the retiring image service left under
  /// `~/.coffeeshop/`. Originals are only read, never modified.
  public static func migrate(
    coffeeShopProviders: URL = ComfyBoxServerConfig.coffeeShopProvidersPath(),
    coffeeShopConfig: URL = ComfyBoxServerConfig.coffeeShopConfigPath(),
    fileManager: FileManager = .default
  ) -> ComfyBoxServerConfig {
    var config = ComfyBoxServerConfig()

    // providers.json → replicate
    if fileManager.fileExists(atPath: coffeeShopProviders.path),
       let data = try? Data(contentsOf: coffeeShopProviders),
       let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let rep = root["replicate"] as? [String: Any] {
      config.replicate = ReplicateProviderConfig(
        apiKey: rep["apiKey"] as? String,
        baseUrl: rep["baseUrl"] as? String,
        model: rep["model"] as? String,
        imageModel: rep["imageModel"] as? String,
        videoModel: rep["videoModel"] as? String
      )
    }

    // image-service config.json → prompt-optimization endpoint + output dir
    if fileManager.fileExists(atPath: coffeeShopConfig.path),
       let data = try? Data(contentsOf: coffeeShopConfig),
       let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
      if let enhancer = root["enhancer"] as? [String: Any],
         let baseUrl = enhancer["baseUrl"] as? String, !baseUrl.isEmpty {
        config.providers.promptOptimization = AIProviderEndpoint(
          baseUrl: baseUrl,
          model: (enhancer["model"] as? String) ?? AIProviderRegistry.lmStudioPromptDefault.model,
          apiKey: enhancer["apiKey"] as? String
        )
      }
      if let outputDir = (root["outputDir"] ?? root["outputDirectory"]) as? String, !outputDir.isEmpty {
        config.allowedOutputDirectory = outputDir
      }
    }

    // Always leave prompt optimization pointing somewhere usable.
    if config.providers.promptOptimization == nil {
      config.providers.promptOptimization = AIProviderRegistry.lmStudioPromptDefault
    }
    return config
  }
}
