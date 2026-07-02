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
    replicate: ReplicateProviderConfig? = nil
  ) {
    self.port = port
    self.host = host
    self.modelSpec = modelSpec
    self.allowedOutputDirectory = allowedOutputDirectory
    self.seedvr2WeightsPath = seedvr2WeightsPath
    self.providers = providers
    self.replicate = replicate
  }

  private enum CodingKeys: String, CodingKey {
    case port, host, modelSpec, allowedOutputDirectory, seedvr2WeightsPath, providers, replicate
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
  }

  // MARK: - Paths

  public static func homeDirectory() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
  }

  /// `~/.comfybox/config.json`.
  public static func defaultPath() -> URL {
    homeDirectory().appendingPathComponent(".comfybox/config.json")
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
