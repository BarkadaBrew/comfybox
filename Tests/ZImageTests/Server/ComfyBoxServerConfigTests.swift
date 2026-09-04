import XCTest
@testable import ZImage

final class ComfyBoxServerConfigTests: XCTestCase {

  private func makeTempDir() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-config-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
  }

  // MARK: - Defaults & constants

  func testDefaultsUseCanonicalPortAndLMStudioProvider() {
    let config = ComfyBoxServerConfig()
    XCTAssertEqual(config.port, 7870)
    XCTAssertEqual(ComfyBoxServerConfig.canonicalPort, 7870)
    XCTAssertEqual(ComfyBoxServerConfig.deprecatedAliasPort, 7862)
    XCTAssertEqual(config.host, "127.0.0.1")
    XCTAssertEqual(config.providers.promptOptimization?.model, "dans-pe-v1.3.0-24b-heresy@8bit")
    XCTAssertEqual(config.providers.promptOptimization?.baseUrl, "http://localhost:1234/v1")
  }

  // MARK: - Codable round-trip & tolerant decode

  func testRoundTripEncodeDecode() throws {
    var config = ComfyBoxServerConfig(port: 7870, host: "0.0.0.0", modelSpec: "z-image-turbo")
    config.providers.vision = AIProviderEndpoint(baseUrl: "http://localhost:1235/v1", model: "moondream")
    config.replicate = ReplicateProviderConfig(apiKey: "k", videoModel: "kwaivgi/kling-v1.6-standard")

    let data = try JSONEncoder().encode(config)
    let decoded = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: data)
    XCTAssertEqual(config, decoded)
  }

  func testPartialJSONDecodesWithDefaults() throws {
    // Only a host is provided; everything else must fall back to defaults.
    let json = Data(#"{ "host": "192.168.1.5" }"#.utf8)
    let config = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: json)
    XCTAssertEqual(config.host, "192.168.1.5")
    XCTAssertEqual(config.port, 7870)
    // Absent providers key seeds the LM Studio default (smooth upgrade from bare configs).
    XCTAssertEqual(config.providers.promptOptimization, AIProviderRegistry.lmStudioPromptDefault)
    XCTAssertNil(config.replicate)
  }

  func testExplicitEmptyProvidersIsRespected() throws {
    let json = Data(#"{ "host": "h", "providers": {} }"#.utf8)
    let config = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: json)
    XCTAssertNil(config.providers.promptOptimization) // explicit empty stays empty
  }

  func testDecodesLegacyDesktopKeys() throws {
    // The desktop's old AppConfig shape must still load (and preserve the output dir).
    let json = Data(#"{ "serverHost": "127.0.0.1", "serverPort": 7870, "outputDirectory": "~/Pictures/ComfyBox" }"#.utf8)
    let config = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: json)
    XCTAssertEqual(config.host, "127.0.0.1")
    XCTAssertEqual(config.port, 7870)
    XCTAssertEqual(config.allowedOutputDirectory, "~/Pictures/ComfyBox")

    // Re-encoding writes canonical keys, not the legacy ones.
    let reEncoded = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(config)) as? [String: Any]
    XCTAssertEqual(reEncoded?["host"] as? String, "127.0.0.1")
    XCTAssertNil(reEncoded?["serverHost"])
    XCTAssertNil(reEncoded?["outputDirectory"])
  }

  // MARK: - Save / load

  func testSaveThenLoadRoundTrips() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("config.json")

    let original = ComfyBoxServerConfig(port: 7870, host: "127.0.0.1", modelSpec: "chroma")
    try original.save(to: path)

    XCTAssertTrue(FileManager.default.fileExists(atPath: path.path))
    let loaded = ComfyBoxServerConfig.loadOrMigrate(at: path)
    XCTAssertEqual(loaded.modelSpec, "chroma")
    XCTAssertEqual(loaded.port, 7870)
  }

  // MARK: - Migration (non-destructive)

  func testMigrateFoldsReplicateAndEnhancerLeavingOriginalsUntouched() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }

    let providers = dir.appendingPathComponent("providers.json")
    let providersJSON = Data(#"""
    { "replicate": { "apiKey": "sk-x", "model": "prunaai/z-image-turbo", "videoModel": "kwaivgi/kling-v1.6-standard" } }
    """#.utf8)
    try providersJSON.write(to: providers)

    let csConfig = dir.appendingPathComponent("image-service-config.json")
    let csJSON = Data(#"""
    { "enhancer": { "baseUrl": "http://localhost:4321/v1", "model": "my-optimizer", "apiKey": "e-key" }, "outputDir": "/tmp/renders" }
    """#.utf8)
    try csJSON.write(to: csConfig)

    let migrated = ComfyBoxServerConfig.migrate(coffeeShopProviders: providers, coffeeShopConfig: csConfig)

    XCTAssertEqual(migrated.replicate?.apiKey, "sk-x")
    XCTAssertEqual(migrated.replicate?.videoModel, "kwaivgi/kling-v1.6-standard")
    XCTAssertEqual(migrated.providers.promptOptimization?.baseUrl, "http://localhost:4321/v1")
    XCTAssertEqual(migrated.providers.promptOptimization?.model, "my-optimizer")
    XCTAssertEqual(migrated.allowedOutputDirectory, "/tmp/renders")

    // Originals are read-only: contents unchanged.
    XCTAssertEqual(try Data(contentsOf: providers), providersJSON)
    XCTAssertEqual(try Data(contentsOf: csConfig), csJSON)
  }

  func testMigrateWithNoSourcesFallsBackToLMStudioDefault() {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-absent-\(UUID().uuidString)", isDirectory: true)
    let migrated = ComfyBoxServerConfig.migrate(
      coffeeShopProviders: dir.appendingPathComponent("providers.json"),
      coffeeShopConfig: dir.appendingPathComponent("config.json")
    )
    XCTAssertEqual(migrated.providers.promptOptimization, AIProviderRegistry.lmStudioPromptDefault)
    XCTAssertNil(migrated.replicate)
  }

  func testLoadOrMigrateWritesFileOnFirstLaunch() throws {
    let dir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("nested/config.json")
    let absent = dir.appendingPathComponent("nope.json")

    XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
    let config = ComfyBoxServerConfig.loadOrMigrate(at: path, coffeeShopProviders: absent, coffeeShopConfig: absent)
    XCTAssertTrue(FileManager.default.fileExists(atPath: path.path), "first launch should persist config")
    XCTAssertEqual(config.port, 7870)
    XCTAssertEqual(config.providers.promptOptimization, AIProviderRegistry.lmStudioPromptDefault)
  }

  // MARK: - Krea 2 model directories (WP-E5)

  func testKrea2ModelsDecodesEncodesAndDefaultsEmpty() throws {
    // Absent key → empty table (the built-in defaults apply at runtime).
    let bare = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: Data(#"{ "host": "h" }"#.utf8))
    XCTAssertEqual(bare.krea2Models, [:])
    XCTAssertFalse(String(data: try JSONEncoder().encode(bare), encoding: .utf8)!.contains("krea2Models"),
                   "an empty table is not written")

    // Declared spec → directory round-trips.
    var config = ComfyBoxServerConfig()
    config.krea2Models = ["krea2-raw": "/Volumes/Bolt/Models/krea2-raw"]
    let data = try JSONEncoder().encode(config)
    let decoded = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: data)
    XCTAssertEqual(decoded, config)
    XCTAssertEqual(decoded.krea2Models["krea2-raw"], "/Volumes/Bolt/Models/krea2-raw")
  }

  // MARK: - Image memory caps (#22)

  func testImageMemoryCapsAbsentKeyDefaultsAndDoesRoundTrip() throws {
    let bare = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: Data(#"{ "host": "h" }"#.utf8))
    XCTAssertEqual(bare.imageMemoryCaps, .default)
    XCTAssertEqual(bare.imageMemoryCaps.maxLongEdge, 4096)
    XCTAssertEqual(bare.imageMemoryCaps.maxPixels, 16_777_216)
    XCTAssertEqual(bare.imageMemoryCaps.minAvailableHeadroomFraction, 0.10, accuracy: 1e-9)

    var config = ComfyBoxServerConfig()
    config.imageMemoryCaps = ImageMemoryCapsConfig(maxLongEdge: 3072, maxPixels: 3072 * 3072, minAvailableHeadroomFraction: 0.2)
    let data = try JSONEncoder().encode(config)
    let decoded = try JSONDecoder().decode(ComfyBoxServerConfig.self, from: data)
    XCTAssertEqual(decoded.imageMemoryCaps, config.imageMemoryCaps)
  }

  func testImageMemoryCapsValidationRejectsNonPositiveAndOutOfRangeHeadroom() {
    var badLongEdge = ComfyBoxServerConfig()
    badLongEdge.imageMemoryCaps.maxLongEdge = 0
    XCTAssertThrowsError(try ServerConfigStore.validate(badLongEdge))

    var badPixels = ComfyBoxServerConfig()
    badPixels.imageMemoryCaps.maxPixels = -1
    XCTAssertThrowsError(try ServerConfigStore.validate(badPixels))

    var badHeadroom = ComfyBoxServerConfig()
    badHeadroom.imageMemoryCaps.minAvailableHeadroomFraction = 1.0
    XCTAssertThrowsError(try ServerConfigStore.validate(badHeadroom))

    var okConfig = ComfyBoxServerConfig()
    okConfig.imageMemoryCaps = ImageMemoryCapsConfig(maxLongEdge: 2048, maxPixels: 2048 * 2048, minAvailableHeadroomFraction: 0.0)
    XCTAssertNoThrow(try ServerConfigStore.validate(okConfig))
  }
}
