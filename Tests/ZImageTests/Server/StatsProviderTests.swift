import XCTest
@testable import ZImage

/// Unit tests for the pure formatting / threshold / summary logic in
/// ``StatsProvider``. Deliberately does not assert on live machine memory —
/// only the deterministic conversions and classifications are exercised.
final class StatsProviderTests: XCTestCase {

  // MARK: - Memory pressure thresholds

  func testDefaultThresholdsMatchNodeImageService() {
    let t = MemoryPressureThresholds.default
    XCTAssertEqual(t.warningRssMb, 8192)
    XCTAssertEqual(t.criticalRssMb, 12288)
    XCTAssertEqual(t.criticalFreeMb, 2048)
  }

  func testEvaluateNormalWhenLowRssAndAmpleFree() {
    let t = MemoryPressureThresholds.default
    XCTAssertEqual(t.evaluate(processRssMb: 4096, systemFreeMb: 32768), .normal)
  }

  func testEvaluateWarningAtRssBoundary() {
    let t = MemoryPressureThresholds.default
    // Just below the warning threshold stays normal.
    XCTAssertEqual(t.evaluate(processRssMb: 8191, systemFreeMb: 32768), .normal)
    // At the warning threshold becomes warning.
    XCTAssertEqual(t.evaluate(processRssMb: 8192, systemFreeMb: 32768), .warning)
  }

  func testEvaluateCriticalAtRssBoundary() {
    let t = MemoryPressureThresholds.default
    // Just below the critical RSS threshold is still only warning.
    XCTAssertEqual(t.evaluate(processRssMb: 12287, systemFreeMb: 32768), .warning)
    // At the critical RSS threshold becomes critical.
    XCTAssertEqual(t.evaluate(processRssMb: 12288, systemFreeMb: 32768), .critical)
  }

  func testEvaluateCriticalWhenFreeMemoryLowEvenWithModestRss() {
    let t = MemoryPressureThresholds.default
    // Low free memory forces critical even though RSS is comfortable.
    XCTAssertEqual(t.evaluate(processRssMb: 1024, systemFreeMb: 2048), .critical)
    // One MB above the free threshold is not critical from free alone.
    XCTAssertEqual(t.evaluate(processRssMb: 1024, systemFreeMb: 2049), .normal)
  }

  func testEvaluateCriticalWinsOverWarning() {
    let t = MemoryPressureThresholds.default
    // RSS in warning range but free memory critical → critical.
    XCTAssertEqual(t.evaluate(processRssMb: 9000, systemFreeMb: 100), .critical)
  }

  func testCustomThresholds() {
    let t = MemoryPressureThresholds(warningRssMb: 1000, criticalRssMb: 2000, criticalFreeMb: 500)
    XCTAssertEqual(t.evaluate(processRssMb: 999, systemFreeMb: 10000), .normal)
    XCTAssertEqual(t.evaluate(processRssMb: 1000, systemFreeMb: 10000), .warning)
    XCTAssertEqual(t.evaluate(processRssMb: 2000, systemFreeMb: 10000), .critical)
    XCTAssertEqual(t.evaluate(processRssMb: 0, systemFreeMb: 500), .critical)
  }

  // MARK: - Byte → MB conversion

  func testMemoryStatusConvertsBytesToMegabytes() {
    let provider = StatsProvider()
    let mb: UInt64 = 1024 * 1024
    let status = provider.memoryStatus(
      processRssBytes: 4096 * mb,
      systemFreeBytes: 32768 * mb,
      systemTotalBytes: 65536 * mb
    )
    XCTAssertEqual(status.processRssMb, 4096)
    XCTAssertEqual(status.systemFreeMb, 32768)
    XCTAssertEqual(status.systemTotalMb, 65536)
    XCTAssertEqual(status.pressureLevel, .normal)
  }

  func testMemoryStatusTruncatesSubMegabyteBytes() {
    let provider = StatsProvider()
    // 1.5 MB truncates to 1 MB.
    let status = provider.memoryStatus(
      processRssBytes: UInt64(1.5 * 1024 * 1024),
      systemFreeBytes: 0,
      systemTotalBytes: 0
    )
    XCTAssertEqual(status.processRssMb, 1)
  }

  func testMemoryStatusAppliesThresholdClassification() {
    let provider = StatsProvider(thresholds: .default)
    let mb: UInt64 = 1024 * 1024
    let status = provider.memoryStatus(
      processRssBytes: 13000 * mb,
      systemFreeBytes: 32768 * mb,
      systemTotalBytes: 65536 * mb
    )
    XCTAssertEqual(status.pressureLevel, .critical)
  }

  // MARK: - Uptime

  func testUptimeSecondsWholeNumber() {
    let start = Date(timeIntervalSince1970: 1000)
    let now = Date(timeIntervalSince1970: 1042)
    XCTAssertEqual(StatsProvider.uptimeSeconds(startTime: start, now: now), 42)
  }

  func testUptimeSecondsNeverNegative() {
    let start = Date(timeIntervalSince1970: 2000)
    let now = Date(timeIntervalSince1970: 1000)
    XCTAssertEqual(StatsProvider.uptimeSeconds(startTime: start, now: now), 0)
  }

  func testSnapshotClampsNegativeUptime() {
    let provider = StatsProvider()
    let memory = MemoryStatus(processRssMb: 100, systemFreeMb: 100, systemTotalMb: 100, pressureLevel: .normal)
    let snapshot = provider.snapshot(memory: memory, uptimeSeconds: -5, config: ComfyBoxServerConfig())
    XCTAssertEqual(snapshot.uptimeSeconds, 0)
  }

  // MARK: - Provider summary

  func testDefaultConfigConfiguresOnlyPromptOptimization() {
    let summary = ProvidersStatusSummarizer.summarize(ComfyBoxServerConfig())
    XCTAssertTrue(summary.promptOptimization.configured)
    XCTAssertFalse(summary.vision.configured)
    XCTAssertFalse(summary.captioning.configured)
    XCTAssertFalse(summary.replicate.configured)
    XCTAssertEqual(summary.configuredCount, 1)
    // Detail surfaces the model + base URL for a configured endpoint.
    XCTAssertEqual(
      summary.promptOptimization.detail,
      "model=dans-pe-v1.3.0-24b-heresy@8bit baseUrl=http://localhost:1234/v1"
    )
  }

  func testUnconfiguredEndpointHasNilDetail() {
    let summary = ProvidersStatusSummarizer.summarize(ComfyBoxServerConfig())
    XCTAssertNil(summary.vision.detail)
  }

  func testAllCapabilitiesConfigured() {
    var config = ComfyBoxServerConfig()
    config.providers.vision = AIProviderEndpoint(baseUrl: "http://localhost:1235/v1", model: "moondream")
    config.providers.captioning = AIProviderEndpoint(baseUrl: "http://localhost:1236/v1", model: "florence")
    config.replicate = ReplicateProviderConfig(apiKey: "secret", imageModel: "black-forest-labs/flux", videoModel: "kwaivgi/kling")

    let summary = ProvidersStatusSummarizer.summarize(config)
    XCTAssertTrue(summary.promptOptimization.configured)
    XCTAssertTrue(summary.vision.configured)
    XCTAssertTrue(summary.captioning.configured)
    XCTAssertTrue(summary.replicate.configured)
    XCTAssertEqual(summary.configuredCount, 4)
    XCTAssertEqual(summary.replicate.detail, "apiKey set imageModel=black-forest-labs/flux videoModel=kwaivgi/kling")
  }

  func testReplicateWithEmptyKeyIsUnconfigured() {
    var config = ComfyBoxServerConfig()
    config.replicate = ReplicateProviderConfig(apiKey: "   ")
    let summary = ProvidersStatusSummarizer.summarize(config)
    XCTAssertFalse(summary.replicate.configured)
    XCTAssertNotNil(summary.replicate.detail)
  }

  func testReplicateConfiguredWithKeyOnly() {
    var config = ComfyBoxServerConfig()
    config.replicate = ReplicateProviderConfig(apiKey: "secret")
    let summary = ProvidersStatusSummarizer.summarize(config)
    XCTAssertTrue(summary.replicate.configured)
    XCTAssertEqual(summary.replicate.detail, "apiKey set")
  }

  func testEndpointWithBlankBaseUrlIsUnconfigured() {
    var config = ComfyBoxServerConfig()
    config.providers.vision = AIProviderEndpoint(baseUrl: "   ", model: "moondream")
    let summary = ProvidersStatusSummarizer.summarize(config)
    XCTAssertFalse(summary.vision.configured)
  }

  func testEndpointWithEmptyModelShowsQuestionMark() {
    var config = ComfyBoxServerConfig()
    config.providers.vision = AIProviderEndpoint(baseUrl: "http://localhost:9/v1", model: "")
    let summary = ProvidersStatusSummarizer.summarize(config)
    XCTAssertTrue(summary.vision.configured)
    XCTAssertEqual(summary.vision.detail, "model=? baseUrl=http://localhost:9/v1")
  }

  // MARK: - Snapshot assembly & Codable

  func testSnapshotCarriesRenderCountsAndProviders() {
    let provider = StatsProvider()
    let memory = MemoryStatus(processRssMb: 500, systemFreeMb: 40000, systemTotalMb: 65536, pressureLevel: .normal)
    let snapshot = provider.snapshot(
      memory: memory,
      uptimeSeconds: 120,
      renderCount: 7,
      failedRenderCount: 1,
      pendingCount: 2,
      config: ComfyBoxServerConfig()
    )
    XCTAssertEqual(snapshot.uptimeSeconds, 120)
    XCTAssertEqual(snapshot.renderCount, 7)
    XCTAssertEqual(snapshot.failedRenderCount, 1)
    XCTAssertEqual(snapshot.pendingCount, 2)
    XCTAssertEqual(snapshot.memory, memory)
    XCTAssertEqual(snapshot.providers.configuredCount, 1)
  }

  func testSnapshotCodableRoundTrip() throws {
    let provider = StatsProvider()
    let memory = MemoryStatus(processRssMb: 9000, systemFreeMb: 3000, systemTotalMb: 65536, pressureLevel: .warning)
    let snapshot = provider.snapshot(
      memory: memory,
      uptimeSeconds: 300,
      renderCount: 42,
      config: ComfyBoxServerConfig()
    )
    let data = try JSONEncoder().encode(snapshot)
    let decoded = try JSONDecoder().decode(ServerStatsSnapshot.self, from: data)
    XCTAssertEqual(snapshot, decoded)
  }

  func testMemoryPressureLevelEncodesAsRawString() throws {
    let data = try JSONEncoder().encode(MemoryPressureLevel.critical)
    XCTAssertEqual(String(data: data, encoding: .utf8), "\"critical\"")
  }
}
