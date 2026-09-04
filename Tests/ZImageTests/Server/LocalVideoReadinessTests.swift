import XCTest
import MLX
@testable import ZImage

/// #298 review — pure `LocalVideoReadiness.compute` coverage (finding 1: real
/// integrity, not filename-suffix matching) plus the lock-backed monitor
/// (finding 3: filesystem work stays off the `/health` request path).
final class LocalVideoReadinessTests: XCTestCase {

  private func tempDir(_ name: String) throws -> URL {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("lvr-\(name)-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  private func writeValidSafetensors(at url: URL) throws {
    let values: [Float] = [1, 2, 3, 4]
    try MLX.save(arrays: ["weight": MLXArray(values, [4]).asType(.bfloat16)], metadata: [:], url: url)
  }

  private func writeCompleteGemmaDir(at gemma: URL) throws {
    try writeValidSafetensors(at: gemma.appendingPathComponent("model.safetensors"))
    try Data("{}".utf8).write(to: gemma.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: gemma.appendingPathComponent("tokenizer.json"))
  }

  // MARK: - not configured

  func testNotConfiguredIsNotReady() {
    let r = LocalVideoReadiness.compute(weightsPath: nil, gemmaPath: nil, upsamplerPath: nil)
    XCTAssertFalse(r.ready)
    XCTAssertEqual(r.reason, "not_configured")
    XCTAssertNotNil(r.checkedAt, "compute() always records that a check ran, even when unconfigured")
  }

  // MARK: - ready

  func testReadyWhenAllRequiredAssetsPresentAndValid() throws {
    let weights = try tempDir("weights")
    let gemma = try tempDir("gemma")
    defer {
      try? FileManager.default.removeItem(at: weights)
      try? FileManager.default.removeItem(at: gemma)
    }
    try writeValidSafetensors(at: weights.appendingPathComponent("local-monolith.safetensors"))
    try writeCompleteGemmaDir(at: gemma)

    let r = LocalVideoReadiness.compute(weightsPath: weights.path, gemmaPath: gemma.path, upsamplerPath: nil)
    XCTAssertTrue(r.ready)
    XCTAssertNil(r.reason)
    XCTAssertEqual(r.requiredAssets.count, 2)
    XCTAssertTrue(r.requiredAssets.allSatisfy(\.valid))
    XCTAssertEqual(r.optionalAssets.first?.valid, true, "an absent optional upsampler must not disable core video")
  }

  // MARK: - truncated shard: the filename is right, the bytes are not

  func testTruncatedWeightsShardFailsReadinessWithTruncatedReason() throws {
    let weights = try tempDir("weights-truncated")
    let gemma = try tempDir("gemma-ok")
    defer {
      try? FileManager.default.removeItem(at: weights)
      try? FileManager.default.removeItem(at: gemma)
    }
    let shard = weights.appendingPathComponent("local-monolith.safetensors")
    try writeValidSafetensors(at: shard)
    let full = try Data(contentsOf: shard)
    try full.dropLast(4).write(to: shard)
    try writeCompleteGemmaDir(at: gemma)

    let r = LocalVideoReadiness.compute(weightsPath: weights.path, gemmaPath: gemma.path, upsamplerPath: nil)
    XCTAssertFalse(r.ready, "a shard whose filename looks right but is truncated must not read as ready")
    XCTAssertEqual(r.reason, "truncated:local-monolith.safetensors")
    let weightsAsset = try XCTUnwrap(r.requiredAssets.first { $0.name == "ltx2_weights" })
    XCTAssertFalse(weightsAsset.valid)
    XCTAssertEqual(weightsAsset.error, "truncated:local-monolith.safetensors")
  }

  // MARK: - missing directories

  func testMissingDirectoriesAreNotReady() throws {
    let root = try tempDir("root")
    defer { try? FileManager.default.removeItem(at: root) }

    let r = LocalVideoReadiness.compute(
      weightsPath: root.appendingPathComponent("missing-weights").path,
      gemmaPath: root.appendingPathComponent("missing-gemma").path,
      upsamplerPath: nil)
    XCTAssertFalse(r.ready)
    XCTAssertTrue(r.requiredAssets.allSatisfy { !$0.valid })
    XCTAssertEqual(r.requiredAssets.first?.error, "path_not_found")
  }

  // MARK: - not_a_directory (#298 review finding 5)

  func testAFileWherADirectoryIsExpectedIsRejected() throws {
    let root = try tempDir("root-file-not-dir")
    defer { try? FileManager.default.removeItem(at: root) }
    let notADirectory = root.appendingPathComponent("not-a-directory")
    try Data("x".utf8).write(to: notADirectory)

    let r = LocalVideoReadiness.compute(weightsPath: notADirectory.path, gemmaPath: nil, upsamplerPath: nil)
    let weightsAsset = try XCTUnwrap(r.requiredAssets.first { $0.name == "ltx2_weights" })
    XCTAssertEqual(weightsAsset.error, "not_a_directory")
  }

  // MARK: - path_not_readable (#298 review finding 5)

  func testAnUnreadableWeightsDirectoryIsRejected() throws {
    let weights = try tempDir("unreadable-weights")
    defer {
      try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: weights.path)
      try? FileManager.default.removeItem(at: weights)
    }
    try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: weights.path)

    let r = LocalVideoReadiness.compute(weightsPath: weights.path, gemmaPath: nil, upsamplerPath: nil)
    let weightsAsset = try XCTUnwrap(r.requiredAssets.first { $0.name == "ltx2_weights" })
    XCTAssertEqual(weightsAsset.error, "path_not_readable")
  }

  // MARK: - tilde expansion (#298 review finding 5)

  func testTildeIsExpandedToTheHomeDirectory() {
    let expanded = LocalVideoReadiness.expandedPath("~/some/nested/path")
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    XCTAssertNotNil(expanded)
    XCTAssertTrue(expanded!.hasPrefix(home), "\(expanded ?? "nil") should start with \(home)")
    XCTAssertTrue(expanded!.hasSuffix("/some/nested/path"))
    XCTAssertFalse(expanded!.hasPrefix("~"))
  }

  func testEmptyAndWhitespacePathsAreTreatedAsUnconfigured() {
    XCTAssertNil(LocalVideoReadiness.expandedPath(nil))
    XCTAssertNil(LocalVideoReadiness.expandedPath(""))
    XCTAssertNil(LocalVideoReadiness.expandedPath("   "))
  }

  // MARK: - optional upsampler present-and-valid (#298 review finding 5)

  func testOptionalUpsamplerPresentAndValid() throws {
    let dir = try tempDir("upsampler-ok")
    defer { try? FileManager.default.removeItem(at: dir) }
    let upsampler = dir.appendingPathComponent("upsampler.safetensors")
    try writeValidSafetensors(at: upsampler)

    let r = LocalVideoReadiness.compute(weightsPath: nil, gemmaPath: nil, upsamplerPath: upsampler.path)
    let optional = try XCTUnwrap(r.optionalAssets.first)
    XCTAssertEqual(optional.name, "ltx2_upsampler")
    XCTAssertFalse(optional.required)
    XCTAssertTrue(optional.configured)
    XCTAssertTrue(optional.valid)
    XCTAssertNil(optional.error)
  }

  func testOptionalUpsamplerTruncatedIsInvalid() throws {
    let dir = try tempDir("upsampler-truncated")
    defer { try? FileManager.default.removeItem(at: dir) }
    let upsampler = dir.appendingPathComponent("upsampler.safetensors")
    try writeValidSafetensors(at: upsampler)
    let full = try Data(contentsOf: upsampler)
    try full.dropLast(4).write(to: upsampler)

    let r = LocalVideoReadiness.compute(weightsPath: nil, gemmaPath: nil, upsamplerPath: upsampler.path)
    let optional = try XCTUnwrap(r.optionalAssets.first)
    XCTAssertFalse(optional.valid)
    XCTAssertEqual(optional.error, "truncated:upsampler.safetensors")
  }

  // MARK: - missing literal gemma files

  func testMissingGemmaConfigIsRejected() throws {
    let gemma = try tempDir("gemma-missing-config")
    defer { try? FileManager.default.removeItem(at: gemma) }
    try writeValidSafetensors(at: gemma.appendingPathComponent("model.safetensors"))
    try Data("{}".utf8).write(to: gemma.appendingPathComponent("tokenizer.json"))

    let r = LocalVideoReadiness.compute(weightsPath: nil, gemmaPath: gemma.path, upsamplerPath: nil)
    let gemmaAsset = try XCTUnwrap(r.requiredAssets.first { $0.name == "gemma_text_encoder" })
    XCTAssertEqual(gemmaAsset.error, "missing_config.json")
  }

  // MARK: - LocalVideoReadinessMonitor: background computation, snapshot-only reads

  func testMonitorPublishesAComputedSnapshotAfterStart() async throws {
    let weights = try tempDir("monitor-weights")
    let gemma = try tempDir("monitor-gemma")
    defer {
      try? FileManager.default.removeItem(at: weights)
      try? FileManager.default.removeItem(at: gemma)
    }
    try writeValidSafetensors(at: weights.appendingPathComponent("local-monolith.safetensors"))
    try writeCompleteGemmaDir(at: gemma)

    let monitor = LocalVideoReadinessMonitor(weightsPath: weights.path, gemmaPath: gemma.path, upsamplerPath: nil)
    XCTAssertEqual(monitor.current(), .unchecked, "no computation has run before start()")

    monitor.start()
    defer { monitor.stop() }

    var snapshot = monitor.current()
    for _ in 0..<100 where snapshot == .unchecked {
      try await Task.sleep(nanoseconds: 20_000_000)
      snapshot = monitor.current()
    }
    XCTAssertTrue(snapshot.ready)
    XCTAssertNotNil(snapshot.checkedAt)
  }

  func testMonitorStartIsIdempotent() {
    let monitor = LocalVideoReadinessMonitor(weightsPath: nil, gemmaPath: nil, upsamplerPath: nil)
    monitor.start()
    monitor.start()  // must not spawn a second loop
    monitor.stop()
  }
}
