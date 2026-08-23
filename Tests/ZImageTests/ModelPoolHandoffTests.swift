import Logging
import XCTest

@testable import ZImage

/// Handoff eviction (2026-08-11): switching between two models that each
/// nearly fill the pool budget must succeed — a load that intends to
/// ACTIVATE may evict the active model once nothing else remains. Without
/// this, swapping krea2 ↔ kroma-v0.2 (both ~22.5GB under the 40GB budget)
/// always 507'd: the active model refused to evict and the incoming load
/// refused to fit.
final class ModelPoolHandoffTests: XCTestCase {

  private func makePool(budgetMB: Int) -> ModelPool {
    ModelPool(
      budgetMB: budgetMB,
      textEncoderPath: nil,
      maxSequenceLength: 512,
      forceTransformerOverrideOnly: false,
      logger: Logger(label: "test"))
  }

  private func registerActive(_ pool: ModelPool, id: String, vramMB: Int) async throws {
    await pool.registerExisting(
      poolKey: id, modelSpec: id, family: .krea2,
      box: PipelineBox(pipeline: NSObject()), vramEstimateMB: vramMB, detectedInfo: nil)
    _ = try await pool.activate(modelId: id)
  }

  func testHandoffEvictsActiveModelWhenNothingElseFits() async throws {
    let pool = makePool(budgetMB: 40960)
    try await registerActive(pool, id: "krea2", vramMB: 22528)

    // A 22.5GB incoming load cannot fit beside the active 22.5GB model —
    // with the handoff flag the active model is evicted instead of throwing.
    try await pool.evictIfNeeded(neededMB: 22528, allowActiveEviction: true)
    let count = await pool.count()
    XCTAssertEqual(count, 0, "active model evicted for the handoff")
  }

  func testNonHandoffLoadStillRefusesToEvictActive() async throws {
    let pool = makePool(budgetMB: 40960)
    try await registerActive(pool, id: "krea2", vramMB: 22528)

    // A background preload (activate:false) must NOT tear down the active
    // model — the original protection stands.
    do {
      try await pool.evictIfNeeded(neededMB: 22528, allowActiveEviction: false)
      XCTFail("expected budgetExceeded")
    } catch let error as ModelPoolError {
      guard case .budgetExceeded = error else {
        return XCTFail("wrong error: \(error)")
      }
    }
    let count = await pool.count()
    XCTAssertEqual(count, 1, "active model untouched")
  }

  func testHandoffPrefersInactiveEvictionFirst() async throws {
    let pool = makePool(budgetMB: 40960)
    // An inactive 12GB model + an active 22.5GB model; a 22.5GB handoff
    // load should evict ONLY what it must: inactive first, then active.
    await pool.registerExisting(
      poolKey: "z-image", modelSpec: "z-image", family: .flux1,
      box: PipelineBox(pipeline: NSObject()), vramEstimateMB: 12288, detectedInfo: nil)
    try await registerActive(pool, id: "krea2", vramMB: 22528)

    try await pool.evictIfNeeded(neededMB: 22528, allowActiveEviction: true)
    let count = await pool.count()
    XCTAssertEqual(count, 0, "both evicted — 22.5 + anything exceeds 40GB with 22.5 incoming")
  }
}

// MARK: - WP-E9 (FDD D17, AC-59): the VAE is reloaded in place, never pool-keyed

extension ModelPoolHandoffTests {

  /// A VAE change must never evict or re-key the resident pipeline: the pool
  /// key has no VAE term, and a failed (missing-file) swap on the resident
  /// slot leaves the pool entry, its instance and its count untouched.
  func testVAESwapDoesNotEvictOrRekey() async throws {
    XCTAssertEqual(ModelPool.poolKey(for: "krea2-raw"), "krea2-raw")
    XCTAssertEqual(ModelPool.poolKey(for: "krea2-raw", quantization: "q8"), "krea2-raw-q8")

    let pool = ModelPool(
      budgetMB: 40960, textEncoderPath: nil, maxSequenceLength: 512,
      forceTransformerOverrideOnly: false, logger: Logger(label: "test"))
    let resident = FileManager.default.temporaryDirectory.appending(path: "resident-\(UUID()).safetensors")
    let slot = Krea2VAESlot(unloaded: Krea2VAE(), file: resident, layout: .qwenDiffusers, source: .modelDir)
    let box = PipelineBox(pipeline: slot)
    await pool.registerExisting(
      poolKey: "krea2-raw", modelSpec: "krea2-raw", family: .krea2,
      box: box, vramEstimateMB: 22528, detectedInfo: Krea2Variant.raw)
    _ = try await pool.activate(modelId: "krea2-raw")

    let missing = FileManager.default.temporaryDirectory.appending(path: "missing-\(UUID()).safetensors")
    XCTAssertThrowsError(try slot.ensure(file: missing, source: .payload))

    let count = await pool.count()
    XCTAssertEqual(count, 1, "a VAE request never evicts")
    let entry = try await pool.activate(modelId: "krea2-raw")
    XCTAssertTrue((entry.box.pipeline as AnyObject) === slot, "same resident instance")
    XCTAssertEqual(slot.reloadCount, 0)
  }
}
