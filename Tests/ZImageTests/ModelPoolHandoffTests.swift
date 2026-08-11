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
