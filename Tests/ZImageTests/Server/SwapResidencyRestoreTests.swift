import XCTest
@testable import ZImage

/// Unit tests for the pure swap-time residency-restore decision in
/// ``SwapResidencyRestore``.
///
/// A swap-first client (kira-daemon renders images as swap → generate) must
/// not fail because the image pipeline is not resident: the only restore path
/// lived in `runGenerate`, so once a video render evicted the image models
/// (#218) — or the engine booted without loading one — every swap threw
/// `krea2NotLoaded`, the client failed the render closed, generate was never
/// called, and image creation deadlocked permanently.
final class SwapResidencyRestoreTests: XCTestCase {

  // MARK: - Video eviction (#218): the live 2026-08-23 outage

  func testEvictedDecidesReload() {
    let d = SwapResidencyRestore.decide(
      imageModelsEvicted: true, familyPipelineMissing: true, restoreSpec: "krea2-raw")
    XCTAssertEqual(d, .reloadEvicted)
  }

  func testEvictedWinsEvenIfPipelinePresent() {
    // `imageModelsEvicted` is the authoritative #218 flag; if it is set the
    // reload path must run so the flag is cleared through the one place that
    // owns it (`reloadImageModelIfEvicted`), never bypassed.
    let d = SwapResidencyRestore.decide(
      imageModelsEvicted: true, familyPipelineMissing: false, restoreSpec: nil)
    XCTAssertEqual(d, .reloadEvicted)
  }

  // MARK: - Fresh boot: family pipeline nil, eviction flag never set

  func testFreshBootMissingPipelineLoadsSpec() {
    let d = SwapResidencyRestore.decide(
      imageModelsEvicted: false, familyPipelineMissing: true, restoreSpec: "krea2-raw")
    XCTAssertEqual(d, .load(modelSpec: "krea2-raw"))
  }

  func testFreshBootMissingPipelineWithoutSpecDoesNothing() {
    // No spec to load — the swap proceeds and fails exactly as before. The
    // decision must not invent a model.
    let d = SwapResidencyRestore.decide(
      imageModelsEvicted: false, familyPipelineMissing: true, restoreSpec: nil)
    XCTAssertEqual(d, .none)
  }

  // MARK: - Healthy path: resident pipeline swaps untouched

  func testResidentPipelineDecidesNone() {
    let d = SwapResidencyRestore.decide(
      imageModelsEvicted: false, familyPipelineMissing: false, restoreSpec: "krea2-raw")
    XCTAssertEqual(d, .none)
  }
}
