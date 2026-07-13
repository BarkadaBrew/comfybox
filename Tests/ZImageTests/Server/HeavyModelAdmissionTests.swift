import XCTest
@testable import ZImage

/// Unit tests for the pure single-heavy-model residency accounting in
/// ``HeavyModelAdmission``. Deliberately uses injected byte figures — no live
/// machine memory — so the eviction/accounting decision is deterministic.
///
/// Issue: #218
final class HeavyModelAdmissionTests: XCTestCase {
  private let gb: UInt64 = 1024 * 1024 * 1024

  // MARK: - Eviction is unconditional when the other class is resident

  func testEvictsOtherWhenResident() {
    let a = HeavyModelAdmission()
    // krea2 image model (~75GB) resident, loading LTX-2 (~65GB), 40GB free now.
    let d = a.decide(needBytes: 65 * gb, availableBytes: 40 * gb, otherResidentBytes: 75 * gb)
    XCTAssertTrue(d.evictOther, "a resident other-class model must be evicted first")
  }

  func testDoesNotEvictWhenNothingResident() {
    let a = HeavyModelAdmission()
    let d = a.decide(needBytes: 65 * gb, availableBytes: 110 * gb, otherResidentBytes: 0)
    XCTAssertFalse(d.evictOther)
    XCTAssertTrue(d.admit)
    XCTAssertTrue(d.hasHeadroom)
  }

  // MARK: - The core OOM scenario the fix targets

  func testImageResidentPlusVideoIsAdmittedOnlyAfterEviction() {
    let a = HeavyModelAdmission(headroomBytes: 15 * gb)
    // This is the live jetsam case: krea2 resident (75GB), only 40GB free.
    // Without eviction 40GB < 65GB need → would OOM. With eviction the freed
    // 75GB brings projected free to 115GB, comfortably above need+headroom.
    let d = a.decide(needBytes: 65 * gb, availableBytes: 40 * gb, otherResidentBytes: 75 * gb)
    XCTAssertTrue(d.admit)
    XCTAssertTrue(d.evictOther)
    XCTAssertTrue(d.hasHeadroom)
    XCTAssertEqual(a.projectedFreeAfterEvict(availableBytes: 40 * gb, otherResidentBytes: 75 * gb), 115 * gb)
  }

  // MARK: - Headroom boundary

  func testTightAdmitWhenNeedFitsButHeadroomDoesNot() {
    let a = HeavyModelAdmission(headroomBytes: 15 * gb)
    // Projected free = 10 + 60 = 70GB. Need 65GB fits, but 65+15=80 does not.
    let d = a.decide(needBytes: 65 * gb, availableBytes: 10 * gb, otherResidentBytes: 60 * gb)
    XCTAssertTrue(d.admit)
    XCTAssertFalse(d.hasHeadroom, "admitted but headroom target not met")
  }

  func testHeadroomExactBoundaryHasHeadroom() {
    let a = HeavyModelAdmission(headroomBytes: 15 * gb)
    // Projected free exactly need + headroom → headroom satisfied.
    let d = a.decide(needBytes: 65 * gb, availableBytes: 0, otherResidentBytes: 80 * gb)
    XCTAssertTrue(d.admit)
    XCTAssertTrue(d.hasHeadroom)
  }

  // MARK: - Refusal when even eviction cannot make room

  func testRefusesWhenProjectedFreeBelowNeed() {
    let a = HeavyModelAdmission()
    // Nothing to evict and only 30GB free, need 65GB → refuse (would OOM).
    let d = a.decide(needBytes: 65 * gb, availableBytes: 30 * gb, otherResidentBytes: 0)
    XCTAssertFalse(d.admit)
    XCTAssertFalse(d.hasHeadroom)
  }

  // MARK: - Post-eviction re-probe gate

  func testAdmitsAfterEvictWhenFreeCoversNeed() {
    let a = HeavyModelAdmission()
    XCTAssertTrue(a.admitsAfterEvict(needBytes: 65 * gb, freeBytes: 100 * gb))
    XCTAssertTrue(a.admitsAfterEvict(needBytes: 65 * gb, freeBytes: 65 * gb))
    XCTAssertFalse(a.admitsAfterEvict(needBytes: 65 * gb, freeBytes: 64 * gb))
  }

  // MARK: - LTX-2 estimate sanity

  func testLtx2EstimateIsHeavy() {
    XCTAssertGreaterThanOrEqual(HeavyModelAdmission.ltx2EstimateBytes, 60 * gb)
  }
}
