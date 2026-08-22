import Foundation
import XCTest

@testable import ZImage

/// WP-E15 (§3.13, D18) — the gate that replaced the tier gate.
///
/// Before T2, `eta != 0` on the Krea 2 path was a flat refusal. It still fails
/// loud, but for a different and narrower reason: RES4LYF's SDE is defined
/// against RES4LYF's own solver, so it is HONOURED on the RES4LYF ports and
/// REFUSED by name on every other sampler. What it is never is ignored — which
/// is the property D18 actually asks for.
final class Krea2EtaGateTests: XCTestCase {


  /// The gate removal has a boundary: RES4LYF's SDE is defined against
  /// RES4LYF's own prepared grid, so `eta != 0` is honoured on the RES4LYF
  /// family and REFUSED — 400, naming the sampler — on the samplers that are
  /// ports of other stacks. It is never silently dropped.
  func testEtaIsRefusedOnNonRES4LYFSamplersAndAcceptedOnTheFamily() throws {
    for kind in SchedulerKind.allCases where kind.isRES4LYFFamily {
      XCTAssertNotNil(
        try Krea2Pipeline.makeSDEInjector(
          eta: 0.5, sampler: kind, stageSeed: 1, layout: .channelsAtAxis1),
        "\(kind.rawValue) is a RES4LYF port and takes the SDE")
    }
    for kind in SchedulerKind.allCases where !kind.isRES4LYFFamily {
      XCTAssertThrowsError(
        try Krea2Pipeline.makeSDEInjector(
          eta: 0.5, sampler: kind, stageSeed: 1, layout: .channelsAtAxis1),
        "\(kind.rawValue) must refuse eta, not ignore it"
      ) { error in
        guard case Krea2ScheduleError.etaUnsupportedSampler(let sampler, let value) = error else {
          return XCTFail("\(kind.rawValue): wrong error \(error)")
        }
        XCTAssertEqual(sampler, kind.rawValue)
        XCTAssertEqual(value, "0.5")
        XCTAssertTrue("\(error)".contains(kind.rawValue), "\(error)")
      }
    }
    // eta 0 is no SDE at all, on every sampler.
    for kind in SchedulerKind.allCases {
      XCTAssertNil(
        try Krea2Pipeline.makeSDEInjector(
          eta: 0, sampler: kind, stageSeed: 1, layout: .channelsAtAxis1),
        "\(kind.rawValue) at eta 0")
    }
  }

  /// T2 has landed, so the pipeline's tier gate no longer refuses `eta`; T3's
  /// `bongmath` is still refused (D18: an unimplemented tier is a 400).
  func testTierGateNoLongerRefusesEtaButStillRefusesBongmath() {
    XCTAssertNoThrow(try Krea2Pipeline.validateTiers(eta: 0.5, bongmath: false))
    XCTAssertNoThrow(try Krea2Pipeline.validateTiers(eta: 0, bongmath: false))
    XCTAssertThrowsError(try Krea2Pipeline.validateTiers(eta: 0.5, bongmath: true)) { error in
      guard case Krea2ScheduleError.tierNotImplemented(let field, _, let tier) = error else {
        return XCTFail("wrong error \(error)")
      }
      XCTAssertEqual(field, "bongmath")
      XCTAssertEqual(tier, "T3")
    }
  }
}
