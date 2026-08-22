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

  /// T2 landed with WP-E15 and **T3 landed with WP-E16**, so the pipeline's
  /// tier gate refuses neither field any more. Neither is ignored either: each
  /// is now refused BY SAMPLER at its own factory, which is what
  /// `Krea2SchedulerResolutionTests` and `RES4LYFBongMathParityTests` assert.
  func testTierGateRefusesNeitherEtaNorBongmath() {
    for eta in [Float(0), 0.5] {
      for bongmath in [false, true] {
        XCTAssertNoThrow(
          try Krea2Pipeline.validateTiers(eta: eta, bongmath: bongmath),
          "eta \(eta) bongmath \(bongmath)")
      }
    }
  }
}
