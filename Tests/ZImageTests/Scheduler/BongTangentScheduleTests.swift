import Foundation
import XCTest

@testable import ZImage

/// WP-E11 — `bong_tangent` (FDD-krea2-raw-recipe §3.11, D6, AC-19, AC-20).
///
/// RES4LYF's `bong_tangent_scheduler` (`sigmas.py:4065-4098`, pinned commit
/// 26036f6) is two arctan arcs joined at exactly `middle = 0.5`, `steps + 1`
/// values ending at 0, with integer truncation that makes it non-smooth in
/// `steps`. It accepts `model_sampling` and never reads it. The Swift port is
/// literal; the oracle is the E18 fixture `comfy_sigmas.json`, dumped from the
/// pinned upstream source.
final class BongTangentScheduleTests: XCTestCase {

  static let fixtureSteps = [2, 6, 8, 9, 10, 12, 20]

  private func fixture() throws -> [String: Any] {
    let all = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    return try XCTUnwrap(all["bong_tangent"] as? [String: Any], "comfy_sigmas.json: bong_tangent")
  }

  // MARK: - AC-19: matches upstream exactly

  /// Every pinned step count matches the upstream dump to 1e-6, element for
  /// element, with `steps + 1` entries, `1.0` first, exactly `0.5` at the join
  /// and exactly `0.0` last.
  func testMatchesUpstreamFixture() throws {
    let bong = try fixture()
    for steps in Self.fixtureSteps {
      let want = try SchedulerOracleFixtures.doubles(bong["\(steps)"], "bong_tangent.\(steps)")
      let got = SigmaSchedule.bongTangent(numSteps: steps)
      XCTAssertEqual(got.count, steps + 1, "bong_tangent(\(steps)): steps+1 elements")
      XCTAssertEqual(got.count, want.count, "bong_tangent(\(steps)): fixture length")
      for (i, (g, w)) in zip(got, want).enumerated() {
        XCTAssertEqual(Double(g), w, accuracy: 1e-6, "bong_tangent(\(steps)) i=\(i)")
      }
      XCTAssertEqual(got.first, 1.0, "bong_tangent(\(steps)): starts at exactly 1.0")
      XCTAssertEqual(got.last, 0.0, "bong_tangent(\(steps)): ends at exactly 0.0")
      XCTAssertTrue(got.contains(0.5), "bong_tangent(\(steps)): exactly 0.5 at the join")
      for i in 1..<got.count {
        XCTAssertLessThan(got[i], got[i - 1], "bong_tangent(\(steps)): not decreasing at \(i)")
      }
    }
  }

  /// The FDD's own quoted pin (§3.11 / AC-19), independent of the fixture file.
  func testMatchesFDDPinnedSix() {
    let pinned: [Float] = [1.0, 0.928970, 0.797686, 0.5, 0.185601, 0.056802, 0.0]
    let got = SigmaSchedule.bongTangent(numSteps: 6)
    XCTAssertEqual(got.count, pinned.count)
    for (i, (g, w)) in zip(got, pinned).enumerated() {
      XCTAssertEqual(g, w, accuracy: 1e-6, "bong_tangent(6) i=\(i)")
    }
  }

  /// The join sits at index `int(0.6·(steps+2))` after the stage-1 truncation —
  /// integer arithmetic, so the schedule is deliberately non-smooth in `steps`
  /// ("port literally, do not clean up"). Read the index off the fixture rather
  /// than re-deriving it, so the test cannot share a bug with the port.
  func testJoinIndexIsIntegerTruncated() throws {
    let bong = try fixture()
    for steps in Self.fixtureSteps {
      let want = try SchedulerOracleFixtures.doubles(bong["\(steps)"], "bong_tangent.\(steps)")
      let got = SigmaSchedule.bongTangent(numSteps: steps)
      let wantJoin = try XCTUnwrap(want.firstIndex(where: { $0 == 0.5 }))
      let gotJoin = try XCTUnwrap(got.firstIndex(of: 0.5))
      XCTAssertEqual(gotJoin, wantJoin, "bong_tangent(\(steps)): join index")
      // Upstream: stage_1_len = (steps+2) - ((steps+2) - int(0.6·(steps+2))) = int(0.6·(steps+2)),
      // and tan_sigmas_1[:-1] drops one, so the join lands at int(0.6·(steps+2)) - 1.
      XCTAssertEqual(gotJoin, Int(Double(steps + 2) * 0.6) - 1, "bong_tangent(\(steps)): truncation rule")
    }
  }

  // MARK: - AC-20: shift-free

  /// Identical sigmas for shift 1.0 / 1.15 / 2.475 and for `useDynamicShifting`
  /// true/false, with and without `mu` — `resolveSigmas` never consults the
  /// model for this schedule (D6). Exact `==`, not a tolerance: there is no
  /// arithmetic path by which the config could enter.
  func testIgnoresModelSampling() throws {
    let reference = SigmaSchedule.bongTangent(numSteps: 9)
    let configs: [(String, ZImageSchedulerConfig)] = [
      ("krea2 (ModelSamplingFlux)", Krea2Sampling.schedulerConfig()),
      ("static shift 1.0", ZImageSchedulerConfig(numTrainTimesteps: 1000, shift: 1.0, useDynamicShifting: false)),
      ("static shift 1.15", ZImageSchedulerConfig(numTrainTimesteps: 1000, shift: 1.15, useDynamicShifting: false)),
      ("static shift 2.475", ZImageSchedulerConfig(numTrainTimesteps: 1000, shift: 2.475, useDynamicShifting: false)),
      ("static shift 3.0 / T=10000", ZImageSchedulerConfig(numTrainTimesteps: 10000, shift: 3.0, useDynamicShifting: false)),
      ("zimage turbo", FlowMatchSchedulerTests.makeConfig()),
    ]
    let mus: [Float?] = [nil, 0.0, 0.5, 1.15, 2.475, Krea2Sampling.mu(seqLen: 4096, align: 16)]
    for (label, config) in configs {
      for mu in mus {
        let sigmas = try SchedulerFactory.resolveSigmas(
          schedule: .bongTangent, numSteps: 9, config: config, mu: mu)
        XCTAssertEqual(sigmas, reference, "bong_tangent moved under \(label), mu=\(String(describing: mu))")
      }
    }
  }

  /// The grid stays in the flow domain on its own terms (it is pure index
  /// arithmetic over `1.0 → 0.5 → 0.0`), for every step count a caller could
  /// reasonably request.
  func testStaysInFlowDomain() {
    for steps in 2...64 {
      let sigmas = SigmaSchedule.bongTangent(numSteps: steps)
      XCTAssertEqual(sigmas.count, steps + 1, "steps=\(steps): count")
      XCTAssertEqual(sigmas[0], 1.0, "steps=\(steps): σ₀")
      XCTAssertEqual(sigmas.last, 0.0, "steps=\(steps): sentinel")
      XCTAssertTrue(sigmas.contains(0.5), "steps=\(steps): join")
      for (i, s) in sigmas.enumerated() {
        XCTAssertFalse(s.isNaN, "steps=\(steps) i=\(i): NaN")
        XCTAssertTrue(s >= 0.0 && s <= 1.0, "steps=\(steps) i=\(i): \(s) left [0,1]")
      }
      for i in 1..<sigmas.count {
        XCTAssertLessThan(sigmas[i], sigmas[i - 1], "steps=\(steps): not decreasing at \(i)")
      }
    }
  }

  // MARK: - Fail loud below the upstream minimum

  /// Upstream raises `ZeroDivisionError` for `steps < 2` (a one-point arc has
  /// `smax == smin`). The factory refuses the same inputs by name rather than
  /// emitting a NaN grid.
  func testFactoryRejectsFewerThanTwoSteps() {
    let config = Krea2Sampling.schedulerConfig()
    for steps in [-1, 0, 1] {
      XCTAssertThrowsError(
        try SchedulerFactory.resolveSigmas(schedule: .bongTangent, numSteps: steps, config: config, mu: nil),
        "steps=\(steps) must throw"
      ) { error in
        XCTAssertEqual(
          error as? SchedulerFactoryError,
          .stepCountBelowMinimum(.bongTangent, steps: steps, minimum: 2))
        XCTAssertTrue("\(error)".contains("bong_tangent"), "error names the schedule: \(error)")
        XCTAssertTrue("\(error)".contains("\(steps)"), "error names the step count: \(error)")
      }
      XCTAssertThrowsError(
        try SchedulerFactory.create(
          kind: .euler, sigmaSchedule: .bongTangent, numInferenceSteps: steps, config: config, mu: nil),
        "create(steps=\(steps)) must throw"
      )
    }
  }

  // MARK: - Wire name and factory seam

  /// Wire name is snake_case `bong_tangent`, matching ComfyUI/RES4LYF so a
  /// value pasted out of the workflow JSON works verbatim (§3.11).
  func testWireName() {
    XCTAssertEqual(SigmaScheduleKind.bongTangent.rawValue, "bong_tangent")
    XCTAssertEqual(SigmaScheduleKind(rawValue: "bong_tangent"), .bongTangent)
    XCTAssertNil(SigmaScheduleKind(rawValue: "bong-tangent"), "the April plan's hyphenated name is not a wire name")
    XCTAssertNil(SigmaScheduleKind(rawValue: "bongTangent"))
    XCTAssertTrue(SigmaScheduleKind.allCases.contains(.bongTangent))
  }

  /// Every sampler builds on the grid, the scheduler's sigmas are the grid
  /// (float32, `steps + 1`), and the euler/flow fast path is not the one taken.
  func testFactoryBuildsEverySamplerOnBongTangent() throws {
    let config = Krea2Sampling.schedulerConfig()
    let grid = SigmaSchedule.bongTangent(numSteps: 9)
    for kind in SchedulerKind.allCases {
      let scheduler = try SchedulerFactory.create(
        kind: kind, sigmaSchedule: .bongTangent, numInferenceSteps: 9, config: config, mu: nil, seed: 1)
      XCTAssertEqual(scheduler.numInferenceSteps, 9, "\(kind.rawValue): steps")
      let sigmas = scheduler.sigmas.asArray(Float.self)
      XCTAssertEqual(sigmas, grid, "\(kind.rawValue): scheduler sigmas are the bong_tangent grid")
    }
  }

  /// `bong_tangent` is a different grid from every model-consulting schedule,
  /// including Krea 2's native warp — so a request for it can never be
  /// silently served by `.flow` or `.krea2`.
  func testDiffersFromModelConsultingSchedules() throws {
    let config = Krea2Sampling.schedulerConfig()
    let mu = Krea2Sampling.mu(seqLen: 4096, align: 16)
    let bong = try SchedulerFactory.resolveSigmas(schedule: .bongTangent, numSteps: 9, config: config, mu: mu)
    for other in SigmaScheduleKind.allCases where other != .bongTangent {
      let sigmas = try SchedulerFactory.resolveSigmas(schedule: other, numSteps: 9, config: config, mu: mu)
      XCTAssertNotEqual(sigmas, bong, "\(other.rawValue) coincides with bong_tangent")
    }
  }
}
