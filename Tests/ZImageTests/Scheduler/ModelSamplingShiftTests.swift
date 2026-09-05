// ModelSamplingShiftTests.swift — comfybox#154
//
// Pins the explicit flow-matching schedule shift (ComfyUI's
// `ModelSamplingAuraFlow` node) on the Z-Image / flow schedule:
//
//   time_snr_shift(alpha, t) = alpha * t / (1 + (alpha - 1) * t),  alpha == 1 -> t
//
// verified against the local ComfyUI checkout at
// `~/Projects/ComfyUI/comfy/model_sampling.py:279-282` (`time_snr_shift`) and
// `comfy_extras/nodes_model_advanced.py:148-160` (`ModelSamplingAuraFlow`
// derives from `ModelSamplingSD3`, which patches `ModelSamplingDiscreteFlow`
// with `set_parameters(shift=shift)`; AuraFlow only changes the `multiplier`,
// which scales the model's timestep input, not the sigma grid).
//
// The engine already had this formula in `SigmaSchedule.flow` behind the
// MODEL's `scheduler_config.json` `shift`; #154 makes it a per-request dial.
// The mechanism is `ZImageSchedulerConfig.applyingExplicitShift(_:)`, which is
// the Swift equivalent of ComfyUI patching `model_sampling`: it replaces the
// model's own shift AND turns off the resolution-dependent (`mu`) dynamic
// shift for that request, exactly as the node replaces the whole
// `model_sampling` object.

import XCTest
@testable import ZImage

final class ModelSamplingShiftTests: XCTestCase {

  // MARK: - Precedence / byte-identity

  /// The whole safety property of #154: `nil` shift changes NOTHING. Not
  /// "within tolerance" — the same Float bit patterns, on both the static and
  /// the dynamic-shifting config.
  func testNilShiftIsByteIdenticalToTodayStatic() {
    let config = FlowMatchSchedulerTests.makeConfig(shift: 3.0)
    for steps in [4, 9, 20, 50] {
      let before = SigmaSchedule.flow(numSteps: steps, config: config)
      let after = SigmaSchedule.flow(numSteps: steps, config: config, explicitShift: nil)
      XCTAssertEqual(before.map { $0.bitPattern }, after.map { $0.bitPattern },
                     "explicitShift: nil moved the grid at \(steps) steps")
    }
  }

  func testNilShiftIsByteIdenticalToTodayDynamic() {
    let config = FlowMatchSchedulerTests.makeConfig(
      shift: 1.0, useDynamicShifting: true,
      baseShift: 0.5, maxShift: 1.15, baseImageSeqLen: 256, maxImageSeqLen: 4096)
    let mu: Float = 0.8
    for steps in [4, 9, 20] {
      let before = SigmaSchedule.flow(numSteps: steps, config: config, mu: mu)
      let after = SigmaSchedule.flow(numSteps: steps, config: config, mu: mu, explicitShift: nil)
      XCTAssertEqual(before.map { $0.bitPattern }, after.map { $0.bitPattern },
                     "explicitShift: nil moved the dynamic grid at \(steps) steps")
    }
  }

  /// The documented precedence: an explicit shift REPLACES the mu-based
  /// dynamic shift for that request. The dynamic config with an explicit
  /// shift must equal the equivalent static config, mu ignored.
  func testExplicitShiftReplacesDynamicShift() {
    let dynamic = FlowMatchSchedulerTests.makeConfig(
      shift: 1.0, useDynamicShifting: true,
      baseShift: 0.5, maxShift: 1.15, baseImageSeqLen: 256, maxImageSeqLen: 4096)
    let staticConfig = FlowMatchSchedulerTests.makeConfig(shift: 3.0)

    let explicit = SigmaSchedule.flow(numSteps: 9, config: dynamic, mu: 0.8, explicitShift: 3.0)
    let baseline = SigmaSchedule.flow(numSteps: 9, config: staticConfig)
    XCTAssertEqual(explicit.map { $0.bitPattern }, baseline.map { $0.bitPattern })

    // And mu genuinely does not matter once a shift is explicit.
    let otherMu = SigmaSchedule.flow(numSteps: 9, config: dynamic, mu: -2.0, explicitShift: 3.0)
    XCTAssertEqual(explicit.map { $0.bitPattern }, otherMu.map { $0.bitPattern })
  }

  /// `applyingExplicitShift` is the pure seam the pipelines use. nil returns
  /// the config untouched; a value replaces `shift` and clears
  /// `useDynamicShifting`.
  func testApplyingExplicitShiftIsTheModelSamplingPatch() {
    let dynamic = FlowMatchSchedulerTests.makeConfig(
      shift: 1.0, useDynamicShifting: true,
      baseShift: 0.5, maxShift: 1.15, baseImageSeqLen: 256, maxImageSeqLen: 4096)

    let untouched = dynamic.applyingExplicitShift(nil)
    XCTAssertEqual(untouched.shift, 1.0)
    XCTAssertTrue(untouched.useDynamicShifting)

    let patched = dynamic.applyingExplicitShift(3.0)
    XCTAssertEqual(patched.shift, 3.0)
    XCTAssertFalse(patched.useDynamicShifting)
    // Everything else survives — the node replaces the shift, not the model.
    XCTAssertEqual(patched.numTrainTimesteps, dynamic.numTrainTimesteps)
    XCTAssertEqual(patched.baseShift, dynamic.baseShift)
    XCTAssertEqual(patched.maxShift, dynamic.maxShift)
    XCTAssertEqual(patched.baseImageSeqLen, dynamic.baseImageSeqLen)
    XCTAssertEqual(patched.maxImageSeqLen, dynamic.maxImageSeqLen)
    XCTAssertEqual(patched.modelSampling, dynamic.modelSampling)
  }

  // MARK: - The formula

  /// `time_snr_shift` itself, against upstream's two branches.
  func testTimeSNRShiftFormula() {
    // alpha == 1.0 is upstream's identity early-return.
    for t in [Float(0.0), 0.001, 0.25, 0.5, 0.9, 1.0] {
      XCTAssertEqual(SigmaSchedule.timeSNRShift(alpha: 1.0, t: t), t, accuracy: 0)
    }
    // alpha * t / (1 + (alpha - 1) * t)
    XCTAssertEqual(SigmaSchedule.timeSNRShift(alpha: 3.0, t: 0.5), 0.75, accuracy: 1e-6)
    XCTAssertEqual(SigmaSchedule.timeSNRShift(alpha: 3.0, t: 1.0), 1.0, accuracy: 1e-6)
    XCTAssertEqual(SigmaSchedule.timeSNRShift(alpha: 3.0, t: 0.0), 0.0, accuracy: 1e-6)
    // 1.73 is ComfyUI's own ModelSamplingAuraFlow default.
    XCTAssertEqual(SigmaSchedule.timeSNRShift(alpha: 1.73, t: 0.5),
                   1.73 * 0.5 / (1 + 0.73 * 0.5), accuracy: 1e-6)
  }

  // MARK: - Grids, computed by hand from the formula

  /// shift 1.0 is the identity: the grid is the raw `linspace(1 → 1/T)`.
  ///
  /// T = 1000, steps = 4:
  ///   sigma_min = 1·(1/1000)/(1 + 0·(1/1000)) = 0.001
  ///   linspace(1000 → 1, 4) = [1000, 667, 334, 1] (step = −333 exactly)
  ///   sigmas   = [1.0, 0.667, 0.334, 0.001] (no warp; alpha == 1)
  func testIdentityShiftGridByHand() {
    let config = FlowMatchSchedulerTests.makeConfig(shift: 1.0)
    let sigmas = SigmaSchedule.flow(numSteps: 4, config: config, explicitShift: 1.0)
    let expected: [Float] = [1.0, 0.667, 0.334, 0.001, 0.0]
    XCTAssertEqual(sigmas.count, expected.count)
    for (i, e) in expected.enumerated() {
      XCTAssertEqual(sigmas[i], e, accuracy: 1e-6, "index \(i)")
    }
  }

  /// shift 3.0 — the Zeta Chroma recommendation (issue #154, Lodestone's
  /// workflow tips). Hand-computed from the formula, T = 1000, steps = 4:
  ///
  ///   sigma_min = 3·0.001 / (1 + 2·0.001)      = 0.0029940119760…
  ///   linspace(1000 → 2.994011976, 4)          = [1000, 667.664670…, 335.329341…, 2.994011…]
  ///   sigmas = t/1000                          = [1.0, 0.667664670…, 0.335329341…, 0.002994011…]
  ///   sigma' = 3σ / (1 + 2σ)                   = [1.0, 0.857692307…, 0.602150537…, 0.008928571…]
  ///                                              (= 1, 11.15/13, 56/93, 1/112)
  func testShift3GridByHand() {
    let config = FlowMatchSchedulerTests.makeConfig(shift: 1.0, useDynamicShifting: true,
                                                   baseShift: 0.5, maxShift: 1.15,
                                                   baseImageSeqLen: 256, maxImageSeqLen: 4096)
    let sigmas = SigmaSchedule.flow(numSteps: 4, config: config, mu: 0.8, explicitShift: 3.0)
    let expected: [Float] = [1.0, 0.857692308, 0.602150538, 0.008928571, 0.0]
    XCTAssertEqual(sigmas.count, expected.count)
    for (i, e) in expected.enumerated() {
      XCTAssertEqual(sigmas[i], e, accuracy: 1e-6, "index \(i)")
    }
  }

  /// Monotone decreasing, endpoints preserved, trailing zero — for every
  /// shift a caller may send, at production step counts.
  func testGridInvariantsAcrossShifts() {
    let config = FlowMatchSchedulerTests.makeConfig(shift: 1.0, useDynamicShifting: true,
                                                   baseShift: 0.5, maxShift: 1.15,
                                                   baseImageSeqLen: 256, maxImageSeqLen: 4096)
    for shift in [Float(0.25), 1.0, 1.73, 3.0, 6.0, 12.0] {
      for steps in [4, 9, 20, 28, 50] {
        let sigmas = SigmaSchedule.flow(numSteps: steps, config: config, mu: 0.8,
                                        explicitShift: shift)
        XCTAssertEqual(sigmas.count, steps + 1, "shift \(shift), steps \(steps)")
        // sigma_max is preserved exactly: shift·1 / (1 + (shift − 1)·1) == 1.
        XCTAssertEqual(sigmas[0], 1.0, accuracy: 1e-6, "shift \(shift) moved sigma_max")
        // The trailing sentinel is an exact zero.
        XCTAssertEqual(sigmas[steps], 0.0)
        // Strictly decreasing through the grid, and the last real sigma is > 0.
        XCTAssertGreaterThan(sigmas[steps - 1], 0, "shift \(shift) collapsed sigma_min")
        for i in 0..<(steps - 1) {
          XCTAssertGreaterThan(sigmas[i], sigmas[i + 1],
                               "shift \(shift), steps \(steps): not decreasing at \(i)")
        }
      }
    }
  }

  /// Higher shift holds more noise for longer — the behaviour the issue
  /// describes ("pushes more noise handling to early sampling steps"). Every
  /// interior sigma is strictly greater under shift 3 than under shift 1.
  func testHigherShiftRaisesEveryInteriorSigma() {
    let config = FlowMatchSchedulerTests.makeConfig(shift: 1.0)
    let low = SigmaSchedule.flow(numSteps: 9, config: config, explicitShift: 1.0)
    let high = SigmaSchedule.flow(numSteps: 9, config: config, explicitShift: 3.0)
    XCTAssertEqual(low[0], high[0], accuracy: 1e-6)
    for i in 1..<9 {
      XCTAssertGreaterThan(high[i], low[i], "index \(i)")
    }
  }

  // MARK: - Through the scheduler

  /// The scheduler the euler + flow path actually builds carries the shifted
  /// grid, and its timesteps are the grid × numTrainTimesteps.
  func testFlowMatchEulerSchedulerHonoursExplicitShift() {
    let config = FlowMatchSchedulerTests.makeConfig(shift: 1.0, useDynamicShifting: true,
                                                   baseShift: 0.5, maxShift: 1.15,
                                                   baseImageSeqLen: 256, maxImageSeqLen: 4096)
    let scheduler = FlowMatchEulerScheduler(
      numInferenceSteps: 4, config: config.applyingExplicitShift(3.0), mu: nil)
    let sigmas = scheduler.sigmas.asArray(Float.self)
    let expected: [Float] = [1.0, 0.857692308, 0.602150538, 0.008928571, 0.0]
    for (i, e) in expected.enumerated() {
      XCTAssertEqual(sigmas[i], e, accuracy: 1e-6, "sigma \(i)")
    }
    let timesteps = scheduler.timesteps.asArray(Float.self)
    XCTAssertEqual(timesteps.count, 4)
    for i in 0..<4 {
      XCTAssertEqual(timesteps[i], expected[i] * 1000, accuracy: 1e-3, "timestep \(i)")
    }
  }

  /// `SchedulerFactory` is where the pipelines meet the schedule: a config
  /// patched with an explicit shift produces the same grid through the factory
  /// as through `SigmaSchedule.flow` directly, and does so for the non-flow
  /// schedules too (they index the model's discrete-flow table, which the
  /// patched `shift` rebuilds — ComfyUI's `simple`/`beta` read the same
  /// patched `model_sampling`).
  func testFactoryGridFollowsThePatchedConfig() throws {
    let config = FlowMatchSchedulerTests.makeConfig(shift: 1.0, useDynamicShifting: true,
                                                   baseShift: 0.5, maxShift: 1.15,
                                                   baseImageSeqLen: 256, maxImageSeqLen: 4096)
    let patched = config.applyingExplicitShift(3.0)

    let flow = try SchedulerFactory.resolveSigmas(
      schedule: .flow, numSteps: 4, config: patched, mu: 0.8)
    XCTAssertEqual(flow.map { $0.bitPattern },
                   SigmaSchedule.flow(numSteps: 4, config: config, mu: 0.8, explicitShift: 3.0)
                     .map { $0.bitPattern })

    // `simple` walks the discrete-flow table, which the patch rebuilds at
    // shift 3 — so it differs from the unpatched table's walk.
    let simplePatched = try SchedulerFactory.resolveSigmas(
      schedule: .simple, numSteps: 4, config: patched, mu: 0.8)
    let simpleBase = try SchedulerFactory.resolveSigmas(
      schedule: .simple, numSteps: 4, config: config, mu: 0.8)
    XCTAssertNotEqual(simplePatched, simpleBase)
    XCTAssertEqual(simplePatched.count, 5)
  }
}
