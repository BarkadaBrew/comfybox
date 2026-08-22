import XCTest
import MLX
@testable import ZImage

/// WP-E1 — `SigmaScheduleKind.krea2` and the mu seam (FDD-krea2-raw-recipe §3.1).
///
/// Weight-free. Pins three things:
///   * AC-3  `SigmaSchedule.krea2(numSteps:mu:)` is element-for-element equal
///           (exact `==` on Float) to the pre-change `Krea2Sampling.timesteps`,
///           whose body is inlined below as the oracle.
///   * AC-4  `Krea2Sampling.mu(seqLen:align:16)` equals
///           `PipelineUtilities.calculateShift(…, 256, 6400, 0.5, 1.15)` exactly.
///   * AC-14 the non-flow schedules stay in the flow domain under
///           `Krea2Sampling.schedulerConfig`.
/// plus the factory seam: `.krea2` requires `mu` (no silent unshifted grid)
/// and the synthetic scheduler config's seven values are the ones §3.1 states.
final class Krea2SigmaScheduleTests: XCTestCase {

  /// The matrix named by AC-3 / AC-4.
  static let stepSet = [4, 6, 9, 12, 20, 52]
  static let seqLenSet = [256, 1024, 4096, 6400, 9216]

  /// Krea 2's token-grid alignment: VAE spatial scale (8) × patch (2).
  static let align = 16

  // MARK: - Oracle: the pre-change `Krea2Sampling.timesteps` body, verbatim.
  //
  // Copied from Krea2Pipeline.swift @ 27868e3 (before this WP). Do not
  // "simplify" it — its float operation order is the thing being pinned.

  static func preChangeTimesteps(
    seqLen: Int, steps: Int, x1: Float, x2: Float,
    y1: Float = 0.5, y2: Float = 1.15, sigma: Float = 1.0, mu muOverride: Float? = nil
  ) -> [Float] {
    let slope = (y2 - y1) / (x2 - x1)
    let mu = muOverride ?? (slope * Float(seqLen) + (y1 - slope * x1))
    let expMu = Foundation.exp(mu)
    var out: [Float] = []
    out.reserveCapacity(steps + 1)
    for i in 0...steps {
      let t = 1.0 - Float(i) / Float(steps)  // linspace 1 -> 0
      if t <= 0 {
        out.append(0)
      } else {
        let warped = expMu / (expMu + Foundation.pow(1.0 / t - 1.0, sigma))
        out.append(warped)
      }
    }
    return out
  }

  /// The x1 / x2 the pipelines compute at align 16: (256/16)² and (1280/16)².
  static let x1 = Float((256 / align) * (256 / align))   // 256
  static let x2 = Float((1280 / align) * (1280 / align)) // 6400

  // MARK: - AC-3

  func testMatchesPreChangeOracle() {
    XCTAssertEqual(Self.x1, 256)
    XCTAssertEqual(Self.x2, 6400)
    for seqLen in Self.seqLenSet {
      let mu = Krea2Sampling.mu(seqLen: seqLen, align: Self.align)
      for steps in Self.stepSet {
        let oracle = Self.preChangeTimesteps(
          seqLen: seqLen, steps: steps, x1: Self.x1, x2: Self.x2)
        let schedule = SigmaSchedule.krea2(numSteps: steps, mu: mu)

        XCTAssertEqual(schedule.count, steps + 1,
                       "seqLen=\(seqLen) steps=\(steps): count must be steps+1")
        XCTAssertEqual(oracle.count, schedule.count)
        XCTAssertEqual(schedule.first, 1.0, "seqLen=\(seqLen) steps=\(steps): starts at 1.0")
        XCTAssertEqual(schedule.last, 0.0, "seqLen=\(seqLen) steps=\(steps): trailing exact 0.0")
        // Exact `==` on Float, element for element — no tolerance.
        for i in 0..<oracle.count {
          XCTAssertTrue(
            oracle[i] == schedule[i],
            "seqLen=\(seqLen) steps=\(steps) i=\(i): oracle \(oracle[i]) != krea2 \(schedule[i])")
        }
      }
    }
  }

  /// Delegation is structural: the surviving `Krea2Sampling.timesteps`
  /// signature must still return exactly what `SigmaSchedule.krea2` does.
  func testTimestepsDelegatesToKrea2Schedule() {
    for seqLen in Self.seqLenSet {
      let mu = Krea2Sampling.mu(seqLen: seqLen, align: Self.align)
      for steps in Self.stepSet {
        let viaSampling = Krea2Sampling.timesteps(
          seqLen: seqLen, steps: steps, x1: Self.x1, x2: Self.x2)
        let viaSchedule = SigmaSchedule.krea2(numSteps: steps, mu: mu)
        XCTAssertEqual(viaSampling.count, viaSchedule.count)
        for i in 0..<viaSampling.count {
          XCTAssertTrue(viaSampling[i] == viaSchedule[i],
                        "seqLen=\(seqLen) steps=\(steps) i=\(i)")
        }
        // And the oracle, through the surviving signature.
        let oracle = Self.preChangeTimesteps(
          seqLen: seqLen, steps: steps, x1: Self.x1, x2: Self.x2)
        for i in 0..<oracle.count {
          XCTAssertTrue(oracle[i] == viaSampling[i],
                        "timesteps drifted from oracle at seqLen=\(seqLen) steps=\(steps) i=\(i)")
        }
      }
    }
  }

  /// `mu` override and `sigma` exponent survive delegation unchanged.
  func testTimestepsHonoursMuOverrideAndSigmaExponent() {
    let steps = 9
    let oracle = Self.preChangeTimesteps(
      seqLen: 4096, steps: steps, x1: Self.x1, x2: Self.x2, sigma: 1.5, mu: 0.25)
    let viaSampling = Krea2Sampling.timesteps(
      seqLen: 4096, steps: steps, x1: Self.x1, x2: Self.x2, sigma: 1.5, mu: 0.25)
    let viaSchedule = SigmaSchedule.krea2(numSteps: steps, mu: 0.25, sigmaExp: 1.5)
    XCTAssertEqual(oracle.count, steps + 1)
    for i in 0..<oracle.count {
      XCTAssertTrue(oracle[i] == viaSampling[i], "i=\(i)")
      XCTAssertTrue(oracle[i] == viaSchedule[i], "i=\(i)")
    }
  }

  // MARK: - AC-4

  func testMuMatchesCalculateShift() {
    for seqLen in Self.seqLenSet {
      let mu = Krea2Sampling.mu(seqLen: seqLen, align: Self.align)
      let reference = PipelineUtilities.calculateShift(
        imageSeqLen: seqLen, baseSeqLen: 256, maxSeqLen: 6400,
        baseShift: 0.5, maxShift: 1.15)
      XCTAssertTrue(mu == reference,
                    "seqLen=\(seqLen): mu \(mu) != calculateShift \(reference)")

      // And against the pre-change inline slope formula — the value the
      // pipelines have always used — so the seam cannot drift either way.
      let slope = (Float(1.15) - Float(0.5)) / (Self.x2 - Self.x1)
      let inline = slope * Float(seqLen) + (Float(0.5) - slope * Self.x1)
      XCTAssertTrue(mu == inline,
                    "seqLen=\(seqLen): mu \(mu) != pre-change inline \(inline)")
    }
    // Anchor values: 256 tokens → 0.5, 6400 tokens → 1.15, 4096 (1024²) → ~0.906.
    XCTAssertEqual(Krea2Sampling.mu(seqLen: 256, align: 16), 0.5, accuracy: 1e-6)
    XCTAssertEqual(Krea2Sampling.mu(seqLen: 6400, align: 16), 1.15, accuracy: 1e-6)
    XCTAssertEqual(Krea2Sampling.mu(seqLen: 4096, align: 16), 0.9062, accuracy: 1e-4)
  }

  // MARK: - §3.1 synthetic scheduler config (pinned values)

  /// The eight values §3.1 (as amended by A.1) specifies. There is no
  /// `shift:` parameter any more: Krea 2's shift is `mu`, carried beside the
  /// config by `ScheduleShift`, and the table-backed schedules build the Flux
  /// table from it — `config.shift` stays 1.0 and is not where the shift lives.
  func testSchedulerConfigPinnedValues() {
    let config = Krea2Sampling.schedulerConfig()
    XCTAssertEqual(config.numTrainTimesteps, 1000)
    XCTAssertEqual(config.shift, 1.0)
    XCTAssertTrue(config.useDynamicShifting)
    XCTAssertEqual(config.baseShift, 0.5)
    XCTAssertEqual(config.maxShift, 1.15)
    XCTAssertEqual(config.baseImageSeqLen, 256)
    XCTAssertEqual(config.maxImageSeqLen, 6400)
    XCTAssertEqual(config.modelSampling, .flux(tableSize: 10000), "ComfyUI registers Krea 2 as ModelSamplingFlux (A.1)")
    XCTAssertEqual(Krea2Sampling.fluxTableSize, 10000)
  }

  /// A config decoded from a `scheduler_config.json` (every non-Krea-2 family)
  /// is `.discreteFlow` — the key is not on the wire and the default is the
  /// table those families always used.
  func testDecodedSchedulerConfigIsDiscreteFlow() throws {
    let decoded = FlowMatchSchedulerTests.makeConfig(
      numTrainTimesteps: 1000, shift: 3.0, useDynamicShifting: true,
      baseShift: 0.5, maxShift: 1.15, baseImageSeqLen: 256, maxImageSeqLen: 4096)
    XCTAssertEqual(decoded.modelSampling, .discreteFlow)
    let minimal = try JSONDecoder().decode(
      ZImageSchedulerConfig.self,
      from: Data(#"{"num_train_timesteps": 1000, "shift": 3.0, "use_dynamic_shifting": false}"#.utf8))
    XCTAssertEqual(minimal.modelSampling, .discreteFlow)
    XCTAssertEqual(minimal.shift, 3.0)
    XCTAssertNil(minimal.baseShift)
    // The memberwise default is the same.
    XCTAssertEqual(ZImageSchedulerConfig(numTrainTimesteps: 1000, shift: 1.0, useDynamicShifting: false).modelSampling, .discreteFlow)
  }

  /// `ZImageSchedulerConfig` now has a public memberwise init; it must agree
  /// with the JSON-decoded form the rest of the suite builds.
  func testSchedulerConfigMemberwiseInitMatchesDecoded() {
    let decoded = FlowMatchSchedulerTests.makeConfig(
      numTrainTimesteps: 1000, shift: 1.15, useDynamicShifting: true,
      baseShift: 0.5, maxShift: 1.15, baseImageSeqLen: 256, maxImageSeqLen: 6400)
    let direct = ZImageSchedulerConfig(
      numTrainTimesteps: 1000, shift: 1.15, useDynamicShifting: true,
      baseShift: 0.5, maxShift: 1.15, baseImageSeqLen: 256, maxImageSeqLen: 6400)
    XCTAssertEqual(decoded.numTrainTimesteps, direct.numTrainTimesteps)
    XCTAssertEqual(decoded.shift, direct.shift)
    XCTAssertEqual(decoded.useDynamicShifting, direct.useDynamicShifting)
    XCTAssertEqual(decoded.baseShift, direct.baseShift)
    XCTAssertEqual(decoded.maxShift, direct.maxShift)
    XCTAssertEqual(decoded.baseImageSeqLen, direct.baseImageSeqLen)
    XCTAssertEqual(decoded.maxImageSeqLen, direct.maxImageSeqLen)

    let minimal = ZImageSchedulerConfig(numTrainTimesteps: 1000, shift: 3.0, useDynamicShifting: false)
    XCTAssertNil(minimal.baseShift)
    XCTAssertNil(minimal.maxShift)
    XCTAssertNil(minimal.baseImageSeqLen)
    XCTAssertNil(minimal.maxImageSeqLen)
  }

  // MARK: - AC-14

  func testNoEDMLeakage() throws {
    let schedules: [SigmaScheduleKind] = [.karras, .exponential, .beta, .beta57]
    let config = Krea2Sampling.schedulerConfig()
    // Dynamic mu at 1024² and the published shift 1.15 (A.1: shift IS mu).
    let mus: [(String, Float)] = [
      ("dynamic", Krea2Sampling.mu(seqLen: 4096, align: Self.align)),
      ("shift 1.15", 1.15),
    ]
    for (label, mu) in mus {
      for schedule in schedules {
        for steps in Self.stepSet {
          let sigmas = try SchedulerFactory.resolveSigmas(
            schedule: schedule, numSteps: steps, config: config, mu: mu)
          XCTAssertEqual(sigmas.count, steps + 1,
                         "\(schedule.rawValue)/\(label)/\(steps): count")
          XCTAssertEqual(sigmas[0], 1.0,
                         "\(schedule.rawValue)/\(label)/\(steps): sigmas[0] must be exactly 1.0")
          XCTAssertEqual(sigmas.last, 0.0,
                         "\(schedule.rawValue)/\(label)/\(steps): trailing 0.0 sentinel")
          for (i, s) in sigmas.enumerated() {
            XCTAssertTrue(s >= 0.0 && s <= 1.0,
                          "\(schedule.rawValue)/\(label)/\(steps) i=\(i): sigma \(s) left the flow domain [0,1]")
          }
        }
      }
    }
  }

  /// The karras/exponential bounds under the Krea 2 family are the Flux
  /// table's ends, `(e^mu / (e^mu + 9999), 1.0)` — ComfyUI's
  /// `model_sampling.sigma_min/sigma_max` under `ModelSamplingFlux` — so they
  /// move with `mu` exactly as `beta` does (D3 as amended). Read through the
  /// lowest non-zero karras sigma, which is the schedule's `sigmaMin` exactly
  /// (t = 1 at the last grid point). At shift 1.15 that is the fixture's
  /// `model_samplings.flux.sigma_min`; at 1024² (mu 0.90625) it is 2.4747e-4.
  func testExplicitShiftMovesNonFlowBounds() throws {
    let steps = 9
    let config = Krea2Sampling.schedulerConfig()
    let dynamicMu = Krea2Sampling.mu(seqLen: 4096, align: Self.align)
    let dynamic = try SchedulerFactory.resolveSigmas(
      schedule: .karras, numSteps: steps, config: config, mu: dynamicMu)
    let explicit = try SchedulerFactory.resolveSigmas(
      schedule: .karras, numSteps: steps, config: config, mu: 1.15)
    let fixture = try SchedulerOracleFixtures.json("comfy_sigmas.json")
    let fluxMin = try XCTUnwrap(
      ((fixture["model_samplings"] as? [String: Any])?["flux"] as? [String: Any])?["sigma_min"] as? Double)
    XCTAssertEqual(Double(explicit[steps - 1]), fluxMin, accuracy: 1e-9, "shift 1.15 → ModelSamplingFlux σ_min 3.1575e-4")
    XCTAssertEqual(Double(dynamic[steps - 1]), exp(0.90625) / (exp(0.90625) + 9999.0), accuracy: 1e-9)
    XCTAssertNotEqual(dynamic[steps - 1], explicit[steps - 1])
    // Neither is the DiscreteFlow bound E12 used (1.15e-3 / (1 + 0.15e-3) = 0.00114983) or the unshifted 0.001.
    XCTAssertNotEqual(explicit[steps - 1], 0.00114983, accuracy: 1e-5)
    XCTAssertNotEqual(dynamic[steps - 1], 0.001, accuracy: 1e-5)
    // exponential shares the bounds.
    let expo = try SchedulerFactory.resolveSigmas(
      schedule: .exponential, numSteps: steps, config: config, mu: 1.15)
    XCTAssertEqual(Double(expo[steps - 1]), fluxMin, accuracy: 1e-9)
  }

  // MARK: - Factory seam

  func testSigmaScheduleKindHasKrea2() {
    XCTAssertEqual(SigmaScheduleKind(rawValue: "krea2"), .krea2)
    XCTAssertTrue(SigmaScheduleKind.allCases.contains(.krea2))
    XCTAssertEqual(SigmaScheduleKind.krea2.rawValue, "krea2")
  }

  /// `mu` is required for `.krea2` — `mu ?? 0` would silently be an
  /// unshifted linear grid, the one silent default this programme forbids.
  func testFactoryKrea2RequiresMu() {
    let config = Krea2Sampling.schedulerConfig()
    XCTAssertThrowsError(
      try SchedulerFactory.resolveSigmas(schedule: .krea2, numSteps: 9, config: config, mu: nil)
    ) { error in
      XCTAssertEqual(error as? SchedulerFactoryError, .missingMu(.krea2))
    }
    XCTAssertThrowsError(
      try SchedulerFactory.create(
        kind: .euler, sigmaSchedule: .krea2, numInferenceSteps: 9, config: config, mu: nil)
    ) { error in
      XCTAssertEqual(error as? SchedulerFactoryError, .missingMu(.krea2))
    }
    // The error names the schedule, so a 400 built from it is self-describing.
    XCTAssertTrue(String(describing: SchedulerFactoryError.missingMu(.krea2)).contains("krea2"))
  }

  /// The factory's `.krea2` grid is `SigmaSchedule.krea2` and reaches the
  /// scheduler's float32 `sigmas` untouched.
  func testFactoryKrea2SigmasMatchSchedule() throws {
    let config = Krea2Sampling.schedulerConfig()
    for seqLen in Self.seqLenSet {
      let mu = Krea2Sampling.mu(seqLen: seqLen, align: Self.align)
      for steps in [6, 9, 20] {
        let expected = SigmaSchedule.krea2(numSteps: steps, mu: mu)

        let resolved = try SchedulerFactory.resolveSigmas(
          schedule: .krea2, numSteps: steps, config: config, mu: mu)
        XCTAssertEqual(resolved, expected, "resolveSigmas seqLen=\(seqLen) steps=\(steps)")

        let scheduler = try SchedulerFactory.create(
          kind: .euler, sigmaSchedule: .krea2, numInferenceSteps: steps, config: config, mu: mu)
        XCTAssertEqual(scheduler.numInferenceSteps, steps)
        let sigmas = scheduler.sigmas.asArray(Float.self)
        XCTAssertEqual(sigmas.count, steps + 1)
        for i in 0..<expected.count {
          XCTAssertTrue(sigmas[i] == expected[i],
                        "seqLen=\(seqLen) steps=\(steps) i=\(i): scheduler sigma \(sigmas[i]) != \(expected[i])")
        }
      }
    }
  }

  /// `.flow` and `.krea2` are not synonyms (§3.1): same warp, different base
  /// grid — `.flow` runs `linspace(1 → shiftedSigmaMin)` + sentinel, `.krea2`
  /// runs `linspace(1 → 0)` inclusive. Count, head and tail agree; the
  /// penultimate sigma does not.
  func testKrea2IsNotFlow() throws {
    let config = Krea2Sampling.schedulerConfig()
    let mu = Krea2Sampling.mu(seqLen: 4096, align: Self.align)
    let steps = 9
    let flow = try SchedulerFactory.resolveSigmas(
      schedule: .flow, numSteps: steps, config: config, mu: mu)
    let krea2 = try SchedulerFactory.resolveSigmas(
      schedule: .krea2, numSteps: steps, config: config, mu: mu)
    XCTAssertEqual(flow.count, krea2.count)
    XCTAssertEqual(flow[0], 1.0)
    XCTAssertEqual(krea2[0], 1.0)
    XCTAssertEqual(flow.last, 0.0)
    XCTAssertEqual(krea2.last, 0.0)
    XCTAssertNotEqual(flow[steps - 1], krea2[steps - 1],
                      "flow and krea2 must differ at the penultimate sigma")
    // Every interior point differs, because the base grids differ everywhere but the endpoints.
    for i in 1..<steps {
      XCTAssertNotEqual(flow[i], krea2[i], "i=\(i)")
    }
  }

  /// The `kind == .euler && schedule == .flow` fast path is untouched (Z-Image
  /// bit-unaffected): the factory still returns the config-built scheduler,
  /// whose sigmas equal the direct construction exactly.
  func testEulerFlowFastPathUnchanged() throws {
    let config = FlowMatchSchedulerTests.makeConfig(
      useDynamicShifting: true, baseShift: 0.5, maxShift: 1.15,
      baseImageSeqLen: 256, maxImageSeqLen: 4096)
    let mu: Float = 0.8
    let direct = FlowMatchEulerScheduler(numInferenceSteps: 9, config: config, mu: mu)
    let factory = try SchedulerFactory.create(
      kind: .euler, sigmaSchedule: .flow, numInferenceSteps: 9, config: config, mu: mu)
    XCTAssertEqual(direct.sigmas.asArray(Float.self), factory.sigmas.asArray(Float.self))
    XCTAssertEqual(direct.timesteps.asArray(Float.self), factory.timesteps.asArray(Float.self))
  }
}
