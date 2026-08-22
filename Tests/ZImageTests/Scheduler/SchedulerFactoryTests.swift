import XCTest
import MLX
@testable import ZImage

final class SchedulerFactoryTests: XCTestCase {

  // MARK: - Factory Creation

  func testEulerFlowIdenticalToLegacy() throws {
    // Factory-created Euler + flow must produce identical sigmas/timesteps
    // to direct FlowMatchEulerScheduler construction.
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let direct = FlowMatchEulerScheduler(numInferenceSteps: steps, config: config)
    let factory = try SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .flow,
      numInferenceSteps: steps,
      config: config
    )

    let directSigmas = direct.sigmas.asArray(Float.self)
    let factorySigmas = factory.sigmas.asArray(Float.self)
    XCTAssertEqual(directSigmas.count, factorySigmas.count)
    for i in 0..<directSigmas.count {
      XCTAssertEqual(directSigmas[i], factorySigmas[i], accuracy: 1e-6,
                     "Sigma mismatch at index \(i)")
    }

    let directTimesteps = direct.timesteps.asArray(Float.self)
    let factoryTimesteps = factory.timesteps.asArray(Float.self)
    XCTAssertEqual(directTimesteps.count, factoryTimesteps.count)
    for i in 0..<directTimesteps.count {
      XCTAssertEqual(directTimesteps[i], factoryTimesteps[i], accuracy: 1e-4,
                     "Timestep mismatch at index \(i)")
    }
  }

  func testEulerFlowWithDynamicShiftingIdenticalToLegacy() throws {
    let config = FlowMatchSchedulerTests.makeConfig(
      useDynamicShifting: true,
      baseShift: 0.5,
      maxShift: 1.15,
      baseImageSeqLen: 256,
      maxImageSeqLen: 4096
    )
    let mu: Float = 0.8
    let steps = 9

    let direct = FlowMatchEulerScheduler(numInferenceSteps: steps, config: config, mu: mu)
    let factory = try SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .flow,
      numInferenceSteps: steps,
      config: config,
      mu: mu
    )

    let directSigmas = direct.sigmas.asArray(Float.self)
    let factorySigmas = factory.sigmas.asArray(Float.self)
    XCTAssertEqual(directSigmas.count, factorySigmas.count)
    for i in 0..<directSigmas.count {
      XCTAssertEqual(directSigmas[i], factorySigmas[i], accuracy: 1e-6)
    }
  }

  func testEulerWithKarrasSchedule() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let scheduler = try SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .karras,
      numInferenceSteps: steps,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, steps)
    XCTAssertEqual(scheduler.sigmas.dim(0), steps + 1)
    XCTAssertEqual(scheduler.timesteps.dim(0), steps)

    let sigmas = scheduler.sigmas.asArray(Float.self)
    // With flow-matching bounds, sigmas are in [0, 1] range.
    XCTAssertEqual(sigmas[0], 1.0, accuracy: 1e-3,
                   "Karras first sigma should be ~1.0 (flow-matching sigmaMax)")
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  func testEulerWithExponentialSchedule() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let scheduler = try SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .exponential,
      numInferenceSteps: steps,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, steps)
    let sigmas = scheduler.sigmas.asArray(Float.self)
    XCTAssertEqual(sigmas[0], 1.0, accuracy: 1e-3,
                   "Exponential first sigma should be ~1.0 (flow-matching sigmaMax)")
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  func testEulerWithBetaSchedule() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let scheduler = try SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .beta,
      numInferenceSteps: steps,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, steps)
    let sigmas = scheduler.sigmas.asArray(Float.self)
    for i in 0..<(sigmas.count - 1) {
      XCTAssertGreaterThan(sigmas[i], 0.0)
    }
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  // MARK: - Enum Parsing

  func testUnknownSchedulerKindFromString() throws {
    XCTAssertNil(SchedulerKind(rawValue: "bogus"))
    XCTAssertNil(SchedulerKind(rawValue: ""))
    XCTAssertNil(SchedulerKind(rawValue: "Euler"))  // case-sensitive
  }

  func testValidSchedulerKindFromString() throws {
    XCTAssertEqual(SchedulerKind(rawValue: "euler"), .euler)
    XCTAssertEqual(SchedulerKind(rawValue: "heun"), .heun)
    XCTAssertEqual(SchedulerKind(rawValue: "dpmpp-2m"), .dpmplusplus2m)
    XCTAssertEqual(SchedulerKind(rawValue: "dpmpp-2s-a"), .dpmplusplus2sa)
    XCTAssertEqual(SchedulerKind(rawValue: "deis"), .deis)
    XCTAssertEqual(SchedulerKind(rawValue: "ddim"), .ddim)
    XCTAssertEqual(SchedulerKind(rawValue: "res_2s"), .res2s)
  }

  func testUnknownSigmaScheduleKindFromString() throws {
    XCTAssertNil(SigmaScheduleKind(rawValue: "bogus"))
    XCTAssertNil(SigmaScheduleKind(rawValue: ""))
  }

  func testValidSigmaScheduleKindFromString() throws {
    XCTAssertEqual(SigmaScheduleKind(rawValue: "flow"), .flow)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "karras"), .karras)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "exponential"), .exponential)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "beta"), .beta)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "beta57"), .beta57)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "krea2"), .krea2)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "bong_tangent"), .bongTangent)
  }

  // MARK: - All Kinds

  func testAllSchedulerKindsCreateSuccessfully() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    for kind in SchedulerKind.allCases {
      let scheduler = try SchedulerFactory.create(
        kind: kind,
        numInferenceSteps: 9,
        config: config
      )
      XCTAssertEqual(scheduler.numInferenceSteps, 9,
                     "Scheduler \(kind.rawValue) should have 9 steps")
      XCTAssertEqual(scheduler.sigmas.dim(0), 10,
                     "Scheduler \(kind.rawValue) should have 10 sigma values")
    }
  }

  func testAllSigmaScheduleKindsCreateSuccessfully() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    for scheduleKind in SigmaScheduleKind.allCases {
      // `.krea2` is defined by mu and throws without it (WP-E1); the other
      // schedules ignore mu under this non-dynamic-shifting config.
      let scheduler = try SchedulerFactory.create(
        kind: .euler,
        sigmaSchedule: scheduleKind,
        numInferenceSteps: 9,
        config: config,
        mu: 0.9
      )
      XCTAssertEqual(scheduler.numInferenceSteps, 9,
                     "Sigma schedule \(scheduleKind.rawValue) should produce 9 steps")
    }
  }

  // MARK: - Phase 2 Sampler Creation

  func testDPMPlusPlus2MCreation() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .dpmplusplus2m,
      numInferenceSteps: 9,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)
    XCTAssertFalse(scheduler.requiresIntermediateEvaluation)
  }

  func testDDIMCreation() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .ddim,
      numInferenceSteps: 9,
      config: config,
      eta: 0.5
    )

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)
    XCTAssertFalse(scheduler.requiresIntermediateEvaluation)
  }

  func testDDIMWithSeed() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .ddim,
      numInferenceSteps: 9,
      config: config,
      seed: 42,
      eta: 1.0
    )

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
  }

  func testDEISCreation() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .deis,
      numInferenceSteps: 9,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)
    XCTAssertFalse(scheduler.requiresIntermediateEvaluation)
  }

  func testDPMPlusPlus2SACreation() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .dpmplusplus2sa,
      numInferenceSteps: 9,
      config: config,
      seed: 42
    )

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)
    XCTAssertFalse(scheduler.requiresIntermediateEvaluation)
  }

  func testHeunCreation() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .heun,
      numInferenceSteps: 9,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)
    XCTAssertTrue(scheduler.requiresIntermediateEvaluation)
  }

  func testRES2sCreation() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .res2s,
      numInferenceSteps: 9,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)
    XCTAssertTrue(scheduler.requiresIntermediateEvaluation)
  }

  func testBeta57ScheduleCreation() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .beta57,
      numInferenceSteps: 9,
      config: config
    )
    let defaultBeta = try SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .beta,
      numInferenceSteps: 9,
      config: config
    )

    let beta57Sigmas = scheduler.sigmas.asArray(Float.self)
    let defaultBetaSigmas = defaultBeta.sigmas.asArray(Float.self)
    XCTAssertEqual(beta57Sigmas.count, 10)
    XCTAssertEqual(beta57Sigmas.last!, 0.0, accuracy: 1e-10)

    var differs = false
    for i in 0..<beta57Sigmas.count {
      if abs(beta57Sigmas[i] - defaultBetaSigmas[i]) > 1e-6 {
        differs = true
        break
      }
    }
    XCTAssertTrue(differs, "Factory beta57 should differ from default beta")
  }

  func testPhase2SamplersWithKarrasSchedule() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let phase2Kinds: [SchedulerKind] = [
      .dpmplusplus2m, .ddim, .deis, .dpmplusplus2sa, .heun, .res2s
    ]

    for kind in phase2Kinds {
      let scheduler = try SchedulerFactory.create(
        kind: kind,
        sigmaSchedule: .karras,
        numInferenceSteps: 9,
        config: config,
        seed: 42
      )
      XCTAssertEqual(scheduler.numInferenceSteps, 9,
                     "Scheduler \(kind.rawValue) with Karras should have 9 steps")
      let sigmas = scheduler.sigmas.asArray(Float.self)
      XCTAssertEqual(sigmas[0], 1.0, accuracy: 1e-3,
                     "Karras first sigma should be ~1.0 for \(kind.rawValue)")
      XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10,
                     "Last sigma should be 0 for \(kind.rawValue)")
    }
  }

  // MARK: - Protocol Conformance

  func testFactoryReturnsZImageScheduler() throws {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = try SchedulerFactory.create(
      kind: .euler,
      numInferenceSteps: 9,
      config: config
    )

    // Verify the returned value conforms to ZImageScheduler
    XCTAssertFalse(scheduler.requiresIntermediateEvaluation)
  }

  func testFactoryDefaultSigmaScheduleIsFlow() throws {
    // When sigmaSchedule is not specified, it should default to .flow
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let withDefault = try SchedulerFactory.create(
      kind: .euler,
      numInferenceSteps: steps,
      config: config
    )
    let withExplicitFlow = try SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .flow,
      numInferenceSteps: steps,
      config: config
    )

    let defaultSigmas = withDefault.sigmas.asArray(Float.self)
    let flowSigmas = withExplicitFlow.sigmas.asArray(Float.self)
    XCTAssertEqual(defaultSigmas.count, flowSigmas.count)
    for i in 0..<defaultSigmas.count {
      XCTAssertEqual(defaultSigmas[i], flowSigmas[i], accuracy: 1e-6)
    }
  }
}
