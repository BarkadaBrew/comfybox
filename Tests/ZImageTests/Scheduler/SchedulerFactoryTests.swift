import XCTest
import MLX
@testable import ZImage

final class SchedulerFactoryTests: XCTestCase {

  // MARK: - Factory Creation

  func testEulerFlowIdenticalToLegacy() {
    // Factory-created Euler + flow must produce identical sigmas/timesteps
    // to direct FlowMatchEulerScheduler construction.
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let direct = FlowMatchEulerScheduler(numInferenceSteps: steps, config: config)
    let factory = SchedulerFactory.create(
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

  func testEulerFlowWithDynamicShiftingIdenticalToLegacy() {
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
    let factory = SchedulerFactory.create(
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

  func testEulerWithKarrasSchedule() {
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let scheduler = SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .karras,
      numInferenceSteps: steps,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, steps)
    XCTAssertEqual(scheduler.sigmas.dim(0), steps + 1)
    XCTAssertEqual(scheduler.timesteps.dim(0), steps)

    let sigmas = scheduler.sigmas.asArray(Float.self)
    // Karras sigmas are much larger than flow sigmas (sigmaMax=100 vs 1.0)
    XCTAssertGreaterThan(sigmas[0], 10.0)
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  func testEulerWithExponentialSchedule() {
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let scheduler = SchedulerFactory.create(
      kind: .euler,
      sigmaSchedule: .exponential,
      numInferenceSteps: steps,
      config: config
    )

    XCTAssertEqual(scheduler.numInferenceSteps, steps)
    let sigmas = scheduler.sigmas.asArray(Float.self)
    XCTAssertGreaterThan(sigmas[0], 10.0)
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  func testEulerWithBetaSchedule() {
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let scheduler = SchedulerFactory.create(
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

  func testUnknownSchedulerKindFromString() {
    XCTAssertNil(SchedulerKind(rawValue: "bogus"))
    XCTAssertNil(SchedulerKind(rawValue: ""))
    XCTAssertNil(SchedulerKind(rawValue: "Euler"))  // case-sensitive
  }

  func testValidSchedulerKindFromString() {
    XCTAssertEqual(SchedulerKind(rawValue: "euler"), .euler)
  }

  func testUnknownSigmaScheduleKindFromString() {
    XCTAssertNil(SigmaScheduleKind(rawValue: "bogus"))
    XCTAssertNil(SigmaScheduleKind(rawValue: ""))
  }

  func testValidSigmaScheduleKindFromString() {
    XCTAssertEqual(SigmaScheduleKind(rawValue: "flow"), .flow)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "karras"), .karras)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "exponential"), .exponential)
    XCTAssertEqual(SigmaScheduleKind(rawValue: "beta"), .beta)
  }

  // MARK: - All Kinds

  func testAllSchedulerKindsCreateSuccessfully() {
    let config = FlowMatchSchedulerTests.makeConfig()
    for kind in SchedulerKind.allCases {
      let scheduler = SchedulerFactory.create(
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

  func testAllSigmaScheduleKindsCreateSuccessfully() {
    let config = FlowMatchSchedulerTests.makeConfig()
    for scheduleKind in SigmaScheduleKind.allCases {
      let scheduler = SchedulerFactory.create(
        kind: .euler,
        sigmaSchedule: scheduleKind,
        numInferenceSteps: 9,
        config: config
      )
      XCTAssertEqual(scheduler.numInferenceSteps, 9,
                     "Sigma schedule \(scheduleKind.rawValue) should produce 9 steps")
    }
  }

  // MARK: - Protocol Conformance

  func testFactoryReturnsZImageScheduler() {
    let config = FlowMatchSchedulerTests.makeConfig()
    let scheduler = SchedulerFactory.create(
      kind: .euler,
      numInferenceSteps: 9,
      config: config
    )

    // Verify the returned value conforms to ZImageScheduler
    XCTAssertFalse(scheduler.requiresIntermediateEvaluation)
  }

  func testFactoryDefaultSigmaScheduleIsFlow() {
    // When sigmaSchedule is not specified, it should default to .flow
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let withDefault = SchedulerFactory.create(
      kind: .euler,
      numInferenceSteps: steps,
      config: config
    )
    let withExplicitFlow = SchedulerFactory.create(
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
