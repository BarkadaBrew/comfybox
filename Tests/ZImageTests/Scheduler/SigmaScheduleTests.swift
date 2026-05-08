import XCTest
@testable import ZImage

final class SigmaScheduleTests: XCTestCase {

  // MARK: - Flow Schedule

  func testFlowScheduleMatchesLegacy() {
    // SigmaSchedule.flow() must produce identical values to the original
    // FlowMatchEulerScheduler sigma computation.
    let config = FlowMatchSchedulerTests.makeConfig()
    let steps = 9

    let sigmaValues = SigmaSchedule.flow(numSteps: steps, config: config)
    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: steps, config: config)
    let schedulerSigmas = scheduler.sigmas.asArray(Float.self)

    XCTAssertEqual(sigmaValues.count, schedulerSigmas.count)
    for i in 0..<sigmaValues.count {
      XCTAssertEqual(sigmaValues[i], schedulerSigmas[i], accuracy: 1e-6,
                     "Sigma mismatch at index \(i)")
    }
  }

  func testFlowScheduleWithShift() {
    let config = FlowMatchSchedulerTests.makeConfig(shift: 3.0)
    let steps = 9

    let sigmaValues = SigmaSchedule.flow(numSteps: steps, config: config)
    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: steps, config: config)
    let schedulerSigmas = scheduler.sigmas.asArray(Float.self)

    XCTAssertEqual(sigmaValues.count, schedulerSigmas.count)
    for i in 0..<sigmaValues.count {
      XCTAssertEqual(sigmaValues[i], schedulerSigmas[i], accuracy: 1e-6,
                     "Sigma mismatch at index \(i) with shift=3.0")
    }
  }

  func testFlowScheduleWithDynamicShifting() {
    let config = FlowMatchSchedulerTests.makeConfig(
      useDynamicShifting: true,
      baseShift: 0.5,
      maxShift: 1.15,
      baseImageSeqLen: 256,
      maxImageSeqLen: 4096
    )
    let mu: Float = 0.8
    let steps = 9

    let sigmaValues = SigmaSchedule.flow(numSteps: steps, config: config, mu: mu)
    let scheduler = FlowMatchEulerScheduler(numInferenceSteps: steps, config: config, mu: mu)
    let schedulerSigmas = scheduler.sigmas.asArray(Float.self)

    XCTAssertEqual(sigmaValues.count, schedulerSigmas.count)
    for i in 0..<sigmaValues.count {
      XCTAssertEqual(sigmaValues[i], schedulerSigmas[i], accuracy: 1e-6,
                     "Sigma mismatch at index \(i) with dynamic shifting")
    }
  }

  func testFlowScheduleCount() {
    let config = FlowMatchSchedulerTests.makeConfig()
    for steps in [1, 4, 9, 20, 50] {
      let sigmas = SigmaSchedule.flow(numSteps: steps, config: config)
      XCTAssertEqual(sigmas.count, steps + 1,
                     "Expected \(steps + 1) sigma values for \(steps) steps")
    }
  }

  func testFlowScheduleTerminalZero() {
    let config = FlowMatchSchedulerTests.makeConfig()
    let sigmas = SigmaSchedule.flow(numSteps: 9, config: config)
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  // MARK: - Karras Schedule

  func testKarrasMonotonicallyDecreasing() {
    let sigmas = SigmaSchedule.karras(numSteps: 20)
    for i in 1..<sigmas.count {
      XCTAssertLessThanOrEqual(sigmas[i], sigmas[i - 1],
                               "Karras sigmas should decrease monotonically at index \(i)")
    }
  }

  func testKarrasCount() {
    for steps in [1, 4, 9, 20, 50] {
      let sigmas = SigmaSchedule.karras(numSteps: steps)
      XCTAssertEqual(sigmas.count, steps + 1,
                     "Expected \(steps + 1) sigma values for \(steps) steps")
    }
  }

  func testKarrasBounds() {
    let sigmaMax: Float = 100.0
    let sigmas = SigmaSchedule.karras(numSteps: 20, sigmaMin: 0.02, sigmaMax: sigmaMax)
    XCTAssertEqual(sigmas.first!, sigmaMax, accuracy: 1e-4,
                   "First Karras sigma should be near sigmaMax")
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10,
                   "Last Karras sigma should be 0.0")
  }

  func testKarrasTerminalZero() {
    let sigmas = SigmaSchedule.karras(numSteps: 10)
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  func testKarrasSingleStep() {
    let sigmas = SigmaSchedule.karras(numSteps: 1)
    XCTAssertEqual(sigmas.count, 2)
    XCTAssertGreaterThan(sigmas[0], 0.0)
    XCTAssertEqual(sigmas[1], 0.0, accuracy: 1e-10)
  }

  func testKarrasZeroSteps() {
    let sigmas = SigmaSchedule.karras(numSteps: 0)
    XCTAssertEqual(sigmas, [0.0])
  }

  // MARK: - Exponential Schedule

  func testExponentialMonotonicallyDecreasing() {
    let sigmas = SigmaSchedule.exponential(numSteps: 20)
    for i in 1..<sigmas.count {
      XCTAssertLessThanOrEqual(sigmas[i], sigmas[i - 1],
                               "Exponential sigmas should decrease monotonically at index \(i)")
    }
  }

  func testExponentialBounds() {
    let sigmaMax: Float = 100.0
    let sigmaMin: Float = 0.02
    let sigmas = SigmaSchedule.exponential(numSteps: 20, sigmaMin: sigmaMin, sigmaMax: sigmaMax)
    XCTAssertEqual(sigmas.first!, sigmaMax, accuracy: 1e-4,
                   "First exponential sigma should be near sigmaMax")
    XCTAssertEqual(sigmas[sigmas.count - 2], sigmaMin, accuracy: 1e-4,
                   "Second-to-last exponential sigma should be near sigmaMin")
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  func testExponentialCount() {
    for steps in [1, 4, 9, 20, 50] {
      let sigmas = SigmaSchedule.exponential(numSteps: steps)
      XCTAssertEqual(sigmas.count, steps + 1)
    }
  }

  func testExponentialSingleStep() {
    let sigmas = SigmaSchedule.exponential(numSteps: 1)
    XCTAssertEqual(sigmas.count, 2)
    XCTAssertGreaterThan(sigmas[0], 0.0)
    XCTAssertEqual(sigmas[1], 0.0, accuracy: 1e-10)
  }

  func testExponentialZeroSteps() {
    let sigmas = SigmaSchedule.exponential(numSteps: 0)
    XCTAssertEqual(sigmas, [0.0])
  }

  // MARK: - Beta Schedule

  func testBetaBounds() {
    let sigmas = SigmaSchedule.beta(numSteps: 20, sigmaMin: 0.02, sigmaMax: 100.0)
    // Beta schedule values should all be positive (except trailing 0)
    for i in 0..<(sigmas.count - 1) {
      XCTAssertGreaterThan(sigmas[i], 0.0, "Beta sigma at index \(i) should be positive")
    }
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  func testBetaCount() {
    for steps in [1, 4, 9, 20, 50] {
      let sigmas = SigmaSchedule.beta(numSteps: steps)
      XCTAssertEqual(sigmas.count, steps + 1)
    }
  }

  func testBetaSingleStep() {
    let sigmas = SigmaSchedule.beta(numSteps: 1)
    XCTAssertEqual(sigmas.count, 2)
    XCTAssertGreaterThan(sigmas[0], 0.0)
    XCTAssertEqual(sigmas[1], 0.0, accuracy: 1e-10)
  }

  func testBetaZeroSteps() {
    let sigmas = SigmaSchedule.beta(numSteps: 0)
    XCTAssertEqual(sigmas, [0.0])
  }

  func testBetaMonotonicallyDecreasing() {
    let sigmas = SigmaSchedule.beta(numSteps: 20, sigmaMin: 0.02, sigmaMax: 100.0)
    for i in 1..<(sigmas.count - 1) {
      XCTAssertLessThanOrEqual(
        sigmas[i], sigmas[i - 1],
        "Beta sigmas should decrease monotonically (index \(i): \(sigmas[i]) > \(sigmas[i-1]))"
      )
    }
  }

  func testBetaFirstAndLastValues() {
    let sigmas = SigmaSchedule.beta(numSteps: 9, sigmaMin: 0.001, sigmaMax: 1.0)
    // First sigma should be close to sigmaMax, last (before sentinel) close to sigmaMin.
    XCTAssertEqual(sigmas[0], 1.0, accuracy: 1e-4, "First sigma should be ~sigmaMax")
    XCTAssertEqual(sigmas[8], 0.001, accuracy: 1e-4, "Last sigma should be ~sigmaMin")
    XCTAssertEqual(sigmas[9], 0.0, accuracy: 1e-10, "Trailing sentinel should be 0")
  }

  func testBetaFlowMatchingBounds() {
    // With flow-matching bounds (0.001 to 1.0), all sigmas should be in [0, 1].
    let sigmas = SigmaSchedule.beta(numSteps: 9, sigmaMin: 0.001, sigmaMax: 1.0)
    for i in 0..<(sigmas.count - 1) {
      XCTAssertGreaterThanOrEqual(sigmas[i], 0.001 - 1e-6)
      XCTAssertLessThanOrEqual(sigmas[i], 1.0 + 1e-6)
    }
  }

  func testBeta57DiffersFromDefaultBeta() {
    let steps = 9
    let beta = SigmaSchedule.beta(
      numSteps: steps,
      sigmaMin: 0.001,
      sigmaMax: 1.0
    )
    let beta57 = SigmaSchedule.beta(
      numSteps: steps,
      sigmaMin: 0.001,
      sigmaMax: 1.0,
      alpha: 0.5,
      betaParam: 0.7
    )

    XCTAssertEqual(beta57.count, steps + 1)
    XCTAssertEqual(beta57.last!, 0.0, accuracy: 1e-10)

    var differs = false
    for i in 0..<beta57.count {
      if abs(beta57[i] - beta[i]) > 1e-6 {
        differs = true
        break
      }
    }
    XCTAssertTrue(differs, "beta57 should differ from default beta")
  }

  func testBeta57MonotonicallyDecreasing() {
    let sigmas = SigmaSchedule.beta(
      numSteps: 20,
      sigmaMin: 0.001,
      sigmaMax: 1.0,
      alpha: 0.5,
      betaParam: 0.7
    )

    XCTAssertEqual(sigmas.count, 21)
    for i in 1..<sigmas.count {
      XCTAssertLessThanOrEqual(
        sigmas[i],
        sigmas[i - 1],
        "beta57 sigmas should decrease monotonically at index \(i)"
      )
    }
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-10)
  }

  // MARK: - Linspace Helper

  func testLinspaceTwoElements() {
    let result = SigmaSchedule.linspace(0.0, 10.0, count: 2)
    XCTAssertEqual(result, [0.0, 10.0])
  }

  func testLinspaceFiveElements() {
    let result = SigmaSchedule.linspace(0.0, 1.0, count: 5)
    XCTAssertEqual(result.count, 5)
    XCTAssertEqual(result[0], 0.0, accuracy: 1e-6)
    XCTAssertEqual(result[1], 0.25, accuracy: 1e-6)
    XCTAssertEqual(result[2], 0.5, accuracy: 1e-6)
    XCTAssertEqual(result[3], 0.75, accuracy: 1e-6)
    XCTAssertEqual(result[4], 1.0, accuracy: 1e-6)
  }

  func testLinspaceSingleElement() {
    let result = SigmaSchedule.linspace(5.0, 10.0, count: 1)
    XCTAssertEqual(result, [5.0])
  }
}
