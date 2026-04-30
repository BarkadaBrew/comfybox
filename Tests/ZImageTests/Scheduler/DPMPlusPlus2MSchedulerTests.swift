import XCTest
import MLX
@testable import ZImage

final class DPMPlusPlus2MSchedulerTests: XCTestCase {

  // MARK: - Initialization

  func testInitializationWithDefaults() {
    let scheduler = Self.makeScheduler(steps: 9)

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)
    XCTAssertEqual(scheduler.timesteps.dim(0), 9)
  }

  func testInitializationWithCustomSteps() {
    for steps in [4, 9, 20, 50] {
      let scheduler = Self.makeScheduler(steps: steps)
      XCTAssertEqual(scheduler.numInferenceSteps, steps)
      XCTAssertEqual(scheduler.sigmas.dim(0), steps + 1)
      XCTAssertEqual(scheduler.timesteps.dim(0), steps)
    }
  }

  // MARK: - Sigma Properties

  func testSigmaMonotonicallyDecreasing() {
    let scheduler = Self.makeScheduler(steps: 9)
    let sigmas = scheduler.sigmas.asArray(Float.self)

    for i in 1..<sigmas.count {
      XCTAssertLessThanOrEqual(
        sigmas[i], sigmas[i - 1],
        "Sigmas should decrease monotonically"
      )
    }
  }

  func testSigmaBounds() {
    let scheduler = Self.makeScheduler(steps: 9)
    let sigmas = scheduler.sigmas.asArray(Float.self)

    XCTAssertGreaterThan(sigmas[0], 0.0)
    XCTAssertEqual(sigmas.last!, 0.0, accuracy: 1e-6)
  }

  // MARK: - Step Behavior

  func testFirstStepMatchesEuler() {
    var dpm = Self.makeScheduler(steps: 9)
    let euler = FlowMatchEulerScheduler(
      numInferenceSteps: 9,
      sigmaValues: Self.defaultSigmas(steps: 9),
      numTrainTimesteps: 1000
    )

    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let dpmResult = dpm.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let eulerResult = euler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let dpmF32 = dpmResult.asType(.float32)
    let eulerF32 = eulerResult.asType(.float32)
    MLX.eval(dpmF32, eulerF32)

    let dpmData = dpmF32.asArray(Float.self)
    let eulerData = eulerF32.asArray(Float.self)

    for i in 0..<dpmData.count {
      XCTAssertEqual(
        dpmData[i], eulerData[i], accuracy: 1e-4,
        "First step should match Euler"
      )
    }
  }

  func testSecondStepDiffersFromEuler() {
    var dpm = Self.makeScheduler(steps: 9)
    var euler = FlowMatchEulerScheduler(
      numInferenceSteps: 9,
      sigmaValues: Self.defaultSigmas(steps: 9),
      numTrainTimesteps: 1000
    )

    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    // Run first step on both.
    var dpmSample = dpm.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    var eulerSample = euler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    // Run second step on both.
    let modelOutput2 = Self.makeModelOutput(values: [0.2, 0.3, 0.4, 0.5])
    dpmSample = dpm.step(
      modelOutput: modelOutput2, timestepIndex: 1, sample: dpmSample
    )
    eulerSample = euler.step(
      modelOutput: modelOutput2, timestepIndex: 1, sample: eulerSample
    )

    let dpmF32 = dpmSample.asType(.float32)
    let eulerF32 = eulerSample.asType(.float32)
    MLX.eval(dpmF32, eulerF32)

    let dpmData = dpmF32.asArray(Float.self)
    let eulerData = eulerF32.asArray(Float.self)

    var allSame = true
    for i in 0..<dpmData.count {
      if abs(dpmData[i] - eulerData[i]) > 1e-4 {
        allSame = false
        break
      }
    }
    XCTAssertFalse(allSame, "Second step should differ from Euler (multistep correction)")
  }

  func testDeterministic() {
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    var dpm1 = Self.makeScheduler(steps: 9)
    var dpm2 = Self.makeScheduler(steps: 9)

    let result1 = dpm1.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let result2 = dpm2.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let r1 = result1.asType(.float32)
    let r2 = result2.asType(.float32)
    MLX.eval(r1, r2)

    let data1 = r1.asArray(Float.self)
    let data2 = r2.asArray(Float.self)

    for i in 0..<data1.count {
      XCTAssertEqual(data1[i], data2[i], accuracy: 1e-6, "DPM++ 2M should be deterministic")
    }
  }

  // MARK: - Shape Preservation

  func testShapePreservation() {
    var scheduler = Self.makeScheduler(steps: 9)
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let result = scheduler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    XCTAssertEqual(result.shape, sample.shape)
  }

  // MARK: - Full Loop

  func testFullLoopCompletion() {
    var scheduler = Self.makeScheduler(steps: 9)
    var sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    for i in 0..<scheduler.numInferenceSteps {
      sample = scheduler.step(
        modelOutput: modelOutput, timestepIndex: i, sample: sample
      )
      MLX.eval(sample.asType(.float32))
    }

    XCTAssertEqual(sample.shape, [1, 1, 2, 2])
  }

  // MARK: - Reset

  func testResetClearsHistory() {
    var scheduler = Self.makeScheduler(steps: 9)
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    // Run one step to establish history.
    _ = scheduler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    // Reset and run first step again -- should match Euler.
    scheduler.reset()

    let euler = FlowMatchEulerScheduler(
      numInferenceSteps: 9,
      sigmaValues: Self.defaultSigmas(steps: 9),
      numTrainTimesteps: 1000
    )

    let resetResult = scheduler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let eulerResult = euler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let resetF32 = resetResult.asType(.float32)
    let eulerF32 = eulerResult.asType(.float32)
    MLX.eval(resetF32, eulerF32)

    let resetData = resetF32.asArray(Float.self)
    let eulerData = eulerF32.asArray(Float.self)

    for i in 0..<resetData.count {
      XCTAssertEqual(
        resetData[i], eulerData[i], accuracy: 1e-4,
        "After reset, first step should match Euler"
      )
    }
  }

  // MARK: - Protocol Conformance

  func testProtocolConformance() {
    let scheduler: any ZImageScheduler = Self.makeScheduler(steps: 9)
    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertFalse(scheduler.requiresIntermediateEvaluation)
  }

  // MARK: - Helpers

  static func defaultSigmas(steps: Int) -> [Float] {
    SigmaSchedule.flow(
      numSteps: steps,
      config: FlowMatchSchedulerTests.makeConfig()
    )
  }

  static func makeScheduler(steps: Int) -> DPMPlusPlus2MScheduler {
    DPMPlusPlus2MScheduler(
      numInferenceSteps: steps,
      sigmaValues: defaultSigmas(steps: steps),
      numTrainTimesteps: 1000
    )
  }

  static func makeSample(values: [Float] = [1.0, 2.0, 3.0, 4.0]) -> MLXArray {
    MLXArray(values, [1, 1, 2, 2]).asType(.bfloat16)
  }

  static func makeModelOutput(values: [Float] = [0.1, 0.2, 0.3, 0.4]) -> MLXArray {
    MLXArray(values, [1, 1, 2, 2]).asType(.bfloat16)
  }
}
