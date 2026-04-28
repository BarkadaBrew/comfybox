import XCTest
import MLX
@testable import ZImage

final class HeunSchedulerTests: XCTestCase {

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

  // MARK: - Multi-Evaluation Protocol

  func testRequiresIntermediateEvaluation() {
    let scheduler = Self.makeScheduler(steps: 9)
    XCTAssertTrue(scheduler.requiresIntermediateEvaluation)
  }

  func testIntermediateStepReturnsNonNil() {
    var scheduler = Self.makeScheduler(steps: 9)
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let intermediate = scheduler.intermediateStep(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    XCTAssertNotNil(intermediate, "Heun's intermediateStep should return non-nil")
  }

  func testIntermediateStepIsEulerStep() {
    // The intermediate step should match a plain Euler step.
    var heun = Self.makeScheduler(steps: 9)
    let euler = FlowMatchEulerScheduler(
      numInferenceSteps: 9,
      sigmaValues: Self.defaultSigmas(steps: 9),
      numTrainTimesteps: 1000
    )

    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let intermediate = heun.intermediateStep(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )!
    let eulerResult = euler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let intF32 = intermediate.asType(.float32)
    let eulerF32 = eulerResult.asType(.float32)
    MLX.eval(intF32, eulerF32)

    let intData = intF32.asArray(Float.self)
    let eulerData = eulerF32.asArray(Float.self)

    for i in 0..<intData.count {
      XCTAssertEqual(
        intData[i], eulerData[i], accuracy: 1e-4,
        "Intermediate step should match Euler"
      )
    }
  }

  // MARK: - Finalize Behavior

  func testFinalizeProducesDifferentResultThanEuler() {
    var heun = Self.makeScheduler(steps: 9)
    let euler = FlowMatchEulerScheduler(
      numInferenceSteps: 9,
      sigmaValues: Self.defaultSigmas(steps: 9),
      numTrainTimesteps: 1000
    )

    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()
    // Simulate a different prediction at the intermediate point.
    let intermediateOutput = Self.makeModelOutput(values: [0.15, 0.25, 0.35, 0.45])

    let heunResult = heun.finalizeStep(
      originalOutput: modelOutput,
      intermediateOutput: intermediateOutput,
      timestepIndex: 0,
      sample: sample
    )
    let eulerResult = euler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let heunF32 = heunResult.asType(.float32)
    let eulerF32 = eulerResult.asType(.float32)
    MLX.eval(heunF32, eulerF32)

    let heunData = heunF32.asArray(Float.self)
    let eulerData = eulerF32.asArray(Float.self)

    var allSame = true
    for i in 0..<heunData.count {
      if abs(heunData[i] - eulerData[i]) > 1e-4 {
        allSame = false
        break
      }
    }
    XCTAssertFalse(allSame, "Heun finalize should differ from Euler when outputs differ")
  }

  func testFinalizeWithIdenticalOutputsMatchesEuler() {
    // When both evaluations return the same output,
    // Heun's average should produce the same result as Euler.
    var heun = Self.makeScheduler(steps: 9)
    let euler = FlowMatchEulerScheduler(
      numInferenceSteps: 9,
      sigmaValues: Self.defaultSigmas(steps: 9),
      numTrainTimesteps: 1000
    )

    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let heunResult = heun.finalizeStep(
      originalOutput: modelOutput,
      intermediateOutput: modelOutput,
      timestepIndex: 0,
      sample: sample
    )
    let eulerResult = euler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let heunF32 = heunResult.asType(.float32)
    let eulerF32 = eulerResult.asType(.float32)
    MLX.eval(heunF32, eulerF32)

    let heunData = heunF32.asArray(Float.self)
    let eulerData = eulerF32.asArray(Float.self)

    for i in 0..<heunData.count {
      XCTAssertEqual(
        heunData[i], eulerData[i], accuracy: 1e-4,
        "Identical outputs should match Euler"
      )
    }
  }

  // MARK: - Single Step Fallback

  func testSingleStepFallbackMatchesEuler() {
    var heun = Self.makeScheduler(steps: 9)
    let euler = FlowMatchEulerScheduler(
      numInferenceSteps: 9,
      sigmaValues: Self.defaultSigmas(steps: 9),
      numTrainTimesteps: 1000
    )

    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let heunResult = heun.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let eulerResult = euler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let heunF32 = heunResult.asType(.float32)
    let eulerF32 = eulerResult.asType(.float32)
    MLX.eval(heunF32, eulerF32)

    let heunData = heunF32.asArray(Float.self)
    let eulerData = eulerF32.asArray(Float.self)

    for i in 0..<heunData.count {
      XCTAssertEqual(
        heunData[i], eulerData[i], accuracy: 1e-4,
        "Step fallback should match Euler"
      )
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

    let intermediate = scheduler.intermediateStep(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )!
    XCTAssertEqual(intermediate.shape, sample.shape)

    let finalized = scheduler.finalizeStep(
      originalOutput: modelOutput,
      intermediateOutput: modelOutput,
      timestepIndex: 0,
      sample: sample
    )
    XCTAssertEqual(finalized.shape, sample.shape)
  }

  // MARK: - Full Loop (Simulated)

  func testFullLoop() {
    // Simulate the full multi-evaluation loop manually.
    var scheduler = Self.makeScheduler(steps: 9)
    var sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    for i in 0..<scheduler.numInferenceSteps {
      if let intermediate = scheduler.intermediateStep(
        modelOutput: modelOutput, timestepIndex: i, sample: sample
      ) {
        // Simulate second model eval with slightly different output.
        let midOutput = Self.makeModelOutput(values: [0.12, 0.22, 0.32, 0.42])
        sample = scheduler.finalizeStep(
          originalOutput: modelOutput,
          intermediateOutput: midOutput,
          timestepIndex: i,
          sample: sample
        )
      } else {
        sample = scheduler.step(
          modelOutput: modelOutput, timestepIndex: i, sample: sample
        )
      }
      MLX.eval(sample.asType(.float32))
    }

    XCTAssertEqual(sample.shape, [1, 1, 2, 2])
  }

  // MARK: - Determinism

  func testDeterministic() {
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()
    let midOutput = Self.makeModelOutput(values: [0.12, 0.22, 0.32, 0.42])

    var heun1 = Self.makeScheduler(steps: 9)
    var heun2 = Self.makeScheduler(steps: 9)

    let _ = heun1.intermediateStep(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let result1 = heun1.finalizeStep(
      originalOutput: modelOutput,
      intermediateOutput: midOutput,
      timestepIndex: 0,
      sample: sample
    )

    let _ = heun2.intermediateStep(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let result2 = heun2.finalizeStep(
      originalOutput: modelOutput,
      intermediateOutput: midOutput,
      timestepIndex: 0,
      sample: sample
    )

    let r1 = result1.asType(.float32)
    let r2 = result2.asType(.float32)
    MLX.eval(r1, r2)

    let data1 = r1.asArray(Float.self)
    let data2 = r2.asArray(Float.self)

    for i in 0..<data1.count {
      XCTAssertEqual(data1[i], data2[i], accuracy: 1e-6, "Heun should be deterministic")
    }
  }

  // MARK: - Protocol Conformance

  func testProtocolConformance() {
    let scheduler: any ZImageScheduler = Self.makeScheduler(steps: 9)
    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertTrue(scheduler.requiresIntermediateEvaluation)
  }

  // MARK: - Helpers

  static func defaultSigmas(steps: Int) -> [Float] {
    SigmaSchedule.flow(
      numSteps: steps,
      config: FlowMatchSchedulerTests.makeConfig()
    )
  }

  static func makeScheduler(steps: Int) -> HeunScheduler {
    HeunScheduler(
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
