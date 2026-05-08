import XCTest
import MLX
@testable import ZImage

final class RES2sSchedulerTests: XCTestCase {

  // MARK: - Phi Functions

  func testPhi1AtZeroUsesTaylorExpansion() {
    XCTAssertEqual(RES2sScheduler.phi1ForTesting(0.0), 1.0, accuracy: 1e-6)
  }

  func testPhi1AtOne() {
    XCTAssertEqual(
      RES2sScheduler.phi1ForTesting(1.0),
      expf(1.0) - 1.0,
      accuracy: 1e-5
    )
  }

  func testPhi2AtZeroUsesTaylorExpansion() {
    XCTAssertEqual(RES2sScheduler.phi2ForTesting(0.0), 0.5, accuracy: 1e-6)
  }

  func testPhi2AtOne() {
    XCTAssertEqual(
      RES2sScheduler.phi2ForTesting(1.0),
      expf(1.0) - 2.0,
      accuracy: 1e-5
    )
  }

  // MARK: - Initialization

  func testInitializationWithDefaultC2() {
    let scheduler = Self.makeScheduler(steps: 9)

    XCTAssertEqual(scheduler.numInferenceSteps, 9)
    XCTAssertEqual(scheduler.sigmas.dim(0), 10)
    XCTAssertEqual(scheduler.timesteps.dim(0), 9)
    XCTAssertEqual(scheduler.c2, 0.5, accuracy: 1e-6)
  }

  // MARK: - Multi-Evaluation Protocol

  func testRequiresIntermediateEvaluation() {
    let scheduler = Self.makeScheduler(steps: 9)
    XCTAssertTrue(scheduler.requiresIntermediateEvaluation)
  }

  func testIntermediateStepReturnsNonNilWithShapePreserved() {
    var scheduler = Self.makeScheduler(steps: 9)
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let intermediate = scheduler.intermediateStep(
      modelOutput: modelOutput,
      timestepIndex: 0,
      sample: sample
    )

    XCTAssertNotNil(intermediate)
    XCTAssertEqual(intermediate?.shape, sample.shape)
  }

  func testIntermediateSigmaUsesGeometricMidpointByDefault() {
    let scheduler = Self.makeScheduler(steps: 9)
    let sigmas = scheduler.sigmas.asArray(Float.self)

    let intermediateSigma = scheduler.intermediateSigma(timestepIndex: 0)

    XCTAssertEqual(
      intermediateSigma!,
      sqrtf(sigmas[0] * sigmas[1]),
      accuracy: 1e-6
    )
  }

  // MARK: - Finalize Behavior

  func testFinalizeStepProducesDifferentOutputThanInput() {
    var scheduler = Self.makeScheduler(steps: 9)
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()
    let intermediateOutput = Self.makeModelOutput(values: [0.15, 0.25, 0.35, 0.45])

    _ = scheduler.intermediateStep(
      modelOutput: modelOutput,
      timestepIndex: 0,
      sample: sample
    )
    let result = scheduler.finalizeStep(
      originalOutput: modelOutput,
      intermediateOutput: intermediateOutput,
      timestepIndex: 0,
      sample: sample
    )

    XCTAssertEqual(result.shape, sample.shape)
    XCTAssertFalse(Self.arraysEqual(result, sample), "RES 2s output should differ from input")
  }

  // MARK: - Full Loop

  func testFullLoopWithIntermediateAndFinalize() {
    var scheduler = Self.makeScheduler(steps: 9)
    var sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()
    let intermediateOutput = Self.makeModelOutput(values: [0.12, 0.22, 0.32, 0.42])

    for i in 0..<scheduler.numInferenceSteps {
      let intermediate = scheduler.intermediateStep(
        modelOutput: modelOutput,
        timestepIndex: i,
        sample: sample
      )
      XCTAssertNotNil(intermediate)

      sample = scheduler.finalizeStep(
        originalOutput: modelOutput,
        intermediateOutput: intermediateOutput,
        timestepIndex: i,
        sample: sample
      )
      MLX.eval(sample.asType(.float32))
    }

    XCTAssertEqual(sample.shape, [1, 1, 2, 2])
  }

  func testShapePreservationThroughFullLoop() {
    var scheduler = Self.makeScheduler(steps: 9)
    var sample = Self.makeSample()
    let originalShape = sample.shape
    let modelOutput = Self.makeModelOutput()
    let intermediateOutput = Self.makeModelOutput(values: [0.12, 0.22, 0.32, 0.42])

    for i in 0..<scheduler.numInferenceSteps {
      let intermediate = scheduler.intermediateStep(
        modelOutput: modelOutput,
        timestepIndex: i,
        sample: sample
      )!
      XCTAssertEqual(intermediate.shape, originalShape)

      sample = scheduler.finalizeStep(
        originalOutput: modelOutput,
        intermediateOutput: intermediateOutput,
        timestepIndex: i,
        sample: sample
      )
      XCTAssertEqual(sample.shape, originalShape)
      MLX.eval(sample.asType(.float32))
    }
  }

  // MARK: - Helpers

  static func defaultSigmas(steps: Int) -> [Float] {
    SigmaSchedule.flow(
      numSteps: steps,
      config: FlowMatchSchedulerTests.makeConfig()
    )
  }

  static func makeScheduler(steps: Int) -> RES2sScheduler {
    RES2sScheduler(
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

  static func arraysEqual(_ lhs: MLXArray, _ rhs: MLXArray, accuracy: Float = 1e-5) -> Bool {
    let lhsF32 = lhs.asType(.float32)
    let rhsF32 = rhs.asType(.float32)
    MLX.eval(lhsF32, rhsF32)

    let lhsData = lhsF32.asArray(Float.self)
    let rhsData = rhsF32.asArray(Float.self)
    return zip(lhsData, rhsData).allSatisfy { pair in
      abs(pair.0 - pair.1) <= accuracy
    }
  }
}
