import XCTest
import MLX
import MLXRandom
@testable import ZImage

final class DDIMSchedulerTests: XCTestCase {

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

  // MARK: - Determinism

  func testEtaZeroDeterministic() {
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    var ddim1 = Self.makeScheduler(steps: 9, eta: 0.0)
    var ddim2 = Self.makeScheduler(steps: 9, eta: 0.0)

    let result1 = ddim1.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let result2 = ddim2.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let r1 = result1.asType(.float32)
    let r2 = result2.asType(.float32)
    MLX.eval(r1, r2)

    let data1 = r1.asArray(Float.self)
    let data2 = r2.asArray(Float.self)

    for i in 0..<data1.count {
      XCTAssertEqual(data1[i], data2[i], accuracy: 1e-5, "eta=0 should be deterministic")
    }
  }

  func testEtaOneDiffersFromEtaZero() {
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let seed: UInt64 = 42
    var ddimDet = Self.makeScheduler(steps: 9, eta: 0.0)
    var ddimStoch = Self.makeScheduler(steps: 9, eta: 1.0, seed: seed)

    let resultDet = ddimDet.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let resultStoch = ddimStoch.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let detF32 = resultDet.asType(.float32)
    let stochF32 = resultStoch.asType(.float32)
    MLX.eval(detF32, stochF32)

    let detData = detF32.asArray(Float.self)
    let stochData = stochF32.asArray(Float.self)

    var allSame = true
    for i in 0..<detData.count {
      if abs(detData[i] - stochData[i]) > 1e-4 {
        allSame = false
        break
      }
    }
    XCTAssertFalse(allSame, "eta=1 should produce different output than eta=0")
  }

  func testReproducibleWithSeed() {
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()
    let seed: UInt64 = 42

    var ddim1 = Self.makeScheduler(steps: 9, eta: 1.0, seed: seed)
    var ddim2 = Self.makeScheduler(steps: 9, eta: 1.0, seed: seed)

    let result1 = ddim1.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let result2 = ddim2.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let r1 = result1.asType(.float32)
    let r2 = result2.asType(.float32)
    MLX.eval(r1, r2)

    let data1 = r1.asArray(Float.self)
    let data2 = r2.asArray(Float.self)

    for i in 0..<data1.count {
      XCTAssertEqual(data1[i], data2[i], accuracy: 1e-5, "Same seed should reproduce output")
    }
  }

  // MARK: - Step Behavior

  func testStepProducesDifferentOutput() {
    var scheduler = Self.makeScheduler(steps: 9)
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let result = scheduler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let resultF32 = result.asType(.float32)
    let sampleF32 = sample.asType(.float32)
    MLX.eval(resultF32, sampleF32)

    let resultData = resultF32.asArray(Float.self)
    let sampleData = sampleF32.asArray(Float.self)

    var allSame = true
    for i in 0..<resultData.count {
      if abs(resultData[i] - sampleData[i]) > 1e-6 {
        allSame = false
        break
      }
    }
    XCTAssertFalse(allSame, "DDIM step should modify the sample")
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

  func testFullLoop() {
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

  static func makeScheduler(
    steps: Int,
    eta: Float = 0.0,
    seed: UInt64? = nil
  ) -> DDIMScheduler {
    DDIMScheduler(
      numInferenceSteps: steps,
      sigmaValues: defaultSigmas(steps: steps),
      numTrainTimesteps: 1000,
      eta: eta,
      randomKey: seed.map { MLXRandom.key($0) }
    )
  }

  static func makeSample(values: [Float] = [1.0, 2.0, 3.0, 4.0]) -> MLXArray {
    MLXArray(values, [1, 1, 2, 2]).asType(.bfloat16)
  }

  static func makeModelOutput(values: [Float] = [0.1, 0.2, 0.3, 0.4]) -> MLXArray {
    MLXArray(values, [1, 1, 2, 2]).asType(.bfloat16)
  }
}
