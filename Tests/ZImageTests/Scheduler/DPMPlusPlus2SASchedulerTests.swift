import XCTest
import MLX
import MLXRandom
@testable import ZImage

final class DPMPlusPlus2SASchedulerTests: XCTestCase {

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

  // MARK: - First Step Behavior

  func testFirstStepMatchesEulerWhenDeterministic() {
    // With eta=0, the first step (no history) should match Euler exactly.
    var dpmsa = Self.makeScheduler(steps: 9, eta: 0.0)
    let euler = FlowMatchEulerScheduler(
      numInferenceSteps: 9,
      sigmaValues: Self.defaultSigmas(steps: 9),
      numTrainTimesteps: 1000
    )

    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    let dpmsaResult = dpmsa.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    let eulerResult = euler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let dpmsaF32 = dpmsaResult.asType(.float32)
    let eulerF32 = eulerResult.asType(.float32)
    MLX.eval(dpmsaF32, eulerF32)

    let dpmsaData = dpmsaF32.asArray(Float.self)
    let eulerData = eulerF32.asArray(Float.self)

    for i in 0..<dpmsaData.count {
      XCTAssertEqual(
        dpmsaData[i], eulerData[i], accuracy: 1e-4,
        "First deterministic step should match Euler"
      )
    }
  }

  // MARK: - Stochastic Behavior

  func testStochasticOutputDiffersFromDeterministic() {
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()
    let seed: UInt64 = 42

    var det = Self.makeScheduler(steps: 9, eta: 0.0)
    var stoch = Self.makeScheduler(steps: 9, eta: 1.0, seed: seed)

    // Run two steps to engage multistep + noise.
    var detSample = det.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )
    var stochSample = stoch.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    let mo2 = Self.makeModelOutput(values: [0.2, 0.3, 0.4, 0.5])
    detSample = det.step(
      modelOutput: mo2, timestepIndex: 1, sample: detSample
    )
    stochSample = stoch.step(
      modelOutput: mo2, timestepIndex: 1, sample: stochSample
    )

    let detF32 = detSample.asType(.float32)
    let stochF32 = stochSample.asType(.float32)
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
    XCTAssertFalse(allSame, "Stochastic output should differ from deterministic")
  }

  func testReproducibleWithSeed() {
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()
    let seed: UInt64 = 42

    var s1 = Self.makeScheduler(steps: 9, eta: 1.0, seed: seed)
    var s2 = Self.makeScheduler(steps: 9, eta: 1.0, seed: seed)

    // Run two steps to engage noise injection.
    var r1 = s1.step(modelOutput: modelOutput, timestepIndex: 0, sample: sample)
    var r2 = s2.step(modelOutput: modelOutput, timestepIndex: 0, sample: sample)

    let mo2 = Self.makeModelOutput(values: [0.2, 0.3, 0.4, 0.5])
    r1 = s1.step(modelOutput: mo2, timestepIndex: 1, sample: r1)
    r2 = s2.step(modelOutput: mo2, timestepIndex: 1, sample: r2)

    let f1 = r1.asType(.float32)
    let f2 = r2.asType(.float32)
    MLX.eval(f1, f2)

    let data1 = f1.asArray(Float.self)
    let data2 = f2.asArray(Float.self)

    for i in 0..<data1.count {
      XCTAssertEqual(data1[i], data2[i], accuracy: 1e-5, "Same seed should reproduce output")
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

  func testFullLoop() {
    var scheduler = Self.makeScheduler(steps: 9, seed: 42)
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
    var scheduler = Self.makeScheduler(steps: 9, eta: 0.0)
    let sample = Self.makeSample()
    let modelOutput = Self.makeModelOutput()

    // Run one step to establish history.
    _ = scheduler.step(
      modelOutput: modelOutput, timestepIndex: 0, sample: sample
    )

    // Reset and verify first step matches Euler again.
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

  static func makeScheduler(
    steps: Int,
    eta: Float = 1.0,
    seed: UInt64? = nil
  ) -> DPMPlusPlus2SAScheduler {
    DPMPlusPlus2SAScheduler(
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
