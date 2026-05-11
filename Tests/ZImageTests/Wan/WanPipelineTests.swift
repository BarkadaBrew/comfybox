import XCTest
import MLX
@testable import ZImage

/// Tests for Wan 2.2 I2V Phase 4: Pipeline components.
///
/// Unit tests verify scheduler math, conditioning shapes, MoE switching,
/// and extend frame calculations against the Python reference implementation.
final class WanPipelineTests: XCTestCase {

  // MARK: - S4.1: FlowUniPCScheduler Tests

  /// Verify sigma computation matches Python for shift=5.0, steps=40.
  ///
  /// Python reference:
  /// ```python
  /// scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1)
  /// scheduler.set_timesteps(40, shift=5.0)
  /// # sigma_max from __init__ = 0.999 (shift=1, no change)
  /// # set_timesteps: linspace(0.999, 0.0, 41)[:-1], then shift=5 applied
  /// ```
  func testSchedulerSigmaComputation() {
    let scheduler = FlowUniPCScheduler(
      numInferenceSteps: 40,
      shift: 5.0,
      numTrainTimesteps: 1000
    )

    // Should have 41 sigma values (40 steps + 1 trailing zero)
    XCTAssertEqual(scheduler.sigmas.dim(0), 41, "Should have numSteps + 1 sigmas")

    // Should have 40 timestep values
    XCTAssertEqual(scheduler.timesteps.dim(0), 40, "Should have numSteps timesteps")

    // First sigma should be close to shift * sigmaMax / (1 + (shift-1) * sigmaMax)
    // sigmaMax = 0.999, shift = 5
    // shifted = 5 * 0.999 / (1 + 4 * 0.999) = 4.995 / 4.996 ≈ 0.9998
    let firstSigma = scheduler.sigmas[0].item(Float.self)
    XCTAssertEqual(firstSigma, 0.9998, accuracy: 0.001, "First sigma should be ~0.9998")

    // Last sigma (before trailing zero) should be small but positive
    let lastSigma = scheduler.sigmas[39].item(Float.self)
    XCTAssertGreaterThan(lastSigma, 0, "Last step sigma should be positive")
    XCTAssertLessThan(lastSigma, 0.2, "Last step sigma should be small")

    // Trailing zero
    let trailingSigma = scheduler.sigmas[40].item(Float.self)
    XCTAssertEqual(trailingSigma, 0.0, accuracy: 1e-6, "Trailing sigma should be 0")

    // First timestep should be floor(sigmas[0] * 1000) — truncated to int like Python
    let firstTimestep = scheduler.timesteps[0].item(Float.self)
    XCTAssertEqual(firstTimestep, Float(Int32(firstSigma * 1000)), accuracy: 0.01, "First timestep = int(sigma * numTrainTimesteps)")

    // Sigmas should be monotonically decreasing
    for i in 0..<40 {
      let s1 = scheduler.sigmas[i].item(Float.self)
      let s2 = scheduler.sigmas[i + 1].item(Float.self)
      XCTAssertGreaterThanOrEqual(s1, s2, "Sigmas should be monotonically decreasing at index \(i)")
    }
  }

  /// Verify scheduler step produces valid output for known inputs.
  func testSchedulerStepOutput() {
    var scheduler = FlowUniPCScheduler(
      numInferenceSteps: 10,
      shift: 5.0,
      numTrainTimesteps: 1000
    )

    // Create a simple test: noise and model output
    let sample = MLXArray.ones([4, 5, 6])
    let modelOutput = MLXArray.ones([4, 5, 6]) * 0.1

    // First step
    let result = scheduler.step(
      modelOutput: modelOutput,
      timestepIndex: 0,
      sample: sample
    )

    // Result should be finite and reasonable
    let resultData = result.asType(.float32)
    eval(resultData)
    let firstVal = resultData[0, 0, 0].item(Float.self)
    XCTAssertFalse(firstVal.isNaN, "Result should not be NaN")
    XCTAssertFalse(firstVal.isInfinite, "Result should not be infinite")
  }

  /// Verify scheduler reset clears state.
  func testSchedulerReset() {
    var scheduler = FlowUniPCScheduler(
      numInferenceSteps: 10,
      shift: 5.0,
      numTrainTimesteps: 1000
    )

    // Do a step to populate state
    let sample = MLXArray.ones([4, 5, 6])
    let modelOutput = MLXArray.ones([4, 5, 6]) * 0.1
    _ = scheduler.step(modelOutput: modelOutput, timestepIndex: 0, sample: sample)

    // Reset
    scheduler.reset()

    // Should be able to step from 0 again
    let result2 = scheduler.step(modelOutput: modelOutput, timestepIndex: 0, sample: sample)
    let val = result2[0, 0, 0].item(Float.self)
    XCTAssertFalse(val.isNaN, "After reset, first step should produce valid output")
  }

  // MARK: - S4.2: WanI2VConditioner Tests

  /// Verify mask shape [4, latent_t, latH, latW] for F=81.
  func testConditionerMaskShape() {
    let frameNum = 81
    let latH = 45  // 360 / 8
    let latW = 80  // 640 / 8

    let mask = WanI2VConditioner.buildMask(
      frameNum: frameNum, latH: latH, latW: latW
    )

    // F=81, latent_t = (81-1)/4 + 1 = 21
    let expectedLatentT = (frameNum - 1) / 4 + 1
    XCTAssertEqual(mask.dim(0), 4, "Mask should have 4 channels")
    XCTAssertEqual(mask.dim(1), expectedLatentT, "Mask latent_t should be \(expectedLatentT)")
    XCTAssertEqual(mask.dim(2), latH, "Mask latH should be \(latH)")
    XCTAssertEqual(mask.dim(3), latW, "Mask latW should be \(latW)")
  }

  /// Verify mask values: first temporal position should be 1, rest 0.
  func testConditionerMaskValues() {
    let mask = WanI2VConditioner.buildMask(
      frameNum: 81, latH: 4, latW: 4
    )

    eval(mask)

    // The first temporal slice should contain 1s (marking the first frame)
    // Due to repeat_interleave(4), the first latent position maps to 4 repeats of frame 0
    // which is all 1s. After reshape/transpose, the first channel at t=0 should be 1.
    let firstVal = mask[0, 0, 0, 0].item(Float.self)
    XCTAssertEqual(firstVal, 1.0, accuracy: 1e-6, "First temporal position should be 1")

    // Later temporal positions should be 0
    let laterVal = mask[0, 2, 0, 0].item(Float.self)
    XCTAssertEqual(laterVal, 0.0, accuracy: 1e-6, "Later temporal positions should be 0")
  }

  /// Verify 36-channel concatenation (noise 16 + mask 4 + VAE 16).
  func testConditionerAssembly() {
    let latH = 4
    let latW = 4
    let latentT = 21

    let mask = MLXArray.ones([4, latentT, latH, latW])
    let vaeEncoded = MLXArray.ones([16, latentT, latH, latW])

    let conditioning = WanI2VConditioner.assembleConditioning(
      mask: mask, vaeEncoded: vaeEncoded
    )

    XCTAssertEqual(conditioning.dim(0), 20, "Conditioning should have 20 channels (4 mask + 16 VAE)")
    XCTAssertEqual(conditioning.dim(1), latentT)
    XCTAssertEqual(conditioning.dim(2), latH)
    XCTAssertEqual(conditioning.dim(3), latW)

    // Noise (16ch) + conditioning (20ch) = 36ch total (done in forward pass)
    let noise = MLXArray.ones([16, latentT, latH, latW])
    let fullInput = MLX.concatenated([noise, conditioning], axis: 0)
    XCTAssertEqual(fullInput.dim(0), 36, "Full input should have 36 channels")
  }

  /// Verify resolution computation.
  func testConditionerResolution() {
    // 720p landscape: 1280x720 image
    let res = WanI2VConditioner.computeResolution(
      imageHeight: 720, imageWidth: 1280, maxArea: 720 * 1280
    )

    // latH and latW should be divisible by patch size (2)
    XCTAssertEqual(res.latH % 2, 0, "latH should be divisible by patch height (2)")
    XCTAssertEqual(res.latW % 2, 0, "latW should be divisible by patch width (2)")

    // Pixel dimensions should match
    XCTAssertEqual(res.pixelH, res.latH * 8, "pixelH = latH * vaeStride")
    XCTAssertEqual(res.pixelW, res.latW * 8, "pixelW = latW * vaeStride")

    // Total pixels should be close to maxArea
    let area = res.pixelH * res.pixelW
    XCTAssertLessThanOrEqual(area, 720 * 1280 * 2, "Area should not exceed 2x maxArea")
  }

  /// Verify noise generation shape.
  func testConditionerNoiseGeneration() {
    let noise = WanI2VConditioner.generateNoise(
      frameNum: 81, latH: 45, latW: 80, seed: 42
    )

    let expectedLatentT = (81 - 1) / 4 + 1  // = 21
    XCTAssertEqual(noise.dim(0), 16, "Noise should have 16 channels")
    XCTAssertEqual(noise.dim(1), expectedLatentT, "Noise latent_t should be \(expectedLatentT)")
    XCTAssertEqual(noise.dim(2), 45, "Noise latH")
    XCTAssertEqual(noise.dim(3), 80, "Noise latW")
  }

  /// Verify sequence length computation.
  func testConditionerSeqLen() {
    let seqLen = WanI2VConditioner.computeSeqLen(
      frameNum: 81, latH: 90, latW: 160
    )
    // latentT = (81-1)/4+1 = 21
    // seqLen = 21 * 90 * 160 / (2 * 2) = 21 * 90 * 40 = 75600
    XCTAssertEqual(seqLen, 21 * 90 * 160 / 4, "Sequence length computation")
  }

  // MARK: - S4.3: MoE Manager Tests

  /// Verify expert switching at boundary threshold.
  func testMoEBoundarySelection() {
    let manager = WanMoEManager(
      modelDir: URL(fileURLWithPath: "/tmp/test-models"),
      boundary: 0.9,
      numTrainTimesteps: 1000,
      lazyLoading: true,
      logger: .init(label: "test")
    )

    // boundary = 0.9 * 1000 = 900
    let highScale = manager.guideScale(forTimestep: 950, scales: (3.0, 5.0))
    XCTAssertEqual(highScale, 5.0, "Timestep >= 900 should use high-noise guide scale")

    let lowScale = manager.guideScale(forTimestep: 850, scales: (3.0, 5.0))
    XCTAssertEqual(lowScale, 3.0, "Timestep < 900 should use low-noise guide scale")

    // Exactly at boundary
    let atBoundary = manager.guideScale(forTimestep: 900, scales: (3.0, 5.0))
    XCTAssertEqual(atBoundary, 5.0, "Timestep == 900 should use high-noise guide scale")

    // Just below boundary
    let belowBoundary = manager.guideScale(forTimestep: 899.9, scales: (3.0, 5.0))
    XCTAssertEqual(belowBoundary, 3.0, "Timestep just below 900 should use low-noise")
  }

  // MARK: - S4.6: Extend Tests

  /// Verify frame counting for various targetSeconds values.
  func testExtendFrameCounting() {
    // Single chunk: 81 frames = 81/16 = 5.0625s
    let chunks1 = WanExtend.chunksNeeded(targetSeconds: 5.0, framesPerChunk: 81, fps: 16)
    XCTAssertEqual(chunks1, 1, "5s should need 1 chunk")

    // Two chunks: first=81, second=80 (drop 1), total=161 frames = 10.0625s
    let chunks2 = WanExtend.chunksNeeded(targetSeconds: 10.0, framesPerChunk: 81, fps: 16)
    XCTAssertEqual(chunks2, 2, "10s should need 2 chunks")

    // Three chunks: 81 + 80 + 80 = 241 frames = 15.0625s
    let chunks3 = WanExtend.chunksNeeded(targetSeconds: 15.0, framesPerChunk: 81, fps: 16)
    XCTAssertEqual(chunks3, 3, "15s should need 3 chunks")

    // Verify totalFrames
    XCTAssertEqual(WanExtend.totalFrames(chunks: 1, framesPerChunk: 81), 81)
    XCTAssertEqual(WanExtend.totalFrames(chunks: 2, framesPerChunk: 81), 161)
    XCTAssertEqual(WanExtend.totalFrames(chunks: 3, framesPerChunk: 81), 241)

    // Edge case: 0 chunks
    XCTAssertEqual(WanExtend.totalFrames(chunks: 0, framesPerChunk: 81), 0)
  }

  /// Verify extend for very short durations.
  func testExtendShortDuration() {
    let chunks = WanExtend.chunksNeeded(targetSeconds: 1.0, framesPerChunk: 81, fps: 16)
    XCTAssertEqual(chunks, 1, "Very short duration should still need 1 chunk")
  }

  /// Verify extend for exact chunk boundary.
  func testExtendExactBoundary() {
    // 81 frames at 16fps = 5.0625s
    let chunks = WanExtend.chunksNeeded(targetSeconds: 5.0625, framesPerChunk: 81, fps: 16)
    XCTAssertEqual(chunks, 1, "Exactly 81 frames duration should need 1 chunk")
  }

  // MARK: - S4.1: Scheduler Multi-Step Test

  /// Verify the scheduler can run multiple steps without producing NaN.
  func testSchedulerMultipleSteps() {
    var scheduler = FlowUniPCScheduler(
      numInferenceSteps: 10,
      shift: 5.0,
      numTrainTimesteps: 1000
    )

    var sample = MLXRandom.normal([16, 21, 45, 80])
    eval(sample)

    for stepIdx in 0..<10 {
      let modelOutput = MLXRandom.normal(sample.shape)
      eval(modelOutput)

      sample = scheduler.step(
        modelOutput: modelOutput,
        timestepIndex: stepIdx,
        sample: sample
      )
      eval(sample)

      // Check for NaN
      let hasNaN = MLX.any(MLX.isNaN(sample)).item(Bool.self)
      XCTAssertFalse(hasNaN, "Step \(stepIdx) should not produce NaN")
    }
  }
}
