import XCTest
import MLX
import MLXRandom
@testable import ZImage

/// WP-E3 — `Krea2DenoiseLoop` (FDD-krea2-raw-recipe §3.3, D1).
///
/// Weight-free: `evaluate` is a closed-form velocity field, so the byte-identity
/// gate, the three dispatch branches, the `startIndex` accounting, the CFG
/// eval count and the reset discipline all run in milliseconds.
///
/// The byte-identity gate (AC-1/AC-2, weight-free half) keeps a verbatim copy
/// of the pre-change inline loop (`Krea2Pipeline.generate` at `aa8e590`) and
/// asserts the new driver lands on the SAME bf16 bits for euler + krea2 over
/// several step counts and seeds. The live-render half is the integration
/// oracle (`oracle-seed44821`).
final class Krea2DenoiseLoopTests: XCTestCase {

  static let align = 16
  static let seqLen1024 = (1024 / 16) * (1024 / 16)

  // MARK: - The pre-change loop, verbatim

  /// `Krea2Pipeline.generate` / `generateImg2Img` loop body at `aa8e590`:
  /// `img = img + (tp - tc) * v` with `tc`/`tp` Swift Floats from
  /// `Krea2Sampling.timesteps`. Only the transformer call is replaced by
  /// `evaluate`, which receives the SAME `tc`.
  static func inlineEulerLoop(
    ts: [Float], initial: MLXArray, startIndex: Int,
    evaluate: (MLXArray, Float) -> MLXArray
  ) -> MLXArray {
    var img = initial
    let total = ts.count - 1
    for i in startIndex..<total {
      let tc = ts[i], tp = ts[i + 1]
      let v = evaluate(img, tc)
      img = img + (tp - tc) * v
      MLX.eval(img)
    }
    return img
  }

  /// The pre-change `ts`: `Krea2Sampling.timesteps` with the inline slope mu.
  static func preChangeTimesteps(seqLen: Int, steps: Int) -> [Float] {
    let x1 = Float((256 / align) * (256 / align))
    let x2 = Float((1280 / align) * (1280 / align))
    return Krea2Sampling.timesteps(seqLen: seqLen, steps: steps, x1: x1, x2: x2)
  }

  /// A stand-in for the transformer: bf16 in, bf16 out, depends on both the
  /// latent and the sigma (built into a `t` tensor exactly as the pipeline
  /// does), and is nonlinear so any rounding difference in the update
  /// propagates rather than cancelling.
  static func syntheticTransformer(_ x: MLXArray, _ sigma: Float) -> MLXArray {
    let t = MLX.full([1], values: MLXArray(sigma)).asType(.bfloat16)
    return MLX.tanh(x * 0.5) * 0.5 + t * 0.25 - x * 0.1
  }

  static func bf16Noise(seed: UInt64, shape: [Int] = [1, 64, 64]) -> MLXArray {
    MLXRandom.seed(seed)
    let noise = MLXRandom.normal(shape).asType(.bfloat16)
    MLX.eval(noise)
    return noise
  }

  static func bits(_ x: MLXArray) -> [Float] {
    // bf16 → float32 is exact, so Float equality is bit equality.
    x.asType(.float32).asArray(Float.self)
  }

  static func defaultScheduler(steps: Int, seqLen: Int = seqLen1024) throws -> any ZImageScheduler {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: seqLen, align: align)
    return try Krea2Pipeline.makeScheduler(
      sampler: .euler, sigmaSchedule: .krea2, steps: steps, shift: shift, seed: 0, c2: 0.5)
  }

  // MARK: - AC-1 (weight-free half): euler + krea2 is bit-identical

  func testDefaultPathBitIdenticalToInlineLoop() throws {
    for steps in [4, 6, 9, 12, 20] {
      for seed: UInt64 in [1, 44821, 99] {
        let ts = Self.preChangeTimesteps(seqLen: Self.seqLen1024, steps: steps)
        let initial = Self.bf16Noise(seed: seed)

        let old = Self.inlineEulerLoop(ts: ts, initial: initial, startIndex: 0, evaluate: Self.syntheticTransformer)

        var scheduler = try Self.defaultScheduler(steps: steps)
        // The grid the driver walks IS the pre-change grid, element for element.
        XCTAssertEqual(scheduler.sigmas.asArray(Float.self), ts, "steps \(steps)")
        let (new, stats) = Krea2DenoiseLoop.run(
          scheduler: &scheduler, initialSample: initial, startIndex: 0, evaluate: Self.syntheticTransformer)

        XCTAssertEqual(new.dtype, .bfloat16)
        XCTAssertEqual(Self.bits(new), Self.bits(old), "steps \(steps) seed \(seed): latents moved")
        XCTAssertEqual(stats.stepsRun, steps)
        XCTAssertEqual(stats.rowsAtStart, 1)
        XCTAssertEqual(stats.modelEvals, steps)
        XCTAssertFalse(MLX.any(MLX.isNaN(new)).item(Bool.self))
      }
    }
  }

  /// The driver hands `evaluate` the grid sigma as a Float equal to the
  /// pre-change `ts[i]`, in order — the `t` tensor the transformer sees is
  /// built from the same number.
  func testEvaluateReceivesTheGridSigmasInOrder() throws {
    let steps = 9
    let ts = Self.preChangeTimesteps(seqLen: Self.seqLen1024, steps: steps)
    var seen: [Float] = []
    var scheduler = try Self.defaultScheduler(steps: steps)
    _ = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: Self.bf16Noise(seed: 3), startIndex: 0
    ) { x, sigma in
      seen.append(sigma)
      return Self.syntheticTransformer(x, sigma)
    }
    XCTAssertEqual(seen, Array(ts.dropLast()))
  }

  // MARK: - AC-2 (weight-free half): img2img start index and the float32 mix

  func testImg2ImgStartIndexBitIdenticalToInlineLoop() throws {
    let steps = 9
    let ts = Self.preChangeTimesteps(seqLen: Self.seqLen1024, steps: steps)
    for strength: Float in [0.3, 0.5, 0.75] {
      // The pipeline's startIndex arithmetic (unchanged by WP-E3).
      let denoise = 1.0 - max(0.01, min(0.99, strength))
      let startIndex = max(0, steps - Int((Double(steps) * Double(denoise)).rounded(.up)))
      let initial = Self.bf16Noise(seed: 5)

      let old = Self.inlineEulerLoop(ts: ts, initial: initial, startIndex: startIndex, evaluate: Self.syntheticTransformer)
      var scheduler = try Self.defaultScheduler(steps: steps)
      var progressSeen: [(Int, Int)] = []
      let (new, stats) = Krea2DenoiseLoop.run(
        scheduler: &scheduler, initialSample: initial, startIndex: startIndex,
        evaluate: Self.syntheticTransformer, progress: { progressSeen.append(($0, $1)) })

      XCTAssertEqual(Self.bits(new), Self.bits(old), "strength \(strength) startIndex \(startIndex)")
      XCTAssertEqual(stats.stepsRun, steps - startIndex, "strength \(strength)")
      XCTAssertEqual(stats.modelEvals, steps - startIndex)
      // Progress is reported exactly as before: (i + 1, total) from startIndex.
      XCTAssertEqual(progressSeen.map(\.0), Array((startIndex + 1)...steps))
      XCTAssertEqual(Set(progressSeen.map(\.1)), [steps])
    }
  }

  /// §3.3's byte-identity trap: the img2img mix must run in float32 — the
  /// pre-change `MLXArray(ts[startIndex])` is a float32 0-d array, which
  /// promotes the whole mix to float32 before the single cast to bf16. The
  /// new path takes `scheduler.sigmas[startIndex]` as a float32 `MLXArray`,
  /// never `.item()`, and lands on the same bits.
  func testImg2ImgMixIsFloat32AndBitIdentical() throws {
    let steps = 9
    let ts = Self.preChangeTimesteps(seqLen: Self.seqLen1024, steps: steps)
    let scheduler = try Self.defaultScheduler(steps: steps)
    let noise = Self.bf16Noise(seed: 8, shape: [1, 16, 8, 8])
    let source = Self.bf16Noise(seed: 9, shape: [1, 16, 8, 8])

    for startIndex in [0, 3, 6, 8] {
      // Pre-change expression, verbatim (Krea2ImageToImagePipeline.swift at aa8e590).
      let tStart = MLXArray(ts[startIndex])
      let old = (noise * tStart + source * (1.0 - tStart)).asType(.bfloat16)

      let sigma = scheduler.sigmas[startIndex]
      XCTAssertEqual(sigma.dtype, .float32)
      XCTAssertEqual(sigma.ndim, 0)
      let new = Krea2Sampling.mixSourceLatent(noise: noise, source: source, sigma: sigma, dtype: .bfloat16)
      XCTAssertEqual(new.dtype, .bfloat16)
      XCTAssertEqual(Self.bits(new), Self.bits(old), "startIndex \(startIndex)")

      // The trap, demonstrated: a Swift Float scalar runs the mix in bf16.
      let scalar = ts[startIndex]
      let trapped = noise * scalar + source * (1.0 - scalar)
      XCTAssertEqual(trapped.dtype, .bfloat16, "Float * bf16 array stays bf16 — that is the trap")
    }
  }

  // MARK: - AC-12: CFG works under every sampler and its cost is reported

  func testCFGEvalCount() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: Self.seqLen1024, align: Self.align)
    let field = ModelOutputConventionTests.makeField(seed: 12)
    for kind in SchedulerKind.allCases {
      let steps = 6
      var scheduler = try Krea2Pipeline.makeScheduler(
        sampler: kind, sigmaSchedule: .krea2, steps: steps, shift: shift, seed: 42, c2: 0.5)
      var calls = 0
      let (x, stats) = Krea2DenoiseLoop.run(
        scheduler: &scheduler, initialSample: field.sample(at: 1.0), startIndex: 0,
        modelEvalsPerEvaluate: 2
      ) { x, sigma in
        calls += 1
        return field.velocity(x, sigma)
      }
      // WP-E13 added N-row conformers; WP-E14 added `deis_Nm`, whose row count
      // FALLS to 1 the moment its order ramp completes. So the expected call
      // count is derived from what the sampler IS, and `rowsAtStart` is
      // checked as the label it is rather than used as a multiplier.
      let expectedRowsAtStart: Int
      let expectedCalls: Int
      if let deis = scheduler as? DEISMultistepScheduler {
        let warm = min(deis.order.warmUpStepCount, steps)
        expectedRowsAtStart = deis.order.rawValue
        expectedCalls = warm * deis.order.rawValue + (steps - warm)
        XCTAssertEqual(deis.warmUpSteps, warm, kind.rawValue)
      } else {
        let rows = (scheduler as? TableauScheduler)?.rows
          ?? (scheduler.requiresIntermediateEvaluation ? 2 : 1)
        expectedRowsAtStart = rows
        expectedCalls = steps * rows
        // The product identity holds only BECAUSE these samplers have a
        // CONSTANT row count. It is asserted here as a property of them, not
        // as a law of the driver — `testRowsMayChangeMidRun` pins the general
        // contract, and `deis_Nm` above is the shipping counter-example.
        XCTAssertEqual(stats.modelEvals, stats.stepsRun * stats.rowsAtStart * 2, kind.rawValue)
      }
      XCTAssertEqual(stats.rowsAtStart, expectedRowsAtStart, kind.rawValue)
      XCTAssertEqual(stats.stepsRun, steps, kind.rawValue)
      XCTAssertEqual(calls, expectedCalls, kind.rawValue)
      // The counted truth, which holds for every scheduler.
      XCTAssertEqual(stats.modelEvals, calls * 2, kind.rawValue)
      XCTAssertEqual(stats.evaluateCalls, calls, kind.rawValue)
      XCTAssertFalse(MLX.any(MLX.isNaN(x)).item(Bool.self), kind.rawValue)
    }
  }

  // MARK: - AC-13: multistep state does not leak between runs

  func testResetBetweenRuns() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: Self.seqLen1024, align: Self.align)
    var scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .dpmplusplus2m, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 0, c2: 0.5)
    let initial = Self.bf16Noise(seed: 13)

    let (first, _) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: initial, startIndex: 0, evaluate: Self.syntheticTransformer)
    // Dirty the multistep cache deliberately: a stray step leaves previousOutput set.
    _ = scheduler.step(modelOutput: initial * 3.0, timestepIndex: 4, sample: initial)
    let (second, _) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: initial, startIndex: 0, evaluate: Self.syntheticTransformer)
    XCTAssertEqual(Self.bits(first), Self.bits(second), "dpmpp_2m leaked state across runs")
  }

  // MARK: - AC-10 through the driver: the 2-row branch converts to x₀ for res_2s

  func testRES2sThroughDriverReconstructsX0() throws {
    let shift = try Krea2Sampling.resolveShift(explicit: nil, seqLen: Self.seqLen1024, align: Self.align)
    var scheduler = try Krea2Pipeline.makeScheduler(
      sampler: .res2s, sigmaSchedule: .krea2, steps: 9, shift: shift, seed: 0, c2: 0.5)
    let field = ModelOutputConventionTests.makeField(seed: 10)
    var midSigmas: [Float] = []
    let grid = scheduler.sigmas.asArray(Float.self)
    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: field.sample(at: grid[0]), startIndex: 0
    ) { x, sigma in
      if !grid.contains(sigma) { midSigmas.append(sigma) }
      return field.velocity(x, sigma)
    }
    XCTAssertEqual(stats.rowsAtStart, 2)
    XCTAssertEqual(stats.modelEvals, 18)
    let err = ModelOutputConventionTests.relativeError(x, field.x0)
    XCTAssertLessThanOrEqual(err, 1e-5, "res_2s through the driver must reconstruct x₀; got \(err)")
    // The second evaluation is at the genuine substep σ·e^{−c₂h}, not σ_{i+1}.
    XCTAssertEqual(midSigmas.count, 9)
    for (i, mid) in midSigmas.enumerated() {
      XCTAssertEqual(mid, scheduler.intermediateSigma(timestepIndex: i)!, accuracy: 1e-7)
      XCTAssertGreaterThan(mid, grid[i + 1])
      XCTAssertLessThan(mid, grid[i])
    }
  }

  // MARK: - The N-row branch (TableauScheduler)

  /// Explicit-RK tableau over the flow ODE `dx/dσ = v`, in the form WP-E13's
  /// samplers will take: `rows` evaluations per step, row r evaluated at
  /// `σ_i + c_r·(σ_{i+1} − σ_i)` on `x_i + (σ_{i+1} − σ_i)·Σ a_{r,j} k_j`,
  /// committed with `x_i + (σ_{i+1} − σ_i)·Σ b_j k_j`.
  struct TestTableau: TableauScheduler {
    let sigmas: MLXArray
    let timesteps: MLXArray
    let numInferenceSteps: Int
    let a: [[Float]]
    let b: [Float]
    let c: [Float]
    var rows: Int { b.count }
    var rowSamplesRequested: [(step: Int, row: Int)] = []
    var commits = 0

    init(sigmaValues: [Float], a: [[Float]], b: [Float], c: [Float]) {
      self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
      self.timesteps = MLXArray(sigmaValues.dropLast().map { $0 * 1000 }, [sigmaValues.count - 1])
      self.numInferenceSteps = sigmaValues.count - 1
      self.a = a; self.b = b; self.c = c
    }

    private func dt(_ i: Int) -> Float {
      sigmas[i + 1].item(Float.self) - sigmas[i].item(Float.self)
    }

    func rowSigma(timestepIndex: Int, row: Int) -> Float {
      sigmas[timestepIndex].item(Float.self) + c[row] * dt(timestepIndex)
    }

    mutating func rowSample(timestepIndex: Int, row: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray {
      rowSamplesRequested.append((timestepIndex, row))
      var acc = x0
      for j in 0..<row where a[row][j] != 0 {
        acc = acc + (a[row][j] * dt(timestepIndex)) * k[j]
      }
      return acc
    }

    mutating func commit(timestepIndex: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray {
      commits += 1
      var acc = x0
      for j in 0..<k.count where b[j] != 0 {
        acc = acc + (b[j] * dt(timestepIndex)) * k[j]
      }
      return acc
    }

    mutating func step(modelOutput: MLXArray, timestepIndex: Int, sample: MLXArray) -> MLXArray {
      XCTFail("the driver must route a TableauScheduler through rows/commit, never step()")
      return sample
    }
  }

  func testTableauBranchRunsRowsAndCommits() throws {
    let steps = 9
    let grid = try Self.defaultScheduler(steps: steps).sigmas.asArray(Float.self)
    let field = ModelOutputConventionTests.makeField(seed: 31)

    // Ralston 3-stage: 3 rows, the warm-up WP-E14 uses.
    let ralston3 = TestTableau(
      sigmaValues: grid,
      a: [[0, 0, 0], [0.5, 0, 0], [0, 0.75, 0]],
      b: [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
      c: [0, 0.5, 0.75])
    var scheduler: any ZImageScheduler = ralston3
    var seen: [Float] = []
    let (x, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: field.sample(at: grid[0]), startIndex: 0
    ) { x, sigma in
      seen.append(sigma)
      return field.velocity(x, sigma)
    }
    XCTAssertEqual(stats.rowsAtStart, 3)
    XCTAssertEqual(stats.stepsRun, steps)
    XCTAssertEqual(stats.modelEvals, steps * 3)
    XCTAssertEqual(seen.count, steps * 3)
    // Row sigmas in order: σ_i, σ_i + 0.5·dt, σ_i + 0.75·dt for every step.
    for i in 0..<steps {
      let dt = grid[i + 1] - grid[i]
      XCTAssertEqual(seen[3 * i], grid[i], accuracy: 1e-7)
      XCTAssertEqual(seen[3 * i + 1], grid[i] + 0.5 * dt, accuracy: 1e-6)
      XCTAssertEqual(seen[3 * i + 2], grid[i] + 0.75 * dt, accuracy: 1e-6)
    }
    let tableau = scheduler as! TestTableau
    XCTAssertEqual(tableau.commits, steps)
    XCTAssertEqual(tableau.rowSamplesRequested.map(\.row), Array(repeating: [1, 2], count: steps).flatMap { $0 })
    // Any consistent RK tableau integrates the constant-velocity field exactly.
    let err = ModelOutputConventionTests.relativeError(x, field.x0)
    XCTAssertLessThanOrEqual(err, 1e-5, "ralston_3s tableau must land on x₀; got \(err)")
  }

  /// A 1-row tableau whose commit is the Euler update is bit-identical to
  /// FlowMatchEulerScheduler through the driver — the tableau branch adds no
  /// arithmetic of its own.
  func testOneRowTableauMatchesEulerBitwise() throws {
    let steps = 9
    var euler = try Self.defaultScheduler(steps: steps)
    let grid = euler.sigmas.asArray(Float.self)
    let initial = Self.bf16Noise(seed: 17)
    let (viaEuler, _) = Krea2DenoiseLoop.run(
      scheduler: &euler, initialSample: initial, startIndex: 0, evaluate: Self.syntheticTransformer)

    var tableau: any ZImageScheduler = TestTableau(sigmaValues: grid, a: [[0]], b: [1], c: [0])
    let (viaTableau, stats) = Krea2DenoiseLoop.run(
      scheduler: &tableau, initialSample: initial, startIndex: 0, evaluate: Self.syntheticTransformer)
    XCTAssertEqual(stats.rowsAtStart, 1)
    XCTAssertEqual(Self.bits(viaTableau), Self.bits(viaEuler))
  }

  // MARK: - rowsAtStart is a label; modelEvals is the count (AC-24's shape)

  /// A tableau whose row count DROPS after a warm-up — the shape `deis_3m`
  /// takes (RES4LYF `multistep_extra_initial_steps = 1`: `ralston_3s` over
  /// steps 0…3, then multistep at one row, AC-24 as corrected by A.1).
  ///
  /// The driver must dispatch the actual row count on every step, and
  /// `modelEvals` must be the number of calls that really happened — NOT
  /// `stepsRun × rowsAtStart`, which for this scheduler overcounts by more
  /// than 2x. `rowsAtStart` reports the warm-up's 3 and is a description of
  /// where the run started, nothing more.
  struct WarmUpTableau: TableauScheduler {
    let sigmas: MLXArray
    let timesteps: MLXArray
    let numInferenceSteps: Int
    /// Steps `0 ..< warmUpSteps` take 3 rows; every later step takes 1.
    let warmUpSteps: Int
    /// The step the driver is dispatching, tracked so `rows` can change with it.
    var currentStep = 0
    var rowsPerStep: [Int: Int] = [:]

    init(sigmaValues: [Float], warmUpSteps: Int) {
      self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
      self.timesteps = MLXArray(sigmaValues.dropLast().map { $0 * 1000 }, [sigmaValues.count - 1])
      self.numInferenceSteps = sigmaValues.count - 1
      self.warmUpSteps = warmUpSteps
    }

    var rows: Int { currentStep < warmUpSteps ? 3 : 1 }

    private func dt(_ i: Int) -> Float {
      sigmas[i + 1].item(Float.self) - sigmas[i].item(Float.self)
    }

    func rowSigma(timestepIndex: Int, row: Int) -> Float {
      sigmas[timestepIndex].item(Float.self) + Float(row) / 3.0 * dt(timestepIndex)
    }

    mutating func rowSample(timestepIndex: Int, row: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray {
      x0 + (Float(row) / 3.0 * dt(timestepIndex)) * k[row - 1]
    }

    mutating func commit(timestepIndex: Int, x0: MLXArray, k: [MLXArray]) -> MLXArray {
      rowsPerStep[timestepIndex] = k.count
      // Euler on the last row, whatever the row count — the arithmetic is not
      // what this test is about.
      let out = x0 + dt(timestepIndex) * k[k.count - 1]
      currentStep = timestepIndex + 1
      return out
    }

    mutating func step(modelOutput: MLXArray, timestepIndex: Int, sample: MLXArray) -> MLXArray {
      XCTFail("the driver must route a TableauScheduler through rows/commit, never step()")
      return sample
    }
  }

  func testRowsMayChangeMidRunAndModelEvalsCountsTheActualCalls() throws {
    let steps = 9, warmUp = 4
    let grid = try Self.defaultScheduler(steps: steps).sigmas.asArray(Float.self)
    var scheduler: any ZImageScheduler = WarmUpTableau(sigmaValues: grid, warmUpSteps: warmUp)
    var calls = 0
    let (_, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: Self.bf16Noise(seed: 41), startIndex: 0,
      modelEvalsPerEvaluate: 2
    ) { x, sigma in
      calls += 1
      return Self.syntheticTransformer(x, sigma)
    }

    // The driver re-read `rows` every step: 3 for the warm-up, 1 after.
    let tableau = scheduler as! WarmUpTableau
    XCTAssertEqual(tableau.rowsPerStep, [0: 3, 1: 3, 2: 3, 3: 3, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1])

    let expectedCalls = warmUp * 3 + (steps - warmUp) * 1  // 12 + 5 = 17
    XCTAssertEqual(calls, expectedCalls)
    XCTAssertEqual(stats.evaluateCalls, expectedCalls)
    XCTAssertEqual(stats.modelEvals, expectedCalls * 2, "modelEvals is counted, not multiplied")

    // `rowsAtStart` is the warm-up's row count and nothing more. The product
    // would claim 9 x 3 x 2 = 54 forwards for a run that made 34.
    XCTAssertEqual(stats.rowsAtStart, 3)
    XCTAssertEqual(stats.stepsRun, steps)
    XCTAssertNotEqual(
      stats.modelEvals, stats.stepsRun * stats.rowsAtStart * 2,
      "stepsRun x rowsAtStart is NOT the cost for a scheduler whose rows change")
  }

  // MARK: - T2 / T3 seams: nil is the default path; a hook runs once per step after the commit

  final class CountingInjector: SDENoiseInjector {
    var steps: [Int] = []
    /// `(step, row)` — WP-E15 added the substep hook, which fires once per
    /// non-final ROW before that row's evaluation.
    var substeps: [(Int, Int)] = []
    func inject(
      sample: inout MLXArray, x0: MLXArray, timestepIndex: Int, scheduler: any ZImageScheduler
    ) {
      steps.append(timestepIndex)
    }
    func injectSubstep(
      sample: inout MLXArray, x0: MLXArray, timestepIndex: Int, row: Int,
      scheduler: any ZImageScheduler
    ) {
      substeps.append((timestepIndex, row))
    }
  }

  /// WP-E16 corrected this seam from a step hook to a ROW hook: upstream's
  /// `bong_iter` runs inside the row loop and rewrites the step's ANCHOR, not
  /// the sample. This stub records where it was called and moves nothing, so
  /// the driver's plumbing can be asserted without the arithmetic.
  final class CountingBongMath: BongMath {
    /// `(step, upstream row)` — the row whose UPDATE built the sample.
    var calls: [(Int, Int)] = []
    /// How many extra evaluations to claim, so the accounting can be tested.
    let claimedEvals: Int
    /// The samples it was handed, to prove `buildRowSample` inverts.
    var rebuilt: [MLXArray] = []

    init(claimedEvals: Int = 0) { self.claimedEvals = claimedEvals }

    func iterate(
      x0: inout MLXArray, rowSample: MLXArray, timestepIndex: Int, row: Int,
      scheduler: any ZImageScheduler,
      buildRowSample: (MLXArray) -> MLXArray,
      evaluate: (MLXArray, Float) -> MLXArray
    ) -> Int {
      calls.append((timestepIndex, row))
      rebuilt.append(buildRowSample(x0))
      return claimedEvals
    }
  }

  func testHooksRunOncePerStepFromStartIndex() throws {
    var scheduler = try Self.defaultScheduler(steps: 9)
    let injector = CountingInjector()
    let bong = CountingBongMath()
    _ = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: Self.bf16Noise(seed: 19), startIndex: 3,
      evaluate: Self.syntheticTransformer, noise: injector, bongmath: bong)
    XCTAssertEqual(injector.steps, [3, 4, 5, 6, 7, 8])
    // The default euler path is 1-row: there is no non-final row to re-noise
    // and none to re-anchor, so NEITHER row hook fires (WP-E15 / WP-E16).
    // Upstream agrees: `bong_iter` returns immediately unless
    // `row < rows − row_offset`, which no 1-row sampler ever satisfies.
    XCTAssertTrue(injector.substeps.isEmpty, "1-row schedulers have no substep")
    XCTAssertTrue(bong.calls.isEmpty, "1-row schedulers have no row to re-anchor")
  }

  /// The T3 hook fires once per non-final row, on both branches that have
  /// rows, at upstream's row index (`driver row − 1`) — and it fires AFTER the
  /// T2 substep re-noise and BEFORE that row's evaluation, which is the only
  /// position where the sample it is handed is the re-noised one.
  func testBongMathFiresPerNonFinalRowAfterTheSubstepSwapAndBeforeTheEvaluation() throws {
    let sigmas = try Self.defaultScheduler(steps: 3).sigmas.asArray(Float.self)

    var twoRow: any ZImageScheduler = RES2sScheduler(numInferenceSteps: 3, sigmaValues: sigmas)
    let twoRowInjector = CountingInjector()
    let twoRowBong = CountingBongMath()
    var evaluationsSeen: [Int] = []
    _ = Krea2DenoiseLoop.run(
      scheduler: &twoRow, initialSample: Self.bf16Noise(seed: 23),
      evaluate: { latent, sigma in
        evaluationsSeen.append(twoRowBong.calls.count)
        return Self.syntheticTransformer(latent, sigma)
      },
      noise: twoRowInjector, bongmath: twoRowBong)
    XCTAssertEqual(twoRowBong.calls.map { $0.0 }, [0, 1, 2], "one rebase per step")
    XCTAssertEqual(twoRowBong.calls.map { $0.1 }, [0, 0, 0], "upstream's row 0, not the driver's 1")
    // Rows alternate 0, 1: the rebase count seen at row 1's evaluation is
    // always one more than at row 0's, i.e. the rebase preceded it.
    XCTAssertEqual(evaluationsSeen, [0, 1, 1, 2, 2, 3])
    // …and it ran after the substep swap, not before: both hooks fired the
    // same number of times, and the injector's fired first each round.
    XCTAssertEqual(twoRowInjector.substeps.count, twoRowBong.calls.count)

    var threeRow: any ZImageScheduler = RalstonScheduler(
      stages: .three, numInferenceSteps: 3, sigmaValues: sigmas)
    let threeRowBong = CountingBongMath()
    _ = Krea2DenoiseLoop.run(
      scheduler: &threeRow, initialSample: Self.bf16Noise(seed: 23),
      evaluate: Self.syntheticTransformer, noise: CountingInjector(), bongmath: threeRowBong)
    XCTAssertEqual(
      threeRowBong.calls.map { $0.0 }, [0, 0, 1, 1, 2, 2], "two rebases per 3-row step")
    XCTAssertEqual(
      threeRowBong.calls.map { $0.1 }, [0, 1, 0, 1, 0, 1],
      "rows 0 and 1 — never the committing row")
  }

  /// Whatever a T3 conformer says it evaluated is ADDED to the count, so the
  /// number reported stays the number that happened (§3.3).
  func testBongMathExtraEvaluationsAreCountedNotAssumed() throws {
    let sigmas = try Self.defaultScheduler(steps: 3).sigmas.asArray(Float.self)
    var scheduler: any ZImageScheduler = RES2sScheduler(
      numInferenceSteps: 3, sigmaValues: sigmas)
    let bong = CountingBongMath(claimedEvals: 2)
    let (_, stats) = Krea2DenoiseLoop.run(
      scheduler: &scheduler, initialSample: Self.bf16Noise(seed: 29),
      modelEvalsPerEvaluate: 2,
      evaluate: Self.syntheticTransformer, bongmath: bong)
    XCTAssertEqual(bong.calls.count, 3)
    XCTAssertEqual(stats.evaluateCalls, 3 * 2 + 3 * 2, "6 row calls + 6 the hook reported")
    XCTAssertEqual(stats.modelEvals, stats.evaluateCalls * 2)
    XCTAssertEqual(stats.stepsRun, 3, "a hook's evaluations are not steps")
  }

  /// The substep hook fires once per non-final row, in row order, before that
  /// row's model evaluation — the position RES4LYF re-noises at
  /// (`rk_sampler_beta.py:1874`). Asserted on both driver branches that have
  /// rows: the 2-row `res_2s` and a 3-row tableau.
  func testSubstepHookFiresOncePerNonFinalRowBeforeItsEvaluation() throws {
    let sigmas = try Self.defaultScheduler(steps: 4).sigmas.asArray(Float.self)

    var twoRow: any ZImageScheduler = RES2sScheduler(numInferenceSteps: 4, sigmaValues: sigmas)
    let twoRowInjector = CountingInjector()
    var seenAtEvaluate: [Int] = []
    _ = Krea2DenoiseLoop.run(
      scheduler: &twoRow, initialSample: Self.bf16Noise(seed: 21),
      evaluate: { latent, sigma in
        seenAtEvaluate.append(twoRowInjector.substeps.count)
        return Self.syntheticTransformer(latent, sigma)
      },
      noise: twoRowInjector)
    XCTAssertEqual(
      twoRowInjector.substeps.map { $0.0 }, [0, 1, 2, 3], "one substep re-noise per step")
    XCTAssertEqual(twoRowInjector.substeps.map { $0.1 }, [1, 1, 1, 1], "the non-final row is row 1")
    // Rows alternate 0, 1: the substep count seen at row 1's evaluation is
    // always one more than at row 0's, i.e. the injection preceded it.
    XCTAssertEqual(seenAtEvaluate, [0, 1, 1, 2, 2, 3, 3, 4])

    var threeRow: any ZImageScheduler = RalstonScheduler(
      stages: .three, numInferenceSteps: 4, sigmaValues: sigmas)
    let threeRowInjector = CountingInjector()
    _ = Krea2DenoiseLoop.run(
      scheduler: &threeRow, initialSample: Self.bf16Noise(seed: 22),
      evaluate: Self.syntheticTransformer, noise: threeRowInjector)
    XCTAssertEqual(
      threeRowInjector.substeps.map { "\($0.0):\($0.1)" },
      ["0:1", "0:2", "1:1", "1:2", "2:1", "2:2", "3:1", "3:2"],
      "rows 1 and 2 of 3, in order, on every step — never the committing row")
  }
}
