import Foundation
import MLX

/// Flow-matching UniPC multi-step scheduler for Wan 2.2 I2V.
///
/// Ported from `wan/utils/fm_solvers_unipc.py` (FlowUniPCMultistepScheduler).
/// A predictor-corrector multi-step ODE solver with order warmup and
/// `lower_order_final` stability mode.
///
/// ## Key Differences from Euler
/// - Multi-step solver (order 2 by default) — uses history of previous model outputs
/// - Predictor-corrector structure (UniP for prediction, UniC for correction)
/// - Maintains `modelOutputs` buffer and `timestepList` history
/// - `lowerOrderFinal` for stability in last few steps
/// - `predictX0 = true` mode with `convertModelOutput` that transforms flow prediction to x0
///
/// ## Sigma Computation
/// ```
/// sigmas = linspace(sigma_max, sigma_min, num_steps + 1)[:-1]
/// sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
/// timesteps = sigmas * num_train_timesteps
/// sigmas = concat([sigmas, [0.0]])
/// ```
public struct FlowUniPCScheduler: ZImageScheduler {

  // MARK: - ZImageScheduler conformance

  public let sigmas: MLXArray
  public let timesteps: MLXArray
  public let numInferenceSteps: Int

  // MARK: - Configuration

  public let solverOrder: Int
  public let predictX0: Bool
  public let lowerOrderFinal: Bool
  public let numTrainTimesteps: Int

  // MARK: - Mutable state

  /// Ring buffer of converted model outputs (x0 predictions).
  private var modelOutputs: [MLXArray?]

  /// Ring buffer of timestep indices used for each stored model output.
  private var timestepList: [Int?]

  /// Number of lower-order steps taken so far (warmup counter).
  private var lowerOrderNums: Int

  /// The sample from the previous step (needed for corrector).
  private var lastSample: MLXArray?

  /// The computed order for the current step.
  private var thisOrder: Int

  /// Internal step counter.
  private var stepIndex: Int

  // MARK: - Init

  /// Creates a FlowUniPC scheduler.
  ///
  /// - Parameters:
  ///   - numInferenceSteps: Number of denoising steps.
  ///   - shift: Noise schedule shift parameter (5.0 for Wan I2V).
  ///   - numTrainTimesteps: Training timesteps (default 1000).
  ///   - solverOrder: Multi-step order (default 2).
  ///   - predictX0: Whether to predict x0 (default true for flow matching).
  ///   - lowerOrderFinal: Use lower order for final steps (default true).
  public init(
    numInferenceSteps: Int,
    shift: Float = 5.0,
    numTrainTimesteps: Int = 1000,
    solverOrder: Int = 2,
    predictX0: Bool = true,
    lowerOrderFinal: Bool = true
  ) {
    precondition(numInferenceSteps > 0, "numInferenceSteps must be positive")
    self.numInferenceSteps = numInferenceSteps
    self.numTrainTimesteps = numTrainTimesteps
    self.solverOrder = solverOrder
    self.predictX0 = predictX0
    self.lowerOrderFinal = lowerOrderFinal

    // Reproduce Python's FlowUniPCMultistepScheduler:
    // __init__(shift=1): sigmas are identity-shifted (no change)
    //   sigma_max = 1 - 1/N = 0.999, sigma_min = 0
    // set_timesteps(steps, shift=5.0):
    //   sigmas = linspace(sigma_max, sigma_min, steps+1)[:-1]
    //   sigmas = shift * sigmas / (1 + (shift-1) * sigmas)
    //   timesteps = sigmas * N
    //   sigmas = concat([sigmas, [0.0]])

    let sigmaMaxBase: Float = 1.0 - 1.0 / Float(numTrainTimesteps)  // 0.999
    let sigmaMinBase: Float = 0.0

    // linspace(0.999, 0.0, steps+1)[:-1] → steps values from 0.999 down
    var sigmaValues = SigmaSchedule.linspace(sigmaMaxBase, sigmaMinBase, count: numInferenceSteps + 1)
    sigmaValues = Array(sigmaValues.dropLast())

    // Apply shift: sigmas = shift * s / (1 + (shift-1) * s)
    sigmaValues = sigmaValues.map { s in
      shift * s / (1.0 + (shift - 1.0) * s)
    }

    // Compute timesteps — truncate to integer to match Python's int64 conversion.
    // Python: self.timesteps = torch.from_numpy(timesteps).to(dtype=torch.int64)
    let timestepValues = sigmaValues.map { Float(Int32($0 * Float(numTrainTimesteps))) }

    // Append trailing zero sentinel
    sigmaValues.append(0.0)

    self.sigmas = MLXArray(sigmaValues, [sigmaValues.count])
    self.timesteps = MLXArray(timestepValues, [timestepValues.count])

    // Initialize mutable state
    self.modelOutputs = Array(repeating: nil, count: solverOrder)
    self.timestepList = Array(repeating: nil, count: solverOrder)
    self.lowerOrderNums = 0
    self.lastSample = nil
    self.thisOrder = 1
    self.stepIndex = 0
  }

  // MARK: - ZImageScheduler Protocol

  /// Perform one denoising step using the UniPC predictor-corrector.
  public mutating func step(
    modelOutput: MLXArray,
    timestepIndex: Int,
    sample: MLXArray
  ) -> MLXArray {
    self.stepIndex = timestepIndex

    // 1. Convert model output (flow prediction -> x0)
    let convertedOutput = convertModelOutput(modelOutput: modelOutput, sample: sample)

    // 2. Apply corrector if we have a previous sample and output
    var correctedSample = sample
    let useCorrector = stepIndex > 0 && lastSample != nil
    if useCorrector {
      correctedSample = multistepUniCBhUpdate(
        thisModelOutput: convertedOutput,
        lastSample: lastSample!,
        thisSample: sample,
        order: thisOrder
      )
    }

    // 3. Shift model output buffer (FIFO ring)
    for i in 0..<(solverOrder - 1) {
      modelOutputs[i] = modelOutputs[i + 1]
      timestepList[i] = timestepList[i + 1]
    }
    modelOutputs[solverOrder - 1] = convertedOutput
    timestepList[solverOrder - 1] = timestepIndex

    // 4. Compute order for this step
    var order: Int
    if lowerOrderFinal {
      order = min(solverOrder, numInferenceSteps - stepIndex)
    } else {
      order = solverOrder
    }
    order = min(order, lowerOrderNums + 1)  // warmup
    precondition(order > 0, "order must be positive")
    self.thisOrder = order

    // 5. Save sample for corrector in next step
    self.lastSample = correctedSample

    // 6. Apply predictor (UniP)
    let prevSample = multistepUniPBhUpdate(
      sample: correctedSample,
      order: order
    )

    // 7. Update warmup counter
    if lowerOrderNums < solverOrder {
      lowerOrderNums += 1
    }

    return prevSample
  }

  public mutating func reset() {
    modelOutputs = Array(repeating: nil, count: solverOrder)
    timestepList = Array(repeating: nil, count: solverOrder)
    lowerOrderNums = 0
    lastSample = nil
    thisOrder = 1
    stepIndex = 0
  }

  // MARK: - Model Output Conversion

  /// Converts flow prediction to x0 prediction.
  ///
  /// For flow matching with predict_x0=True:
  ///   `x0_pred = sample - sigma_t * model_output`
  func convertModelOutput(modelOutput: MLXArray, sample: MLXArray) -> MLXArray {
    let sigma = sigmas[stepIndex]
    if predictX0 {
      return sample - sigma * modelOutput
    } else {
      return sample - (1.0 - sigma) * modelOutput
    }
  }

  // MARK: - UniP Predictor (B(h) version, bh2)

  /// One step of the UniP predictor.
  ///
  /// For order 1: simple first-order update.
  /// For order 2: uses rhos_p = [0.5] (hardcoded, no linalg.solve needed).
  func multistepUniPBhUpdate(
    sample: MLXArray,
    order: Int
  ) -> MLXArray {
    let m0 = modelOutputs[solverOrder - 1]!

    let sigmaT = sigmas[stepIndex + 1]  // next sigma
    let sigmaS0 = sigmas[stepIndex]     // current sigma

    let alphaT = 1.0 - sigmaT
    let alphaS0 = 1.0 - sigmaS0

    let lambdaT = MLX.log(alphaT) - MLX.log(sigmaT)
    let lambdaS0 = MLX.log(alphaS0) - MLX.log(sigmaS0)
    let h = lambdaT - lambdaS0

    let hh = predictX0 ? -h : h
    let hPhi1 = MLX.exp(hh) - 1.0  // e^h - 1
    let B_h = MLX.exp(hh) - 1.0    // bh2: B(h) = expm1(h)

    if predictX0 {
      var xT = sigmaT / sigmaS0 * sample - alphaT * hPhi1 * m0

      if order >= 2 {
        // For order 2: rhos_p = [0.5]
        // D1 = (m_{i-1} - m0) / rk, predRes = rhos_p * D1
        var D1s: [MLXArray] = []
        for i in 1..<order {
          let si = stepIndex - i
          let mi = modelOutputs[solverOrder - 1 - i]!
          let sigmaSi = sigmas[si]
          let alphaSi = 1.0 - sigmaSi
          let lambdaSi = MLX.log(alphaSi) - MLX.log(sigmaSi)
          let rk = (lambdaSi - lambdaS0) / h
          D1s.append((mi - m0) / rk)
        }

        if !D1s.isEmpty {
          // For order 2, rhos_p = [0.5] (from Python)
          let predRes = D1s[0] * 0.5
          xT = xT - alphaT * B_h * predRes
        }
      }

      return xT
    } else {
      var xT = alphaT / alphaS0 * sample - sigmaT * hPhi1 * m0

      if order >= 2 {
        var D1s: [MLXArray] = []
        for i in 1..<order {
          let si = stepIndex - i
          let mi = modelOutputs[solverOrder - 1 - i]!
          let sigmaSi = sigmas[si]
          let alphaSi = 1.0 - sigmaSi
          let lambdaSi = MLX.log(alphaSi) - MLX.log(sigmaSi)
          let rk = (lambdaSi - lambdaS0) / h
          D1s.append((mi - m0) / rk)
        }

        if !D1s.isEmpty {
          let predRes = D1s[0] * 0.5
          xT = xT - sigmaT * B_h * predRes
        }
      }

      return xT
    }
  }

  // MARK: - UniC Corrector (B(h) version, bh2)

  /// One step of the UniC corrector.
  ///
  /// For order 1: rhos_c = [0.5] (from Python).
  /// For order 2: uses explicit 2x2 solve.
  func multistepUniCBhUpdate(
    thisModelOutput: MLXArray,
    lastSample: MLXArray,
    thisSample: MLXArray,
    order: Int
  ) -> MLXArray {
    let m0 = modelOutputs[solverOrder - 1]!

    let sigmaT = sigmas[stepIndex]
    let sigmaS0 = sigmas[stepIndex - 1]

    let alphaT = 1.0 - sigmaT
    let alphaS0 = 1.0 - sigmaS0

    let lambdaT = MLX.log(alphaT) - MLX.log(sigmaT)
    let lambdaS0 = MLX.log(alphaS0) - MLX.log(sigmaS0)
    let h = lambdaT - lambdaS0

    let hh = predictX0 ? -h : h
    let hPhi1 = MLX.exp(hh) - 1.0
    let B_h = MLX.exp(hh) - 1.0

    let D1t = thisModelOutput - m0

    // Solve for rhos_c coefficients matching Python's torch.linalg.solve(R, b)
    let rhosC = solveCorrector(order: order, hh: hh, hPhi1: hPhi1, B_h: B_h, h: h, lambdaS0: lambdaS0)

    if predictX0 {
      var xT = sigmaT / sigmaS0 * lastSample - alphaT * hPhi1 * m0

      if order == 1 {
        // rhos_c = [0.5], single coefficient applied to D1_t
        xT = xT - alphaT * B_h * (rhosC.last! * D1t)
      } else {
        // order >= 2: rhos_c[:-1] weights history D1s, rhos_c[-1] weights D1_t
        var D1s: [MLXArray] = []
        for i in 1..<order {
          let si = stepIndex - (i + 1)
          let mi = modelOutputs[solverOrder - 1 - i]!
          let sigmaSi = sigmas[si]
          let alphaSi = 1.0 - sigmaSi
          let lambdaSi = MLX.log(alphaSi) - MLX.log(sigmaSi)
          let rk = (lambdaSi - lambdaS0) / h
          D1s.append((mi - m0) / rk)
        }

        if !D1s.isEmpty {
          var corrRes = D1s[0] * rhosC[0]
          for j in 1..<D1s.count {
            corrRes = corrRes + D1s[j] * rhosC[j]
          }
          xT = xT - alphaT * B_h * (corrRes + rhosC.last! * D1t)
        } else {
          xT = xT - alphaT * B_h * (rhosC.last! * D1t)
        }
      }

      return xT
    } else {
      var xT = alphaT / alphaS0 * lastSample - sigmaT * hPhi1 * m0

      if order == 1 {
        xT = xT - sigmaT * B_h * (rhosC.last! * D1t)
      } else {
        var D1s: [MLXArray] = []
        for i in 1..<order {
          let si = stepIndex - (i + 1)
          let mi = modelOutputs[solverOrder - 1 - i]!
          let sigmaSi = sigmas[si]
          let alphaSi = 1.0 - sigmaSi
          let lambdaSi = MLX.log(alphaSi) - MLX.log(sigmaSi)
          let rk = (lambdaSi - lambdaS0) / h
          D1s.append((mi - m0) / rk)
        }

        if !D1s.isEmpty {
          var corrRes = D1s[0] * rhosC[0]
          for j in 1..<D1s.count {
            corrRes = corrRes + D1s[j] * rhosC[j]
          }
          xT = xT - sigmaT * B_h * (corrRes + rhosC.last! * D1t)
        } else {
          xT = xT - sigmaT * B_h * (rhosC.last! * D1t)
        }
      }

      return xT
    }
  }

  // MARK: - Corrector Coefficient Solver

  /// Solves for the UniC corrector coefficients `rhos_c` matching Python's
  /// `torch.linalg.solve(R, b)`.
  ///
  /// For order 1: returns `[0.5]`.
  /// For order 2: builds the 2x2 R matrix and 2-element b vector from the
  /// step size ratios and h_phi_k recurrence, then solves via Cramer's rule.
  private func solveCorrector(
    order: Int,
    hh: MLXArray,
    hPhi1: MLXArray,
    B_h: MLXArray,
    h: MLXArray,
    lambdaS0: MLXArray
  ) -> [MLXArray] {
    if order == 1 {
      return [MLXArray(0.5)]
    }

    // Build rks: step size ratios for history entries, plus 1.0 for current
    var rks: [MLXArray] = []
    for i in 1..<order {
      let si = stepIndex - (i + 1)
      let sigmaSi = sigmas[si]
      let alphaSi = 1.0 - sigmaSi
      let lambdaSi = MLX.log(alphaSi) - MLX.log(sigmaSi)
      let rk = (lambdaSi - lambdaS0) / h
      rks.append(rk)
    }
    rks.append(MLXArray(1.0))

    // Build R matrix (order x order) and b vector (order elements)
    // Python: for i in range(1, order+1):
    //   R.append(torch.pow(rks, i-1))
    //   b.append(h_phi_k * factorial_i / B_h)
    //   factorial_i *= (i+1)
    //   h_phi_k = h_phi_k / hh - 1/factorial_i
    var hPhiK = hPhi1 / hh - 1.0
    var factorialI: Float = 1.0

    var bVec: [MLXArray] = []
    var R: [[MLXArray]] = []
    for i in 1...order {
      var row: [MLXArray] = []
      for rk in rks {
        row.append(MLX.pow(rk, MLXArray(Float(i - 1))))
      }
      R.append(row)
      bVec.append(hPhiK * MLXArray(factorialI) / B_h)
      factorialI *= Float(i + 1)
      hPhiK = hPhiK / hh - MLXArray(1.0 / factorialI)
    }

    // For order 2: solve 2x2 system R * rhos = b via Cramer's rule
    // det = R[0][0]*R[1][1] - R[0][1]*R[1][0]
    // rhos[0] = (b[0]*R[1][1] - b[1]*R[0][1]) / det
    // rhos[1] = (R[0][0]*b[1] - R[1][0]*b[0]) / det
    precondition(order == 2, "UniC corrector only supports order <= 2")
    let det = R[0][0] * R[1][1] - R[0][1] * R[1][0]
    let rho0 = (bVec[0] * R[1][1] - bVec[1] * R[0][1]) / det
    let rho1 = (R[0][0] * bVec[1] - R[1][0] * bVec[0]) / det
    return [rho0, rho1]
  }
}
