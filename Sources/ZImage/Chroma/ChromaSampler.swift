import Foundation
import MLX
import MLXRandom

/// Scheduler type for Chroma denoising.
public enum ChromaSchedulerType: String, Sendable {
    /// Standard linear flow matching (default for Chroma).
    case euler
    /// Heun (2nd-order) with time-shifted sigmas.
    case heun
    /// Beta-distributed sigmas — pair with Heun for flash-heun LoRA.
    case beta
}

/// Chroma-specific flow matching sampler.
///
/// Supports multiple scheduling strategies:
/// - **Euler**: Standard first-order flow matching with time-shifted sigmas.
/// - **Heun**: Second-order (trapezoidal rule) for better accuracy at fewer steps.
/// - **Beta**: Beta-distributed sigma schedule — designed for flash-heun LoRA at 8 steps.
///
/// Uses time-shifted sigmas with linear interpolation for mu based on
/// image sequence length. Different from Flux 1/2 scheduling.
public struct ChromaSampler {
    public let baseShift: Float
    public let maxShift: Float

    public init(baseShift: Float = 0.5, maxShift: Float = 1.15) {
        self.baseShift = baseShift
        self.maxShift = maxShift
    }

    // MARK: - Timestep Generation

    /// Compute time-shifted timesteps for Chroma (standard linear schedule).
    public func timesteps(numSteps: Int, imageSequenceLength: Int, start: Float = 1.0, stop: Float = 0.0) -> [Float] {
        let mu = computeMu(imageSequenceLength: imageSequenceLength)
        let raw = linearTimesteps(numSteps: numSteps, start: start, stop: stop)
        return raw.map { timeShift($0, mu: mu) }
    }

    /// Compute timesteps with beta distribution schedule.
    ///
    /// Beta schedule concentrates steps where they matter most — early denoising
    /// gets more steps, late cleanup gets fewer. Paired with Heun sampler for
    /// the flash-heun LoRA (8 steps, CFG 1).
    ///
    /// - Parameters:
    ///   - numSteps: Number of denoising steps.
    ///   - imageSequenceLength: Latent sequence length for mu computation.
    ///   - alpha: Beta distribution alpha parameter (default 0.6).
    ///   - beta: Beta distribution beta parameter (default 0.6).
    /// - Returns: Array of `numSteps + 1` timestep values.
    public func betaTimesteps(
        numSteps: Int,
        imageSequenceLength: Int,
        alpha: Float = 0.6,
        beta: Float = 0.6
    ) -> [Float] {
        let mu = computeMu(imageSequenceLength: imageSequenceLength)

        // Generate beta-distributed timesteps using inverse CDF (quantile function).
        // We approximate using regularized incomplete beta function via power transform.
        var timesteps: [Float] = []
        for i in 0...numSteps {
            let u = 1.0 - Float(i) / Float(numSteps)  // 1.0 → 0.0
            let betaVal = betaQuantile(u, alpha: alpha, beta: beta)
            timesteps.append(timeShift(betaVal, mu: mu))
        }
        return timesteps
    }

    // MARK: - Stepping

    /// Euler step: x_t + (t_prev - t) * pred
    public func step(pred: MLXArray, xT: MLXArray, t: Float, tPrev: Float) -> MLXArray {
        xT.asType(pred.dtype) + MLXArray(tPrev - t) * pred
    }

    /// Heun step (second-order): requires two model evaluations.
    ///
    /// 1. Compute d1 = pred at current timestep
    /// 2. Euler step to get intermediate sample: x_mid = x + dt * d1
    /// 3. Evaluate model at intermediate to get d2
    /// 4. Average: x_next = x + dt * (d1 + d2) / 2
    ///
    /// The caller provides both predictions; this method does the final combination.
    public func heunStep(
        pred1: MLXArray,
        pred2: MLXArray,
        xT: MLXArray,
        t: Float,
        tPrev: Float
    ) -> MLXArray {
        let dt = MLXArray(tPrev - t)
        let avgPred = (pred1 + pred2) / 2.0
        return xT.asType(avgPred.dtype) + dt * avgPred
    }

    /// Euler intermediate step for Heun: step to next timestep to get intermediate sample.
    public func eulerIntermediate(pred: MLXArray, xT: MLXArray, t: Float, tPrev: Float) -> MLXArray {
        step(pred: pred, xT: xT, t: t, tPrev: tPrev)
    }

    // MARK: - Prior Sampling

    /// Sample from prior (standard normal)
    public func samplePrior(shape: [Int], seed: UInt64? = nil) -> MLXArray {
        let key = seed.map { MLXRandom.key($0) }
        if let key = key {
            return MLXRandom.normal(shape, key: key).asType(.bfloat16)
        }
        return MLXRandom.normal(shape).asType(.bfloat16)
    }

    // MARK: - Private

    /// Compute mu for time-shifting based on image sequence length.
    private func computeMu(imageSequenceLength: Int) -> Float {
        let x1: Float = 256.0, y1 = baseShift
        let x2: Float = 4096.0, y2 = maxShift
        let m = (y2 - y1) / (x2 - x1)
        let b = y1 - m * x1
        return m * Float(imageSequenceLength) + b
    }

    /// Linear timesteps from start to stop.
    private func linearTimesteps(numSteps: Int, start: Float, stop: Float) -> [Float] {
        (0...numSteps).map { i in
            start + (stop - start) * Float(i) / Float(numSteps)
        }
    }

    /// Apply logit-normal time shift.
    private func timeShift(_ t: Float, mu: Float) -> Float {
        if t <= 0.0 { return 0.0 }
        if t >= 1.0 { return 1.0 }
        return exp(mu) / (exp(mu) + pow(1.0 / t - 1.0, 1.0))
    }

    /// Approximate beta distribution quantile function (inverse CDF).
    ///
    /// Uses Newton's method on the regularized incomplete beta function
    /// approximation. For the parameters used in diffusion scheduling
    /// (alpha ≈ 0.6, beta ≈ 0.6), this converges in ~10 iterations.
    private func betaQuantile(_ p: Float, alpha a: Float, beta b: Float) -> Float {
        guard p > 0 else { return 0.0 }
        guard p < 1 else { return 1.0 }

        // Initial estimate using normal approximation
        var x = p

        // Newton iterations
        for _ in 0..<20 {
            let cdf = regularizedBeta(x, a: a, b: b)
            let pdf = betaPDF(x, a: a, b: b)
            guard pdf > 1e-10 else { break }
            let delta = (cdf - p) / pdf
            x -= delta
            x = max(1e-8, min(1.0 - 1e-8, x))
            if abs(delta) < 1e-7 { break }
        }

        return x
    }

    /// Beta distribution PDF.
    private func betaPDF(_ x: Float, a: Float, b: Float) -> Float {
        guard x > 0 && x < 1 else { return 0 }
        let logPDF = (a - 1) * log(x) + (b - 1) * log(1 - x) - logBeta(a, b)
        return exp(logPDF)
    }

    /// Regularized incomplete beta function (numerical approximation).
    /// Uses continued fraction expansion for good convergence.
    private func regularizedBeta(_ x: Float, a: Float, b: Float) -> Float {
        guard x > 0 else { return 0 }
        guard x < 1 else { return 1 }

        // Use the symmetry relation if x > (a+1)/(a+b+2)
        if x > (a + 1) / (a + b + 2) {
            return 1 - regularizedBeta(1 - x, a: b, b: a)
        }

        // Continued fraction (Lentz's method)
        let logPrefix = a * log(x) + b * log(1 - x) - logBeta(a, b) - log(a)
        let prefix = exp(logPrefix)

        var f: Float = 1
        var c: Float = 1
        var d: Float = 1 - (a + b) * x / (a + 1)
        if abs(d) < 1e-30 { d = 1e-30 }
        d = 1 / d
        f = d

        for m in 1...100 {
            let mf = Float(m)

            // Even step
            var num = mf * (b - mf) * x / ((a + 2 * mf - 1) * (a + 2 * mf))
            d = 1 + num * d
            if abs(d) < 1e-30 { d = 1e-30 }
            c = 1 + num / c
            if abs(c) < 1e-30 { c = 1e-30 }
            d = 1 / d
            f *= c * d

            // Odd step
            num = -(a + mf) * (a + b + mf) * x / ((a + 2 * mf) * (a + 2 * mf + 1))
            d = 1 + num * d
            if abs(d) < 1e-30 { d = 1e-30 }
            c = 1 + num / c
            if abs(c) < 1e-30 { c = 1e-30 }
            d = 1 / d
            let delta = c * d
            f *= delta

            if abs(delta - 1) < 1e-7 { break }
        }

        return prefix * f
    }

    /// Log of the beta function B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b).
    private func logBeta(_ a: Float, _ b: Float) -> Float {
        lgammaf(a) + lgammaf(b) - lgammaf(a + b)
    }
}
