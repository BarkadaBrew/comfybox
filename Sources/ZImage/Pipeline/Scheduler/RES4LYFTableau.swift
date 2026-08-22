// RES4LYFTableau.swift — the shared N-row explicit-tableau machinery
//
// WP-E13 (docs/FDD-krea2-raw-recipe.md §3.12, D1, D20). One driver core for
// every RES4LYF tableau conformer: `RalstonScheduler` (linear frame) and
// `RES3sScheduler` (exponential frame) differ only in `frame` and in how their
// Butcher coefficients are produced. `Krea2DenoiseLoop` dispatches both through
// the E3 `TableauScheduler` protocol.
//
// Port source: RES4LYF at the commit E18 pinned its fixtures to,
// `26036f647ca15d3048a193daf99a40cecfc3820d` —
//   * `beta/rk_coefficients_beta.py` (the tableaus, `get_rk_methods_beta`),
//   * `beta/phi_functions.py` (`Phi.__call__`, `phi_mpmath_series`),
//   * `beta/rk_method_beta.py` semantics for the epsilon anchoring, verified
//     numerically against E18's step traces rather than read off the page.
//
// THE ANCHORING, which is the whole subtlety of this file.
//
// RES4LYF's model call returns `denoised` (the data prediction x₀), and the
// row derivative it feeds the tableau is anchored at the STEP's start sample
// `x_0` and the STEP's sigma — not at the row's own sample and substep sigma
// (`noise_anchor = 1.0`):
//
//     linear       εⱼ = (x_0 − denoisedⱼ) / σ            h = σ' − σ
//     exponential  εⱼ = denoisedⱼ − x_0                  h = −log(σ'/σ)
//
//     x_row = x_0 + h · Σⱼ aᵢⱼ εⱼ        x' = x_0 + h · Σⱼ bⱼ εⱼ
//
// Both frames are therefore `.dataPrediction` conformers — including the
// linear ralstons, whose name suggests otherwise. Handed a row-local velocity
// instead, the linear frame is off by 0.17 at row 1 of the first traced step;
// this is checked, not assumed (`RalstonTraceParityTests`).
//
// In the exponential frame the anchoring is exact: because
// `Σⱼ aᵢⱼ = cᵢ·φ₁(−cᵢh)` (RES4LYF's `gen_first_col_exp`), the update is
// algebraically `x_row = e^{−cᵢh}·x_0 + h·Σⱼ aᵢⱼ·denoisedⱼ`, the textbook
// exponential-RK step. In the linear frame it is NOT: it costs the ralston
// family its classical order (they are all second order in practice). That is
// upstream's behaviour and the recipe's behaviour, so it is ours —
// `ExplicitRKSchedulerTests` measures it in both directions so the loss can
// never be mistaken for a transcription bug.

import Foundation
import MLX

/// Which frame a tableau conformer integrates in — RES4LYF's `EXPONENTIAL`
/// flag, which selects both the step size and the epsilon anchoring.
public enum TableauFrame: Sendable, Equatable {
  /// `h = σ' − σ`, row sigmas `σ + cᵣh`, `ε = (x₀ − denoised)/σ`.
  /// `ralston_2s/3s/4s`, and the DEIS multistep warm-up (WP-E14).
  case linear
  /// `h = −log(σ'/σ)`, row sigmas `σ·e^{−cᵣh}`, `ε = denoised − x₀`.
  /// `res_2s`, `res_3s`.
  case exponential
}

/// Frame arithmetic and the φ functions, shared by every tableau conformer.
/// `internal` on purpose: the conformers are the public surface.
enum RES4LYFTableau {

  /// RES4LYF clamps a sigma to a small positive value before taking a log;
  /// `RES2sScheduler` already uses this floor and the exponential frame here
  /// matches it so `res_2s` and `res_3s` treat a σ' = 0 grid tail identically.
  static let sigmaFloor: Float = 1e-8

  /// `NS.h`: `σ' − σ` in the linear frame, `−log(σ'/σ)` in the exponential one.
  ///
  /// The linear frame does NOT clamp `σ'`: a grid whose last entry is 0 gives
  /// `h = −σ`, which is the correct final Euler step onto the data prediction.
  /// The exponential frame has no finite `h` there and clamps, as `res_2s` does.
  static func stepSize(frame: TableauFrame, sigma: Float, sigmaNext: Float) -> Float {
    switch frame {
    case .linear:
      return sigmaNext - sigma
    case .exponential:
      let s = max(sigma, sigmaFloor)
      let sNext = max(sigmaNext, sigmaFloor)
      return -logf(sNext / s)
    }
  }

  /// The sigma row `c` of a step is evaluated at (`substep_sigmas`).
  static func rowSigma(frame: TableauFrame, sigma: Float, h: Float, c: Float) -> Float {
    switch frame {
    case .linear: return sigma + c * h
    case .exponential: return sigma * expf(-c * h)
    }
  }

  /// RES4LYF's x₀-anchored epsilon for one row's data prediction.
  static func epsilon(
    frame: TableauFrame, dataPrediction: MLXArray, x0: MLXArray, sigma: Float
  ) -> MLXArray {
    switch frame {
    case .linear:
      return (x0 - dataPrediction) / scalar(max(sigma, sigmaFloor), like: x0)
    case .exponential:
      return dataPrediction - x0
    }
  }

  /// `x₀ + h·Σⱼ wⱼ·εⱼ` — one tableau row or the step commit.
  ///
  /// Weights are `Double` (the coefficients are computed there); the sum is
  /// accumulated in the sample's dtype and scaled by `h` once at the end,
  /// mirroring upstream's `x_0 + h * sum(...)` rather than folding `h` into
  /// each weight.
  static func advance(
    frame: TableauFrame, x0: MLXArray, dataPredictions: [MLXArray],
    weights: [Double], h: Float, sigma: Float
  ) -> MLXArray {
    var accumulator: MLXArray?
    for (j, w) in weights.enumerated() where w != 0 {
      // A nonzero weight on a row that has not been evaluated yet is a broken
      // tableau (not strictly lower-triangular), never something to skip.
      precondition(
        j < dataPredictions.count,
        "tableau weight \(w) at column \(j) needs an output this row does not have "
          + "(\(dataPredictions.count) available)")
      let eps = epsilon(
        frame: frame, dataPrediction: dataPredictions[j], x0: x0, sigma: sigma)
      let term = scalar(Float(w), like: x0) * eps
      accumulator = accumulator.map { $0 + term } ?? term
    }
    guard let accumulator else { return x0 }
    return x0 + scalar(h, like: x0) * accumulator
  }

  /// `φⱼ(z) = Σ_{k≥0} z^k/(k+j)!` — `phi_functions.py`'s `phi_mpmath_series`,
  /// which is the branch upstream takes by default (`use_analytic_solution`
  /// defaults on). Evaluated in `Double`.
  ///
  /// The series is used for `|z| < 1/2`, where it converges in a handful of
  /// terms and has no cancellation; above that the recurrence
  /// `φ_{j+1}(z) = (φⱼ(z) − 1/j!)/z` from `φ₁(z) = (e^z − 1)/z` is exact to
  /// machine precision and does not overflow for the large `h` a σ' → 0 tail
  /// produces. Upstream's own closed form (incomplete gamma) is the one branch
  /// NOT reproduced: it loses ~1e-12 near zero, and it is not the default.
  static func phi(_ j: Int, _ z: Double) -> Double {
    precondition(j > 0, "φ is defined for j ≥ 1")
    if abs(z) < 0.5 {
      var term = 1.0 / factorial(j)
      var sum = 0.0
      for k in 0..<80 {
        sum += term
        term = term * z / Double(k + j + 1)
        if abs(term) < 1e-30 * max(abs(sum), 1e-30) { break }
      }
      return sum
    }
    var value = (Foundation.exp(z) - 1.0) / z
    for i in 1..<max(j, 1) {
      value = (value - 1.0 / factorial(i)) / z
    }
    return value
  }

  private static func factorial(_ n: Int) -> Double {
    var out = 1.0
    if n > 1 { for i in 2...n { out *= Double(i) } }
    return out
  }

  /// `Float * MLXArray` would cast the scalar to the array's dtype anyway;
  /// going through `MLXArray(_:).asType` is what the existing schedulers do
  /// and keeps bf16 samples in bf16.
  static func scalar(_ value: Float, like sample: MLXArray) -> MLXArray {
    MLXArray(value).asType(sample.dtype)
  }
}
