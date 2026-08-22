// DEISCoefficients.swift — RES4LYF's `rhoab` DEIS coefficients, closed form.
//
// WP-E14 (docs/FDD-krea2-raw-recipe.md §3.12, AC-23). Port source: RES4LYF
// `beta/deis_coefficients.py:86-121` at the pinned commit
// `26036f647ca15d3048a193daf99a40cecfc3820d` — the `deis_mode == "rhoab"`
// branch, which is the one `rk_coefficients_beta.py:1480` calls:
//
//     coeff_list = get_deis_coeff_list(sigmas, multistep_stages + 1, deis_mode="rhoab")
//
// The PRD assumed ComfyUI's `comfy/k_diffusion/deis.py`; Engine C established
// from the workflow JSON that both sampler nodes are RES4LYF's
// `ClownsharKSampler_Beta`, and RES4LYF asks for `rhoab`. That branch is a
// CLOSED FORM: no `edm2t` VP change of variable, no autograd, no 10 000-point
// quadrature. Only the `tab` branch needs any of that, and nothing here calls
// it.
//
// WHAT THE COEFFICIENTS ARE. At step `i` of order `p`, the interpolation nodes
// are the sigmas `σᵢ, σᵢ₋₁, …, σᵢ₋ₚ₊₁` and `coeffⱼ` is the integral over
// `[σᵢ, σᵢ₊₁]` of the Lagrange basis polynomial that is 1 at node `j` and 0 at
// the others. Two consequences are load-bearing and both are asserted:
//
//   * `Σⱼ coeffⱼ = σᵢ₊₁ − σᵢ = h` — the basis polynomials sum to the constant
//     1, whose integral is the interval length. This is why a constant data
//     prediction integrates EXACTLY (`DEISMultistepSchedulerTests`).
//   * The order RAMP is upstream's `order = min(i + 1, max_order)`, with
//     `order == 1` contributing NO coefficients (the caller falls back to a
//     one-row method). RES4LYF's comment `#fixed order calcs` marks its
//     divergence from ComfyUI core's `min(i, max_order)`; ours is RES4LYF's.
//
// `prev_t` is built as `t_steps[[i - k for k in range(order + 1)]]` upstream —
// `order + 1` entries of which only `0 ..< order` are ever read. The extra
// entry is what would index negatively (torch wrapping to the tail) at small
// `i`; because `order ≤ i + 1` the entries that ARE read are always in range,
// so this port builds only those and needs no wrap-around.

import Foundation

/// RES4LYF's `get_deis_coeff_list(..., deis_mode="rhoab")`.
///
/// Everything here is `Double` (AC-23: "computed in `Double`"); upstream runs
/// it in float64 too, on sigma values that arrive as float32.
enum DEISCoefficients {

  /// Upstream's `max_order` ceiling. `1 <= max_order <= 4`, and order 1 is the
  /// degenerate "no coefficients" case.
  static let maximumOrder = 4

  /// The whole list, one entry per step `i ∈ 0 ..< sigmas.count − 1`, exactly
  /// as upstream builds it — including the empty entries the ramp produces.
  ///
  /// Upstream is handed the sampler's full sigma array (ending at the `0`
  /// sentinel). Entry `i` reads only `sigmas[i + 1]` and `sigmas[i - k]`, so a
  /// solver grid that has had its sentinel dropped (`RES4LYFSigmaPreparation`)
  /// produces identical entries for every index it still has.
  static func rhoabList(sigmas: [Double], maxOrder: Int) -> [[Double]] {
    precondition(sigmas.count >= 2, "a DEIS coefficient list needs at least one interval")
    return (0..<(sigmas.count - 1)).map { rhoab(sigmas: sigmas, index: $0, maxOrder: maxOrder) }
  }

  /// One entry of that list. The scheduler calls this rather than building the
  /// whole table per step; ``rhoabList(sigmas:maxOrder:)`` is the same function
  /// mapped, and the two are pinned equal by test.
  ///
  /// - Parameters:
  ///   - sigmas: the sampler's sigma grid, descending.
  ///   - index: `i`, the step whose interval `[σᵢ, σᵢ₊₁]` is integrated over.
  ///   - maxOrder: `multistep_stages + 1` upstream — 2, 3 or 4.
  /// - Returns: `min(i + 1, maxOrder)` coefficients, ordered
  ///   `[current, previous, previous², previous³]`; empty when that order is 1.
  static func rhoab(sigmas: [Double], index i: Int, maxOrder: Int) -> [Double] {
    precondition(maxOrder >= 1 && maxOrder <= maximumOrder, "max_order must be 1...4")
    precondition(i >= 0 && i + 1 < sigmas.count, "step \(i) outside the grid")

    // `order = min(i+1, max_order)` — RES4LYF's corrected ramp (ComfyUI core
    // uses `min(i, max_order)`, which is off by one and is not what runs).
    let order = min(i + 1, maxOrder)
    if order == 1 { return [] }

    let tCur = sigmas[i]
    let tNext = sigmas[i + 1]
    // `prev_t[k] = t_steps[i - k]`; `order ≤ i + 1` keeps every read in range.
    let prev = (0..<order).map { sigmas[i - $0] }

    switch order {
    case 2:
      let p1 = prev[1]
      let coeffCur =
        ((tNext - p1) * (tNext - p1) - (tCur - p1) * (tCur - p1)) / (2 * (tCur - p1))
      let coeffPrev1 = (tNext - tCur) * (tNext - tCur) / (2 * (p1 - tCur))
      return [coeffCur, coeffPrev1]

    case 3:
      let p1 = prev[1], p2 = prev[2]
      return [
        definiteIntegral2(a: p1, b: p2, start: tCur, end: tNext, c: tCur),
        definiteIntegral2(a: tCur, b: p2, start: tCur, end: tNext, c: p1),
        definiteIntegral2(a: tCur, b: p1, start: tCur, end: tNext, c: p2),
      ]

    default:
      let p1 = prev[1], p2 = prev[2], p3 = prev[3]
      return [
        definiteIntegral3(a: p1, b: p2, c: p3, start: tCur, end: tNext, d: tCur),
        definiteIntegral3(a: tCur, b: p2, c: p3, start: tCur, end: tNext, d: p1),
        definiteIntegral3(a: tCur, b: p1, c: p3, start: tCur, end: tNext, d: p2),
        definiteIntegral3(a: tCur, b: p1, c: p2, start: tCur, end: tNext, d: p3),
      ]
    }
  }

  /// `get_def_integral_2` — `∫ (τ−a)(τ−b) dτ / ((c−a)(c−b))` over
  /// `[start, end]`, expanded exactly as upstream expands it so the float64
  /// rounding is the same.
  private static func definiteIntegral2(
    a: Double, b: Double, start: Double, end: Double, c: Double
  ) -> Double {
    let coeff =
      (end * end * end - start * start * start) / 3
      - (end * end - start * start) * (a + b) / 2
      + (end - start) * a * b
    return coeff / ((c - a) * (c - b))
  }

  /// `get_def_integral_3` — the cubic sibling.
  private static func definiteIntegral3(
    a: Double, b: Double, c: Double, start: Double, end: Double, d: Double
  ) -> Double {
    let e2 = end * end, s2 = start * start
    let e3 = e2 * end, s3 = s2 * start
    let e4 = e3 * end, s4 = s3 * start
    let coeff =
      (e4 - s4) / 4
      - (e3 - s3) * (a + b + c) / 3
      + (e2 - s2) * (a * b + a * c + b * c) / 2
      - (end - start) * a * b * c
    return coeff / ((d - a) * (d - b) * (d - c))
  }
}
