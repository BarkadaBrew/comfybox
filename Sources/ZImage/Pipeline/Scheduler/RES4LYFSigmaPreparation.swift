// RES4LYFSigmaPreparation.swift — RES4LYF's `prepare_sigmas`, ported exactly.
//
// Task S-FIX-1 (codex engine review, "Production RES4LYF execution skips
// `sigma_min` preparation and the final linear tail"; FDD-krea2-raw-recipe
// §3.3, §3.12, Addendum A.1).
//
// A ComfyUI schedule ends at an exact `0.0` sentinel. RES4LYF does NOT solve
// through that zero. Before sampling it rewrites the grid
// (`beta/rk_noise_sampler_beta.py:916`, pinned commit
// `26036f647ca15d3048a193daf99a40cecfc3820d`):
//
//     if sigmas[-1] == 0:
//         if sigmas[-2] < SIGMA_MIN:                       # REPLACE, length kept
//             sigmas[-2] = SIGMA_MIN
//         elif (sigmas[-2] - SIGMA_MIN).abs() > 1e-4:      # INSERT, length + 1
//             sigmas = cat((sigmas[:-1], SIGMA_MIN, sigmas[-1:]))
//
// then runs `len(sigmas) - 2` solver steps — the last one landing ON
// `sigma_min`, never on zero (`beta/rk_sampler_beta.py:480`) — and finally
// converts `sigma_min → 0` with **no model call** at all
// (`beta/rk_sampler_beta.py:2202`):
//
//     x = model_sampling.calculate_denoised(sigma_min, eps, x)     # = x − σ_min·eps
//     eps = (x_0 − x_next) / (σ − σ_next)                          # :1997, the step's own
//
// Concretely, for `res_2s + beta`, 6 steps, `shift = 1.15`: the published tail
// is `0.241540268 → 0`; RES4LYF solves `0.241540268 → 0.000315751` and then
// converts linearly to zero. Solving straight through the zero (what this
// engine did before this task) evaluates the exponential frame's last rows
// against a `1e-8` floor instead of the model's own `sigma_min`, so the final
// model inputs — and the final latent — differ.
//
// This file is the grid half. The model-free conversion itself is
// `Krea2DenoiseLoop`'s (it needs the step's `x₀`/`x_next`), and the schedulers
// carry the conversion sigma forward as `finalConversionSigma`.

import Foundation

/// The grid a RES4LYF sampler actually walks, plus the model-free conversion
/// that finishes it.
///
/// ``solverSigmas`` deliberately does NOT carry the trailing zero: the zero is
/// a sentinel that marks where the schedule ends, and RES4LYF never takes a
/// solver step onto it. It comes back as ``walkedGrid``'s last element so
/// provenance can report the whole thing.
public struct PreparedSigmaGrid: Equatable, Sendable {
  /// The sigmas the solver steps through, `numInferenceSteps + 1` of them,
  /// ending on `sigma_min` when a conversion follows and on the schedule's own
  /// penultimate sigma when one does not.
  public let solverSigmas: [Float]

  /// The sigma the model-free `σ_min → 0` conversion starts from, or `nil`
  /// when this grid ends without one (no trailing zero to convert to, or a
  /// penultimate sigma that upstream leaves alone).
  public let finalConversionSigma: Float?

  /// Steps a scheduler built on this grid will take.
  public var numInferenceSteps: Int { solverSigmas.count - 1 }

  /// The grid as walked end to end — the solver sigmas plus the zero the
  /// conversion lands on. This is what `Krea2RunTrace` reports (AC-26's
  /// provenance obligation: the record must describe the grid that ran).
  public var walkedGrid: [Float] {
    finalConversionSigma != nil ? solverSigmas + [0.0] : solverSigmas
  }
}

/// RES4LYF's `prepare_sigmas`, for the samplers that are ports of RES4LYF.
public enum RES4LYFSigmaPreparation {

  /// Upstream's `(sigmas[-2] - SIGMA_MIN).abs() > 1e-4` gate: a penultimate
  /// sigma already this close to `sigma_min` is left exactly as it is — no
  /// insertion, and therefore no final conversion either (the tail at
  /// `rk_sampler_beta.py:2202` requires `sigmas[-2] == NS.sigma_min`).
  public static let sigmaMinMatchTolerance: Float = 1e-4

  /// Prepare a published schedule for a RES4LYF sampler.
  ///
  /// - Parameters:
  ///   - published: the schedule as its producer emitted it — monotonically
  ///     decreasing with a trailing `0.0` sentinel.
  ///   - sigmaMin: the ACTIVE model-sampling table's first entry
  ///     (`ModelSamplingFlux(shift = mu).sigmas[0]`, `3.1575e-4` at Krea 2's
  ///     registered 1.15; Addendum A.1). Not `1e-8`, and not a constant.
  /// - Returns: the solver grid and the conversion sigma, if any.
  public static func prepare(published: [Float], sigmaMin: Float) -> PreparedSigmaGrid {
    // `consecutive_duplicate_mask` (rk_noise_sampler_beta.py:912). ComfyUI's
    // `beta` already de-duplicates, but a schedule that repeats a sigma would
    // otherwise give `h = 0` and a division by zero in the linear frame.
    var sigmas = dedupeConsecutive(published)

    // No trailing zero: upstream takes `len(sigmas) - 1` steps and runs no
    // tail. Nothing to prepare.
    guard sigmas.count >= 2, sigmas[sigmas.count - 1] == 0 else {
      return PreparedSigmaGrid(solverSigmas: sigmas, finalConversionSigma: nil)
    }

    let penultimate = sigmas[sigmas.count - 2]
    if penultimate < sigmaMin {
      sigmas[sigmas.count - 2] = sigmaMin
    } else if abs(penultimate - sigmaMin) > sigmaMinMatchTolerance {
      sigmas.insert(sigmaMin, at: sigmas.count - 1)
    }

    // The zero is the sentinel the model-free conversion lands on, never a
    // solver target: `num_steps = len(sigmas) - 2` (rk_sampler_beta.py:480).
    var solver = sigmas
    solver.removeLast()

    // Degenerate: dropping the zero would leave no step to take (a one-step
    // schedule whose only sigma was already below σ_min). Upstream would run a
    // zero-step loop; every scheduler here preconditions on at least one step,
    // so leave the schedule untouched rather than construct an empty solver.
    guard solver.count >= 2 else {
      return PreparedSigmaGrid(solverSigmas: dedupeConsecutive(published), finalConversionSigma: nil)
    }

    // `sigmas[-2] == NS.sigma_min` is upstream's own guard on the tail: the
    // "already close enough" branch above leaves a penultimate that is NOT
    // σ_min, and upstream then finishes at that sigma with no conversion.
    let tail = solver[solver.count - 1] == sigmaMin ? sigmaMin : nil
    return PreparedSigmaGrid(solverSigmas: solver, finalConversionSigma: tail)
  }

  /// `torch.cat((True, diff(sigmas) != 0))` — drop each sigma equal to the one
  /// before it.
  static func dedupeConsecutive(_ values: [Float]) -> [Float] {
    var out: [Float] = []
    out.reserveCapacity(values.count)
    for v in values where out.last != v {
      out.append(v)
    }
    return out
  }
}
