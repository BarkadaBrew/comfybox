import Foundation

// comfybox#307 (point 4): the two-stage refine's skip conditions used to be
// two ad hoc `if`/`guard` branches duplicated in both refine call sites (the
// T2V inline pass and the shared `applyTwoStageRefine` for I2V/continuation
// chunks) — one of them (`self.upsampler == nil`) logged nothing at all, and
// the other (the volume gate) logged only an `.info` line nothing else could
// see. A caller who turned on `two_stage` got a phantom single-pass render
// with no signal anywhere in `/health`, the render trace, or the job status
// that the refine never ran. Extracted here as a pure decision so the skip
// reasons are unit-testable without model weights and consistent between the
// two call sites.

/// Outcome of `LTX2RefineGate.decide`.
public enum LTX2RefineGateDecision: Equatable, Sendable {
  /// `two_stage` was not requested — nothing to report; this is normal
  /// single-pass operation, not a skip.
  case notRequested
  /// `two_stage` was requested and the refine denoise runs normally.
  case run
  /// `two_stage` was requested but the refine could not run. `reason` is a
  /// short, greppable, machine-parseable string suitable for a log line and
  /// for the render's `refine_skipped` trace/status field.
  case skip(reason: String)
}

public enum LTX2RefineGate {
  /// Pure gate decision mirroring the two skip conditions that guard the
  /// refine denoise:
  ///   1. The refine machinery isn't available — `two_stage` resolved true
  ///      but no upsampler is loaded (missing/invalid `LTX2_UPSAMPLER_PATH`,
  ///      or the per-request lazy load — Codex finding #18 — hasn't run yet).
  ///   2. The volume gate — the pre-refine latent volume (at the resolved
  ///      `refine_scale`) exceeds `refine_max_vol`, so running the refine
  ///      denoise at that resolution would risk an OOM.
  ///
  /// `preVolume`/`maxVolume` are ignored unless `twoStage` is true and the
  /// upsampler is loaded — matching the existing call sites, which compute
  /// the volume gate only after confirming the refine is otherwise eligible.
  public static func decide(
    twoStage: Bool, upsamplerLoaded: Bool, preVolume: Int, maxVolume: Int
  ) -> LTX2RefineGateDecision {
    guard twoStage else { return .notRequested }
    guard upsamplerLoaded else {
      return .skip(reason: "upsampler_unavailable (two_stage requested but no upsampler loaded — check LTX2_UPSAMPLER_PATH)")
    }
    guard preVolume <= maxVolume else {
      return .skip(reason: "volume_gate (pre-refine volume \(preVolume) > refine_max_vol \(maxVolume))")
    }
    return .run
  }
}
