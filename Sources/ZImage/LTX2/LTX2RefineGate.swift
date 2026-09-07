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

  /// The refine scale the pipeline actually uses: the resolved value clamped
  /// to [1, 2] (`LTX2Pipeline`). One definition, so the gate, the pipeline and
  /// the server's prediction cannot disagree about what "1.35" means.
  public static func clampedRefineScale(_ raw: Float) -> Float {
    max(1.0, min(2.0, raw))
  }

  /// The pre-refine latent volume the volume gate compares against
  /// `refine_max_vol`, in LATENT units.
  ///
  /// comfybox#405 (review round 3, item 1): this arithmetic used to be
  /// duplicated inline at both refine call sites in `LTX2Pipeline`, which
  /// meant the server had no way to ask "will the refine run?" without
  /// re-implementing it — so `refineWillSkip` defaulted to false and the
  /// predicted output size was wrong for exactly the renders that skip. Both
  /// pipeline sites and `WarmServer.prepareLocalVideo` now call this.
  ///
  /// The learned upsampler is fixed 2x, but the refine DENOISE runs at
  /// `rScale`, so the gate reflects the real refine size (Todd 2026-08-07).
  public static func preRefineVolume(
    latentFrames: Int, latentHeight: Int, latentWidth: Int, refineScale: Float
  ) -> Int {
    let s = scaledLatentDims(
      latentHeight: latentHeight, latentWidth: latentWidth, refineScale: refineScale)
    return max(1, latentFrames) * s.height * s.width
  }

  /// The latent grid the refine denoise runs on — the 2x upsampled latent
  /// resized to `refine_scale`. The pipeline resizes to exactly these dims,
  /// and the volume gate above measures exactly these dims; one definition so
  /// the two cannot drift apart (comfybox#405 review round 3).
  public static func scaledLatentDims(
    latentHeight: Int, latentWidth: Int, refineScale: Float
  ) -> (height: Int, width: Int) {
    let rScale = clampedRefineScale(refineScale)
    return (
      height: max(1, Int((Float(latentHeight) * rScale).rounded())),
      width: max(1, Int((Float(latentWidth) * rScale).rounded()))
    )
  }

  /// The same volume from PIXEL dims — what a caller outside the pipeline
  /// (the warm server, deciding what size to predict) has in hand.
  public static func preRefineVolume(
    width: Int, height: Int, frames: Int, refineScale: Float,
    spatialCompression: Int = 32, temporalCompression: Int = 8
  ) -> Int {
    let comp = max(1, spatialCompression)
    let tComp = max(1, temporalCompression)
    return preRefineVolume(
      latentFrames: (max(1, frames) - 1) / tComp + 1,
      latentHeight: max(1, height / comp),
      latentWidth: max(1, width / comp),
      refineScale: refineScale)
  }

  /// Whether the refine will be SKIPPED for a render of these pixel dims —
  /// the question the warm server needs answered before the render starts, in
  /// terms of the same gate the pipeline applies.
  public static func willSkip(
    twoStage: Bool, upsamplerAvailable: Bool,
    width: Int, height: Int, frames: Int,
    refineScale: Float, refineMaxVolume: Int,
    spatialCompression: Int = 32, temporalCompression: Int = 8
  ) -> Bool {
    guard twoStage else { return false }
    let volume = preRefineVolume(
      width: width, height: height, frames: frames, refineScale: refineScale,
      spatialCompression: spatialCompression, temporalCompression: temporalCompression)
    if case .skip = decide(
      twoStage: true, upsamplerLoaded: upsamplerAvailable,
      preVolume: volume, maxVolume: refineMaxVolume
    ) { return true }
    return false
  }
}
