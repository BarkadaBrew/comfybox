// LTX2NAG.swift — Normalized Attention Guidance
//
// The reference PinkCherry recipe (verified identical across v1.6, v1.7 and
// v1.8 by reading the API graph embedded in the author's published mp4s) runs
// CFG 1.0 on both passes and gets prompt adherence from NAG instead:
// `LTX2_NAG {nag_scale: 11.0, nag_alpha: 0.25, nag_tau: 2.5, inplace: true}`.
//
// Why it matters here: CFG extrapolates the final prediction with no bound on
// magnitude, so pushing it high to buy adherence drives latent channels into
// saturation — measured 2026-08-02 at cfg 3.5 as cyan chroma blotches across
// 9.3% of the frame, which a global guidance-rescale could not repair because
// it only shifts the mean. NAG extrapolates INSIDE cross-attention and then
// renormalises per attention vector, so adherence rises while magnitude stays
// clamped at tau x the conditional norm.
//
// Reference: "Normalized Attention Guidance: Universal Negative Guidance for
// Diffusion Models" (2025).

import Foundation
import MLX

/// Apply Normalized Attention Guidance to a pair of cross-attention outputs.
///
///     z_guid  = z_pos * scale - z_neg * (scale - 1)
///     ratio   = ||z_guid|| / ||z_pos||        (per attention vector)
///     z_clamp = ratio > tau ? z_guid * (tau / ratio) : z_guid
///     z_out   = alpha * z_clamp + (1 - alpha) * z_pos
///
/// Norms are taken over the LAST axis so each attention vector is clamped on
/// its own. A global norm would let one large vector's ratio mask another's
/// excursion — the precise flaw in the global-std guidance rescale this
/// replaces.
///
/// - Parameters:
///   - positive: Cross-attention output for the positive context.
///   - negative: Cross-attention output for the negative context, same shape.
///   - scale: Extrapolation strength (reference: 11.0). 1.0 disables.
///   - alpha: Blend toward the guided vector (reference: 0.25). 0 disables.
///   - tau: Norm ceiling as a multiple of the positive norm (reference: 2.5).
/// - Returns: Guided output, same shape and dtype as `positive`.
public func ltx2ApplyNAG(
  positive: MLXArray,
  negative: MLXArray,
  scale: Float,
  alpha: Float,
  tau: Float
) -> MLXArray {
  guard alpha > 0, scale != 1.0 else { return positive }

  let dtype = positive.dtype
  let pos = positive.asType(.float32)
  let neg = negative.asType(.float32)

  let guided = pos * MLXArray(scale) - neg * MLXArray(scale - 1)

  let normGuided = MLX.sqrt((guided * guided).sum(axis: -1, keepDims: true))
  let normPos = MLX.sqrt((pos * pos).sum(axis: -1, keepDims: true))
  let ratio = normGuided / (normPos + MLXArray(Float(1e-6)))

  // Scale down only when the ratio exceeds tau; never scale up.
  let factor = MLX.minimum(MLXArray(tau) / (ratio + MLXArray(Float(1e-6))), MLXArray(Float(1)))
  let clamped = guided * factor

  let out = MLXArray(alpha) * clamped + MLXArray(1 - alpha) * pos
  return out.asType(dtype)
}

/// NAG settings resolved from environment, with the reference recipe as the
/// documented default. Disabled unless `LTX2_NAG_SCALE` is set, so existing
/// renders are byte-identical until NAG is explicitly turned on.
public struct LTX2NAGConfig: Sendable {
  public let scale: Float
  public let alpha: Float
  public let tau: Float

  public var isEnabled: Bool { scale != 1.0 && alpha > 0 }

  public init(scale: Float, alpha: Float, tau: Float) {
    self.scale = scale
    self.alpha = alpha
    self.tau = tau
  }

  /// Reference recipe (PinkCherry v1.6/v1.7/v1.8, ComfyUI `LTX2_NAG` node).
  public static let reference = LTX2NAGConfig(scale: 11.0, alpha: 0.25, tau: 2.5)

  /// Off — `scale == 1` makes `ltx2ApplyNAG` an identity.
  public static let disabled = LTX2NAGConfig(scale: 1.0, alpha: 0.0, tau: 2.5)

  /// Reads `LTX2_NAG_SCALE` / `_ALPHA` / `_TAU`. Values are trimmed before
  /// parsing: padded plist env values silently defeated comparisons in this
  /// codebase before (2026-08-01).
  public static func fromEnvironment(
    _ env: [String: String] = ProcessInfo.processInfo.environment
  ) -> LTX2NAGConfig {
    func f(_ key: String) -> Float? {
      guard let raw = env[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
            !raw.isEmpty, let v = Float(raw), v.isFinite else { return nil }
      return v
    }
    guard let scale = f("LTX2_NAG_SCALE") else { return .disabled }
    return LTX2NAGConfig(
      scale: scale,
      alpha: f("LTX2_NAG_ALPHA") ?? LTX2NAGConfig.reference.alpha,
      tau: f("LTX2_NAG_TAU") ?? LTX2NAGConfig.reference.tau
    )
  }
}
