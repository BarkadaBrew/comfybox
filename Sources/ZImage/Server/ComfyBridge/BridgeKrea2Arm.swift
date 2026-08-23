// BridgeKrea2Arm.swift — the Krita bridge's variant-aware Krea 2 arm
//
// WP-E19 (docs/FDD-krea2-raw-recipe.md §3.5 "Bridge", D13, AC-5a). Before
// this, `bridgeGenerate`'s `.krea2` arm pinned guidance to 0, dropped the
// negative prompt and clamped steps to 12 for the WHOLE family — correct for
// Krea-2-Turbo (distilled, no CFG), and exactly wrong for Krea-2-Raw, which
// honours CFG and wants the steps Krita asked for. The arm now splits on the
// physical variant the engine actually loaded (`Krea2Variant`, WP-E5):
//
//   .turbo  byte-identical to 296735d: steps > 0 ? min(steps, 12) : 9,
//           guidance 0.0, negative nil.
//   .raw    request-sourced, no clamp, live negative. Falls back to the
//           variant defaults (30 / 1.0) ONLY when Krita sends nothing — and
//           Krita sends both on every render, so in practice this is a
//           pass-through with the negative prompt restored. There is no
//           "active preset" on a bridge render to read from (D13, corrected
//           in FDD v2: `ComfyBridgeGenerateRequest` has no preset field and
//           image presets resolve daemon-side).
//   nil     the family is krea2 but no variant is known — a fault, thrown,
//           never "turbo" (WP-E5's recorded discrepancy for this WP).
//
// The arm is a pure function and the payload constructor is shared by every
// family arm in `bridgeGenerate`, so AC-5a's field-for-field comparison in
// `BridgeKrea2VariantTests` exercises the real construction with no weights.

import Foundation

/// What the bridge's `.krea2` arm resolved for one request.
struct BridgeKrea2Resolution: Sendable, Equatable {
  /// The physical variant the arm switched on.
  let variant: Krea2Variant
  let steps: Int
  let guidance: Float
  let negativePrompt: String?
  /// Forwarded verbatim; name resolution (fail-loud, WP-E4) runs on the
  /// payload afterwards, so an unknown sampler throws and never becomes euler.
  let sampler: String?
  /// True when `.turbo`'s runaway-KSampler clamp actually changed `steps`.
  let stepsClamped: Bool
}

enum BridgeKrea2Arm {

  /// Resolve steps / guidance / negative / sampler for a bridge render on the
  /// krea2 family. `variant == nil` is a fault (the coordinator reports the
  /// krea2 family resident but no variant) and throws
  /// `WarmServerError.krea2VariantUnknown`.
  static func resolve(_ request: ComfyBridgeGenerateRequest, variant: Krea2Variant?) throws -> BridgeKrea2Resolution {
    guard let variant else { throw WarmServerError.krea2VariantUnknown }

    switch variant {
    case .turbo:
      // Krea-2-Turbo: 8-step distilled, no CFG. Clamp runaway KSampler
      // defaults. Byte-identical to 296735d (AC-5a).
      let requested = request.steps > 0 ? request.steps : variant.defaultSteps  // 9
      let clamp = variant.bridgeStepClamp ?? Int.max                             // 12
      let steps = min(requested, clamp)
      return BridgeKrea2Resolution(
        variant: .turbo,
        steps: steps,
        guidance: 0.0,
        negativePrompt: nil,
        sampler: request.sampler,
        stepsClamped: steps != requested)

    case .raw:
      // Krea-2-Raw: what Krita sent, no clamp, CFG and the negative prompt
      // live. Variant defaults only on an absent value (D13).
      return BridgeKrea2Resolution(
        variant: .raw,
        steps: request.steps > 0 ? request.steps : variant.defaultSteps,            // 30
        guidance: request.guidance > 0 ? request.guidance : variant.defaultGuidance, // 1.0
        negativePrompt: request.negativePrompt,
        sampler: request.sampler,
        stepsClamped: false)
    }
  }
}

extension ComfyBridgeGenerateRequest {

  /// THE bridge payload constructor — `bridgeGenerate` builds every family's
  /// `GeneratePayload` through this, with the four family-resolved values
  /// passed in and everything else lifted from the request exactly as
  /// 296735d did (`WarmServer.swift:3277-3297` at that commit). Fields the
  /// bridge never set stay nil.
  func makeGeneratePayload(
    width: Int, height: Int,
    steps: Int, guidance: Float, negativePrompt: String?, sampler: String?
  ) -> GeneratePayload {
    GeneratePayload(
      prompt: prompt,
      negativePrompt: negativePrompt,
      width: width,
      height: height,
      steps: steps,
      guidance: guidance,
      seed: seed,
      outputPath: nil,
      levelsMin: levelsMin,
      levelsMax: levelsMax,
      scheduler: sampler,
      sigmaSchedule: sigmaSchedule,
      inpaintImageData: inpaintImageData,
      maskData: maskImageData,
      denoise: denoise,
      maskGrow: maskGrow,
      maskFeather: maskFeather,
      maskCropX: maskCropX,
      maskCropY: maskCropY
    )
  }
}
