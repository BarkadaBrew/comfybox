// PresetEffectiveRecipe.swift — the "effective recipe" view model (#277)
//
// The preset editor persists a DECLARED `ImagePreset`; what the engine
// actually renders is that preset RESOLVED against system defaults
// (`PresetStore.resolve`, `POST /v1/presets/resolve`) and then, when the
// preset is named on `/v1/generate`, EXPANDED into one concrete LoRA stack —
// or left a label-only preset with `preset_unresolved`/`preset_unresolved_reason`
// when the engine cannot tell which base its adapters belong to (#286,
// `PresetLoRAStack.decide`).
//
// ComfyBoxDesktop already links `ZImage`, so this reuses the engine's own
// `ResolvedPreset` and `PresetLoRAStack` — not a hand-rolled re-implementation
// of the wire contract that could drift from it. Running the same Swift code
// the server runs is the strongest form of "shows the SAME resolution the
// engine applies": there is no serialization step for either side to diverge
// on.
import Foundation
import ZImage

/// One adapter in the effective, resolved-and-expanded LoRA stack.
public struct EffectiveLoRA: Equatable, Identifiable, Sendable {
  public var id: String { "\(filename)@\(role ?? "-")" }
  public var filename: String
  public var scale: Double
  public var role: String?

  public init(filename: String, scale: Double, role: String?) {
    self.filename = filename
    self.scale = scale
    self.role = role
  }
}

/// Why the engine would render this preset as a label only, never expanding
/// its model/LoRA stack — `PresetExpansion.Unresolved`, mirrored for the UI.
public struct EffectiveRecipeUnresolved: Equatable, Sendable {
  /// Machine-readable code (`no_model`, `kroma_file_missing`, `engine:mflux`, …).
  public var code: String
  /// Full sentence, the same one the engine logs.
  public var message: String
  /// #359: when the reason is that the engine can't tell which model family
  /// the preset's adapters belong to, tell the user the one field that fixes
  /// it — the same follow-up already filed against the 26 desktop presets
  /// that carry only `custom_model_path`.
  public var hint: String?

  public init(code: String, message: String, hint: String? = nil) {
    self.code = code
    self.message = message
    self.hint = hint
  }
}

/// The effective Krea recipe: what `/v1/generate {"preset": id}` would
/// actually run, with no other request overrides.
public struct EffectiveRecipe: Equatable, Sendable {
  public var model: String?
  public var checkpointFamily: String?
  public var mediaKind: String
  public var loraStack: [EffectiveLoRA]
  public var steps: Int
  public var guidance: Double?
  public var sampler: String?
  public var sigmaSchedule: String?
  public var shift: Double?
  public var eta: Double?
  public var bongmath: Bool?
  public var stage2: PresetStage?
  /// nil = the engine would expand this preset as a whole (model + stack).
  /// Non-nil = it stays a provenance label, and this says why.
  public var unresolved: EffectiveRecipeUnresolved?

  public init(
    model: String?, checkpointFamily: String?, mediaKind: String,
    loraStack: [EffectiveLoRA], steps: Int, guidance: Double?,
    sampler: String?, sigmaSchedule: String?, shift: Double?, eta: Double?,
    bongmath: Bool?, stage2: PresetStage?, unresolved: EffectiveRecipeUnresolved?
  ) {
    self.model = model
    self.checkpointFamily = checkpointFamily
    self.mediaKind = mediaKind
    self.loraStack = loraStack
    self.steps = steps
    self.guidance = guidance
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
    self.shift = shift
    self.eta = eta
    self.bongmath = bongmath
    self.stage2 = stage2
    self.unresolved = unresolved
  }
}

public enum PresetEffectiveRecipePresenter {

  /// #359: the one hint the UI adds to the engine's `no_model` reason —
  /// naming the specific fix, not just the symptom.
  static let checkpointFamilyHint =
    "Add checkpoint_family to make this preset expandable."

  /// Compute the effective recipe purely from the declared preset — no
  /// network round trip needed: `ResolvedPreset(preset:)` and
  /// `PresetLoRAStack.decide` are exactly what the engine runs for
  /// `POST /v1/presets/resolve` and `POST /v1/generate {"preset": id}`
  /// respectively. Recomputing this on every editor keystroke is what makes
  /// the panel update live.
  public static func compute(
    declared: ImagePreset, defaults: PresetDefaults = .standard
  ) -> EffectiveRecipe {
    compute(resolved: ResolvedPreset(preset: declared, defaults: defaults), declared: declared)
  }

  /// Compute from an already-resolved preset — the shape `POST
  /// /v1/presets/resolve` actually returns — plus the declared preset the
  /// engine needs to run the same expansion decision. Split out so the
  /// network path (decode the server's response, reuse this) and the local
  /// live-preview path (`compute(declared:)`) share one decision function.
  public static func compute(resolved: ResolvedPreset, declared: ImagePreset) -> EffectiveRecipe {
    // `decide` treats an empty presetId as "no preset named" (`.unchanged`),
    // which would skip the expansion decision entirely — a brand-new,
    // not-yet-saved preset has no id yet, but the panel still needs to show
    // what WOULD happen, so it is never left empty here.
    let id = declared.id.isEmpty ? "draft" : declared.id
    let decision = PresetLoRAStack.decide(
      presetId: id,
      lookup: .resolved(resolved, declared: declared),
      requestLoras: nil
    )

    switch decision {
    case .unchanged, .modelConflict:
      // Unreachable with a non-empty id and no requestModel — but a resolved
      // panel beats a crash if `decide`'s contract ever changes underfoot.
      return EffectiveRecipe(
        model: resolved.model, checkpointFamily: resolved.checkpointFamily,
        mediaKind: resolved.mediaKind, loraStack: [], steps: resolved.steps,
        guidance: resolved.guidance, sampler: resolved.sampler,
        sigmaSchedule: resolved.sigmaSchedule, shift: resolved.shift, eta: resolved.eta,
        bongmath: resolved.bongmath, stage2: resolved.stage2, unresolved: nil)

    case .apply(let expansion):
      if let reason = expansion.unresolved {
        let hint = reason.code == "no_model" ? checkpointFamilyHint : nil
        return EffectiveRecipe(
          model: resolved.model, checkpointFamily: resolved.checkpointFamily,
          mediaKind: resolved.mediaKind, loraStack: [], steps: resolved.steps,
          guidance: resolved.guidance, sampler: resolved.sampler,
          sigmaSchedule: resolved.sigmaSchedule, shift: resolved.shift, eta: resolved.eta,
          bongmath: resolved.bongmath, stage2: resolved.stage2,
          unresolved: EffectiveRecipeUnresolved(code: reason.code, message: reason.message, hint: hint))
      }
      let stack = (expansion.loras ?? resolved.loras).map {
        EffectiveLoRA(filename: $0.filename, scale: $0.scale, role: $0.role)
      }
      return EffectiveRecipe(
        model: expansion.model ?? resolved.model, checkpointFamily: resolved.checkpointFamily,
        mediaKind: resolved.mediaKind, loraStack: stack,
        steps: expansion.steps ?? resolved.steps, guidance: expansion.guidance ?? resolved.guidance,
        sampler: resolved.sampler, sigmaSchedule: resolved.sigmaSchedule,
        shift: resolved.shift, eta: resolved.eta, bongmath: resolved.bongmath,
        stage2: resolved.stage2, unresolved: nil)
    }
  }
}
