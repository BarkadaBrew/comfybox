// StudioPackRecipe.swift — Resolves a StudioPack (optionally through a
// slot-based template, FR-2 / #198) into a concrete generation recipe
// against the locally available LoRA library.
//
// Pure, side-effect-free: resolving a pack never mutates a preset, the LoRA
// library, or generation state. Callers (Desktop "Apply" action, future
// WarmServer/MCP endpoints) decide what to do with the resolved recipe.

import Foundation

/// A concrete, ready-to-generate recipe produced by resolving a StudioPack
/// (and optionally one of its templates) against the local LoRA library.
public struct StudioPackRecipe: Sendable, Equatable {
  public var packId: String
  /// The template that produced this recipe's prompt, if any (raw subject
  /// text was used otherwise).
  public var templateId: String?
  public var prompt: String
  public var negativePrompt: String?
  public var model: String?
  public var steps: Int?
  public var guidance: Float?
  public var scheduler: String?
  public var width: Int?
  public var height: Int?
  /// Only LoRAs actually found in the local library — never a dangling reference.
  public var loras: [StudioPackLoRARef]
  public var svgDefaults: StudioPackSVGDefaults?
  public var cameraAngle: String?
  public var cameraOrientation: String?
  public var lightingStyle: String?
  /// Human-readable notes about anything that couldn't be fully resolved
  /// (e.g. a missing optional LoRA). Never blocks applying the pack.
  public var warnings: [String]
}

public enum StudioPackResolver {
  /// Resolve a pack into a recipe from free-text subject/scenario input.
  ///
  /// - Parameters:
  ///   - pack: The Studio Pack to apply.
  ///   - subject: Free-text subject/scenario the user typed; composed with
  ///     the pack's prompt prefix/suffix.
  ///   - availableLoraIds: The ids currently present in the local LoRA
  ///     library, used to filter the pack's recommended LoRA stack down to
  ///     what's actually installed.
  public static func resolve(
    pack: StudioPack,
    subject: String = "",
    availableLoraIds: Set<String>
  ) -> StudioPackRecipe {
    resolve(pack: pack, prompt: pack.composePrompt(subject: subject), templateId: nil, availableLoraIds: availableLoraIds)
  }

  /// Resolve a pack into a recipe by rendering one of its templates with the
  /// given slot values, then composing the result with the pack's prompt
  /// prefix/suffix (the template supplies the subject, the pack still
  /// contributes the consistent style layer).
  ///
  /// - Parameters:
  ///   - pack: The Studio Pack to apply.
  ///   - template: One of `pack.templates` to render.
  ///   - slotValues: User-entered values keyed by slot id; missing/blank
  ///     values fall back to the slot's default.
  ///   - availableLoraIds: See `resolve(pack:subject:availableLoraIds:)`.
  public static func resolve(
    pack: StudioPack,
    template: StudioPackTemplate,
    slotValues: [String: String],
    availableLoraIds: Set<String>
  ) -> StudioPackRecipe {
    let rendered = template.render(slotValues: slotValues)
    return resolve(
      pack: pack, prompt: pack.composePrompt(subject: rendered),
      templateId: template.id, availableLoraIds: availableLoraIds
    )
  }

  private static func resolve(
    pack: StudioPack,
    prompt: String,
    templateId: String?,
    availableLoraIds: Set<String>
  ) -> StudioPackRecipe {
    var warnings: [String] = []
    var resolvedLoras: [StudioPackLoRARef] = []

    for ref in pack.loraStack {
      if availableLoraIds.contains(ref.loraId) {
        resolvedLoras.append(ref)
      } else if ref.optional {
        warnings.append("LoRA '\(ref.loraId)' not found locally — skipped (optional).")
      } else {
        warnings.append("LoRA '\(ref.loraId)' not found locally — required by this pack but missing.")
      }
    }

    return StudioPackRecipe(
      packId: pack.id,
      templateId: templateId,
      prompt: prompt,
      negativePrompt: pack.negativePrompt,
      model: pack.model,
      steps: pack.steps,
      guidance: pack.guidance,
      scheduler: pack.scheduler,
      width: pack.width,
      height: pack.height,
      loras: resolvedLoras,
      svgDefaults: pack.svgDefaults,
      cameraAngle: pack.cameraAngle,
      cameraOrientation: pack.cameraOrientation,
      lightingStyle: pack.lightingStyle,
      warnings: warnings
    )
  }
}
