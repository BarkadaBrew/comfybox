// ModelFamilyDetection.swift — GET /v1/model/family (comfybox#359).
//
// The desktop's preset editor can save `custom_model_path` (an on-disk
// directory) without ever naming an engine `model` spec or a
// `checkpoint_family` — `PresetLoRAStack.declaredFamily` then has neither to
// go on, so `/v1/generate {"preset": id}` can never expand the preset (it
// stays a `preset_unresolved_reason: "no_model"` label forever, even though
// the engine could load the checkpoint just fine via Apply/Generate).
//
// Rather than have the desktop re-implement `Krea2ModelDetection`'s
// fail-closed resolution client-side (or worse, guess a family from a
// filename — exactly the F3 bug that detection exists to prevent), this
// route lets it ask the engine what a model spec resolves to. The desktop
// still owns the final `checkpoint_family` policy label: the accel/stock
// split within Krea-2 "raw" depends on whether the PRESET's own `loras[]`
// declares an accelerator (`role: "accel"`), which only the desktop/preset
// document has — the engine has no opinion on it here.
//
// File-existence checks only (`Krea2ModelDetection.detect` stats files and
// reads `model_index.json`) — never a weight load, never a pool mutation, so
// this is safe to call once per preset in a "Backfill all" batch without
// disturbing whatever is resident.

import Foundation

/// `GET /v1/model/family?model=<spec>` response body.
struct ModelFamilyDetectionResponse: Encodable, Sendable, Equatable {
  /// Echoed verbatim (untrimmed) — the caller's own request value.
  let model: String
  /// Broad family the engine classifies this spec as: `"krea2"` | `"z-image"`.
  /// nil when unclassifiable.
  let family: String?
  /// The physical/declared variant within that family: krea2 `"turbo"` |
  /// `"raw"`; z-image `"turbo"` | `"base"`. nil when `family` is nil, or the
  /// spec resolves to the family but not to a determinable variant.
  let variant: String?
}

enum ModelFamilyDetector {
  /// z-image alias/variant classification. Deliberately independent of both
  /// `ComfyBox/main.swift`'s CLI alias resolution and the
  /// `ComfyBoxModelRegistry` catalog — this route needs neither the CLI
  /// target nor a catalog lookup, just the same aliases they already use.
  private static let baseAliases: Set<String> = ["z-image-base", "zimage-base"]
  private static let turboAliases: Set<String> = ["z-image-turbo", "zimage-turbo", "z-image"]

  /// nil when `spec` does not name z-image at all.
  static func zImageVariant(for spec: String) -> String? {
    let lower = spec.lowercased()
    if baseAliases.contains(lower) { return "base" }
    if turboAliases.contains(lower) { return "turbo" }
    guard lower.contains("z-image") || lower.contains("zimage") else { return nil }
    // An id that names z-image but neither alias exactly (e.g. a HF id like
    // "Tongyi-MAI/Z-Image-Turbo-BF16", or a catalog id like
    // "z-image-base-bf16") — read "base" vs "turbo" off the text itself,
    // same as the alias table above, just not an exact match.
    return lower.contains("base") ? "base" : "turbo"
  }

  /// Detect a spec's family + variant without loading it. Deliberately does
  /// NOT delegate to `PresetLoRAStack.modelFamily`: that function classifies
  /// a `/v1/generate` request's own `model` field, which is always an engine
  /// spec (alias, catalog id, HF id) — never a raw `custom_model_path`
  /// directory, because "the engine never loads from it" there. This route
  /// exists specifically to answer for THAT shape too, so `detectVariant`'s
  /// existing-directory branch (mirroring `resolve(spec:)`'s own precedence)
  /// runs unconditionally, not gated behind a family guess first.
  static func detect(spec: String) -> ModelFamilyDetectionResponse {
    let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)

    if let variant = Krea2ModelDetection.detectVariant(spec: trimmed) {
      return ModelFamilyDetectionResponse(model: spec, family: "krea2", variant: variant.rawValue)
    }
    if Krea2ModelDetection.isKnownKrea2Model(trimmed) {
      // Known by alias/table (or a "krea-2-turbo" substring) but its
      // directory could not be read (e.g. a declared alias whose directory
      // is missing) — still krea2, variant unknown rather than a guess.
      return ModelFamilyDetectionResponse(model: spec, family: "krea2", variant: nil)
    }
    if let variant = zImageVariant(for: trimmed) {
      return ModelFamilyDetectionResponse(model: spec, family: "z-image", variant: variant)
    }
    return ModelFamilyDetectionResponse(model: spec, family: nil, variant: nil)
  }
}
