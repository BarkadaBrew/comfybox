// CivitAITaxonomy.swift — best-effort "act" category inference for harvested
// CivitAI models (#234).
//
// CivitAI does not publish a stable pose/act taxonomy over its API, and
// nothing else in this codebase defined one before this file (checked
// FamilyRecipeMatrix.swift, ContentModeStore.swift, PromptLibraryStore.swift,
// CivitAIModels.swift — none carry a category enum that fits). This is
// therefore a NEW, modest, keyword-driven classifier — a decision flagged
// for review in the #234 build report, not a port of an existing taxonomy.
// It exists so harvested prompt-repository entries are at least coarsely
// browsable/filterable by category; it is not meant to be authoritative, and
// `actTaxonomy` on a `PromptRepositoryEntry` is optional precisely because
// this can (and will) sometimes match nothing.

import Foundation

public enum CivitAITaxonomy {
  /// Fixed category vocabulary `actTaxonomy` values are drawn from.
  public enum Act: String, CaseIterable, Sendable {
    case pose
    case action
    case clothing
    case body
    case character
    case style
    case concept
  }

  /// Infer a coarse category from a model's type/name/tags. Checks, in
  /// order, the most specific keyword buckets first, then falls back to the
  /// CivitAI `type` field (Checkpoint -> style, LORA/LoCon/LyCORIS ->
  /// concept, TextualInversion -> concept). Returns nil only when there was
  /// no signal to classify at all (empty type, name and tags).
  public static func inferAct(modelType: String, name: String, tags: [String]) -> String? {
    let type = modelType.lowercased()
    guard !type.isEmpty || !name.isEmpty || !tags.isEmpty else { return nil }

    let haystack = ([name] + tags).joined(separator: " ").lowercased()
    func any(_ keywords: [String]) -> Bool { keywords.contains { haystack.contains($0) } }

    if any(["pose", "posing", "position"]) { return Act.pose.rawValue }
    if any(["sex", "fuck", "oral", "penetration", "orgasm", "cumshot", "scene", "motion", "grinding", "handjob", "blowjob"]) {
      return Act.action.rawValue
    }
    if any(["outfit", "dress", "lingerie", "costume", "uniform", "clothing", "bikini", "swimsuit", "wardrobe"]) {
      return Act.clothing.rawValue
    }
    if any(["body type", "anatomy", "breast", "figure", "physique", "muscular", "curvy", "athletic body"]) {
      return Act.body.rawValue
    }
    if any(["style", "aesthetic", "art style", "photography", "cinematic", "film look"]) {
      return Act.style.rawValue
    }
    if any(["character", "waifu", "oc", "original character"]) { return Act.character.rawValue }

    switch type {
    case "checkpoint": return Act.style.rawValue
    case "textualinversion": return Act.concept.rawValue
    case "lora", "locon", "lycoris", "dora": return Act.concept.rawValue
    default: return type.isEmpty ? nil : Act.concept.rawValue
    }
  }
}
