// BuiltInStudioPacks.swift — Compiled-in default Studio Packs.
//
// Built-in packs ship as Swift literals rather than bundled JSON resources
// (avoids SPM resource-bundle wiring for a single small pack). User packs
// under ~/.comfybox/studio-packs override/extend these — see StudioPackLibrary.

import Foundation

public enum BuiltInStudioPacks {
  /// Life Design / Healthcare Training — faceless figures, flat shading,
  /// clean vector-friendly output, SVG-first export. Ships with prompt/SVG
  /// defaults only; no dedicated LoRA exists yet (loraStack is empty and
  /// stays that way until one does — see PRD open questions).
  public static let lifeDesignHealthcare = StudioPack(
    id: "life-design-healthcare",
    name: "Life Design — Healthcare Training",
    description: "Faceless, flat-shaded healthcare training illustrations with clean, vector-friendly output for SVG export.",
    domain: "healthcare-training",
    version: 1,
    promptSuffix: "faceless figures, flat shading, clean vector-friendly illustration, healthcare training scenario, simple flat colors, no gradients",
    negativePrompt: "photorealistic, detailed face, facial features, gradient shading, photograph, realistic skin texture, 3d render",
    model: "z-image-turbo",
    steps: 9,
    guidance: 0.0,
    width: 1024,
    height: 1024,
    loraStack: [],
    svgDefaults: StudioPackSVGDefaults(enabled: true, preset: "simplified"),
    templateCategories: [
      "cpr", "hospital-bed", "seated-patient", "clinical-handoff", "medical-equipment-tutorial",
    ],
    qaRules: [
      StudioPackQARule(id: "faceless", description: "Prompt requests faceless/no facial features", required: false),
      StudioPackQARule(id: "flat-vector-terms", description: "Prompt includes flat/vector-friendly terms", required: false),
      StudioPackQARule(id: "no-photorealism", description: "Prompt discourages gradients/photorealistic terms", required: false),
      StudioPackQARule(id: "svg-export-requested", description: "SVG export is requested for vector templates", required: false),
      StudioPackQARule(id: "has-pack-metadata", description: "Output carries pack metadata", required: false),
    ],
    mcpTags: ["life-design", "healthcare", "vector", "svg", "faceless"]
  )

  /// All built-in packs, keyed by id.
  public static let all: [StudioPack] = [lifeDesignHealthcare]
}
