// LTX2ImagePresetExpansion.swift — named-preset support for native LTX images.

import Foundation

extension WarmServer {
  /// Expand an LTX image preset without routing it through the warm image
  /// model's model/adapter decision. LTX's configured weights are its base,
  /// so a valid LTX preset intentionally has no `model` field.
  static func expandLTX2ImagePreset(
    _ payload: GeneratePayload,
    store: PresetStore,
    stageNearline: ([LoRAEntry]) -> [LoRAEntry] = { $0 },
    loraExists: (LoRAEntry) -> Bool = WarmServer.loRASourceExists,
    log: (String) -> Void = { _ in }
  ) -> GeneratePayload {
    let id = payload.preset?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard !id.isEmpty else { return payload }

    let lookup = store.lookup(id)
    guard let preset = lookup.preset else {
      return payload.asUnresolvedPreset(
        id,
        PresetExpansion.Unresolved(
          code: "unknown_preset",
          message: "unknown preset '\(id)' — not in this engine's store (GET /v1/presets)"),
        log: log)
    }
    if let reason = lookup.invalidReason {
      return payload.asUnresolvedPreset(
        id,
        PresetExpansion.Unresolved(
          code: "invalid_preset", message: "preset '\(id)' is flagged invalid: \(reason)"),
        log: log)
    }
    guard LTX2ImageRecipe.isEngineName(preset.engine) else {
      let engine = preset.engine ?? "default"
      return payload.asUnresolvedPreset(
        id,
        PresetExpansion.Unresolved(
          code: "engine:\(engine)",
          message: "preset '\(id)' declares engine '\(engine)', not native LTX-2 image generation"),
        log: log)
    }
    guard preset.mediaKind?.lowercased() != "video" else {
      return payload.asUnresolvedPreset(
        id,
        PresetExpansion.Unresolved(
          code: "media_kind:video", message: "preset '\(id)' is a video preset, not an image preset"),
        log: log)
    }
    if let provider = preset.provider?.trimmingCharacters(in: .whitespacesAndNewlines),
       !provider.isEmpty, provider.lowercased() != "local" {
      return payload.asUnresolvedPreset(
        id,
        PresetExpansion.Unresolved(
          code: "provider:\(provider)", message: "preset '\(id)' is not a local preset"),
        log: log)
    }

    let presetEntries = preset.loras.map {
      LoRAEntry(path: $0.filename, scale: Float($0.scale), role: $0.role)
    }
    let staged: [LoRAEntry]
    if payload.loras == nil {
      staged = presetEntries.isEmpty ? [] : stageNearline(presetEntries)
      if let missing = staged.first(where: { !loraExists($0) }) {
        let name = (missing.path as NSString).lastPathComponent
        return payload.asUnresolvedPreset(
          id,
          PresetExpansion.Unresolved(
            code: "missing_lora:\(name)",
            message: "preset '\(id)' names LoRA '\(missing.path)', which is not available locally"),
          log: log)
      }
    } else {
      staged = presetEntries
    }

    var out = payload
    if let explicit = payload.loras {
      let requestStack = explicit.map {
        LoraReference(filename: $0.path, scale: Double($0.scale ?? 1), role: $0.role)
      }
      if !PresetLoRAStack.isSameStack(requestStack, preset.loras) {
        out.presetStackMismatch = true
      }
    } else {
      out.loras = staged
      out.presetStackApplied = true
      log("Preset '\(id)': applying its native LTX LoRA stack — "
        + PresetLoRAStack.describe(preset.loras))
    }

    if out.negativePrompt == nil { out.negativePrompt = preset.negativePrompt }
    if out.width == nil { out.width = preset.width }
    if out.height == nil { out.height = preset.height }
    if out.steps == nil { out.steps = preset.steps }
    if out.guidance == nil { out.guidance = preset.guidance.map(Float.init) }
    if out.seed == nil, let seed = preset.seed, seed >= 0 { out.seed = UInt64(seed) }
    if out.contentMode == nil { out.contentMode = preset.contentMode }

    var recipeApplied: [String] = []
    if out.scheduler == nil,
       let sampler = LTX2ImageRecipe.normalized(preset.sampler)
        ?? LTX2ImageRecipe.normalized(preset.scheduler) {
      out.scheduler = sampler
      recipeApplied.append("scheduler")
    }
    if out.sigmaSchedule == nil,
       let schedule = LTX2ImageRecipe.normalized(preset.sigmaSchedule) {
      out.sigmaSchedule = schedule
      recipeApplied.append("sigma_schedule")
    }
    if !recipeApplied.isEmpty { out.presetRecipeApplied = recipeApplied }
    return out
  }
}
