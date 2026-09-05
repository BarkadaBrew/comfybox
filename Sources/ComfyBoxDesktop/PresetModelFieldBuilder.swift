// PresetModelFieldBuilder.swift — comfybox#359
//
// How the preset editor's `model` text field maps onto the wire fields the
// engine actually reads for `/v1/generate {preset}` expansion:
// `ServerPreset.model` / `.customModelPath` / `.checkpointFamily`. Pure and
// synchronous, deliberately independent of any live engine call, so
// `ServerPresetEditor.buildPreset()` stays a plain function of its `@State`
// and the save button never blocks on network — the engine's answer
// (`ModelFamilyInfo`, from `GET /v1/model/family`) is fetched ahead of time
// (`.task(id: model)`) and handed in already resolved.
//
// FIX ROUND 1: a typed PATH now also populates `model` (with the canonical
// engine spec the engine returned — a declared alias where one matches,
// otherwise the standardized absolute path), not just `custom_model_path`.
// That is the field `PresetLoRAStack.decide` reads first; without it a
// path-only preset stays a `no_model` label no matter what
// `checkpoint_family` says. `custom_model_path` is still written so the
// desktop's own Apply/Set-as-Warm path is unchanged.
//
// FIX ROUND 2, two rules that matter more than they look:
//
//   1. (CRITICAL) A detection MISS must never erase a `model` that is
//      already there. `buildPreset()` runs on every Save, and detection is
//      nil whenever the engine is unreachable or the user saves before
//      `.task(id: model)` answers. Writing nil then reverted a
//      just-backfilled preset straight back to `no_model` — this PR's own
//      bug, reintroduced by its own editor. So `model` is cleared ONLY when
//      the user actually changed the path; otherwise `fallbackModel` (the
//      preset's existing `model`) stands.
//   3. An un-roled accelerator LoRA no longer gates anything. `model` is
//      written whenever the engine says the spec is loadable; only the
//      `checkpoint_family` LABEL is deferred, and only where the label
//      genuinely depends on the role (Krea-2 "raw"). Expandability never
//      waits on a label — `declaredFamily` maps `raw-accel` and `raw-stock`
//      to the same "krea2" anyway.

import Foundation

public enum PresetModelFieldBuilder {

    public struct Result: Equatable {
        public var model: String?
        public var customModelPath: String?
        public var checkpointFamily: String?

        public init(model: String?, customModelPath: String?, checkpointFamily: String?) {
            self.model = model
            self.customModelPath = customModelPath
            self.checkpointFamily = checkpointFamily
        }
    }

    /// - Parameters:
    ///   - modelText: the editor's `model` field, as typed.
    ///   - loras: the preset's current LoRA rows — only their `role` matters
    ///     here (accel vs not), for `CheckpointFamilyResolver`.
    ///   - detection: the engine's answer (`GET /v1/model/family`) for the
    ///     spec THIS builder is about to derive from `modelText`, kept fresh
    ///     by `ServerPresetEditor`'s `.task(id: model)`. nil when not yet
    ///     resolved or the engine was unreachable.
    ///   - fallbackModel: `original.model` — the preset's existing `model`,
    ///     kept when detection can't supply a fresh one AND the typed path is
    ///     unchanged. This is the ruling-1 guard against erasing a backfill.
    ///   - fallbackCustomModelPath: `original.customModelPath` — what
    ///     "unchanged" is measured against.
    ///   - fallbackCheckpointFamily: `original.checkpointFamily` — carried
    ///     through UNCHANGED when no label can be derived, so an already
    ///     valid, manually-declared value (or one an earlier backfill wrote)
    ///     is never clobbered by a transient detection miss or by a deferred
    ///     accel/stock decision.
    public static func build(
        modelText: String,
        loras: [ServerPresetLora],
        detection: ModelFamilyInfo?,
        fallbackModel: String?,
        fallbackCustomModelPath: String?,
        fallbackCheckpointFamily: String?
    ) -> Result {
        let modelValue = modelText.trimmingCharacters(in: .whitespacesAndNewlines)
        let isPath = modelValue.hasPrefix("/") || modelValue.hasPrefix("~")

        var model: String?
        var customModelPath: String?
        if isPath {
            customModelPath = modelValue
            // The canonical spec the engine gave us for THIS path — an alias
            // when the path is a declared install, else the standardized
            // absolute path. Only when the engine said it would load it.
            if let detection, detection.loadable,
               case let spec = detection.spec.trimmingCharacters(in: .whitespacesAndNewlines),
               !spec.isEmpty {
                model = spec
            } else if isUnchangedPath(modelValue, from: fallbackCustomModelPath) {
                // Ruling 1: nothing fresh to write, and the user did not
                // repoint the preset — keep whatever `model` it already has.
                // Clearing here is what silently un-fixed a backfilled preset.
                model = fallbackModel
            } else {
                // The path genuinely changed and we have no answer for the new
                // one; the old base no longer describes this preset.
                model = nil
            }
        } else {
            // A bare spec IS the model, exactly as typed — including when it
            // is empty, which is an explicit "clear this field", not a miss.
            model = modelValue.isEmpty ? nil : modelValue
            customModelPath = nil
        }

        return Result(
            model: model,
            customModelPath: customModelPath,
            checkpointFamily: derivedLabel(detection: detection, loras: loras)
                ?? fallbackCheckpointFamily
        )
    }

    /// The `checkpoint_family` label this save can derive with certainty, or
    /// nil to fall back to whatever the preset already declared.
    private static func derivedLabel(detection: ModelFamilyInfo?, loras: [ServerPresetLora]) -> String? {
        guard let detection else { return nil }
        // Ruling 3: defer the label — never guess it — but only where it
        // actually turns on the role.
        if CheckpointFamilyResolver.dependsOnAccelRole(family: detection.family, variant: detection.variant),
           !AcceleratorLoRAHeuristic.unroledAcceleratorCandidates(loras).isEmpty {
            return nil
        }
        return CheckpointFamilyResolver.resolve(
            family: detection.family, variant: detection.variant, loras: loras)
    }

    /// Same path, modulo the whitespace the editor trims anyway.
    private static func isUnchangedPath(_ typed: String, from stored: String?) -> Bool {
        guard let stored = stored?.trimmingCharacters(in: .whitespacesAndNewlines), !stored.isEmpty
        else { return false }
        return typed == stored
    }
}
