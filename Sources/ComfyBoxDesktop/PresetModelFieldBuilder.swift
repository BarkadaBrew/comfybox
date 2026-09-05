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
// desktop's own Apply/Set-as-Warm path is unchanged, and `model` is only
// written when the engine said the spec is `loadable` — a preset that names
// a base the engine would refuse is worse than one that names none.

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
    ///     resolved or the engine was unreachable — in which case a typed
    ///     path is stored as `custom_model_path` only, exactly as before this
    ///     change, and the preset's existing `checkpoint_family` survives.
    ///   - fallbackCheckpointFamily: `original.checkpointFamily` — carried
    ///     through UNCHANGED when detection can't resolve one, so an already
    ///     valid, manually-declared value (or one an earlier backfill wrote)
    ///     is never clobbered by a transient detection miss.
    public static func build(
        modelText: String,
        loras: [ServerPresetLora],
        detection: ModelFamilyInfo?,
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
            if let detection, detection.loadable {
                let spec = detection.spec.trimmingCharacters(in: .whitespacesAndNewlines)
                model = spec.isEmpty ? nil : spec
            }
        } else {
            model = modelValue.isEmpty ? nil : modelValue
            customModelPath = nil
        }

        let resolvedFamily = CheckpointFamilyResolver.resolve(
            family: detection?.family, variant: detection?.variant, loras: loras)

        return Result(
            model: model,
            customModelPath: customModelPath,
            checkpointFamily: resolvedFamily ?? fallbackCheckpointFamily
        )
    }
}
