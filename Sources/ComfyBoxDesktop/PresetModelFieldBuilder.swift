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
    ///   - detectedFamily / detectedVariant: the engine's answer for the spec
    ///     THIS builder is about to derive from `modelText` (or, when
    ///     `modelText` is empty, whatever the caller last resolved for the
    ///     preset's existing `customModelPath` — see
    ///     `ServerPresetEditor.detectedModelFamily`). nil/nil when not yet
    ///     resolved, or the engine could not classify it.
    ///   - fallbackCheckpointFamily: `original.checkpointFamily` — carried
    ///     through UNCHANGED when detection can't resolve one, so an already
    ///     valid, manually-declared value (or one an earlier backfill wrote)
    ///     is never clobbered by a transient detection miss.
    public static func build(
        modelText: String,
        loras: [ServerPresetLora],
        detectedFamily: String?,
        detectedVariant: String?,
        fallbackCheckpointFamily: String?
    ) -> Result {
        let modelValue = modelText.trimmingCharacters(in: .whitespacesAndNewlines)
        let model: String?
        let customModelPath: String?
        if modelValue.hasPrefix("/") || modelValue.hasPrefix("~") {
            customModelPath = modelValue
            model = nil
        } else {
            model = modelValue.isEmpty ? nil : modelValue
            customModelPath = nil
        }

        let resolvedFamily = CheckpointFamilyResolver.resolve(
            family: detectedFamily, variant: detectedVariant, loras: loras)

        return Result(
            model: model,
            customModelPath: customModelPath,
            checkpointFamily: resolvedFamily ?? fallbackCheckpointFamily
        )
    }
}
