// PresetModelFieldBuilderTests.swift — comfybox#359
//
// `ServerPresetEditor.buildPreset()`'s model/checkpoint_family mapping,
// extracted so it's testable without a live engine or a SwiftUI view.
//
// FIX ROUND 1: the editor must write `model` for a typed PATH too (the
// canonical spec the engine returned), because that is the only field
// `PresetLoRAStack.decide` reads before it gives up with `no_model`.

import Testing
@testable import ComfyBoxDesktop

@Suite("PresetModelFieldBuilder")
struct PresetModelFieldBuilderTests {

    private func detection(
        model: String, family: String?, variant: String?,
        spec: String, loadable: Bool = true, reason: String? = nil
    ) -> ModelFamilyInfo {
        ModelFamilyInfo(model: model, family: family, variant: variant,
                        spec: spec, loadable: loadable, reason: reason)
    }

    @Test("a bare alias goes to model, not custom_model_path — unchanged from today")
    func aliasGoesToModel() {
        let result = PresetModelFieldBuilder.build(
            modelText: "krea2-raw", loras: [],
            detection: detection(model: "krea2-raw", family: "krea2", variant: "raw", spec: "krea2-raw"),
            fallbackCheckpointFamily: nil)
        #expect(result.model == "krea2-raw")
        #expect(result.customModelPath == nil)
        #expect(result.checkpointFamily == "raw-stock")
    }

    @Test("a path fills BOTH model (canonical spec) and custom_model_path — the fix")
    func pathFillsModelAndCustomPath() {
        let loras = [ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: "accel")]
        let result = PresetModelFieldBuilder.build(
            modelText: "/Users/todd/LocalModels/krea2-raw", loras: loras,
            detection: detection(model: "/Users/todd/LocalModels/krea2-raw",
                                 family: "krea2", variant: "raw", spec: "krea2-raw"),
            fallbackCheckpointFamily: nil)
        #expect(result.model == "krea2-raw", "the engine's canonical spec, not the raw path")
        #expect(result.customModelPath == "/Users/todd/LocalModels/krea2-raw")
        #expect(result.checkpointFamily == "raw-accel")
    }

    @Test("a path the engine would NOT load leaves model empty rather than writing a bad base")
    func unloadablePathDoesNotFillModel() {
        let result = PresetModelFieldBuilder.build(
            modelText: "/Users/todd/LocalModels/gone", loras: [],
            detection: detection(model: "/Users/todd/LocalModels/gone", family: nil, variant: nil,
                                 spec: "/Users/todd/LocalModels/gone", loadable: false,
                                 reason: "does not exist on this machine"),
            fallbackCheckpointFamily: nil)
        #expect(result.model == nil)
        #expect(result.customModelPath == "/Users/todd/LocalModels/gone")
    }

    @Test("with no detection in hand, a typed path behaves exactly as it did before this change")
    func noDetectionKeepsThePreExistingBehavior() {
        let result = PresetModelFieldBuilder.build(
            modelText: "/Users/todd/LocalModels/krea2-raw", loras: [],
            detection: nil, fallbackCheckpointFamily: "raw-stock")
        #expect(result.model == nil)
        #expect(result.customModelPath == "/Users/todd/LocalModels/krea2-raw")
        #expect(result.checkpointFamily == "raw-stock", "a detection miss never erases a valid declaration")
    }

    @Test("a tilde path is also treated as a path, and takes the engine's expanded spec")
    func tildePathIsAPath() {
        let result = PresetModelFieldBuilder.build(
            modelText: "~/LocalModels/krea2-raw", loras: [],
            detection: detection(model: "~/LocalModels/krea2-raw", family: "krea2", variant: "raw",
                                 spec: "/Users/todd/LocalModels/krea2-raw"),
            fallbackCheckpointFamily: nil)
        #expect(result.customModelPath == "~/LocalModels/krea2-raw")
        #expect(result.model == "/Users/todd/LocalModels/krea2-raw",
                "ModelResolution does not expand ~, so the engine's expanded spec is what gets stored")
    }

    @Test("an empty model field clears both model and custom_model_path")
    func emptyModelClearsBoth() {
        let result = PresetModelFieldBuilder.build(
            modelText: "   ", loras: [],
            detection: nil, fallbackCheckpointFamily: "raw-stock")
        #expect(result.model == nil)
        #expect(result.customModelPath == nil)
        // No model at all to detect from — the existing declaration (if any)
        // survives; nothing here should silently erase it.
        #expect(result.checkpointFamily == "raw-stock")
    }

    @Test("when detection cannot classify the spec, the preset's existing checkpoint_family is kept")
    func undetectedFallsBackToExisting() {
        let result = PresetModelFieldBuilder.build(
            modelText: "some-unknown-checkpoint", loras: [],
            detection: detection(model: "some-unknown-checkpoint", family: nil, variant: nil,
                                 spec: "some-unknown-checkpoint", loadable: false,
                                 reason: "not an engine-known model spec"),
            fallbackCheckpointFamily: "zimage-base")
        #expect(result.checkpointFamily == "zimage-base")
        // A bare (non-path) spec is stored verbatim, exactly as the user
        // typed it — this builder never second-guesses a typed alias.
        #expect(result.model == "some-unknown-checkpoint")
    }

    @Test("a fresh detection overrides a stale fallback rather than being ignored")
    func detectionOverridesFallback() {
        let result = PresetModelFieldBuilder.build(
            modelText: "krea2", loras: [],
            detection: detection(model: "krea2", family: "krea2", variant: "turbo", spec: "krea2"),
            fallbackCheckpointFamily: "raw-stock")
        #expect(result.checkpointFamily == "turbo")
    }

    @Test("z-image base path, no accel lora needed")
    func zImageBasePath() {
        let result = PresetModelFieldBuilder.build(
            modelText: "z-image-base", loras: [],
            detection: detection(model: "z-image-base", family: "z-image", variant: "base",
                                 spec: "z-image-base"),
            fallbackCheckpointFamily: nil)
        #expect(result.model == "z-image-base")
        #expect(result.checkpointFamily == "zimage-base")
    }
}
