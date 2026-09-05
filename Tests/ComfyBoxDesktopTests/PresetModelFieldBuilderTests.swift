// PresetModelFieldBuilderTests.swift — comfybox#359
//
// `ServerPresetEditor.buildPreset()`'s model/checkpoint_family mapping,
// extracted so it's testable without a live engine or a SwiftUI view.

import Testing
@testable import ComfyBoxDesktop

@Suite("PresetModelFieldBuilder")
struct PresetModelFieldBuilderTests {

    @Test("a bare alias goes to model, not custom_model_path — unchanged from today")
    func aliasGoesToModel() {
        let result = PresetModelFieldBuilder.build(
            modelText: "krea2-raw", loras: [],
            detectedFamily: "krea2", detectedVariant: "raw",
            fallbackCheckpointFamily: nil)
        #expect(result.model == "krea2-raw")
        #expect(result.customModelPath == nil)
        #expect(result.checkpointFamily == "raw-stock")
    }

    @Test("a path goes to custom_model_path, never model — the real shape of the 26 desktop presets")
    func pathGoesToCustomModelPath() {
        let loras = [ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: "accel")]
        let result = PresetModelFieldBuilder.build(
            modelText: "/Users/todd/LocalModels/krea2-raw", loras: loras,
            detectedFamily: "krea2", detectedVariant: "raw",
            fallbackCheckpointFamily: nil)
        #expect(result.model == nil)
        #expect(result.customModelPath == "/Users/todd/LocalModels/krea2-raw")
        #expect(result.checkpointFamily == "raw-accel")
    }

    @Test("a tilde path is also treated as a path")
    func tildePathIsAPath() {
        let result = PresetModelFieldBuilder.build(
            modelText: "~/LocalModels/krea2-raw", loras: [],
            detectedFamily: "krea2", detectedVariant: "raw",
            fallbackCheckpointFamily: nil)
        #expect(result.model == nil)
        #expect(result.customModelPath == "~/LocalModels/krea2-raw")
    }

    @Test("an empty model field clears both model and custom_model_path")
    func emptyModelClearsBoth() {
        let result = PresetModelFieldBuilder.build(
            modelText: "   ", loras: [],
            detectedFamily: nil, detectedVariant: nil,
            fallbackCheckpointFamily: "raw-stock")
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
            detectedFamily: nil, detectedVariant: nil,
            fallbackCheckpointFamily: "zimage-base")
        #expect(result.checkpointFamily == "zimage-base")
    }

    @Test("a fresh detection overrides a stale fallback rather than being ignored")
    func detectionOverridesFallback() {
        let result = PresetModelFieldBuilder.build(
            modelText: "krea2", loras: [],
            detectedFamily: "krea2", detectedVariant: "turbo",
            fallbackCheckpointFamily: "raw-stock")
        #expect(result.checkpointFamily == "turbo")
    }

    @Test("z-image base path, no accel lora needed")
    func zImageBasePath() {
        let result = PresetModelFieldBuilder.build(
            modelText: "z-image-base", loras: [],
            detectedFamily: "z-image", detectedVariant: "base",
            fallbackCheckpointFamily: nil)
        #expect(result.model == "z-image-base")
        #expect(result.checkpointFamily == "zimage-base")
    }
}
