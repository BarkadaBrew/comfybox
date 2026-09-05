// PresetModelFieldBuilderTests.swift — comfybox#359
//
// `ServerPresetEditor.buildPreset()`'s model/checkpoint_family mapping,
// extracted so it's testable without a live engine or a SwiftUI view.
//
// FIX ROUND 1: the editor must write `model` for a typed PATH too (the
// canonical spec the engine returned), because that is the only field
// `PresetLoRAStack.decide` reads before it gives up with `no_model`.
//
// FIX ROUND 2: (1) save must never ERASE a `model` a backfill wrote, and
// (3) an un-roled accelerator LoRA skips the family LABEL only — `model` is
// written regardless, so the preset is expandable on the first save.

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

    private func build(
        _ modelText: String,
        loras: [ServerPresetLora] = [],
        detection: ModelFamilyInfo? = nil,
        fallbackModel: String? = nil,
        fallbackCustomModelPath: String? = nil,
        fallbackCheckpointFamily: String? = nil
    ) -> PresetModelFieldBuilder.Result {
        PresetModelFieldBuilder.build(
            modelText: modelText, loras: loras, detection: detection,
            fallbackModel: fallbackModel,
            fallbackCustomModelPath: fallbackCustomModelPath,
            fallbackCheckpointFamily: fallbackCheckpointFamily)
    }

    // MARK: - Field mapping

    @Test("a bare alias goes to model, not custom_model_path — unchanged from today")
    func aliasGoesToModel() {
        let result = build(
            "krea2-raw",
            detection: detection(model: "krea2-raw", family: "krea2", variant: "raw", spec: "krea2-raw"))
        #expect(result.model == "krea2-raw")
        #expect(result.customModelPath == nil)
        #expect(result.checkpointFamily == "raw-stock")
    }

    @Test("a path fills BOTH model (canonical spec) and custom_model_path — the round-1 fix")
    func pathFillsModelAndCustomPath() {
        let loras = [ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: "accel")]
        let result = build(
            "/Users/todd/LocalModels/krea2-raw", loras: loras,
            detection: detection(model: "/Users/todd/LocalModels/krea2-raw",
                                 family: "krea2", variant: "raw", spec: "krea2-raw"))
        #expect(result.model == "krea2-raw", "the engine's canonical spec, not the raw path")
        #expect(result.customModelPath == "/Users/todd/LocalModels/krea2-raw")
        #expect(result.checkpointFamily == "raw-accel")
    }

    @Test("a tilde path is also treated as a path, and takes the engine's expanded spec")
    func tildePathIsAPath() {
        let result = build(
            "~/LocalModels/krea2-raw",
            detection: detection(model: "~/LocalModels/krea2-raw", family: "krea2", variant: "raw",
                                 spec: "/Users/todd/LocalModels/krea2-raw"))
        #expect(result.customModelPath == "~/LocalModels/krea2-raw")
        #expect(result.model == "/Users/todd/LocalModels/krea2-raw",
                "ModelResolution does not expand ~, so the engine's expanded spec is what gets stored")
    }

    @Test("an empty model field clears both model and custom_model_path")
    func emptyModelClearsBoth() {
        let result = build("   ", fallbackModel: "krea2-raw", fallbackCheckpointFamily: "raw-stock")
        #expect(result.model == nil, "emptying the field is an explicit user action, not a detection miss")
        #expect(result.customModelPath == nil)
        // No model at all to detect from — the existing declaration (if any)
        // survives; nothing here should silently erase it.
        #expect(result.checkpointFamily == "raw-stock")
    }

    @Test("a bare spec the engine does not know is still stored verbatim")
    func unknownBareSpecIsStoredVerbatim() {
        let result = build(
            "some-unknown-checkpoint",
            detection: detection(model: "some-unknown-checkpoint", family: nil, variant: nil,
                                 spec: "some-unknown-checkpoint", loadable: false,
                                 reason: "not an engine-known model spec"),
            fallbackCheckpointFamily: "zimage-base")
        #expect(result.model == "some-unknown-checkpoint", "this builder never second-guesses a typed alias")
        #expect(result.checkpointFamily == "zimage-base")
    }

    @Test("a fresh detection overrides a stale fallback rather than being ignored")
    func detectionOverridesFallback() {
        let result = build(
            "krea2",
            detection: detection(model: "krea2", family: "krea2", variant: "turbo", spec: "krea2"),
            fallbackCheckpointFamily: "raw-stock")
        #expect(result.checkpointFamily == "turbo")
    }

    @Test("z-image base alias, no accel lora needed")
    func zImageBaseAlias() {
        let result = build(
            "z-image-base",
            detection: detection(model: "z-image-base", family: "z-image", variant: "base",
                                 spec: "z-image-base"))
        #expect(result.model == "z-image-base")
        #expect(result.checkpointFamily == "zimage-base")
    }

    @Test("a z-image checkpoint with no declared variant gets model but no label (ruling 5)")
    func zImageWithoutAVariantGetsNoLabel() {
        let path = "/Users/todd/Models-working/cyberrealistic-z-image/cyberrealisticZImage_v50.safetensors"
        let result = build(
            path,
            detection: detection(model: path, family: "z-image", variant: nil, spec: path))
        #expect(result.model == path, "still expandable")
        #expect(result.checkpointFamily == nil, "cyberrealistic is served as BASE — never guess turbo from a filename")
    }

    // MARK: - Round 2, ruling 1 (CRITICAL): save must never erase a
    // backfilled `model`.
    //
    // `buildPreset()` runs on every Save. If the engine is unreachable — or
    // the user hits Save before `.task(id: model)` has answered — detection
    // is nil. Leaving `model = nil` then wrote a nil over the spec a backfill
    // had just put there and reverted the preset to `no_model`: the bug this
    // whole PR exists to fix, reintroduced by its own editor.

    @Test("a detection miss on an UNCHANGED path keeps the backfilled model")
    func detectionMissKeepsTheExistingModel() {
        let result = build(
            "/Users/todd/LocalModels/krea2-raw",
            detection: nil,
            fallbackModel: "krea2-raw",
            fallbackCustomModelPath: "/Users/todd/LocalModels/krea2-raw",
            fallbackCheckpointFamily: "raw-stock")
        #expect(result.model == "krea2-raw", "the engine was silent — do not erase what a backfill wrote")
        #expect(result.customModelPath == "/Users/todd/LocalModels/krea2-raw")
        #expect(result.checkpointFamily == "raw-stock", "a detection miss never erases a valid declaration")
    }

    @Test("a detection miss on a CHANGED path clears the stale model")
    func detectionMissOnAChangedPathClearsTheModel() {
        let result = build(
            "/Users/todd/LocalModels/some-other-checkpoint",
            detection: nil,
            fallbackModel: "krea2-raw",
            fallbackCustomModelPath: "/Users/todd/LocalModels/krea2-raw",
            fallbackCheckpointFamily: "raw-stock")
        #expect(result.model == nil,
                "the user pointed the preset somewhere else — the old base no longer describes it")
        #expect(result.customModelPath == "/Users/todd/LocalModels/some-other-checkpoint")
    }

    @Test("an unloadable answer on an unchanged path also keeps the existing model")
    func unloadableOnAnUnchangedPathKeepsTheExistingModel() {
        // The engine says the PATH is not loadable; the stored `model` may be
        // a declared alias that is. Only a path change clears it.
        let result = build(
            "/Users/todd/LocalModels/krea2-raw",
            detection: detection(model: "/Users/todd/LocalModels/krea2-raw", family: nil, variant: nil,
                                 spec: "/Users/todd/LocalModels/krea2-raw", loadable: false,
                                 reason: "does not exist on this machine"),
            fallbackModel: "krea2-raw",
            fallbackCustomModelPath: "/Users/todd/LocalModels/krea2-raw")
        #expect(result.model == "krea2-raw")
    }

    @Test("whitespace around an unchanged path is not a path change")
    func whitespaceIsNotAPathChange() {
        let result = build(
            "  /Users/todd/LocalModels/krea2-raw  ",
            detection: nil,
            fallbackModel: "krea2-raw",
            fallbackCustomModelPath: "  /Users/todd/LocalModels/krea2-raw")
        #expect(result.model == "krea2-raw")
    }

    @Test("a detection for a DIFFERENT spec is ignored, never applied to the new path")
    func staleDetectionIsIgnored() {
        // Round 3, item 1. `.task(id: model)` is async: repoint the preset and
        // hit Save immediately and `detectedModelFamily` still holds the
        // answer for the OLD path. Pairing the new `custom_model_path` with
        // the old `model` would point the preset at a base nobody chose —
        // silently, and reporting success.
        let stale = detection(model: "/Users/todd/LocalModels/krea2-raw",
                              family: "krea2", variant: "raw", spec: "krea2-raw")
        let result = build(
            "/Users/todd/LocalModels/some-other-checkpoint",
            detection: stale,
            fallbackModel: "krea2-raw",
            fallbackCustomModelPath: "/Users/todd/LocalModels/krea2-raw")
        #expect(result.customModelPath == "/Users/todd/LocalModels/some-other-checkpoint")
        #expect(result.model == nil, "must never pair the new path with the old spec")
        #expect(result.checkpointFamily == nil, "and the stale LABEL is not applied either")
    }

    @Test("a stale detection cannot label a bare spec it was not asked about")
    func staleDetectionDoesNotLabelABareSpec() {
        let stale = detection(model: "krea2", family: "krea2", variant: "turbo", spec: "krea2")
        let result = build("z-image-base", detection: stale, fallbackCheckpointFamily: "zimage-base")
        #expect(result.model == "z-image-base")
        #expect(result.checkpointFamily == "zimage-base", "the preset's own declaration, not krea2 turbo")
    }

    @Test("a detection whose echoed model matches — modulo whitespace — is applied")
    func matchingDetectionIsApplied() {
        let result = build(
            "  /Users/todd/LocalModels/krea2-raw ",
            detection: detection(model: "/Users/todd/LocalModels/krea2-raw",
                                 family: "krea2", variant: "raw", spec: "krea2-raw"))
        #expect(result.model == "krea2-raw")
        #expect(result.checkpointFamily == "raw-stock")
    }

    @Test("a preset whose PATH lives in model (no custom_model_path) does not lose it on save")
    func pathOnlyInModelSurvivesADetectionMiss() {
        // Round 3, item 2. `ServerPresetEditor` seeds its field from
        // `customModelPath ?? model`, so a preset that carries a path in
        // `model` alone shows that path — and `isUnchangedPath` used to
        // compare it against a nil `customModelPath`, call it "changed", and
        // clear `model` on a plain Save with the engine down.
        let path = "/Users/todd/LocalModels/krea2-raw"
        let result = build(path, detection: nil, fallbackModel: path, fallbackCustomModelPath: nil)
        #expect(result.model == path)
        #expect(result.customModelPath == path)
    }

    @Test("a path typed over a preset whose model was a bare ALIAS is a real change")
    func typingAPathOverAnAliasStillClearsTheModel() {
        let result = build(
            "/Users/todd/LocalModels/krea2-raw",
            detection: nil, fallbackModel: "krea2-raw", fallbackCustomModelPath: nil)
        #expect(result.model == nil, "the alias never described this path")
    }

    @Test("with no fallback model, a detection miss on a path behaves as it always did")
    func noFallbackModelStillYieldsNil() {
        let result = build("/Users/todd/LocalModels/krea2-raw", detection: nil)
        #expect(result.model == nil)
        #expect(result.customModelPath == "/Users/todd/LocalModels/krea2-raw")
    }

    // MARK: - Round 2, ruling 3: no gate on the accel role — write `model`,
    // skip only the LABEL.

    @Test("an un-roled accelerator skips the family label but still writes model")
    func unroledAcceleratorSkipsOnlyTheLabel() {
        let loras = [
            ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: nil),
            ServerPresetLora(filename: "cutifier_krea2.safetensors", scale: 0.8, role: nil),
        ]
        let result = build(
            "/Users/todd/LocalModels/krea2-raw", loras: loras,
            detection: detection(model: "/Users/todd/LocalModels/krea2-raw",
                                 family: "krea2", variant: "raw", spec: "krea2-raw"))
        #expect(result.model == "krea2-raw", "expandability never waits on a label")
        #expect(result.checkpointFamily == nil, "raw-accel vs raw-stock is unknowable here — do not guess")
    }

    @Test("an un-roled accelerator keeps an existing label rather than clearing it")
    func unroledAcceleratorKeepsAnExistingLabel() {
        let loras = [ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: nil)]
        let result = build(
            "krea2-raw", loras: loras,
            detection: detection(model: "krea2-raw", family: "krea2", variant: "raw", spec: "krea2-raw"),
            fallbackCheckpointFamily: "raw-accel")
        #expect(result.checkpointFamily == "raw-accel")
    }

    @Test("an un-roled accelerator does NOT block a turbo label — there is no accel/stock split there")
    func unroledAcceleratorDoesNotBlockATurboLabel() {
        let loras = [ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: nil)]
        let result = build(
            "krea2", loras: loras,
            detection: detection(model: "krea2", family: "krea2", variant: "turbo", spec: "krea2"))
        #expect(result.checkpointFamily == "turbo")
    }

    @Test("an un-roled accelerator does NOT block a z-image label either")
    func unroledAcceleratorDoesNotBlockAZImageLabel() {
        let loras = [ServerPresetLora(filename: "lightning_8step.safetensors", scale: 1.0, role: nil)]
        let result = build(
            "z-image-base", loras: loras,
            detection: detection(model: "z-image-base", family: "z-image", variant: "base",
                                 spec: "z-image-base"))
        #expect(result.checkpointFamily == "zimage-base")
    }
}
