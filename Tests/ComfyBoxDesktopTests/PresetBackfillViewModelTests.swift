// PresetBackfillViewModelTests.swift — comfybox#359
//
// "Make Expandable" / "Backfill all". Network calls (`detectFamily`, `save`)
// are injected fakes — no live engine.
//
// FIX ROUND 1: these cover the real seam end to end — after a backfill,
// `PresetEffectiveRecipePresenter.compute` must report `unresolved == nil`,
// i.e. `/v1/generate {"preset": id}` really does expand the preset now, and
// it must stop being a backfill candidate.
//
// FIX ROUND 2: the fixture shaped like the actual 26 presets
// (`custom_model_path` only, an accelerator LoRA with `role: null`) is now
// UPDATED on the first click — `model` written, `checkpoint_family` LABEL
// deferred with a note — instead of being held back as `needsReview`. The
// accel/stock split is engine-inert (`declaredFamily` maps both to "krea2"),
// so blocking the fix on it left the preset broken over a cosmetic field.

import XCTest
@testable import ComfyBoxDesktop

@MainActor
final class PresetBackfillViewModelTests: XCTestCase {

    /// The real shape of the 26 desktop presets: `custom_model_path` only,
    /// an accel LoRA declared with no `role`, no `checkpoint_family`.
    private func labelOnlyPreset(accelRole: String? = nil) -> ServerPreset {
        ServerPreset(
            name: "test 01",
            customModelPath: "/Users/todd/LocalModels/krea2-raw",
            loras: [
                ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: accelRole),
                ServerPresetLora(filename: "cutifier_krea2.safetensors", scale: 0.8, role: nil),
            ]
        )
    }

    /// A label-only preset whose stack contains nothing accelerator-shaped —
    /// so the role heuristic has no question to raise.
    private func styleOnlyLabelPreset() -> ServerPreset {
        ServerPreset(
            name: "test 02",
            customModelPath: "/Users/todd/LocalModels/krea2-raw",
            loras: [ServerPresetLora(filename: "cutifier_krea2.safetensors", scale: 0.8, role: nil)]
        )
    }

    /// Mirrors the live `krea-kira` shape — already expandable.
    private func expandablePreset() -> ServerPreset {
        ServerPreset(name: "Krea-Kira", model: "krea2-raw", checkpointFamily: "raw-accel")
    }

    /// What the engine returns for `/Users/todd/LocalModels/krea2-raw` on a
    /// machine where that path is the declared `krea2-raw` install.
    private func rawDetection(loadable: Bool = true, reason: String? = nil) -> ModelFamilyInfo {
        ModelFamilyInfo(model: "/Users/todd/LocalModels/krea2-raw", family: "krea2", variant: "raw",
                        spec: "krea2-raw", loadable: loadable, reason: reason)
    }

    // MARK: isExpandable / isBackfillable

    func testIsExpandableMatchesTheEffectiveRecipeDecision() {
        XCTAssertFalse(PresetBackfillViewModel.isExpandable(labelOnlyPreset()))
        XCTAssertTrue(PresetBackfillViewModel.isExpandable(expandablePreset()))
    }

    func testIsBackfillableOnlyForTheNoModelReason() {
        XCTAssertTrue(PresetBackfillViewModel.isBackfillable(labelOnlyPreset()))
        XCTAssertFalse(PresetBackfillViewModel.isBackfillable(expandablePreset()))

        // A video preset is unresolved for a DIFFERENT reason (media_kind) —
        // backfilling a model cannot fix that, so it must not be offered the
        // action.
        var video = labelOnlyPreset()
        video.mediaKind = "video"
        XCTAssertFalse(PresetBackfillViewModel.isBackfillable(video))
    }

    /// The bug this whole round exists for: `checkpoint_family` on its own
    /// does NOT make a model-less preset expandable, because
    /// `PresetLoRAStack.decide` returns `no_model` before reading it.
    func testCheckpointFamilyAloneDoesNotMakeAPresetExpandable() {
        var labelled = labelOnlyPreset()
        labelled.checkpointFamily = "raw-stock"
        XCTAssertFalse(PresetBackfillViewModel.isExpandable(labelled),
                       "checkpoint_family is read only for a {preset, model} request")
        XCTAssertTrue(PresetBackfillViewModel.isBackfillable(labelled),
                      "still a no_model label, so still a candidate")
    }

    func testBackfillCandidatesFiltersToNoModelOnly() {
        let candidates = PresetBackfillViewModel.backfillCandidates([labelOnlyPreset(), expandablePreset()])
        XCTAssertEqual(candidates.map(\.name), ["test 01"])
    }

    // MARK: round 2 — an un-roled accelerator defers the LABEL, never the fix

    func testTheRealFixtureIsFixedOnTheFirstClickWithTheLabelDeferred() async {
        let vm = PresetBackfillViewModel()
        var saved: ServerPreset?
        let outcome = await vm.backfill(
            labelOnlyPreset(),
            detectFamily: { _ in self.rawDetection() },
            save: { saved = $0 }
        )
        guard case .updated(let model, let label, let note) = outcome.status else {
            return XCTFail("expected .updated, got \(outcome.status)")
        }
        XCTAssertEqual(model, "krea2-raw")
        XCTAssertNil(label, "raw-accel vs raw-stock is unknowable — do not guess it")
        XCTAssertTrue(note?.contains("krea2_turbo_distill_r256.safetensors") == true, note ?? "nil")

        guard let saved else { return XCTFail("nothing was saved") }
        XCTAssertNil(saved.checkpointFamily, "no label was invented")
        XCTAssertTrue(PresetBackfillViewModel.isExpandable(saved),
                      "expandability must never wait on a cosmetic label")
    }

    func testADeferredLabelDoesNotClobberAnExistingOne() async {
        var preset = labelOnlyPreset()
        preset.checkpointFamily = "raw-accel"
        let vm = PresetBackfillViewModel()
        var saved: ServerPreset?
        _ = await vm.backfill(preset, detectFamily: { _ in self.rawDetection() }, save: { saved = $0 })
        XCTAssertEqual(saved?.checkpointFamily, "raw-accel")
    }

    func testAStackWithNoAcceleratorShapedLoraGetsItsLabel() async {
        let vm = PresetBackfillViewModel()
        let outcome = await vm.backfill(
            styleOnlyLabelPreset(),
            detectFamily: { _ in self.rawDetection() },
            save: { _ in }
        )
        guard case .updated(_, let label, let note) = outcome.status else {
            return XCTFail("expected .updated, got \(outcome.status)")
        }
        XCTAssertEqual(label, "raw-stock", "no accel role declared and none implied")
        XCTAssertNil(note)
    }

    func testAnUnroledAcceleratorDoesNotDeferATurboLabel() async {
        // Only krea2 "raw" splits accel/stock; turbo is `turbo` regardless.
        let vm = PresetBackfillViewModel()
        let turbo = ModelFamilyInfo(model: "krea2", family: "krea2", variant: "turbo",
                                    spec: "krea2", loadable: true)
        let outcome = await vm.backfill(
            labelOnlyPreset(), detectFamily: { _ in turbo }, save: { _ in })
        guard case .updated(_, let label, _) = outcome.status else {
            return XCTFail("expected .updated, got \(outcome.status)")
        }
        XCTAssertEqual(label, "turbo")
    }

    /// Ruling 5 at the desktop seam: a z-image checkpoint the engine will not
    /// pin to a variant still gets `model`, just no label.
    func testAZImagePresetWithNoVariantIsStillMadeExpandable() async {
        let path = "/Users/todd/Models-working/cyberrealistic-z-image/cyberrealisticZImage_v50.safetensors"
        var preset = styleOnlyLabelPreset()
        preset.customModelPath = path
        let vm = PresetBackfillViewModel()
        var saved: ServerPreset?
        let outcome = await vm.backfill(
            preset,
            detectFamily: { _ in ModelFamilyInfo(model: path, family: "z-image", variant: nil,
                                                 spec: path, loadable: true) },
            save: { saved = $0 })
        guard case .updated(let model, let label, let note) = outcome.status else {
            return XCTFail("expected .updated, got \(outcome.status)")
        }
        XCTAssertEqual(model, path)
        XCTAssertNil(label)
        XCTAssertTrue(note?.contains("variant") == true, note ?? "nil")
        XCTAssertTrue(PresetBackfillViewModel.isExpandable(saved!))
    }

    // MARK: the write itself — model, not just checkpoint_family (ruling 2)

    func testBackfillWritesTheCanonicalModelSpecAndMeasuresSuccess() async {
        let vm = PresetBackfillViewModel()
        var saved: ServerPreset?
        let outcome = await vm.backfill(
            labelOnlyPreset(accelRole: "accel"),
            detectFamily: { spec in
                XCTAssertEqual(spec, "/Users/todd/LocalModels/krea2-raw")
                return self.rawDetection()
            },
            save: { preset in saved = preset }
        )
        guard case .updated(let model, let label, _) = outcome.status else {
            return XCTFail("expected .updated, got \(outcome.status)")
        }
        XCTAssertEqual(model, "krea2-raw", "the ENGINE's canonical spec, not the raw path")
        XCTAssertEqual(label, "raw-accel")

        guard let written = saved else { return XCTFail("nothing was saved") }
        XCTAssertEqual(written.model, "krea2-raw")
        XCTAssertEqual(written.checkpointFamily, "raw-accel")
        XCTAssertEqual(written.customModelPath, "/Users/todd/LocalModels/krea2-raw",
                       "custom_model_path is left alone — the desktop's Apply path still reads it")

        // The measurement the controller asked for: the written preset really
        // does expand, and is no longer a candidate.
        XCTAssertTrue(PresetBackfillViewModel.isExpandable(written))
        XCTAssertFalse(PresetBackfillViewModel.isBackfillable(written))
    }

    func testBackfillRefusesASpecTheEngineWouldNotLoad() async {
        let vm = PresetBackfillViewModel()
        let outcome = await vm.backfill(
            labelOnlyPreset(accelRole: "accel"),
            detectFamily: { _ in
                self.rawDetection(loadable: false,
                                  reason: "'/Users/todd/LocalModels/krea2-raw' does not exist on this machine")
            },
            save: { _ in XCTFail("must not write a model spec the engine would refuse") }
        )
        guard case .failed(let message) = outcome.status else {
            return XCTFail("expected .failed, got \(outcome.status)")
        }
        XCTAssertTrue(message.contains("does not exist"), message)
    }

    /// Ruling 3: if the write would not actually have made the preset
    /// expandable, that is a FAILURE — not a success with a shrug.
    func testAWriteThatWouldNotFixThePresetIsReportedAsAFailure() async {
        var preset = labelOnlyPreset(accelRole: "accel")
        // A video preset can never expand on the image path, whatever model
        // it names. `plan` must notice after building the document.
        preset.mediaKind = "video"
        let planned = PresetBackfillViewModel.plan(preset, detection: rawDetection())
        guard case .stop(.failed(let message)) = planned else {
            return XCTFail("expected .failed, got \(planned)")
        }
        XCTAssertTrue(message.hasPrefix("still not expandable:"), message)
        XCTAssertTrue(message.contains("media_kind"), message)
    }

    func testBackfillSkipsAnAlreadyExpandablePreset() async {
        let vm = PresetBackfillViewModel()
        var detectFamilyCalled = false
        let outcome = await vm.backfill(
            expandablePreset(),
            detectFamily: { _ in detectFamilyCalled = true; return self.rawDetection() },
            save: { _ in XCTFail("must not save an already-expandable preset") }
        )
        guard case .skipped = outcome.status else {
            return XCTFail("expected .skipped, got \(outcome.status)")
        }
        XCTAssertFalse(detectFamilyCalled, "no reason to call the engine for a preset that needs no backfill")
    }

    func testBackfillStillWritesTheModelWhenTheFamilyLabelCannotBeDerived() async {
        // The engine knows it can load the spec but cannot classify it into
        // one of the five policy labels. `model` alone is enough for
        // `decide`, so this must still succeed — with no checkpoint_family.
        let vm = PresetBackfillViewModel()
        var saved: ServerPreset?
        let detection = ModelFamilyInfo(
            model: "/Users/todd/LocalModels/krea2-raw", family: nil, variant: nil,
            spec: "/Users/todd/LocalModels/krea2-raw", loadable: true)
        let outcome = await vm.backfill(
            styleOnlyLabelPreset(), detectFamily: { _ in detection }, save: { saved = $0 })
        guard case .updated(let model, let label, _) = outcome.status else {
            return XCTFail("expected .updated, got \(outcome.status)")
        }
        XCTAssertEqual(model, "/Users/todd/LocalModels/krea2-raw")
        XCTAssertNil(label)
        XCTAssertNil(saved?.checkpointFamily)
        XCTAssertTrue(PresetBackfillViewModel.isExpandable(saved!))
    }

    func testBackfillFailsWhenTheEngineIsUnreachable() async {
        let vm = PresetBackfillViewModel()
        let outcome = await vm.backfill(
            labelOnlyPreset(accelRole: "accel"),
            detectFamily: { _ in nil },
            save: { _ in XCTFail("must not save without a detection result") }
        )
        guard case .failed = outcome.status else {
            return XCTFail("expected .failed, got \(outcome.status)")
        }
    }

    func testBackfillFailsWhenSaveThrows() async {
        struct SaveError: Error {}
        let vm = PresetBackfillViewModel()
        let outcome = await vm.backfill(
            labelOnlyPreset(accelRole: "accel"),
            detectFamily: { _ in self.rawDetection() },
            save: { _ in throw SaveError() }
        )
        guard case .failed = outcome.status else {
            return XCTFail("expected .failed, got \(outcome.status)")
        }
    }

    /// An older engine that doesn't report `spec`/`loadable` must not be
    /// treated as permission to write.
    func testAnEngineThatDoesNotReportLoadabilityIsRefused() throws {
        let json = Data("""
        {"model":"/Users/todd/LocalModels/krea2-raw","family":"krea2","variant":"raw"}
        """.utf8)
        let decoded = try JSONDecoder().decode(ModelFamilyInfo.self, from: json)
        XCTAssertFalse(decoded.loadable)
        XCTAssertEqual(decoded.spec, "/Users/todd/LocalModels/krea2-raw")
        guard case .stop(.failed) = PresetBackfillViewModel.plan(
            styleOnlyLabelPreset(), detection: decoded) else {
            return XCTFail("expected .failed for an engine that cannot vouch for the spec")
        }
    }

    func testANewEngineResponseDecodesSpecAndLoadable() throws {
        let json = Data("""
        {"model":"~/LocalModels/krea2-raw","family":"krea2","variant":"raw",
         "spec":"krea2-raw","loadable":true,"reason":null}
        """.utf8)
        let decoded = try JSONDecoder().decode(ModelFamilyInfo.self, from: json)
        XCTAssertEqual(decoded.spec, "krea2-raw")
        XCTAssertTrue(decoded.loadable)
        XCTAssertNil(decoded.reason)
    }

    // MARK: backfillAll — the batch action

    func testBackfillAllOnlyTouchesCandidatesAndReportsEachResult() async {
        let vm = PresetBackfillViewModel()
        var savedNames: [String] = []
        let presets = [styleOnlyLabelPreset(), expandablePreset()]
        let results = await vm.backfillAll(
            presets,
            detectFamily: { _ in self.rawDetection() },
            save: { preset in savedNames.append(preset.name) }
        )
        XCTAssertEqual(results.map(\.name), ["test 02"], "the already-expandable preset is not a candidate at all")
        XCTAssertEqual(savedNames, ["test 02"])
        XCTAssertEqual(vm.lastResults, results)
        XCTAssertFalse(vm.isRunning)
    }

    /// The batch's third bucket is "Updated (label pending)", not a hold-back:
    /// BOTH presets are written and expandable; one just has no label yet.
    func testBackfillAllReportsLabelPendingAlongsideFullyLabelledUpdates() async {
        let vm = PresetBackfillViewModel()
        var savedNames: [String] = []
        let results = await vm.backfillAll(
            [labelOnlyPreset(), styleOnlyLabelPreset()],
            detectFamily: { _ in self.rawDetection() },
            save: { savedNames.append($0.name) }
        )
        XCTAssertEqual(results.count, 2)
        guard case .updated(_, let pending, let note) = results[0].status else {
            return XCTFail("expected test 01 updated, got \(results[0].status)")
        }
        XCTAssertNil(pending)
        XCTAssertNotNil(note)
        guard case .updated(_, let labelled, _) = results[1].status else {
            return XCTFail("expected test 02 updated, got \(results[1].status)")
        }
        XCTAssertEqual(labelled, "raw-stock")
        XCTAssertEqual(savedNames, ["test 01", "test 02"], "round 2: both are fixed")
    }

    /// Ruling 7: an engine that ANSWERED with an error is not "unreachable",
    /// and its own words reach the Failed row.
    func testAnEngineRefusalSurfacesTheEnginesOwnMessage() async {
        let refusal = EngineService.refusal(
            spec: "/nope", status: 400, message: "model query parameter is required")
        XCTAssertFalse(refusal.loadable)
        let vm = PresetBackfillViewModel()
        let outcome = await vm.backfill(
            styleOnlyLabelPreset(),
            detectFamily: { _ in refusal },
            save: { _ in XCTFail("must not save on a refusal") })
        guard case .failed(let message) = outcome.status else {
            return XCTFail("expected .failed, got \(outcome.status)")
        }
        XCTAssertTrue(message.contains("model query parameter is required"), message)
        XCTAssertTrue(message.contains("400"), message)
    }

    func testBackfillAllOfNoCandidatesReturnsEmptyWithoutTouchingTheNetwork() async {
        let vm = PresetBackfillViewModel()
        var called = false
        let results = await vm.backfillAll(
            [expandablePreset()],
            detectFamily: { _ in called = true; return self.rawDetection() },
            save: { _ in called = true }
        )
        XCTAssertTrue(results.isEmpty)
        XCTAssertFalse(called)
    }
}
