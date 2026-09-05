// PresetBackfillViewModelTests.swift — comfybox#359
//
// "Make Expandable" / "Backfill all". Network calls (`detectFamily`, `save`)
// are injected fakes — no live engine.

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

    /// Mirrors the live `krea-kira` shape — already expandable.
    private func expandablePreset() -> ServerPreset {
        ServerPreset(name: "Krea-Kira", model: "krea2-raw", checkpointFamily: "raw-accel")
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
        // backfilling checkpoint_family cannot fix that, so it must not be
        // offered the action.
        var video = labelOnlyPreset()
        video.mediaKind = "video"
        XCTAssertFalse(PresetBackfillViewModel.isBackfillable(video))
    }

    func testBackfillCandidatesFiltersToNoModelOnly() {
        let candidates = PresetBackfillViewModel.backfillCandidates([labelOnlyPreset(), expandablePreset()])
        XCTAssertEqual(candidates.map(\.name), ["test 01"])
    }

    // MARK: backfill(_:detectFamily:save:) — one preset

    func testBackfillWritesCheckpointFamilyAndReportsUpdated() async {
        let vm = PresetBackfillViewModel()
        var saved: ServerPreset?
        let outcome = await vm.backfill(
            labelOnlyPreset(),
            detectFamily: { spec in
                XCTAssertEqual(spec, "/Users/todd/LocalModels/krea2-raw")
                return ModelFamilyInfo(model: spec, family: "krea2", variant: "raw")
            },
            save: { preset in saved = preset }
        )
        guard case .updated(let family) = outcome.status else {
            return XCTFail("expected .updated, got \(outcome.status)")
        }
        XCTAssertEqual(family, "raw-stock", "no lora declares role accel in this fixture")
        XCTAssertEqual(saved?.checkpointFamily, "raw-stock")
    }

    func testBackfillReadsTheAccelRoleWhenPresent() async {
        let vm = PresetBackfillViewModel()
        let outcome = await vm.backfill(
            labelOnlyPreset(accelRole: "accel"),
            detectFamily: { ModelFamilyInfo(model: $0, family: "krea2", variant: "raw") },
            save: { _ in }
        )
        guard case .updated(let family) = outcome.status else {
            return XCTFail("expected .updated, got \(outcome.status)")
        }
        XCTAssertEqual(family, "raw-accel")
    }

    func testBackfillSkipsAnAlreadyExpandablePreset() async {
        let vm = PresetBackfillViewModel()
        var detectFamilyCalled = false
        let outcome = await vm.backfill(
            expandablePreset(),
            detectFamily: { spec in detectFamilyCalled = true; return ModelFamilyInfo(model: spec, family: "krea2", variant: "raw") },
            save: { _ in XCTFail("must not save an already-expandable preset") }
        )
        guard case .skipped = outcome.status else {
            return XCTFail("expected .skipped, got \(outcome.status)")
        }
        XCTAssertFalse(detectFamilyCalled, "no reason to call the engine for a preset that needs no backfill")
    }

    func testBackfillFailsWhenTheEngineCannotClassifyTheSpec() async {
        let vm = PresetBackfillViewModel()
        let outcome = await vm.backfill(
            labelOnlyPreset(),
            detectFamily: { ModelFamilyInfo(model: $0, family: nil, variant: nil) },
            save: { _ in XCTFail("must not save when nothing was resolved") }
        )
        guard case .failed = outcome.status else {
            return XCTFail("expected .failed, got \(outcome.status)")
        }
    }

    func testBackfillFailsWhenTheEngineIsUnreachable() async {
        let vm = PresetBackfillViewModel()
        let outcome = await vm.backfill(
            labelOnlyPreset(),
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
            labelOnlyPreset(),
            detectFamily: { ModelFamilyInfo(model: $0, family: "krea2", variant: "raw") },
            save: { _ in throw SaveError() }
        )
        guard case .failed = outcome.status else {
            return XCTFail("expected .failed, got \(outcome.status)")
        }
    }

    // MARK: backfillAll — the batch action

    func testBackfillAllOnlyTouchesCandidatesAndReportsEachResult() async {
        let vm = PresetBackfillViewModel()
        var savedNames: [String] = []
        let presets = [labelOnlyPreset(), expandablePreset()]
        let results = await vm.backfillAll(
            presets,
            detectFamily: { ModelFamilyInfo(model: $0, family: "krea2", variant: "raw") },
            save: { preset in savedNames.append(preset.name) }
        )
        XCTAssertEqual(results.map(\.name), ["test 01"], "the already-expandable preset is not a candidate at all")
        XCTAssertEqual(savedNames, ["test 01"])
        XCTAssertEqual(vm.lastResults, results)
        XCTAssertFalse(vm.isRunning)
    }

    func testBackfillAllOfNoCandidatesReturnsEmptyWithoutTouchingTheNetwork() async {
        let vm = PresetBackfillViewModel()
        var called = false
        let results = await vm.backfillAll(
            [expandablePreset()],
            detectFamily: { called = true; return ModelFamilyInfo(model: $0, family: "krea2", variant: "raw") },
            save: { _ in called = true }
        )
        XCTAssertTrue(results.isEmpty)
        XCTAssertFalse(called)
    }
}
