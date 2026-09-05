// PresetBackfillViewModel.swift — comfybox#359
//
// "Make expandable" (one preset) / "Backfill all" (the Presets tab header):
// for a preset that is currently label-only because the engine cannot tell
// which base its adapters belong to (`preset_unresolved_reason: "no_model"`),
// write the fields that actually make `POST /v1/generate {"preset": id}`
// expand it, then PROVE it expanded.
//
// FIX ROUND 1 — what changed and why it had to:
//
//   * `checkpoint_family` alone is not the fix. `PresetLoRAStack.decide`
//     (Sources/ZImage/Server/PresetLoRAStack.swift) returns `no_model`
//     whenever the preset's `model` is empty AND the request names none —
//     BEFORE `checkpoint_family` is consulted. `checkpoint_family` only ever
//     matters for a `{preset, model}` request. The clients that send
//     `{preset}` alone (MCP, this desktop app, other REST callers) would have
//     seen exactly nothing change. So the backfill writes `model` — the
//     canonical spec the engine hands back from `GET /v1/model/family` — and
//     `checkpoint_family` alongside it. `custom_model_path` is left as-is;
//     the desktop's own Apply path still reads it.
//   * It refuses to write a spec the engine would not accept. `loadable`
//     false ⇒ `.failed(reason)`, never a cosmetic write.
//   * Success is MEASURED, not assumed: after building the updated preset it
//     re-runs the engine's own decision (`PresetEffectiveRecipePresenter
//     .compute` → `PresetLoRAStack.decide`) and only reports `.updated` when
//     `unresolved == nil`. A write that would not have fixed anything is
//     reported as a failure with the code that is still blocking it.
//   * It never guesses an accelerator role. `raw-accel` vs `raw-stock` turns
//     on `loras[].role == "accel"`, and the 26 affected presets declare no
//     roles at all — see `AcceleratorLoRAHeuristic`.
//
// FIX ROUND 2 — the ambiguity no longer blocks the fix. `declaredFamily`
// maps `raw-accel` and `raw-stock` BOTH to "krea2", so the accel/stock split
// is engine-inert: it is a record for humans and the recipe matrix, not
// something `decide` reads. Gating the whole backfill on it left the preset
// broken over a label that would not have changed the render. So: `model` is
// ALWAYS written when the spec is loadable, and only the LABEL is deferred
// (with a note) when it genuinely depends on an undeclared role. The preset
// becomes expandable on the first click; the label can follow later.
//
// Network calls are injected (`detectFamily` / `save`) so all of the above —
// the actual logic worth testing — runs with no live server.

import Foundation
import SwiftUI

@MainActor
@Observable
public final class PresetBackfillViewModel {

    public struct Outcome: Identifiable, Equatable {
        public enum Status: Equatable {
            /// The preset now expands: `model` was written, and re-running
            /// the engine's decision confirmed it.
            ///
            /// - `model`: the spec written.
            /// - `label`: the `checkpoint_family` this run DERIVED, or nil
            ///   when none could be — any pre-existing label is left as it
            ///   was. nil is the "Updated (label pending)" row.
            /// - `note`: why there is no label. nil when there is one.
            case updated(model: String, label: String?, note: String?)
            /// Nothing to do — already expandable, or not a `no_model` label.
            case skipped(String)
            /// Could not resolve, could not save, or the write would not have
            /// made the preset expandable after all.
            case failed(String)
        }
        public let id: String
        public let name: String
        public let status: Status

        public init(id: String, name: String, status: Status) {
            self.id = id
            self.name = name
            self.status = status
        }
    }

    public private(set) var isRunning = false
    public private(set) var lastResults: [Outcome] = []

    public init() {}

    /// #359 badge: would `/v1/generate {"preset": id}` expand this preset as
    /// a whole (model + LoRA stack), or does it stay a provenance label? Runs
    /// the SAME decision (`PresetLoRAStack.decide`, via
    /// `PresetEffectiveRecipePresenter`) the Effective-recipe panel already
    /// shows for the editor's live field values.
    public static func isExpandable(_ preset: ServerPreset) -> Bool {
        PresetEffectiveRecipePresenter.compute(declared: preset.toImagePreset()).unresolved == nil
    }

    /// A preset this action can plausibly fix: label-only specifically
    /// because the engine can't tell which base its adapters belong to.
    /// Other unresolved reasons (`media_kind:video`, `bypass_declared`,
    /// `engine:mflux`, …) are not model problems and are left alone.
    public static func isBackfillable(_ preset: ServerPreset) -> Bool {
        PresetEffectiveRecipePresenter.compute(declared: preset.toImagePreset()).unresolved?.code == "no_model"
    }

    public static func backfillCandidates(_ presets: [ServerPreset]) -> [ServerPreset] {
        presets.filter(isBackfillable)
    }

    /// The document this backfill would POST, plus what it could and could
    /// not label.
    public struct PlannedWrite: Equatable {
        public var preset: ServerPreset
        /// The `checkpoint_family` this run derived and wrote. nil = none
        /// could be derived; any pre-existing value is left untouched.
        public var label: String?
        /// Why there is no label. nil when there is one.
        public var note: String?
    }

    /// The preset this backfill would write, or the reason it cannot be
    /// built. Pure — `detection` is already in hand. Split out from
    /// `backfill` so the whole decision is testable without the network
    /// fakes, and so the editor could reuse it later.
    public enum Plan: Equatable {
        /// The document to POST.
        case write(PlannedWrite)
        /// Do not write. The status to report instead.
        case stop(Outcome.Status)
    }

    public static func plan(_ preset: ServerPreset, detection: ModelFamilyInfo) -> Plan {
        guard detection.loadable else {
            return .stop(.failed(detection.reason
                ?? "the engine would not accept '\(detection.model)' as a model"))
        }
        let spec = detection.spec.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !spec.isEmpty else {
            return .stop(.failed("the engine returned no canonical model spec for '\(detection.model)'"))
        }

        var updated = preset
        // THE fix: `model` is what `decide` reads first, and round 2 writes it
        // unconditionally once the engine vouches for the spec.
        // `custom_model_path` stays exactly as it was — the desktop's
        // Apply/Set-as-Warm path still uses it, and rewriting it here would be
        // a second, unasked-for change to the document.
        updated.model = spec

        // `checkpoint_family` is the policy label the recipe matrix and
        // `{preset, model}` requests read. Derived only when it can be —
        // never a guess, and never clobbering an existing declaration with
        // nothing.
        var label: String?
        var note: String?
        let unroled = AcceleratorLoRAHeuristic.unroledAcceleratorCandidates(preset.loras)
        if CheckpointFamilyResolver.dependsOnAccelRole(
            family: detection.family, variant: detection.variant), !unroled.isEmpty {
            note = "family label skipped — set the accelerator LoRA's role "
                + "(\(unroled.joined(separator: ", "))) to record accel vs stock"
        } else if let derived = CheckpointFamilyResolver.resolve(
            family: detection.family, variant: detection.variant, loras: preset.loras) {
            label = derived
            updated.checkpointFamily = derived
        } else {
            note = "family label skipped — the engine could not determine the checkpoint "
                + "variant for '\(spec)'"
        }

        // MEASURED, not assumed. This is the engine's own decision
        // (`PresetLoRAStack.decide`) re-run against the document we are about
        // to POST. If it still would not expand, we have learned something
        // and must not report success.
        let recipe = PresetEffectiveRecipePresenter.compute(declared: updated.toImagePreset())
        if let unresolved = recipe.unresolved {
            return .stop(.failed("still not expandable: \(unresolved.code) — \(unresolved.message)"))
        }
        return .write(PlannedWrite(preset: updated, label: label, note: note))
    }

    /// One preset's attempt. `detectFamily` is `EngineService.fetchModelFamily`;
    /// `save` is `EngineService.savePreset`.
    public func backfill(
        _ preset: ServerPreset,
        detectFamily: (String) async -> ModelFamilyInfo?,
        save: (ServerPreset) async throws -> Void
    ) async -> Outcome {
        func outcome(_ status: Outcome.Status) -> Outcome {
            Outcome(id: preset.id, name: preset.name, status: status)
        }
        guard Self.isBackfillable(preset) else {
            return outcome(.skipped("Already expandable, or not a model-family problem"))
        }
        guard let spec = preset.effectiveModelSpec else {
            return outcome(.failed("This preset declares no model or path to resolve"))
        }
        guard let detection = await detectFamily(spec) else {
            return outcome(.failed("Could not reach the engine to classify '\(spec)'"))
        }
        let planned: PlannedWrite
        switch Self.plan(preset, detection: detection) {
        case .stop(let status): return outcome(status)
        case .write(let p): planned = p
        }
        do {
            try await save(planned.preset)
            return outcome(.updated(
                model: planned.preset.model ?? "", label: planned.label, note: planned.note))
        } catch {
            return outcome(.failed(error.localizedDescription))
        }
    }

    /// "Backfill all": every candidate in `presets`, sequentially (each is a
    /// network round trip; there is no batch route to fan these out into).
    /// `save` should also refresh the caller's own preset list on success —
    /// this view model tracks nothing about a live list itself.
    @discardableResult
    public func backfillAll(
        _ presets: [ServerPreset],
        detectFamily: @escaping (String) async -> ModelFamilyInfo?,
        save: @escaping (ServerPreset) async throws -> Void
    ) async -> [Outcome] {
        isRunning = true
        defer { isRunning = false }
        var results: [Outcome] = []
        for preset in Self.backfillCandidates(presets) {
            results.append(await backfill(preset, detectFamily: detectFamily, save: save))
        }
        lastResults = results
        return results
    }
}
