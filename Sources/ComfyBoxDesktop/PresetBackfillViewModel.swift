// PresetBackfillViewModel.swift — comfybox#359
//
// "Make expandable" (one preset) / "Backfill all" (the Presets tab header):
// for a preset that is currently label-only because the engine cannot tell
// which model family its adapters belong to (`preset_unresolved_reason:
// "no_model"`), resolve `checkpoint_family` via the engine's own model
// detection (`GET /v1/model/family`) and the preset's own declared LoRA
// roles (`CheckpointFamilyResolver`), then patch it through the existing
// preset update route (`POST /v1/presets`, upsert).
//
// Network calls are injected (`detectFamily` / `save`) so what counts as
// backfillable and how one attempt's outcome is classified — the actual
// logic worth testing — runs with no live server.

import Foundation
import SwiftUI

@MainActor
@Observable
public final class PresetBackfillViewModel {

    public struct Outcome: Identifiable, Equatable {
        public enum Status: Equatable {
            /// `checkpoint_family` was written; the value is the label used.
            case updated(String)
            /// Nothing to do — already expandable, or not a `no_model` label.
            case skipped(String)
            /// Could not resolve or could not save.
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
    /// because the engine can't tell which family its adapters belong to.
    /// Other unresolved reasons (`media_kind:video`, `bypass_declared`,
    /// `engine:mflux`, …) are not model-family problems and are left alone.
    public static func isBackfillable(_ preset: ServerPreset) -> Bool {
        PresetEffectiveRecipePresenter.compute(declared: preset.toImagePreset()).unresolved?.code == "no_model"
    }

    public static func backfillCandidates(_ presets: [ServerPreset]) -> [ServerPreset] {
        presets.filter(isBackfillable)
    }

    /// One preset's attempt. `detectFamily` is `EngineService.fetchModelFamily`;
    /// `save` is `EngineService.savePreset`.
    public func backfill(
        _ preset: ServerPreset,
        detectFamily: (String) async -> ModelFamilyInfo?,
        save: (ServerPreset) async throws -> Void
    ) async -> Outcome {
        guard Self.isBackfillable(preset) else {
            return Outcome(id: preset.id, name: preset.name, status: .skipped("Already expandable, or not a model-family problem"))
        }
        guard let spec = preset.effectiveModelSpec else {
            return Outcome(id: preset.id, name: preset.name, status: .failed("This preset declares no model or path to resolve"))
        }
        guard let detection = await detectFamily(spec) else {
            return Outcome(id: preset.id, name: preset.name, status: .failed("Could not reach the engine to classify '\(spec)'"))
        }
        guard let family = CheckpointFamilyResolver.resolve(
            family: detection.family, variant: detection.variant, loras: preset.loras
        ) else {
            return Outcome(id: preset.id, name: preset.name, status: .failed("The engine did not recognize '\(spec)' as a known checkpoint family"))
        }
        var updated = preset
        updated.checkpointFamily = family
        do {
            try await save(updated)
            return Outcome(id: preset.id, name: preset.name, status: .updated(family))
        } catch {
            return Outcome(id: preset.id, name: preset.name, status: .failed(error.localizedDescription))
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
