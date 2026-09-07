// PresetSamplingRules.swift — family-aware gating, editor state and
// Save-blocking validation for the sampler/schedule recipe fields the
// desktop edits (#419).
//
// One pure, testable decision surface shared by the preset editor
// (`ServerPresetEditor`), the Generate panel and the shared
// `SamplingAdvancedControls` subview, so "what does this family do with eta
// here?" is answered in exactly one place. Every answer is a
// `RecipeFieldStatus` from `SamplingRecipeCatalog` (ZImage), each pinned by
// test to the engine gate it mirrors — the desktop refuses to SAVE only what
// the engine would 400 on at render, labels what the engine accepts and
// ignores, and never invents a stricter rule of its own.
//
// Loaded values are never erased by a gate (review B2): a closed gate greys
// the control, offers Clear, and blocks Save with the engine's own wording.
// The only automatic reset is on a USER change of a sampler picker.
//
// Empty string / nil / the neutral sentinel = model default throughout; the
// editor never persists a default value into the store.

import Foundation
import ZImage

/// The gates a UI needs to enable/disable/label each knob for the current
/// family + sampler. Thin names over the catalog so call sites read as the
/// field they gate.
enum SamplingGate {
    typealias Status = SamplingRecipeCatalog.RecipeFieldStatus

    static func eta(modelFamily: String?, sampler: String) -> Status {
        SamplingRecipeCatalog.etaStatus(sampler: sampler.isEmpty ? nil : sampler, forModelFamily: modelFamily)
    }

    static func bongmath(modelFamily: String?, sampler: String) -> Status {
        SamplingRecipeCatalog.bongmathStatus(sampler: sampler.isEmpty ? nil : sampler, forModelFamily: modelFamily)
    }

    static func noise(modelFamily: String?) -> Status {
        SamplingRecipeCatalog.noiseRecipeStatus(forModelFamily: modelFamily)
    }

    static func shift(modelFamily: String?, sigmaSchedule: String) -> Status {
        SamplingRecipeCatalog.shiftStatus(sigmaSchedule: sigmaSchedule.isEmpty ? nil : sigmaSchedule, forModelFamily: modelFamily)
    }

    static func stage2(modelFamily: String?) -> Status {
        SamplingRecipeCatalog.stage2Status(forModelFamily: modelFamily)
    }

    static func vae(modelFamily: String?) -> Status {
        SamplingRecipeCatalog.vaeStatus(forModelFamily: modelFamily)
    }

    /// The Generate panel's one automatic reset, as a pure function: the
    /// user just picked `sampler` (the picker's own edit — never a
    /// programmatic write, see `SamplingRecipePicker.userEditBinding`). If
    /// the new sampler refuses eta/bongmath on the RESIDENT family, drop
    /// them. While an apply / model switch is in flight the resident family
    /// is the OLD one, so nothing is judged (`applyInFlight`).
    static func userSamplerChange(
        to sampler: String, modelFamily: String?, eta: Double, bongmath: Bool, applyInFlight: Bool
    ) -> (eta: Double, bongmath: Bool) {
        guard !applyInFlight else { return (eta, bongmath) }
        return (
            self.eta(modelFamily: modelFamily, sampler: sampler).isRefused ? 0 : eta,
            self.bongmath(modelFamily: modelFamily, sampler: sampler).isRefused ? false : bongmath
        )
    }

    /// Stage-2 `eta` is gated on the sampler the stage ACTUALLY runs — its
    /// own when named, else the render's (`WarmServer.stage2Gate`'s
    /// effective-sampler rule). The refusal wording is the engine's.
    static func stage2Eta(modelFamily: String?, stage2Sampler: String, stage1Sampler: String) -> Status {
        let effective = stage2Sampler.isEmpty ? stage1Sampler : stage2Sampler
        let status = eta(modelFamily: modelFamily, sampler: effective)
        guard status.isRefused else { return status }
        let name = SamplingRecipeCatalog.isRES4LYFSampler(effective) ? effective : (effective.isEmpty ? "euler" : effective)
        return .refused(
            "eta is RES4LYF's SDE (parity tier T2) and applies to the RES4LYF samplers only; stage 2 runs "
            + "'\(name)', which is not one of them. Send stage2.eta 0, or a stage2 sampler from "
            + SamplingRecipeCatalog.res4lyfSamplerList)
    }
}

/// The editable recipe with the neutral sentinels already collapsed to nil /
/// "" (= model default), plus the family to judge it against. What the
/// validator sees is exactly what `buildPreset()` writes.
struct PresetSamplingDraft: Equatable, Sendable {
    var modelFamily: String?
    var sampler: String = ""
    var sigmaSchedule: String = ""
    var shift: Double?
    var eta: Double?
    var bongmath: Bool?
    var noiseType: String?
    var noiseAlpha: Double?
    var implicitSteps: Int?
    var c2: Double?
    var projectorScale: Double?
    var vae: String?
    var stage2Enabled: Bool = false
    var stage2Steps: Int?
    var stage2Denoise: Double?
    var stage2Sampler: String = ""
    var stage2SigmaSchedule: String = ""
    var stage2Eta: Double?
    /// Preserved from the store, never written by a control: the engine
    /// 400s it as unimplemented, so the editor shows it with Clear.
    var stage2Bongmath: Bool?
}

/// The preset editor's dials as SwiftUI state — seeded from the stored
/// preset in `init(original:)`, written back verbatim by `write(into:)`.
/// Pure value type so the seed → edit → write cycle is unit-testable without
/// a view: in particular that NO gate ever erases a loaded value (review B2)
/// and that a preset saved with the engine unreachable is byte-for-byte what
/// was loaded.
struct PresetSamplingEditorState: Equatable, Sendable {
    var shift: Double?
    var eta: Double
    var bongmath: Bool
    var noiseType: String
    var noiseAlpha: Double
    var implicitSteps: Double
    var c2: Double
    var projectorScale: Double
    var vae: String
    var stage2Enabled: Bool
    var stage2StepsText: String
    var stage2DenoiseText: String
    var stage2Sampler: String
    var stage2SigmaSchedule: String
    var stage2Eta: Double
    var stage2Bongmath: Bool?

    init(original: ServerPreset) {
        shift = original.shift
        eta = original.eta ?? 0
        bongmath = original.bongmath ?? false
        noiseType = original.noiseType ?? "gaussian"
        noiseAlpha = original.noiseAlpha ?? 0
        implicitSteps = Double(original.implicitSteps ?? 0)
        c2 = original.c2 ?? 0.5
        projectorScale = original.projectorScale ?? 1.0
        vae = original.vae ?? ""
        stage2Enabled = original.stage2 != nil
        stage2StepsText = original.stage2?.steps.map(String.init) ?? ""
        stage2DenoiseText = original.stage2?.denoise.map { String(format: "%g", $0) } ?? ""
        stage2Sampler = original.stage2?.sampler ?? ""
        stage2SigmaSchedule = original.stage2?.sigmaSchedule ?? ""
        stage2Eta = original.stage2?.eta ?? 0
        stage2Bongmath = original.stage2?.bongmath
    }

    /// The draft the validator judges and `write(into:)` persists.
    func draft(modelFamily: String?, sampler: String, sigmaSchedule: String) -> PresetSamplingDraft {
        let trimmedVAE = vae.trimmingCharacters(in: .whitespacesAndNewlines)
        return PresetSamplingDraft(
            modelFamily: modelFamily,
            sampler: sampler,
            sigmaSchedule: sigmaSchedule,
            shift: shift,
            eta: eta == 0 ? nil : eta,
            bongmath: bongmath ? true : nil,
            noiseType: noiseType == "gaussian" ? nil : noiseType,
            noiseAlpha: noiseAlpha == 0 ? nil : noiseAlpha,
            implicitSteps: implicitSteps.rounded() == 0 ? nil : Int(implicitSteps.rounded()),
            c2: c2 == 0.5 ? nil : c2,
            projectorScale: projectorScale == 1.0 ? nil : projectorScale,
            vae: trimmedVAE.isEmpty ? nil : trimmedVAE,
            stage2Enabled: stage2Enabled,
            stage2Steps: Int(stage2StepsText.trimmingCharacters(in: .whitespaces)),
            stage2Denoise: Double(stage2DenoiseText.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: ",", with: ".")),
            stage2Sampler: stage2Sampler,
            stage2SigmaSchedule: stage2SigmaSchedule,
            stage2Eta: stage2Eta == 0 ? nil : stage2Eta,
            stage2Bongmath: stage2Enabled ? stage2Bongmath : nil
        )
    }

    /// Write the dials onto a preset — every field, nil where neutral, no
    /// family involved: what the user sees is what is saved. `stage2.bongmath`
    /// is preserved exactly as loaded (never set by a control; cleared only
    /// by the explicit Clear button) so a value the engine will refuse is
    /// surfaced, not silently dropped.
    func write(into p: inout ServerPreset) {
        let d = draft(modelFamily: nil, sampler: "", sigmaSchedule: "")
        p.shift = d.shift
        p.eta = d.eta
        p.bongmath = d.bongmath
        p.noiseType = d.noiseType
        p.noiseAlpha = d.noiseAlpha
        p.implicitSteps = d.implicitSteps
        p.c2 = d.c2
        p.projectorScale = d.projectorScale
        p.vae = d.vae
        p.stage2 = d.stage2Enabled
            ? ServerPresetStage(
                sampler: d.stage2Sampler.isEmpty ? nil : d.stage2Sampler,
                sigmaSchedule: d.stage2SigmaSchedule.isEmpty ? nil : d.stage2SigmaSchedule,
                steps: d.stage2Steps,
                denoise: d.stage2Denoise,
                eta: d.stage2Eta,
                bongmath: d.stage2Bongmath)
            : nil
    }

    /// The ONE automatic reset (review B2): the user just picked a stage-1
    /// sampler. If the new sampler refuses eta/bongmath on this family, drop
    /// them — the user made the change that closed the gate. Never called
    /// on load or on a family answer arriving.
    mutating func samplerDidChange(to sampler: String, modelFamily: String?) {
        if SamplingGate.eta(modelFamily: modelFamily, sampler: sampler).isRefused { eta = 0 }
        if SamplingGate.bongmath(modelFamily: modelFamily, sampler: sampler).isRefused { bongmath = false }
    }

    /// Same rule for the stage-2 picker.
    mutating func stage2SamplerDidChange(to stage2Sampler: String, stage1Sampler: String, modelFamily: String?) {
        if SamplingGate.stage2Eta(modelFamily: modelFamily, stage2Sampler: stage2Sampler, stage1Sampler: stage1Sampler).isRefused {
            stage2Eta = 0
        }
    }
}

enum PresetSamplingValidator {
    /// The Krea 2 `c2` pole: `res_3s`'s tableau divides by `2 − 3·c2`.
    static let c2Pole = 2.0 / 3.0

    /// nil when the engine would accept this recipe on the draft's family;
    /// otherwise the sentence the editor shows and blocks Save with — the
    /// engine's own wording wherever a gate has one. Only `.refused` statuses
    /// and out-of-range values block; `.inert` fields are labelled, not
    /// refused (the engine accepts and ignores them). The family-pair check
    /// runs first (it is what `SamplingRecipePicker` already reports).
    static func validationError(_ d: PresetSamplingDraft) -> String? {
        let familyName = SamplingRecipeCatalog.canonicalFamily(d.modelFamily) ?? "this model"

        // Sampler / schedule pair (tableau samplers and krea2/bong_tangent
        // schedules are krea2-only; chroma's heun+beta pairing rule).
        if !d.sampler.isEmpty || !d.sigmaSchedule.isEmpty {
            if !SamplingRecipeCatalog.supports(
                sampler: d.sampler.isEmpty ? nil : d.sampler,
                sigmaSchedule: d.sigmaSchedule.isEmpty ? nil : d.sigmaSchedule,
                forModelFamily: d.modelFamily
            ) {
                return "The selected sampler/scheduler pair is not supported by \(familyName)."
            }
        }

        // shift — a positive number the family reads and the schedule honours.
        if let shift = d.shift {
            guard shift.isFinite, shift > 0 else {
                return "shift must be a positive number (got \(format(shift))); omit it for the resolution-dependent default"
            }
            if case .refused(let why) = SamplingGate.shift(modelFamily: d.modelFamily, sigmaSchedule: d.sigmaSchedule) {
                return why
            }
        }

        // eta / bongmath — the Krea 2 tier gates.
        if let eta = d.eta, eta != 0 {
            guard eta.isFinite, eta >= 0 else { return "eta must be a finite number >= 0 (got \(format(eta)))" }
            if case .refused(let why) = SamplingGate.eta(modelFamily: d.modelFamily, sampler: d.sampler) { return why }
        }
        if d.bongmath == true, case .refused(let why) = SamplingGate.bongmath(modelFamily: d.modelFamily, sampler: d.sampler) {
            return why
        }

        // RES4LYF noise / implicit-RK / c2 / projector — range checks only
        // (`PresetStore.validateRecipeFields`); no family gate exists.
        if let noiseType = d.noiseType, !noiseType.isEmpty,
           !["gaussian", "fractal", "pyramid"].contains(noiseType) {
            return "noise_type must be one of gaussian, fractal, pyramid (got \(noiseType))"
        }
        if let noiseAlpha = d.noiseAlpha, !noiseAlpha.isFinite {
            return "noise_alpha must be a finite number"
        }
        if let implicitSteps = d.implicitSteps, !(0...8).contains(implicitSteps) {
            return "implicit_steps must be an integer in 0...8 (got \(implicitSteps)); 0 is the explicit render"
        }
        if let c2 = d.c2, !(c2.isFinite && c2 > 0 && c2 <= 1 && abs(c2 - c2Pole) >= 1e-6) {
            return "c2 must be a finite number in (0, 1] other than 2/3 (got \(format(c2))); 0.5 is the default"
        }
        if let projectorScale = d.projectorScale, !(projectorScale.isFinite && (0.0...3.0).contains(projectorScale)) {
            return "projector_scale must be a finite number in 0.0...3.0 (got \(format(projectorScale))); 1.0 is neutral"
        }

        // vae — a Krea 2 request field (vaeGate).
        if let vae = d.vae, !vae.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           case .refused(let why) = SamplingGate.vae(modelFamily: d.modelFamily) {
            return why
        }

        // stage 2 — krea2 only (stage2Gate); needs steps AND denoise; own
        // pair; eta on the EFFECTIVE sampler with the EFFECTIVE eta
        // (`stage2.eta ?? eta`, WarmServer.stage2Gate); bongmath unimplemented.
        if d.stage2Enabled {
            if case .refused(let why) = SamplingGate.stage2(modelFamily: d.modelFamily) { return why }
            guard let steps = d.stage2Steps, steps > 0 else {
                return "stage2.steps must be positive — a detail pass needs a step count"
            }
            guard let denoise = d.stage2Denoise, denoise.isFinite, denoise > 0, denoise <= 1 else {
                return "stage2.denoise is the fraction of the schedule the stage runs and must be in (0, 1]"
            }
            if !d.stage2Sampler.isEmpty || !d.stage2SigmaSchedule.isEmpty {
                if !SamplingRecipeCatalog.supports(
                    sampler: d.stage2Sampler.isEmpty ? nil : d.stage2Sampler,
                    sigmaSchedule: d.stage2SigmaSchedule.isEmpty ? nil : d.stage2SigmaSchedule,
                    forModelFamily: d.modelFamily
                ) {
                    return "The stage 2 sampler/scheduler pair is not supported by \(familyName)."
                }
            }
            if d.stage2Bongmath == true {
                return "'stage2.bongmath' = 'true' is not supported: bongmath is parity tier T3 (WP-E16) and is not implemented yet; omit it or send false"
            }
            let effectiveEta = d.stage2Eta ?? d.eta ?? 0
            if effectiveEta != 0 {
                guard effectiveEta.isFinite, effectiveEta >= 0 else { return "stage2.eta must be a finite number >= 0" }
                if case .refused(let why) = SamplingGate.stage2Eta(
                    modelFamily: d.modelFamily, stage2Sampler: d.stage2Sampler, stage1Sampler: d.sampler
                ) {
                    return why
                }
            }
        }
        return nil
    }

    private static func format(_ value: Double) -> String {
        String(format: "%g", value)
    }
}
