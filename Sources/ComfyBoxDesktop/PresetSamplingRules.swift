// PresetSamplingRules.swift — family-aware gating + validation for the
// sampler/schedule recipe fields the desktop edits (#419).
//
// One pure, testable decision surface shared by the preset editor
// (`ServerPresetEditor`), the Generate panel and the shared
// `SamplingAdvancedControls` subview, so "may eta be set here?" is answered
// in exactly one place. Every rule mirrors an engine gate by name (see
// `SamplingRecipeCatalog`'s family helpers in ZImage) — the desktop refuses
// to SAVE what the engine would 400 on at render, rather than letting a
// preset carry a recipe that only fails once Bree renders it.
//
// Empty string / nil = model default throughout; the editor never persists a
// default value into the store.

import Foundation
import ZImage

/// The gates a UI needs to enable/disable each knob for the current family +
/// sampler. Pure functions over strings so previews and tests need no engine.
enum SamplingGate {
    /// `eta` / `bongmath` are editable only when the family runs the RES4LYF
    /// tier gates (krea2) AND the selected sampler is a RES4LYF port —
    /// `WarmServer.validateKrea2TierGates` refuses either on `euler`.
    static func sdeAllowed(modelFamily: String?, sampler: String) -> Bool {
        SamplingRecipeCatalog.supportsRES4LYFTiers(forModelFamily: modelFamily)
            && SamplingRecipeCatalog.isRES4LYFSampler(sampler)
    }

    /// Stage-2 `eta` follows the same rule on the sampler the stage will
    /// ACTUALLY run: its own when named, else the render's (the engine's
    /// effective-sampler rule in `validateStage2`).
    static func stage2SDEAllowed(modelFamily: String?, stage2Sampler: String, stage1Sampler: String) -> Bool {
        let effective = stage2Sampler.isEmpty ? stage1Sampler : stage2Sampler
        return sdeAllowed(modelFamily: modelFamily, sampler: effective)
    }

    /// The RES4LYF noise / implicit-RK / c2 / projector-scale dials.
    static func noiseAllowed(modelFamily: String?) -> Bool {
        SamplingRecipeCatalog.supportsRES4LYFNoise(forModelFamily: modelFamily)
    }

    static func shiftAllowed(modelFamily: String?) -> Bool {
        SamplingRecipeCatalog.acceptsShift(forModelFamily: modelFamily)
    }

    static func stage2Allowed(modelFamily: String?) -> Bool {
        SamplingRecipeCatalog.supportsStage2(forModelFamily: modelFamily)
    }

    static func vaeAllowed(modelFamily: String?) -> Bool {
        SamplingRecipeCatalog.supportsVAEOverride(forModelFamily: modelFamily)
    }

    /// The one-line reason shown under a disabled eta/bongmath control.
    static func sdeDisabledReason(modelFamily: String?, sampler: String) -> String {
        if !SamplingRecipeCatalog.supportsRES4LYFTiers(forModelFamily: modelFamily) {
            let family = SamplingRecipeCatalog.canonicalFamily(modelFamily) ?? "this model"
            return "Eta and bongmath are Krea 2 RES4LYF settings; \(family) does not read them."
        }
        return "Eta and bongmath need a RES4LYF sampler (res_2s, res_3s, ralston_*, deis_*m)."
    }
}

/// The editable recipe as the preset editor holds it — Double/Bool/String
/// with nil / "" meaning "model default". `Sendable` + `Equatable` so tests
/// can build one per rule and compare.
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
}

enum PresetSamplingValidator {
    /// The Krea 2 `c2` pole: `res_3s`'s tableau divides by `2 − 3·c2`.
    static let c2Pole = 2.0 / 3.0

    /// nil when the engine would accept this recipe on the draft's family;
    /// otherwise the sentence the editor shows and blocks Save with. The
    /// family-pair check runs first (it is what `SamplingRecipePicker`
    /// already reports), then every field the editor added in #419.
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
                return "Shift must be a positive number (leave it empty for the model default)."
            }
            guard SamplingGate.shiftAllowed(modelFamily: d.modelFamily) else {
                return "Shift is a krea2 / Z-Image schedule field; \(familyName) does not honour it."
            }
            guard SamplingRecipeCatalog.shiftIsHonoured(sigmaSchedule: d.sigmaSchedule, forModelFamily: d.modelFamily) else {
                return "Shift is not read by the '\(d.sigmaSchedule)' schedule on \(familyName) — drop it, or choose a schedule that honours it."
            }
        }

        // eta / bongmath — krea2 + RES4LYF sampler only.
        if let eta = d.eta, eta != 0 {
            guard eta.isFinite, eta >= 0 else { return "Eta must be a finite number ≥ 0." }
            guard SamplingGate.sdeAllowed(modelFamily: d.modelFamily, sampler: d.sampler) else {
                return "Eta \(format(eta)) needs a Krea 2 RES4LYF sampler; '\(d.sampler.isEmpty ? "model default (euler)" : d.sampler)' on \(familyName) would be refused."
            }
        }
        if d.bongmath == true, !SamplingGate.sdeAllowed(modelFamily: d.modelFamily, sampler: d.sampler) {
            return "Bongmath needs a Krea 2 RES4LYF sampler; '\(d.sampler.isEmpty ? "model default (euler)" : d.sampler)' on \(familyName) would be refused."
        }

        // RES4LYF noise / implicit-RK / c2 / projector — krea2 only, in range.
        let noiseAllowed = SamplingGate.noiseAllowed(modelFamily: d.modelFamily)
        if let noiseType = d.noiseType, !noiseType.isEmpty {
            guard ["gaussian", "fractal", "pyramid"].contains(noiseType) else {
                return "Noise type must be gaussian, fractal or pyramid."
            }
            guard noiseAllowed else { return "Noise type is a Krea 2 setting; \(familyName) does not read it." }
        }
        if let noiseAlpha = d.noiseAlpha, noiseAlpha != 0 {
            guard noiseAlpha.isFinite else { return "Noise alpha must be a finite number." }
            guard noiseAllowed else { return "Noise alpha is a Krea 2 setting; \(familyName) does not read it." }
        }
        if let implicitSteps = d.implicitSteps, implicitSteps != 0 {
            guard (0...8).contains(implicitSteps) else { return "Implicit steps must be an integer in 0…8." }
            guard noiseAllowed else { return "Implicit steps is a Krea 2 setting; \(familyName) does not read it." }
        }
        if let c2 = d.c2 {
            guard c2.isFinite, c2 > 0, c2 <= 1, abs(c2 - c2Pole) >= 1e-6 else {
                return "C2 must be in (0, 1] and not 2/3 (the res_3s tableau pole)."
            }
            if c2 != 0.5, !noiseAllowed { return "C2 is a Krea 2 setting; \(familyName) does not read it." }
        }
        if let projectorScale = d.projectorScale {
            guard projectorScale.isFinite, (0.0...3.0).contains(projectorScale) else {
                return "Projector scale must be in 0…3 (1.0 is neutral)."
            }
            if projectorScale != 1.0, !noiseAllowed { return "Projector scale is a Krea 2 setting; \(familyName) does not read it." }
        }

        // vae — krea2 only, non-blank when present.
        if let vae = d.vae, !vae.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           !SamplingGate.vaeAllowed(modelFamily: d.modelFamily) {
            return "A VAE override is a Krea 2 setting; \(familyName) does not read it."
        }

        // stage 2 — krea2 only; needs steps AND denoise; own pair + eta rule.
        if d.stage2Enabled {
            guard SamplingGate.stage2Allowed(modelFamily: d.modelFamily) else {
                return "A detail pass (stage 2) is a Krea 2 mechanism; \(familyName) has no such seam."
            }
            guard let steps = d.stage2Steps, steps > 0 else {
                return "Stage 2 needs a positive step count."
            }
            guard let denoise = d.stage2Denoise, denoise.isFinite, denoise > 0, denoise <= 1 else {
                return "Stage 2 needs a denoise fraction in (0, 1]."
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
            if let eta = d.stage2Eta, eta != 0 {
                guard eta.isFinite, eta >= 0 else { return "Stage 2 eta must be a finite number ≥ 0." }
                guard SamplingGate.stage2SDEAllowed(
                    modelFamily: d.modelFamily, stage2Sampler: d.stage2Sampler, stage1Sampler: d.sampler
                ) else {
                    let effective = d.stage2Sampler.isEmpty ? d.sampler : d.stage2Sampler
                    return "Stage 2 eta needs a RES4LYF sampler; stage 2 would run '\(effective.isEmpty ? "model default (euler)" : effective)'."
                }
            }
        }
        return nil
    }

    private static func format(_ value: Double) -> String {
        String(format: "%g", value)
    }
}
