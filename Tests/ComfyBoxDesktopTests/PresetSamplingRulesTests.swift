// PresetSamplingRulesTests.swift — #419: the family-aware gates and the
// Save-blocking validator behind the preset editor's Sampling / Detail pass
// sections, exercised at REAL recipe shapes (the live krea-kira presets),
// not convenient ones.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("SamplingGate")
struct SamplingGateTests {
    @Test("eta/bongmath: krea2 AND a RES4LYF sampler; euler (model default) never")
    func sdeGate() {
        #expect(SamplingGate.sdeAllowed(modelFamily: "krea2", sampler: "res_2s"))
        #expect(SamplingGate.sdeAllowed(modelFamily: "krea2-raw", sampler: "res_3s"))
        #expect(SamplingGate.sdeAllowed(modelFamily: "krea2", sampler: "exponential/res_2s"))
        #expect(!SamplingGate.sdeAllowed(modelFamily: "krea2", sampler: ""))
        #expect(!SamplingGate.sdeAllowed(modelFamily: "krea2", sampler: "euler"))
        #expect(!SamplingGate.sdeAllowed(modelFamily: "krea2", sampler: "dpmpp_2m"))
        // res_2s is the one RES4LYF sampler flux1 also runs — but the tier
        // gates (eta/bongmath) are krea2's, so it is still refused there.
        #expect(!SamplingGate.sdeAllowed(modelFamily: "flux1", sampler: "res_2s"))
        #expect(!SamplingGate.sdeAllowed(modelFamily: "Tongyi-MAI/Z-Image-Turbo", sampler: "res_2s"))
        #expect(!SamplingGate.sdeAllowed(modelFamily: "chroma", sampler: "res_2s"))
    }

    @Test("unknown family is permissive on the family half, never on the sampler half")
    func unknownFamily() {
        #expect(SamplingGate.sdeAllowed(modelFamily: nil, sampler: "res_2s"))
        #expect(!SamplingGate.sdeAllowed(modelFamily: nil, sampler: "euler"))
        #expect(SamplingGate.stage2Allowed(modelFamily: nil))
        #expect(SamplingGate.shiftAllowed(modelFamily: "/Models/mystery.safetensors"))
    }

    @Test("stage-2 eta follows the sampler the stage actually runs (its own, else the render's)")
    func stage2EffectiveSampler() {
        #expect(SamplingGate.stage2SDEAllowed(modelFamily: "krea2", stage2Sampler: "", stage1Sampler: "res_2s"))
        #expect(!SamplingGate.stage2SDEAllowed(modelFamily: "krea2", stage2Sampler: "", stage1Sampler: "euler"))
        #expect(!SamplingGate.stage2SDEAllowed(modelFamily: "krea2", stage2Sampler: "", stage1Sampler: ""))
        #expect(SamplingGate.stage2SDEAllowed(modelFamily: "krea2", stage2Sampler: "deis_3m", stage1Sampler: "euler"))
        #expect(!SamplingGate.stage2SDEAllowed(modelFamily: "krea2", stage2Sampler: "euler", stage1Sampler: "res_2s"))
    }

    @Test("family-only gates: krea2 for noise/stage2/vae; krea2 + flux1 for shift")
    func familyGates() {
        #expect(SamplingGate.noiseAllowed(modelFamily: "krea2"))
        #expect(!SamplingGate.noiseAllowed(modelFamily: "flux1"))
        #expect(SamplingGate.stage2Allowed(modelFamily: "krea2"))
        #expect(!SamplingGate.stage2Allowed(modelFamily: "flux1"))
        #expect(SamplingGate.vaeAllowed(modelFamily: "krea2"))
        #expect(!SamplingGate.vaeAllowed(modelFamily: "flux1"))
        #expect(SamplingGate.shiftAllowed(modelFamily: "krea2"))
        #expect(SamplingGate.shiftAllowed(modelFamily: "flux1"))
        #expect(!SamplingGate.shiftAllowed(modelFamily: "chroma"))
        #expect(!SamplingGate.shiftAllowed(modelFamily: "flux2"))
    }
}

@Suite("PresetSamplingValidator")
struct PresetSamplingValidatorTests {
    /// The live `krea-kira-sfw` shape: euler on the native schedule with the
    /// reference mu, no SDE.
    private static var kreaKiraSFW: PresetSamplingDraft {
        PresetSamplingDraft(modelFamily: "krea2-raw", sampler: "euler", sigmaSchedule: "", shift: 1.15)
    }

    private func error(_ draft: PresetSamplingDraft) -> String? {
        PresetSamplingValidator.validationError(draft)
    }

    @Test("krea-kira-sfw: euler + shift 1.15, no eta — valid")
    func kreaKiraSFWIsValid() {
        #expect(error(Self.kreaKiraSFW) == nil)
    }

    @Test("euler + eta 0.5 on krea2 — invalid (the engine 400s eta on a non-RES4LYF sampler)")
    func eulerWithEtaIsInvalid() {
        var d = Self.kreaKiraSFW
        d.eta = 0.5
        #expect(error(d)?.contains("RES4LYF") == true)
        // Model default (euler) is the same refusal.
        d.sampler = ""
        #expect(error(d) != nil)
    }

    @Test("res_2s + eta 0.5 on krea2 — valid (the Clownshark recipe)")
    func res2sWithEtaIsValid() {
        var d = Self.kreaKiraSFW
        d.sampler = "res_2s"; d.sigmaSchedule = "beta57"; d.eta = 0.5; d.bongmath = true
        #expect(error(d) == nil)
    }

    @Test("res_3s on flux1 — invalid (N-row tableaus are krea2-only)")
    func res3sOnFlux1IsInvalid() {
        let d = PresetSamplingDraft(modelFamily: "Tongyi-MAI/Z-Image-Turbo", sampler: "res_3s")
        #expect(error(d) != nil)
    }

    @Test("res_2s runs on flux1, but eta/bongmath there are still refused")
    func res2sOnFlux1WithoutSDEIsValidWithSDEIsNot() {
        var d = PresetSamplingDraft(modelFamily: "flux1", sampler: "res_2s", sigmaSchedule: "beta")
        #expect(error(d) == nil)
        d.eta = 0.3
        #expect(error(d) != nil)
        d.eta = nil; d.bongmath = true
        #expect(error(d) != nil)
    }

    @Test("bongmath on euler — invalid even with no eta")
    func bongmathOnEulerIsInvalid() {
        var d = Self.kreaKiraSFW
        d.bongmath = true
        #expect(error(d)?.contains("Bongmath") == true)
    }

    @Test("eta 0 / bongmath false are neutral, never a refusal")
    func neutralSDEIsFine() {
        var d = Self.kreaKiraSFW
        d.eta = 0; d.bongmath = false
        #expect(error(d) == nil)
    }

    @Test("shift: positive only, krea2/flux1 only, and honoured by the schedule on flux1")
    func shiftRules() {
        var d = Self.kreaKiraSFW
        d.shift = 0
        #expect(error(d)?.contains("positive") == true)
        d.shift = -1
        #expect(error(d) != nil)
        d.shift = 1.15
        #expect(error(d) == nil)

        // chroma runs a fixed schedule — shift refused by family.
        #expect(error(PresetSamplingDraft(modelFamily: "chroma", shift: 3.0)) != nil)

        // flux1 honours it on flow/beta/karras…; a mu-defined grid drops it.
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sigmaSchedule: "beta", shift: 1.5)) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sigmaSchedule: "", shift: 1.5)) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sigmaSchedule: "krea2", shift: 1.5)) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sigmaSchedule: "bong_tangent", shift: 1.5)) != nil)
        // The same schedules ARE fine with a shift on krea2 (shift is mu there).
        #expect(error(PresetSamplingDraft(modelFamily: "krea2", sigmaSchedule: "bong_tangent", shift: 1.15)) == nil)
    }

    @Test("krea2 / bong_tangent schedules outside krea2 — invalid")
    func krea2SchedulesOutsideKrea2() {
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sigmaSchedule: "krea2")) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sigmaSchedule: "bong_tangent")) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: "krea2", sigmaSchedule: "bong_tangent")) == nil)
    }

    @Test("noise / implicit / c2 / projector: ranges, and krea2 only")
    func noiseRecipeRules() {
        var d = PresetSamplingDraft(modelFamily: "krea2", sampler: "res_2s")
        d.projectorScale = 3.5
        #expect(error(d)?.contains("Projector") == true)
        d.projectorScale = 2.0
        #expect(error(d) == nil)
        d.implicitSteps = 9
        #expect(error(d)?.contains("Implicit") == true)
        d.implicitSteps = 8
        #expect(error(d) == nil)
        d.c2 = 2.0 / 3.0
        #expect(error(d)?.contains("C2") == true)
        d.c2 = 0
        #expect(error(d) != nil)
        d.c2 = 1.0
        #expect(error(d) == nil)
        d.noiseType = "perlin"
        #expect(error(d) != nil)
        d.noiseType = "fractal"; d.noiseAlpha = 0.5
        #expect(error(d) == nil)

        // The same, non-neutral, on Z-Image — refused by family.
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", noiseType: "fractal")) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", noiseAlpha: 0.5)) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", implicitSteps: 2)) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", c2: 0.6)) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", projectorScale: 1.4)) != nil)
        // Neutral values are not a declaration.
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", c2: 0.5, projectorScale: 1.0)) == nil)
    }

    @Test("vae override is krea2 only")
    func vaeRule() {
        #expect(error(PresetSamplingDraft(modelFamily: "krea2", vae: "/vae/Wan2_1_VAE_fp32.safetensors")) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", vae: "/vae/Wan2_1_VAE_fp32.safetensors")) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", vae: "  ")) == nil)
    }

    @Test("stage 2: krea2 only; needs steps AND denoise in range; own pair; eta on the effective sampler")
    func stage2Rules() {
        var d = PresetSamplingDraft(modelFamily: "krea2", sampler: "res_2s", stage2Enabled: true)
        #expect(error(d)?.contains("step") == true)          // no steps
        d.stage2Steps = 0
        #expect(error(d) != nil)
        d.stage2Steps = 2
        #expect(error(d)?.contains("denoise") == true)       // no denoise
        d.stage2Denoise = 1.5
        #expect(error(d) != nil)
        d.stage2Denoise = 0
        #expect(error(d) != nil)
        d.stage2Denoise = 0.2
        #expect(error(d) == nil)

        // Own pair on the family.
        d.stage2Sampler = "res_3s"; d.stage2SigmaSchedule = "karras"
        #expect(error(d) == nil)
        d.stage2Sampler = "uni_pc"
        #expect(error(d) != nil)

        // eta: inherits the render's res_2s when the stage names none.
        d.stage2Sampler = ""; d.stage2SigmaSchedule = ""; d.stage2Eta = 0.5
        #expect(error(d) == nil)
        d.stage2Sampler = "euler"
        #expect(error(d)?.contains("Stage 2 eta") == true)
        d.stage2Sampler = ""; d.sampler = "euler"
        #expect(error(d) != nil)

        // Outside krea2 the whole stage is refused.
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", stage2Enabled: true, stage2Steps: 2, stage2Denoise: 0.2)) != nil)
        // Disabled = not declared, whatever the stale fields hold.
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", stage2Enabled: false, stage2Steps: 0, stage2Denoise: 9)) == nil)
    }

    @Test("unknown family: only the sampler half of the SDE rule applies")
    func unknownFamily() {
        #expect(error(PresetSamplingDraft(modelFamily: nil, sampler: "res_2s", eta: 0.5)) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: nil, sampler: "euler", eta: 0.5)) != nil)
        #expect(error(PresetSamplingDraft(modelFamily: nil, shift: 1.15, stage2Enabled: true, stage2Steps: 2, stage2Denoise: 0.2)) == nil)
    }
}
