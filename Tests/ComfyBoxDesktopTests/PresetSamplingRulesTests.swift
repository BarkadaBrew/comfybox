// PresetSamplingRulesTests.swift — #419: the family-aware gates, the
// editor's seed → write cycle, and the Save-blocking validator behind the
// preset editor's Sampling / Detail pass sections, exercised at REAL recipe
// shapes (the live krea-kira presets), not convenient ones.

import Testing
import Foundation
import ZImage
@testable import ComfyBoxDesktop

@Suite("SamplingGate")
struct SamplingGateTests {
    @Test("eta on krea2: honoured on a RES4LYF sampler, refused on euler (model default) and every other")
    func etaOnKrea2() {
        #expect(SamplingGate.eta(modelFamily: "krea2", sampler: "res_2s").isHonoured)
        #expect(SamplingGate.eta(modelFamily: "krea2", sampler: "heun_3s").isHonoured)
        #expect(SamplingGate.eta(modelFamily: "krea2", sampler: "exponential/res_2s").isHonoured)
        #expect(SamplingGate.eta(modelFamily: "krea2", sampler: "").isRefused)
        #expect(SamplingGate.eta(modelFamily: "krea2", sampler: "euler").isRefused)
        #expect(SamplingGate.eta(modelFamily: "krea2", sampler: "dpmpp_2m").isRefused)
        #expect(SamplingGate.eta(modelFamily: "krea2", sampler: "ddim").isRefused)
    }

    /// Review B1: eta is NOT krea2-only. Z-Image has no eta gate — the value
    /// is forwarded, read by DDIM / DPM++ 2S-A (their own η) and ignored by
    /// every other sampler (inert, labelled, never refused).
    @Test("eta on flux1: honoured on ddim / dpmpp-2s-a, inert (never refused) elsewhere")
    func etaOnFlux1() {
        #expect(SamplingGate.eta(modelFamily: "flux1", sampler: "ddim").isHonoured)
        #expect(SamplingGate.eta(modelFamily: "Tongyi-MAI/Z-Image-Turbo", sampler: "dpmpp-2s-a").isHonoured)
        #expect(SamplingGate.eta(modelFamily: "flux1", sampler: "dpmpp_2s_ancestral").isHonoured)
        let res2s = SamplingGate.eta(modelFamily: "flux1", sampler: "res_2s")
        #expect(!res2s.isRefused && !res2s.isHonoured)
        #expect(res2s.note?.contains("ignored") == true)
        let euler = SamplingGate.eta(modelFamily: "flux1", sampler: "euler")
        #expect(!euler.isRefused && !euler.isHonoured)
    }

    @Test("bongmath: krea2 + RES4LYF only; inert on the other families")
    func bongmath() {
        #expect(SamplingGate.bongmath(modelFamily: "krea2", sampler: "res_2s").isHonoured)
        #expect(SamplingGate.bongmath(modelFamily: "krea2", sampler: "euler").isRefused)
        #expect(SamplingGate.bongmath(modelFamily: "krea2", sampler: "").isRefused)
        let flux1 = SamplingGate.bongmath(modelFamily: "flux1", sampler: "res_2s")
        #expect(!flux1.isRefused && !flux1.isHonoured)
        #expect(!SamplingGate.bongmath(modelFamily: "chroma", sampler: "res_2s").isRefused)
    }

    @Test("unknown family is permissive everywhere (pending / offline / no answer)")
    func unknownFamily() {
        #expect(SamplingGate.eta(modelFamily: nil, sampler: "euler").isHonoured)
        #expect(SamplingGate.bongmath(modelFamily: nil, sampler: "").isHonoured)
        #expect(SamplingGate.stage2(modelFamily: nil).isHonoured)
        #expect(SamplingGate.vae(modelFamily: nil).isHonoured)
        #expect(SamplingGate.shift(modelFamily: "/Models/mystery.safetensors", sigmaSchedule: "krea2").isHonoured)
    }

    /// Review B4: the stage-2 eta gate keys on the EFFECTIVE sampler (the
    /// stage's own, else the render's), with the engine's wording.
    @Test("stage-2 eta follows the sampler the stage actually runs")
    func stage2EffectiveSampler() {
        #expect(SamplingGate.stage2Eta(modelFamily: "krea2", stage2Sampler: "", stage1Sampler: "res_2s").isHonoured)
        #expect(SamplingGate.stage2Eta(modelFamily: "krea2", stage2Sampler: "", stage1Sampler: "euler").isRefused)
        #expect(SamplingGate.stage2Eta(modelFamily: "krea2", stage2Sampler: "", stage1Sampler: "").isRefused)
        #expect(SamplingGate.stage2Eta(modelFamily: "krea2", stage2Sampler: "deis_3m", stage1Sampler: "euler").isHonoured)
        let refused = SamplingGate.stage2Eta(modelFamily: "krea2", stage2Sampler: "euler", stage1Sampler: "res_2s")
        #expect(refused.isRefused)
        #expect(refused.note?.contains("stage 2 runs 'euler'") == true)
        #expect(refused.note?.contains("heun_2s / heun_3s") == true)
    }

    @Test("family-only gates: krea2 for stage2/vae (refused elsewhere); noise inert elsewhere; shift krea2 + flux1")
    func familyGates() {
        #expect(SamplingGate.noise(modelFamily: "krea2").isHonoured)
        #expect(!SamplingGate.noise(modelFamily: "flux1").isRefused)
        #expect(!SamplingGate.noise(modelFamily: "flux1").isHonoured)
        #expect(SamplingGate.stage2(modelFamily: "krea2").isHonoured)
        #expect(SamplingGate.stage2(modelFamily: "flux1").isRefused)
        #expect(SamplingGate.vae(modelFamily: "krea2").isHonoured)
        #expect(SamplingGate.vae(modelFamily: "flux1").isRefused)
        #expect(SamplingGate.shift(modelFamily: "krea2", sigmaSchedule: "bong_tangent").isHonoured)
        #expect(SamplingGate.shift(modelFamily: "flux1", sigmaSchedule: "beta").isHonoured)
        #expect(SamplingGate.shift(modelFamily: "flux1", sigmaSchedule: "krea2").isRefused)
        #expect(SamplingGate.shift(modelFamily: "flux1", sigmaSchedule: "bong_tangent").isRefused)
        #expect(SamplingGate.shift(modelFamily: "chroma", sigmaSchedule: "").isRefused)
        #expect(SamplingGate.shift(modelFamily: "flux2", sigmaSchedule: "").isRefused)
    }
}

@Suite("PresetSamplingValidator")
struct PresetSamplingValidatorTests {
    /// The live `krea-kira-sfw` shape: euler on the native schedule with the
    /// reference mu, no SDE.
    private static var kreaKiraSFW: PresetSamplingDraft {
        PresetSamplingDraft(modelFamily: "krea2", sampler: "euler", sigmaSchedule: "", shift: 1.15)
    }

    private func error(_ draft: PresetSamplingDraft) -> String? {
        PresetSamplingValidator.validationError(draft)
    }

    @Test("krea-kira-sfw: euler + shift 1.15, no eta — valid")
    func kreaKiraSFWIsValid() {
        #expect(error(Self.kreaKiraSFW) == nil)
    }

    @Test("euler + eta 0.5 on krea2 — refused with the engine's tier-gate wording")
    func eulerWithEtaIsInvalid() {
        var d = Self.kreaKiraSFW
        d.eta = 0.5
        let message = error(d)
        #expect(message?.contains("parity tier T2") == true)
        #expect(message?.contains("'euler'") == true)
        // Model default (euler) is the same refusal.
        d.sampler = ""
        #expect(error(d) != nil)
    }

    @Test("res_2s + eta 0.5 (+ bongmath) on krea2 — valid (the Clownshark recipe)")
    func res2sWithEtaIsValid() {
        var d = Self.kreaKiraSFW
        d.sampler = "res_2s"; d.sigmaSchedule = "beta57"; d.eta = 0.5; d.bongmath = true
        #expect(error(d) == nil)
    }

    @Test("res_3s on flux1 — invalid (N-row tableaus are krea2-only)")
    func res3sOnFlux1IsInvalid() {
        let d = PresetSamplingDraft(modelFamily: "flux1", sampler: "res_3s")
        #expect(error(d) != nil)
    }

    /// Review B1: on Z-Image there is no eta gate. res_2s + eta is accepted
    /// and inert (labelled, not refused); ddim + eta is DDIM's own η.
    @Test("eta on flux1 is never refused: res_2s inert, ddim honoured; bongmath inert")
    func etaOnFlux1IsNeverRefused() {
        var d = PresetSamplingDraft(modelFamily: "flux1", sampler: "res_2s", sigmaSchedule: "beta")
        #expect(error(d) == nil)
        d.eta = 0.3
        #expect(error(d) == nil)
        d.bongmath = true
        #expect(error(d) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sampler: "ddim", eta: 0.5)) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sampler: "euler", eta: 0.5)) == nil)
    }

    @Test("bongmath on euler (krea2) — refused even with no eta")
    func bongmathOnEulerIsInvalid() {
        var d = Self.kreaKiraSFW
        d.bongmath = true
        #expect(error(d)?.contains("parity tier T3") == true)
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

        // chroma runs a fixed schedule — shift refused by family (validateShift).
        #expect(error(PresetSamplingDraft(modelFamily: "chroma", shift: 3.0))?.contains("not honoured by model family 'chroma'") == true)

        // flux1 honours it on flow/beta/karras…; a mu-defined grid drops it (#154).
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sigmaSchedule: "beta", shift: 1.5)) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", sigmaSchedule: "", shift: 3.0)) == nil)
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

    /// Ranges are `PresetStore.validateRecipeFields`'s; there is no family
    /// gate for these fields anywhere in the engine, so on Z-Image they are
    /// accepted (inert), never refused.
    @Test("noise / implicit / c2 / projector: engine ranges; never refused by family")
    func noiseRecipeRules() {
        var d = PresetSamplingDraft(modelFamily: "krea2", sampler: "res_2s")
        d.projectorScale = 3.5
        #expect(error(d)?.contains("projector_scale") == true)
        d.projectorScale = 2.0
        #expect(error(d) == nil)
        d.implicitSteps = 9
        #expect(error(d)?.contains("implicit_steps") == true)
        d.implicitSteps = 8
        #expect(error(d) == nil)
        d.c2 = 2.0 / 3.0
        #expect(error(d)?.contains("c2") == true)
        d.c2 = 0
        #expect(error(d) != nil)
        d.c2 = 1.0
        #expect(error(d) == nil)
        d.noiseType = "perlin"
        #expect(error(d) != nil)
        d.noiseType = "fractal"; d.noiseAlpha = 0.5
        #expect(error(d) == nil)

        #expect(error(PresetSamplingDraft(modelFamily: "flux1", noiseType: "fractal", noiseAlpha: 0.5, implicitSteps: 2, c2: 0.6, projectorScale: 1.4)) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", implicitSteps: 9)) != nil)
    }

    @Test("vae override is refused outside krea2 with vaeGate's wording")
    func vaeRule() {
        #expect(error(PresetSamplingDraft(modelFamily: "krea2", vae: "/vae/Wan2_1_VAE_fp32.safetensors")) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", vae: "/vae/Wan2_1_VAE_fp32.safetensors"))?.contains("WP-E9") == true)
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", vae: "  ")) == nil)
    }

    @Test("stage 2: krea2 only; needs steps AND denoise in range; own pair; bongmath refused")
    func stage2Rules() {
        var d = PresetSamplingDraft(modelFamily: "krea2", sampler: "res_2s", stage2Enabled: true)
        #expect(error(d)?.contains("stage2.steps") == true)          // no steps
        d.stage2Steps = 0
        #expect(error(d) != nil)
        d.stage2Steps = 2
        #expect(error(d)?.contains("stage2.denoise") == true)       // no denoise
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
        d.stage2Sampler = ""; d.stage2SigmaSchedule = ""

        // A stored stage2.bongmath is surfaced, not dropped.
        d.stage2Bongmath = true
        #expect(error(d)?.contains("stage2.bongmath") == true)
        d.stage2Bongmath = nil

        // Outside krea2 the whole stage is refused (stage2Gate).
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", stage2Enabled: true, stage2Steps: 2, stage2Denoise: 0.2))?.contains("WP-E17") == true)
        // Disabled = not declared, whatever the stale fields hold.
        #expect(error(PresetSamplingDraft(modelFamily: "flux1", stage2Enabled: false, stage2Steps: 0, stage2Denoise: 9)) == nil)
    }

    /// Review B4 — the reachable case: krea2, res_2s + eta 0.5 on the render,
    /// stage 2 on euler with its eta left empty. The stage inherits eta 0.5
    /// (`stage2.eta ?? eta`) onto a sampler RES4LYF's SDE is not defined
    /// for — refused, with the engine's wording.
    @Test("stage 2 inherits the render's eta onto its own sampler")
    func stage2InheritsEta() {
        var d = PresetSamplingDraft(modelFamily: "krea2", sampler: "res_2s", eta: 0.5,
                                    stage2Enabled: true, stage2Steps: 2, stage2Denoise: 0.2, stage2Sampler: "euler")
        let message = error(d)
        #expect(message?.contains("stage 2 runs 'euler'") == true)
        #expect(message?.contains("parity tier T2") == true)
        // Explicit stage-2 eta 0 overrides the inheritance: valid.
        d.stage2Eta = 0
        #expect(error(d) == nil)
        // Stage sampler left at model default inherits res_2s: valid.
        d.stage2Eta = nil; d.stage2Sampler = ""
        #expect(error(d) == nil)
        // Stage names its own eta on euler while the render has none: refused.
        d.sampler = "euler"; d.eta = nil; d.stage2Sampler = ""; d.stage2Eta = 0.5
        #expect(error(d) != nil)
    }

    @Test("unknown family: only ranges apply")
    func unknownFamily() {
        #expect(error(PresetSamplingDraft(modelFamily: nil, sampler: "euler", eta: 0.5)) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: nil, shift: 3.0, stage2Enabled: true, stage2Steps: 2, stage2Denoise: 0.2)) == nil)
        #expect(error(PresetSamplingDraft(modelFamily: nil, implicitSteps: 9)) != nil)
    }
}

@Suite("PresetSamplingEditorState")
struct PresetSamplingEditorStateTests {
    /// #154's Zeta Chroma: a Z-Image-BASED checkpoint whose path contains
    /// "chroma", declaring the author's recommended linear shift 3.0.
    private static let zetaChroma = ServerPreset(
        id: "zeta-chroma", name: "Zeta Chroma",
        customModelPath: "/Models/zeta-chroma-v2.safetensors",
        steps: 20, guidance: 4.0, sampler: "euler", sigmaSchedule: "beta", shift: 3.0)

    /// Review B2: a loaded value survives every family answer — pending,
    /// wrong-looking text, the engine's real answer — and a refused family
    /// blocks Save rather than erasing it.
    @Test("shift 3.0 is never erased by a family answer; write carries it unchanged")
    func loadedShiftIsNeverErased() {
        let state = PresetSamplingEditorState(original: Self.zetaChroma)
        #expect(state.shift == 3.0)

        // Engine unreachable / answer pending: family nil, permissive — Save
        // allowed, and what is written is exactly what was loaded.
        let pending = state.draft(modelFamily: nil, sampler: "euler", sigmaSchedule: "beta")
        #expect(PresetSamplingValidator.validationError(pending) == nil)
        var saved = Self.zetaChroma
        state.write(into: &saved)
        #expect(saved.shift == 3.0)
        #expect(saved == Self.zetaChroma, "a save with the engine unreachable is byte-for-byte the loaded preset")

        // The engine answers flux1 (the truth): honoured, still 3.0.
        let flux1 = state.draft(modelFamily: "flux1", sampler: "euler", sigmaSchedule: "beta")
        #expect(PresetSamplingValidator.validationError(flux1) == nil)
        var afterAnswer = Self.zetaChroma
        state.write(into: &afterAnswer)
        #expect(afterAnswer.shift == 3.0)

        // Had the engine said chroma: refused → Save blocked, value STILL
        // present (Clear is the user's move, not the editor's).
        let chroma = state.draft(modelFamily: "chroma", sampler: "euler", sigmaSchedule: "beta")
        #expect(PresetSamplingValidator.validationError(chroma)?.contains("chroma") == true)
        var stillThere = Self.zetaChroma
        state.write(into: &stillThere)
        #expect(stillThere.shift == 3.0)
    }

    @Test("seed → write round-trips every recipe field, including a stored stage2.bongmath")
    func seedWriteRoundTrip() {
        let original = ServerPreset(
            id: "krea-clown", name: "Clown", model: "krea2-raw",
            projectorScale: 1.2, noiseType: "fractal", noiseAlpha: 0.5, implicitSteps: 2, c2: 0.4,
            vae: "/vae/Wan2_1_VAE_fp32.safetensors",
            sampler: "res_2s", sigmaSchedule: "beta57", shift: 1.15, eta: 0.5, bongmath: true,
            stage2: ServerPresetStage(sampler: "res_3s", sigmaSchedule: "bong_tangent", steps: 6, denoise: 0.4, eta: 0.3, bongmath: true))
        let state = PresetSamplingEditorState(original: original)
        var written = original
        state.write(into: &written)
        #expect(written == original)
        // The stored stage2.bongmath is surfaced by validation, not dropped.
        let draft = state.draft(modelFamily: "krea2", sampler: "res_2s", sigmaSchedule: "beta57")
        #expect(PresetSamplingValidator.validationError(draft)?.contains("stage2.bongmath") == true)
        // Clear (the user's explicit act) is what removes it.
        var cleared = state
        cleared.stage2Bongmath = nil
        var savedClean = original
        cleared.write(into: &savedClean)
        #expect(savedClean.stage2?.bongmath == nil)
        #expect(PresetSamplingValidator.validationError(
            cleared.draft(modelFamily: "krea2", sampler: "res_2s", sigmaSchedule: "beta57")) == nil)
    }

    @Test("neutral dials are written as nil, never frozen defaults; implicit steps round")
    func neutralWritesNil() {
        var state = PresetSamplingEditorState(original: ServerPreset(id: "b", name: "B"))
        state.implicitSteps = 2.4
        var p = ServerPreset(id: "b", name: "B")
        state.write(into: &p)
        #expect(p.eta == nil && p.bongmath == nil && p.shift == nil && p.vae == nil && p.stage2 == nil)
        #expect(p.noiseType == nil && p.noiseAlpha == nil && p.c2 == nil && p.projectorScale == nil)
        #expect(p.implicitSteps == 2)
    }

    @Test("the only automatic reset is the user's own sampler change")
    func samplerChangeIsTheOnlyReset() {
        var state = PresetSamplingEditorState(original: ServerPreset(
            id: "k", name: "K", model: "krea2-raw", sampler: "res_2s", eta: 0.5, bongmath: true))
        // Family answers arriving change nothing.
        #expect(state.eta == 0.5 && state.bongmath)
        // Picking euler on krea2 closes the gate: eta/bongmath drop.
        state.samplerDidChange(to: "euler", modelFamily: "krea2")
        #expect(state.eta == 0 && !state.bongmath)
        // Picking a RES4LYF sampler keeps them.
        var kept = PresetSamplingEditorState(original: ServerPreset(id: "k", name: "K", eta: 0.5, bongmath: true))
        kept.samplerDidChange(to: "res_3s", modelFamily: "krea2")
        #expect(kept.eta == 0.5 && kept.bongmath)
        // On flux1 nothing is refused, so nothing resets.
        var flux1 = PresetSamplingEditorState(original: ServerPreset(id: "k", name: "K", eta: 0.5, bongmath: true))
        flux1.samplerDidChange(to: "euler", modelFamily: "flux1")
        #expect(flux1.eta == 0.5 && flux1.bongmath)
    }
}
