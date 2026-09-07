// SamplingUserEditTests.swift — #419 (review blocker): the one automatic
// eta/bongmath reset fires for the USER's own sampler pick and never for a
// programmatic write (an applied preset, a model switch settling), and is
// suppressed while an apply is in flight.

import Testing
import SwiftUI
@testable import ComfyBoxDesktop

@Suite("Sampler user-edit routing")
struct SamplingUserEditTests {
    @Test("the picker's proxy binding reports user picks only; programmatic writes bypass it")
    func proxyBindingDistinguishesUserEdits() {
        var value = ""
        var reported: [String] = []
        let base = Binding(get: { value }, set: { value = $0 })
        let proxy = SamplingRecipePicker.userEditBinding(base) { reported.append($0) }

        // Programmatic (applyPreset) write: the base changes, nothing is reported.
        base.wrappedValue = "ddim"
        #expect(value == "ddim")
        #expect(reported.isEmpty)
        #expect(proxy.wrappedValue == "ddim")

        // User pick through the Picker: base changes AND it is reported.
        proxy.wrappedValue = "res_2s"
        #expect(value == "res_2s")
        #expect(reported == ["res_2s"])
    }

    /// The blocker scenario: krea2 is resident; the user applies a Z-Image
    /// preset (ddim + eta 0.6). Its model takes ~70 s to load, so at apply
    /// time the resident family is still krea2 — against which `ddim + eta`
    /// would be refused. The apply is a programmatic write, so no reset
    /// runs; when flux1 becomes resident the eta is honoured, still 0.6.
    @Test("applying ddim + eta 0.6 over a resident krea2 never erases eta; honoured once flux1 is resident")
    func applyPresetDoesNotResetAgainstTheOldFamily() {
        var sampler = ""
        var eta = 0.0
        var bongmath = false
        var residentFamily: String? = "krea2"
        var applyInFlight = false
        let base = Binding(get: { sampler }, set: { sampler = $0 })
        // Exactly what GenerationView hangs on the picker.
        let picker = SamplingRecipePicker.userEditBinding(base) { newSampler in
            let outcome = SamplingGate.userSamplerChange(
                to: newSampler, modelFamily: residentFamily, eta: eta, bongmath: bongmath, applyInFlight: applyInFlight)
            eta = outcome.eta; bongmath = outcome.bongmath
        }

        // applyPreset: programmatic writes + model switch begins.
        applyInFlight = true
        base.wrappedValue = "ddim"
        eta = 0.6
        #expect(eta == 0.6, "no reset against the still-resident krea2")
        #expect(SamplingGate.eta(modelFamily: residentFamily, sampler: sampler).isRefused,
                "precondition: krea2 WOULD refuse ddim + eta — the old rule erased it here")

        // Even a user pick during the switch is not judged against the old family.
        picker.wrappedValue = "ddim"
        #expect(eta == 0.6)

        // Model switch settles: flux1 resident, ddim's own η honoured.
        residentFamily = "flux1"
        applyInFlight = false
        #expect(SamplingGate.eta(modelFamily: residentFamily, sampler: sampler).isHonoured)
        #expect(eta == 0.6)
        #expect(PresetSamplingValidator.validationError(
            PresetSamplingDraft(modelFamily: residentFamily, sampler: sampler, eta: eta)) == nil)

        // A user pick on flux1 of a sampler that ignores eta: inert, kept.
        picker.wrappedValue = "euler"
        #expect(eta == 0.6)

        // Back on krea2, the user's own pick of euler is the one reset.
        residentFamily = "krea2"
        eta = 0.5; bongmath = true
        picker.wrappedValue = "euler"
        #expect(eta == 0 && !bongmath)
        // …and picking a RES4LYF sampler keeps values.
        eta = 0.5; bongmath = true
        picker.wrappedValue = "res_3s"
        #expect(eta == 0.5 && bongmath)
    }

    @Test("userSamplerChange is a no-op while an apply is in flight")
    func suppressedWhileApplying() {
        let kept = SamplingGate.userSamplerChange(
            to: "euler", modelFamily: "krea2", eta: 0.5, bongmath: true, applyInFlight: true)
        #expect(kept.eta == 0.5 && kept.bongmath)
        let reset = SamplingGate.userSamplerChange(
            to: "euler", modelFamily: "krea2", eta: 0.5, bongmath: true, applyInFlight: false)
        #expect(reset.eta == 0 && !reset.bongmath)
    }
}
