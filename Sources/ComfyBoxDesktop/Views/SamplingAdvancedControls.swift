// SamplingAdvancedControls.swift — the shared "RES4LYF knobs" subview (#419)
//
// One family-aware view for the Clownshark recipe dials, used by BOTH the
// Generate panel and the server preset editor so the eta/bongmath gating
// exists once (`SamplingGate`). Bindings use the panel's neutral sentinels —
// eta 0, bongmath false, gaussian, alpha 0, implicit 0, c2 0.5, projector 1.0,
// shift nil — which both callers already map to "omit from the wire /
// preset" (never freezing today's default into a recipe).
//
// When a gate closes (the family or sampler changes under an active value)
// the affected dials are RESET to neutral, not merely greyed out: a value that
// is disabled but still set would be silently saved into a preset the engine
// then 400s on at render.

import SwiftUI
import ZImage

struct SamplingAdvancedControls: View {
    @Binding var shift: Double?
    @Binding var projectorScale: Double
    @Binding var eta: Double
    @Binding var bongmath: Bool
    @Binding var noiseType: String
    @Binding var noiseAlpha: Double
    @Binding var implicitSteps: Double
    @Binding var c2: Double
    /// The current stage-1 sampler ("" = model default), which decides the
    /// RES4LYF gate.
    var sampler: String
    var sigmaSchedule: String = ""
    /// `/health` model_family, a model id/path, or nil (unknown = permissive).
    var modelFamily: String?
    /// The caller's own enablement (Generate disables everything for a cloud
    /// backend). Gates below are AND-ed with it.
    var isEnabled: Bool = true
    /// Whether to show the `shift` row at all (Generate hides it — the panel
    /// has no shift dial of its own beyond what an applied preset carries).
    var showsShift: Bool = true

    private var sdeAllowed: Bool {
        SamplingGate.sdeAllowed(modelFamily: modelFamily, sampler: sampler)
    }
    private var noiseAllowed: Bool {
        SamplingGate.noiseAllowed(modelFamily: modelFamily)
    }
    private var shiftAllowed: Bool {
        SamplingGate.shiftAllowed(modelFamily: modelFamily)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if showsShift {
                shiftRow
            }

            // Projector scale — CFG-free prompt-adherence gain (Krea2). 1.0 = off.
            NumericSliderField(label: "Projector Scale", value: $projectorScale, range: 0...3, step: 0.05, fractionDigits: 2)
                .disabled(!isEnabled || !noiseAllowed)

            // RES4LYF SDE / bongmath (the Clownshark recipe): eta>0 turns on SDE
            // noise re-injection, bongmath aligns substeps. Both are sampler-gated
            // 400s on the engine (krea2 + a RES4LYF sampler), never no-ops.
            NumericSliderField(label: "Eta (SDE)", value: $eta, range: 0...1, step: 0.05, fractionDigits: 2)
                .disabled(!isEnabled || !sdeAllowed)
            Toggle("Bongmath", isOn: $bongmath)
                .font(.caption)
                .disabled(!isEnabled || !sdeAllowed)
            if !sdeAllowed {
                Text(SamplingGate.sdeDisabledReason(modelFamily: modelFamily, sampler: sampler))
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            HStack(spacing: 8) {
                Text("Noise Type")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(width: 76, alignment: .leading)
                Picker("Noise Type", selection: $noiseType) {
                    Text("Gaussian").tag("gaussian")
                    Text("Fractal").tag("fractal")
                    Text("Pyramid").tag("pyramid")
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .disabled(!isEnabled || !noiseAllowed)
            NumericSliderField(
                label: "Noise Alpha", value: $noiseAlpha,
                range: -2...2, step: 0.1, fractionDigits: 1
            )
            .disabled(!isEnabled || !noiseAllowed)
            NumericSliderField(
                label: "Implicit Steps", value: $implicitSteps,
                range: 0...8, step: 1
            )
            .disabled(!isEnabled || !noiseAllowed)
            NumericSliderField(
                label: "C2", value: $c2,
                range: 0.05...1, step: 0.05, fractionDigits: 2
            )
            .disabled(!isEnabled || !noiseAllowed)
            if !noiseAllowed {
                Text("Noise, implicit steps, C2 and projector scale are Krea 2 settings; \(SamplingRecipeCatalog.canonicalFamily(modelFamily) ?? "this model") does not read them.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .onChange(of: sdeAllowed, initial: true) { _, allowed in
            if !allowed { eta = 0; bongmath = false }
        }
        .onChange(of: noiseAllowed, initial: true) { _, allowed in
            if !allowed {
                noiseType = "gaussian"; noiseAlpha = 0; implicitSteps = 0; c2 = 0.5; projectorScale = 1.0
            }
        }
        .onChange(of: shiftAllowed, initial: true) { _, allowed in
            if !allowed { shift = nil }
        }
    }

    @ViewBuilder
    private var shiftRow: some View {
        HStack(spacing: 8) {
            Text(SamplingRecipeCatalog.shiftLabel(forModelFamily: modelFamily))
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(width: 96, alignment: .leading)
            TextField("Model default", value: $shift, format: .number.precision(.fractionLength(0...3)))
                .textFieldStyle(.roundedBorder)
                .font(.subheadline.monospacedDigit())
                .multilineTextAlignment(.trailing)
                .frame(width: 80)
            if shift != nil {
                Button {
                    shift = nil
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .help("Clear — use the model's resolution-dependent default")
            }
            Spacer()
        }
        .disabled(!isEnabled || !shiftAllowed)
        .help(shiftHelp)
    }

    private var shiftHelp: String {
        switch SamplingRecipeCatalog.canonicalFamily(modelFamily) {
        case "krea2":
            return "Krea 2 reads shift as mu (the reference recipe states 1.15). Empty = the resolution-dependent default."
        case "flux1":
            return "Z-Image reads shift as a linear sigma warp (1.0 = identity). Empty = the model's own shift."
        case .some(let family):
            return "\(family) runs a fixed schedule and does not honour shift."
        case nil:
            return "Empty = the model's own resolution-dependent shift."
        }
    }
}
