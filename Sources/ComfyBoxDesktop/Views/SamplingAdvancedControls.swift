// SamplingAdvancedControls.swift — the shared "RES4LYF knobs" subview (#419)
//
// One family-aware view for the Clownshark recipe dials, used by BOTH the
// Generate panel and the server preset editor so the eta/bongmath gating
// exists once (`SamplingGate` → `SamplingRecipeCatalog`). Bindings use the
// panel's neutral sentinels — eta 0, bongmath false, gaussian, alpha 0,
// implicit 0, c2 0.5, projector 1.0, shift nil — which both callers already
// map to "omit from the wire / preset" (never freezing today's default).
//
// Gate semantics (review B2): a loaded value is NEVER erased by a gate. A
// `.refused` field is greyed, shows the engine's reason and a Clear button
// next to the offending value; `.inert` (accepted-and-ignored by this
// family) stays editable with a label. The one automatic reset is a USER
// change of the sampler picker (`onSamplerChanged`), which the caller wires.

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
    /// The ENGINE's answer for the model (`/health` model_family or
    /// `/v1/model/family`), or nil while unknown/pending — nil is permissive.
    var modelFamily: String?
    /// The caller's own enablement (Generate disables everything for a cloud
    /// backend). Gates below are AND-ed with it.
    var isEnabled: Bool = true

    private var etaStatus: SamplingGate.Status { SamplingGate.eta(modelFamily: modelFamily, sampler: sampler) }
    private var bongmathStatus: SamplingGate.Status { SamplingGate.bongmath(modelFamily: modelFamily, sampler: sampler) }
    private var noiseStatus: SamplingGate.Status { SamplingGate.noise(modelFamily: modelFamily) }
    private var shiftStatus: SamplingGate.Status { SamplingGate.shift(modelFamily: modelFamily, sigmaSchedule: sigmaSchedule) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            shiftRow

            // Projector scale — CFG-free prompt-adherence gain (Krea2). 1.0 = off.
            NumericSliderField(label: "Projector Scale", value: $projectorScale, range: 0...3, step: 0.05, fractionDigits: 2)
                .disabled(!isEnabled)

            // RES4LYF SDE / bongmath (the Clownshark recipe): eta>0 turns on SDE
            // noise re-injection, bongmath aligns substeps. On Krea 2 both are
            // sampler-gated 400s (RES4LYF samplers only); on Z-Image eta is DDIM /
            // DPM++ 2S-A's own η and inert elsewhere; bongmath is inert.
            NumericSliderField(label: "Eta (SDE)", value: $eta, range: 0...1, step: 0.05, fractionDigits: 2)
                .disabled(!isEnabled || etaStatus.isRefused)
            statusRow(etaStatus, isSet: eta != 0, clearLabel: "Clear eta") { eta = 0 }

            Toggle("Bongmath", isOn: $bongmath)
                .font(.caption)
                .disabled(!isEnabled || bongmathStatus.isRefused)
            statusRow(bongmathStatus, isSet: bongmath, clearLabel: "Clear bongmath") { bongmath = false }

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
            .disabled(!isEnabled)
            NumericSliderField(
                label: "Noise Alpha", value: $noiseAlpha,
                range: -2...2, step: 0.1, fractionDigits: 1
            )
            .disabled(!isEnabled)
            NumericSliderField(
                label: "Implicit Steps", value: $implicitSteps,
                range: 0...8, step: 1
            )
            .disabled(!isEnabled)
            NumericSliderField(
                label: "C2", value: $c2,
                range: 0.05...1, step: 0.05, fractionDigits: 2
            )
            .disabled(!isEnabled)
            if let note = noiseStatus.note {
                Text(note).font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }

    /// Under a gated control: the engine's reason (orange when refused and
    /// a value is set — that is the Save-blocking path — tertiary when
    /// merely inert) and, for a refused value, the Clear button that fixes it.
    @ViewBuilder
    private func statusRow(_ status: SamplingGate.Status, isSet: Bool, clearLabel: String, clear: @escaping () -> Void) -> some View {
        if let note = status.note {
            HStack(alignment: .top, spacing: 6) {
                if status.isRefused, isSet {
                    Label(note, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption2).foregroundStyle(.orange)
                    Spacer(minLength: 0)
                    Button(clearLabel, action: clear)
                        .controlSize(.small)
                        .disabled(!isEnabled)
                } else {
                    Text(note).font(.caption2).foregroundStyle(.tertiary)
                }
            }
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
                .disabled(!isEnabled || shiftStatus.isRefused)
            if shift != nil {
                // Always operable, even when the field itself is greyed —
                // this is the on-screen way out of a refused value.
                Button {
                    shift = nil
                } label: {
                    Image(systemName: "xmark.circle.fill")
                }
                .buttonStyle(.borderless)
                .disabled(!isEnabled)
                .help("Clear — use the model's resolution-dependent default")
            }
            Spacer()
        }
        .help(shiftHelp)
        statusRow(shiftStatus, isSet: shift != nil, clearLabel: "Clear shift") { shift = nil }
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
