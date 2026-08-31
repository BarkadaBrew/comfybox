import SwiftUI
import ZImage

/// Shared sampler + sigma-schedule controls used by both Generate and Presets.
/// Empty strings are the UI sentinel for "model default" and are omitted from
/// the wire/preset rather than freezing today's default into every recipe.
struct SamplingRecipePicker: View {
    @Binding var sampler: String
    @Binding var sigmaSchedule: String
    /// `/health` model_family or a model id/path. The engine catalog accepts
    /// either and maps it to the same family capability table used at render.
    var modelFamily: String?
    var showsExplanation: Bool = true

    private var samplerNames: [String] {
        SamplingRecipeCatalog.samplerNames(forModelFamily: modelFamily)
    }

    private var scheduleNames: [String] {
        SamplingRecipeCatalog.sigmaScheduleNames(
            forModelFamily: modelFamily,
            sampler: sampler.isEmpty ? nil : sampler
        )
    }

    private var canonicalFamily: String? {
        SamplingRecipeCatalog.canonicalFamily(modelFamily)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            samplingRow(title: "Sampler", selection: $sampler, names: samplerNames) {
                let value = SamplingRecipeCatalog.defaultSamplerName(forModelFamily: modelFamily)
                return "Model Default (\(Self.displayName(value)))"
            }
            samplingRow(title: "Scheduler", selection: $sigmaSchedule, names: scheduleNames) {
                let value = SamplingRecipeCatalog.defaultSigmaScheduleName(forModelFamily: modelFamily)
                return "Model Default (\(Self.displayName(value)))"
            }

            if let error = validationError {
                Label(error, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            } else if showsExplanation {
                Text("Sampler chooses the solver; Scheduler chooses its sigma/noise schedule. Model Default leaves that choice to the active model.")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .onChange(of: sampler) { _, _ in
            // A sampler change can invalidate a family-specific pair (Chroma
            // euler+beta). Reset only the schedule half to model default; the
            // user's explicit sampler choice remains visible.
            if !sigmaSchedule.isEmpty, !scheduleNames.contains(sigmaSchedule) {
                sigmaSchedule = ""
            }
        }
    }

    var validationError: String? {
        guard canonicalFamily != nil else { return nil }
        guard !sampler.isEmpty || !sigmaSchedule.isEmpty else { return nil }
        guard !SamplingRecipeCatalog.supports(
            sampler: sampler.isEmpty ? nil : sampler,
            sigmaSchedule: sigmaSchedule.isEmpty ? nil : sigmaSchedule,
            forModelFamily: modelFamily
        ) else { return nil }
        return "This sampler/scheduler pair is not supported by \(canonicalFamily ?? "the active model")."
    }

    @ViewBuilder
    private func samplingRow(
        title: String,
        selection: Binding<String>,
        names: [String],
        defaultLabel: () -> String
    ) -> some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(width: 76, alignment: .leading)
            Picker(title, selection: selection) {
                Text(defaultLabel()).tag("")
                Divider()
                ForEach(names, id: \.self) { name in
                    Text(Self.optionLabel(name)).tag(name)
                }
                if !selection.wrappedValue.isEmpty, !names.contains(selection.wrappedValue) {
                    Divider()
                    Text("Unsupported: \(selection.wrappedValue)").tag(selection.wrappedValue)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
            .help(selection.wrappedValue.isEmpty
                ? defaultLabel()
                : "Wire value: \(selection.wrappedValue)")
        }
    }

    private static func optionLabel(_ raw: String) -> String {
        "\(displayName(raw))  ·  \(raw)"
    }

    private static func displayName(_ raw: String) -> String {
        switch raw {
        case "euler": return "Euler"
        case "heun": return "Heun"
        case "dpmpp-2m": return "DPM++ 2M"
        case "dpmpp-2s-a": return "DPM++ 2S Ancestral"
        case "deis": return "DEIS"
        case "ddim": return "DDIM"
        case "res_2s": return "RES 2S"
        case "res_3s": return "RES 3S"
        case "ralston_2s": return "Ralston 2S"
        case "ralston_3s": return "Ralston 3S"
        case "ralston_4s": return "Ralston 4S"
        case "heun_2s": return "Heun 2S"
        case "heun_3s": return "Heun 3S"
        case "deis_2m": return "DEIS 2M"
        case "deis_3m": return "DEIS 3M"
        case "deis_4m": return "DEIS 4M"
        case "flow": return "Flow"
        case "karras": return "Karras"
        case "exponential": return "Exponential"
        case "beta": return "Beta"
        case "beta57": return "Beta 5.7"
        case "krea2": return "Krea 2 Native"
        case "bong_tangent": return "Bong Tangent"
        case "simple": return "Simple"
        default: return raw.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}
