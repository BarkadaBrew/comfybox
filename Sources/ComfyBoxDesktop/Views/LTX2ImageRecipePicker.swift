import SwiftUI
import ZImage

/// Native LTX image sampler controls. The schedule picker intentionally has
/// one real choice: LTX image sigmas are token-shifted flow sigmas.
struct LTX2ImageRecipePicker: View {
    @Binding var sampler: String
    @Binding var sigmaSchedule: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            row(title: "Sampler", selection: $sampler, names: LTX2ImageRecipe.samplerNames)
            row(title: "Scheduler", selection: $sigmaSchedule, names: LTX2ImageRecipe.sigmaScheduleNames)
            Text("Euler variants share LTX's native token-shifted flow schedule. CFG++ adds a negative-conditioning pass; ancestral variants add seeded SDE noise.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .onAppear {
            if sampler.isEmpty { sampler = LTX2ImageRecipe.defaultSampler }
            if sigmaSchedule.isEmpty { sigmaSchedule = LTX2ImageRecipe.defaultSigmaSchedule }
        }
    }

    @ViewBuilder
    private func row(title: String, selection: Binding<String>, names: [String]) -> some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .frame(width: 76, alignment: .leading)
            Picker(title, selection: selection) {
                ForEach(names, id: \.self) { name in
                    Text(name.replacingOccurrences(of: "_", with: " ").capitalized + "  ·  " + name)
                        .tag(name)
                }
                if !selection.wrappedValue.isEmpty, !names.contains(selection.wrappedValue) {
                    Divider()
                    Text("Unsupported: \(selection.wrappedValue)").tag(selection.wrappedValue)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
