// NumericSliderField.swift — Slider with a synchronized editable number field
//
// Everywhere the UI sliders a numeric value, the exact number can also be
// typed. The field commits on Return / focus loss and clamps into the
// slider's range so a typo can't push an out-of-range value to the server.

import SwiftUI

struct NumericSliderField: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    /// Digits shown/parsed after the decimal point (0 = integer semantics).
    var fractionDigits: Int = 0

    @State private var text: String = ""
    @FocusState private var fieldFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                TextField("", text: $text)
                    .textFieldStyle(.roundedBorder)
                    .font(.subheadline.monospacedDigit())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 64)
                    .focused($fieldFocused)
                    .onSubmit { commitText() }
                    .onChange(of: fieldFocused) { _, focused in
                        if !focused { commitText() }
                    }
            }
            Slider(value: $value, in: range, step: step)
        }
        .onAppear { text = format(value) }
        .onChange(of: value) { _, newValue in
            // Keep the field in sync with slider drags / programmatic sets,
            // but never fight the user mid-edit.
            if !fieldFocused { text = format(newValue) }
        }
    }

    private func format(_ v: Double) -> String {
        String(format: "%.\(fractionDigits)f", v)
    }

    /// Parse, clamp into range, snap the display. Unparseable input reverts.
    private func commitText() {
        let normalized = text.replacingOccurrences(of: ",", with: ".")
        if let parsed = Double(normalized) {
            value = min(max(parsed, range.lowerBound), range.upperBound)
        }
        text = format(value)
    }
}
