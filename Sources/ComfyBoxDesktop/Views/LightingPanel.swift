// LightingPanel.swift — Lighting direction control
//
// Pick a light direction, quality, and mood/time; a live lighting phrase is
// composed and appended to the prompt. Mirrors CameraPanel.

import SwiftUI

struct LightingPanel: View {
    var onInsert: (LightingDirective) -> Void

    @State private var directive = LightingDirective()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            picker("Dir", selection: $directive.direction, cases: LightingDirective.Direction.allCases) { $0.label }
            picker("Light", selection: $directive.quality, cases: LightingDirective.Quality.allCases) { $0.label }
            picker("Mood", selection: $directive.mood, cases: LightingDirective.Mood.allCases) { $0.label }

            // Live preview of the composed phrase.
            Text(directive.isEmpty ? "No lighting selected" : directive.phrase)
                .font(.caption)
                .foregroundStyle(directive.isEmpty ? .tertiary : .secondary)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))

            Button {
                onInsert(directive)
            } label: {
                Label("Add Lighting to Prompt", systemImage: "text.badge.plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .disabled(directive.isEmpty)
        }
    }

    private func picker<T: Hashable>(
        _ title: String,
        selection: Binding<T>,
        cases: [T],
        label: @escaping (T) -> String
    ) -> some View {
        HStack {
            Text(title).font(.caption).foregroundStyle(.secondary).frame(width: 46, alignment: .leading)
            Picker("", selection: selection) {
                ForEach(cases, id: \.self) { Text(label($0)).tag($0) }
            }
            .labelsHidden()
            .fixedSize()
            Spacer()
        }
    }
}
