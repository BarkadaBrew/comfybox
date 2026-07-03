// CameraPanel.swift — Camera placement control (v1)
//
// Pick orientation, angle, shot size, and lens; a live camera phrase is
// composed and appended to the prompt. Imported shot templates (cinematic
// directives from the legacy image service) sit below as one-tap inserts.
// v2 will pair this with an img2img/Klein reference so the SAME subject is
// re-rendered from the chosen viewpoint.

import SwiftUI

struct CameraPanel: View {
    var shotTemplates: ShotTemplateStore
    var onInsert: (CameraDirective) -> Void
    var onInsertTemplate: (ShotTemplate) -> Void

    @State private var directive = CameraDirective()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            picker("Shot", selection: $directive.shotSize, cases: CameraDirective.ShotSize.allCases) { $0.label }
            picker("View", selection: $directive.orientation, cases: CameraDirective.Orientation.allCases) { $0.label }
            picker("Angle", selection: $directive.angle, cases: CameraDirective.Angle.allCases) { $0.label }

            HStack {
                Text("Lens").font(.caption).foregroundStyle(.secondary).frame(width: 46, alignment: .leading)
                Picker("", selection: $directive.lens) {
                    ForEach(CameraDirective.Lens.allCases, id: \.self) { Text($0.label).tag($0) }
                }
                .labelsHidden()
                .fixedSize()
                Spacer()
            }

            // Live preview of the composed phrase.
            Text(directive.phrase)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(6)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))

            Button {
                onInsert(directive)
            } label: {
                Label("Add Camera to Prompt", systemImage: "text.badge.plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)

            if !shotTemplates.templates.isEmpty {
                Divider()
                Text("Shot Templates")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                ForEach(shotTemplates.templates) { template in
                    Button {
                        onInsertTemplate(template)
                    } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Text(template.name).font(.caption.weight(.medium))
                                if let mode = template.contentMode {
                                    Text(mode).font(.system(size: 9))
                                        .padding(.horizontal, 4).padding(.vertical, 1)
                                        .background(.quaternary, in: Capsule())
                                }
                            }
                            Text(template.directive)
                                .font(.caption2).foregroundStyle(.secondary)
                                .lineLimit(2).multilineTextAlignment(.leading)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.plain)
                    .padding(6)
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 6))
                }
            }
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
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }
}
