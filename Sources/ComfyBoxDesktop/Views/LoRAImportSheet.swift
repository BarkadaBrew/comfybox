// LoRAImportSheet.swift — the Models tab's "Import LoRA…" flow (spec
// 2026-08-10): a resolved batch of .safetensors files, a category to file
// them under, and per-file outcomes as the batch runs. The heavy lifting is
// server-side (POST /v1/loras/import copies + rescans); this sheet is the
// picker, the filing decision, and the receipt.

import SwiftUI

struct LoRAImportSheet: View {
    let engine: EngineService
    let expansion: LoRAImportPlanner.Expansion
    let onDone: () -> Void

    @State private var category: String = "vault"
    @State private var newCategory: String = ""
    @State private var running = false
    @State private var finished = false
    @State private var outcomes: [String: String] = [:]  // filename → outcome line
    @Environment(\.dismiss) private var dismiss

    /// The folder the batch files into: the free-text field wins when set.
    private var effectiveCategory: String {
        let typed = newCategory.trimmingCharacters(in: .whitespaces)
        return typed.isEmpty ? category : typed
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Import \(expansion.files.count) LoRA\(expansion.files.count == 1 ? "" : "s")")
                .font(.headline)
            if expansion.skipped > 0 {
                Text("\(expansion.skipped) selected item\(expansion.skipped == 1 ? "" : "s") skipped (not .safetensors, or duplicate names).")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if expansion.files.isEmpty {
                Text("Nothing to import — the selection contained no .safetensors files.")
                    .foregroundStyle(.secondary)
            } else {
                HStack {
                    Picker("Category", selection: $category) {
                        ForEach(LoRAImportPlanner.categories(from: engine.availableLoras), id: \.self) {
                            Text($0).tag($0)
                        }
                    }
                    .frame(maxWidth: 220)
                    .disabled(running || !newCategory.trimmingCharacters(in: .whitespaces).isEmpty)
                    TextField("New category…", text: $newCategory)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 180)
                        .disabled(running)
                }
                Text("Files land in ~/Models/loras/\(effectiveCategory)/ and appear in the library, picker, and presets after import.")
                    .font(.caption2).foregroundStyle(.tertiary)

                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(expansion.files, id: \.self) { file in
                            HStack(spacing: 8) {
                                Image(systemName: icon(for: file))
                                    .foregroundStyle(iconColor(for: file))
                                    .frame(width: 14)
                                Text(file.lastPathComponent).font(.caption)
                                Spacer()
                                Text(outcomes[file.lastPathComponent] ?? sizeLabel(file))
                                    .font(.caption2).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .frame(minHeight: 120, maxHeight: 260)
            }

            HStack {
                Spacer()
                Button(finished ? "Done" : "Cancel") {
                    dismiss()
                    if finished { onDone() }
                }
                .keyboardShortcut(finished ? .defaultAction : .cancelAction)
                if !expansion.files.isEmpty && !finished {
                    Button(running ? "Importing…" : "Import") { Task { await runBatch() } }
                        .keyboardShortcut(.defaultAction)
                        .disabled(running)
                }
            }
        }
        .padding(20)
        .frame(width: 520)
    }

    private func icon(for file: URL) -> String {
        guard let outcome = outcomes[file.lastPathComponent] else { return "doc" }
        if outcome.hasPrefix("failed") { return "exclamationmark.triangle.fill" }
        return "checkmark.circle.fill"
    }

    private func iconColor(for file: URL) -> Color {
        guard let outcome = outcomes[file.lastPathComponent] else { return .secondary }
        return outcome.hasPrefix("failed") ? .orange : .green
    }

    private func sizeLabel(_ file: URL) -> String {
        let bytes = (try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }

    private func runBatch() async {
        running = true
        defer { running = false; finished = true }
        let known = Set(engine.availableLoras.map(\.filename))
        for file in expansion.files {
            let name = file.lastPathComponent
            do {
                let entry = try await engine.importLora(path: file.path, category: effectiveCategory)
                if known.contains(entry.filename) {
                    outcomes[name] = "already in library"
                } else {
                    let family = entry.modelCompatibility.joined(separator: ", ")
                    outcomes[name] = family.isEmpty ? "imported" : "imported · \(family)"
                }
            } catch {
                // One bad file never stops the batch (spec: per-file outcomes).
                outcomes[name] = "failed: \(error.localizedDescription)"
            }
        }
        await engine.refreshLoras()
    }
}
