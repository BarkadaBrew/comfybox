// LoRAPicker.swift — LoRA selection and scale control
//
// Lists available LoRAs from the server's LoRA library. Each row shows
// LoRA name, checkbox to enable, and scale slider. Selected LoRAs are
// tracked as LoRASelection instances and passed to generation requests.
// Active LoRAs (currently loaded on the server) appear at the top.

import SwiftUI

struct LoRAPicker: View {
    @Bindable var engine: EngineService
    @Binding var selectedLoras: [LoRASelection]

    @State private var searchText: String = ""
    @State private var errorMessage: String?
    @State private var compatibleOnly: Bool = true

    /// The active model's identifier for compatibility checks (family or name).
    private var activeModel: String? {
        engine.currentModelFamily ?? engine.currentModel
    }

    private func compatStatus(_ lora: LoRAInfo) -> LoRACompatibility.Status {
        LoRACompatibility.status(loraCompatibility: lora.modelCompatibility, modelIdentifier: activeModel)
    }

    private func isIncompatible(_ lora: LoRAInfo) -> Bool {
        if case .incompatible = compatStatus(lora) { return true }
        return false
    }

    private var filteredLoras: [LoRAInfo] {
        var base = engine.availableLoras.filter { !$0.quarantined }

        // Hide LoRAs designed for a different model family (keep unknowns +
        // selected ones so nothing already chosen silently vanishes).
        if compatibleOnly {
            base = base.filter { lora in
                !isIncompatible(lora) || selectedLoras.contains { $0.id == lora.id }
            }
        }

        guard !searchText.isEmpty else {
            return sortedLoras(base)
        }

        let query = searchText.lowercased()
        let filtered = base.filter { lora in
            lora.id.lowercased().contains(query)
                || lora.filename.lowercased().contains(query)
                || lora.tags.contains { $0.lowercased().contains(query) }
                || lora.category.lowercased().contains(query)
                || lora.triggerwords.contains { $0.lowercased().contains(query) }
        }
        return sortedLoras(filtered)
    }

    /// Stable order: active on server first, then alphabetical. Deliberately does
    /// NOT sort by selection — reordering a row at the moment it's selected (while
    /// its content also changes to reveal the scale slider) leaves SwiftUI showing
    /// a stale row until the list is rebuilt. Keeping order stable lets the row
    /// gain its slider in place.
    private func sortedLoras(_ loras: [LoRAInfo]) -> [LoRAInfo] {
        loras.sorted { a, b in
            if a.isActive != b.isActive { return a.isActive }
            return a.id < b.id
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Header with refresh
            HStack {
                Text("LoRA Adapters")
                    .font(.subheadline)
                    .fontWeight(.semibold)

                Spacer()

                if !selectedLoras.isEmpty {
                    Text("\(selectedLoras.count) selected")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }

                Button(action: { Task { await engine.refreshLoras() } }) {
                    Image(systemName: "arrow.clockwise")
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .help("Refresh LoRA library")
            }

            // Compatibility filter — hide LoRAs built for a different model.
            if let model = activeModel, !model.isEmpty {
                Toggle(isOn: $compatibleOnly) {
                    Text("Only \(LoRACompatibility.label(for: LoRACompatibility.family(from: model))) LoRAs")
                        .font(.caption2)
                }
                .toggleStyle(.checkbox)
                .help("Hide LoRAs designed for a different model family")
            }

            // Search
            if engine.availableLoras.count > 5 {
                HStack(spacing: 4) {
                    Image(systemName: "magnifyingglass")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("Search LoRAs...", text: $searchText)
                        .textFieldStyle(.plain)
                        .font(.caption)
                    if !searchText.isEmpty {
                        Button(action: { searchText = "" }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(4)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 4))
            }

            // LoRA list
            if engine.availableLoras.isEmpty {
                if engine.connectionState.isConnected {
                    Text("No LoRAs available")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                } else {
                    Text("Connect to server to load LoRAs")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
            } else if filteredLoras.isEmpty {
                Text("No LoRAs match \"\(searchText)\"")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                ScrollView {
                    VStack(spacing: 2) {
                        ForEach(filteredLoras) { lora in
                            loraRow(lora)
                        }
                    }
                }
                .frame(maxHeight: 240)
            }

            // LoRAs are applied automatically at Generate time — no separate step.
            if !selectedLoras.isEmpty {
                Text("\(selectedLoras.count) LoRA\(selectedLoras.count == 1 ? "" : "s") selected — applied automatically on Generate")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }

            // Error display
            if let error = errorMessage {
                Text(error)
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
    }

    // MARK: - LoRA Row

    private func loraRow(_ lora: LoRAInfo) -> some View {
        let isSelected = selectedLoras.contains { $0.id == lora.id }
        let selectionIndex = selectedLoras.firstIndex { $0.id == lora.id }

        return VStack(spacing: 4) {
            HStack(spacing: 6) {
                // Checkbox
                Button(action: { toggleLora(lora) }) {
                    Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                        .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
                        .font(.caption)
                }
                .buttonStyle(.plain)

                // Name and info
                VStack(alignment: .leading, spacing: 1) {
                    HStack(spacing: 4) {
                        Text(lora.id)
                            .font(.caption)
                            .lineLimit(1)
                            .truncationMode(.tail)

                        if lora.isActive {
                            Text("active")
                                .font(.system(size: 9))
                                .foregroundStyle(.green)
                                .padding(.horizontal, 4)
                                .padding(.vertical, 1)
                                .background(Color.green.opacity(0.1))
                                .clipShape(RoundedRectangle(cornerRadius: 3))
                        }
                        compatibilityBadge(lora)
                    }

                    HStack(spacing: 4) {
                        Text(formattedSize(lora.sizeBytes))
                            .font(.caption2)
                            .foregroundStyle(.secondary)

                        if !lora.category.isEmpty {
                            Text(lora.category)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }

                        if !lora.triggerwords.isEmpty {
                            Text(lora.triggerwords.joined(separator: ", "))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .truncationMode(.tail)
                        }
                    }
                }

                Spacer()
            }

            // Scale slider when selected
            if isSelected, let index = selectionIndex {
                HStack(spacing: 6) {
                    Text("Scale")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .frame(width: 32, alignment: .leading)

                    Slider(
                        value: Binding(
                            get: { selectedLoras[index].scale },
                            set: { newValue in
                                // Magnetic detent at 0 so it's easy to neutralize/disable a
                                // LoRA; otherwise round to 0.01 for fine-grained control across
                                // the wider -5...5 range (sliders often need negatives/overdrive).
                                selectedLoras[index].scale = abs(newValue) < 0.08
                                    ? 0.0
                                    : (newValue * 100).rounded() / 100
                            }
                        ),
                        in: -5.0...5.0
                    )
                    .controlSize(.mini)

                    TextField(
                        "",
                        value: Binding(
                            get: { selectedLoras[safe: index]?.scale ?? 1.0 },
                            set: { selectedLoras[index].scale = min(max($0, -5.0), 5.0) }
                        ),
                        format: .number.precision(.fractionLength(0...2))
                    )
                    .textFieldStyle(.roundedBorder)
                    .font(.caption2.monospacedDigit())
                    .multilineTextAlignment(.trailing)
                    .frame(width: 44)
                }
                .padding(.leading, 22)
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(isSelected ? Color.accentColor.opacity(0.06) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    @ViewBuilder
    private func compatibilityBadge(_ lora: LoRAInfo) -> some View {
        switch compatStatus(lora) {
        case .incompatible(let loraFam, _):
            Text("⚠ \(LoRACompatibility.label(for: loraFam))")
                .font(.system(size: 9))
                .foregroundStyle(.orange)
                .padding(.horizontal, 4).padding(.vertical, 1)
                .background(Color.orange.opacity(0.12))
                .clipShape(RoundedRectangle(cornerRadius: 3))
                .help("This LoRA is built for \(LoRACompatibility.label(for: loraFam)), not the active model.")
        case .compatible, .unknown:
            EmptyView()
        }
    }

    // MARK: - Actions

    private func toggleLora(_ lora: LoRAInfo) {
        if let index = selectedLoras.firstIndex(where: { $0.id == lora.id }) {
            selectedLoras.remove(at: index)
            errorMessage = nil
        } else {
            // Warn (don't block) when selecting a LoRA for a different model.
            if case .incompatible(let loraFam, let modelFam) = compatStatus(lora) {
                errorMessage = "⚠ \(lora.id) is a \(LoRACompatibility.label(for: loraFam)) LoRA but the active model is \(LoRACompatibility.label(for: modelFam)) — results may be poor."
            }
            selectedLoras.append(LoRASelection(
                id: lora.id,
                filename: lora.filename,
                scale: lora.recommendedScale
            ))
        }
    }

    // MARK: - Helpers

    private func formattedSize(_ bytes: Int) -> String {
        if bytes < 1024 { return "\(bytes) B" }
        let kb = Double(bytes) / 1024
        if kb < 1024 { return String(format: "%.0f KB", kb) }
        let mb = kb / 1024
        if mb < 1024 { return String(format: "%.1f MB", mb) }
        let gb = mb / 1024
        return String(format: "%.1f GB", gb)
    }
}

// MARK: - Safe Array Access

private extension Array {
    subscript(safe index: Int) -> Element? {
        guard index >= 0 && index < count else { return nil }
        return self[index]
    }
}
