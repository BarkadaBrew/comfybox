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

    private var filteredLoras: [LoRAInfo] {
        let nonQuarantined = engine.availableLoras.filter { !$0.quarantined }

        guard !searchText.isEmpty else {
            return sortedLoras(nonQuarantined)
        }

        let query = searchText.lowercased()
        let filtered = nonQuarantined.filter { lora in
            lora.id.lowercased().contains(query)
                || lora.filename.lowercased().contains(query)
                || lora.tags.contains { $0.lowercased().contains(query) }
                || lora.category.lowercased().contains(query)
                || lora.triggerwords.contains { $0.lowercased().contains(query) }
        }
        return sortedLoras(filtered)
    }

    /// Sort: selected first, then active on server, then alphabetical.
    private func sortedLoras(_ loras: [LoRAInfo]) -> [LoRAInfo] {
        loras.sorted { a, b in
            let aSelected = selectedLoras.contains { $0.id == a.id }
            let bSelected = selectedLoras.contains { $0.id == b.id }
            if aSelected != bSelected { return aSelected }
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

            // Apply button if selections differ from server state
            if !selectedLoras.isEmpty {
                Button(action: { Task { await applyLoras() } }) {
                    HStack(spacing: 4) {
                        if engine.isSwappingLoras {
                            ProgressView()
                                .controlSize(.mini)
                        }
                        Text(engine.isSwappingLoras ? "Applying..." : "Apply LoRAs")
                    }
                    .frame(maxWidth: .infinity)
                    .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(engine.isSwappingLoras)
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
                            set: { selectedLoras[index].scale = $0 }
                        ),
                        in: 0.0...2.0,
                        step: 0.05
                    )
                    .controlSize(.mini)

                    Text(String(format: "%.2f", selectedLoras[safe: index]?.scale ?? 1.0))
                        .font(.caption2)
                        .monospacedDigit()
                        .frame(width: 32)
                }
                .padding(.leading, 22)
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(isSelected ? Color.accentColor.opacity(0.06) : Color.clear)
        .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    // MARK: - Actions

    private func toggleLora(_ lora: LoRAInfo) {
        if let index = selectedLoras.firstIndex(where: { $0.id == lora.id }) {
            selectedLoras.remove(at: index)
        } else {
            selectedLoras.append(LoRASelection(
                id: lora.id,
                filename: lora.filename,
                scale: lora.recommendedScale
            ))
        }
    }

    private func applyLoras() async {
        errorMessage = nil
        do {
            try await engine.swapLoras(selectedLoras)
        } catch {
            errorMessage = error.localizedDescription
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
