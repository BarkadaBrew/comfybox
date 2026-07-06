// ServerView.swift — Server status and model-pool control: view loaded models,
// activate/unload, and load available models. Ports the Electron Server view.

import SwiftUI
import AppKit

struct ServerView: View {
    @Bindable var engine: EngineService

    @State private var busy: String?          // id of a model with an op in flight
    @State private var actionError: String?

    // Nearline storage
    @State private var nearline: EngineService.NearlineCatalog?
    @State private var nearlineBusy: String?
    @State private var nearlineFilter: String = ""
    @State private var nearlineSortBySize = true
    // LoRA library
    @State private var loraFilter: String = ""
    @State private var loraBusy: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                statusSection
                storageSection
                loadedPoolSection
                availableModelsSection
                loraLibrarySection
                nearlineSection
            }
            .padding(20)
        }
        .navigationTitle("Server")
        .task {
            await refreshAll()
            await engine.refreshLoras()
            nearline = await engine.fetchNearline()
        }
    }

    // MARK: - Storage overview (reduce primary-drive overuse)

    private var storageSection: some View {
        let disk = EngineService.primaryDiskInfo()
        let stagedGB = (nearline?.stagedMB ?? 0) / 1024
        let limitGB = nearline?.cacheLimitGB ?? 0
        let stagedFrac = limitGB > 0 ? min(1, stagedGB / limitGB) : 0
        return VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Storage").font(.headline)
                Spacer()
                if let n = nearline, n.items.contains(where: { $0.staged }) {
                    Button(role: .destructive) {
                        Task { await evictAll() }
                    } label: { Label("Evict all staged", systemImage: "internaldrive.badge.xmark") }
                        .controlSize(.small)
                        .disabled(nearlineBusy != nil)
                }
            }
            if let disk {
                let usedFrac = disk.totalGB > 0 ? (disk.totalGB - disk.freeGB) / disk.totalGB : 0
                gauge("Primary drive",
                      detail: String(format: "%.0f GB free of %.0f GB", disk.freeGB, disk.totalGB),
                      fraction: usedFrac,
                      warn: disk.freeGB < 50)
            }
            if limitGB > 0 {
                gauge("Nearline staged (on primary drive)",
                      detail: String(format: "%.1f / %.0f GB", stagedGB, limitGB),
                      fraction: stagedFrac,
                      warn: stagedFrac > 0.85)
                Text("Stage models/LoRAs from attached storage only when needed; evict to reclaim the primary drive.")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
            if disk == nil && limitGB == 0 {
                Text("Connect to the server to see storage usage.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func gauge(_ title: String, detail: String, fraction: Double, warn: Bool) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text(title).font(.caption.weight(.medium))
                Spacer()
                Text(detail).font(.caption.monospacedDigit()).foregroundStyle(warn ? .orange : .secondary)
            }
            ProgressView(value: fraction)
                .tint(warn ? .orange : .blue)
        }
    }

    private func evictAll() async {
        nearlineBusy = "__all__"
        defer { nearlineBusy = nil }
        if let result = try? await engine.evictAllStaged() { nearline = result.catalog }
    }

    // MARK: - LoRA library

    private var loraLibrarySection: some View {
        let loras = filteredLoras()
        let totalGB = engine.availableLoras.reduce(0.0) { $0 + Double($1.sizeBytes) } / 1_073_741_824
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("LoRA Library").font(.headline)
                Text("\(engine.availableLoras.count) · \(String(format: "%.1f GB", totalGB))")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                TextField("Filter…", text: $loraFilter).textFieldStyle(.roundedBorder).frame(width: 150)
                Button { Task { try? await engine.scanLoras(); await engine.refreshLoras() } } label: {
                    Label("Scan", systemImage: "arrow.clockwise")
                }.controlSize(.small).disabled(!engine.connectionState.isConnected)
            }
            if loras.isEmpty {
                Text(engine.connectionState.isConnected ? "No LoRAs match." : "Connect to browse the LoRA library.")
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                ForEach(loras.prefix(60)) { lora in loraRow(lora) }
                if loras.count > 60 {
                    Text("+\(loras.count - 60) more — filter to narrow.").font(.caption2).foregroundStyle(.tertiary)
                }
            }
        }
    }

    private func loraRow(_ lora: LoRAInfo) -> some View {
        HStack(spacing: 8) {
            Image(systemName: lora.quarantined ? "exclamationmark.triangle.fill" : "square.stack.3d.up")
                .foregroundStyle(lora.quarantined ? .orange : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(lora.filename).font(.callout).lineLimit(1).truncationMode(.middle)
                HStack(spacing: 8) {
                    Text(lora.modelCompatibility).font(.caption2).foregroundStyle(.secondary)
                    if lora.rank > 0 { Text("rank \(lora.rank)").font(.caption2).foregroundStyle(.tertiary) }
                    if lora.isActive { Text("active").font(.caption2).foregroundStyle(.green) }
                }
            }
            Spacer()
            Text(loraSizeLabel(lora)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            if loraBusy == lora.id {
                ProgressView().controlSize(.small)
            } else {
                Button(lora.quarantined ? "Release" : "Quarantine") {
                    Task { await toggleQuarantine(lora) }
                }
                .controlSize(.small)
                Button { revealLora(lora) } label: { Image(systemName: "magnifyingglass") }
                    .controlSize(.small).help("Reveal in Finder")
            }
        }
        .padding(8)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
    }

    private func filteredLoras() -> [LoRAInfo] {
        let base = loraFilter.isEmpty
            ? engine.availableLoras
            : engine.availableLoras.filter { $0.filename.lowercased().contains(loraFilter.lowercased()) }
        return base.sorted { $0.sizeBytes > $1.sizeBytes }
    }

    private func loraSizeLabel(_ lora: LoRAInfo) -> String {
        let mb = Double(lora.sizeBytes) / 1_048_576
        return mb >= 1024 ? String(format: "%.1f GB", mb / 1024) : String(format: "%.0f MB", mb)
    }

    private func toggleQuarantine(_ lora: LoRAInfo) async {
        loraBusy = lora.id
        defer { loraBusy = nil }
        try? await engine.quarantineLora(id: lora.id, quarantine: !lora.quarantined)
    }

    private func revealLora(_ lora: LoRAInfo) {
        // LoRAs live in the server library; reveal the filename in the default dir.
        let dir = NSString(string: "~/.comfybox/loras").expandingTildeInPath
        let path = (dir as NSString).appendingPathComponent(lora.filename)
        if FileManager.default.fileExists(atPath: path) {
            NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
        } else {
            NSWorkspace.shared.selectFile(nil, inFileViewerRootedAtPath: dir)
        }
    }

    // MARK: - Nearline storage

    private var nearlineSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text("Nearline Storage").font(.headline)
                if let nearline {
                    Text("\(nearline.items.count) items · \(String(format: "%.1f", nearline.stagedMB / 1024)) / \(Int(nearline.cacheLimitGB)) GB staged")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                TextField("Filter…", text: $nearlineFilter)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 160)
                Button {
                    Task { nearline = try? await engine.scanNearline() }
                } label: { Label("Scan", systemImage: "arrow.clockwise") }
                    .controlSize(.small)
                    .disabled(!engine.connectionState.isConnected)
            }

            if let nearline {
                if let root = nearline.roots.first {
                    Text(root)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                let visible = filteredNearlineItems(nearline.items)
                if visible.isEmpty {
                    emptyRow(nearline.items.isEmpty
                             ? "No catalog yet — press Scan to index attached storage."
                             : "No items match the filter.")
                } else {
                    VStack(spacing: 0) {
                        // Staged first, then by name; cap the list to keep the pane light.
                        ForEach(visible.prefix(60)) { item in
                            nearlineRow(item)
                            if item.id != visible.prefix(60).last?.id { Divider() }
                        }
                    }
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                    if visible.count > 60 {
                        Text("\(visible.count - 60) more — narrow with the filter.")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
            } else {
                emptyRow("Connect to the server to browse nearline storage.")
            }
        }
    }

    private func filteredNearlineItems(_ items: [EngineService.NearlineEntry]) -> [EngineService.NearlineEntry] {
        let base = nearlineFilter.isEmpty
            ? items
            : items.filter { $0.name.lowercased().contains(nearlineFilter.lowercased()) }
        return base.sorted {
            if $0.staged != $1.staged { return $0.staged }
            return $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending
        }
    }

    private func nearlineRow(_ item: EngineService.NearlineEntry) -> some View {
        HStack(spacing: 8) {
            Image(systemName: item.kind == "model" ? "cube" : "square.stack.3d.up")
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 18)
            Text(item.name)
                .font(.callout)
                .lineLimit(1)
                .truncationMode(.middle)
            if item.staged {
                Text("staged")
                    .font(.caption2)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(.green.opacity(0.2), in: Capsule())
                    .foregroundStyle(.green)
            }
            Spacer()
            Text(item.sizeLabel)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            if nearlineBusy == item.name {
                ProgressView().controlSize(.small)
            } else if item.staged {
                Button("Evict") { Task { await nearlineAct("evict", item) } }
                    .controlSize(.small)
            } else {
                Button("Stage") { Task { await nearlineAct("stage", item) } }
                    .controlSize(.small)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private func nearlineAct(_ action: String, _ item: EngineService.NearlineEntry) async {
        nearlineBusy = item.name
        defer { nearlineBusy = nil }
        do {
            nearline = try await engine.nearlineAction(action, name: item.name)
        } catch {
            actionError = error.localizedDescription
        }
    }

    // MARK: - Status

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle()
                    .fill(engine.connectionState.isConnected ? Color.green : Color.gray)
                    .frame(width: 10, height: 10)
                Text(engine.connectionState.label).font(.headline)
                Spacer()
                Button {
                    engine.connectionState.isConnected ? engine.disconnect() : engine.connect()
                } label: {
                    Text(engine.connectionState.isConnected ? "Disconnect" : "Connect")
                }
            }
            if let q = engine.queueInfo {
                HStack(spacing: 12) {
                    StatTile(title: "Status", value: q.isRendering ? "Rendering" : "Idle",
                             systemImage: "bolt.fill", tint: q.isRendering ? .orange : .green)
                    StatTile(title: "Pending", value: "\(q.pendingCount)", systemImage: "tray.full")
                    StatTile(title: "Memory",
                             value: q.memoryUsageMB >= 1024 ? String(format: "%.1f GB", Double(q.memoryUsageMB) / 1024) : "\(q.memoryUsageMB) MB",
                             systemImage: "memorychip")
                }
            }
            if let err = actionError {
                Text(err).font(.caption).foregroundStyle(.red)
            }
        }
    }

    // MARK: - Loaded pool

    private var loadedPoolSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Model Pool").font(.headline)
                Spacer()
                Button { Task { await engine.refreshPool() } } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .disabled(!engine.connectionState.isConnected)
            }

            if engine.poolModels.isEmpty {
                emptyRow("No models loaded.")
            } else {
                ForEach(engine.poolModels) { m in
                    HStack(spacing: 10) {
                        Circle().fill(m.active ? Color.green : Color.secondary.opacity(0.4))
                            .frame(width: 8, height: 8)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.model).lineLimit(1).truncationMode(.middle)
                            Text("\(m.family) · \(m.vramMB >= 1024 ? String(format: "%.1f GB", Double(m.vramMB) / 1024) : "\(m.vramMB) MB")")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        if m.active {
                            Text("Active").font(.caption).foregroundStyle(.green)
                        } else {
                            Button("Activate") { Task { await run(m.id) { try await engine.activateModel(id: m.id) } } }
                                .buttonStyle(.bordered)
                        }
                        Button {
                            Task { await run(m.id) { try await engine.unloadModel(id: m.id) } }
                        } label: {
                            Image(systemName: "eject")
                        }
                        .buttonStyle(.borderless)
                        .help("Unload")
                        if busy == m.id { ProgressView().controlSize(.small) }
                    }
                    .padding(10)
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    // MARK: - Available models

    private var availableModelsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Available Models").font(.headline)
                Spacer()
                Button { Task { await engine.refreshModels() } } label: {
                    Image(systemName: "arrow.clockwise")
                }
                .buttonStyle(.borderless)
                .disabled(!engine.connectionState.isConnected)
            }

            if engine.availableModels.isEmpty {
                emptyRow("No catalog models reported.")
            } else {
                ForEach(engine.availableModels) { m in
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(m.displayName).lineLimit(1)
                            Text("\(m.family) · \(m.quantization) · \(String(format: "%.1f", m.estimatedVRAM_GB)) GB")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button("Load") { Task { await run(m.id) { try await engine.loadModel(id: m.id) } } }
                            .buttonStyle(.bordered)
                            .disabled(!engine.connectionState.isConnected)
                        if busy == m.id { ProgressView().controlSize(.small) }
                    }
                    .padding(10)
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
                }
            }
        }
    }

    private func emptyRow(_ text: String) -> some View {
        Text(text).font(.callout).foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 4)
    }

    // MARK: - Actions

    private func refreshAll() async {
        guard engine.connectionState.isConnected else { return }
        await engine.refreshModels()
        await engine.refreshPool()
    }

    private func run(_ id: String, _ op: @escaping () async throws -> Void) async {
        busy = id
        actionError = nil
        defer { busy = nil }
        do {
            try await op()
            await engine.refreshPool()
        } catch {
            actionError = error.localizedDescription
        }
    }
}
