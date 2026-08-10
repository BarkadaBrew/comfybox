// ModelsView.swift — Model, LoRA, and storage management: pool (load/activate/
// unload), available models, the LoRA library, and nearline staging with a
// primary-drive storage overview to curb overuse. (Server status lives in the
// Server tab.)

import SwiftUI
import AppKit
import UniformTypeIdentifiers

struct ModelsView: View {
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
    // Import LoRA… flow (spec 2026-08-10): nil = sheet closed.
    @State private var loraImportExpansion: LoRAImportPlanner.Expansion?
    // Folded LoRA families, persisted across launches (comma-joined — family
    // tokens never contain commas). An active search overrides folds so a
    // match is never hidden inside a collapsed group.
    @AppStorage("modelsView.collapsedLoraFamilies") private var collapsedLoraCSV: String = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if let err = actionError {
                    Text(err).font(.caption).foregroundStyle(.red)
                }
                storageSection
                featuredModelsSection
                loadedPoolSection
                availableModelsSection
                loraLibrarySection
                nearlineSection
            }
            .padding(20)
        }
        .navigationTitle("Models & LoRAs")
        .task {
            await refreshAll()
            await engine.refreshLoras()
            nearline = await engine.fetchNearline()
        }
        .sheet(isPresented: Binding(
            get: { loraImportExpansion != nil },
            set: { if !$0 { loraImportExpansion = nil } }
        )) {
            if let expansion = loraImportExpansion {
                LoRAImportSheet(engine: engine, expansion: expansion) {
                    Task { await engine.refreshLoras() }
                }
            }
        }
    }

    // MARK: - Import LoRA… (spec 2026-08-10)

    /// The /v1/loras/import route copies from a path on the ENGINE's disk.
    private var engineIsLocal: Bool {
        ["127.0.0.1", "localhost", "::1"].contains(engine.serverHost)
    }

    private func pickLorasToImport() {
        let panel = NSOpenPanel()
        panel.title = "Import LoRAs"
        panel.message = "Choose .safetensors files, or folders to import every LoRA inside"
        panel.allowsMultipleSelection = true
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        // Filter files to .safetensors (folders stay selectable); the planner
        // re-filters anyway, so a failed UTType lookup just means no dimming.
        if let type = UTType(filenameExtension: "safetensors") {
            panel.allowedContentTypes = [type]
        }
        guard panel.runModal() == .OK else { return }
        loraImportExpansion = LoRAImportPlanner.expand(urls: panel.urls)
    }

    // MARK: - Featured art models (Zeta-Chroma, CoffeeShop)

    private var featuredModelsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Art Models").font(.headline)
            ForEach(FeaturedModels.all) { fm in featuredRow(fm) }
        }
    }

    private func featuredRow(_ fm: FeaturedModel) -> some View {
        // Match against the live catalog, pool, and nearline.
        let catalogMatch = engine.availableModels.first { fm.matches($0.id) || fm.matches($0.displayName) }
        let poolMatch = engine.poolModels.first { fm.matches($0.id) || fm.matches($0.model) }
        let nearMatch = nearline?.items.first { fm.matches($0.name) }
        let (statusText, statusColor): (String, Color) =
            poolMatch?.active == true ? ("active", .green)
            : poolMatch != nil ? ("loaded", .green)
            : catalogMatch != nil ? ("available", .blue)
            : nearMatch?.staged == true ? ("staged", .blue)
            : nearMatch != nil ? ("on attached storage", .orange)
            : ("not found — mount its drive to stage", .secondary)

        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: "paintpalette.fill").foregroundStyle(.pink)
                Text(fm.name).font(.callout.weight(.medium))
                Text(fm.family).font(.caption2).padding(.horizontal, 5).padding(.vertical, 1)
                    .background(.quaternary, in: Capsule())
                Spacer()
                Text(statusText).font(.caption).foregroundStyle(statusColor)
            }
            Text(fm.blurb).font(.caption2).foregroundStyle(.secondary)
            HStack(spacing: 8) {
                if let cat = catalogMatch, poolMatch == nil {
                    Button("Load") { Task { await run(cat.id) { try await engine.loadModel(id: cat.id) } } }
                        .controlSize(.small)
                } else if let pool = poolMatch, pool.active != true {
                    Button("Activate") { Task { await run(pool.id) { try await engine.activateModel(id: pool.id) } } }
                        .controlSize(.small)
                } else if let near = nearMatch, !near.staged {
                    Button("Stage") { Task { await nearlineAct("stage", near) } }
                        .controlSize(.small).disabled(nearlineBusy != nil)
                } else if catalogMatch == nil && poolMatch == nil && nearMatch == nil, let hint = fm.nearlineHint {
                    Text(hint).font(.caption2).foregroundStyle(.tertiary).lineLimit(1).truncationMode(.middle)
                }
            }
        }
        .padding(10)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 10))
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
        let groups = groupedLoras(loras)
        let totalGB = engine.availableLoras.reduce(0.0) { $0 + Double($1.sizeBytes) } / 1_073_741_824
        let searching = !loraFilter.isEmpty
        return VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("LoRA Library").font(.headline)
                Text("\(engine.availableLoras.count) · \(String(format: "%.1f GB", totalGB))")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                // Searches filename, category, and trigger words — typing
                // "krea2" surfaces the whole family.
                TextField("Search name or category…", text: $loraFilter)
                    .textFieldStyle(.roundedBorder).frame(width: 190)
                if !groups.isEmpty && !searching {
                    // One-click fold state for the whole library. Hidden while
                    // searching (search force-expands, the buttons would lie).
                    let allCollapsed = groups.allSatisfy { collapsedLoraFamilies.contains($0.family) }
                    Button(allCollapsed ? "Expand all" : "Collapse all") {
                        collapsedLoraCSV = allCollapsed ? "" : groups.map(\.family).joined(separator: ",")
                    }.controlSize(.small)
                }
                Button { Task { try? await engine.scanLoras(); await engine.refreshLoras() } } label: {
                    Label("Scan", systemImage: "arrow.clockwise")
                }.controlSize(.small).disabled(!engine.connectionState.isConnected)
                // Import passes a server-LOCAL path — only offered when the
                // engine runs on this machine (spec 2026-08-10).
                Button { pickLorasToImport() } label: {
                    Label("Import LoRA…", systemImage: "square.and.arrow.down")
                }
                .controlSize(.small)
                .disabled(!engine.connectionState.isConnected || !engineIsLocal)
                .help(engineIsLocal
                      ? "Copy .safetensors files or folders into the library"
                      : "Import needs the engine on this machine (it copies from a local path)")
            }
            if loras.isEmpty {
                // An empty list must say WHY. A failed fetch used to look
                // identical to "there are no LoRAs" (2026-08-10).
                Text(engine.loraLoadError
                     ?? (engine.connectionState.isConnected
                         ? (searching ? "Nothing matches “\(loraFilter)”." : "No LoRAs match.")
                         : "Connect to browse the LoRA library."))
                    .font(.caption).foregroundStyle(.secondary)
            } else {
                // LoRAs are subordinate to their requisite model — group under
                // the model family they're designed for. Groups fold; an
                // active search force-expands so a hit is never hidden.
                ForEach(groups, id: \.family) { group in
                    let expanded = searching || !collapsedLoraFamilies.contains(group.family)
                    VStack(alignment: .leading, spacing: 4) {
                        Button {
                            toggleLoraGroup(group.family)
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "chevron.right")
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                    .rotationEffect(.degrees(expanded ? 90 : 0))
                                Image(systemName: "cube.fill").font(.caption2).foregroundStyle(.secondary)
                                Text(group.title).font(.subheadline.weight(.semibold))
                                Text("\(group.loras.count) · \(group.sizeLabel)")
                                    .font(.caption2).foregroundStyle(.tertiary)
                                Spacer()
                            }
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(searching)   // folds are meaningless mid-search
                        .padding(.top, 4)
                        if expanded {
                            ForEach(group.loras.prefix(40)) { lora in loraRow(lora) }
                            if group.loras.count > 40 {
                                Text("+\(group.loras.count - 40) more — search to narrow.")
                                    .font(.caption2).foregroundStyle(.tertiary)
                            }
                        }
                    }
                }
            }
        }
    }

    private var collapsedLoraFamilies: Set<String> {
        Set(collapsedLoraCSV.split(separator: ",").map(String.init))
    }

    private func toggleLoraGroup(_ family: String) {
        var set = collapsedLoraFamilies
        if set.contains(family) { set.remove(family) } else { set.insert(family) }
        collapsedLoraCSV = set.sorted().joined(separator: ",")
    }

    private struct LoRAGroup { let family: String; let title: String; let loras: [LoRAInfo]; let sizeLabel: String }

    /// Group LoRAs by the model family they're built for, active model's family
    /// first, then by total size. Unknowns fall into an "Uncategorized" group.
    private func groupedLoras(_ loras: [LoRAInfo]) -> [LoRAGroup] {
        let activeFamily = (engine.currentModelFamily ?? engine.currentModel).map { LoRACompatibility.family(from: $0) }
        let buckets = Dictionary(grouping: loras) { LoRACompatibility.family(from: $0.modelCompatibility) }
        return buckets.map { fam, items in
            let bytes = items.reduce(0) { $0 + $1.sizeBytes }
            let mb = Double(bytes) / 1_048_576
            let size = mb >= 1024 ? String(format: "%.1f GB", mb / 1024) : String(format: "%.0f MB", mb)
            let title = fam.isEmpty ? "Uncategorized" : LoRACompatibility.label(for: fam)
            return LoRAGroup(family: fam.isEmpty ? "~" : fam, title: title,
                             loras: items.sorted { $0.sizeBytes > $1.sizeBytes }, sizeLabel: size)
        }
        .sorted { a, b in
            if let af = activeFamily {                     // active model's family first
                if (a.family == af) != (b.family == af) { return a.family == af }
            }
            if (a.family == "~") != (b.family == "~") { return b.family == "~" }  // Uncategorized last
            return a.title < b.title
        }
    }

    private func loraRow(_ lora: LoRAInfo) -> some View {
        HStack(spacing: 8) {
            Image(systemName: lora.quarantined ? "exclamationmark.triangle.fill" : "square.stack.3d.up")
                .foregroundStyle(lora.quarantined ? .orange : .secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(lora.filename).font(.callout).lineLimit(1).truncationMode(.middle)
                HStack(spacing: 8) {
                    let fam = LoRACompatibility.family(from: lora.modelCompatibility)
                    Text(fam.isEmpty ? lora.modelCompatibility : LoRACompatibility.label(for: fam))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 4).padding(.vertical, 1)
                        .background(.quaternary.opacity(0.5), in: Capsule())
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

    /// Search matches filename, family/category (both raw and display label,
    /// so "Krea 2" and "krea2" both hit), server category, and trigger words —
    /// the things you actually remember about a LoRA.
    private func filteredLoras() -> [LoRAInfo] {
        let q = loraFilter.lowercased()
        let base = q.isEmpty
            ? engine.availableLoras
            : engine.availableLoras.filter { lora in
                if lora.filename.lowercased().contains(q) { return true }
                if lora.category.lowercased().contains(q) { return true }
                if lora.modelCompatibility.lowercased().contains(q) { return true }
                let fam = LoRACompatibility.family(from: lora.modelCompatibility)
                if !fam.isEmpty, fam.contains(q) || LoRACompatibility.label(for: fam).lowercased().contains(q) { return true }
                return lora.triggerwords.contains { $0.lowercased().contains(q) }
            }
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
