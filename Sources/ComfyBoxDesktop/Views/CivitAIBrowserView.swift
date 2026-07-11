// CivitAIBrowserView.swift — Browse, scrape, and download from CivitAI
//
// Search models/LoRAs (defaults to the Z-Image family), inspect versions
// with trigger words and sample-image prompts (copy or save to the Prompt
// Library), and download files straight into the local LoRA library
// (~/.comfybox/loras) followed by a server rescan. Works keyless; the API
// key in Settings unlocks auth-gated listings and downloads.

import SwiftUI

/// A user-facing filter label mapped to the CivitAI API value (nil = no filter).
struct CivitAIFilterOption: Hashable {
    let label: String
    let apiValue: String?
}

/// Which CivitAI host to query. `.red` is the uncensored mirror; it defaults
/// NSFW on. Both speak the same /api/v1 shape.
enum CivitAISource: String, CaseIterable, Identifiable {
    case com = "civitai.com"
    case red = "civitai.red"
    var id: String { rawValue }
    var baseURL: URL { URL(string: "https://\(rawValue)")! }
    var defaultsNSFW: Bool { self == .red }
}

struct CivitAIBrowserView: View {
    @Bindable var engine: EngineService
    var promptLibrary: PromptLibraryStore?

    @State private var query: String = ""
    @State private var source: CivitAISource = .com
    @State private var typeFilter = CivitAIFilterOption(label: "LoRA", apiValue: "LORA")
    @State private var baseModelFilter = CivitAIFilterOption(label: "Z-Image Turbo", apiValue: "ZImageTurbo")
    @State private var sort: CivitAIClient.SortOrder = .mostDownloaded
    @State private var period: CivitAIClient.Period = .allTime
    @State private var includeNSFW = false
    @State private var results: [CivitAIModel] = []
    @State private var nextCursor: String?
    @State private var isLoading = false
    @State private var loadError: String?
    @State private var selected: CivitAIModel?

    private static let typeOptions: [CivitAIFilterOption] = [
        .init(label: "LoRA", apiValue: "LORA"),
        .init(label: "LyCORIS", apiValue: "LoCon"),
        .init(label: "Checkpoint", apiValue: "Checkpoint"),
        .init(label: "Embedding", apiValue: "TextualInversion"),
        .init(label: "ControlNet", apiValue: "Controlnet"),
        .init(label: "VAE", apiValue: "VAE"),
        .init(label: "Poses", apiValue: "Poses"),
        .init(label: "Wildcards", apiValue: "Wildcards"),
        .init(label: "All types", apiValue: nil),
    ]

    // Values are CivitAI's exact `baseModels` API strings (verified against the
    // live API), NOT display guesses — e.g. Z-Image is "ZImageBase"/"ZImageTurbo".
    private static let baseModelOptions: [CivitAIFilterOption] = [
        .init(label: "Z-Image Turbo", apiValue: "ZImageTurbo"),
        .init(label: "Z-Image Base", apiValue: "ZImageBase"),
        .init(label: "Flux.1 Dev", apiValue: "Flux.1 D"),
        .init(label: "Flux.1 Schnell", apiValue: "Flux.1 S"),
        .init(label: "Krea 2", apiValue: "Krea 2"),
        .init(label: "Flux.1 Krea", apiValue: "Flux.1 Krea"),
        .init(label: "Qwen", apiValue: "Qwen"),
        .init(label: "Wan Video", apiValue: "Wan Video"),
        .init(label: "LTXV 2.3", apiValue: "LTXV 2.3"),
        .init(label: "HiDream", apiValue: "HiDream"),
        .init(label: "SDXL 1.0", apiValue: "SDXL 1.0"),
        .init(label: "Pony", apiValue: "Pony"),
        .init(label: "Illustrious", apiValue: "Illustrious"),
        .init(label: "NoobAI", apiValue: "NoobAI"),
        .init(label: "SD 3.5", apiValue: "SD 3.5"),
        .init(label: "SD 1.5", apiValue: "SD 1.5"),
        .init(label: "All base models", apiValue: nil),
    ]

    private func sortLabel(_ s: CivitAIClient.SortOrder) -> String { s.rawValue }
    private func periodLabel(_ p: CivitAIClient.Period) -> String {
        p == .allTime ? "All Time" : p.rawValue
    }

    private var client: CivitAIClient {
        CivitAIClient(baseURL: source.baseURL, apiKey: AppSecrets.civitai)
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            if let loadError {
                Text(loadError).font(.caption).foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12).padding(.vertical, 6)
            }
            if results.isEmpty && !isLoading {
                emptyState
            } else {
                resultsGrid
            }
        }
        .navigationTitle("CivitAI")
        .task { await search(reset: true) }
        .sheet(item: $selected) { model in
            CivitAIModelSheet(
                model: model,
                client: client,
                engine: engine,
                promptLibrary: promptLibrary
            )
        }
    }

    private var toolbar: some View {
        VStack(spacing: 8) {
            // Row 1: source + search + refresh
            HStack(spacing: 10) {
                Picker("", selection: $source) {
                    ForEach(CivitAISource.allCases) { src in
                        Text(src.rawValue).tag(src)
                    }
                }
                .pickerStyle(.segmented)
                .fixedSize()
                .onChange(of: source) { _, newValue in
                    includeNSFW = newValue.defaultsNSFW
                    Task { await search(reset: true) }
                }

                HStack {
                    Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                    TextField("Search \(source.rawValue)…", text: $query)
                        .textFieldStyle(.plain)
                        .onSubmit { Task { await search(reset: true) } }
                    if !query.isEmpty {
                        Button { query = ""; Task { await search(reset: true) } } label: {
                            Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary)
                        }.buttonStyle(.plain)
                    }
                }
                .padding(6)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6))

                if isLoading { ProgressView().controlSize(.small) }
                Button { Task { await search(reset: true) } } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(.borderless)
                    .help("Refresh")
            }

            // Row 2: filter pulldowns
            HStack(spacing: 10) {
                labeledPicker("Type", systemImage: "square.stack.3d.up") {
                    Picker("", selection: $typeFilter) {
                        ForEach(Self.typeOptions, id: \.self) { Text($0.label).tag($0) }
                    }.labelsHidden().fixedSize()
                    .onChange(of: typeFilter) { _, _ in Task { await search(reset: true) } }
                }
                labeledPicker("Base", systemImage: "cube") {
                    Picker("", selection: $baseModelFilter) {
                        ForEach(Self.baseModelOptions, id: \.self) { Text($0.label).tag($0) }
                    }.labelsHidden().fixedSize()
                    .onChange(of: baseModelFilter) { _, _ in Task { await search(reset: true) } }
                }
                labeledPicker("Sort", systemImage: "arrow.up.arrow.down") {
                    Picker("", selection: $sort) {
                        ForEach(CivitAIClient.SortOrder.allCases, id: \.self) { Text(sortLabel($0)).tag($0) }
                    }.labelsHidden().fixedSize()
                    .onChange(of: sort) { _, _ in Task { await search(reset: true) } }
                }
                labeledPicker("Period", systemImage: "calendar") {
                    Picker("", selection: $period) {
                        ForEach(CivitAIClient.Period.allCases, id: \.self) { Text(periodLabel($0)).tag($0) }
                    }.labelsHidden().fixedSize()
                    .onChange(of: period) { _, _ in Task { await search(reset: true) } }
                }

                Spacer()

                Toggle(isOn: $includeNSFW) {
                    Label("NSFW", systemImage: includeNSFW ? "eye" : "eye.slash")
                }
                .toggleStyle(.button)
                .controlSize(.small)
                .onChange(of: includeNSFW) { _, _ in Task { await search(reset: true) } }
                .help("Include adult content")
            }
        }
        .padding(12)
    }

    private func labeledPicker<Content: View>(_ title: String, systemImage: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(spacing: 4) {
            Image(systemName: systemImage).font(.caption2).foregroundStyle(.secondary)
            content()
        }
    }

    private var resultsGrid: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 210), spacing: 12)], spacing: 12) {
                ForEach(results) { model in
                    CivitAIModelCard(model: model)
                        .onTapGesture { selected = model }
                }
            }
            .padding(12)

            if nextCursor != nil {
                Button("Load More") { Task { await search(reset: false) } }
                    .disabled(isLoading)
                    .padding(.bottom, 16)
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "globe").font(.system(size: 32)).foregroundStyle(.tertiary)
            Text(isLoading ? "Searching…" : "No results")
                .font(.subheadline).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func search(reset: Bool) async {
        if reset { nextCursor = nil }
        isLoading = true
        defer { isLoading = false }
        loadError = nil
        do {
            let page = try await client.searchModels(
                query: query,
                types: typeFilter.apiValue.map { [$0] } ?? [],
                baseModel: baseModelFilter.apiValue,
                sort: sort,
                period: period,
                nsfw: includeNSFW,
                cursor: reset ? nil : nextCursor
            )
            var items = page.items
            // The server-side baseModels filter is skipped for text queries
            // (see CivitAIClient.searchModels) — apply it client-side here.
            let trimmedQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmedQuery.isEmpty, let baseModel = baseModelFilter.apiValue {
                items = items.filter { model in
                    model.modelVersions.contains { $0.baseModel == baseModel }
                }
            }
            results = reset ? items : results + items
            nextCursor = page.nextCursor
        } catch {
            loadError = "\(source.rawValue) request failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Card

private struct CivitAIModelCard: View {
    let model: CivitAIModel

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ZStack {
                Rectangle().fill(.quaternary.opacity(0.4))
                if let url = model.thumbnailURL {
                    AsyncImage(url: url) { phase in
                        if case .success(let image) = phase {
                            image.resizable().aspectRatio(contentMode: .fill)
                        } else {
                            Image(systemName: "photo").foregroundStyle(.tertiary)
                        }
                    }
                } else {
                    Image(systemName: "square.stack.3d.up").font(.title).foregroundStyle(.tertiary)
                }
            }
            .frame(height: 150)
            .clipShape(RoundedRectangle(cornerRadius: 8))

            Text(model.name)
                .font(.callout.weight(.medium))
                .lineLimit(2)
            HStack(spacing: 8) {
                Text(model.type)
                    .font(.caption2)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(.quaternary, in: Capsule())
                if let creator = model.creatorName {
                    Text(creator).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
                }
                Spacer()
                Label("\(model.downloadCount)", systemImage: "arrow.down.circle")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(8)
        .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
        .contentShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Detail sheet

private struct CivitAIModelSheet: View {
    let model: CivitAIModel
    let client: CivitAIClient
    @Bindable var engine: EngineService
    var promptLibrary: PromptLibraryStore?

    @Environment(\.dismiss) private var dismiss
    @State private var selectedVersionId: Int?
    @State private var scrapedImages: [CivitAIImage] = []
    @State private var downloadProgress: Double?
    @State private var statusMessage: String?
    @State private var statusIsError = false

    private var version: CivitAIModelVersion? {
        model.modelVersions.first(where: { $0.id == selectedVersionId }) ?? model.modelVersions.first
    }

    /// The model's page on whichever CivitAI host we're browsing.
    private var modelWebURL: URL? {
        guard model.id > 0 else { return nil }
        var s = "\(client.baseURL.absoluteString)/models/\(model.id)"
        if let vid = version?.id, vid > 0 { s += "?modelVersionId=\(vid)" }
        return URL(string: s)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.name).font(.headline)
                    HStack(spacing: 8) {
                        Text(model.type).font(.caption).foregroundStyle(.secondary)
                        if let creator = model.creatorName {
                            Text("by \(creator)").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                Spacer()
                if let url = modelWebURL {
                    Link(destination: url) {
                        Label("View on CivitAI", systemImage: "safari")
                    }
                    .font(.callout)
                }
                Button { dismiss() } label: { Image(systemName: "xmark.circle.fill").font(.title3) }
                    .buttonStyle(.plain).foregroundStyle(.secondary)
            }
            .padding()
            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if model.modelVersions.count > 1 {
                        Picker("Version", selection: Binding(
                            get: { version?.id ?? 0 },
                            set: { selectedVersionId = $0 }
                        )) {
                            ForEach(model.modelVersions) { v in
                                Text("\(v.name)  (\(v.baseModel))").tag(v.id)
                            }
                        }
                        .fixedSize()
                    }

                    if let version {
                        if !version.trainedWords.isEmpty {
                            labeledRow("Trigger words") {
                                Text(version.trainedWords.joined(separator: ", "))
                                    .font(.callout.monospaced())
                                    .textSelection(.enabled)
                            }
                        }

                        if let file = version.primaryFile {
                            labeledRow("File") {
                                HStack(spacing: 10) {
                                    Text("\(file.name)  ·  \(file.sizeLabel)")
                                        .font(.callout)
                                    Spacer()
                                    if let progress = downloadProgress {
                                        ProgressView(value: progress)
                                            .frame(width: 120)
                                        Text("\(Int(progress * 100))%")
                                            .font(.caption.monospacedDigit())
                                    } else {
                                        Button {
                                            Task { await download(file) }
                                        } label: {
                                            Label("Download to Library", systemImage: "arrow.down.circle")
                                        }
                                        .buttonStyle(.borderedProminent)
                                        .controlSize(.small)
                                    }
                                }
                            }
                        }

                        promptSection(version)
                    }

                    if let statusMessage {
                        Label(statusMessage, systemImage: statusIsError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                            .font(.callout)
                            .foregroundStyle(statusIsError ? .orange : .green)
                    }
                }
                .padding()
            }
        }
        .frame(minWidth: 620, idealWidth: 700, minHeight: 480, idealHeight: 600)
        .task(id: version?.id) { await scrapePrompts() }
    }

    @ViewBuilder
    private func promptSection(_ version: CivitAIModelVersion) -> some View {
        let inline = version.images.compactMap(\.prompt)
        let scraped = scrapedImages.compactMap(\.prompt)
        let prompts = Array(NSOrderedSet(array: inline + scraped)) as? [String] ?? inline

        if !prompts.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Sample prompts")
                    .font(.subheadline.weight(.semibold))
                ForEach(Array(prompts.prefix(8).enumerated()), id: \.offset) { _, prompt in
                    HStack(alignment: .top, spacing: 8) {
                        Text(prompt)
                            .font(.caption)
                            .lineLimit(4)
                            .textSelection(.enabled)
                        Spacer()
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(prompt, forType: .string)
                        } label: { Image(systemName: "doc.on.doc") }
                            .buttonStyle(.borderless)
                            .help("Copy prompt")
                        if let promptLibrary {
                            Button {
                                promptLibrary.add(text: prompt, tags: ["civitai", model.name])
                            } label: { Image(systemName: "bookmark") }
                                .buttonStyle(.borderless)
                                .help("Save to Prompt Library")
                        }
                    }
                    .padding(8)
                    .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 6))
                }
            }
        }
    }

    private func labeledRow<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.subheadline.weight(.semibold))
            content()
        }
    }

    private func scrapePrompts() async {
        guard let version, version.images.compactMap(\.prompt).count < 3 else { return }
        scrapedImages = (try? await client.images(modelVersionId: version.id)) ?? []
    }

    /// All prompts we've collected for the current version (inline + scraped).
    private var collectedPrompts: [String] {
        guard let version else { return [] }
        let inline = version.images.compactMap(\.prompt)
        let scraped = scrapedImages.compactMap(\.prompt)
        return (Array(NSOrderedSet(array: inline + scraped)) as? [String]) ?? inline
    }

    /// Persist a LoRA's trigger words + sample prompts alongside the downloaded
    /// file (a `<name>.civitai.json` sidecar) and into the Prompt Library, so
    /// they travel with the import automatically.
    private func saveLoraMetadata(next file: URL) {
        let triggers = version?.trainedWords ?? []
        let prompts = Array(collectedPrompts.prefix(8))
        let meta: [String: Any] = [
            "model": model.name,
            "modelId": model.id,
            "version": version?.name ?? "",
            "baseModel": version?.baseModel ?? "",
            "creator": model.creatorName ?? "",
            "triggerWords": triggers,
            "prompts": prompts,
            "url": modelWebURL?.absoluteString ?? "",
        ]
        let sidecar = file.deletingPathExtension().appendingPathExtension("civitai.json")
        if let data = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: sidecar)
        }
        // Push into the Prompt Library, tagged with the LoRA name + trigger words.
        if let promptLibrary {
            let tags = ["civitai", "lora", model.name] + triggers
            if !triggers.isEmpty {
                _ = promptLibrary.add(title: "\(model.name) — triggers",
                                      text: triggers.joined(separator: ", "), tags: tags)
            }
            for prompt in prompts.prefix(4) {
                _ = promptLibrary.add(title: model.name, text: prompt, tags: tags)
            }
        }
    }

    private func download(_ file: CivitAIFile) async {
        downloadProgress = 0
        statusMessage = nil
        do {
            let destination = try await client.download(file: file) { fraction in
                Task { @MainActor in downloadProgress = fraction }
            }
            downloadProgress = nil
            statusMessage = "Saved to \(destination.path). Rescanning library…"
            statusIsError = false
            saveLoraMetadata(next: destination)
            try? await engine.scanLoras()
            statusMessage = "Saved \(destination.lastPathComponent) — trigger words + \(min(collectedPrompts.count, 8)) sample prompts imported."
        } catch {
            downloadProgress = nil
            statusMessage = "Download failed: \(error.localizedDescription)"
            statusIsError = true
        }
    }
}
