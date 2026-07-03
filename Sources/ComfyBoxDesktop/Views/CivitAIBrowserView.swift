// CivitAIBrowserView.swift — Browse, scrape, and download from CivitAI
//
// Search models/LoRAs (defaults to the Z-Image family), inspect versions
// with trigger words and sample-image prompts (copy or save to the Prompt
// Library), and download files straight into the local LoRA library
// (~/.comfybox/loras) followed by a server rescan. Works keyless; the API
// key in Settings unlocks auth-gated listings and downloads.

import SwiftUI

struct CivitAIBrowserView: View {
    @Bindable var engine: EngineService
    var promptLibrary: PromptLibraryStore?

    @State private var query: String = ""
    @State private var typeFilter: String = "LORA"
    @State private var baseModelFilter: String = "Z-Image"
    @State private var sort: CivitAIClient.SortOrder = .mostDownloaded
    @State private var results: [CivitAIModel] = []
    @State private var nextCursor: String?
    @State private var isLoading = false
    @State private var loadError: String?
    @State private var selected: CivitAIModel?

    private static let typeOptions = ["LORA", "Checkpoint", "All"]
    private static let baseModelOptions = ["Z-Image", "All"]

    private var client: CivitAIClient {
        CivitAIClient(apiKey: DesktopSettings.load().civitaiApiKey)
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
        HStack(spacing: 10) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search CivitAI…", text: $query)
                    .textFieldStyle(.plain)
                    .frame(maxWidth: 220)
                    .onSubmit { Task { await search(reset: true) } }
            }
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6))

            Picker("Type", selection: $typeFilter) {
                ForEach(Self.typeOptions, id: \.self) { Text($0).tag($0) }
            }
            .fixedSize()
            .onChange(of: typeFilter) { _, _ in Task { await search(reset: true) } }

            Picker("Base", selection: $baseModelFilter) {
                ForEach(Self.baseModelOptions, id: \.self) { Text($0).tag($0) }
            }
            .fixedSize()
            .onChange(of: baseModelFilter) { _, _ in Task { await search(reset: true) } }

            Picker("Sort", selection: $sort) {
                ForEach(CivitAIClient.SortOrder.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }
            .fixedSize()
            .onChange(of: sort) { _, _ in Task { await search(reset: true) } }

            Spacer()
            if isLoading { ProgressView().controlSize(.small) }
            Button { Task { await search(reset: true) } } label: { Image(systemName: "arrow.clockwise") }
                .buttonStyle(.borderless)
        }
        .padding(12)
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
                types: typeFilter == "All" ? [] : [typeFilter],
                baseModel: baseModelFilter == "All" ? nil : baseModelFilter,
                sort: sort,
                cursor: reset ? nil : nextCursor
            )
            results = reset ? page.items : results + page.items
            nextCursor = page.nextCursor
        } catch {
            loadError = "CivitAI request failed: \(error.localizedDescription)"
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
            try? await engine.scanLoras()
            statusMessage = "Saved to \(destination.lastPathComponent) and library rescanned."
        } catch {
            downloadProgress = nil
            statusMessage = "Download failed: \(error.localizedDescription)"
            statusIsError = true
        }
    }
}
