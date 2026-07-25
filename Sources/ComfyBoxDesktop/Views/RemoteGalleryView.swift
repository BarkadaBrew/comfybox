// RemoteGalleryView.swift — Browse + pull images from a remote ComfyBox server.

import SwiftUI

struct RemoteGalleryView: View {
    @State private var remote: RemoteGalleryService
    var ingestor: AssetIngestor?
    @State private var status: String?
    @State private var lightboxIndex: Int?

    private let columns = [GridItem(.adaptive(minimum: 180), spacing: 10)]

    init(engine: EngineService, ingestor: AssetIngestor?) {
        _remote = State(initialValue: RemoteGalleryService(engine: engine))
        self.ingestor = ingestor
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
        }
        .navigationTitle("Remote Gallery")
        .task { await remote.load() }
        .overlay {
            if let idx = lightboxIndex {
                RemoteGalleryLightbox(
                    assets: remote.assets,
                    index: idx,
                    remote: remote,
                    onSave: { asset in Task { await pull(asset) } },
                    onIndexChange: { lightboxIndex = $0 },
                    onClose: { lightboxIndex = nil }
                )
                .transition(.opacity)
            }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Remote Gallery").font(.headline)
                Text(remote.baseURL + (remote.isLocalServer ? " (this Mac)" : ""))
                    .font(.caption.monospaced()).foregroundStyle(.secondary)
            }
            Spacer()
            if let status { Text(status).font(.caption).foregroundStyle(.green) }
            if remote.isLoading { ProgressView().controlSize(.small) }
            Button { Task { await remote.load() } } label: { Label("Refresh", systemImage: "arrow.clockwise") }
                .controlSize(.small)
        }
        .padding(12)
    }

    @ViewBuilder private var content: some View {
        if let err = remote.error {
            ContentUnavailableView("Couldn't load remote gallery", systemImage: "wifi.exclamationmark", description: Text(err))
        } else if remote.assets.isEmpty && !remote.isLoading {
            ContentUnavailableView("No media on the server", systemImage: "photo.on.rectangle")
        } else {
            ScrollView {
                LazyVGrid(columns: columns, spacing: 10) {
                    ForEach(remote.assets) { asset in
                        cell(asset)
                    }
                }
                .padding(12)
            }
        }
    }

    private func cell(_ asset: RemoteGalleryService.RemoteAsset) -> some View {
        VStack(spacing: 4) {
            ZStack {
                RoundedRectangle(cornerRadius: 8).fill(.quaternary.opacity(0.3))
                if asset.kind == "video" {
                    Image(systemName: "film").font(.largeTitle).foregroundStyle(.secondary)
                } else {
                    AsyncImage(url: remote.fileURL(for: asset.path)) { phase in
                        switch phase {
                        case .success(let img): img.resizable().scaledToFill()
                        case .failure: Image(systemName: "exclamationmark.triangle").foregroundStyle(.orange)
                        default: ProgressView().controlSize(.small)
                        }
                    }
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                }
            }
            .frame(height: 160)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .contentShape(Rectangle())
            .onTapGesture {
                lightboxIndex = remote.assets.firstIndex(where: { $0.id == asset.id })
            }
            .contentGated(cornerRadius: 8)
            GatedText(asset.filename, font: .caption2).lineLimit(1)
            if !remote.isLocalServer, ingestor != nil {
                Button { Task { await pull(asset) } } label: {
                    Label("Save locally", systemImage: "square.and.arrow.down").font(.caption2)
                }.controlSize(.small).buttonStyle(.borderless)
            }
        }
        .contextMenu {
            if ingestor != nil { Button("Save to local gallery") { Task { await pull(asset) } } }
        }
    }

    private func pull(_ asset: RemoteGalleryService.RemoteAsset) async {
        let dir = DesktopSettings.load().outputDirectory
        do {
            let local = try await remote.pull(asset, to: dir)
            try? await ingestor?.ingestFile(at: local)
            status = "Saved \(asset.filename)"
        } catch {
            status = nil
            remote.error = "Save failed: \(error.localizedDescription)"
        }
    }
}
