// AssetDetailView.swift — Full asset detail and metadata editing
//
// Displays a full-size image preview with all generation metadata.
// Provides editable fields for rating (0-5 stars), favorite toggle,
// and notes. Changes are saved back to DAMStore.
//
// REMOTE ROWS. 1,278 of the converged catalog's 2,994 rows have no copy in
// this Mac's own library; they are the server's, reachable only through the
// smbfs mount at /Volumes/todd or the engine's stream route. This view used to
// open `absolutePath` with NSImage and nothing else, which meant two empty
// panes: any row whose path wasn't readable (mount down, server-only path),
// and — 535 rows of them — every VIDEO, because NSImage cannot open an mp4 and
// there was no player here at all. It now resolves through `AssetMediaSource`,
// the same local/remote decision the grid cell makes, plays video, streams from
// the engine when the bytes are only there, and says why when it can't.

import SwiftUI
import AVKit

struct AssetDetailView: View {
    /// The full set the card can navigate through (the gallery's filtered list).
    let assets: [DAMAsset]
    /// Thumbnail path for a given asset (fallback when the full image fails).
    let thumbnailProvider: (DAMAsset) -> String?
    /// Where the gallery decided this row's bytes are. Defaults to "its own
    /// path, nothing remote", which is exactly the old behaviour for callers
    /// that have no catalog.
    var mediaLocationProvider: (DAMAsset) -> AssetMediaLocation = {
        AssetMediaLocation(localPath: $0.absolutePath, remoteURL: nil)
    }
    let onUpdate: (DAMAsset) -> Void
    /// Open the given asset in the full-screen lightbox.
    var onFullScreen: ((DAMAsset) -> Void)?
    /// Send this asset's full recipe (prompt, params, LoRAs, content mode) to Generate.
    var onSendToGenerate: ((DAMAsset) -> Void)?

    @State private var currentIndex: Int
    @State private var rating: Int = 0
    @State private var isFavorite: Bool = false
    @State private var notes: String = ""
    @State private var fullImage: NSImage?
    @State private var isLoadingImage: Bool = false
    @State private var usedThumbnailFallback: Bool = false
    /// One player per asset, owned here. Building AVPlayer inline in the body
    /// (one per re-render) is a known crash in _AVKit_SwiftUI — the lightbox
    /// learned this the hard way; same rule applies here.
    @State private var player: AVPlayer?
    /// What went wrong fetching a server-side asset, shown instead of a blank pane.
    @State private var mediaError: String?
    @Environment(\.dismiss) private var dismiss
    @Environment(AppContentGate.self) private var contentGate

    /// Single-asset convenience (no navigation).
    init(asset: DAMAsset, thumbnailPath: String?, onUpdate: @escaping (DAMAsset) -> Void) {
        self.assets = [asset]
        self.thumbnailProvider = { _ in thumbnailPath }
        self.onUpdate = onUpdate
        self.onFullScreen = nil
        self.onSendToGenerate = nil
        self._currentIndex = State(initialValue: 0)
    }

    /// Navigable initializer used by the gallery.
    init(
        assets: [DAMAsset],
        index: Int,
        thumbnailProvider: @escaping (DAMAsset) -> String?,
        mediaLocationProvider: @escaping (DAMAsset) -> AssetMediaLocation = {
            AssetMediaLocation(localPath: $0.absolutePath, remoteURL: nil)
        },
        onUpdate: @escaping (DAMAsset) -> Void,
        onFullScreen: ((DAMAsset) -> Void)? = nil,
        onSendToGenerate: ((DAMAsset) -> Void)? = nil
    ) {
        self.assets = assets
        self.thumbnailProvider = thumbnailProvider
        self.mediaLocationProvider = mediaLocationProvider
        self.onUpdate = onUpdate
        self.onFullScreen = onFullScreen
        self.onSendToGenerate = onSendToGenerate
        self._currentIndex = State(initialValue: index)
    }

    /// The asset currently shown.
    private var asset: DAMAsset { assets[min(max(currentIndex, 0), assets.count - 1)] }
    private var thumbnailPath: String? { thumbnailProvider(asset) }

    /// Where this asset's bytes come from — gate first, then disk, then engine.
    private var source: AssetMediaSource {
        AssetMediaSource.resolve(mediaLocationProvider(asset), gateRevealed: contentGate.revealed)
    }

    /// Nil when Copy / Reveal can run; otherwise why they can't.
    private var localOnlyReason: String? { source.localOnlyReason }

    var body: some View {
        VStack(spacing: 0) {
            navigationBar
            Divider()
            HSplitView {
                // Left: Full-size image
                imagePanel
                    .frame(minWidth: 400)

                // Right: Metadata and controls
                metadataPanel
                    .frame(minWidth: 280, maxWidth: 360)
            }
        }
        .frame(minWidth: 800, minHeight: 500)
        // Keyed on the gate as well as the asset: closing the gate mid-session
        // must tear the loaded bytes (and a playing video) down, not merely
        // blur them, and opening it must then load what was withheld.
        .task(id: "\(asset.id)|\(contentGate.revealed)") {
            syncEditableState()
            await loadFullImage()
        }
        .onKeyPress(.leftArrow) { step(-1); return .handled }
        .onKeyPress(.rightArrow) { step(1); return .handled }
        .onKeyPress(keys: ["c"]) { press in
            guard press.modifiers.contains(.command) else { return .ignored }
            copyToClipboard()
            return .handled
        }
    }

    // MARK: - Navigation bar

    private var navigationBar: some View {
        HStack(spacing: 12) {
            Button { step(-1) } label: { Image(systemName: "chevron.left") }
                .disabled(currentIndex <= 0)
            Button { step(1) } label: { Image(systemName: "chevron.right") }
                .disabled(currentIndex >= assets.count - 1)
            Text("\(currentIndex + 1) / \(assets.count)")
                .font(.callout.monospacedDigit())
                .foregroundStyle(.secondary)
            Text(asset.filename)
                .font(.callout.weight(.medium))
                .lineLimit(1)
                .truncationMode(.middle)
            if source.isRemote {
                Label("On server", systemImage: "externaldrive.connected.to.line.below")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help("Streamed from the engine — no copy of this file on this Mac.")
            }
            Spacer()
            // Copy and Reveal both need a real file on this disk. For a
            // server-side row they would do nothing at all, so they say why
            // instead of failing silently.
            Button { copyToClipboard() } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            .disabled(localOnlyReason != nil)
            .help(localOnlyReason ?? "Copy image (⌘C)")
            Button { revealInFinder() } label: {
                Label("Reveal", systemImage: "magnifyingglass")
            }
            .disabled(localOnlyReason != nil)
            .help(localOnlyReason ?? "Reveal in Finder")
            if let onSendToGenerate {
                Button { onSendToGenerate(asset) } label: {
                    Label("Send to Generate", systemImage: "wand.and.stars")
                }
            }
            if let onFullScreen {
                Button { onFullScreen(asset) } label: {
                    Label("Full Screen", systemImage: "arrow.up.left.and.arrow.down.right")
                }
            }
            Button { dismiss() } label: { Image(systemName: "xmark.circle.fill") }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .padding(12)
    }

    private func step(_ delta: Int) {
        let next = currentIndex + delta
        guard assets.indices.contains(next) else { return }
        currentIndex = next
    }

    private func syncEditableState() {
        rating = asset.rating
        isFavorite = asset.favorite
        notes = ""
    }

    private func revealInFinder() {
        guard case .local(let path) = source else { return }
        NSWorkspace.shared.selectFile(path, inFileViewerRootedAtPath: "")
    }

    /// Copy the current image to the clipboard as both a file reference and,
    /// when it loads, its bitmap.
    private func copyToClipboard() {
        guard case .local(let path) = source else { return }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        var items: [NSPasteboardWriting] = [url as NSURL]
        if let image = fullImage ?? NSImage(contentsOf: url) { items.append(image) }
        pasteboard.writeObjects(items)
    }

    // MARK: - Image Panel

    private var imagePanel: some View {
        ZStack {
            Color(nsColor: .controlBackgroundColor)

            if let player {
                SafeVideoPlayer(player: player)
                    .aspectRatio(contentMode: .fit)
                    .padding(8)
            } else if let image = fullImage {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .padding(16)
                    .overlay(alignment: .bottom) {
                        if usedThumbnailFallback {
                            Text("Preview from thumbnail — original file not found")
                                .font(.caption)
                                .foregroundStyle(.orange)
                                .padding(6)
                                .background(.black.opacity(0.4), in: Capsule())
                                .padding(.bottom, 20)
                        }
                    }
            } else if isLoadingImage {
                VStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.large)
                    Text(source.isRemote
                         ? (asset.kind == "video" ? "Fetching video from server..." : "Loading from server...")
                         : "Loading image...")
                        .foregroundStyle(.secondary)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: asset.kind == "video" ? "film" : "photo")
                        .font(.system(size: 48))
                        .foregroundStyle(.tertiary)
                    Text(mediaError ?? "Image not available")
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 20)
                    Text(asset.absolutePath)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                        .padding(.horizontal, 20)
                }
            }
        }
        .contentGated(cornerRadius: 0)
    }

    // MARK: - Metadata Panel

    private var metadataPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // File info header
                fileInfoSection

                Divider()

                // Rating and favorite
                annotationSection

                Divider()

                // Generation parameters
                generationSection

                Divider()

                // Prompt
                promptSection

                // Vision caption & tags (from the local vision model)
                captionTagsSection

                Spacer()

                // Save button
                saveButton
            }
            .padding()
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var fileInfoSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(asset.filename)
                .font(.headline)
                .lineLimit(2)
                .truncationMode(.middle)

            HStack(spacing: 12) {
                if let w = asset.width, let h = asset.height {
                    Label("\(w) x \(h)", systemImage: "aspectratio")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Label(formattedFileSize, systemImage: "doc")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Text(formattedDate(asset.createdAt))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var annotationSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Annotation")
                .font(.subheadline)
                .fontWeight(.semibold)

            // Star rating
            HStack(spacing: 4) {
                Text("Rating")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                starRatingPicker
            }

            // Favorite toggle
            Toggle(isOn: $isFavorite) {
                Label("Favorite", systemImage: isFavorite ? "heart.fill" : "heart")
            }
            .toggleStyle(.switch)
        }
    }

    private var starRatingPicker: some View {
        HStack(spacing: 2) {
            ForEach(1...5, id: \.self) { star in
                Image(systemName: star <= rating ? "star.fill" : "star")
                    .foregroundStyle(star <= rating ? .yellow : .gray)
                    .font(.body)
                    .onTapGesture {
                        rating = star == rating ? 0 : star
                    }
            }
        }
    }

    private var generationSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Generation")
                .font(.subheadline)
                .fontWeight(.semibold)

            metadataRow("Model", value: asset.modelFamily)
            if let loras = lorasSummary { metadataRow("LoRAs", value: loras) }
            metadataRow("Steps", value: asset.steps.map(String.init))
            metadataRow("Guidance", value: asset.guidance.map { String(format: "%.1f", $0) })
            metadataRow("Seed", value: asset.seed.map(String.init))
            metadataRow("Content Mode", value: asset.contentMode)
            metadataRow("Character", value: asset.characterName)
        }
    }

    /// Vision caption + tags (read from Finder-aligned metadata). Populated by
    /// "Auto-caption & Tag" in the gallery.
    @ViewBuilder
    private var captionTagsSection: some View {
        let caption = FinderTags.caption(atPath: asset.absolutePath)
        let tags = FinderTags.textTags(atPath: asset.absolutePath)
        if caption != nil || !tags.isEmpty {
            Divider()
            VStack(alignment: .leading, spacing: 8) {
                Text("Caption & Tags").font(.subheadline).fontWeight(.semibold)
                if let caption {
                    Text(caption).font(.caption).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(8)
                        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                }
                if !tags.isEmpty {
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 5) {
                            ForEach(tags, id: \.self) { tag in
                                Text(tag)
                                    .font(.caption2)
                                    .padding(.horizontal, 7).padding(.vertical, 2)
                                    .background(.quaternary, in: Capsule())
                            }
                        }
                    }
                }
            }
        }
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prompt")
                .font(.subheadline)
                .fontWeight(.semibold)

            if let prompt = asset.prompt {
                GatedText(prompt, font: .caption)
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            } else {
                Text("No prompt recorded")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            if let negPrompt = asset.negativePrompt {
                Text("Negative Prompt")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.top, 4)
                Text(negPrompt)
                    .font(.caption)
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(Color(nsColor: .controlBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private var saveButton: some View {
        Button(action: saveChanges) {
            HStack {
                Image(systemName: "checkmark.circle")
                Text("Save Changes")
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
        }
        .buttonStyle(.borderedProminent)
        .disabled(!hasChanges)
    }

    // MARK: - Helpers

    private var hasChanges: Bool {
        rating != asset.rating || isFavorite != asset.favorite
    }

    private var formattedFileSize: String {
        let bytes = asset.fileSize
        if bytes < 1024 { return "\(bytes) B" }
        let kb = Double(bytes) / 1024
        if kb < 1024 { return String(format: "%.0f KB", kb) }
        let mb = kb / 1024
        return String(format: "%.1f MB", mb)
    }

    private func formattedDate(_ date: Date) -> String {
        let fmt = DateFormatter()
        fmt.dateStyle = .medium
        fmt.timeStyle = .short
        return fmt.string(from: date)
    }

    /// LoRAs read on demand from the asset's embedded metadata / sidecar.
    private var lorasSummary: String? {
        let loras = AssetIngestor.embeddedLoras(imagePath: asset.absolutePath)
        return loras.isEmpty ? nil : loras.joined(separator: ", ")
    }

    @ViewBuilder
    private func metadataRow(_ label: String, value: String?) -> some View {
        if let value = value {
            HStack {
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 90, alignment: .leading)
                Text(value)
                    .font(.caption)
                    .foregroundStyle(.primary)
                Spacer()
            }
        }
    }

    /// Load whatever this asset is, from wherever it is.
    ///
    /// Four outcomes, in the order `AssetMediaSource.resolve` decides them:
    /// gated (nothing is read — not disk, not network), local file, engine
    /// stream, or nothing at all.
    private func loadFullImage() async {
        // Tear the previous asset down first; a player left running would keep
        // playing over the next row (or past a closing gate).
        player?.pause()
        player = nil
        fullImage = nil
        usedThumbnailFallback = false
        mediaError = nil
        isLoadingImage = false

        let isVideo = asset.kind == "video"
        let thumb = thumbnailPath

        switch source {
        case .gated:
            // Rated G. Read nothing.
            return

        case .local(let path):
            isLoadingImage = true
            if isVideo {
                player = AVPlayer(url: URL(fileURLWithPath: path))
                isLoadingImage = false
                return
            }
            let (image, fellBack) = await Task.detached { () -> (NSImage?, Bool) in
                if let full = NSImage(contentsOfFile: path) { return (full, false) }
                // Original gone — show the cached thumbnail so the card isn't blank.
                if let thumb, let preview = NSImage(contentsOfFile: thumb) { return (preview, true) }
                return (nil, false)
            }.value
            fullImage = image
            usedThumbnailFallback = fellBack
            if image == nil { mediaError = "Image not available" }
            isLoadingImage = false

        case .remote(let url):
            isLoadingImage = true
            if isVideo {
                // Pulled to a temp file rather than streamed: the engine route
                // serves whole bodies with no Range support, and AVPlayer needs
                // ranges to play over HTTP.
                do {
                    let file = try await RemoteMediaCache.localCopy(of: url, filename: asset.filename)
                    guard !Task.isCancelled else { return }
                    player = AVPlayer(url: file)
                } catch {
                    mediaError = "Couldn't fetch this video from the server: \(error.localizedDescription)"
                }
                isLoadingImage = false
                return
            }
            let image = await Self.remoteImage(from: url)
            guard !Task.isCancelled else { return }
            if let image {
                fullImage = image
            } else if let thumb, let preview = NSImage(contentsOfFile: thumb) {
                fullImage = preview
                usedThumbnailFallback = true
            } else {
                mediaError = "Couldn't fetch this image from the server."
            }
            isLoadingImage = false

        case .missing:
            if let thumb, let preview = NSImage(contentsOfFile: thumb) {
                fullImage = preview
                usedThumbnailFallback = true
            } else {
                mediaError = "This asset's file isn't on this Mac, and no server copy is known."
            }
        }
    }

    /// Bytes from the engine's `/v1/gallery/file` — the same route (and the
    /// same 200-only rule) the grid's thumbnails use.
    private static func remoteImage(from url: URL) async -> NSImage? {
        guard let (data, response) = try? await URLSession.shared.data(from: url),
              (response as? HTTPURLResponse)?.statusCode ?? 200 == 200 else { return nil }
        return NSImage(data: data)
    }

    private func saveChanges() {
        let updated = DAMAsset(
            id: asset.id,
            kind: asset.kind,
            filename: asset.filename,
            absolutePath: asset.absolutePath,
            fileSize: asset.fileSize,
            sha256: asset.sha256,
            width: asset.width,
            height: asset.height,
            createdAt: asset.createdAt,
            modifiedAt: Date(),
            ingestedAt: asset.ingestedAt,
            orphaned: asset.orphaned,
            prompt: asset.prompt,
            negativePrompt: asset.negativePrompt,
            seed: asset.seed,
            steps: asset.steps,
            guidance: asset.guidance,
            modelFamily: asset.modelFamily,
            rating: rating,
            favorite: isFavorite,
            contentMode: asset.contentMode,
            characterName: asset.characterName
        )
        onUpdate(updated)
    }
}
