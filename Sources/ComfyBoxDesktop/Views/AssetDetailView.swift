// AssetDetailView.swift — Full asset detail and metadata editing
//
// Displays a full-size image preview with all generation metadata.
// Provides editable fields for rating (0-5 stars), favorite toggle,
// and notes. Changes are saved back to DAMStore.

import SwiftUI

struct AssetDetailView: View {
    let asset: DAMAsset
    let thumbnailPath: String?
    let onUpdate: (DAMAsset) -> Void

    @State private var rating: Int
    @State private var isFavorite: Bool
    @State private var notes: String
    @State private var fullImage: NSImage?
    @State private var isLoadingImage: Bool = false
    @Environment(\.dismiss) private var dismiss

    init(asset: DAMAsset, thumbnailPath: String?, onUpdate: @escaping (DAMAsset) -> Void) {
        self.asset = asset
        self.thumbnailPath = thumbnailPath
        self.onUpdate = onUpdate
        self._rating = State(initialValue: asset.rating)
        self._isFavorite = State(initialValue: asset.favorite)
        self._notes = State(initialValue: "")
    }

    var body: some View {
        HSplitView {
            // Left: Full-size image
            imagePanel
                .frame(minWidth: 400)

            // Right: Metadata and controls
            metadataPanel
                .frame(minWidth: 280, maxWidth: 360)
        }
        .frame(minWidth: 800, minHeight: 500)
        .task {
            await loadFullImage()
        }
    }

    // MARK: - Image Panel

    private var imagePanel: some View {
        ZStack {
            Color(nsColor: .controlBackgroundColor)

            if let image = fullImage {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .padding(16)
            } else if isLoadingImage {
                VStack(spacing: 8) {
                    ProgressView()
                        .controlSize(.large)
                    Text("Loading image...")
                        .foregroundStyle(.secondary)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "photo")
                        .font(.system(size: 48))
                        .foregroundStyle(.tertiary)
                    Text("Image not available")
                        .foregroundStyle(.secondary)
                }
            }
        }
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
            metadataRow("Steps", value: asset.steps.map(String.init))
            metadataRow("Guidance", value: asset.guidance.map { String(format: "%.1f", $0) })
            metadataRow("Seed", value: asset.seed.map(String.init))
            metadataRow("Content Mode", value: asset.contentMode)
            metadataRow("Character", value: asset.characterName)
        }
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Prompt")
                .font(.subheadline)
                .fontWeight(.semibold)

            if let prompt = asset.prompt {
                Text(prompt)
                    .font(.caption)
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

    private func loadFullImage() async {
        isLoadingImage = true
        let path = asset.absolutePath
        let image = await Task.detached {
            NSImage(contentsOfFile: path)
        }.value
        await MainActor.run {
            fullImage = image
            isLoadingImage = false
        }
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
