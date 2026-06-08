// ComparisonGridView.swift — Side-by-side image comparison
//
// Displays 2-4 selected images in a grid for A/B testing prompt
// variations. Shows metadata diff highlighting what changed between
// images (seed, prompt, LoRAs, steps, guidance).

import SwiftUI

struct ComparisonGridView: View {
    let store: DAMStore
    let ingestor: AssetIngestor

    @State private var selectedAssets: [DAMAsset] = []
    @State private var allAssets: [DAMAsset] = []
    @State private var showingPicker: Bool = true
    @State private var showMetadataDiff: Bool = true
    @State private var isLoading: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            // Toolbar
            comparisonToolbar
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color(nsColor: .windowBackgroundColor))

            Divider()

            if selectedAssets.count < 2 {
                pickerView
            } else {
                comparisonGrid
            }
        }
        .task {
            await loadAssets()
        }
    }

    // MARK: - Toolbar

    private var comparisonToolbar: some View {
        HStack(spacing: 12) {
            Text("Compare")
                .font(.headline)

            Spacer()

            if selectedAssets.count >= 2 {
                Toggle("Show Diff", isOn: $showMetadataDiff)
                    .toggleStyle(.switch)
                    .controlSize(.small)

                Button("Clear Selection") {
                    selectedAssets.removeAll()
                }
                .controlSize(.small)
            }

            Text("\(selectedAssets.count)/4 selected")
                .font(.caption)
                .foregroundStyle(.secondary)
                .monospacedDigit()
        }
    }

    // MARK: - Picker (select images to compare)

    private var pickerView: some View {
        VStack(spacing: 12) {
            if selectedAssets.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: "square.grid.2x2")
                        .font(.system(size: 48))
                        .foregroundStyle(.tertiary)
                    Text("Select 2-4 images to compare")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    Text("Click images below to add them to the comparison.")
                        .font(.body)
                        .foregroundStyle(.tertiary)
                }
                .padding(.top, 40)
            } else {
                // Show current selection as small thumbnails
                HStack(spacing: 8) {
                    ForEach(selectedAssets) { asset in
                        VStack {
                            thumbnailImage(for: asset)
                                .frame(width: 80, height: 80)
                                .clipShape(RoundedRectangle(cornerRadius: 6))
                            Text(asset.filename)
                                .font(.caption2)
                                .lineLimit(1)
                        }
                        .onTapGesture {
                            selectedAssets.removeAll { $0.id == asset.id }
                        }
                    }

                    if selectedAssets.count >= 2 {
                        Button(action: { /* comparison auto-shows at 2+ */ }) {
                            Label("Compare", systemImage: "arrow.right.circle.fill")
                        }
                        .buttonStyle(.borderedProminent)
                    }
                }
                .padding(.vertical, 8)
            }

            Divider()

            // Grid of all assets to pick from
            ScrollView {
                let columns = [GridItem(.adaptive(minimum: 120, maximum: 160), spacing: 8)]
                LazyVGrid(columns: columns, spacing: 8) {
                    ForEach(allAssets) { asset in
                        let isSelected = selectedAssets.contains { $0.id == asset.id }
                        PickerCell(
                            asset: asset,
                            thumbnailPath: ingestor.thumbnailPath(for: asset.id),
                            isSelected: isSelected
                        )
                        .onTapGesture {
                            toggleSelection(asset)
                        }
                    }
                }
                .padding(12)
            }
        }
    }

    // MARK: - Comparison Grid

    private var comparisonGrid: some View {
        VStack(spacing: 0) {
            // Images side by side
            GeometryReader { geometry in
                let columns = min(selectedAssets.count, 4)
                let cellWidth = (geometry.size.width - CGFloat(columns - 1) * 8 - 24) / CGFloat(columns)

                HStack(alignment: .top, spacing: 8) {
                    ForEach(selectedAssets) { asset in
                        VStack(spacing: 4) {
                            thumbnailImage(for: asset)
                                .frame(width: cellWidth, height: cellWidth)
                                .clipShape(RoundedRectangle(cornerRadius: 8))

                            Text(asset.filename)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        .frame(width: cellWidth)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.top, 8)
            }
            .frame(maxHeight: .infinity)

            // Metadata diff
            if showMetadataDiff {
                Divider()
                metadataDiffView
                    .frame(height: 200)
            }
        }
    }

    // MARK: - Metadata Diff

    private var metadataDiffView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                Text("Metadata Comparison")
                    .font(.subheadline.weight(.medium))
                    .padding(.horizontal, 12)
                    .padding(.top, 8)

                // Table-style diff
                Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                    // Header row
                    GridRow {
                        Text("Field")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ForEach(selectedAssets) { asset in
                            Text(asset.filename)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }

                    Divider()

                    // Prompt
                    diffRow(label: "Prompt") { asset in
                        asset.prompt ?? "-"
                    }

                    // Seed
                    diffRow(label: "Seed") { asset in
                        asset.seed.map { "\($0)" } ?? "random"
                    }

                    // Steps
                    diffRow(label: "Steps") { asset in
                        asset.steps.map { "\($0)" } ?? "-"
                    }

                    // Guidance
                    diffRow(label: "Guidance") { asset in
                        asset.guidance.map { String(format: "%.1f", $0) } ?? "-"
                    }

                    // Model
                    diffRow(label: "Model") { asset in
                        asset.modelFamily ?? "-"
                    }

                    // Content Mode
                    diffRow(label: "Mode") { asset in
                        asset.contentMode ?? "-"
                    }

                    // Character
                    diffRow(label: "Character") { asset in
                        asset.characterName ?? "-"
                    }

                    // Resolution
                    diffRow(label: "Resolution") { asset in
                        if let w = asset.width, let h = asset.height {
                            return "\(w)x\(h)"
                        }
                        return "-"
                    }
                }
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
            }
        }
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    private func diffRow(label: String, value: @escaping (DAMAsset) -> String) -> some View {
        GridRow {
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 70, alignment: .leading)

            ForEach(selectedAssets) { asset in
                let val = value(asset)
                let isDifferent = selectedAssets.contains { value($0) != val }
                Text(val)
                    .font(.caption)
                    .lineLimit(2)
                    .foregroundStyle(isDifferent ? .primary : .secondary)
                    .fontWeight(isDifferent ? .medium : .regular)
                    .background(isDifferent ? Color.yellow.opacity(0.1) : Color.clear)
            }
        }
    }

    // MARK: - Helpers

    private func loadAssets() async {
        isLoading = true
        do {
            allAssets = try await store.fetchAssets(limit: 200)
        } catch {
            // Non-fatal — empty grid shown.
        }
        isLoading = false
    }

    private func toggleSelection(_ asset: DAMAsset) {
        if let index = selectedAssets.firstIndex(where: { $0.id == asset.id }) {
            selectedAssets.remove(at: index)
        } else if selectedAssets.count < 4 {
            selectedAssets.append(asset)
        }
    }

    private func thumbnailImage(for asset: DAMAsset) -> some View {
        AsyncThumbnail(
            thumbnailPath: ingestor.thumbnailPath(for: asset.id),
            fallbackPath: asset.absolutePath
        )
    }
}

// MARK: - Picker Cell

private struct PickerCell: View {
    let asset: DAMAsset
    let thumbnailPath: String
    let isSelected: Bool

    @State private var thumbnail: NSImage?

    var body: some View {
        ZStack(alignment: .topTrailing) {
            VStack(spacing: 2) {
                ZStack {
                    Color(nsColor: .controlBackgroundColor)
                    if let thumb = thumbnail {
                        Image(nsImage: thumb)
                            .resizable()
                            .aspectRatio(contentMode: .fill)
                    } else {
                        Image(systemName: "photo")
                            .foregroundStyle(.tertiary)
                    }
                }
                .frame(height: 100)
                .clipShape(RoundedRectangle(cornerRadius: 6))

                Text(asset.filename)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            if isSelected {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.white, Color.accentColor)
                    .font(.title3)
                    .padding(4)
            }
        }
        .padding(4)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? Color.accentColor : Color.clear, lineWidth: 2)
        )
        .task {
            thumbnail = await Task.detached {
                NSImage(contentsOfFile: thumbnailPath) ?? NSImage(contentsOfFile: asset.absolutePath)
            }.value
        }
    }
}

// MARK: - Async Thumbnail Helper

struct AsyncThumbnail: View {
    let thumbnailPath: String
    let fallbackPath: String

    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Color(nsColor: .controlBackgroundColor)
            if let img = image {
                Image(nsImage: img)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Image(systemName: "photo")
                    .foregroundStyle(.tertiary)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .task {
            image = await Task.detached {
                NSImage(contentsOfFile: thumbnailPath) ?? NSImage(contentsOfFile: fallbackPath)
            }.value
        }
    }
}
