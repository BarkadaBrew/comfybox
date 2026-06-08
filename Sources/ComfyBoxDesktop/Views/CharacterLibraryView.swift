// CharacterLibraryView.swift — Character library browser
//
// Displays registered characters from the engine's character registry.
// Each character shows name, description, default LoRAs, and a sample
// prompt snippet. Tapping "Insert" appends the character's prompt
// snippet into the active generation prompt.

import SwiftUI

/// A character definition from the character registry.
public struct CharacterEntry: Identifiable, Sendable {
    public let id: String
    public let name: String
    public let description: String
    public let defaultLoras: [String]
    public let promptSnippet: String
    public let tags: [String]

    public init(
        id: String,
        name: String,
        description: String = "",
        defaultLoras: [String] = [],
        promptSnippet: String = "",
        tags: [String] = []
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.defaultLoras = defaultLoras
        self.promptSnippet = promptSnippet
        self.tags = tags
    }
}

struct CharacterLibraryView: View {
    var characters: [CharacterEntry]
    var onInsert: ((CharacterEntry) -> Void)?

    @State private var searchText: String = ""
    @State private var selectedCharacter: CharacterEntry?

    var body: some View {
        VStack(spacing: 0) {
            // Header with search
            HStack(spacing: 8) {
                Text("Characters")
                    .font(.headline)
                Spacer()
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search...", text: $searchText)
                        .textFieldStyle(.plain)
                        .frame(maxWidth: 160)
                }
                .padding(4)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)

            Divider()

            if filteredCharacters.isEmpty {
                emptyState
            } else {
                characterList
            }
        }
    }

    private var filteredCharacters: [CharacterEntry] {
        guard !searchText.isEmpty else { return characters }
        let query = searchText.lowercased()
        return characters.filter { entry in
            entry.name.lowercased().contains(query)
                || entry.description.lowercased().contains(query)
                || entry.tags.contains { $0.lowercased().contains(query) }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "person.2.crop.square.stack")
                .font(.system(size: 32))
                .foregroundStyle(.tertiary)
            if characters.isEmpty {
                Text("No characters registered")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Text("Characters are registered through the engine API.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            } else {
                Text("No matches for \"\(searchText)\"")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var characterList: some View {
        ScrollView {
            LazyVStack(spacing: 8) {
                ForEach(filteredCharacters) { entry in
                    CharacterCard(
                        entry: entry,
                        isSelected: selectedCharacter?.id == entry.id,
                        onInsert: { onInsert?(entry) }
                    )
                    .onTapGesture {
                        withAnimation(.easeInOut(duration: 0.15)) {
                            selectedCharacter = selectedCharacter?.id == entry.id ? nil : entry
                        }
                    }
                }
            }
            .padding(12)
        }
    }
}

// MARK: - Character Card

private struct CharacterCard: View {
    let entry: CharacterEntry
    let isSelected: Bool
    var onInsert: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Name and insert button
            HStack {
                Image(systemName: "person.crop.circle")
                    .font(.title2)
                    .foregroundStyle(Color.accentColor)

                VStack(alignment: .leading, spacing: 2) {
                    Text(entry.name)
                        .font(.subheadline.weight(.semibold))
                    if !entry.description.isEmpty {
                        Text(entry.description)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(isSelected ? nil : 2)
                    }
                }

                Spacer()

                Button(action: onInsert) {
                    Label("Insert", systemImage: "text.insert")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }

            // Expanded details
            if isSelected {
                VStack(alignment: .leading, spacing: 6) {
                    // Tags
                    if !entry.tags.isEmpty {
                        FlowLayout(spacing: 4) {
                            ForEach(entry.tags, id: \.self) { tag in
                                Text(tag)
                                    .font(.caption2)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.accentColor.opacity(0.15))
                                    .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }
                    }

                    // Default LoRAs
                    if !entry.defaultLoras.isEmpty {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Default LoRAs")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            ForEach(entry.defaultLoras, id: \.self) { lora in
                                Text("  \(lora)")
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }
                        }
                    }

                    // Prompt snippet
                    if !entry.promptSnippet.isEmpty {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Prompt Snippet")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Text(entry.promptSnippet)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .padding(6)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color(nsColor: .controlBackgroundColor))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                        }
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(isSelected ? Color.accentColor.opacity(0.08) : Color(nsColor: .controlBackgroundColor).opacity(0.5))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(isSelected ? Color.accentColor.opacity(0.3) : Color.clear, lineWidth: 1)
        )
    }
}

// MARK: - Simple Flow Layout (for tags)

struct FlowLayout: Layout {
    var spacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let result = layout(proposal: proposal, subviews: subviews)
        return result.size
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = layout(proposal: proposal, subviews: subviews)
        for (index, position) in result.positions.enumerated() {
            subviews[index].place(
                at: CGPoint(x: bounds.minX + position.x, y: bounds.minY + position.y),
                proposal: .unspecified
            )
        }
    }

    private func layout(proposal: ProposedViewSize, subviews: Subviews) -> (size: CGSize, positions: [CGPoint]) {
        let maxWidth = proposal.width ?? .infinity
        var positions: [CGPoint] = []
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth, x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            positions.append(CGPoint(x: x, y: y))
            rowHeight = max(rowHeight, size.height)
            x += size.width + spacing
        }

        return (CGSize(width: maxWidth, height: y + rowHeight), positions)
    }
}
