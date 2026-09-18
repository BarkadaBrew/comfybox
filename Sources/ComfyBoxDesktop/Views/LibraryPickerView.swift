// LibraryPickerView.swift — the one picker every surface opens
// (PRD docs/PRD-creative-library.md, L4).
//
// One sheet, filtered to the kinds the caller wants: a template for the prompt,
// an outfit or wardrobe piece for a slot, a scene component to insert, a look
// to apply. Its primary action is USE, not "view" — the PRD's guard against a
// library that becomes a museum.

import SwiftUI

public struct LibraryPickerView: View {
    public enum Mode: Equatable {
        case templates
        case components(kind: String?)
        case wardrobe
        case outfits
        case looks
        case recipes

        var kinds: [String] {
            switch self {
            case .templates: return ["template"]
            case .components: return ["component"]
            case .wardrobe: return ["wardrobe_item"]
            case .outfits: return ["outfit"]
            case .looks: return ["look"]
            case .recipes: return ["recipe"]
            }
        }

        var title: String {
            switch self {
            case .templates: return "Templates"
            case .components(let kind): return kind.map { $0.capitalized + "s" } ?? "Components"
            case .wardrobe: return "Wardrobe"
            case .outfits: return "Outfits"
            case .looks: return "Looks"
            case .recipes: return "Recipes"
            }
        }

        var componentKind: String? {
            if case .components(let kind) = self { return kind }
            return nil
        }
    }

    let mode: Mode
    let client: LibraryClient
    let onUse: (LibraryItemDTO) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var search = ""
    @State private var selectedCategory: String?
    @State private var selectedFacets: [String: String] = [:]
    @State private var favoritesOnly = false
    @State private var order = "recent"

    public init(
        mode: Mode, client: LibraryClient, onUse: @escaping (LibraryItemDTO) -> Void
    ) {
        self.mode = mode
        self.client = client
        self.onUse = onUse
    }

    private var spec: LibraryQuerySpec {
        var spec = LibraryQuerySpec(kinds: mode.kinds, text: search.isEmpty ? nil : search)
        spec.componentKind = mode.componentKind
        spec.category = selectedCategory
        spec.facets = selectedFacets.mapValues { [$0] }
        spec.favoritesOnly = favoritesOnly
        spec.order = order
        return spec
    }

    /// Wardrobe browses by its own category tree; everything else by facets.
    private var categories: [String] {
        Array(Set(client.items.compactMap(\.category))).sorted()
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if client.isLoading && client.items.isEmpty {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if client.items.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .frame(minWidth: 520, minHeight: 420)
        .task { await reload() }
        .onChange(of: search) { _, _ in Task { await reload() } }
        .onChange(of: selectedCategory) { _, _ in Task { await reload() } }
        .onChange(of: favoritesOnly) { _, _ in Task { await reload() } }
        .onChange(of: order) { _, _ in Task { await reload() } }
    }

    private var header: some View {
        VStack(spacing: 8) {
            HStack {
                Text(mode.title).font(.headline)
                Spacer()
                Button("Done") { dismiss() }.keyboardShortcut(.cancelAction)
            }
            HStack(spacing: 8) {
                TextField("Search", text: $search)
                    .textFieldStyle(.roundedBorder)
                Toggle(isOn: $favoritesOnly) { Image(systemName: "star") }
                    .toggleStyle(.button)
                    .help("Favourites only")
                Picker("", selection: $order) {
                    Text("Recent").tag("recent")
                    Text("Top rated").tag("rating")
                    Text("Most used").tag("use_count")
                    Text("Name").tag("name")
                }
                .labelsHidden()
                .frame(width: 130)
            }
            if mode == .wardrobe, !categories.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        chip(title: "All", isOn: selectedCategory == nil) { selectedCategory = nil }
                        ForEach(categories, id: \.self) { category in
                            chip(title: category, isOn: selectedCategory == category) {
                                selectedCategory = selectedCategory == category ? nil : category
                            }
                        }
                    }
                }
            }
        }
        .padding(12)
    }

    private func chip(title: String, isOn: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title).font(.caption)
                .padding(.horizontal, 8).padding(.vertical, 3)
                .background(isOn ? Color.accentColor.opacity(0.25) : Color.secondary.opacity(0.12))
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "tray").font(.largeTitle).foregroundStyle(.secondary)
            Text("Nothing here yet").font(.headline)
            Text("Import a pack, or save something from a render you liked.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var list: some View {
        List(client.items) { item in
            HStack(alignment: .top, spacing: 10) {
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(item.name).font(.body).lineLimit(1)
                        if item.favorite { Image(systemName: "star.fill").font(.caption2).foregroundStyle(.yellow) }
                        if item.rating > 0 {
                            Text(String(repeating: "★", count: item.rating))
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                    if !item.subtitle.isEmpty {
                        Text(item.subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                    }
                    if !item.value.isEmpty {
                        Text(item.value).font(.caption2).foregroundStyle(.tertiary).lineLimit(2)
                    }
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 4) {
                    Button("Use") {
                        client.markUsed(item.id)
                        onUse(item)
                        dismiss()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.small)
                    if item.useCount > 0 {
                        Text("used \(item.useCount)×").font(.caption2).foregroundStyle(.tertiary)
                    }
                }
            }
            .contentShape(Rectangle())
            .onTapGesture(count: 2) {
                client.markUsed(item.id)
                onUse(item)
                dismiss()
            }
        }
        .listStyle(.inset)
    }

    private func reload() async {
        await client.search(spec)
    }
}
