// PromptLibraryView.swift — Saved prompts + render history
//
// Two sections: prompts the user explicitly saved (PromptLibraryStore,
// ~/.comfybox/prompt-library.json) and a history of distinct prompts mined
// from the DAM with use counts. Either kind can be inserted into the
// Generate tab, copied, or (for history) promoted into the saved library.

import SwiftUI

struct PromptLibraryView: View {
    @Bindable var library: PromptLibraryStore
    var store: DAMStore?
    /// Insert a prompt into the Generate tab (switches tabs).
    var onInsert: ((String) -> Void)?

    @State private var search: String = ""
    @State private var history: [(prompt: String, count: Int, lastUsed: Date)] = []
    @State private var editing: PromptLibraryEntry?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    let saved = filteredSaved
                    if !saved.isEmpty {
                        sectionHeader("Saved", count: saved.count)
                        ForEach(saved) { savedRow($0) }
                    }
                    let hist = filteredHistory
                    if !hist.isEmpty {
                        sectionHeader("History", count: hist.count)
                            .padding(.top, saved.isEmpty ? 0 : 10)
                        ForEach(hist, id: \.prompt) { historyRow($0) }
                    }
                    if saved.isEmpty && hist.isEmpty {
                        emptyState
                    }
                }
                .padding(12)
            }
        }
        .navigationTitle("Prompt Library")
        .task { await loadHistory() }
        .sheet(item: $editing) { entry in
            PromptEditorSheet(
                original: entry,
                onSave: { library.update($0); editing = nil },
                onCancel: { editing = nil }
            )
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search prompts…", text: $search).textFieldStyle(.plain).frame(maxWidth: 240)
            }
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
            Spacer()
            Button { Task { await loadHistory() } } label: { Image(systemName: "arrow.clockwise") }
                .buttonStyle(.borderless)
            Button {
                editing = library.add(text: "")
            } label: { Label("New Prompt", systemImage: "plus") }
                .buttonStyle(.borderedProminent)
        }
        .padding(12)
    }

    private var filteredSaved: [PromptLibraryEntry] {
        guard !search.isEmpty else { return library.entries }
        let q = search.lowercased()
        return library.entries.filter {
            $0.title.lowercased().contains(q) || $0.text.lowercased().contains(q)
                || $0.tags.contains { $0.lowercased().contains(q) }
        }
    }

    private var filteredHistory: [(prompt: String, count: Int, lastUsed: Date)] {
        // Hide prompts already promoted into the saved library.
        let savedTexts = Set(library.entries.map(\.text))
        let base = history.filter { !savedTexts.contains($0.prompt) }
        guard !search.isEmpty else { return base }
        let q = search.lowercased()
        return base.filter { $0.prompt.lowercased().contains(q) }
    }

    private func loadHistory() async {
        guard let store else { return }
        history = (try? await store.promptHistory(limit: 200)) ?? []
    }

    // MARK: - Rows

    private func sectionHeader(_ title: String, count: Int) -> some View {
        HStack(spacing: 6) {
            Text(title).font(.headline)
            Text("\(count)").font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 2)
    }

    private func savedRow(_ entry: PromptLibraryEntry) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "bookmark.fill")
                .font(.callout).foregroundStyle(Color.accentColor).frame(width: 20)
            VStack(alignment: .leading, spacing: 3) {
                Text(entry.displayTitle).font(.callout.weight(.medium))
                Text(entry.text).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                HStack(spacing: 8) {
                    if !entry.tags.isEmpty {
                        Text(entry.tags.joined(separator: " · ")).font(.caption2).foregroundStyle(.tertiary)
                    }
                    if entry.useCount > 0 {
                        Text("used \(entry.useCount)×").font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
                    }
                }
            }
            Spacer()
            rowButtons(
                insert: {
                    library.recordUse(id: entry.id)
                    onInsert?(entry.text)
                },
                copy: { copyToPasteboard(entry.text) }
            )
            Button { editing = entry } label: { Image(systemName: "pencil") }
                .buttonStyle(.borderless)
            Button(role: .destructive) { library.delete(id: entry.id) } label: { Image(systemName: "trash") }
                .buttonStyle(.borderless)
        }
        .padding(10)
        .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
    }

    private func historyRow(_ item: (prompt: String, count: Int, lastUsed: Date)) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "clock")
                .font(.callout).foregroundStyle(.secondary).frame(width: 20)
            VStack(alignment: .leading, spacing: 3) {
                Text(item.prompt).font(.caption).lineLimit(3)
                Text("\(item.count) render\(item.count == 1 ? "" : "s") · \(item.lastUsed.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption2.monospacedDigit()).foregroundStyle(.tertiary)
            }
            Spacer()
            rowButtons(
                insert: { onInsert?(item.prompt) },
                copy: { copyToPasteboard(item.prompt) }
            )
            Button {
                library.add(text: item.prompt)
            } label: { Image(systemName: "bookmark") }
                .buttonStyle(.borderless)
                .help("Save to library")
        }
        .padding(10)
        .background(.quaternary.opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
    }

    @ViewBuilder
    private func rowButtons(insert: @escaping () -> Void, copy: @escaping () -> Void) -> some View {
        Button("Insert", action: insert)
            .controlSize(.small)
            .buttonStyle(.borderedProminent)
        Button { copy() } label: { Image(systemName: "doc.on.doc") }
            .buttonStyle(.borderless)
            .help("Copy prompt")
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "text.book.closed").font(.system(size: 32)).foregroundStyle(.tertiary)
            Text(search.isEmpty ? "No prompts yet" : "No matches for \"\(search)\"")
                .font(.subheadline).foregroundStyle(.secondary)
            if search.isEmpty {
                Text("Prompts from your renders appear as history; save the keepers.")
                    .font(.caption).foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity).padding(.top, 60)
    }

    private func copyToPasteboard(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

// MARK: - Editor

private struct PromptEditorSheet: View {
    let original: PromptLibraryEntry
    let onSave: (PromptLibraryEntry) -> Void
    let onCancel: () -> Void

    @State private var title: String
    @State private var text: String
    @State private var tagsText: String

    init(original: PromptLibraryEntry,
         onSave: @escaping (PromptLibraryEntry) -> Void, onCancel: @escaping () -> Void) {
        self.original = original
        self.onSave = onSave
        self.onCancel = onCancel
        _title = State(initialValue: original.title)
        _text = State(initialValue: original.text)
        _tagsText = State(initialValue: original.tags.joined(separator: ", "))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Edit Prompt").font(.headline).padding()
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                TextField("Title (optional)", text: $title)
                    .textFieldStyle(.roundedBorder)
                TextEditor(text: $text)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                    .padding(6)
                    .frame(minHeight: 140)
                    .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color(nsColor: .separatorColor), lineWidth: 1))
                TextField("Tags (comma-separated)", text: $tagsText)
                    .textFieldStyle(.roundedBorder)
            }
            .padding(16)
            Divider()
            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                Button("Save") {
                    var entry = original
                    entry.title = title.trimmingCharacters(in: .whitespacesAndNewlines)
                    entry.text = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    entry.tags = tagsText.split(separator: ",")
                        .map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
                    onSave(entry)
                }
                .buttonStyle(.borderedProminent)
                .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            .padding()
        }
        .frame(minWidth: 480, idealWidth: 540, minHeight: 380)
    }
}
