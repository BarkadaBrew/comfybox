// CharactersView.swift — Full character management (list, create, edit, delete)
// against the /v1/characters backend. Ports the Electron Characters view.

import SwiftUI

struct CharactersView: View {
    @Bindable var engine: EngineService

    @State private var characters: [CharacterEntry] = []
    @State private var search: String = ""
    @State private var editing: CharacterEntry?
    @State private var isNew: Bool = false
    @State private var loadError: String?
    @State private var isLoading = false

    private var filtered: [CharacterEntry] {
        guard !search.isEmpty else { return characters }
        let q = search.lowercased()
        return characters.filter {
            $0.name.lowercased().contains(q) || $0.displayDescription.lowercased().contains(q)
                || $0.tags.contains { $0.lowercased().contains(q) }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if let loadError {
                errorBar(loadError)
            }
            if filtered.isEmpty {
                emptyState
            } else {
                list
            }
        }
        .navigationTitle("Characters")
        .task { await reload() }
        .onChange(of: engine.connectionState.isConnected) { _, connected in
            // The connection is established asynchronously after launch; reload once it's up.
            if connected { Task { await reload() } }
        }
        .sheet(item: $editing) { entry in
            CharacterEditor(
                original: entry,
                isNew: isNew,
                onSave: { updated in Task { await save(updated) } },
                onCancel: { editing = nil }
            )
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            HStack {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Search characters…", text: $search).textFieldStyle(.plain).frame(maxWidth: 220)
            }
            .padding(6)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
            Spacer()
            if isLoading { ProgressView().controlSize(.small) }
            Button { Task { await reload() } } label: { Image(systemName: "arrow.clockwise") }
                .buttonStyle(.borderless)
            Button {
                isNew = true
                editing = CharacterEntry(id: "", name: "")
            } label: { Label("New Character", systemImage: "plus") }
                .buttonStyle(.borderedProminent)
                .disabled(!engine.connectionState.isConnected)
        }
        .padding(12)
    }

    private var list: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                let characters = filtered.filter { $0.kind != "scene" }
                let scenes = filtered.filter { $0.kind == "scene" }
                if !characters.isEmpty {
                    groupHeader("Characters", count: characters.count)
                    ForEach(characters) { row($0) }
                }
                if !scenes.isEmpty {
                    groupHeader("Scenes", count: scenes.count)
                        .padding(.top, characters.isEmpty ? 0 : 10)
                    ForEach(scenes) { row($0) }
                }
            }
            .padding(12)
        }
    }

    private func groupHeader(_ title: String, count: Int) -> some View {
        HStack(spacing: 6) {
            Text(title).font(.headline)
            Text("\(count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 2)
    }

    private func row(_ c: CharacterEntry) -> some View {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: c.kind == "scene" ? "photo.artframe" : "person.crop.square")
                            .font(.title2).foregroundStyle(.secondary).frame(width: 28)
                        VStack(alignment: .leading, spacing: 4) {
                            Text(c.name).font(.headline)
                            if !c.displayDescription.isEmpty {
                                Text(c.displayDescription).font(.caption).foregroundStyle(.secondary)
                                    .lineLimit(2)
                            }
                            if !c.tags.isEmpty {
                                Text(c.tags.joined(separator: " · ")).font(.caption2).foregroundStyle(.tertiary)
                            }
                            if !c.defaultLoras.isEmpty {
                                Label("\(c.defaultLoras.count) LoRA\(c.defaultLoras.count == 1 ? "" : "s")",
                                      systemImage: "square.stack.3d.up")
                                    .font(.caption2).foregroundStyle(.tertiary)
                            }
                        }
                        Spacer()
                        Button { isNew = false; editing = c } label: { Image(systemName: "pencil") }
                            .buttonStyle(.borderless)
                        Button(role: .destructive) { Task { await delete(c) } } label: { Image(systemName: "trash") }
                            .buttonStyle(.borderless)
                    }
                    .padding(12)
                    .background(.quaternary.opacity(0.35), in: RoundedRectangle(cornerRadius: 8))
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "person.2.crop.square.stack").font(.system(size: 32)).foregroundStyle(.tertiary)
            Text(search.isEmpty ? "No characters yet" : "No matches for \"\(search)\"")
                .font(.subheadline).foregroundStyle(.secondary)
            if search.isEmpty {
                Text("Create one with “New Character”.").font(.caption).foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity).padding()
    }

    private func errorBar(_ text: String) -> some View {
        Text(text).font(.caption).foregroundStyle(.red)
            .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal, 12).padding(.vertical, 6)
    }

    // MARK: - Actions

    private func reload() async {
        guard engine.connectionState.isConnected else { loadError = "Connect to the server to manage characters."; return }
        isLoading = true; defer { isLoading = false }
        loadError = nil
        characters = await engine.fetchCharacters().sorted { $0.name.lowercased() < $1.name.lowercased() }
    }

    private func save(_ entry: CharacterEntry) async {
        do {
            try await engine.saveCharacter(entry)
            editing = nil
            await reload()
        } catch {
            loadError = "Save failed: \(error.localizedDescription)"
        }
    }

    private func delete(_ entry: CharacterEntry) async {
        do {
            try await engine.deleteCharacter(id: entry.id)
            await reload()
        } catch {
            loadError = "Delete failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Editor

private struct CharacterEditor: View {
    let original: CharacterEntry
    let isNew: Bool
    let onSave: (CharacterEntry) -> Void
    let onCancel: () -> Void

    @State private var name: String
    @State private var kind: String
    @State private var baseText: String
    @State private var bananaText: String
    @State private var avocadoText: String
    @State private var promptSnippet: String
    @State private var tagsText: String
    @State private var lorasText: String

    init(original: CharacterEntry, isNew: Bool, onSave: @escaping (CharacterEntry) -> Void, onCancel: @escaping () -> Void) {
        self.original = original
        self.isNew = isNew
        self.onSave = onSave
        self.onCancel = onCancel
        _name = State(initialValue: original.name)
        _kind = State(initialValue: original.kind)
        // The canonical text lives in the base tier; legacy entries only have
        // the flat description. Edit whichever exists, save into `base`.
        _baseText = State(initialValue: original.displayDescription)
        _bananaText = State(initialValue: original.banana ?? "")
        _avocadoText = State(initialValue: original.avocado ?? "")
        _promptSnippet = State(initialValue: original.promptSnippet)
        _tagsText = State(initialValue: original.tags.joined(separator: ", "))
        _lorasText = State(initialValue: original.defaultLoras.joined(separator: ", "))
    }

    private var trimmedName: String { name.trimmingCharacters(in: .whitespacesAndNewlines) }

    /// Existing characters keep their id; new ones derive a slug from the name.
    private var resolvedId: String {
        if !original.id.isEmpty { return original.id }
        return trimmedName.lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .filter { $0.isLetter || $0.isNumber || $0 == "-" }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(isNew ? "New Character" : "Edit Character").font(.headline).padding()
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    labeled("Name") {
                        TextField("Name", text: $name)
                            .textFieldStyle(.roundedBorder)
                    }
                    labeled("Type") {
                        Picker("Type", selection: $kind) {
                            Label("Character", systemImage: "person.crop.square").tag("character")
                            Label("Scene", systemImage: "photo.artframe").tag("scene")
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: 260)
                    }
                    labeled("Appearance — always used") {
                        editor($baseText, minHeight: 140)
                    }
                    labeled("Suggestive additions — banana mode") {
                        editor($bananaText, minHeight: 60)
                    }
                    labeled("Explicit additions — avocado mode") {
                        editor($avocadoText, minHeight: 60)
                    }
                    labeled("Prompt snippet") {
                        editor($promptSnippet, minHeight: 48)
                    }
                    labeled("Tags (comma-separated)") {
                        TextField("tag1, tag2", text: $tagsText)
                            .textFieldStyle(.roundedBorder)
                    }
                    labeled("Default LoRAs (comma-separated filenames)") {
                        TextField("lora.safetensors", text: $lorasText)
                            .textFieldStyle(.roundedBorder)
                    }
                }
                .padding(16)
            }
            Divider()
            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                Button("Save") {
                    let base = baseText.trimmingCharacters(in: .whitespacesAndNewlines)
                    let banana = bananaText.trimmingCharacters(in: .whitespacesAndNewlines)
                    let avocado = avocadoText.trimmingCharacters(in: .whitespacesAndNewlines)
                    let entry = CharacterEntry(
                        id: resolvedId,
                        name: trimmedName,
                        kind: kind,
                        // The edited text is saved as the base tier (which takes
                        // precedence everywhere); the flat description passes
                        // through untouched for legacy compatibility.
                        description: original.description,
                        base: base,
                        banana: banana.isEmpty ? nil : banana,
                        avocado: avocado.isEmpty ? nil : avocado,
                        defaultLoras: splitCSV(lorasText),
                        promptSnippet: promptSnippet.trimmingCharacters(in: .whitespacesAndNewlines),
                        negativePrompt: original.negativePrompt,
                        triggerWords: original.triggerWords,
                        tags: splitCSV(tagsText)
                    )
                    onSave(entry)
                }
                .buttonStyle(.borderedProminent)
                .disabled(trimmedName.isEmpty)
            }
            .padding()
        }
        .frame(minWidth: 560, idealWidth: 640, maxWidth: 900,
               minHeight: 560, idealHeight: 700, maxHeight: 1000)
    }

    /// A caption label above a control.
    private func labeled<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            content()
        }
    }

    /// A bordered, scrollable multi-line text editor for long prose fields.
    private func editor(_ text: Binding<String>, minHeight: CGFloat) -> some View {
        TextEditor(text: text)
            .font(.body)
            .scrollContentBackground(.hidden)
            .padding(6)
            .frame(minHeight: minHeight, maxHeight: minHeight * 2.5)
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
            )
    }

    private func splitCSV(_ s: String) -> [String] {
        s.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    }
}
