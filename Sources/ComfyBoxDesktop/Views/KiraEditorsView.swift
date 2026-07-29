// KiraEditorsView.swift — editable Kira-tab surfaces (Todd 2026-07-27):
//
//  - HerNowEditor:       pin mood valence/energy + set/clear the arcPhase
//                        override (PUT /v1/kira/state/now).
//  - KiraCharacterCard:  the tiered image-prompt description, fully editable
//                        (PUT/DELETE /v1/kira/character; override-aware).
//  - KiraLorebookCard:   canonical facts injected into her context by keyword
//                        match (existing /v1/lorebook/* CRUD on the daemon).
//
// All three are THIN CLIENTS of the daemon API, same as the rest of the tab —
// no state of their own beyond edit drafts.

import SwiftUI

// MARK: - Her Now editor

struct HerNowEditor: View {
    @Bindable var client: KiraClient
    @State private var valence: Double = 0
    @State private var energy: Double = 0
    @State private var arcPhaseDraft: String = ""
    @State private var loadedFor: KiraNowState?

    var body: some View {
        if let now = client.nowState {
            VStack(alignment: .leading, spacing: 6) {
                Divider()
                Text("Adjust — pins her current mood; it drifts back to baseline on the normal half-life.")
                    .font(.caption2).foregroundStyle(.tertiary)
                HStack(spacing: 10) {
                    Text("valence").font(.caption).foregroundStyle(.secondary)
                    Slider(value: $valence, in: -1...1, step: 0.05).frame(width: 140)
                    Text(String(format: "%+.2f", valence)).font(.caption.monospacedDigit())
                    Text("energy").font(.caption).foregroundStyle(.secondary)
                    Slider(value: $energy, in: -1...1, step: 0.05).frame(width: 140)
                    Text(String(format: "%+.2f", energy)).font(.caption.monospacedDigit())
                    Button("Set mood") {
                        Task { await client.saveMood(valence: valence, energy: energy) }
                    }
                    .disabled(client.actionInFlight)
                }
                HStack(spacing: 8) {
                    Text("arc phase").font(.caption).foregroundStyle(.secondary)
                    TextField("override (empty = derived)", text: $arcPhaseDraft)
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 260)
                        .onSubmit { saveArcPhase() }
                    Button("Set") { saveArcPhase() }
                        .disabled(client.actionInFlight)
                    if !now.arcPhase.isEmpty {
                        Button("Clear override") {
                            Task { await client.saveArcPhase(nil) }
                        }
                        .disabled(client.actionInFlight)
                    }
                }
            }
            .onAppear { syncDrafts(now) }
            .onChange(of: client.nowState) { _, updated in
                if let updated { syncDrafts(updated) }
            }
        }
    }

    private func saveArcPhase() {
        let trimmed = arcPhaseDraft.trimmingCharacters(in: .whitespaces)
        Task { await client.saveArcPhase(trimmed.isEmpty ? nil : trimmed) }
    }

    /// Re-seed drafts only when the server state actually changed — otherwise
    /// the 5s dashboard poll would stomp in-progress slider drags.
    private func syncDrafts(_ now: KiraNowState) {
        guard loadedFor != now else { return }
        loadedFor = now
        valence = now.valence
        energy = now.energy
        arcPhaseDraft = now.arcPhase
    }
}

// MARK: - Character description editor

struct KiraCharacterCard: View {
    @Bindable var client: KiraClient
    @State private var draft = KiraCharacterDescription()
    @State private var loadedFor: KiraCharacterDescription?
    @State private var anchorsDraft: String = ""

    var body: some View {
        if client.character != nil {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text(client.characterSource == "override"
                         ? "Edited (override active — the code default is preserved)"
                         : "Canonical code default")
                        .font(.caption2).foregroundStyle(.tertiary)
                    Spacer()
                    if client.characterSource == "override" {
                        Button("Revert to default") {
                            Task { await client.resetCharacter() }
                        }
                        .disabled(client.actionInFlight)
                    }
                }
                editor("Base (identity anchors — always injected)", text: $draft.base, height: 70)
                ForEach(KiraCharacterDescription.regionKeys, id: \.self) { key in
                    editor("Region: \(key)", text: regionBinding(key), height: 44)
                }
                editor("Banana tier (suggestive additions)", text: $draft.banana, height: 36)
                editor("Avocado tier (explicit additions)", text: $draft.avocado, height: 36)
                HStack(spacing: 8) {
                    Text("preserve anchors").font(.caption).foregroundStyle(.secondary)
                    TextField("comma-separated phrases the optimizer must keep", text: $anchorsDraft)
                        .textFieldStyle(.roundedBorder)
                }
                HStack(spacing: 8) {
                    Text("avocado anchor").font(.caption).foregroundStyle(.secondary)
                    TextField("canonical explicit-anatomy phrase", text: $draft.avocadoAnchor)
                        .textFieldStyle(.roundedBorder)
                }
                HStack {
                    Text("Writing guidelines: concrete visuals only, 30–60 words, no garments, light facial detail (face consistency is the LoRA's job).")
                        .font(.caption2).foregroundStyle(.tertiary)
                    Spacer()
                    Button("Save") {
                        var out = draft
                        out.preserveAnchors = anchorsDraft
                            .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                            .filter { !$0.isEmpty }
                        Task { await client.saveCharacter(out) }
                    }
                    .keyboardShortcut("s", modifiers: [.command])
                    .disabled(client.actionInFlight
                              || draft.base.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            .contentGated()   // description text is tiered; gate like other Kira surfaces
            .onAppear { syncDraft() }
            .onChange(of: client.character) { _, _ in syncDraft() }
        } else {
            Text("character description unavailable (daemon predates A6?)")
                .font(.caption).foregroundStyle(.tertiary)
        }
    }

    private func regionBinding(_ key: String) -> Binding<String> {
        Binding(
            get: { draft.regions[key] ?? "" },
            set: { draft.regions[key] = $0 })
    }

    private func editor(_ label: String, text: Binding<String>, height: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption).foregroundStyle(.secondary)
            TextEditor(text: text)
                .font(.caption)
                .frame(height: height)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(.quaternary))
        }
    }

    private func syncDraft() {
        guard let server = client.character, loadedFor != server else { return }
        loadedFor = server
        draft = server
        anchorsDraft = server.preserveAnchors.joined(separator: ", ")
    }
}

// MARK: - Lorebook editor

struct KiraLorebookCard: View {
    @Bindable var client: KiraClient
    @State private var editingID: String?
    @State private var draft = KiraLorebookEntry(id: "", title: "", enabled: true,
                                                 pinned: false, keywords: [], content: "")
    @State private var keywordsDraft: String = ""
    @State private var adding = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Canonical facts injected into her context by keyword match; pinned entries always ride along.")
                .font(.caption2).foregroundStyle(.tertiary)

            ForEach(client.lorebookEntries) { entry in
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Toggle("", isOn: Binding(
                            get: { entry.enabled },
                            set: { on in
                                var e = entry; e.enabled = on
                                Task { await client.updateLorebookEntry(e) }
                            }))
                            .toggleStyle(.checkbox).labelsHidden()
                            .help(entry.enabled ? "enabled" : "disabled")
                        Text(entry.title).font(.caption.bold())
                        if entry.pinned {
                            Image(systemName: "pin.fill").font(.caption2).foregroundStyle(.orange)
                        }
                        Spacer()
                        Button(editingID == entry.id ? "Close" : "Edit") {
                            if editingID == entry.id {
                                editingID = nil
                            } else {
                                editingID = entry.id
                                draft = entry
                                keywordsDraft = entry.keywords.joined(separator: ", ")
                                adding = false
                            }
                        }
                        .font(.caption)
                        Button(role: .destructive) {
                            Task { await client.deleteLorebookEntry(entry.id) }
                        } label: { Image(systemName: "trash") }
                            .font(.caption)
                            .disabled(client.actionInFlight)
                    }
                    if editingID != entry.id {
                        Text(entry.content).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                    }
                    if editingID == entry.id {
                        entryEditor(isNew: false)
                    }
                }
                .padding(6)
                .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.3)))
            }

            if adding {
                entryEditor(isNew: true)
                    .padding(6)
                    .background(RoundedRectangle(cornerRadius: 6).fill(.quaternary.opacity(0.3)))
            } else {
                Button {
                    adding = true
                    editingID = nil
                    draft = KiraLorebookEntry(id: "", title: "", enabled: true,
                                              pinned: false, keywords: [], content: "")
                    keywordsDraft = ""
                } label: {
                    Label("Add entry", systemImage: "plus")
                }
                .font(.caption)
            }
        }
        .contentGated()
    }

    @ViewBuilder private func entryEditor(isNew: Bool) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            TextField("Title", text: $draft.title).textFieldStyle(.roundedBorder)
            TextField("Keywords (comma-separated)", text: $keywordsDraft).textFieldStyle(.roundedBorder)
            TextEditor(text: $draft.content)
                .font(.caption)
                .frame(height: 70)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(.quaternary))
            HStack {
                Toggle("Pinned (always injected)", isOn: $draft.pinned)
                    .toggleStyle(.checkbox).font(.caption)
                Spacer()
                Button(isNew ? "Add" : "Save") {
                    let keywords = keywordsDraft
                        .split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                        .filter { !$0.isEmpty }
                    if isNew {
                        Task {
                            await client.addLorebookEntry(
                                title: draft.title, keywords: keywords,
                                content: draft.content, pinned: draft.pinned)
                        }
                        adding = false
                    } else {
                        var e = draft; e.keywords = keywords
                        Task { await client.updateLorebookEntry(e) }
                        editingID = nil
                    }
                }
                .disabled(client.actionInFlight
                          || draft.title.trimmingCharacters(in: .whitespaces).isEmpty
                          || draft.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }
}

// MARK: - World map (Todd 2026-07-29)

/// Her people + places (`/v1/kira/world`) — the persisted map that rides every
/// turn's state block. Full-replacement PUT: edit drafts locally, Apply saves
/// the whole map; the daemon caps/validates and the next poll reads back what
/// it kept. Kira edits the same store live via her `update_world` tool, so
/// Revert also picks up anything she added since the drafts loaded.
struct KiraWorldMapCard: View {
    @Bindable var client: KiraClient
    @State private var draft: [KiraWorldEntity] = []
    @State private var loadedFor: [KiraWorldEntity]?
    @State private var newName = ""
    @State private var newKind = "place"

    private static let kinds = ["friend", "person", "pet", "place"]

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            section("People", entities: draft.filter { !$0.isPlace })
            section("Places", entities: draft.filter { $0.isPlace })
            HStack(spacing: 8) {
                TextField("new entry name", text: $newName)
                    .textFieldStyle(.roundedBorder)
                    .frame(maxWidth: 180)
                Picker("", selection: $newKind) {
                    ForEach(Self.kinds, id: \.self) { Text($0) }
                }
                .frame(width: 90)
                Button("Add") {
                    let name = newName.trimmingCharacters(in: .whitespaces)
                    guard !name.isEmpty,
                          !draft.contains(where: { $0.name.lowercased() == name.lowercased() })
                    else { return }
                    draft.append(KiraWorldEntity(
                        name: name, kind: newKind, persona: "",
                        relation: "", location: "", facts: []))
                    newName = ""
                }
                .disabled(newName.trimmingCharacters(in: .whitespaces).isEmpty)
                Spacer()
                Button("Revert") { draft = client.world; loadedFor = client.world }
                    .disabled(draft == client.world)
                Button("Apply") { Task { await client.saveWorld(draft) } }
                    .buttonStyle(.borderedProminent)
                    .disabled(draft == client.world || client.actionInFlight)
            }
            Text("She maintains this map herself with update_world — Revert pulls in anything she added. Deleting here sticks: seeds never resurrect.")
                .font(.caption2).foregroundStyle(.tertiary)
        }
        .onAppear { syncDrafts() }
        .onChange(of: client.world) { syncDrafts() }
    }

    /// Load drafts from the live map, but never clobber unsaved local edits
    /// (same guard pattern as the character card's loadedFor).
    private func syncDrafts() {
        guard loadedFor != client.world else { return }
        if draft == (loadedFor ?? []) { draft = client.world }
        loadedFor = client.world
    }

    @ViewBuilder private func section(_ title: String, entities: [KiraWorldEntity]) -> some View {
        if !entities.isEmpty {
            Text(title).font(.caption).bold().foregroundStyle(.secondary)
            ForEach(entities) { entity in
                row(entity)
            }
        }
    }

    @ViewBuilder private func row(_ entity: KiraWorldEntity) -> some View {
        if let idx = draft.firstIndex(where: { $0.id == entity.id }) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(draft[idx].name).font(.callout).bold()
                    Text(draft[idx].kind)
                        .font(.caption2)
                        .padding(.horizontal, 5).padding(.vertical, 1)
                        .background(.quaternary, in: Capsule())
                    Spacer()
                    Button(role: .destructive) {
                        draft.remove(at: idx)
                    } label: { Image(systemName: "trash") }
                        .buttonStyle(.borderless)
                        .help("Remove from her map (Apply to persist — the deletion sticks)")
                }
                TextField("persona — short sketch", text: $draft[idx].persona)
                    .textFieldStyle(.roundedBorder).font(.caption)
                if draft[idx].isPlace {
                    TextField("where — e.g. above Barkada Brew", text: $draft[idx].location)
                        .textFieldStyle(.roundedBorder).font(.caption)
                } else {
                    TextField("relation — how they stand to her", text: $draft[idx].relation)
                        .textFieldStyle(.roundedBorder).font(.caption)
                }
                if !draft[idx].facts.isEmpty {
                    Text(draft[idx].facts.joined(separator: " · "))
                        .font(.caption2).foregroundStyle(.tertiary)
                        .lineLimit(2)
                        .help("Facts accumulate from her update_world calls")
                }
            }
            .padding(6)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
        }
    }
}
