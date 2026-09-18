// LibraryPromptSection.swift — the Library inside Generate
// (PRD docs/PRD-creative-library.md, L4 — "holistic, not a tab").
//
// A prompt is assembled from material instead of typed from nothing: choose a
// template, fill its slots from the wardrobe and components, apply a look, and
// save anything that worked back. The engine owns the material; this view owns
// only the assembly.

import SwiftUI

@MainActor
@Observable
public final class LibraryPromptModel {
    public var template: LibraryItemDTO?
    public var slotValues: [String: String] = [:]
    public var appliedLook: LibraryItemDTO?
    /// Ids used while assembling, so a render can record what made it.
    public private(set) var usedItemIds: Set<String> = []

    public init() {}

    public func use(_ item: LibraryItemDTO) {
        usedItemIds.insert(item.id)
    }

    public func chooseTemplate(_ item: LibraryItemDTO) {
        template = item
        use(item)
        // Seed each slot with its default so the fields are never blank-by-surprise.
        for slot in item.slots ?? [] where slotValues[slot.id] == nil {
            slotValues[slot.id] = slot.defaultValue ?? ""
        }
    }

    public func clearTemplate() {
        template = nil
        slotValues = [:]
    }

    /// The prompt a look produces around an assembled body.
    public nonisolated static func applyLook(_ look: LibraryItemDTO, to body: String) -> String {
        var parts: [String] = []
        if let prefix = look.promptPrefix?.trimmingCharacters(in: .whitespaces), !prefix.isEmpty {
            parts.append(prefix)
        }
        let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedBody.isEmpty { parts.append(trimmedBody) }
        let extra = look.value.trimmingCharacters(in: .whitespaces)
        if !extra.isEmpty { parts.append(extra) }
        if let suffix = look.promptSuffix?.trimmingCharacters(in: .whitespaces), !suffix.isEmpty {
            parts.append(suffix)
        }
        return parts.joined(separator: ", ")
            .replacingOccurrences(of: ", ,", with: ",")
    }

    /// Insert a component's text at the end of the prompt, without doubling
    /// punctuation.
    public nonisolated static func insert(_ value: String, into prompt: String) -> String {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedValue = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedValue.isEmpty else { return prompt }
        guard !trimmedPrompt.isEmpty else { return trimmedValue }
        let separator = trimmedPrompt.hasSuffix(",") || trimmedPrompt.hasSuffix(".") ? " " : ", "
        return trimmedPrompt + separator + trimmedValue
    }
}

public struct LibraryPromptSection: View {
    @Bindable var model: LibraryPromptModel
    let client: LibraryClient
    @Binding var prompt: String
    @Binding var negativePrompt: String
    @Binding var isExpanded: Bool

    @State private var picker: LibraryPickerView.Mode?
    @State private var slotBeingFilled: String?
    @State private var saveSheet = false
    @State private var status: String?

    public init(
        model: LibraryPromptModel, client: LibraryClient, prompt: Binding<String>,
        negativePrompt: Binding<String>, isExpanded: Binding<Bool>
    ) {
        self.model = model
        self.client = client
        _prompt = prompt
        _negativePrompt = negativePrompt
        _isExpanded = isExpanded
    }

    public var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            VStack(alignment: .leading, spacing: 10) {
                templateRow
                if let template = model.template, let slots = template.slots, !slots.isEmpty {
                    slotFields(slots)
                    Button {
                        Task { await applyTemplate() }
                    } label: {
                        Label("Write prompt from template", systemImage: "text.append")
                    }
                    .controlSize(.small)
                    .buttonStyle(.borderedProminent)
                }
                Divider()
                insertRow
                if let status {
                    Text(status).font(.caption2).foregroundStyle(.secondary)
                }
            }
            .padding(.top, 6)
        } label: {
            HStack(spacing: 6) {
                Label("Library", systemImage: "books.vertical")
                if let template = model.template {
                    Text(template.name).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
                if let look = model.appliedLook {
                    Text("· \(look.name)").font(.caption).foregroundStyle(.secondary).lineLimit(1)
                }
            }
        }
        .sheet(item: $picker) { mode in
            LibraryPickerView(mode: mode, client: client) { item in
                handle(item, from: mode)
            }
        }
        .sheet(isPresented: $saveSheet) {
            LibrarySaveSheet(client: client, prompt: prompt, negativePrompt: negativePrompt) { saved in
                status = "Saved \(saved.kind.replacingOccurrences(of: "_", with: " ")) “\(saved.name)”"
            }
        }
    }

    private var templateRow: some View {
        HStack(spacing: 8) {
            Button {
                picker = .templates
            } label: {
                Label(model.template == nil ? "Choose template…" : "Change template…",
                      systemImage: "doc.text")
            }
            .controlSize(.small)
            if model.template != nil {
                Button("Clear") { model.clearTemplate() }
                    .controlSize(.small)
            }
            Spacer()
            Button { saveSheet = true } label: {
                Label("Save as…", systemImage: "square.and.arrow.down")
            }
            .controlSize(.small)
            .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    private func slotFields(_ slots: [LibrarySlotDTO]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(slots) { slot in
                HStack(spacing: 6) {
                    Text(slot.label ?? slot.id)
                        .font(.caption)
                        .frame(width: 92, alignment: .trailing)
                        .foregroundStyle(.secondary)
                    TextField(
                        slot.placeholder ?? slot.id.lowercased(),
                        text: Binding(
                            get: { model.slotValues[slot.id] ?? "" },
                            set: { model.slotValues[slot.id] = $0 }))
                        .textFieldStyle(.roundedBorder)
                    if let options = slot.options, !options.isEmpty {
                        Menu {
                            ForEach(options, id: \.self) { option in
                                Button(option) { model.slotValues[slot.id] = option }
                            }
                        } label: {
                            Image(systemName: "chevron.down.circle")
                        }
                        .menuStyle(.borderlessButton)
                        .frame(width: 22)
                    }
                    Button {
                        slotBeingFilled = slot.id
                        picker = pickerMode(for: slot.id)
                    } label: {
                        Image(systemName: "books.vertical")
                    }
                    .buttonStyle(.borderless)
                    .help("Fill from the library")
                }
            }
        }
    }

    /// A slot named OUTFIT wants the wardrobe; SCENE wants scene components.
    /// Anything else opens components, because that is the general drawer.
    func pickerMode(for slotId: String) -> LibraryPickerView.Mode {
        switch slotId.uppercased() {
        case "OUTFIT", "WARDROBE", "CLOTHING": return .outfits
        case "SCENE", "SETTING", "LOCATION": return .components(kind: "scene")
        case "LIGHTING", "LIGHT": return .components(kind: "lighting")
        case "POSE": return .components(kind: "pose")
        default: return .components(kind: nil)
        }
    }

    private var insertRow: some View {
        HStack(spacing: 8) {
            Button { picker = .components(kind: "scene") } label: {
                Label("Scene", systemImage: "mountain.2")
            }
            Button { picker = .wardrobe } label: {
                Label("Wardrobe", systemImage: "tshirt")
            }
            Button { picker = .outfits } label: {
                Label("Outfit", systemImage: "person.crop.rectangle.stack")
            }
            Button { picker = .looks } label: {
                Label("Look", systemImage: "camera.filters")
            }
            Spacer()
        }
        .controlSize(.small)
        .buttonStyle(.bordered)
    }

    private func handle(_ item: LibraryItemDTO, from mode: LibraryPickerView.Mode) {
        model.use(item)
        switch mode {
        case .templates:
            model.chooseTemplate(item)
            status = "Template “\(item.name)” — fill its slots, then write the prompt."
        case .looks:
            model.appliedLook = item
            prompt = LibraryPromptModel.applyLook(item, to: prompt)
            if let negative = item.negativePrompt, !negative.isEmpty {
                negativePrompt = negativePrompt.isEmpty
                    ? negative
                    : negativePrompt + ", " + negative
            }
            status = "Applied look “\(item.name)”."
        default:
            if let slot = slotBeingFilled {
                model.slotValues[slot] = item.value
                slotBeingFilled = nil
                status = "\(slot): \(item.name)"
            } else {
                prompt = LibraryPromptModel.insert(item.value, into: prompt)
                status = "Inserted “\(item.name)”."
            }
        }
    }

    private func applyTemplate() async {
        guard let template = model.template else { return }
        if let result = await client.fill(templateId: template.id, values: model.slotValues) {
            prompt = result.prompt
            status = result.unfilled.isEmpty
                ? "Prompt written from “\(template.name)”."
                : "Unfilled: \(result.unfilled.joined(separator: ", ")) — they stay visible in the prompt."
        } else {
            status = "Could not reach the library to fill that template."
        }
    }
}

// MARK: - Save sheet

/// "Save as…" — turning what just worked back into material.
struct LibrarySaveSheet: View {
    let client: LibraryClient
    let prompt: String
    let negativePrompt: String
    let onSaved: (LibraryItemDTO) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var kind = "component"
    @State private var name = ""
    @State private var value = ""
    @State private var componentKind = "scene"
    @State private var category = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Save to library").font(.headline)
            Picker("As", selection: $kind) {
                Text("Component").tag("component")
                Text("Template").tag("template")
                Text("Wardrobe item").tag("wardrobe_item")
                Text("Outfit").tag("outfit")
                Text("Look").tag("look")
            }
            .pickerStyle(.segmented)
            TextField("Name", text: $name).textFieldStyle(.roundedBorder)
            if kind == "component" {
                Picker("Kind", selection: $componentKind) {
                    ForEach(["scene", "pose", "lighting", "camera", "expression", "styling"], id: \.self) {
                        Text($0.capitalized).tag($0)
                    }
                }
            }
            if kind == "wardrobe_item" {
                TextField("Category (e.g. Tops)", text: $category).textFieldStyle(.roundedBorder)
            }
            Text(kind == "template"
                 ? "Slots use {NAME} markers; an unfilled slot stays visible."
                 : "Text this contributes to a prompt.")
                .font(.caption2).foregroundStyle(.secondary)
            TextEditor(text: $value)
                .font(.body)
                .frame(minHeight: 90)
                .border(Color(nsColor: .separatorColor))
            HStack {
                Spacer()
                Button("Cancel") { dismiss() }.keyboardShortcut(.cancelAction)
                Button("Save") { Task { await save() } }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(name.trimmingCharacters(in: .whitespaces).isEmpty
                              || value.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(16)
        .frame(width: 460)
        .onAppear { if value.isEmpty { value = prompt } }
    }

    private func save() async {
        var item = LibraryItemDTO(
            id: "local:\(kind):\(UUID().uuidString.prefix(8))",
            kind: kind, name: name.trimmingCharacters(in: .whitespaces), value: value)
        if kind == "component" { item.componentKind = componentKind }
        if kind == "wardrobe_item", !category.isEmpty { item.category = category }
        if kind == "look", !negativePrompt.isEmpty { item.negativePrompt = negativePrompt }
        if kind == "template" {
            // Declare every {SLOT} the body uses, or the engine refuses it.
            let markers = LibrarySlotScanner.markers(in: value)
            item.slots = markers.sorted().map { LibrarySlotDTO(id: $0, label: $0.capitalized) }
        }
        if await client.save(item) {
            onSaved(item)
            dismiss()
        }
    }
}

/// `{SLOT}` markers in a body — the desktop's copy of the engine's rule, so
/// "Save as template" cannot produce something the engine will refuse.
public enum LibrarySlotScanner {
    public static func markers(in body: String) -> Set<String> {
        guard let regex = try? NSRegularExpression(pattern: "\\{([A-Za-z0-9_-]+)\\}") else {
            return []
        }
        let ns = body as NSString
        return Set(
            regex.matches(in: body, range: NSRange(location: 0, length: ns.length))
                .map { ns.substring(with: $0.range(at: 1)) })
    }
}

extension LibraryPickerView.Mode: Identifiable {
    public var id: String {
        switch self {
        case .templates: return "templates"
        case .components(let kind): return "components:\(kind ?? "all")"
        case .wardrobe: return "wardrobe"
        case .outfits: return "outfits"
        case .looks: return "looks"
        case .recipes: return "recipes"
        }
    }
}
