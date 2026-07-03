// PromptLibraryStore.swift — Saved prompt snippets
//
// A small persisted library of reusable prompts (~/.comfybox/
// prompt-library.json — under the shared config home so other coffeeshop
// tools can read it). History comes from the DAM separately; this store
// only holds what the user explicitly saved.

import Foundation

public struct PromptLibraryEntry: Codable, Identifiable, Sendable, Equatable {
    public var id: String
    public var title: String
    public var text: String
    public var tags: [String]
    public var createdAt: Date
    public var useCount: Int

    public init(
        id: String = UUID().uuidString,
        title: String = "",
        text: String,
        tags: [String] = [],
        createdAt: Date = Date(),
        useCount: Int = 0
    ) {
        self.id = id
        self.title = title
        self.text = text
        self.tags = tags
        self.createdAt = createdAt
        self.useCount = useCount
    }

    /// Title when set, else the first few words of the prompt.
    public var displayTitle: String {
        if !title.isEmpty { return title }
        let words = text.split(separator: " ").prefix(8).joined(separator: " ")
        return words.isEmpty ? "Untitled" : words
    }
}

@Observable
@MainActor
public final class PromptLibraryStore {
    public private(set) var entries: [PromptLibraryEntry] = []

    private let path: URL

    public nonisolated static func defaultPath() -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".comfybox/prompt-library.json")
    }

    public init(path: URL = PromptLibraryStore.defaultPath()) {
        self.path = path
        if let data = try? Data(contentsOf: path) {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            entries = (try? decoder.decode([PromptLibraryEntry].self, from: data)) ?? []
        }
    }

    /// Save a new prompt. Duplicate text is a no-op returning the existing entry.
    @discardableResult
    public func add(title: String = "", text: String, tags: [String] = []) -> PromptLibraryEntry {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if let existing = entries.first(where: { $0.text == trimmed }) {
            return existing
        }
        let entry = PromptLibraryEntry(title: title, text: trimmed, tags: tags)
        entries.insert(entry, at: 0)
        persist()
        return entry
    }

    public func update(_ entry: PromptLibraryEntry) {
        guard let index = entries.firstIndex(where: { $0.id == entry.id }) else { return }
        entries[index] = entry
        persist()
    }

    public func delete(id: String) {
        entries.removeAll { $0.id == id }
        persist()
    }

    /// Bump the use counter when a prompt is inserted into Generate.
    public func recordUse(id: String) {
        guard let index = entries.firstIndex(where: { $0.id == id }) else { return }
        entries[index].useCount += 1
        persist()
    }

    private func persist() {
        let dir = path.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(entries) {
            try? data.write(to: path, options: .atomic)
        }
    }
}
