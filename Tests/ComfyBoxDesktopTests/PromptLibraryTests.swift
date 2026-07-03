// PromptLibraryTests.swift — Saved-prompt store + DAM prompt history

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("PromptLibraryStore")
@MainActor
struct PromptLibraryStoreTests {
    private func tempPath() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("prompts-\(UUID().uuidString).json")
    }

    @Test("add, update, delete, and use counting")
    func crud() {
        let store = PromptLibraryStore(path: tempPath())
        let entry = store.add(title: "Espresso", text: "kira making espresso", tags: ["kira"])
        #expect(store.entries.count == 1)
        #expect(entry.displayTitle == "Espresso")

        var edited = entry
        edited.title = "Morning espresso"
        store.update(edited)
        #expect(store.entries.first?.title == "Morning espresso")

        store.recordUse(id: entry.id)
        store.recordUse(id: entry.id)
        #expect(store.entries.first?.useCount == 2)

        store.delete(id: entry.id)
        #expect(store.entries.isEmpty)
    }

    @Test("duplicate text is a no-op returning the existing entry")
    func dedupe() {
        let store = PromptLibraryStore(path: tempPath())
        let first = store.add(text: "same prompt")
        let second = store.add(text: "  same prompt  ")  // whitespace-normalized
        #expect(first.id == second.id)
        #expect(store.entries.count == 1)
    }

    @Test("displayTitle falls back to leading words")
    func displayTitleFallback() {
        let entry = PromptLibraryEntry(text: "a very long prompt with many descriptive words that keeps going and going")
        #expect(entry.displayTitle == "a very long prompt with many descriptive words")
    }

    @Test("persists across instances")
    func persistence() {
        let path = tempPath()
        let store1 = PromptLibraryStore(path: path)
        store1.add(title: "Keep", text: "persisted prompt", tags: ["t"])

        let store2 = PromptLibraryStore(path: path)
        #expect(store2.entries.count == 1)
        #expect(store2.entries.first?.title == "Keep")
        #expect(store2.entries.first?.tags == ["t"])
    }
}

@Suite("DAM prompt history")
struct DAMPromptHistoryTests {
    @Test("groups identical prompts with counts, newest first")
    func history() async throws {
        let tmpDir = NSTemporaryDirectory()
        let dbPath = (tmpDir as NSString).appendingPathComponent("test-dam-\(UUID().uuidString).sqlite3")
        let store = try await DAMStore.open(path: dbPath)

        let old = Date(timeIntervalSinceNow: -3600)
        try await store.insertAsset(DAMAsset(
            id: "h1", filename: "h1.png", absolutePath: "/tmp/h1.png",
            createdAt: old, prompt: "red cube"))
        try await store.insertAsset(DAMAsset(
            id: "h2", filename: "h2.png", absolutePath: "/tmp/h2.png",
            createdAt: Date(), prompt: "red cube"))
        try await store.insertAsset(DAMAsset(
            id: "h3", filename: "h3.png", absolutePath: "/tmp/h3.png",
            createdAt: Date(timeIntervalSinceNow: -60), prompt: "blue sphere"))
        try await store.insertAsset(DAMAsset(
            id: "h4", filename: "h4.png", absolutePath: "/tmp/h4.png",
            createdAt: Date(), prompt: nil))  // no prompt — excluded

        let history = try await store.promptHistory(limit: 10)
        #expect(history.count == 2)
        #expect(history[0].prompt == "red cube")
        #expect(history[0].count == 2)
        #expect(history[1].prompt == "blue sphere")
        #expect(history[1].count == 1)
        try? FileManager.default.removeItem(atPath: dbPath)
    }
}
