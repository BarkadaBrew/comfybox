// CharacterEntryTests.swift — Tests for CharacterEntry

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("CharacterEntry")
struct CharacterEntryTests {
    @Test("default initialization")
    func defaults() {
        let entry = CharacterEntry(id: "c1", name: "Alice")
        #expect(entry.id == "c1")
        #expect(entry.name == "Alice")
        #expect(entry.description == "")
        #expect(entry.defaultLoras.isEmpty)
        #expect(entry.promptSnippet == "")
        #expect(entry.tags.isEmpty)
    }

    @Test("full initialization preserves all fields")
    func fullInit() {
        let entry = TestData.makeCharacter(id: "c2", name: "Bob", description: "A character", promptSnippet: "bob prompt")
        #expect(entry.id == "c2")
        #expect(entry.name == "Bob")
        #expect(entry.description == "A character")
        #expect(entry.promptSnippet == "bob prompt")
        #expect(!entry.defaultLoras.isEmpty)
        #expect(!entry.tags.isEmpty)
    }

    @Test("identifiable via id")
    func identifiable() {
        let entry = CharacterEntry(id: "unique", name: "Test")
        #expect(entry.id == "unique")
    }

    @Test("sendable conformance")
    func sendable() {
        let entry = TestData.makeCharacter()
        let _: any Sendable = entry
        _ = entry
    }

    @Test("displayDescription prefers the base tier over the flat description")
    func displayPrefersBase() {
        let tiered = CharacterEntry(id: "t", name: "T", description: "", base: "SFW base text")
        #expect(tiered.displayDescription == "SFW base text")

        let both = CharacterEntry(id: "b", name: "B", description: "flat", base: "base wins")
        #expect(both.displayDescription == "base wins")
    }

    @Test("displayDescription falls back to flat description when no base")
    func displayFallsBackToFlat() {
        let flat = CharacterEntry(id: "f", name: "F", description: "flat only")
        #expect(flat.displayDescription == "flat only")

        let emptyBase = CharacterEntry(id: "e", name: "E", description: "flat", base: "")
        #expect(emptyBase.displayDescription == "flat")
    }

    @Test("tier and passthrough fields default to nil")
    func tierDefaults() {
        let entry = CharacterEntry(id: "c1", name: "Alice")
        #expect(entry.base == nil)
        #expect(entry.banana == nil)
        #expect(entry.avocado == nil)
        #expect(entry.negativePrompt == nil)
        #expect(entry.triggerWords == nil)
    }
}
