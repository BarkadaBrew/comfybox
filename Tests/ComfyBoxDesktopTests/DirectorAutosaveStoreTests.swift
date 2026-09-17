// DirectorAutosaveStoreTests.swift — WP3: debounced autosave of the open timeline.

import Foundation
import Testing
import ZImage
@testable import ComfyBoxDesktop

@MainActor
@Suite("DirectorAutosaveStore")
struct DirectorAutosaveStoreTests {

    private func tempURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("director-autosave-\(UUID().uuidString)")
            .appendingPathComponent("director-autosave.cbdirector")
    }

    @Test("schedule then flushNow writes the timeline")
    func scheduleThenFlushWrites() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let store = DirectorAutosaveStore(path: url, debounce: 60)
        store.schedule(directorTestTimeline())
        #expect(FileManager.default.fileExists(atPath: url.path) == false)
        store.flushNow()
        #expect(store.load() == directorTestTimeline())
    }

    @Test("load returns nil when no autosave exists")
    func loadReturnsNilWhenAbsent() {
        #expect(DirectorAutosaveStore(path: tempURL()).load() == nil)
    }

    @Test("clear removes the file and any pending write")
    func clearRemovesFile() throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let store = DirectorAutosaveStore(path: url, debounce: 60)
        store.schedule(directorTestTimeline())
        store.flushNow()
        #expect(FileManager.default.fileExists(atPath: url.path))
        store.schedule(directorTestTimeline(length: 97))
        store.clear()
        store.flushNow()
        #expect(FileManager.default.fileExists(atPath: url.path) == false)
    }

    @Test("rapid schedules coalesce into one write of the latest timeline")
    func debounceCoalesces() async throws {
        let url = tempURL()
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let store = DirectorAutosaveStore(path: url, debounce: 0.05)
        for len in [97, 145, 289] {
            store.schedule(directorTestTimeline(length: len))
        }
        try await Task.sleep(for: .milliseconds(400))
        #expect(store.writeCount == 1)
        #expect(store.load()?.settings.lengthFrames == 289)
    }
}
