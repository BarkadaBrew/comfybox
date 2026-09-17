// DirectorProjectRoundTripTests.swift — WP3: .cbdirector save/load + autosave
// wiring through DirectorDocumentModel.

import Foundation
import Testing
import ZImage
@testable import ComfyBoxDesktop

@MainActor
@Suite("Director project files")
struct DirectorProjectRoundTripTests {

    private func tempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("director-proj-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test("load -> edit -> save -> load equals the in-memory timeline")
    func loadEditSaveLoadEquals() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("scene.cbdirector")
        try DirectorDocument.write(directorTestTimeline(), to: url)

        let model = DirectorDocumentModel(timeline: directorTestTimeline(length: 97))
        try model.load(url: url)
        #expect(model.timeline == directorTestTimeline())
        #expect(model.projectURL == url)
        #expect(model.isDirty == false)
        #expect(model.canUndo == false)

        _ = model.addKeyframe(imagePath: "/tmp/a.png", atFrame: 0)
        _ = model.addPromptSegment(startFrame: 0, lengthFrames: 96, prompt: "she turns")
        try model.save(to: url)
        #expect(model.isDirty == false)

        let reloaded = DirectorDocumentModel(timeline: directorTestTimeline(length: 97))
        try reloaded.load(url: url)
        #expect(reloaded.timeline == model.timeline)
    }

    @Test("desktop-written files decode under the engine's snake_case decoder")
    func savedFileDecodesUnderEngineDecoder() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("scene.cbdirector")
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        _ = model.addKeyframe(imagePath: "/tmp/a.png", atFrame: 8)
        _ = model.addAudioClip(path: "/tmp/v.wav", atFrame: 0, lengthFrames: 48)
        try model.save(to: url)

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase  // WarmServer.decode's strategy
        let decoded = try decoder.decode(DirectorTimeline.self, from: Data(contentsOf: url))
        #expect(decoded == model.timeline)
    }

    @Test("save never embeds base64 unless asked")
    func saveNeverEmbedsBase64UnlessAsked() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let image = dir.appendingPathComponent("k.png")
        let bytes = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3, 4])
        try bytes.write(to: image)
        let model = DirectorDocumentModel(timeline: directorTestTimeline())
        _ = model.addKeyframe(imagePath: image.path, atFrame: 0)

        let plain = dir.appendingPathComponent("plain.cbdirector")
        try model.save(to: plain)
        #expect(String(decoding: try Data(contentsOf: plain), as: UTF8.self).contains("image_base64") == false)

        let embedded = dir.appendingPathComponent("embedded.cbdirector")
        try model.save(to: embedded, embedAssets: true)
        let read = try DirectorDocument.read(from: embedded)
        #expect(read.keyframes.first?.imageBase64 == bytes.base64EncodedString())
        // In memory the timeline never carries the payload.
        #expect(model.timeline.keyframes.first?.imageBase64 == nil)
    }

    @Test("load does not trigger an autosave")
    func loadDoesNotTriggerAutosave() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let autosaveURL = dir.appendingPathComponent("autosave.cbdirector")
        let store = DirectorAutosaveStore(path: autosaveURL, debounce: 0)
        let url = dir.appendingPathComponent("scene.cbdirector")
        try DirectorDocument.write(directorTestTimeline(), to: url)

        let model = DirectorDocumentModel(timeline: directorTestTimeline(), autosave: store)
        try model.load(url: url)
        store.flushNow()
        #expect(FileManager.default.fileExists(atPath: autosaveURL.path) == false)
    }

    @Test("an edit triggers an autosave")
    func editTriggersAutosave() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let autosaveURL = dir.appendingPathComponent("autosave.cbdirector")
        let store = DirectorAutosaveStore(path: autosaveURL, debounce: 0)
        let model = DirectorDocumentModel(timeline: directorTestTimeline(), autosave: store)
        model.setGlobalPrompt("edited prompt")
        store.flushNow()
        #expect(store.load()?.globalPrompt == "edited prompt")
    }

    @Test("restoreAutosaveIfAny restores when no project is open")
    func restoreAutosaveIfAnyRestoresWhenNoProjectURL() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let autosaveURL = dir.appendingPathComponent("autosave.cbdirector")
        var saved = directorTestTimeline()
        saved.globalPrompt = "unsaved work"
        try DirectorDocument.write(saved, to: autosaveURL)
        let store = DirectorAutosaveStore(path: autosaveURL, debounce: 0)

        let model = DirectorDocumentModel(timeline: directorTestTimeline(), autosave: store)
        #expect(model.restoreAutosaveIfAny())
        #expect(model.timeline.globalPrompt == "unsaved work")
        #expect(model.isDirty)
        #expect(model.canUndo == false)

        // With a project open the autosave is left alone.
        let projectURL = dir.appendingPathComponent("p.cbdirector")
        try DirectorDocument.write(directorTestTimeline(), to: projectURL)
        let opened = DirectorDocumentModel(timeline: directorTestTimeline(), autosave: store)
        try opened.load(url: projectURL)
        #expect(opened.restoreAutosaveIfAny() == false)
        #expect(opened.timeline.globalPrompt == directorTestTimeline().globalPrompt)
    }

    @Test("new() resets the document and clears the autosave")
    func newClearsAutosave() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let autosaveURL = dir.appendingPathComponent("autosave.cbdirector")
        let store = DirectorAutosaveStore(path: autosaveURL, debounce: 0)
        let model = DirectorDocumentModel(timeline: directorTestTimeline(), autosave: store)
        model.setGlobalPrompt("edited")
        store.flushNow()
        #expect(FileManager.default.fileExists(atPath: autosaveURL.path))
        model.new()
        #expect(FileManager.default.fileExists(atPath: autosaveURL.path) == false)
        #expect(model.isDirty == false)
        #expect(model.projectURL == nil)
        #expect(model.timeline.keyframes.isEmpty)
        #expect(model.canUndo == false)
    }
}
