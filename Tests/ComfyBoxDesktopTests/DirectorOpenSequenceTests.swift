// DirectorOpenSequenceTests.swift — WP13 (docs/FDD-ltx-director-tab.md §4.9.2)
//
// Reopening a render: drop the mp4 on the Director tab and get the timeline
// that made it, the way a ComfyUI PNG restores its graph. The sequence is
// stored beside the clip as `clip.sequence.json`.

import Foundation
import Testing
import ZImage

@testable import ComfyBoxDesktop

@MainActor
@Suite("Director: open a rendered clip's sequence")
struct DirectorOpenSequenceTests {

    /// A scratch directory that cleans itself up.
    private func withDirectory(_ body: (URL) throws -> Void) throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("director-open-sequence-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    private func timeline(prompt: String) -> DirectorTimeline {
        DirectorTimeline(
            settings: .init(width: 576, height: 896, lengthFrames: 289),
            globalPrompt: prompt,
            keyframes: [.init(id: "k1", imagePath: "/img/a.png", frame: 0)])
    }

    /// Write a render's sidecar; return the mp4 path it sits beside.
    private func render(in directory: URL, named name: String, prompt: String) -> URL {
        let media = directory.appendingPathComponent("\(name).mp4")
        let document = SequenceDocument(
            id: "seq-1", name: name, timeline: timeline(prompt: prompt),
            chunks: [
                SequenceChunkRecord(index: 0, startFrame: 0, frames: 145, seed: 7),
                SequenceChunkRecord(index: 1, startFrame: 144, frames: 145, seed: 8),
            ])
        #expect(SequenceSidecar.write(document, forMediaAt: media.path))
        return media
    }

    @Test("opening a render restores the timeline that made it")
    func opensTheTimeline() throws {
        try withDirectory { directory in
            let media = render(in: directory, named: "clip", prompt: "she talks to camera")
            let model = DirectorDocumentModel()

            let document = model.loadSequence(forMediaAt: media)

            #expect(document?.chunks.count == 2)
            #expect(model.timeline.globalPrompt == "she talks to camera")
            #expect(model.timeline.keyframes.first?.imagePath == "/img/a.png")
        }
    }

    @Test("a reopened sequence is not a project file")
    func doesNotAdoptTheRendersSidecarAsItsProject() throws {
        // Saving must ask where to put it rather than writing over the render's
        // own sidecar.
        try withDirectory { directory in
            let media = render(in: directory, named: "clip", prompt: "x")
            let model = DirectorDocumentModel(timeline: timeline(prompt: "y"))
            try model.save(to: directory.appendingPathComponent("old.director.json"))
            #expect(model.projectURL != nil)

            _ = model.loadSequence(forMediaAt: media)

            #expect(model.projectURL == nil)
            #expect(model.canUndo == false, "a reopened render starts a fresh editing history")
        }
    }

    @Test("a clip with no sequence is reported, not opened as an empty timeline")
    func reportsAClipRenderedBeforeSequences() throws {
        try withDirectory { directory in
            let model = DirectorDocumentModel(timeline: timeline(prompt: "untouched"))
            let media = directory.appendingPathComponent("older.mp4")

            #expect(model.loadSequence(forMediaAt: media) == nil)
            #expect(
                model.timeline.globalPrompt == "untouched",
                "a clip rendered before sequences leaves the open document alone")
        }
    }

    @Test("the sidecar is found beside the media by name")
    func sidecarPath() {
        #expect(SequenceSidecar.path(forMediaAt: "/a/b/clip.mp4") == "/a/b/clip.sequence.json")
    }
}
