// AssetSequenceIngestTests.swift — WP13 (docs/FDD-ltx-director-tab.md §4.9.2)
//
// The catalog's backfill is a manual CLI; THIS poller is what sees a fresh
// Director render. If the sequence is not picked up here, a clip is not
// reopenable until someone runs a command nobody runs.

import Foundation
import Testing

@testable import ComfyBoxDesktop

@Suite("Ingest: a Director render carries its sequence")
struct AssetSequenceIngestTests {

    private func withDirectory(_ body: (String) throws -> Void) throws {
        let directory = NSTemporaryDirectory() + "ingest-sequence-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: directory) }
        try body(directory)
    }

    @discardableResult
    private func write(_ json: String, besides clip: String, in directory: String) throws -> String {
        let media = directory + "/" + clip
        FileManager.default.createFile(atPath: media, contents: Data("mp4".utf8))
        let sidecar = (media as NSString).deletingPathExtension + ".sequence.json"
        try Data(json.utf8).write(to: URL(fileURLWithPath: sidecar))
        return media
    }

    private var valid: String {
        """
        {"schema":"comfybox.sequence","version":1,"id":"seq-7","kind":"director",
         "name":"morning monologue",
         "chunks":[{"index":0},{"index":1},{"index":2}]}
        """
    }

    @Test("a rendered sequence is read from beside the clip")
    func readsTheSidecar() throws {
        try withDirectory { directory in
            let media = try write(valid, besides: "clip.mp4", in: directory)
            let sequence = try #require(AssetIngestor.readSequenceSidecar(for: media))
            #expect(sequence.id == "seq-7")
            #expect(sequence.name == "morning monologue")
            #expect(sequence.chunks == 3)
        }
    }

    @Test("an ordinary clip has no sequence")
    func plainClip() throws {
        try withDirectory { directory in
            let media = directory + "/plain.mp4"
            FileManager.default.createFile(atPath: media, contents: Data("mp4".utf8))
            #expect(AssetIngestor.readSequenceSidecar(for: media) == nil)
        }
    }

    @Test("a broken sidecar leaves the clip an ordinary video")
    func brokenSidecar() throws {
        try withDirectory { directory in
            let media = try write("{ not json", besides: "clip.mp4", in: directory)
            #expect(AssetIngestor.readSequenceSidecar(for: media) == nil)
        }
    }

    @Test("another tool's .sequence.json is not read as ours")
    func foreignSidecar() throws {
        try withDirectory { directory in
            let media = try write(#"{"id":"x","chunks":[]}"#, besides: "clip.mp4", in: directory)
            #expect(AssetIngestor.readSequenceSidecar(for: media) == nil, "no schema, not ours")
        }
    }

    @Test("isSequence is what the UI asks")
    func isSequenceFlag() {
        let director = DAMAsset(
            kind: "video", filename: "c.mp4", absolutePath: "/a/c.mp4", sequenceID: "seq-7")
        let plain = DAMAsset(kind: "video", filename: "p.mp4", absolutePath: "/a/p.mp4")
        #expect(director.isSequence)
        #expect(plain.isSequence == false)
        #expect(director.kind == "video", "a Director render is still a video")
    }
}
