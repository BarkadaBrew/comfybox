// ArchiveManifestTests.swift — Tests for the .cbarchive manifest types + JSONL codec

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("ArchiveManifest")
struct ArchiveManifestTests {

    // MARK: - Round trip

    @Test("full round-trip of ArchivedAsset covers all 23 DAM fields including source")
    func fullRoundTrip() throws {
        let asset = DAMAsset(
            id: "9F3CA1B2-0000-0000-0000-0000000000D4",
            kind: "video",
            filename: "kira-0042.mp4",
            absolutePath: "/Users/todd/Pictures/ComfyBox/kira-0042.mp4",
            fileSize: 2_841_193,
            sha256: "abc123",
            width: 1280,
            height: 1280,
            createdAt: Date(timeIntervalSince1970: 1_700_000_000),
            modifiedAt: Date(timeIntervalSince1970: 1_700_000_100),
            ingestedAt: Date(timeIntervalSince1970: 1_700_000_200),
            orphaned: true,
            prompt: "a portrait of Kira at dusk",
            negativePrompt: "blurry, low quality",
            seed: 88123,
            steps: 9,
            guidance: 3.5,
            modelFamily: "z-image-turbo",
            rating: 4,
            favorite: true,
            contentMode: "banana",
            characterName: "Kira",
            source: "kira"
        )

        let entry = ArchivedAsset(from: asset, folderId: "B1C4", relativeRoot: "assets/\(asset.id)")

        let encoded = try ArchiveJSONL.encodeLine(entry)
        var line = String(data: encoded, encoding: .utf8)!
        #expect(line.hasSuffix("\n"))
        #expect(!line.dropLast().contains("\n"))
        line.removeLast()

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(ArchivedAsset.self, from: Data(line.utf8))

        #expect(decoded.id == asset.id)
        #expect(decoded.relativePath == "assets/\(asset.id)/kira-0042.mp4")
        #expect(decoded.sidecarRelativePath == "assets/\(asset.id)/kira-0042.json")
        #expect(decoded.thumbnailRelativePath == "assets/\(asset.id)/thumb.jpg")
        #expect(decoded.originalPath != nil)
        #expect(decoded.kind == "video")
        #expect(decoded.filename == "kira-0042.mp4")
        #expect(decoded.fileSize == 2_841_193)
        #expect(decoded.sha256 == "abc123")
        #expect(decoded.width == 1280)
        #expect(decoded.height == 1280)
        #expect(decoded.createdAt == 1_700_000_000.0)
        #expect(decoded.modifiedAt == 1_700_000_100.0)
        #expect(decoded.ingestedAt == 1_700_000_200.0)
        #expect(decoded.orphaned == true)
        #expect(decoded.prompt == "a portrait of Kira at dusk")
        #expect(decoded.negativePrompt == "blurry, low quality")
        #expect(decoded.seed == 88123)
        #expect(decoded.steps == 9)
        #expect(decoded.guidance == 3.5)
        #expect(decoded.modelFamily == "z-image-turbo")
        #expect(decoded.rating == 4)
        #expect(decoded.favorite == true)
        #expect(decoded.contentMode == "banana")
        #expect(decoded.characterName == "Kira")
        #expect(decoded.source == "kira")
        #expect(decoded.folderId == "B1C4")

        let restored = decoded.toDAMAsset(absolutePath: "/restored/path/kira-0042.mp4")
        #expect(restored.id == asset.id)
        #expect(restored.kind == "video")
        #expect(restored.filename == "kira-0042.mp4")
        #expect(restored.absolutePath == "/restored/path/kira-0042.mp4")
        #expect(restored.fileSize == 2_841_193)
        #expect(restored.sha256 == "abc123")
        #expect(restored.width == 1280)
        #expect(restored.height == 1280)
        #expect(restored.createdAt == Date(timeIntervalSince1970: 1_700_000_000))
        #expect(restored.modifiedAt == Date(timeIntervalSince1970: 1_700_000_100))
        #expect(restored.ingestedAt == Date(timeIntervalSince1970: 1_700_000_200))
        #expect(restored.orphaned == true)
        #expect(restored.prompt == "a portrait of Kira at dusk")
        #expect(restored.negativePrompt == "blurry, low quality")
        #expect(restored.seed == 88123)
        #expect(restored.steps == 9)
        #expect(restored.guidance == 3.5)
        #expect(restored.modelFamily == "z-image-turbo")
        #expect(restored.rating == 4)
        #expect(restored.favorite == true)
        #expect(restored.contentMode == "banana")
        #expect(restored.characterName == "Kira")
        #expect(restored.source == "kira")

        // toDAMAsset honors an id override (restore-with-new-id path)
        let withOverride = decoded.toDAMAsset(absolutePath: "/x.mp4", id: "new-id")
        #expect(withOverride.id == "new-id")
    }

    @Test("timestamps survive as exact Doubles")
    func exactTimestamps() throws {
        let asset = DAMAsset(
            filename: "t.png", absolutePath: "/tmp/t.png",
            createdAt: Date(timeIntervalSince1970: 1_700_000_000),
            modifiedAt: Date(timeIntervalSince1970: 1_700_000_000),
            ingestedAt: Date(timeIntervalSince1970: 1_700_000_000)
        )
        let entry = ArchivedAsset(from: asset, folderId: nil, relativeRoot: "assets/\(asset.id)")
        let data = try ArchiveJSONL.encodeLine(entry)
        let decoded = try JSONDecoder().decode(ArchivedAsset.self, from: data)
        #expect(decoded.createdAt == 1_700_000_000.0)
        #expect(decoded.modifiedAt == 1_700_000_000.0)
        #expect(decoded.ingestedAt == 1_700_000_000.0)
    }

    // MARK: - Forward/backward JSON compatibility

    @Test("unknown JSON keys are ignored")
    func unknownKeysIgnored() throws {
        let json = """
        {"id":"a1","relativePath":"assets/a1/x.png","kind":"image","filename":"x.png",
         "fileSize":100,"createdAt":1.0,"modifiedAt":1.0,"ingestedAt":1.0,"orphaned":false,
         "rating":0,"favorite":false,"archivedAt":2.0,"futureField":1}
        """
        let decoded = try JSONDecoder().decode(ArchivedAsset.self, from: Data(json.utf8))
        #expect(decoded.id == "a1")
        #expect(decoded.filename == "x.png")
    }

    @Test("missing optional keys decode to nil (minimal v1 line)")
    func minimalLineDecodesNils() throws {
        let json = """
        {"id":"a1","relativePath":"assets/a1/x.png","kind":"image","filename":"x.png",
         "fileSize":100,"createdAt":1.0,"modifiedAt":1.0,"ingestedAt":1.0,"orphaned":false,
         "rating":0,"favorite":false,"archivedAt":2.0}
        """
        let decoded = try JSONDecoder().decode(ArchivedAsset.self, from: Data(json.utf8))
        #expect(decoded.sidecarRelativePath == nil)
        #expect(decoded.thumbnailRelativePath == nil)
        #expect(decoded.originalPath == nil)
        #expect(decoded.sha256 == nil)
        #expect(decoded.width == nil)
        #expect(decoded.height == nil)
        #expect(decoded.prompt == nil)
        #expect(decoded.negativePrompt == nil)
        #expect(decoded.seed == nil)
        #expect(decoded.steps == nil)
        #expect(decoded.guidance == nil)
        #expect(decoded.modelFamily == nil)
        #expect(decoded.contentMode == nil)
        #expect(decoded.characterName == nil)
        #expect(decoded.source == nil)
        #expect(decoded.folderId == nil)
    }

    // MARK: - Schema version policy

    @Test("schemaVersion 2 is rejected as unsupported")
    func rejectsFutureVersion() throws {
        let json = """
        {"schemaVersion":2,"archiveId":"id","name":"n","createdAt":1.0,
         "producer":"ComfyBoxDesktop","assetCount":0,"totalBytes":0,"folders":[]}
        """
        #expect(throws: ArchiveError.unsupportedSchemaVersion(2)) {
            _ = try ArchiveManifest.decode(Data(json.utf8))
        }
    }

    @Test("schemaVersion 1 is accepted")
    func acceptsCurrentVersion() throws {
        let json = """
        {"schemaVersion":1,"archiveId":"id","name":"n","createdAt":1.0,
         "producer":"ComfyBoxDesktop","assetCount":0,"totalBytes":0,"folders":[]}
        """
        let manifest = try ArchiveManifest.decode(Data(json.utf8))
        #expect(manifest.schemaVersion == 1)
        #expect(manifest.archiveId == "id")
    }

    @Test("missing schemaVersion is rejected as unreadable")
    func rejectsMissingVersion() throws {
        let json = """
        {"archiveId":"id","name":"n","createdAt":1.0,
         "producer":"ComfyBoxDesktop","assetCount":0,"totalBytes":0,"folders":[]}
        """
        #expect(throws: ArchiveError.unreadableManifest) {
            _ = try ArchiveManifest.decode(Data(json.utf8))
        }
    }

    @Test("schemaVersion below 1 is rejected as unreadable")
    func rejectsZeroVersion() throws {
        let json = """
        {"schemaVersion":0,"archiveId":"id","name":"n","createdAt":1.0,
         "producer":"ComfyBoxDesktop","assetCount":0,"totalBytes":0,"folders":[]}
        """
        #expect(throws: ArchiveError.unreadableManifest) {
            _ = try ArchiveManifest.decode(Data(json.utf8))
        }
    }

    // MARK: - JSONL codec

    @Test("ArchiveJSONL.read handles 3 good lines + 1 malformed")
    func readSkipsMalformedLines() throws {
        let good = """
        {"id":"a1","relativePath":"assets/a1/x.png","kind":"image","filename":"x.png",\
        "fileSize":1,"createdAt":1.0,"modifiedAt":1.0,"ingestedAt":1.0,"orphaned":false,\
        "rating":0,"favorite":false,"archivedAt":1.0}
        """
        let lines = [
            good.replacingOccurrences(of: "\"a1\"", with: "\"a1\""),
            good.replacingOccurrences(of: "\"a1\"", with: "\"a2\""),
            "{ this is not valid json ",
            good.replacingOccurrences(of: "\"a1\"", with: "\"a3\""),
        ]
        let tmpPath = NSTemporaryDirectory() + "archive-jsonl-\(UUID().uuidString).jsonl"
        try lines.joined(separator: "\n").appending("\n").write(toFile: tmpPath, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(atPath: tmpPath) }

        var seenIds: [String] = []
        let result = try ArchiveJSONL.read(at: tmpPath) { entry in
            seenIds.append(entry.id)
        }
        #expect(result.count == 3)
        #expect(result.skipped == 1)
        #expect(seenIds == ["a1", "a2", "a3"])
    }

    // MARK: - Path traversal guard

    @Test("traversal guard rejects escaping relative paths", arguments: [
        "../../etc/passwd",
        "/etc/passwd",
        "assets/../../x",
    ])
    func rejectsTraversal(_ relativePath: String) throws {
        let root = URL(fileURLWithPath: "/tmp/some-bundle.cbarchive")
        #expect(throws: (any Error).self) {
            _ = try ArchivePaths.resolveEntryPath(relativePath, in: root)
        }
    }

    @Test("traversal guard accepts a well-formed relative path")
    func acceptsWellFormedPath() throws {
        let root = URL(fileURLWithPath: "/tmp/some-bundle.cbarchive")
        let resolved = try ArchivePaths.resolveEntryPath("assets/9F3C/kira-0042.png", in: root)
        #expect(resolved.path == "/tmp/some-bundle.cbarchive/assets/9F3C/kira-0042.png")
    }
}
