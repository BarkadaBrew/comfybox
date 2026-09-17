// FolderGalleryIndexTests.swift — the gallery that lives on the drive
// (FDD-remote-galleries §3.2).
//
// Todd 2026-09-17: "a gallery that I can send images and videos to but not as
// an archive." So: plain files under their own names, a sidecar beside each,
// and an index any machine can read — not a bundle that has to be restored.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("FolderGalleryIndex")
struct FolderGalleryIndexTests {

    private func tempRoot() -> String {
        (NSTemporaryDirectory() as NSString).appendingPathComponent("rg-\(UUID().uuidString)")
    }

    private func entry(_ id: String, _ relative: String, kind: String = "image") -> FolderGalleryIndex.Entry {
        FolderGalleryIndex.Entry(assetID: id, mediaPath: relative, sidecarPath: relative + ".json",
                                 thumbnailPath: "thumbnails/\(id).jpg", kind: kind,
                                 filename: (relative as NSString).lastPathComponent,
                                 fileSize: 10, sha256: "abc", sentAt: Date(timeIntervalSince1970: 1_700_000_000))
    }

    @Test("creating a gallery lays out its directories and an empty index")
    func create() throws {
        let root = tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        let gallery = try FolderGalleryIndex.create(at: root, name: "Vault SSD")
        #expect(gallery.name == "Vault SSD")
        #expect(gallery.entries.isEmpty)
        for sub in ["media", "sidecars", "thumbnails"] {
            #expect(FileManager.default.fileExists(atPath: (root as NSString).appendingPathComponent(sub)), "missing \(sub)")
        }
        #expect(FileManager.default.fileExists(atPath: FolderGalleryIndex.indexPath(inGalleryRoot: root)))
    }

    @Test("creating over an existing gallery opens it instead of wiping it")
    func createIsIdempotent() throws {
        let root = tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        var gallery = try FolderGalleryIndex.create(at: root, name: "Vault SSD")
        try gallery.append(entry("a", "media/2026/09/a.png"))

        let reopened = try FolderGalleryIndex.create(at: root, name: "Ignored")
        #expect(reopened.entries.count == 1, "an existing gallery is never re-created empty")
        #expect(reopened.name == "Vault SSD", "its own name wins")
        #expect(reopened.id == gallery.id)
    }

    @Test("entries survive a reload, in order")
    func appendAndReload() throws {
        let root = tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        var gallery = try FolderGalleryIndex.create(at: root, name: "Vault")
        try gallery.append(entry("a", "media/2026/09/a.png"))
        try gallery.append(entry("b", "media/2026/09/b.mp4", kind: "video"))

        let reloaded = try FolderGalleryIndex.load(fromGalleryRoot: root)
        #expect(reloaded.entries.map(\.assetID) == ["a", "b"])
        #expect(reloaded.entries.last?.kind == "video")
    }

    @Test("appending the same asset twice replaces its entry")
    func appendIsIdempotentPerAsset() throws {
        let root = tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        var gallery = try FolderGalleryIndex.create(at: root, name: "Vault")
        try gallery.append(entry("a", "media/2026/09/a.png"))
        try gallery.append(entry("a", "media/2026/09/a-1.png"))
        #expect(gallery.entries.count == 1, "a resumed transfer must not double-list its asset")
        #expect(gallery.entries.first?.mediaPath == "media/2026/09/a-1.png")
    }

    @Test("a half-written index never replaces a good one")
    func atomicWrite() throws {
        let root = tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        var gallery = try FolderGalleryIndex.create(at: root, name: "Vault")
        try gallery.append(entry("a", "media/2026/09/a.png"))

        // A leftover temp file from an interrupted write is ignored.
        let temp = FolderGalleryIndex.indexPath(inGalleryRoot: root) + ".tmp"
        FileManager.default.createFile(atPath: temp, contents: Data("{ this is not json".utf8))

        let reloaded = try FolderGalleryIndex.load(fromGalleryRoot: root)
        #expect(reloaded.entries.count == 1)
    }

    @Test("the media path is dated, keeps the original name, and never collides")
    func mediaPaths() {
        let sent = Date(timeIntervalSince1970: 1_757_000_000)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .current
        let parts = calendar.dateComponents([.year, .month], from: sent)
        let expectedFolder = String(format: "media/%04d/%02d/", parts.year!, parts.month!)

        let first = FolderGalleryIndex.mediaRelativePath(filename: "kira-01.png", sentAt: sent, taken: [])
        #expect(first.hasPrefix(expectedFolder), "dated folder: \(first)")
        #expect(first.hasSuffix("kira-01.png"))

        let second = FolderGalleryIndex.mediaRelativePath(filename: "kira-01.png", sentAt: sent, taken: [first])
        #expect(second != first)
        #expect(second.contains("kira-01-1.png"), "a collision takes a numeric suffix: \(second)")

        let third = FolderGalleryIndex.mediaRelativePath(filename: "kira-01.png", sentAt: sent, taken: [first, second])
        #expect(third.contains("kira-01-2.png"))
    }

    @Test("a gallery is recognised by its index, not by its folder name")
    func detection() throws {
        let root = tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }
        #expect(!FolderGalleryIndex.exists(atGalleryRoot: root))
        _ = try FolderGalleryIndex.create(at: root, name: "Vault")
        #expect(FolderGalleryIndex.exists(atGalleryRoot: root))
    }
}
