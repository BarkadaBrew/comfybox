// ArchiveStoreTests.swift — Tests for ArchiveStore (T5)
//
// Covers reload() over hand-built fixture directories: listing across
// multiple roots, decoding manifest.json headers only (never entries.jsonl),
// isIncomplete/hasPendingRemoval flags, ignoring non-.cbarchive entries,
// newest-first ordering, and the marker-only (INCOMPLETE.json, no
// manifest.json) crash-recovery case.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("ArchiveStore")
struct ArchiveStoreTests {
    private func makeTempDir(_ name: String) -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("archivestore-test-\(UUID().uuidString)-\(name)")
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }

    private func manifest(name: String, createdAt: Double, assetCount: Int = 1, totalBytes: Int64 = 100) -> ArchiveManifest {
        ArchiveManifest(
            archiveId: UUID().uuidString, name: name, createdAt: createdAt,
            producer: "test", assetCount: assetCount, totalBytes: totalBytes, folders: []
        )
    }

    /// Builds a `<dirName>` bundle directory in `root` with an optional
    /// manifest.json and optional INCOMPLETE.json / PENDING_REMOVAL.json
    /// markers. Returns the bundle's full path.
    @discardableResult
    private func makeBundle(
        in root: String,
        dirName: String,
        manifest: ArchiveManifest? = nil,
        incomplete: Bool = false,
        pendingRemoval: Bool = false
    ) throws -> String {
        let bundlePath = (root as NSString).appendingPathComponent(dirName)
        try FileManager.default.createDirectory(atPath: bundlePath, withIntermediateDirectories: true)
        if let manifest {
            let data = try JSONEncoder().encode(manifest)
            try data.write(to: URL(fileURLWithPath: (bundlePath as NSString).appendingPathComponent("manifest.json")))
        }
        if incomplete {
            FileManager.default.createFile(
                atPath: (bundlePath as NSString).appendingPathComponent("INCOMPLETE.json"),
                contents: Data(#"{"startedAt":0,"name":"x","assetCount":0}"#.utf8)
            )
        }
        if pendingRemoval {
            FileManager.default.createFile(
                atPath: (bundlePath as NSString).appendingPathComponent("PENDING_REMOVAL.json"),
                contents: Data(#"{"assetIds":[]}"#.utf8)
            )
        }
        return bundlePath
    }

    // MARK: - Listing across roots

    @Test("reload lists bundles across two roots")
    @MainActor
    func listsAcrossTwoRoots() async throws {
        let root1 = makeTempDir("root1")
        let root2 = makeTempDir("root2")
        try makeBundle(in: root1, dirName: "one-20260101-000000.cbarchive", manifest: manifest(name: "One", createdAt: 100))
        try makeBundle(in: root2, dirName: "two-20260101-000000.cbarchive", manifest: manifest(name: "Two", createdAt: 200))

        let store = ArchiveStore(roots: [root1, root2])
        await store.reload()

        #expect(store.archives.count == 2)
        #expect(Set(store.archives.map { $0.manifest.name }) == ["One", "Two"])
        #expect(store.error == nil)
    }

    // MARK: - Header-only decode

    @Test("reload decodes only manifest.json, never touching entries.jsonl")
    @MainActor
    func decodesHeaderOnly() async throws {
        let root = makeTempDir("root")
        let bundlePath = try makeBundle(
            in: root, dirName: "test-20260101-000000.cbarchive",
            manifest: manifest(name: "Test", createdAt: 100, assetCount: 5, totalBytes: 5000)
        )
        // Deliberately corrupt entries.jsonl — reload() must not choke on it.
        FileManager.default.createFile(
            atPath: (bundlePath as NSString).appendingPathComponent("entries.jsonl"),
            contents: Data("not valid jsonl at all {{{".utf8)
        )

        let store = ArchiveStore(roots: [root])
        await store.reload()

        #expect(store.archives.count == 1)
        #expect(store.archives.first?.manifest.assetCount == 5)
        #expect(store.archives.first?.manifest.totalBytes == 5000)
        #expect(store.error == nil)
    }

    // MARK: - Marker flags

    @Test("reload flags isIncomplete and hasPendingRemoval independently")
    @MainActor
    func flagsMarkers() async throws {
        let root = makeTempDir("root")
        try makeBundle(in: root, dirName: "plain-20260101-000000.cbarchive", manifest: manifest(name: "Plain", createdAt: 100))
        try makeBundle(
            in: root, dirName: "incomplete-20260101-000000.cbarchive",
            manifest: manifest(name: "Incomplete", createdAt: 200), incomplete: true
        )
        try makeBundle(
            in: root, dirName: "pending-20260101-000000.cbarchive",
            manifest: manifest(name: "Pending", createdAt: 300), pendingRemoval: true
        )

        let store = ArchiveStore(roots: [root])
        await store.reload()

        #expect(store.archives.count == 3)
        let byName = Dictionary(uniqueKeysWithValues: store.archives.map { ($0.manifest.name, $0) })
        #expect(byName["Plain"]?.isIncomplete == false)
        #expect(byName["Plain"]?.hasPendingRemoval == false)
        #expect(byName["Incomplete"]?.isIncomplete == true)
        #expect(byName["Incomplete"]?.hasPendingRemoval == false)
        #expect(byName["Pending"]?.isIncomplete == false)
        #expect(byName["Pending"]?.hasPendingRemoval == true)
    }

    // MARK: - Non-bundle entries ignored

    @Test("reload ignores non-.cbarchive directories and stray files")
    @MainActor
    func ignoresNonBundles() async throws {
        let root = makeTempDir("root")
        try makeBundle(in: root, dirName: "real-20260101-000000.cbarchive", manifest: manifest(name: "Real", createdAt: 100))
        try FileManager.default.createDirectory(
            atPath: (root as NSString).appendingPathComponent("not-a-bundle"), withIntermediateDirectories: true
        )
        FileManager.default.createFile(atPath: (root as NSString).appendingPathComponent("stray.txt"), contents: Data("hi".utf8))
        // A *file* (not a directory) that happens to end in .cbarchive must not be listed.
        FileManager.default.createFile(
            atPath: (root as NSString).appendingPathComponent("fakefile.cbarchive"), contents: Data("not a dir".utf8)
        )

        let store = ArchiveStore(roots: [root])
        await store.reload()

        #expect(store.archives.count == 1)
        #expect(store.archives.first?.manifest.name == "Real")
    }

    // MARK: - Ordering

    @Test("reload orders archives newest-first by manifest.createdAt")
    @MainActor
    func newestFirstOrdering() async throws {
        let root = makeTempDir("root")
        try makeBundle(in: root, dirName: "old-20260101-000000.cbarchive", manifest: manifest(name: "Old", createdAt: 100))
        try makeBundle(in: root, dirName: "new-20260102-000000.cbarchive", manifest: manifest(name: "New", createdAt: 300))
        try makeBundle(in: root, dirName: "mid-20260101-120000.cbarchive", manifest: manifest(name: "Mid", createdAt: 200))

        let store = ArchiveStore(roots: [root])
        await store.reload()

        #expect(store.archives.map { $0.manifest.name } == ["New", "Mid", "Old"])
    }

    // MARK: - Marker-only bundle (crash before manifest.json was written)

    @Test("a bundle with only INCOMPLETE.json (no manifest.json yet) is still listed with a synthesized summary")
    @MainActor
    func markerOnlyBundleListed() async throws {
        let root = makeTempDir("root")
        let bundlePath = try makeBundle(
            in: root, dirName: "crashed-20260101-000000.cbarchive", manifest: nil, incomplete: true
        )

        let store = ArchiveStore(roots: [root])
        await store.reload()

        #expect(store.archives.count == 1)
        let summary = try #require(store.archives.first)
        #expect(summary.bundlePath == bundlePath)
        #expect(summary.id == bundlePath)
        #expect(summary.isIncomplete == true)
        #expect(summary.hasPendingRemoval == false)
        #expect(summary.manifest.assetCount == 0)
        #expect(store.error == nil)
    }

    // MARK: - Unreadable manifest, no INCOMPLETE marker: skipped + surfaced

    @Test("a bundle with an unreadable manifest and no INCOMPLETE marker is skipped and surfaced via error")
    @MainActor
    func unreadableManifestSkippedWithError() async throws {
        let root = makeTempDir("root")
        let bundlePath = (root as NSString).appendingPathComponent("bad-20260101-000000.cbarchive")
        try FileManager.default.createDirectory(atPath: bundlePath, withIntermediateDirectories: true)
        try Data("not json at all".utf8).write(
            to: URL(fileURLWithPath: (bundlePath as NSString).appendingPathComponent("manifest.json"))
        )

        let store = ArchiveStore(roots: [root])
        await store.reload()

        #expect(store.archives.isEmpty)
        #expect(store.error != nil)
    }

    // MARK: - Default init reads DesktopSettings

    @Test("default init falls back to DesktopSettings.defaultArchiveRoot when archiveRoots is unset")
    @MainActor
    func defaultInitUsesSettingsFallback() {
        // Isolated from whatever ~/.comfybox/desktop-config.json exists on
        // this machine would be flaky; this only exercises the no-settings
        // (missing config file) path, which load() also falls back through.
        let store = ArchiveStore()
        #expect(!store.roots.isEmpty)
    }

    // MARK: - Export as Zip (T12)

    @Test("dittoArguments builds the exact ditto invocation vector")
    func dittoArgumentsExactVector() {
        let args = ArchiveStore.dittoArguments(
            bundlePath: "/Volumes/Bolt/ComfyBox/foo-20260101-000000.cbarchive",
            destination: "/Users/todd/Desktop/foo-20260101-000000.zip"
        )
        #expect(args == [
            "-c", "-k", "--sequesterRsrc", "--keepParent",
            "/Volumes/Bolt/ComfyBox/foo-20260101-000000.cbarchive",
            "/Users/todd/Desktop/foo-20260101-000000.zip",
        ])
    }

    @Test("exportAsZip shells ditto and produces a real zip of the bundle directory")
    @MainActor
    func exportAsZipProducesRealZip() async throws {
        let root = makeTempDir("export-root")
        let bundlePath = try makeBundle(
            in: root, dirName: "export-me-20260101-000000.cbarchive",
            manifest: manifest(name: "Export Me", createdAt: 100)
        )
        let destDir = makeTempDir("export-dest")
        let destination = (destDir as NSString).appendingPathComponent("out.zip")

        let store = ArchiveStore(roots: [root])
        try await store.exportAsZip(bundlePath: bundlePath, destination: destination)

        #expect(FileManager.default.fileExists(atPath: destination))
    }

    @Test("exportAsZip throws ExportError.failed when ditto exits non-zero")
    @MainActor
    func exportAsZipThrowsOnFailure() async {
        let store = ArchiveStore(roots: [])
        let missingBundle = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).cbarchive")
        let destination = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("wont-be-written-\(UUID().uuidString).zip")

        await #expect(throws: ArchiveStore.ExportError.self) {
            try await store.exportAsZip(bundlePath: missingBundle, destination: destination)
        }
    }
}
