// ArchiveBrowserViewTests.swift — Tests for ArchiveBrowserView's pure
// helpers (T10). The view itself is SwiftUI (no snapshot/unit harness in
// this repo, confirmed by every other *ViewTests.swift file here); this
// covers the two pieces of non-view logic it adds: the entries.jsonl pager
// and the restore-result summary line.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("ArchiveBrowserView.loadEntriesPage")
struct ArchiveBrowserViewEntriesPageTests {
    private func makeEntriesFile(count: Int) throws -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("archivebrowser-entries-test-\(UUID().uuidString).jsonl")
        FileManager.default.createFile(atPath: path, contents: nil)
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: path))
        for i in 0..<count {
            let asset = TestData.makeAsset(id: "asset-\(i)", filename: "photo-\(i).png")
            let entry = ArchivedAsset(from: asset, folderId: nil, relativeRoot: "assets/asset-\(i)")
            try handle.write(contentsOf: ArchiveJSONL.encodeLine(entry))
        }
        try handle.close()
        return path
    }

    @Test("first page of a file smaller than the page size returns everything and reports last page")
    func firstPageSmallerThanLimit() throws {
        let path = try makeEntriesFile(count: 10)
        let page = try ArchiveBrowserView.loadEntriesPage(entriesPath: path, skip: 0, limit: 500)
        #expect(page.entries.count == 10)
        #expect(page.isLastPage == true)
        #expect(page.entries.map(\.id) == (0..<10).map { "asset-\($0)" })
    }

    @Test("a file exactly the page size in length is still the last page")
    func exactlyOnePage() throws {
        let path = try makeEntriesFile(count: 500)
        let page = try ArchiveBrowserView.loadEntriesPage(entriesPath: path, skip: 0, limit: 500)
        #expect(page.entries.count == 500)
        #expect(page.isLastPage == true)
    }

    @Test("a file one row past the page size reports a non-last first page of exactly the limit")
    func onePastLimitNotLastPage() throws {
        let path = try makeEntriesFile(count: 501)
        let page = try ArchiveBrowserView.loadEntriesPage(entriesPath: path, skip: 0, limit: 500)
        #expect(page.entries.count == 500)
        #expect(page.isLastPage == false)
    }

    @Test("second page picks up exactly where the first left off, with no gap or duplicate")
    func secondPageContinuesFromFirst() throws {
        let path = try makeEntriesFile(count: 750)
        let first = try ArchiveBrowserView.loadEntriesPage(entriesPath: path, skip: 0, limit: 500)
        #expect(first.entries.count == 500)
        #expect(first.isLastPage == false)

        let second = try ArchiveBrowserView.loadEntriesPage(entriesPath: path, skip: first.entries.count, limit: 500)
        #expect(second.entries.count == 250)
        #expect(second.isLastPage == true)

        let combinedIds = (first.entries + second.entries).map(\.id)
        #expect(combinedIds == (0..<750).map { "asset-\($0)" })
        #expect(Set(combinedIds).count == 750)   // no duplicates
    }

    @Test("skipping past the end of the file returns an empty last page")
    func skipPastEnd() throws {
        let path = try makeEntriesFile(count: 10)
        let page = try ArchiveBrowserView.loadEntriesPage(entriesPath: path, skip: 20, limit: 500)
        #expect(page.entries.isEmpty)
        #expect(page.isLastPage == true)
    }

    @Test("an empty file returns an empty last page")
    func emptyFile() throws {
        let path = try makeEntriesFile(count: 0)
        let page = try ArchiveBrowserView.loadEntriesPage(entriesPath: path, skip: 0, limit: 500)
        #expect(page.entries.isEmpty)
        #expect(page.isLastPage == true)
    }

    @Test("a missing entries.jsonl throws rather than returning an empty page")
    func missingFileThrows() {
        let missingPath = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("does-not-exist-\(UUID().uuidString).jsonl")
        #expect(throws: (any Error).self) {
            try ArchiveBrowserView.loadEntriesPage(entriesPath: missingPath, skip: 0, limit: 500)
        }
    }
}

@Suite("ArchiveBrowserView.restoreSummaryLine")
struct ArchiveBrowserViewRestoreSummaryTests {
    @Test("a clean restore with nothing else shows only the restored count")
    func onlyRestored() {
        let line = ArchiveBrowserView.restoreSummaryLine(.init(restored: 40))
        #expect(line == "Restored 40")
    }

    @Test("every non-zero clause is appended in order, separated by middle dots")
    func allClauses() {
        let line = ArchiveBrowserView.restoreSummaryLine(
            .init(restored: 40, skipped: 2, reIdentified: 1, renamed: 3, failed: ["a.png"])
        )
        #expect(line == "Restored 40 · 2 skipped · 1 re-identified · 3 renamed · 1 failed")
    }

    @Test("zero-count clauses are omitted entirely")
    func zeroClausesOmitted() {
        let line = ArchiveBrowserView.restoreSummaryLine(.init(restored: 5, skipped: 0, reIdentified: 0, renamed: 0, failed: []))
        #expect(line == "Restored 5")
        #expect(!line.contains("skipped"))
        #expect(!line.contains("re-identified"))
        #expect(!line.contains("renamed"))
        #expect(!line.contains("failed"))
    }

    @Test("multiple failures count correctly, not by joining filenames")
    func multipleFailures() {
        let line = ArchiveBrowserView.restoreSummaryLine(.init(restored: 0, failed: ["a.png", "b.png", "c.png"]))
        #expect(line == "Restored 0 · 3 failed")
    }
}
