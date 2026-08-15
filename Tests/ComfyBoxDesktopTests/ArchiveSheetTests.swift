// ArchiveSheetTests.swift — Tests for ArchiveSheet's pure summary-line helper.
// The sheet itself is a SwiftUI view (no snapshot/unit harness in this repo,
// confirmed by Tests/ having no view tests); this covers the one piece of
// non-view logic it adds.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("ArchiveSheet.summaryLine")
struct ArchiveSheetTests {

    @Test("no secured assets omits the trailing clause")
    func noSecured() {
        let line = ArchiveSheet.summaryLine(assetCount: 412, totalBytes: 3_000_000_000, securedCount: 0)
        #expect(line.contains("412 images"))
        #expect(line.contains("GB"))
        #expect(!line.contains("secured"))
    }

    @Test("secured assets appended as a will-be-skipped clause")
    func withSecured() {
        let line = ArchiveSheet.summaryLine(assetCount: 412, totalBytes: 3_000_000_000, securedCount: 3)
        #expect(line.contains("412 images"))
        #expect(line.contains("3 secured images will be skipped"))
    }

    @Test("singular wording for a single asset and a single secured asset")
    func singularWording() {
        let line = ArchiveSheet.summaryLine(assetCount: 1, totalBytes: 1_024, securedCount: 1)
        #expect(line.contains("1 image ·"))
        #expect(line.contains("1 secured image will be skipped"))
        #expect(!line.contains("1 images"))
        #expect(!line.contains("1 secured images"))
    }

    @Test("zero assets still produces a well-formed line")
    func zeroAssets() {
        let line = ArchiveSheet.summaryLine(assetCount: 0, totalBytes: 0, securedCount: 0)
        #expect(line.hasPrefix("0 images"))
    }
}
