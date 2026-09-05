// DAMAssetConstructionSiteLintTests.swift — #372.
//
// `DAMAsset.init` defaults all 23 parameters, so a hand-rebuilt
// `DAMAsset(...)` literal that forgets a field compiles silently instead of
// failing to build — `source` was dropped this way at three separate call
// sites, twice (#268). `DAMAsset.copy(with:)` fixes the call sites that are
// COPIES of an existing asset, but nothing stops a future edit from adding
// another hand-rebuilt `DAMAsset(...)` literal instead of reaching for it.
//
// This test scans every .swift file under Sources/ComfyBoxDesktop for
// `DAMAsset(` construction literals and fails if the count for any file
// differs from the allowlist below — so a new site (or a moved/removed one)
// is a deliberate, reviewed, one-line change to the allowlist rather than a
// silent reintroduction of the bug class.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("DAMAsset construction-site lint (#372)")
struct DAMAssetConstructionSiteLintTests {
    private static let sourcesDirectory = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent() // ComfyBoxDesktopTests/
        .deletingLastPathComponent() // Tests/
        .deletingLastPathComponent() // repo root
        .appendingPathComponent("Sources/ComfyBoxDesktop")

    /// Files allowed to hand-construct `DAMAsset(...)` directly, and exactly
    /// how many times. None of these are copies of an existing `DAMAsset`
    /// instance, so `copy(with:)` does not apply to them:
    ///   - row/record mapping from a different, external representation
    ///     (a SQL row, an archive manifest entry), or a different in-memory
    ///     type (a catalog row); or
    ///   - a fresh ingest with no prior `DAMAsset` to copy from; or
    ///   - `copy(with:)`'s own implementation — the one place that must
    ///     enumerate every field, covered directly by the field-preservation
    ///     tests in DAMAssetTests.swift.
    /// Bump a count (or add a file) only for a new, reviewed reason — never
    /// to silence this test. A copy of an existing asset belongs in
    /// `DAMAsset.copy(with:)`, not in this allowlist.
    private static let allowlist: [String: Int] = [
        "DAM/DAMAsset.swift": 1,             // copy(with:) — the canonical field-by-field builder.
        "DAM/DAMStore.swift": 1,             // assetFromRow — maps a raw SQL row, not a DAMAsset copy.
        "DAM/AssetIngestor.swift": 1,        // buildAsset — fresh ingest from a file on disk.
        "DAM/CatalogBrowser.swift": 1,       // damAsset(for: CatalogAsset) — converts a different type.
        "Archive/ArchiveManifest.swift": 1   // toDAMAsset — deserializes an archived manifest entry.
    ]

    /// Matches a `DAMAsset(` construction literal, including `DAMAsset.init(`
    /// and any amount of whitespace/newlines between `DAMAsset` and the
    /// paren (e.g. a call broken across lines) — but not a suffixed
    /// identifier like `toDAMAsset(` or a member access like
    /// `DAMAsset.Mutation(`: the character immediately before "DAMAsset"
    /// must not be a letter, digit, or underscore, and the only thing
    /// allowed between "DAMAsset" and "(" is an optional literal ".init"
    /// plus whitespace. `constructionPatternMatchesAllConstructionForms`
    /// tests this pattern directly against each of these forms.
    private static let constructionPattern = try! NSRegularExpression(
        pattern: "(?<![A-Za-z0-9_])DAMAsset(?:\\.init)?\\s*\\("
    )

    @Test("construction pattern matches DAMAsset(, DAMAsset.init(, and whitespace/newline before the paren, but not toDAMAsset( or DAMAsset.Mutation(")
    func constructionPatternMatchesAllConstructionForms() {
        let shouldMatch = [
            "DAMAsset(",
            "DAMAsset.init(",
            "DAMAsset (",
            "DAMAsset\n(",
            "DAMAsset.init\n(",
            "  return DAMAsset(\n    id: id\n  )",
        ]
        for sample in shouldMatch {
            let range = NSRange(sample.startIndex..<sample.endIndex, in: sample)
            let matches = Self.constructionPattern.numberOfMatches(in: sample, range: range)
            #expect(matches == 1, Comment(rawValue: "expected exactly one match in \(sample.debugDescription), found \(matches)"))
        }

        let shouldNotMatch = [
            "toDAMAsset(",
            "entry.toDAMAsset(absolutePath: path)",
            "DAMAsset.Mutation(",
            "DAMAsset.Mutation()",
            "someDAMAsset(",
            "-> DAMAsset {",
        ]
        for sample in shouldNotMatch {
            let range = NSRange(sample.startIndex..<sample.endIndex, in: sample)
            let matches = Self.constructionPattern.numberOfMatches(in: sample, range: range)
            #expect(matches == 0, Comment(rawValue: "expected no match in \(sample.debugDescription), found \(matches)"))
        }
    }

    @Test("no DAMAsset(...) construction literal outside the allowlist")
    func onlyAllowlistedSitesConstructDAMAssetDirectly() throws {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: Self.sourcesDirectory, includingPropertiesForKeys: nil
        ) else {
            Issue.record("Could not enumerate \(Self.sourcesDirectory.path) — is the test running from the package checkout?")
            return
        }

        var counts: [String: Int] = [:]
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            let matches = Self.constructionPattern.numberOfMatches(in: text, range: range)
            guard matches > 0 else { continue }
            let relativePath = url.path.replacingOccurrences(
                of: Self.sourcesDirectory.path + "/", with: ""
            )
            counts[relativePath] = matches
        }

        #expect(
            !counts.isEmpty,
            Comment(rawValue:
                "Scan found zero DAMAsset( sites under \(Self.sourcesDirectory.path) — "
                + "the scan path is almost certainly wrong, not that the codebase has none.")
        )

        // Every occurrence found must be allowlisted for exactly that count —
        // an unexpected new occurrence (new file, or an already-allowlisted
        // file with one more than expected) fails here.
        for (file, count) in counts.sorted(by: { $0.key < $1.key }) {
            let allowed = Self.allowlist[file] ?? 0
            #expect(
                count == allowed,
                Comment(rawValue:
                    "\(file): found \(count) DAMAsset(...) construction site(s), allowlisted for \(allowed). "
                    + "A copy of an existing DAMAsset must use DAMAsset.copy(with:) instead of a hand-rebuilt "
                    + "literal (see #372/#268). A genuinely new ingest/row-mapping site is a deliberate, "
                    + "reviewed addition to DAMAssetConstructionSiteLintTests.allowlist.")
            )
        }

        // And every allowlisted entry must still be backed by a real
        // occurrence at that count — a site that moved, was migrated to
        // copy(with:), or was deleted must update the allowlist too.
        for (file, allowed) in Self.allowlist {
            #expect(
                counts[file] == allowed,
                Comment(rawValue:
                    "\(file): allowlisted for \(allowed) DAMAsset(...) site(s) but found \(counts[file] ?? 0) — "
                    + "update the allowlist, this file's site moved, was migrated, or was removed.")
            )
        }
    }
}
