import XCTest
@testable import ComfyBoxDesktop

final class FuzzyMatcherTests: XCTestCase {

    func testExactAndSubsequenceMatch() {
        XCTAssertNotNil(FuzzyMatcher.score("gen", "Generate"))
        XCTAssertNotNil(FuzzyMatcher.score("glry", "Gallery"))   // subsequence
        XCTAssertNil(FuzzyMatcher.score("zzz", "Generate"))
    }

    func testEmptyQueryMatches() {
        XCTAssertEqual(FuzzyMatcher.score("", "anything"), 0)
    }

    func testContiguousBeatsScattered() {
        // "Generate" = contiguous g-e-n; "gaeanb" spreads them with no word
        // boundaries, so it can't match the compounding run bonus.
        let contiguous = FuzzyMatcher.score("gen", "Generate")!
        let scattered = FuzzyMatcher.score("gen", "gaeanb")!
        XCTAssertGreaterThan(contiguous, scattered)
    }

    func testWordStartBoost() {
        // "rc" hits the start of "Restart Comfybox" words → beats mid-word.
        let boundary = FuzzyMatcher.score("rc", "Restart Comfybox")!
        let midword = FuzzyMatcher.score("rc", "arcane")!
        XCTAssertGreaterThan(boundary, midword)
    }

    func testRankFiltersAndOrders() {
        let cmds = [
            PaletteCommand(title: "Gallery", systemImage: "photo", action: {}),
            PaletteCommand(title: "Generate", systemImage: "wand", action: {}),
            PaletteCommand(title: "Health", systemImage: "heart", action: {}),
        ]
        let ranked = FuzzyMatcher.rank(cmds, query: "gen")
        XCTAssertEqual(ranked.first?.title, "Generate")
        XCTAssertFalse(ranked.contains { $0.title == "Health" })
    }

    func testRankEmptyQueryKeepsAll() {
        let cmds = [
            PaletteCommand(title: "A", systemImage: "a", action: {}),
            PaletteCommand(title: "B", systemImage: "b", action: {}),
        ]
        XCTAssertEqual(FuzzyMatcher.rank(cmds, query: "  ").count, 2)
    }

    func testKeywordsAreSearchable() {
        let cmd = PaletteCommand(title: "Restart Server", systemImage: "x",
                                 keywords: ["daemon", "launchctl"], action: {})
        XCTAssertNotNil(FuzzyMatcher.score("daemon", cmd.haystack))
    }
}
