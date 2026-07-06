import XCTest
@testable import ComfyBoxDesktop

final class DecoupageServiceTests: XCTestCase {

    func testGenerateArgsCore() {
        let args = DecoupageService.generateArgs(.init(
            description: "a seated figure", recipe: "/r/oxman.yaml"))
        XCTAssertEqual(args, ["generate", "a seated figure", "--recipe", "/r/oxman.yaml"])
    }

    func testGenerateArgsFull() {
        let args = DecoupageService.generateArgs(.init(
            description: "figure", recipe: "/r/oxman.yaml", output: "/o.png",
            seed: 7, anchor: "/a.png", noAnchor: false, printPrep: true))
        XCTAssertEqual(valueAfter("--output", args), "/o.png")
        XCTAssertEqual(valueAfter("--seed", args), "7")
        XCTAssertEqual(valueAfter("--anchor", args), "/a.png")
        XCTAssertTrue(args.contains("--print-prep"))
        XCTAssertFalse(args.contains("--no-anchor"))
    }

    func testGenerateArgsNoAnchor() {
        let args = DecoupageService.generateArgs(.init(description: "x", recipe: "/r.yaml", noAnchor: true))
        XCTAssertTrue(args.contains("--no-anchor"))
    }

    func testCompositeArgs() {
        let args = DecoupageService.compositeArgs(.init(
            figure: "/fig.png", recipe: "/r.yaml", seed: 3, printPrep: true))
        XCTAssertEqual(args.prefix(4).map { $0 }, ["composite", "/fig.png", "--recipe", "/r.yaml"])
        XCTAssertEqual(valueAfter("--seed", args), "3")
        XCTAssertTrue(args.contains("--print-prep"))
    }

    func testGenElementsArgs() {
        let args = DecoupageService.genElementsArgs(
            recipe: "/r.yaml", category: "florals", prompt: "wallpaper roses", count: 3)
        XCTAssertEqual(args, ["gen-elements", "--recipe", "/r.yaml", "florals", "wallpaper roses", "--count", "3"])
    }

    func testGenElementsCountFloor() {
        let args = DecoupageService.genElementsArgs(recipe: "/r.yaml", category: "c", prompt: "p", count: 0)
        XCTAssertEqual(valueAfter("--count", args), "1")
    }

    private func valueAfter(_ flag: String, _ args: [String]) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
    }
}
