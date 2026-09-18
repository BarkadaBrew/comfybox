import Foundation
import XCTest

@testable import ZImage

/// Placement: how a piece of library material gets into a prompt, and what the
/// composer reports back. Adapted from ComfyUI-SickOllie's placement state
/// machine (behaviour, not code).
final class LibraryComposerTests: XCTestCase {

  private let body = "a woman wearing {OUTFIT}, in {SCENE}, shot on {CAMERA}"

  private func use(
    _ slot: String, _ value: String, _ placement: LibraryPlacement = .smart, id: String? = nil
  ) -> LibraryComponentUse {
    LibraryComponentUse(slot: slot, value: value, placement: placement, itemId: id)
  }

  // MARK: smart

  func testSmartSubstitutesWhenTheMarkerIsThere() {
    let result = LibraryComposer.compose(
      template: body, components: [use("OUTFIT", "a grey tee")])
    XCTAssertEqual(result.prompt, "a woman wearing a grey tee, in {SCENE}, shot on {CAMERA}")
    XCTAssertEqual(result.placements.first?.action, .substituted)
    XCTAssertEqual(result.unfilled, ["CAMERA", "SCENE"], "what is still open stays visible")
  }

  func testSmartAppendsWhenThereIsNoMarker() {
    // The whole point: a wardrobe item is usable against a template that never
    // declared {OUTFIT}.
    let result = LibraryComposer.compose(
      template: "a woman at a counter", components: [use("OUTFIT", "a grey tee")])
    XCTAssertEqual(result.prompt, "a woman at a counter, a grey tee")
    XCTAssertEqual(result.placements.first?.action, .appended)
  }

  // MARK: token, append, prepend

  func testTokenRefusesRatherThanAppending() {
    let result = LibraryComposer.compose(
      template: "a woman at a counter", components: [use("OUTFIT", "a grey tee", .token)])
    XCTAssertEqual(result.prompt, "a woman at a counter", "nothing is placed")
    XCTAssertEqual(result.placements.first?.action, .missingMarker)
    XCTAssertFalse(result.placements.first?.used == true)
  }

  func testAppendAndPrependIgnoreTheMarker() {
    let appended = LibraryComposer.compose(
      template: body, components: [use("OUTFIT", "a grey tee", .append)])
    XCTAssertTrue(appended.prompt.hasSuffix("a grey tee"))
    XCTAssertTrue(appended.prompt.contains("{OUTFIT}"), "append leaves the marker for later")

    let prepended = LibraryComposer.compose(
      template: "a woman at a counter", components: [use("LOOK", "shot on film", .prepend)])
    XCTAssertEqual(prepended.prompt, "shot on film, a woman at a counter")
  }

  // MARK: off

  func testOffRemovesTheMarkerAndItsPunctuation() {
    let result = LibraryComposer.compose(
      template: body, components: [use("SCENE", "", .off)])
    XCTAssertEqual(result.prompt, "a woman wearing {OUTFIT}, shot on {CAMERA}")
    XCTAssertEqual(result.placements.first?.action, .removed)
    XCTAssertFalse(result.prompt.contains(", ,"), "no comma litter where it used to be")
  }

  func testOffOnAMarkerThatIsNotThereIsASkip() {
    let result = LibraryComposer.compose(
      template: "a woman at a counter", components: [use("SCENE", "", .off)])
    XCTAssertEqual(result.prompt, "a woman at a counter")
    XCTAssertEqual(result.placements.first?.action, .skipped)
  }

  func testRemovingTheLastMarkerDoesNotLeaveATrailingComma() {
    let result = LibraryComposer.compose(
      template: "a quiet cafe, {SCENE}", components: [use("SCENE", "", .off)])
    XCTAssertEqual(result.prompt, "a quiet cafe")
  }

  // MARK: values and provenance

  func testABlankValueIsSkippedRatherThanErasingTheMarker() {
    let result = LibraryComposer.compose(
      template: body, components: [use("OUTFIT", "   ")])
    XCTAssertTrue(result.prompt.contains("{OUTFIT}"))
    XCTAssertEqual(result.placements.first?.action, .skipped)
  }

  func testEveryPlacedItemIsRecordedForProvenance() {
    let result = LibraryComposer.compose(
      template: body,
      components: [
        use("OUTFIT", "a grey tee", .smart, id: "w1"),
        use("SCENE", "a quiet cafe", .smart, id: "c1"),
        use("CAMERA", "", .off, id: "c2"),
        use("MOOD", "warm", .token, id: "c3"),
      ])
    XCTAssertEqual(result.usedItemIds, ["w1", "c1"], "only what landed is recorded")
    XCTAssertEqual(
      result.placements.map(\.action), [.substituted, .substituted, .removed, .missingMarker])
  }

  func testComponentsApplyInOrder() {
    let result = LibraryComposer.compose(
      template: "a woman",
      components: [use("A", "one", .append), use("B", "two", .append), use("C", "zero", .prepend)])
    XCTAssertEqual(result.prompt, "zero, a woman, one, two")
  }

  // MARK: templates and defaults

  func testATemplatesDefaultsFillWhatTheCallerDidNot() {
    let template = LibraryEntry(
      id: "t1", kind: .template, name: "portrait", value: body,
      slots: [
        LibrarySlot(id: "OUTFIT"),
        LibrarySlot(id: "SCENE", defaultValue: "a quiet cafe"),
        LibrarySlot(id: "CAMERA", defaultValue: "a 50mm lens"),
      ])
    let result = LibraryComposer.compose(
      template: template, components: [use("OUTFIT", "a grey tee")])
    XCTAssertEqual(result.prompt, "a woman wearing a grey tee, in a quiet cafe, shot on a 50mm lens")
    XCTAssertTrue(result.unfilled.isEmpty)
  }

  func testACallerBeatsTheDefault() {
    let template = LibraryEntry(
      id: "t1", kind: .template, name: "p", value: "in {SCENE}",
      slots: [LibrarySlot(id: "SCENE", defaultValue: "a quiet cafe")])
    let result = LibraryComposer.compose(
      template: template, components: [use("SCENE", "a neon alley")])
    XCTAssertEqual(result.prompt, "in a neon alley")
  }

  // MARK: tidying

  func testJoinNeverDoublesASeparator() {
    XCTAssertEqual(LibraryComposer.join("a woman,", "a grey tee"), "a woman, a grey tee")
    XCTAssertEqual(LibraryComposer.join("a woman.", "a grey tee"), "a woman. a grey tee")
    XCTAssertEqual(LibraryComposer.join("", "a grey tee"), "a grey tee")
    XCTAssertEqual(LibraryComposer.join("a woman", ""), "a woman")
  }

  func testTidyCollapsesTheDamage() {
    XCTAssertEqual(LibraryComposer.tidy("a woman ,  in a cafe"), "a woman, in a cafe")
    XCTAssertEqual(LibraryComposer.tidy("a woman,, in a cafe"), "a woman, in a cafe")
    XCTAssertEqual(LibraryComposer.tidy(" a woman in a cafe, "), "a woman in a cafe")
  }
}
