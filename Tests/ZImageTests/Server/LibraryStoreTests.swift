import Foundation
import XCTest

@testable import ZImage

/// The Creative Library store (PRD docs/PRD-creative-library.md, L1).
final class LibraryStoreTests: XCTestCase {

  private var dir: URL!

  override func setUpWithError() throws {
    dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("library-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: dir)
  }

  private func store() -> LibraryStore { LibraryStore(directory: dir) }

  private func wardrobe(_ id: String, _ name: String, category: String = "Tops",
                        subtype: String = "Tees", value: String? = nil) -> LibraryEntry {
    LibraryEntry(id: id, kind: .wardrobeItem, name: name, value: value ?? name,
                category: category, subtype: subtype, itemType: "piece")
  }

  // MARK: persistence

  func testUpsertPersistsAndReloads() throws {
    let s = store()
    try s.upsert(wardrobe("w1", "grey tee"))
    XCTAssertEqual(s.item(id: "w1")?.name, "grey tee")

    let reopened = store()
    XCTAssertEqual(reopened.item(id: "w1")?.category, "Tops")
    XCTAssertEqual(reopened.allItems().count, 1)
  }

  func testUnknownKeysSurviveARoundTrip() throws {
    // #459's lesson: a newer client's field must not be wiped by an older save.
    let json = """
      {"items":[{"id":"w9","kind":"wardrobe_item","name":"future","value":"v",
      "future_field":{"nested":[1,2]},"another":"keep me"}]}
      """
    try Data(json.utf8).write(to: dir.appendingPathComponent("items.json"))
    let s = store()
    XCTAssertEqual(s.item(id: "w9")?.extras["another"], .string("keep me"))
    // Touch an unrelated item, forcing a full rewrite of the file.
    try s.upsert(wardrobe("w1", "grey tee"))
    let reopened = store()
    XCTAssertEqual(reopened.item(id: "w9")?.extras["another"], .string("keep me"))
    XCTAssertNotNil(reopened.item(id: "w9")?.extras["future_field"])
  }

  func testCreatedAtAndUseCountSurviveAnUpdate() throws {
    let s = store()
    let first = try s.upsert(wardrobe("w1", "grey tee"))
    try s.markUsed(id: "w1")
    var edit = wardrobe("w1", "grey tee v2")
    edit.createdAt = Date(timeIntervalSince1970: 0)   // a client that forgot to send it
    let updated = try s.upsert(edit)
    XCTAssertEqual(updated.createdAt, first.createdAt, "createdAt is the store's, not the client's")
    XCTAssertEqual(updated.useCount, 1, "a use count is never reset by an edit")
    XCTAssertEqual(updated.name, "grey tee v2")
  }

  func testMalformedFileIsNotSilentlyEmptied() throws {
    try Data("{ not json".utf8).write(to: dir.appendingPathComponent("items.json"))
    let s = store()
    XCTAssertTrue(s.allItems().isEmpty)
    // The unreadable file is still on disk; nothing overwrote it until a write.
    let raw = try String(contentsOf: dir.appendingPathComponent("items.json"), encoding: .utf8)
    XCTAssertEqual(raw, "{ not json")
  }

  func testBatchUpsertWritesOnceAndReportsRejects() throws {
    let s = store()
    let good = (0..<50).map { wardrobe("w\($0)", "item \($0)") }
    var bad = wardrobe("bad", "nameless")
    bad.name = "  "
    let (saved, rejected) = try s.upsert(contentsOf: good + [bad])
    XCTAssertEqual(saved.count, 50)
    XCTAssertEqual(rejected.count, 1, "one bad item is returned, not thrown — the batch survives")
    XCTAssertEqual(rejected.first?.0.id, "bad")
    XCTAssertEqual(store().allItems().count, 50, "persisted in one write")
  }

  func testBatchUpsertKeepsExistingCreatedAtAndUseCount() throws {
    let s = store()
    let first = try s.upsert(wardrobe("w1", "grey tee"))
    try s.markUsed(id: "w1")
    let (saved, _) = try s.upsert(contentsOf: [wardrobe("w1", "grey tee v2")])
    XCTAssertEqual(saved.first?.createdAt, first.createdAt)
    XCTAssertEqual(saved.first?.useCount, 1)
  }

  func testBatchCollectionUpsertSkipsInvalidOnesAndWritesOnce() throws {
    let s = store()
    let count = try s.upsertCollections([
      LibraryCollection(id: "a", name: "A"),
      LibraryCollection(id: "", name: "no id"),
      LibraryCollection(id: "b", name: "B", parentId: "a"),
      LibraryCollection(id: "c", name: "C", parentId: "c"),
    ])
    XCTAssertEqual(count, 2, "the blank id and the self-parent are skipped")
    XCTAssertEqual(store().allCollections().map(\.id), ["a", "b"])
  }

  // MARK: validation

  func testValidation() throws {
    let s = store()
    XCTAssertThrowsError(try s.upsert(LibraryEntry(id: " ", kind: .component, name: "x", value: "y")))
    XCTAssertThrowsError(try s.upsert(LibraryEntry(id: "c1", kind: .component, name: " ", value: "y")))
    var overRated = wardrobe("w1", "grey tee")
    overRated.rating = 9
    XCTAssertThrowsError(try s.upsert(overRated))
  }

  func testTemplateBodyCannotUseAnUndeclaredSlot() throws {
    let s = store()
    let bad = LibraryEntry(
      id: "t1", kind: .template, name: "portrait", value: "a woman wearing {OUTFIT} in {SCENE}",
      slots: [LibrarySlot(id: "OUTFIT")])
    XCTAssertThrowsError(try s.upsert(bad)) { error in
      XCTAssertEqual(
        (error as? LibraryError), .invalid("template body uses undeclared slot(s): SCENE"))
    }
  }

  // MARK: templates

  func testFillUsesValuesThenDefaultsAndLeavesUnfilledSlotsVisible() {
    let t = LibraryEntry(
      id: "t1", kind: .template, name: "portrait",
      value: "a woman wearing {OUTFIT}, {SCENE}, shot on {CAMERA}",
      slots: [LibrarySlot(id: "OUTFIT"), LibrarySlot(id: "SCENE", defaultValue: "a quiet cafe"),
              LibrarySlot(id: "CAMERA")])
    let out = LibraryStore.fill(template: t, values: ["OUTFIT": "a grey tee"])
    XCTAssertEqual(out, "a woman wearing a grey tee, a quiet cafe, shot on {CAMERA}",
                   "an unfilled slot stays visible — never silently blank")
  }

  func testFillTreatsBlankValuesAsUnsupplied() {
    let t = LibraryEntry(
      id: "t1", kind: .template, name: "p", value: "wearing {OUTFIT}",
      slots: [LibrarySlot(id: "OUTFIT", defaultValue: "a slip dress")])
    XCTAssertEqual(LibraryStore.fill(template: t, values: ["OUTFIT": "   "]), "wearing a slip dress")
  }

  func testOutfitRendersItsItemsInOrder() throws {
    let s = store()
    try s.upsert(wardrobe("w1", "cropped tee", value: "cropped white tee"))
    try s.upsert(wardrobe("w2", "denim skirt", category: "Bottoms", value: "washed denim skirt"))
    let outfit = LibraryEntry(id: "o1", kind: .outfit, name: "weekend", value: "",
                             items: ["w1", "w2"])
    try s.upsert(outfit)
    XCTAssertEqual(s.renderOutfit(outfit), "cropped white tee, washed denim skirt")

    var written = outfit
    written.value = "a hand-written phrase"
    XCTAssertEqual(s.renderOutfit(written), "a hand-written phrase", "own text wins")
  }

  // MARK: query

  private func seed(_ s: LibraryStore) throws {
    var a = LibraryEntry(id: "c1", kind: .component, name: "quarry dusk",
                        value: "a quiet marble quarry at dusk",
                        facets: ["mood": ["Calm"], "scene": ["Nature"]], componentKind: "scene")
    a.rating = 5
    var b = LibraryEntry(id: "c2", kind: .component, name: "neon alley",
                        value: "a neon-lit alley at night",
                        facets: ["mood": ["Bold"], "scene": ["Urban"]], componentKind: "scene")
    b.rating = 2
    b.favorite = true
    var c = wardrobe("w1", "latex bustier", category: "Bodywear", subtype: "Corsets")
    c.archived = true
    try s.upsert(a); try s.upsert(b); try s.upsert(c)
  }

  func testSearchFiltersByKindFacetTextAndRating() throws {
    let s = store()
    try seed(s)
    XCTAssertEqual(s.search(LibraryQuery(kinds: [.component])).map(\.id).sorted(), ["c1", "c2"])
    XCTAssertEqual(s.search(LibraryQuery(facets: ["mood": ["Calm"]])).map(\.id), ["c1"])
    XCTAssertEqual(
      s.search(LibraryQuery(facets: ["mood": ["Calm", "Bold"]])).map(\.id).sorted(), ["c1", "c2"],
      "values within one axis are ORed")
    XCTAssertEqual(
      s.search(LibraryQuery(facets: ["mood": ["Calm"], "scene": ["Urban"]])).map(\.id), [],
      "axes are ANDed")
    XCTAssertEqual(s.search(LibraryQuery(text: "NEON")).map(\.id), ["c2"], "text search ignores case")
    XCTAssertEqual(s.search(LibraryQuery(minRating: 4)).map(\.id), ["c1"])
    XCTAssertEqual(s.search(LibraryQuery(favoritesOnly: true)).map(\.id), ["c2"])
  }

  func testArchivedIsHiddenUnlessAsked() throws {
    let s = store()
    try seed(s)
    XCTAssertFalse(s.search(LibraryQuery()).contains { $0.id == "w1" })
    XCTAssertTrue(s.search(LibraryQuery(includeArchived: true)).contains { $0.id == "w1" })
  }

  func testOrderingAndLimit() throws {
    let s = store()
    try seed(s)
    XCTAssertEqual(s.search(LibraryQuery(order: "rating")).first?.id, "c1")
    XCTAssertEqual(s.search(LibraryQuery(order: "name")).first?.id, "c2", "neon before quarry")
    XCTAssertEqual(s.search(LibraryQuery(limit: 1)).count, 1)
    try s.markUsed(id: "c2")
    XCTAssertEqual(s.search(LibraryQuery(order: "use_count")).first?.id, "c2")
  }

  func testFacetCountsIgnoreArchived() throws {
    let s = store()
    try seed(s)
    var extra = LibraryEntry(id: "c3", kind: .component, name: "third", value: "x",
                            facets: ["mood": ["Calm"]], componentKind: "scene")
    extra.archived = true
    try s.upsert(extra)
    XCTAssertEqual(s.facetCounts()["mood"]?["Calm"], 1)
    XCTAssertEqual(s.facetCounts()["scene"]?["Urban"], 1)
  }

  // MARK: collections

  func testCollectionDeleteUnfilesItemsButKeepsPackMembership() throws {
    let s = store()
    try s.upsertCollection(LibraryCollection(id: "col1", name: "Showcase", color: "#f6e65a"))
    var item = wardrobe("w1", "grey tee")
    item.collections = ["col1"]
    item.packCollections = ["col1"]
    try s.upsert(item)

    XCTAssertTrue(try s.deleteCollection(id: "col1"))
    let after = try XCTUnwrap(s.item(id: "w1"))
    XCTAssertEqual(after.collections, [], "the user's filing is cleared")
    XCTAssertEqual(after.packCollections, ["col1"], "pack membership survives, so a reimport reconciles")
    XCTAssertFalse(try s.deleteCollection(id: "col1"))
  }

  func testSearchByCollectionMatchesUserOrPackMembership() throws {
    let s = store()
    var mine = wardrobe("w1", "mine"); mine.collections = ["col1"]
    var theirs = wardrobe("w2", "theirs"); theirs.packCollections = ["col1"]
    try s.upsert(mine); try s.upsert(theirs)
    XCTAssertEqual(s.search(LibraryQuery(collection: "col1")).map(\.id).sorted(), ["w1", "w2"])
  }

  func testCollectionValidation() throws {
    let s = store()
    XCTAssertThrowsError(try s.upsertCollection(LibraryCollection(id: "", name: "x")))
    XCTAssertThrowsError(try s.upsertCollection(LibraryCollection(id: "c", name: " ")))
    XCTAssertThrowsError(
      try s.upsertCollection(LibraryCollection(id: "c", name: "c", parentId: "c")))
  }

  // MARK: use tracking

  func testMarkUsedCountsAndStamps() throws {
    let s = store()
    try s.upsert(wardrobe("w1", "grey tee"))
    let used = try s.markUsed(id: "w1")
    XCTAssertEqual(used.useCount, 1)
    XCTAssertNotNil(used.lastUsedAt)
    XCTAssertThrowsError(try s.markUsed(id: "nope")) { XCTAssertEqual($0 as? LibraryError, .notFound("nope")) }
    XCTAssertEqual(store().item(id: "w1")?.useCount, 1, "persisted")
  }

  func testDelete() throws {
    let s = store()
    try s.upsert(wardrobe("w1", "grey tee"))
    XCTAssertTrue(try s.delete(id: "w1"))
    XCTAssertFalse(try s.delete(id: "w1"))
    XCTAssertTrue(store().allItems().isEmpty)
  }
}
