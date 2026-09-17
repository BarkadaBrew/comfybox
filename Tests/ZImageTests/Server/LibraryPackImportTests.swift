import Foundation
import XCTest

@testable import ZImage

/// Third-party pack import (PRD docs/PRD-creative-library.md, L7).
final class LibraryPackImportTests: XCTestCase {

  private var root: URL!
  private var storeDir: URL!
  private var store: LibraryStore!

  override func setUpWithError() throws {
    root = FileManager.default.temporaryDirectory
      .appendingPathComponent("pack-\(UUID().uuidString)")
    storeDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("packstore-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: root.appendingPathComponent("library"), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: storeDir, withIntermediateDirectories: true)
    store = LibraryStore(directory: storeDir)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: root)
    try? FileManager.default.removeItem(at: storeDir)
  }

  private func write(_ file: String, _ object: Any) throws {
    let url = file == "manifest.json"
      ? root.appendingPathComponent(file)
      : root.appendingPathComponent("library").appendingPathComponent(file)
    try JSONSerialization.data(withJSONObject: object).write(to: url)
  }

  private func writeManifest(format: String = LibraryPackImporter.supportedFormat) throws {
    try write("manifest.json", [
      "format": format, "schema_version": 3, "pack_id": "soslibrary:abc",
      "name": "Starter", "creator": "Sick Ollie",
    ])
  }

  /// A minimal pack in the reference format's real shape.
  private func writeFullPack() throws {
    try writeManifest()
    try write("component-collections.json", [
      ["name": "Showcase Looks", "parent_name": "", "color": "#f6e65a", "kind": "outfit"],
    ])
    try write("prompts.json", [
      [
        "source_prompt_id": "p1", "kind": "template",
        "value": "A young woman wearing OUTFIT in a quarry at dusk, NAME on her jacket.",
        "placeholder_signature": "NAME+OUTFIT",
        "facets": ["mood": ["Calm"], "shot": ["Portrait"]],
        "primary_parent": "Creators", "primary_subcategory": "dr0s",
        "rating": 3, "archived": false, "note": "",
        "collections": [["name": "Showcase Looks"]],
      ],
      [
        "source_prompt_id": "p2", "kind": "prompt",
        "value": "photorealistic studio portrait, soft key light, shallow depth of field",
        "facets": ["style": ["Editorial"]],
      ],
    ])
    try write("components.json", [
      ["source_component_id": "c1", "kind": "scene", "value": "a neon-lit alley after rain",
       "rating": "4", "preview_ref": "prev1.webp"],
      ["source_component_id": "c2", "kind": "outfit", "value": "cherry halter top, tennis skirt",
       "pack_collections": "[{'name': 'Showcase Looks', 'color': '#f6e65a'}]"],
    ])
    try write("wardrobe-items.json", [
      ["source_wardrobe_id": "w1", "item_type": "piece", "category": "Bodywear",
       "subtype": "Corsets & Bustiers", "value": "latex bustier", "rating": "0"],
    ])
    try write("recipes.json", [
      ["source_recipe_id": "r1", "name": "quarry look",
       "payload": [
         "_sickollie_seed_source": "imported-image",
         "migration": ["loras": [["name": "Savannah_krea2_epoch_08", "strength": 1.0]]],
         "nodes": [["type": "SOPromptLogEngineStudio"]],
       ]],
    ])
  }

  // MARK: format

  func testUnsupportedFormatIsRefused() throws {
    try writeManifest(format: "some-other-pack")
    XCTAssertThrowsError(try LibraryPackImporter.importDirectory(root, into: store)) { error in
      XCTAssertEqual(error as? LibraryPackError, .unsupportedFormat("some-other-pack"))
    }
    XCTAssertTrue(store.allItems().isEmpty)
  }

  func testMissingManifestIsRefused() {
    XCTAssertThrowsError(try LibraryPackImporter.importDirectory(root, into: store))
  }

  // MARK: mapping

  func testImportsEveryKindWithProvenance() throws {
    try writeFullPack()
    let report = try LibraryPackImporter.importDirectory(root, into: store)

    XCTAssertEqual(report.packName, "Starter")
    XCTAssertEqual(report.creator, "Sick Ollie")
    XCTAssertEqual(report.imported["template"], 1)
    XCTAssertEqual(report.imported["component"], 2, "the prompt row plus the scene component")
    XCTAssertEqual(report.imported["outfit"], 1)
    XCTAssertEqual(report.imported["wardrobe_item"], 1)
    XCTAssertEqual(report.imported["recipe"], 1)
    XCTAssertEqual(report.collections, 1)
    XCTAssertEqual(report.total, 6)
    XCTAssertEqual(report.skipped.map(\.what), [], "nothing in a well-formed pack is skipped")

    for item in store.allItems() {
      XCTAssertEqual(item.source, "pack:soslibrary:abc", "\(item.id) records where it came from")
    }
  }

  func testTemplatePlaceholdersBecomeSlots() throws {
    try writeFullPack()
    try LibraryPackImporter.importDirectory(root, into: store)
    let template = try XCTUnwrap(store.item(id: "soslibrary:abc#p1"))
    XCTAssertEqual(template.kind, .template)
    XCTAssertEqual(
      template.value,
      "A young woman wearing {OUTFIT} in a quarry at dusk, {NAME} on her jacket.")
    XCTAssertEqual(template.slots?.map(\.id).sorted(), ["NAME", "OUTFIT"])
    // And it fills like any template authored here.
    XCTAssertEqual(
      LibraryStore.fill(template: template, values: ["OUTFIT": "a grey tee", "NAME": "KIRA"]),
      "A young woman wearing a grey tee in a quarry at dusk, KIRA on her jacket.")
  }

  func testFacetsCarryOverAlongWithParentAndSubcategory() throws {
    try writeFullPack()
    try LibraryPackImporter.importDirectory(root, into: store)
    let template = try XCTUnwrap(store.item(id: "soslibrary:abc#p1"))
    XCTAssertEqual(template.facets["mood"], ["Calm"])
    XCTAssertEqual(template.facets["parent"], ["Creators"])
    XCTAssertEqual(template.facets["subcategory"], ["dr0s"])
    XCTAssertEqual(template.rating, 3)
  }

  func testCollectionMembershipLandsInPackCollectionsOnly() throws {
    try writeFullPack()
    try LibraryPackImporter.importDirectory(root, into: store)
    let outfit = try XCTUnwrap(store.item(id: "soslibrary:abc#c2"))
    XCTAssertEqual(outfit.kind, .outfit)
    XCTAssertEqual(
      outfit.packCollections, ["soslibrary:abc#collection:showcase-looks"],
      "a stringified list is parsed, and pack filing stays out of the user's own collections")
    XCTAssertEqual(outfit.collections, [])
    let collection = try XCTUnwrap(store.allCollections().first)
    XCTAssertEqual(collection.name, "Showcase Looks")
    XCTAssertEqual(collection.membership, "pack")
    XCTAssertEqual(collection.color, "#f6e65a")
  }

  func testWardrobeKeepsItsCategoryStructure() throws {
    try writeFullPack()
    try LibraryPackImporter.importDirectory(root, into: store)
    let item = try XCTUnwrap(store.item(id: "soslibrary:abc#w1"))
    XCTAssertEqual(item.kind, .wardrobeItem)
    XCTAssertEqual(item.category, "Bodywear")
    XCTAssertEqual(item.subtype, "Corsets & Bustiers")
    XCTAssertEqual(item.itemType, "piece")
  }

  func testRecipeKeepsItsForeignPayloadAndIsLabelledReferenceOnly() throws {
    try writeFullPack()
    try LibraryPackImporter.importDirectory(root, into: store)
    let recipe = try XCTUnwrap(store.item(id: "soslibrary:abc#r1"))
    XCTAssertEqual(recipe.kind, .recipe)
    XCTAssertEqual(recipe.seedSource, "imported-image")
    XCTAssertEqual(recipe.loras?.first?.filename, "Savannah_krea2_epoch_08")
    XCTAssertEqual(recipe.loras?.first?.scale, 1.0)
    XCTAssertTrue(recipe.rawPayload?.contains("SOPromptLogEngineStudio") == true)
    XCTAssertTrue(recipe.notes?.contains("not replayable") == true)
  }

  // MARK: reimport

  func testDryRunCountsWithoutWriting() throws {
    try writeFullPack()
    let report = try LibraryPackImporter.importDirectory(root, into: store, dryRun: true)
    XCTAssertEqual(report.total, 6)
    XCTAssertTrue(report.dryRun)
    XCTAssertTrue(store.allItems().isEmpty, "a dry run writes nothing")
    XCTAssertTrue(store.allCollections().isEmpty)
  }

  func testReimportUpdatesInPlaceAndKeepsTheUsersOwnEdits() throws {
    try writeFullPack()
    try LibraryPackImporter.importDirectory(root, into: store)
    let count = store.allItems().count

    // Todd rates it and files it in his own collection.
    var mine = try XCTUnwrap(store.item(id: "soslibrary:abc#w1"))
    mine.rating = 5
    mine.collections = ["my-favourites"]
    try store.upsert(mine)

    try LibraryPackImporter.importDirectory(root, into: store)
    XCTAssertEqual(store.allItems().count, count, "reimport updates in place, never duplicates")
    let after = try XCTUnwrap(store.item(id: "soslibrary:abc#w1"))
    XCTAssertEqual(after.collections, [], "the pack row is authoritative for its own fields")
    XCTAssertEqual(
      after.packCollections, [], "and for pack membership")
    XCTAssertEqual(after.createdAt, mine.createdAt, "createdAt survives, so history is intact")
  }

  // MARK: helpers

  func testPlaceholderRewriteIsWholeTokenOnly() {
    let body = "wearing OUTFIT, not OUTFITTED, already {NAME}"
    let out = LibraryPackImporter.rewritePlaceholders(body, tokens: ["OUTFIT", "NAME"])
    XCTAssertEqual(out, "wearing {OUTFIT}, not OUTFITTED, already {NAME}")
  }

  func testPlaceholderTokensMustActuallyAppearInTheBody() {
    let tokens = LibraryPackImporter.placeholderTokens(
      "NAME+OUTFIT+BRAND", body: "wearing OUTFIT with NAME")
    XCTAssertEqual(tokens, ["NAME", "OUTFIT"], "BRAND is declared but absent, so it is not a slot")
  }

  func testShortNameTrimsProseToALabel() {
    XCTAssertEqual(LibraryPackImporter.shortName("a neon-lit alley, after rain"), "a neon-lit alley")
    XCTAssertEqual(LibraryPackImporter.shortName(String(repeating: "x", count: 80)).count, 58)
  }

  func testStringifiedListParsing() {
    XCTAssertEqual(
      LibraryPackImporter.namesFromStringifiedList("[{'name': 'Showcase Looks', 'color': '#fff'}]"),
      ["Showcase Looks"])
    XCTAssertEqual(LibraryPackImporter.namesFromStringifiedList("[]"), [])
  }

  func testMalformedLibraryFilesAreReportedNotFatal() throws {
    try writeManifest()
    try Data("not json".utf8).write(
      to: root.appendingPathComponent("library/prompts.json"))
    let report = try LibraryPackImporter.importDirectory(root, into: store)
    XCTAssertTrue(report.skipped.contains { $0.what == "prompts.json" })
    XCTAssertEqual(report.total, 0)
  }
}
