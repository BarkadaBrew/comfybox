import Foundation
import XCTest

@testable import ZImage

/// The library's HTTP surface (PRD docs/PRD-creative-library.md, L2). These
/// exercise the pure route helpers, so no server or port is needed.
final class LibraryRouteTests: XCTestCase {

  private var dir: URL!
  private var store: LibraryStore!

  override func setUpWithError() throws {
    dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("library-routes-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    store = LibraryStore(directory: dir)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: dir)
  }

  private func body(_ object: Any) throws -> Data {
    try JSONSerialization.data(withJSONObject: object)
  }

  private func decodeJSON(_ response: RoutedResponse) throws -> Any {
    switch response {
    case .json(let http), .error(let http), .shutdown(let http):
      return try JSONSerialization.jsonObject(with: http.body)
    case .websocketUpgrade:
      XCTFail("expected a JSON response")
      return [:]
    }
  }

  private func status(_ response: RoutedResponse) -> Int {
    switch response {
    case .json(let http), .error(let http), .shutdown(let http): return http.status
    case .websocketUpgrade: return -1
    }
  }

  // MARK: upsert

  func testUpsertAcceptsSnakeCaseWireShape() throws {
    let (response, saved) = WarmServer.libraryUpsert(
      store: store,
      body: try body([
        "id": "w1", "kind": "wardrobe_item", "name": "latex bustier",
        "value": "a latex bustier", "category": "Bodywear", "subtype": "Corsets",
        "item_type": "piece", "pack_collections": ["showcase"],
        "facets": ["mood": ["Bold"]],
      ]))
    XCTAssertEqual(status(response), 200)
    XCTAssertEqual(saved?.category, "Bodywear")
    XCTAssertEqual(saved?.itemType, "piece")
    XCTAssertEqual(saved?.packCollections, ["showcase"])
    XCTAssertEqual(store.item(id: "w1")?.facets["mood"], ["Bold"])
  }

  func testUpsertRefusesAnInvalidItemWithA400AndStoresNothing() throws {
    let (response, saved) = WarmServer.libraryUpsert(
      store: store,
      body: try body([
        "id": "t1", "kind": "template", "name": "bad", "value": "wearing {OUTFIT}",
      ]))
    XCTAssertEqual(status(response), 400)
    XCTAssertNil(saved)
    XCTAssertTrue(store.allItems().isEmpty)
  }

  func testUpsertRefusesGarbage() throws {
    let (response, saved) = WarmServer.libraryUpsert(store: store, body: Data("{".utf8))
    XCTAssertEqual(status(response), 400)
    XCTAssertNil(saved)
  }

  // MARK: list

  private func seed() throws {
    try store.upsert(LibraryEntry(
      id: "c1", kind: .component, name: "quarry dusk", value: "a marble quarry at dusk",
      facets: ["mood": ["Calm"], "scene": ["Nature"]], rating: 5, componentKind: "scene"))
    try store.upsert(LibraryEntry(
      id: "c2", kind: .component, name: "neon alley", value: "a neon alley",
      facets: ["mood": ["Bold"], "scene": ["Urban"]], componentKind: "scene"))
    try store.upsert(LibraryEntry(
      id: "w1", kind: .wardrobeItem, name: "grey tee", value: "a grey tee", category: "Tops"))
  }

  func testListFiltersByKindFacetAndText() throws {
    try seed()
    func ids(_ query: [String: String]) throws -> [String] {
      let json = try decodeJSON(WarmServer.libraryList(store: store, query: query))
      return ((json as? [[String: Any]]) ?? []).compactMap { $0["id"] as? String }.sorted()
    }
    XCTAssertEqual(try ids([:]), ["c1", "c2", "w1"])
    XCTAssertEqual(try ids(["kind": "component"]), ["c1", "c2"])
    XCTAssertEqual(try ids(["kind": "component,wardrobe_item"]), ["c1", "c2", "w1"])
    XCTAssertEqual(try ids(["facet.mood": "Calm"]), ["c1"])
    XCTAssertEqual(try ids(["facet.mood": "Calm,Bold"]), ["c1", "c2"])
    XCTAssertEqual(try ids(["facet.mood": "Calm", "facet.scene": "Urban"]), [])
    XCTAssertEqual(try ids(["q": "neon"]), ["c2"])
    XCTAssertEqual(try ids(["min_rating": "4"]), ["c1"])
    XCTAssertEqual(try ids(["category": "Tops"]), ["w1"])
    XCTAssertEqual(try ids(["limit": "1"]).count, 1)
  }

  func testListClampsTheLimit() throws {
    try seed()
    let json = try decodeJSON(WarmServer.libraryList(store: store, query: ["limit": "99999"]))
    XCTAssertEqual((json as? [[String: Any]])?.count, 3, "a huge limit is clamped, not an error")
  }

  // MARK: fill

  func testFillSubstitutesSlotsAndReportsWhatIsStillOpen() throws {
    try store.upsert(LibraryEntry(
      id: "t1", kind: .template, name: "portrait",
      value: "a woman wearing {OUTFIT} in {SCENE}, shot on {CAMERA}",
      slots: [LibrarySlot(id: "OUTFIT"), LibrarySlot(id: "SCENE", defaultValue: "a quiet cafe"),
              LibrarySlot(id: "CAMERA")]))
    let response = WarmServer.libraryFill(
      store: store,
      body: try body(["template_id": "t1", "values": ["OUTFIT": "a grey tee"]]))
    let json = try XCTUnwrap(try decodeJSON(response) as? [String: Any])
    XCTAssertEqual(
      json["prompt"] as? String, "a woman wearing a grey tee in a quiet cafe, shot on {CAMERA}")
    XCTAssertEqual(json["unfilled"] as? [String], ["CAMERA"])
  }

  func testFillResolvesAnOutfitIntoTheOutfitSlot() throws {
    try store.upsert(LibraryEntry(id: "w1", kind: .wardrobeItem, name: "tee", value: "a cropped tee"))
    try store.upsert(LibraryEntry(id: "w2", kind: .wardrobeItem, name: "skirt", value: "a denim skirt"))
    try store.upsert(LibraryEntry(id: "o1", kind: .outfit, name: "weekend", value: "", items: ["w1", "w2"]))
    try store.upsert(LibraryEntry(
      id: "t1", kind: .template, name: "p", value: "wearing {OUTFIT}",
      slots: [LibrarySlot(id: "OUTFIT")]))
    let json = try XCTUnwrap(try decodeJSON(WarmServer.libraryFill(
      store: store, body: try body(["template_id": "t1", "outfit_id": "o1"]))) as? [String: Any])
    XCTAssertEqual(json["prompt"] as? String, "wearing a cropped tee, a denim skirt")
    XCTAssertEqual(json["unfilled"] as? [String], [])
  }

  func testFillCallerValueBeatsTheOutfit() throws {
    try store.upsert(LibraryEntry(id: "w1", kind: .wardrobeItem, name: "tee", value: "a cropped tee"))
    try store.upsert(LibraryEntry(id: "o1", kind: .outfit, name: "weekend", value: "", items: ["w1"]))
    try store.upsert(LibraryEntry(
      id: "t1", kind: .template, name: "p", value: "wearing {OUTFIT}",
      slots: [LibrarySlot(id: "OUTFIT")]))
    let json = try XCTUnwrap(try decodeJSON(WarmServer.libraryFill(
      store: store,
      body: try body(["template_id": "t1", "outfit_id": "o1", "values": ["OUTFIT": "a gown"]])))
      as? [String: Any])
    XCTAssertEqual(json["prompt"] as? String, "wearing a gown")
  }

  func testFillRejectsAMissingOrNonTemplateItem() throws {
    XCTAssertEqual(
      status(WarmServer.libraryFill(store: store, body: try body(["template_id": "nope"]))), 404)
    try store.upsert(LibraryEntry(id: "c1", kind: .component, name: "x", value: "y"))
    XCTAssertEqual(
      status(WarmServer.libraryFill(store: store, body: try body(["template_id": "c1"]))), 400)
  }
}
