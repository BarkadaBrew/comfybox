// ContentModeStoreMutationTests.swift — FDD-ui-api-parity §3.3 (Phase 3, Class E):
// PUT/DELETE /v1/content-modes/{mode}. `ContentModeStore.save()` shipped
// uncalled (FDD §2.6); these tests exercise the load->merge->save mutation
// path directly, over a temp file — never the real `~/.comfybox/content-modes.json`.

import XCTest
@testable import ZImage

final class ContentModeStoreMutationTests: XCTestCase {

  private func makeTempPath() -> URL {
    FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-content-modes-mutation-tests-\(UUID().uuidString)", isDirectory: true)
      .appendingPathComponent("content-modes.json")
  }

  func testUpdateSetsOnlyTheSuppliedFields() throws {
    let path = makeTempPath()
    let updated = try ContentModeStore.update(mode: .banana, guidanceBoost: 3.0, at: path)
    XCTAssertEqual(updated.guidanceBoost, 3.0)
    // Fields not supplied keep the built-in value.
    XCTAssertEqual(updated.styleVariant, .sensual)
    XCTAssertEqual(updated.promptHint, "sensual, intimate, suggestive")

    // Persisted, and re-loadable.
    let reloaded = ContentModeStore.loadOrCreate(at: path)
    XCTAssertEqual(reloaded.definition(for: .banana).guidanceBoost, 3.0)
    // Other modes are untouched.
    XCTAssertEqual(reloaded.definition(for: .avocado).guidanceBoost, 2.5)
  }

  func testUpdateIsTolerantPartial() throws {
    let path = makeTempPath()
    _ = try ContentModeStore.update(mode: .avocado, guidanceBoost: 4.0, at: path)
    // A second update touching a DIFFERENT field must not clobber the first.
    let second = try ContentModeStore.update(mode: .avocado, promptHint: "custom hint", at: path)
    XCTAssertEqual(second.guidanceBoost, 4.0, "an update omitting guidanceBoost must preserve it")
    XCTAssertEqual(second.promptHint, "custom hint")
  }

  func testUpdateRejectsOutOfRangeGuidanceBoost() {
    let path = makeTempPath()
    XCTAssertThrowsError(try ContentModeStore.update(mode: .banana, guidanceBoost: 250, at: path)) { error in
      guard case ContentModeStore.ValidationError.guidanceBoostOutOfRange(let value) = error else {
        return XCTFail("expected .guidanceBoostOutOfRange, got \(error)")
      }
      XCTAssertEqual(value, 250)
    }
    // Rejected — nothing persisted.
    XCTAssertFalse(FileManager.default.fileExists(atPath: path.path))
  }

  func testUpdateAcceptsBoundaryValues() throws {
    let path = makeTempPath()
    XCTAssertNoThrow(try ContentModeStore.update(mode: .banana, guidanceBoost: ContentModeStore.guidanceBoostRange.lowerBound, at: path))
    XCTAssertNoThrow(try ContentModeStore.update(mode: .banana, guidanceBoost: ContentModeStore.guidanceBoostRange.upperBound, at: path))
  }

  func testUnknownStyleVariantRawValueFailsToParse() {
    // The route handler's own gate: ContentStyleVariant(rawValue:) returning
    // nil for anything outside {neutral, sensual, nsfw} is what produces the
    // "unknown styleVariant" 400 — pinned here independent of the HTTP layer.
    XCTAssertNil(ContentStyleVariant(rawValue: "bogus"))
    XCTAssertNotNil(ContentStyleVariant(rawValue: "nsfw"))
  }

  func testResetRevertsToBuiltinWithoutRemovingTheEntry() throws {
    let path = makeTempPath()
    _ = try ContentModeStore.update(mode: .avocado, guidanceBoost: 9.0, promptHint: "edited", at: path)
    XCTAssertEqual(ContentModeStore.loadOrCreate(at: path).definition(for: .avocado).guidanceBoost, 9.0)

    let reverted = ContentModeStore.reset(mode: .avocado, at: path)
    XCTAssertEqual(reverted.guidanceBoost, 2.5, "reset must restore the built-in value")
    XCTAssertEqual(reverted.promptHint, "explicit, uncensored, anatomically detailed")

    let reloaded = ContentModeStore.loadOrCreate(at: path)
    XCTAssertEqual(reloaded.modes.count, 3, "reset must not remove the entry — always exactly one per mode")
    XCTAssertEqual(reloaded.definition(for: .avocado).guidanceBoost, 2.5)
  }

  func testResetOnlyAffectsTheNamedMode() throws {
    let path = makeTempPath()
    _ = try ContentModeStore.update(mode: .banana, guidanceBoost: 5.0, at: path)
    _ = try ContentModeStore.update(mode: .avocado, guidanceBoost: 6.0, at: path)
    _ = ContentModeStore.reset(mode: .banana, at: path)

    let reloaded = ContentModeStore.loadOrCreate(at: path)
    XCTAssertEqual(reloaded.definition(for: .banana).guidanceBoost, 1.5, "banana reverted to its built-in")
    XCTAssertEqual(reloaded.definition(for: .avocado).guidanceBoost, 6.0, "avocado's edit must survive banana's reset")
  }

  func testConcurrentUpdatesToDifferentModesBothLand() throws {
    let path = makeTempPath()
    let e1 = expectation(description: "banana")
    let e2 = expectation(description: "avocado")
    DispatchQueue.global().async {
      _ = try? ContentModeStore.update(mode: .banana, guidanceBoost: 3.3, at: path)
      e1.fulfill()
    }
    DispatchQueue.global().async {
      _ = try? ContentModeStore.update(mode: .avocado, guidanceBoost: 4.4, at: path)
      e2.fulfill()
    }
    wait(for: [e1, e2], timeout: 10)

    let reloaded = ContentModeStore.loadOrCreate(at: path)
    XCTAssertEqual(reloaded.definition(for: .banana).guidanceBoost, 3.3)
    XCTAssertEqual(reloaded.definition(for: .avocado).guidanceBoost, 4.4)
  }
}
