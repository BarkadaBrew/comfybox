import XCTest
@testable import ZImage

final class GalleryStoreTests: XCTestCase {

  private func makeTempPath() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-gallery-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir.appendingPathComponent("galleries.json")
  }

  func testCreateAssignsSlugIdAndCreatesDirectory() throws {
    let store = GalleryStore(path: try makeTempPath())
    let entry = try store.create(name: "Kira Portraits!", hidden: false, password: nil)
    XCTAssertEqual(entry.id, "kira-portraits")
    XCTAssertTrue(FileManager.default.fileExists(atPath: entry.directoryPath))
    XCTAssertNil(entry.passwordHash)
  }

  func testDuplicateNameIsRejectedCaseInsensitively() throws {
    let store = GalleryStore(path: try makeTempPath())
    try store.create(name: "Nightly", hidden: false, password: nil)
    XCTAssertThrowsError(try store.create(name: "nightly", hidden: false, password: nil)) { error in
      XCTAssertEqual(error as? GalleryStoreError, .duplicateName("nightly"))
    }
  }

  func testEmptyNameIsRejected() throws {
    let store = GalleryStore(path: try makeTempPath())
    XCTAssertThrowsError(try store.create(name: "   ", hidden: false, password: nil)) { error in
      XCTAssertEqual(error as? GalleryStoreError, .invalidName)
    }
  }

  func testListOmitsHiddenUnlessRequested() throws {
    let store = GalleryStore(path: try makeTempPath())
    try store.create(name: "Public", hidden: false, password: nil)
    try store.create(name: "Secret", hidden: true, password: nil)

    XCTAssertEqual(store.list(includeHidden: false).map { $0.name }, ["Public"])
    XCTAssertEqual(Set(store.list(includeHidden: true).map { $0.name }), ["Public", "Secret"])
  }

  func testAuthorizeRequiresCorrectPassword() throws {
    let store = GalleryStore(path: try makeTempPath())
    let entry = try store.create(name: "Vault", hidden: false, password: "sesame")
    XCTAssertTrue(entry.passwordHash != nil)

    XCTAssertThrowsError(try store.authorize(id: entry.id, password: nil)) { error in
      XCTAssertEqual(error as? GalleryStoreError, .unauthorized)
    }
    XCTAssertThrowsError(try store.authorize(id: entry.id, password: "wrong")) { error in
      XCTAssertEqual(error as? GalleryStoreError, .unauthorized)
    }
    XCTAssertNoThrow(try store.authorize(id: entry.id, password: "sesame"))
  }

  func testAuthorizeUnlockedGalleryIgnoresPassword() throws {
    let store = GalleryStore(path: try makeTempPath())
    let entry = try store.create(name: "Open", hidden: false, password: nil)
    XCTAssertNoThrow(try store.authorize(id: entry.id, password: nil))
  }

  func testAuthorizeMissingGalleryThrowsNotFound() throws {
    let store = GalleryStore(path: try makeTempPath())
    XCTAssertThrowsError(try store.authorize(id: "nope", password: nil)) { error in
      XCTAssertEqual(error as? GalleryStoreError, .notFound("nope"))
    }
  }

  func testDeleteRequiresPasswordWhenLocked() throws {
    let store = GalleryStore(path: try makeTempPath())
    let entry = try store.create(name: "Locked", hidden: false, password: "pw")

    XCTAssertThrowsError(try store.delete(id: entry.id, password: nil)) { error in
      XCTAssertEqual(error as? GalleryStoreError, .unauthorized)
    }
    XCTAssertNoThrow(try store.delete(id: entry.id, password: "pw"))
    XCTAssertTrue(store.list(includeHidden: true).isEmpty)
  }

  func testStoreRoundTripsThroughDisk() throws {
    let path = try makeTempPath()
    let store = GalleryStore(path: path)
    try store.create(name: "Alpha", hidden: false, password: nil)
    try store.create(name: "Beta", hidden: true, password: "hunter2")

    let reopened = GalleryStore(path: path)
    XCTAssertEqual(
      Set(reopened.list(includeHidden: true).map { $0.name }),
      Set(store.list(includeHidden: true).map { $0.name })
    )
    // Password survives the round trip (still locked after reload).
    let beta = reopened.list(includeHidden: true).first { $0.name == "Beta" }
    XCTAssertEqual(beta?.locked, true)
  }
}
