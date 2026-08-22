import XCTest
@testable import ZImage

/// WP-E10 "E9b" (FDD Addendum A.2, E5 review MAJOR; AC-34b second half):
/// `/health.model` carries the RESOLVED directory path once a krea2 alias has
/// been loaded through `parseModelSpec` (`"krea2-raw"` → `~/LocalModels/krea2-raw`),
/// so the name the caller used is gone from the wire. `/health.model_alias`
/// restores it by reverse-looking the path up in THE spec→directory table —
/// never by guessing from a filename. Pure, table-injected, weight-free.
final class Krea2ModelAliasTests: XCTestCase {

  private var scratch: URL!
  private var table: [String: String]!

  override func setUpWithError() throws {
    scratch = FileManager.default.temporaryDirectory
      .appending(path: "krea2-alias-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch.appending(path: "krea2-raw"), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: scratch.appending(path: "kroma-v0.2"), withIntermediateDirectories: true)
    table = [
      "krea2-raw": scratch.appending(path: "krea2-raw").path,
      "kroma-v0.2-turbo": scratch.appending(path: "kroma-v0.2").path,
    ]
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: scratch)
  }

  func testDeclaredAliasIsItself() {
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: "krea2-raw", table: table), "krea2-raw")
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: "KREA2-RAW", table: table), "krea2-raw", "case-insensitive, canonical spelling returned")
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: "kroma-v0.2-turbo", table: table), "kroma-v0.2-turbo")
  }

  func testResolvedDirectoryReverseLooksUpToItsAlias() {
    let dir = scratch.appending(path: "krea2-raw").path
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: dir, table: table), "krea2-raw")
    // Trailing slash / non-standardised spellings of the same directory.
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: dir + "/", table: table), "krea2-raw")
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: scratch.path + "/./krea2-raw", table: table), "krea2-raw")
  }

  func testTildeFormsMatch() {
    let home = FileManager.default.homeDirectoryForCurrentUser.path
    let tildeTable = ["krea2-raw": "~/LocalModels/krea2-raw"]
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: home + "/LocalModels/krea2-raw", table: tildeTable), "krea2-raw")
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: "~/LocalModels/krea2-raw", table: tildeTable), "krea2-raw")
  }

  func testTurboHFAliasesAreTheirOwnAlias() {
    for name in Krea2ModelDetection.turboAliases {
      XCTAssertEqual(Krea2ModelDetection.alias(forSpec: name, table: table), name)
    }
  }

  func testUnknownSpecHasNoAlias() {
    XCTAssertNil(Krea2ModelDetection.alias(forSpec: "/nowhere/at/all", table: table))
    XCTAssertNil(Krea2ModelDetection.alias(forSpec: "z-image-turbo", table: table))
    XCTAssertNil(Krea2ModelDetection.alias(forSpec: scratch.path, table: table), "a parent dir is not the alias's dir")
  }

  func testDefaultTableCoversTheLiveInstalls() {
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: "krea2-raw"), "krea2-raw")
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: "~/LocalModels/krea2-raw"), "krea2-raw")
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: "~/LocalModels/kroma-v0.2"), "kroma-v0.2-turbo")
  }
}
