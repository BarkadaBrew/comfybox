import Foundation
import XCTest

@testable import ZImage

/// Task #15 (spec #9 Tier D): prompt templates externalized to
/// ~/.comfybox/prompts/<id>.md with shipped builtins, loud fallback, and a
/// content hash recorded per resolution so any render's prompt text is
/// attributable. Ships byte-identical builtin content — externalization
/// itself must not change a single render.
final class PromptTemplateStoreTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("prompt-templates-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  func testBuiltinWhenNoFileExists() {
    let store = PromptTemplateStore(directory: tempDir)
    let t = store.template("video-banana")
    XCTAssertEqual(t.source, .builtin)
    XCTAssertTrue(t.text.contains("SUGGESTIVE"), "builtin content is the shipped template")
    XCTAssertFalse(t.hash.isEmpty)
  }

  func testFileOverrideWins() throws {
    let custom = "CUSTOM TEMPLATE {{LTX_RULES}} end"
    try custom.write(to: tempDir.appendingPathComponent("video-banana.md"), atomically: true, encoding: .utf8)

    let store = PromptTemplateStore(directory: tempDir)
    let t = store.template("video-banana")
    XCTAssertEqual(t.source, .file)
    XCTAssertTrue(t.text.hasPrefix("CUSTOM TEMPLATE"))
    XCTAssertFalse(t.text.contains("{{LTX_RULES}}"), "placeholder must be substituted")
    XCTAssertTrue(t.text.contains("MOTION MUST BE NATURAL"), "rules block injected")
  }

  func testEmptyFileFallsBackToBuiltin() throws {
    try "   \n ".write(to: tempDir.appendingPathComponent("video-avocado.md"), atomically: true, encoding: .utf8)

    let store = PromptTemplateStore(directory: tempDir)
    let t = store.template("video-avocado")
    XCTAssertEqual(t.source, .builtin, "blank file is malformed — loud fallback, never an empty system prompt")
  }

  func testHashChangesWithContentAndIsStable() throws {
    let store = PromptTemplateStore(directory: tempDir)
    let a1 = store.template("video-neutral")
    let a2 = store.template("video-neutral")
    XCTAssertEqual(a1.hash, a2.hash, "same content, same hash")

    try "different".write(to: tempDir.appendingPathComponent("video-neutral.md"), atomically: true, encoding: .utf8)
    let b = store.template("video-neutral")
    XCTAssertNotEqual(a1.hash, b.hash)
  }

  func testRulesTemplateItselfIsOverridable() throws {
    try "MINIMAL RULES ONLY".write(to: tempDir.appendingPathComponent("ltx-rules.md"), atomically: true, encoding: .utf8)
    let store = PromptTemplateStore(directory: tempDir)
    let t = store.template("video-banana")
    XCTAssertTrue(t.text.contains("MINIMAL RULES ONLY"))
    XCTAssertFalse(t.text.contains("MOTION MUST BE NATURAL"))
  }

  func testExternalizationIsByteIdenticalToLegacyConstants() {
    // The non-negotiable: with no files on disk, every template the store
    // yields is EXACTLY what PromptOptimizer.selectSystemPrompt produced
    // before externalization.
    let store = PromptTemplateStore(directory: tempDir)
    for (id, mode, kind) in [
      ("image-neutral", "neutral", "image"), ("image-banana", "banana", "image"),
      ("image-avocado", "avocado", "image"), ("video-neutral", "neutral", "video"),
      ("video-banana", "banana", "video"), ("video-avocado", "avocado", "video"),
      ("video-i2v", "banana", "video-i2v"),
    ] {
      XCTAssertEqual(
        store.template(id).text,
        PromptOptimizer.selectSystemPrompt(contentMode: mode, mediaKind: kind),
        "template \(id) must be byte-identical to the legacy constant")
    }
  }
}
