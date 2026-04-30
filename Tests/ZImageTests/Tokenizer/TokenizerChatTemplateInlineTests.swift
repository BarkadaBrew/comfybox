import XCTest
@testable import ZImage

/// Covers `QwenTokenizer.ensureInlineChatTemplate`, the idempotent merger
/// that hoists `chat_template.jinja` into `tokenizer_config.json` so
/// swift-transformers' `applyChatTemplate` can find the template inline.
///
/// Regression: upstream HF snapshots of `Tongyi-MAI/Z-Image-Turbo` ship
/// the template as a sidecar `.jinja` file, and swift-transformers only
/// reads the inline field — producing `missingChatTemplate` pipeline
/// errors. Operators previously patched the config by hand via
/// `/tmp/inject_chat_template.py`; this helper does the same merge at
/// load time, preserving a `.bak` of the original on first patch.
final class TokenizerChatTemplateInlineTests: XCTestCase {

  func testInlinesJinjaWhenConfigMissingChatTemplate() throws {
    let tokenizerDir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: tokenizerDir) }

    let configURL = tokenizerDir.appending(path: "tokenizer_config.json")
    let jinjaURL = tokenizerDir.appending(path: "chat_template.jinja")

    let originalConfig: [String: Any] = ["pad_token": "<|endoftext|>", "model_max_length": 131072]
    try write(json: originalConfig, to: configURL)
    let jinjaBody = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"
    try jinjaBody.write(to: jinjaURL, atomically: true, encoding: .utf8)

    try QwenTokenizer.ensureInlineChatTemplate(in: tokenizerDir, environment: [:])

    let patched = try readJSON(from: configURL)
    XCTAssertEqual(patched["chat_template"] as? String, jinjaBody,
      "jinja contents must be inlined verbatim into tokenizer_config.json")
    XCTAssertEqual(patched["pad_token"] as? String, "<|endoftext|>",
      "pre-existing fields must be preserved")
    XCTAssertEqual(patched["model_max_length"] as? Int, 131072)

    let backupURL = configURL.appendingPathExtension("bak")
    XCTAssertTrue(FileManager.default.fileExists(atPath: backupURL.path),
      ".bak must be written on first patch so the original config is recoverable")
    let backup = try readJSON(from: backupURL)
    XCTAssertNil(backup["chat_template"], "backup must be the pre-patch config (no chat_template key)")
  }

  func testIdempotentWhenChatTemplateAlreadyInline() throws {
    let tokenizerDir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: tokenizerDir) }

    let configURL = tokenizerDir.appending(path: "tokenizer_config.json")
    let jinjaURL = tokenizerDir.appending(path: "chat_template.jinja")

    let originalTemplate = "inline-original"
    let originalConfig: [String: Any] = ["chat_template": originalTemplate, "pad_token": "<|endoftext|>"]
    try write(json: originalConfig, to: configURL)
    try "sidecar-should-be-ignored".write(to: jinjaURL, atomically: true, encoding: .utf8)

    try QwenTokenizer.ensureInlineChatTemplate(in: tokenizerDir, environment: [:])

    let after = try readJSON(from: configURL)
    XCTAssertEqual(after["chat_template"] as? String, originalTemplate,
      "existing inline template must not be overwritten by the sidecar")

    let backupURL = configURL.appendingPathExtension("bak")
    XCTAssertFalse(FileManager.default.fileExists(atPath: backupURL.path),
      "no backup should be written when no patch is necessary")
  }

  func testNoOpWhenJinjaSidecarMissing() throws {
    let tokenizerDir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: tokenizerDir) }

    let configURL = tokenizerDir.appending(path: "tokenizer_config.json")
    let originalConfig: [String: Any] = ["pad_token": "<|endoftext|>"]
    try write(json: originalConfig, to: configURL)

    // No chat_template.jinja in the directory.
    try QwenTokenizer.ensureInlineChatTemplate(in: tokenizerDir, environment: [:])

    let after = try readJSON(from: configURL)
    XCTAssertNil(after["chat_template"],
      "config must be untouched when no sidecar is available to merge")
    XCTAssertFalse(FileManager.default.fileExists(atPath: configURL.appendingPathExtension("bak").path))
  }

  func testOptOutViaEnvironmentVariable() throws {
    let tokenizerDir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: tokenizerDir) }

    let configURL = tokenizerDir.appending(path: "tokenizer_config.json")
    let jinjaURL = tokenizerDir.appending(path: "chat_template.jinja")

    try write(json: ["pad_token": "<|endoftext|>"], to: configURL)
    try "body".write(to: jinjaURL, atomically: true, encoding: .utf8)

    try QwenTokenizer.ensureInlineChatTemplate(
      in: tokenizerDir,
      environment: ["ZIMAGE_NO_TOKENIZER_PATCH": "1"]
    )

    let after = try readJSON(from: configURL)
    XCTAssertNil(after["chat_template"],
      "ZIMAGE_NO_TOKENIZER_PATCH=1 must suppress the auto-patch")
  }

  func testDoesNotOverwriteExistingBackup() throws {
    let tokenizerDir = try makeTempDir()
    defer { try? FileManager.default.removeItem(at: tokenizerDir) }

    let configURL = tokenizerDir.appending(path: "tokenizer_config.json")
    let jinjaURL = tokenizerDir.appending(path: "chat_template.jinja")
    let backupURL = configURL.appendingPathExtension("bak")

    // Seed a pre-existing backup with a sentinel payload — if the helper
    // clobbered it, we'd lose an operator-saved snapshot from an earlier
    // migration.
    let sentinel: [String: Any] = ["sentinel": "preserve-me"]
    try write(json: sentinel, to: backupURL)

    try write(json: ["pad_token": "<|endoftext|>"], to: configURL)
    try "body".write(to: jinjaURL, atomically: true, encoding: .utf8)

    try QwenTokenizer.ensureInlineChatTemplate(in: tokenizerDir, environment: [:])

    let backup = try readJSON(from: backupURL)
    XCTAssertEqual(backup["sentinel"] as? String, "preserve-me",
      "existing .bak must not be overwritten by a subsequent patch")
  }

  // MARK: - Helpers

  private func makeTempDir() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  private func write(json object: [String: Any], to url: URL) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted])
    try data.write(to: url, options: [.atomic])
  }

  private func readJSON(from url: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: url)
    guard let json = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any] else {
      XCTFail("\(url.path) did not parse as JSON dictionary")
      return [:]
    }
    return json
  }
}
