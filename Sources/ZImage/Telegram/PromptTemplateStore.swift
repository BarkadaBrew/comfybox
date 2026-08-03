import CryptoKit
import Foundation
import Logging

/// Task #15 (spec #9 Tier D): prompt templates as versioned files with
/// shipped builtins.
///
/// - Files live in `~/.comfybox/prompts/<id>.md`; editing one takes effect on
///   the NEXT optimize call — no rebuild, no restart (templates are read per
///   resolution; they are tiny).
/// - The shipped builtins are the exact legacy `PromptOptimizer` constants —
///   externalization is byte-identical until someone writes a file.
/// - A malformed (empty/whitespace) file falls back to the builtin LOUDLY.
/// - Every resolution carries a content hash so a render's prompt text is
///   attributable in traces ("which template version produced this?").
/// - File templates may reference `{{LTX_RULES}}`; it expands to the shared
///   video rules block (itself overridable via `ltx-rules.md`).
public final class PromptTemplateStore {

  public enum Source: String, Sendable { case file, builtin }

  public struct Resolved: Sendable {
    public let id: String
    public let text: String
    public let source: Source
    public let hash: String   // 12-hex-char SHA256 prefix of `text`
  }

  public static let shared = PromptTemplateStore()

  private let directory: URL
  private let logger = Logger(label: "z-image.prompt-templates")

  public init(directory: URL? = nil) {
    self.directory = directory
      ?? FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent(".comfybox/prompts")
  }

  private static let rulesPlaceholder = "{{LTX_RULES}}"

  /// Builtins keyed by template id — the legacy constants, verbatim
  /// (video ones in {{LTX_RULES}} placeholder form).
  private static let builtins: [String: String] = [
    "image-neutral": PromptOptimizer.systemPromptNeutral,
    "image-banana": PromptOptimizer.systemPromptBanana,
    "image-avocado": PromptOptimizer.systemPromptAvocado,
    "video-neutral": PromptOptimizer.systemPromptVideoNeutral,
    "video-banana": PromptOptimizer.systemPromptVideoBanana,
    "video-avocado": PromptOptimizer.systemPromptVideoAvocado,
    "video-i2v": PromptOptimizer.systemPromptVideoI2V,
    "ltx-rules": PromptOptimizer.ltxRules,
  ]

  /// The template id `PromptOptimizer` uses for a given mode/kind — one
  /// mapping, mirrored from `selectSystemPrompt`.
  public static func templateId(contentMode: String, mediaKind: String) -> String {
    let kind = mediaKind.lowercased()
    if kind == "video-i2v" { return "video-i2v" }
    let prefix = (kind == "video" || kind == "video-t2v") ? "video" : "image"
    switch contentMode.lowercased() {
    case "avocado": return "\(prefix)-avocado"
    case "banana": return "\(prefix)-banana"
    default: return "\(prefix)-neutral"
    }
  }

  public func template(_ id: String) -> Resolved {
    let builtin = Self.builtins[id] ?? ""

    let fileURL = directory.appendingPathComponent("\(id).md")
    if let raw = try? String(contentsOf: fileURL, encoding: .utf8) {
      let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      if trimmed.isEmpty {
        logger.error("prompt template '\(id)' at \(fileURL.path) is empty — using builtin. Delete the file or fill it in.")
      } else {
        let text = substituteRules(in: raw)
        return Resolved(id: id, text: text, source: .file, hash: Self.hash(text))
      }
    }

    // Builtin video templates carry the same {{LTX_RULES}} placeholder as
    // file templates, so a rules-file override reaches them too.
    let text = substituteRules(in: builtin)
    return Resolved(id: id, text: text, source: .builtin, hash: Self.hash(text))
  }

  private func substituteRules(in text: String) -> String {
    guard text.contains(Self.rulesPlaceholder) else { return text }
    let rules: String
    let rulesURL = directory.appendingPathComponent("ltx-rules.md")
    if let raw = try? String(contentsOf: rulesURL, encoding: .utf8),
       !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      rules = raw
    } else {
      rules = PromptOptimizer.ltxRules
    }
    return text.replacingOccurrences(of: Self.rulesPlaceholder, with: rules)
  }

  private static func hash(_ text: String) -> String {
    let digest = SHA256.hash(data: Data(text.utf8))
    return digest.map { String(format: "%02x", $0) }.joined().prefix(12).lowercased()
  }
}
