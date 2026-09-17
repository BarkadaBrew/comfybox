// VisionChat.swift — the one place that knows how to ask the local vision model
// about an image (OpenAI-compatible /chat/completions with an image_url part).
//
// FDD-mlx-serve-tuning WP4 (§0.5, D7). Two defects this replaces:
//  1. The MCP repair path read COMFYBOX_VISION_URL or fell back to a hard-coded
//     LM Studio http://127.0.0.1:1234/v1 — never the configured provider — so
//     with LM Studio empty, repair_image ran blind (diagnosis silently nil).
//  2. Both callers sent max_tokens 300/320. The configured model (Glimmer) is a
//     reasoning model: at 900 it was measured returning HTTP 200 with empty
//     `content` and ~950 chars of `reasoning_content`; 3000 answered in 28 s.
//     So: 1200 tokens + reasoning_budget 256, and an empty-after-reasoning reply
//     is named instead of swallowed.
//
// Everything here is pure so it is unit-tested without a server.

import Foundation

public enum VisionChat {

  public static let maxTokens = 1200
  public static let reasoningBudget = 256

  public struct Endpoint: Equatable, Sendable {
    public enum Source: Equatable, Sendable { case environment, config }
    public var baseURL: String
    public var model: String
    public var apiKey: String?
    public var source: Source
  }

  /// Resolution order: explicit env override (both COMFYBOX_VISION_URL and
  /// COMFYBOX_VISION_MODEL set) → config `providers.vision` → `providers.captioning`
  /// → nil. There is deliberately no hard-coded default endpoint.
  /// - Parameter configJSON: the body of the warm server's `GET /v1/config`
  ///   (or the config file), nil when unavailable.
  public static func resolveEndpoint(environment: [String: String], configJSON: Data?) -> Endpoint? {
    if let url = environment["COMFYBOX_VISION_URL"], !url.isEmpty,
       let model = environment["COMFYBOX_VISION_MODEL"], !model.isEmpty {
      return Endpoint(baseURL: trimSlash(url), model: model, apiKey: environment["COMFYBOX_VISION_API_KEY"], source: .environment)
    }
    guard let configJSON,
          let obj = (try? JSONSerialization.jsonObject(with: configJSON)) as? [String: Any],
          let providers = obj["providers"] as? [String: Any] else { return nil }
    for key in ["vision", "captioning"] {
      if let p = providers[key] as? [String: Any],
         let base = p["baseUrl"] as? String, !base.isEmpty,
         let model = p["model"] as? String, !model.isEmpty {
        let key = (p["apiKey"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        return Endpoint(baseURL: trimSlash(base), model: model, apiKey: key, source: .config)
      }
    }
    return nil
  }

  /// The request body. `prompt` is the text part; the image rides as a data URL.
  public static func body(model: String, prompt: String, base64PNG: String) -> [String: Any] {
    [
      "model": model,
      "temperature": 0.2,
      "max_tokens": maxTokens,
      "reasoning_budget": reasoningBudget,
      "messages": [[
        "role": "user",
        "content": [
          ["type": "text", "text": prompt],
          ["type": "image_url", "image_url": ["url": "data:image/png;base64,\(base64PNG)"]],
        ] as [Any],
      ]],
    ]
  }

  public enum Reply: Equatable, Sendable {
    /// Trimmed, non-empty answer text.
    case text(String)
    /// HTTP 200 but no answer: the model spent its budget reasoning (char count).
    case emptyAfterReasoning(Int)
    /// Not a chat-completions body, or no content at all.
    case malformed
  }

  public static func parseReply(_ data: Data) -> Reply {
    guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
          let choices = obj["choices"] as? [[String: Any]],
          let message = choices.first?["message"] as? [String: Any],
          let content = message["content"] as? String else { return .malformed }
    let trimmed = content.trimmingCharacters(in: .whitespacesAndNewlines)
    if !trimmed.isEmpty { return .text(trimmed) }
    let reasoning = (message["reasoning_content"] as? String) ?? ""
    return reasoning.isEmpty ? .malformed : .emptyAfterReasoning(reasoning.count)
  }

  private static func trimSlash(_ s: String) -> String {
    var s = s
    while s.hasSuffix("/") { s.removeLast() }
    return s
  }
}
