// ComfyBridgeOptimizerClient.swift — CoffeeShop prompt optimizer bridge

import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

struct ComfyBridgeOptimizerRequest: Sendable {
  let nodeId: String
  let rawPrompt: String
  let preset: String
  let contentMode: String
  let sceneHint: String
  let aspectRatio: String
  let character: String?
  let characterDescription: String?
}

final class ComfyBridgeOptimizerClient {
  private let endpoint: URL
  private let session: URLSession

  init(
    baseURL: URL? = nil,
    session: URLSession = .shared
  ) {
    let configuredURL = ProcessInfo.processInfo.environment["COFFEESHOP_DAEMON_URL"]
      ?? "http://10.0.100.232:3777"
    let root = baseURL ?? URL(string: configuredURL) ?? URL(string: "http://10.0.100.232:3777")!
    self.endpoint = root.appendingPathComponent("v1").appendingPathComponent("optimize")
    self.session = session
  }

  func optimize(_ optimizerRequest: ComfyBridgeOptimizerRequest) async throws -> String {
    var payload: [String: Any] = [
      "raw_prompt": optimizerRequest.rawPrompt,
      "preset": optimizerRequest.preset,
      "content_mode": optimizerRequest.contentMode,
      "scene_hint": optimizerRequest.sceneHint,
      "aspect_ratio": optimizerRequest.aspectRatio
    ]
    if let character = optimizerRequest.character, !character.isEmpty {
      payload["character"] = character
    }
    if let characterDescription = optimizerRequest.characterDescription, !characterDescription.isEmpty {
      payload["character_description"] = characterDescription
    }

    var request = URLRequest(url: endpoint)
    request.httpMethod = "POST"
    request.timeoutInterval = 60
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.httpBody = try JSONSerialization.data(withJSONObject: payload)

    let (data, response) = try await session.data(for: request)
    if let httpResponse = response as? HTTPURLResponse,
       !(200..<300).contains(httpResponse.statusCode) {
      let message = String(data: data, encoding: .utf8) ?? "HTTP \(httpResponse.statusCode)"
      throw OptimizerError.requestFailed(message)
    }

    if let object = try? JSONSerialization.jsonObject(with: data),
       let optimized = Self.extractOptimizedPrompt(from: object) {
      return optimized
    }

    if let text = String(data: data, encoding: .utf8), !text.isEmpty {
      return text
    }

    throw OptimizerError.invalidResponse
  }

  private static func extractOptimizedPrompt(from object: Any) -> String? {
    if let dict = object as? [String: Any] {
      for key in ["optimized_prompt", "prompt", "result", "text", "optimized"] {
        if let value = dict[key] as? String, !value.isEmpty {
          return value
        }
      }
      for key in ["data", "output"] {
        if let nested = dict[key], let value = extractOptimizedPrompt(from: nested) {
          return value
        }
      }
    }
    if let array = object as? [Any] {
      for value in array {
        if let value = extractOptimizedPrompt(from: value) {
          return value
        }
      }
    }
    return nil
  }

  enum OptimizerError: Error, CustomStringConvertible {
    case invalidResponse
    case requestFailed(String)

    var description: String {
      switch self {
      case .invalidResponse:
        return "CoffeeShop optimizer returned no optimized prompt"
      case .requestFailed(let message):
        return "CoffeeShop optimizer request failed: \(message)"
      }
    }
  }
}
