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

struct ComfyBridgeOptimizerResponse: Sendable {
  let optimizedPrompt: String
  let contextBlock: String
  let photoBlock: String
  let enhanced: Bool
  let note: String?
}

final class ComfyBridgeOptimizerClient {
  private let directEndpoint: URL
  private let toolEndpoint: URL
  private let authToken: String?
  private let session: URLSession

  init(
    baseURL: URL? = nil,
    session: URLSession = .shared
  ) {
    let configuredURL = ProcessInfo.processInfo.environment["COFFEESHOP_DAEMON_URL"]
      ?? ProcessInfo.processInfo.environment["BREE_DAEMON_URL"]
      ?? "http://10.0.100.232:3777"
    let root = baseURL ?? URL(string: configuredURL) ?? URL(string: "http://10.0.100.232:3777")!
    self.directEndpoint = root.appendingPathComponent("v1").appendingPathComponent("optimize")
    self.toolEndpoint = root
      .appendingPathComponent("v1")
      .appendingPathComponent("tools")
      .appendingPathComponent("execute")
    self.authToken = ProcessInfo.processInfo.environment["COFFEESHOP_DAEMON_TOKEN"]
      ?? ProcessInfo.processInfo.environment["BREE_TOOL_TOKEN"]
      ?? ProcessInfo.processInfo.environment["COMFYBOX_OPTIMIZER_TOKEN"]
    self.session = session
  }

  func optimize(_ optimizerRequest: ComfyBridgeOptimizerRequest) async throws -> ComfyBridgeOptimizerResponse {
    do {
      return try await postJSON(to: directEndpoint, payload: directPayload(for: optimizerRequest))
    } catch {
      return try await postJSON(to: toolEndpoint, payload: toolPayload(for: optimizerRequest))
    }
  }

  private func directPayload(for optimizerRequest: ComfyBridgeOptimizerRequest) -> [String: Any] {
    [
      "raw_prompt": optimizerRequest.rawPrompt,
      "preset": optimizerRequest.preset,
      "content_mode": optimizerRequest.contentMode,
      "scene_hint": optimizerRequest.sceneHint,
      "aspect_ratio": optimizerRequest.aspectRatio,
      "character": optimizerRequest.character ?? "",
      "character_description": optimizerRequest.characterDescription ?? "",
    ]
  }

  private func toolPayload(for optimizerRequest: ComfyBridgeOptimizerRequest) -> [String: Any] {
    var arguments: [String: Any] = [
      "prompt": optimizerRequest.rawPrompt,
      "style": optimizerRequest.preset,
      "content_mode": optimizerRequest.contentMode,
    ]
    if let characterDescription = optimizerRequest.characterDescription, !characterDescription.isEmpty {
      arguments["character_description"] = characterDescription
    }
    return [
      "name": "studio_enhance",
      "arguments": arguments
    ]
  }

  private func postJSON(to url: URL, payload: [String: Any]) async throws -> ComfyBridgeOptimizerResponse {
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.timeoutInterval = 60
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    if let authToken, !authToken.isEmpty {
      request.setValue("Bearer \(authToken)", forHTTPHeaderField: "Authorization")
    }
    request.httpBody = try JSONSerialization.data(withJSONObject: payload)

    let (data, response) = try await session.data(for: request)
    if let httpResponse = response as? HTTPURLResponse,
       !(200..<300).contains(httpResponse.statusCode) {
      let message = String(data: data, encoding: .utf8) ?? "HTTP \(httpResponse.statusCode)"
      throw OptimizerError.requestFailed(message)
    }

    guard let object = try? JSONSerialization.jsonObject(with: data),
          let optimizerResponse = Self.extractOptimizerResponse(from: object) else {
      throw OptimizerError.invalidResponse
    }

    return optimizerResponse
  }

  private static func extractOptimizerResponse(from object: Any) -> ComfyBridgeOptimizerResponse? {
    guard let optimizedPrompt = extractString(
      from: object,
      keys: ["optimized_prompt", "prompt", "enhanced_prompt", "text", "optimized"]
    ) else {
      return nil
    }
    return ComfyBridgeOptimizerResponse(
      optimizedPrompt: optimizedPrompt,
      contextBlock: extractString(from: object, keys: ["context_block", "context"]) ?? "",
      photoBlock: extractString(from: object, keys: ["photo_block", "photo"]) ?? "",
      enhanced: extractBool(from: object, keys: ["enhanced", "was_optimized"]) ?? true,
      note: extractString(from: object, keys: ["note"])
    )
  }

  private static func extractString(from object: Any, keys: [String]) -> String? {
    if let dict = object as? [String: Any] {
      for key in keys {
        if let value = dict[key] as? String, !value.isEmpty {
          return value
        }
      }
      for nestedKey in ["result", "data", "output"] {
        if let nested = dict[nestedKey],
           let value = extractString(from: nested, keys: keys) {
          return value
        }
      }
    }
    if let array = object as? [Any] {
      for item in array {
        if let value = extractString(from: item, keys: keys) {
          return value
        }
      }
    }
    return nil
  }

  private static func extractBool(from object: Any, keys: [String]) -> Bool? {
    if let dict = object as? [String: Any] {
      for key in keys {
        if let value = dict[key] as? Bool {
          return value
        }
      }
      for nestedKey in ["result", "data", "output"] {
        if let nested = dict[nestedKey],
           let value = extractBool(from: nested, keys: keys) {
          return value
        }
      }
    }
    if let array = object as? [Any] {
      for item in array {
        if let value = extractBool(from: item, keys: keys) {
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
