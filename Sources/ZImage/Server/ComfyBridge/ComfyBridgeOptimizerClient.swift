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

  func optimize(_ optimizerRequest: ComfyBridgeOptimizerRequest) async throws -> ComfyBridgeOptimizerResponse {
    let payload: [String: Any] = [
      "raw_prompt": optimizerRequest.rawPrompt,
      "preset": optimizerRequest.preset,
      "content_mode": optimizerRequest.contentMode,
      "scene_hint": optimizerRequest.sceneHint,
      "aspect_ratio": optimizerRequest.aspectRatio,
      "character": optimizerRequest.character ?? "",
      "character_description": optimizerRequest.characterDescription ?? "",
    ]

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

    guard let object = try? JSONSerialization.jsonObject(with: data),
          let dict = object as? [String: Any],
          let optimizedPrompt = dict["optimized_prompt"] as? String,
          let contextBlock = dict["context_block"] as? String,
          let photoBlock = dict["photo_block"] as? String else {
      throw OptimizerError.invalidResponse
    }

    return ComfyBridgeOptimizerResponse(
      optimizedPrompt: optimizedPrompt,
      contextBlock: contextBlock,
      photoBlock: photoBlock,
      enhanced: (dict["enhanced"] as? Bool) ?? false,
      note: dict["note"] as? String
    )
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
