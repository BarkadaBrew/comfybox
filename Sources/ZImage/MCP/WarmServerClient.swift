// WarmServerClient.swift — Minimal HTTP client for localhost WarmServer
//
// Uses Foundation URLSession for HTTP calls to the WarmServer REST API.
// All requests are localhost-only. Timeout: 300 seconds (renders can take minutes).

import Foundation

/// The HTTP surface `MCPToolExecutor` needs, behind a protocol so tool
/// composites can be driven end-to-end in unit tests with a stub instead of a
/// live engine (PR #367 review r1, item 5). `WarmServerClient` is the only
/// production implementation.
public protocol WarmServerTransport: Sendable {
  func get(_ path: String) async throws -> (Int, Data)
  func post(_ path: String, body: Data) async throws -> (Int, Data)
  func put(_ path: String, body: Data) async throws -> (Int, Data)
  func patch(_ path: String, body: Data) async throws -> (Int, Data)
  func delete(_ path: String) async throws -> (Int, Data)
  func send(method: String, path: String, body: Data, headers: [String: String]) async throws
    -> (Int, Data, [String: String])
}

/// Lightweight HTTP client targeting the local WarmServer instance.
public final class WarmServerClient: WarmServerTransport, @unchecked Sendable {
  public let host: String
  public let port: UInt16

  private let session: URLSession
  private let baseURL: String

  public init(host: String = "127.0.0.1", port: UInt16 = 7870) {
    self.host = host
    self.port = port
    self.baseURL = "http://\(host):\(port)"

    let config = URLSessionConfiguration.ephemeral
    config.timeoutIntervalForRequest = 300
    config.timeoutIntervalForResource = 300
    self.session = URLSession(configuration: config)
  }

  // MARK: - Public API

  /// Perform a GET request. Returns (HTTP status code, response body).
  public func get(_ path: String) async throws -> (Int, Data) {
    guard let url = URL(string: baseURL + path) else {
      throw WarmServerClientError.invalidURL(path)
    }
    var request = URLRequest(url: url)
    request.httpMethod = "GET"
    request.setValue("application/json", forHTTPHeaderField: "Accept")

    let (data, response) = try await perform(request)
    return (response.statusCode, data)
  }

  /// Perform a POST request with a raw Data body. Returns (HTTP status code, response body).
  public func post(_ path: String, body: Data) async throws -> (Int, Data) {
    guard let url = URL(string: baseURL + path) else {
      throw WarmServerClientError.invalidURL(path)
    }
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    request.httpBody = body

    let (data, response) = try await perform(request)
    return (response.statusCode, data)
  }

  /// Perform a PUT request. Returns (HTTP status code, response body).
  public func put(_ path: String, body: Data) async throws -> (Int, Data) {
    guard let url = URL(string: baseURL + path) else {
      throw WarmServerClientError.invalidURL(path)
    }
    var request = URLRequest(url: url)
    request.httpMethod = "PUT"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    request.httpBody = body

    let (data, response) = try await perform(request)
    return (response.statusCode, data)
  }

  /// Perform an arbitrary-method request with caller-supplied request headers,
  /// returning the response headers too (keys lowercased). Added for
  /// `set_warm_preset`'s conditional PUT (If-Match from the GET's ETag —
  /// adversarial review F2, 2026-08-30): the fixed-shape helpers above can
  /// neither send extra request headers nor surface response headers.
  public func send(
    method: String, path: String, body: Data, headers: [String: String] = [:]
  ) async throws -> (Int, Data, [String: String]) {
    guard let url = URL(string: baseURL + path) else {
      throw WarmServerClientError.invalidURL(path)
    }
    var request = URLRequest(url: url)
    request.httpMethod = method
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    if method != "GET", method != "DELETE" {
      request.setValue("application/json", forHTTPHeaderField: "Content-Type")
      request.httpBody = body
    }
    for (name, value) in headers {
      request.setValue(value, forHTTPHeaderField: name)
    }
    let (data, response) = try await perform(request)
    var responseHeaders: [String: String] = [:]
    for (name, value) in response.allHeaderFields {
      if let name = name as? String, let value = value as? String {
        responseHeaders[name.lowercased()] = value
      }
    }
    return (response.statusCode, data, responseHeaders)
  }

  /// Perform a PATCH request (RFC 7386 JSON Merge Patch body). Returns
  /// (HTTP status code, response body).
  public func patch(_ path: String, body: Data) async throws -> (Int, Data) {
    guard let url = URL(string: baseURL + path) else {
      throw WarmServerClientError.invalidURL(path)
    }
    var request = URLRequest(url: url)
    request.httpMethod = "PATCH"
    request.setValue("application/merge-patch+json", forHTTPHeaderField: "Content-Type")
    request.setValue("application/json", forHTTPHeaderField: "Accept")
    request.httpBody = body

    let (data, response) = try await perform(request)
    return (response.statusCode, data)
  }

  /// Perform a DELETE request. Returns (HTTP status code, response body).
  public func delete(_ path: String) async throws -> (Int, Data) {
    guard let url = URL(string: baseURL + path) else {
      throw WarmServerClientError.invalidURL(path)
    }
    var request = URLRequest(url: url)
    request.httpMethod = "DELETE"
    request.setValue("application/json", forHTTPHeaderField: "Accept")

    let (data, response) = try await perform(request)
    return (response.statusCode, data)
  }

  // MARK: - Private

  private func perform(_ request: URLRequest) async throws -> (Data, HTTPURLResponse) {
    do {
      let (data, response) = try await session.data(for: request)
      guard let httpResponse = response as? HTTPURLResponse else {
        throw WarmServerClientError.invalidResponse
      }
      return (data, httpResponse)
    } catch let error as URLError where error.code == .cannotConnectToHost || error.code == .networkConnectionLost {
      throw WarmServerClientError.connectionRefused(host: host, port: port)
    } catch let error as URLError where error.code == .timedOut {
      // comfybox#389: a mid-boot engine (socket accepting, slow to answer
      // /health) throws `.timedOut` — the same actionable situation as
      // `.connectionRefused` above ("the engine is not answering yet"),
      // but kept as a DISTINCT case rather than folded into it, because
      // `.connectionRefused`'s meaning must not change for other
      // consumers (ComfyBoxDesktop, the Telegram bot).
      // `MCPBridgeStartupPolicy.nothingListeningMessage(for:)` classifies
      // both cases and gives each its own hint wording.
      //
      // Deliberately NOT included here: `.cannotFindHost`. A DNS/resolver
      // failure for "localhost" (comfybox#389 review, 2026-09-05 Pi-hole
      // outage precedent) is a broken resolver, not a not-yet-booted
      // engine — `launchctl kickstart`ing the managed engine cannot fix a
      // resolver problem, so it must not get that hint. It falls through
      // to `.networkError` below, unchanged from pre-#389 behavior.
      throw WarmServerClientError.timedOut(host: host, port: port)
    } catch let error as WarmServerClientError {
      throw error
    } catch {
      throw WarmServerClientError.networkError(error.localizedDescription)
    }
  }
}

// MARK: - Errors

public enum WarmServerClientError: Error, LocalizedError {
  case invalidURL(String)
  case invalidResponse
  case connectionRefused(host: String, port: UInt16)
  /// The port answered but no response came back in time — a mid-boot
  /// engine (comfybox#389). Kept distinct from `.connectionRefused`: that
  /// case's meaning must not change for other consumers (ComfyBoxDesktop,
  /// the Telegram bot), so timeouts get their own case rather than being
  /// folded into it. Deliberately does NOT cover `.cannotFindHost` (a DNS/
  /// resolver failure) — that is a broken resolver, not a not-yet-booted
  /// engine, and stays `.networkError` (see `perform()`).
  case timedOut(host: String, port: UInt16)
  case networkError(String)

  public var errorDescription: String? {
    switch self {
    case .invalidURL(let path):
      return "Invalid URL path: \(path)"
    case .invalidResponse:
      return "Invalid HTTP response"
    case .connectionRefused(let host, let port):
      return "WarmServer not running at \(host):\(port)"
    case .timedOut(let host, let port):
      return "WarmServer at \(host):\(port) did not respond in time"
    case .networkError(let message):
      return "Network error: \(message)"
    }
  }
}
