// WarmServerClient.swift — Minimal HTTP client for localhost WarmServer
//
// Uses Foundation URLSession for HTTP calls to the WarmServer REST API.
// All requests are localhost-only. Timeout: 300 seconds (renders can take minutes).

import Foundation

/// Lightweight HTTP client targeting the local WarmServer instance.
public final class WarmServerClient: @unchecked Sendable {
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
  case networkError(String)

  public var errorDescription: String? {
    switch self {
    case .invalidURL(let path):
      return "Invalid URL path: \(path)"
    case .invalidResponse:
      return "Invalid HTTP response"
    case .connectionRefused(let host, let port):
      return "WarmServer not running at \(host):\(port)"
    case .networkError(let message):
      return "Network error: \(message)"
    }
  }
}
