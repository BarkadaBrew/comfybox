// GalleryHubService.swift — CRUD client for named galleries (/v1/galleries).
//
// A named gallery is a scope on the ComfyBox server a caller (Kira, Desktop,
// any /v1/generate client) can render into instead of the default output
// folder — see GalleryStore.swift server-side. This service lists/creates/
// deletes them against the currently-connected server; GalleryHubView is the
// card-grid UI over it, and RemoteGalleryService (galleryId/galleryPassword)
// is what actually browses one once selected.

import Foundation

@Observable
@MainActor
public final class GalleryHubService {
    public let engine: EngineService
    public private(set) var galleries: [GallerySummary] = []
    public private(set) var isLoading = false
    public var error: String?

    public struct GallerySummary: Identifiable, Sendable, Equatable, Decodable {
        public let id: String
        public let name: String
        public let hidden: Bool
        public let locked: Bool
    }

    public init(engine: EngineService) { self.engine = engine }

    public var baseURL: String { "http://\(engine.serverHost):\(engine.serverPort)" }

    /// Whether the connected server is this same Mac — GalleryArchiver moves
    /// files with a plain local FileManager call, which only makes sense when
    /// Desktop and the server share a filesystem (matches RemoteGalleryService's
    /// own check).
    public var isLocalServer: Bool {
        ["127.0.0.1", "localhost", "::1"].contains(engine.serverHost)
    }

    /// Lists everything, including hidden galleries — this IS the management
    /// surface, unlike a plain discovery listing that would omit them.
    public func load() async {
        isLoading = true
        defer { isLoading = false }
        error = nil
        guard let url = URL(string: baseURL + "/v1/galleries?includeHidden=1") else { return }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                error = "Server returned \(http.statusCode)"; return
            }
            struct ListResponse: Decodable { let galleries: [GallerySummary] }
            galleries = try JSONDecoder().decode(ListResponse.self, from: data).galleries
        } catch {
            self.error = error.localizedDescription
        }
    }

    @discardableResult
    public func create(name: String, hidden: Bool, password: String?) async throws -> GallerySummary {
        guard let url = URL(string: baseURL + "/v1/galleries") else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        var body: [String: Any] = ["name": name, "hidden": hidden]
        if let password, !password.isEmpty { body["password"] = password }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw GalleryHubError.server(status: http.statusCode, message: Self.errorMessage(from: data))
        }
        let entry = try JSONDecoder().decode(GallerySummary.self, from: data)
        galleries.append(entry)
        return entry
    }

    public func delete(id: String, password: String?) async throws {
        var c = URLComponents(string: baseURL + "/v1/galleries/\(id)")
        if let password, !password.isEmpty {
            c?.queryItems = [URLQueryItem(name: "password", value: password)]
        }
        guard let url = c?.url else { throw URLError(.badURL) }
        var request = URLRequest(url: url)
        request.httpMethod = "DELETE"

        let (data, response) = try await URLSession.shared.data(for: request)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw GalleryHubError.server(status: http.statusCode, message: Self.errorMessage(from: data))
        }
        galleries.removeAll { $0.id == id }
    }

    private static func errorMessage(from data: Data) -> String {
        (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] as? String ?? "Request failed"
    }
}

public enum GalleryHubError: LocalizedError {
    case server(status: Int, message: String)

    public var errorDescription: String? {
        switch self {
        case .server(let status, let message): return "\(message) (\(status))"
        }
    }
}
