// RemoteGalleryService.swift — Browse a ComfyBox server's output folder remotely
//
// When the desktop is pointed at a remote ComfyBox, the local DAM can't see the
// server's files. This fetches the server's gallery listing (/v1/gallery/list)
// and builds URLs to stream each file (/v1/gallery/file?path=). Images can be
// pulled down into the local gallery on demand.
//
// galleryId/galleryPassword scope every request to a named gallery (see
// GalleryStore.swift server-side) instead of the server's default output
// folder — nil (the default) preserves the original single-folder behavior.

import Foundation

@Observable
@MainActor
public final class RemoteGalleryService {
    public let engine: EngineService
    /// Named gallery to browse instead of the default output folder, and its
    /// password if it's locked. Set by GalleryHubView before load()/pull().
    public var galleryId: String?
    public var galleryPassword: String?
    public private(set) var assets: [RemoteAsset] = []
    public private(set) var isLoading = false
    public var error: String?

    public struct RemoteAsset: Identifiable, Sendable, Equatable {
        public let path: String
        public let filename: String
        public let kind: String       // "image" | "video"
        public let size: Int
        public let modified: String
        public var id: String { path }
    }

    public init(engine: EngineService, galleryId: String? = nil, galleryPassword: String? = nil) {
        self.engine = engine
        self.galleryId = galleryId
        self.galleryPassword = galleryPassword
    }

    public var baseURL: String { "http://\(engine.serverHost):\(engine.serverPort)" }

    private var galleryQueryItems: [URLQueryItem] {
        guard let galleryId, !galleryId.isEmpty else { return [] }
        var items = [URLQueryItem(name: "gallery", value: galleryId)]
        if let galleryPassword, !galleryPassword.isEmpty {
            items.append(URLQueryItem(name: "password", value: galleryPassword))
        }
        return items
    }

    /// URL that streams a server-side file's bytes.
    public func fileURL(for path: String) -> URL? {
        var c = URLComponents(string: baseURL + "/v1/gallery/file")
        c?.queryItems = [URLQueryItem(name: "path", value: path)] + galleryQueryItems
        return c?.url
    }

    /// Whether the browsed server is this same Mac (then "pull" is a no-op copy).
    public var isLocalServer: Bool {
        ["127.0.0.1", "localhost", "::1"].contains(engine.serverHost)
    }

    public func load() async {
        isLoading = true
        defer { isLoading = false }
        error = nil
        var c = URLComponents(string: baseURL + "/v1/gallery/list")
        c?.queryItems = [URLQueryItem(name: "limit", value: "1000")] + galleryQueryItems
        guard let url = c?.url else { return }
        do {
            let (data, response) = try await URLSession.shared.data(from: url)
            if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                error = "Server returned \(http.statusCode)"; return
            }
            let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let items = (obj?["items"] as? [[String: Any]]) ?? []
            assets = items.compactMap { d in
                guard let p = d["path"] as? String, let f = d["filename"] as? String else { return nil }
                return RemoteAsset(path: p, filename: f,
                                   kind: d["kind"] as? String ?? "image",
                                   size: d["size"] as? Int ?? 0,
                                   modified: d["modified"] as? String ?? "")
            }
        } catch {
            self.error = error.localizedDescription
        }
    }

    /// Download a remote asset into `destinationDir`; returns the local path.
    @discardableResult
    public func pull(_ asset: RemoteAsset, to destinationDir: String) async throws -> String {
        guard let url = fileURL(for: asset.path) else { throw URLError(.badURL) }
        let (data, _) = try await URLSession.shared.data(from: url)
        let dest = (destinationDir as NSString).appendingPathComponent(asset.filename)
        try data.write(to: URL(fileURLWithPath: dest))
        return dest
    }
}
