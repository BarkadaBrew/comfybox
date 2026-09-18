// ImmichThumbnailProtocol.swift — thumbnails for assets that live on Immich.
//
// Immich requires its API key as a REQUEST HEADER, and SwiftUI's image loading
// gives no way to set one. So the grid points at a private scheme instead, and
// this URLProtocol serves it: it looks the remote up in the desktop settings,
// reads its key from the keychain, and fetches the real thumbnail.
//
//   immich-thumb://<remote config id>/<immich asset id>?size=preview
//
// Registered once at app start (ComfyBoxDesktopApp). Without it, an Immich
// remote's assets would render as broken cells — which is why they used to be
// hidden (FDD-remote-galleries §9).

import Foundation

final class ImmichThumbnailProtocol: URLProtocol, @unchecked Sendable {

    static let scheme = "immich-thumb"

    /// The URL the grid uses for one asset.
    static func url(remoteID: String, immichAssetID: String, size: String = "preview") -> URL? {
        URL(string: "\(scheme)://\(remoteID)/\(immichAssetID)?size=\(size)")
    }

    /// The same asset at full size — what a download must use, since the grid's
    /// URL is a rendered preview.
    static func originalURL(from url: URL) -> URL? {
        guard url.scheme == scheme, var parts = URLComponents(url: url, resolvingAgainstBaseURL: false) else { return nil }
        parts.queryItems = [URLQueryItem(name: "size", value: "original")]
        return parts.url
    }

    /// Call once at startup.
    static func register() {
        URLProtocol.registerClass(ImmichThumbnailProtocol.self)
    }

    override class func canInit(with request: URLRequest) -> Bool {
        request.url?.scheme == scheme
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    private var fetchTask: Task<Void, Never>?

    override func startLoading() {
        let request = self.request
        fetchTask = Task { [weak self] in
            guard let self else { return }
            do {
                let data = try await Self.fetch(request)
                let response = HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil,
                                               headerFields: ["Content-Type": "image/jpeg"])!
                self.client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .allowedInMemoryOnly)
                self.client?.urlProtocol(self, didLoad: data)
                self.client?.urlProtocolDidFinishLoading(self)
            } catch {
                self.client?.urlProtocol(self, didFailWithError: error)
            }
        }
    }

    override func stopLoading() {
        fetchTask?.cancel()
        fetchTask = nil
    }

    private enum ProtocolError: LocalizedError {
        case notConfigured
        case http(Int)

        var errorDescription: String? {
            switch self {
            case .notConfigured: return "No Immich remote matches this thumbnail request"
            case .http(let code): return "Immich answered HTTP \(code) for a thumbnail"
            }
        }
    }

    /// Resolve the remote, add the key, fetch the real thumbnail.
    private static func fetch(_ request: URLRequest) async throws -> Data {
        guard let url = request.url, let remoteID = url.host else { throw ProtocolError.notConfigured }
        let immichAssetID = url.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let size = URLComponents(url: url, resolvingAgainstBaseURL: false)?
            .queryItems?.first(where: { $0.name == "size" })?.value ?? "preview"

        let remotes = await MainActor.run { DesktopSettings.load().remoteGalleries ?? [] }
        // "original" is the full file (what a download needs); anything else is
        // one of Immich's rendered thumbnails.
        let endpoint = size == "original"
            ? "/api/assets/\(immichAssetID)/original"
            : "/api/assets/\(immichAssetID)/thumbnail?size=\(size)"
        guard let remote = remotes.first(where: { $0.id == remoteID }),
              let base = remote.normalizedBaseURL,
              let key = Keychain.get(remote.keychainAccount),
              let target = URL(string: base + endpoint)
        else { throw ProtocolError.notConfigured }

        var upstream = URLRequest(url: target)
        upstream.setValue(key, forHTTPHeaderField: "x-api-key")
        upstream.timeoutInterval = 20
        let (data, response) = try await URLSession.shared.data(for: upstream)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else { throw ProtocolError.http(code) }
        return data
    }
}
