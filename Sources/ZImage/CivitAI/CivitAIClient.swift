// CivitAIClient.swift — civitai.com / civitai.red API client
//
// Search + browse models and LoRAs, scrape prompts from posted images, and
// download model files into the local LoRA library. Works keyless for
// public content; an API key unlocks auth-gated listings and downloads.
// The base URL is configurable so civitai.red (same API) works.
//
// Lives in the ZImage library target (moved from ComfyBoxDesktop in #234) so
// the headless warm server / MCP layer can act as a CivitAI conduit, not just
// the desktop app. Desktop keeps sourcing the key from its Keychain-backed
// `AppSecrets.civitai`; the server resolves it via `CivitAISecrets`.

import Foundation

public struct CivitAIClient: Sendable {
    public enum SortOrder: String, CaseIterable, Sendable {
        case highestRated = "Highest Rated"
        case mostDownloaded = "Most Downloaded"
        case mostLiked = "Most Liked"
        case newest = "Newest"
    }

    /// Time window the sort applies over (CivitAI `period`).
    public enum Period: String, CaseIterable, Sendable {
        case allTime = "AllTime"
        case year = "Year"
        case month = "Month"
        case week = "Week"
        case day = "Day"
    }

    public let baseURL: URL
    public let apiKey: String?

    public init(baseURL: URL = URL(string: "https://civitai.com")!, apiKey: String? = nil) {
        self.baseURL = baseURL
        self.apiKey = apiKey?.isEmpty == true ? nil : apiKey
    }

    // MARK: - Requests

    private func request(path: String, query: [URLQueryItem]) -> URLRequest? {
        var components = URLComponents(url: baseURL.appendingPathComponent(path),
                                       resolvingAgainstBaseURL: false)
        components?.queryItems = query
        guard let url = components?.url else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        request.setValue("ComfyBox/1.0 (macOS)", forHTTPHeaderField: "User-Agent")
        if let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// Search models. Empty query lists by sort order. `types` e.g. ["LORA"],
    /// `baseModel` e.g. "Z-Image" filters to a model family.
    public func searchModels(
        query: String = "",
        types: [String] = [],
        baseModel: String? = nil,
        sort: SortOrder = .mostDownloaded,
        period: Period = .allTime,
        nsfw: Bool = false,
        cursor: String? = nil,
        limit: Int = 24
    ) async throws -> CivitAIModelsPage {
        var items = [
            URLQueryItem(name: "limit", value: String(limit)),
            URLQueryItem(name: "sort", value: sort.rawValue),
            URLQueryItem(name: "period", value: period.rawValue),
        ]
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { items.append(URLQueryItem(name: "query", value: trimmed)) }
        for type in types { items.append(URLQueryItem(name: "types", value: type)) }
        // CivitAI's API silently returns zero results (HTTP 200, empty items)
        // whenever both `query` and `baseModels` are present. Only send
        // `baseModels` for a plain browse (no text query); text searches
        // filter by base model client-side instead (see CivitAIBrowserView).
        if let baseModel, !baseModel.isEmpty, trimmed.isEmpty {
            items.append(URLQueryItem(name: "baseModels", value: baseModel))
        }
        // `nsfw=true` includes adult content; omitting it lets the server default
        // (SFW). civitai.red callers pass true.
        items.append(URLQueryItem(name: "nsfw", value: nsfw ? "true" : "false"))
        if let cursor { items.append(URLQueryItem(name: "cursor", value: cursor)) }

        guard let request = request(path: "api/v1/models", query: items) else {
            throw URLError(.badURL)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(CivitAIModelsPage.self, from: data)
    }

    /// Posted images for a model version — the prompt-scraping source.
    public func images(modelVersionId: Int, limit: Int = 20) async throws -> [CivitAIImage] {
        let items = [
            URLQueryItem(name: "modelVersionId", value: String(modelVersionId)),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        guard let request = request(path: "api/v1/images", query: items) else {
            throw URLError(.badURL)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(CivitAIImagesPage.self, from: data).items
    }

    // MARK: - Download

    /// Where downloaded LoRAs land — the ComfyBox server's library root.
    public static func loraLibraryDirectory() -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".comfybox/loras", isDirectory: true)
    }

    /// Download a model file into `directory`, reporting fractional progress.
    /// Auth rides along both as a `?token=` query parameter and an
    /// `Authorization: Bearer` header; a redirect delegate strips the header
    /// when CivitAI hands off to its CDN (which uses its own query-param auth
    /// and would reject a stray Bearer header). Returns the final file URL.
    public func download(
        file: CivitAIFile,
        to directory: URL = CivitAIClient.loraLibraryDirectory(),
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> URL {
        guard var components = URLComponents(string: file.downloadUrl) else {
            throw CivitAIDownloadError.badURL
        }
        if let apiKey {
            var query = components.queryItems ?? []
            query.append(URLQueryItem(name: "token", value: apiKey))
            components.queryItems = query
        }
        guard let url = components.url else { throw CivitAIDownloadError.badURL }

        var request = URLRequest(url: url)
        request.timeoutInterval = 3600
        request.setValue("ComfyBox/1.0 (macOS)", forHTTPHeaderField: "User-Agent")
        if let apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        let session = URLSession(
            configuration: .default,
            delegate: CivitAIRedirectStripper(originalHost: url.host),
            delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let (bytes, response) = try await session.bytes(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw CivitAIDownloadError.http(-1, "No HTTP response")
        }
        guard http.statusCode == 200 else {
            // Read a little of the body for a useful message.
            var snippet = ""
            var count = 0
            for try await byte in bytes {
                snippet.append(Character(UnicodeScalar(byte)))
                count += 1
                if count >= 300 { break }
            }
            throw CivitAIDownloadError.forStatus(http.statusCode, body: snippet, hasKey: apiKey != nil)
        }

        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let filename = file.name.isEmpty ? (url.lastPathComponent + ".safetensors") : file.name
        let destination = directory.appendingPathComponent(filename)
        let temp = directory.appendingPathComponent(".\(filename).download")

        FileManager.default.createFile(atPath: temp.path, contents: nil)
        let handle = try FileHandle(forWritingTo: temp)
        defer { try? handle.close() }

        let expected = http.expectedContentLength
        var received: Int64 = 0
        var buffer = Data(capacity: 1 << 20)
        var lastReported = -1.0

        for try await byte in bytes {
            buffer.append(byte)
            if buffer.count >= 1 << 20 {
                try handle.write(contentsOf: buffer)
                received += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                if expected > 0 {
                    let fraction = Double(received) / Double(expected)
                    if fraction - lastReported >= 0.01 {
                        lastReported = fraction
                        progress?(min(fraction, 1.0))
                    }
                }
            }
        }
        if !buffer.isEmpty {
            try handle.write(contentsOf: buffer)
        }
        try handle.close()

        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: temp, to: destination)
        progress?(1.0)
        return destination
    }
}

/// Strips the `Authorization` header when a redirect crosses to a different
/// host — CivitAI's CDN (Backblaze B2 / Cloudflare) authenticates via its own
/// signed query params and rejects a forwarded Bearer header.
final class CivitAIRedirectStripper: NSObject, URLSessionTaskDelegate {
    private let originalHost: String?
    init(originalHost: String?) { self.originalHost = originalHost }

    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        var newRequest = request
        if request.url?.host != originalHost {
            newRequest.setValue(nil, forHTTPHeaderField: "Authorization")
        }
        completionHandler(newRequest)
    }
}

public enum CivitAIDownloadError: LocalizedError {
    case badURL
    case authRequired(hasKey: Bool)
    case notFound
    case http(Int, String)

    /// Map an HTTP status to a friendly error.
    static func forStatus(_ code: Int, body: String, hasKey: Bool) -> CivitAIDownloadError {
        switch code {
        case 401, 403: return .authRequired(hasKey: hasKey)
        case 404: return .notFound
        default: return .http(code, body.trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }

    public var errorDescription: String? {
        switch self {
        case .badURL: return "Invalid download URL."
        case .authRequired(let hasKey):
            return hasKey
                ? "CivitAI rejected the download — your API key may be invalid, or this model is early-access/gated for your account."
                : "This download needs a CivitAI API key. Add one in Settings → CivitAI (some NSFW/gated models require it)."
        case .notFound: return "That file was not found on CivitAI (it may have been removed)."
        case .http(let code, let body):
            let extra = body.isEmpty ? "" : " — \(body.prefix(120))"
            return "CivitAI download failed (HTTP \(code))\(extra)"
        }
    }
}
