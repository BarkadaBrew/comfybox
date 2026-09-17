// ImmichClient.swift — the Immich half of a remote gallery.
//
// Todd 2026-09-17: "we can use the immich store on the server as an additional
// remote gallery." Immich 2.3.1 runs at http://10.0.100.232:2283.
//
// Corrections from the Codex review of the spec (FDD-remote-galleries §3.4):
//   - Immich's own duplicate detection header `x-immich-checksum` is SHA-1,
//     not the catalog's SHA-256. Both are computed for a send.
//   - Album membership is PUT /api/albums/{id}/assets with {"ids": [...]}.
//   - The asset endpoints are versioned, so the client refuses a major version
//     it was not built against rather than guessing.

import Foundation
import CryptoKit

public struct ImmichAsset: Sendable, Equatable {
    public let id: String
    /// "created" or "duplicate" — both mean the bytes are on the server.
    public let status: String
    public var isPresent: Bool { status == "created" || status == "duplicate" }
}

public enum ImmichError: LocalizedError, Equatable {
    case notConfigured
    case unsupportedServerVersion(major: Int)
    case http(Int, String)
    case malformedResponse

    public var errorDescription: String? {
        switch self {
        case .notConfigured:
            return "This Immich remote has no server or API key configured"
        case .unsupportedServerVersion(let major):
            return "Immich major version \(major) is not supported by this build"
        case .http(let code, let body):
            return "Immich answered HTTP \(code)\(body.isEmpty ? "" : ": \(body)")"
        case .malformedResponse:
            return "Immich sent a response this build could not read"
        }
    }
}

public struct ImmichClient: Sendable {

    /// Major versions whose asset API this build speaks.
    public static let supportedMajorVersions: Set<Int> = [1, 2]

    private let baseURL: String
    private let apiKey: String
    private let session: URLSession

    public init(baseURL: String, apiKey: String, session: URLSession = .shared) {
        self.baseURL = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        self.apiKey = apiKey
        self.session = session
    }

    // MARK: - Server

    /// `(major, minor, patch)`. Also the reachability check with a key.
    public func serverVersion() async throws -> (major: Int, minor: Int, patch: Int) {
        let json = try await getJSON(path: "/api/server/version")
        guard let major = json["major"] as? Int,
              let minor = json["minor"] as? Int,
              let patch = json["patch"] as? Int else { throw ImmichError.malformedResponse }
        return (major, minor, patch)
    }

    /// Throws unless this build speaks the server's asset API.
    public func assertSupportedVersion() async throws {
        let version = try await serverVersion()
        guard Self.supportedMajorVersions.contains(version.major) else {
            throw ImmichError.unsupportedServerVersion(major: version.major)
        }
    }

    // MARK: - Upload

    /// Upload one file. `sidecar`, when given, rides along as `sidecarData`.
    public func upload(fileAt path: String,
                       assetID: String,
                       createdAt: Date,
                       modifiedAt: Date,
                       sidecar: Data?) async throws -> ImmichAsset {
        let url = try endpoint("/api/assets")
        let fileURL = URL(fileURLWithPath: path)
        let filename = fileURL.lastPathComponent
        let data = try Data(contentsOf: fileURL)

        let boundary = "comfybox-\(UUID().uuidString)"
        var body = Data()
        func field(_ name: String, _ value: String) {
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
            body.append("\(value)\r\n")
        }
        func file(_ name: String, _ filename: String, _ payload: Data, _ type: String) {
            body.append("--\(boundary)\r\n")
            body.append("Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n")
            body.append("Content-Type: \(type)\r\n\r\n")
            body.append(payload)
            body.append("\r\n")
        }

        let formatter = ISO8601DateFormatter()
        // `metadata` is REQUIRED by AssetMediaCreateDto in Immich 2.3.1
        // (verified against the server's own /api/spec.json). It is a list of
        // upsert items; ComfyBox's own metadata rides in `sidecarData`, so an
        // empty list is what this send means.
        field("metadata", "[]")
        field("deviceAssetId", assetID)
        field("deviceId", "comfybox-desktop")
        field("fileCreatedAt", formatter.string(from: createdAt))
        field("fileModifiedAt", formatter.string(from: modifiedAt))
        file("assetData", filename, data, "application/octet-stream")
        if let sidecar { file("sidecarData", filename + ".json", sidecar, "application/json") }
        body.append("--\(boundary)--\r\n")

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        // Immich's duplicate detection is SHA-1 (Codex review, finding 9).
        request.setValue(Self.sha1Base64(of: data), forHTTPHeaderField: "x-immich-checksum")
        request.httpBody = body

        let json = try await send(request)
        guard let id = json["id"] as? String else { throw ImmichError.malformedResponse }
        let status = (json["status"] as? String) ?? "created"
        return ImmichAsset(id: id, status: status)
    }

    /// What the server says it holds for an asset.
    public struct RemoteAssetDetails: Sendable, Equatable {
        public let id: String
        /// Base64 SHA-1, the same spelling Immich takes in `x-immich-checksum`.
        public let checksum: String?
    }

    /// Confirm the asset is really there before anything local is deleted.
    public func assetDetails(id: String) async throws -> RemoteAssetDetails? {
        do {
            let json = try await getJSON(path: "/api/assets/\(id)")
            guard (json["id"] as? String) == id else { return nil }
            return RemoteAssetDetails(id: id, checksum: json["checksum"] as? String)
        } catch ImmichError.http(let code, _) where code == 404 {
            return nil
        }
    }

    /// True when the server holds this asset with these exact bytes. A server
    /// that reports no checksum is trusted on the id alone; one that reports a
    /// DIFFERENT checksum is not (Codex review of the implementation).
    public func assetMatches(id: String, sha1Base64: String) async throws -> Bool {
        guard let details = try await assetDetails(id: id) else { return false }
        guard let remote = details.checksum, !remote.isEmpty else { return true }
        return remote == sha1Base64
    }

    public func assetExists(id: String) async throws -> Bool {
        try await assetDetails(id: id) != nil
    }

    // MARK: - Albums

    /// The album id, creating the album the first time.
    public func ensureAlbum(named name: String, cachedID: String?) async throws -> String {
        if let cachedID, !cachedID.isEmpty, try await albumExists(id: cachedID) { return cachedID }
        let url = try endpoint("/api/albums")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["albumName": name])
        let json = try await send(request)
        guard let id = json["id"] as? String else { throw ImmichError.malformedResponse }
        return id
    }

    public func albumExists(id: String) async throws -> Bool {
        do {
            let json = try await getJSON(path: "/api/albums/\(id)")
            return (json["id"] as? String) == id
        } catch ImmichError.http(let code, _) where code == 400 || code == 404 {
            return false
        }
    }

    public func addToAlbum(albumID: String, assetIDs: [String]) async throws {
        guard !assetIDs.isEmpty else { return }
        let url = try endpoint("/api/albums/\(albumID)/assets")
        var request = URLRequest(url: url)
        request.httpMethod = "PUT"
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["ids": assetIDs])
        _ = try await sendAllowingArray(request)
    }

    // MARK: - Plumbing

    /// Streamed SHA-1 of a file, in Immich's base64 spelling.
    public static func sha1Base64(ofFileAt path: String) throws -> String {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        var hasher = Insecure.SHA1()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return Data(hasher.finalize()).base64EncodedString()
    }

    static func sha1Base64(of data: Data) -> String {
        Data(Insecure.SHA1.hash(data: data)).base64EncodedString()
    }

    private func endpoint(_ path: String) throws -> URL {
        guard !baseURL.isEmpty, !apiKey.isEmpty, let url = URL(string: baseURL + path) else {
            throw ImmichError.notConfigured
        }
        return url
    }

    private func getJSON(path: String) async throws -> [String: Any] {
        var request = URLRequest(url: try endpoint(path))
        request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
        return try await send(request)
    }

    private func send(_ request: URLRequest) async throws -> [String: Any] {
        let value = try await sendAllowingArray(request)
        guard let object = value as? [String: Any] else { throw ImmichError.malformedResponse }
        return object
    }

    private func sendAllowingArray(_ request: URLRequest) async throws -> Any {
        let (data, response) = try await session.data(for: request)
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            throw ImmichError.http(code, String(decoding: data.prefix(200), as: UTF8.self))
        }
        if data.isEmpty { return [String: Any]() }
        return (try? JSONSerialization.jsonObject(with: data)) ?? [String: Any]()
    }
}

private extension Data {
    mutating func append(_ string: String) {
        append(Data(string.utf8))
    }
}
