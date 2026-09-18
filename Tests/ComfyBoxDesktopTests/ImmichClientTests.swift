// ImmichClientTests.swift — talking to Immich without a live server
// (FDD-remote-galleries §3.4). Every request is intercepted, so these run
// anywhere; the live server is only needed for the first real send.

import Testing
import Foundation
@testable import ComfyBoxDesktop

/// Intercepts every request made through the session it configures.
final class ImmichStubProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: (@Sendable (URLRequest) -> (Int, Data))?
    nonisolated(unsafe) static var seen: [URLRequest] = []
    nonisolated(unsafe) static var bodies: [String: Data] = [:]

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        var request = self.request
        // URLProtocol strips httpBody into a stream; read it back for assertions.
        if let stream = request.httpBodyStream {
            stream.open()
            var data = Data()
            let size = 4096
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
            while stream.hasBytesAvailable {
                let read = stream.read(buffer, maxLength: size)
                if read <= 0 { break }
                data.append(buffer, count: read)
            }
            buffer.deallocate()
            stream.close()
            request.httpBody = data
        }
        Self.seen.append(request)
        if let url = request.url?.absoluteString, let body = request.httpBody {
            Self.bodies[url] = body
        }
        let (code, data) = Self.handler?(request) ?? (200, Data("{}".utf8))
        let response = HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}

    static func reset() { seen = []; bodies = [:]; handler = nil }

    static func session() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [ImmichStubProtocol.self]
        return URLSession(configuration: config)
    }
}

@Suite("ImmichClient", .serialized)
struct ImmichClientTests {

    private func client() -> ImmichClient {
        ImmichClient(baseURL: "http://10.0.100.232:2283/", apiKey: "secret-key", session: ImmichStubProtocol.session())
    }

    private func tempFile(_ contents: String) -> String {
        let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("immich-\(UUID().uuidString).png")
        FileManager.default.createFile(atPath: path, contents: Data(contents.utf8))
        return path
    }

    @Test("the version is read, and a future major version is refused")
    func versionGate() async throws {
        ImmichStubProtocol.reset()
        ImmichStubProtocol.handler = { _ in (200, Data(#"{"major":2,"minor":3,"patch":1}"#.utf8)) }
        let version = try await client().serverVersion()
        #expect(version.major == 2 && version.minor == 3)
        try await client().assertSupportedVersion()

        ImmichStubProtocol.handler = { _ in (200, Data(#"{"major":9,"minor":0,"patch":0}"#.utf8)) }
        await #expect(throws: ImmichError.unsupportedServerVersion(major: 9)) {
            try await client().assertSupportedVersion()
        }
    }

    @Test("an upload carries the key, the ids, the file and the SHA-1 checksum")
    func uploadShape() async throws {
        ImmichStubProtocol.reset()
        ImmichStubProtocol.handler = { _ in (201, Data(#"{"id":"immich-1","status":"created"}"#.utf8)) }
        let path = tempFile("pretend png")

        let asset = try await client().upload(fileAt: path, assetID: "a1",
                                              createdAt: Date(timeIntervalSince1970: 1_700_000_000),
                                              modifiedAt: Date(timeIntervalSince1970: 1_700_000_000),
                                              sidecar: nil)
        #expect(asset.id == "immich-1")
        #expect(asset.isPresent)

        let request = try #require(ImmichStubProtocol.seen.last)
        #expect(request.httpMethod == "POST")
        #expect(request.url?.path == "/api/assets")
        #expect(request.value(forHTTPHeaderField: "x-api-key") == "secret-key")
        let checksum = try #require(request.value(forHTTPHeaderField: "x-immich-checksum"))
        #expect(checksum == ImmichClient.sha1Base64(of: Data("pretend png".utf8)), "SHA-1, not SHA-256")

        let body = String(decoding: try #require(request.httpBody), as: UTF8.self)
        #expect(body.contains("name=\"deviceAssetId\""))
        #expect(body.contains("a1"))
        #expect(body.contains("name=\"assetData\""))
        // Two shapes learned from the live server on 2026-09-17: an EMPTY
        // metadata list makes Immich fail its own insert (HTTP 500), and a JSON
        // sidecar is rejected outright ("Unsupported file type"). So neither is
        // sent; the recipe goes on the description.
        #expect(!body.contains("name=\"metadata\""))
        #expect(!body.contains("name=\"sidecarData\""))
    }

    @Test("the recipe is put on the asset description")
    func descriptionCarriesTheRecipe() async throws {
        ImmichStubProtocol.reset()
        ImmichStubProtocol.handler = { _ in (200, Data("{}".utf8)) }
        try await client().setDescription(assetID: "immich-1", text: #"{"prompt":"a barista"}"#)
        let request = try #require(ImmichStubProtocol.seen.last)
        #expect(request.httpMethod == "PUT")
        #expect(request.url?.path == "/api/assets/immich-1")
        let body = String(decoding: try #require(request.httpBody), as: UTF8.self)
        #expect(body.contains("description"))
        #expect(body.contains("a barista"))
    }

    @Test("a duplicate upload counts as present")
    func duplicateIsPresent() async throws {
        ImmichStubProtocol.reset()
        ImmichStubProtocol.handler = { _ in (200, Data(#"{"id":"immich-1","status":"duplicate"}"#.utf8)) }
        let asset = try await client().upload(fileAt: tempFile("x"), assetID: "a1",
                                              createdAt: Date(), modifiedAt: Date(), sidecar: nil)
        #expect(asset.status == "duplicate")
        #expect(asset.isPresent, "the bytes are on the server; the local copy may go")
    }

    @Test("a rejected upload throws with the server's status")
    func uploadFailure() async throws {
        ImmichStubProtocol.reset()
        ImmichStubProtocol.handler = { _ in (413, Data(#"{"message":"too large"}"#.utf8)) }
        await #expect(throws: ImmichError.self) {
            _ = try await client().upload(fileAt: tempFile("x"), assetID: "a1",
                                          createdAt: Date(), modifiedAt: Date(), sidecar: nil)
        }
    }

    @Test("verification asks for the asset, and a 404 is a plain no")
    func verification() async throws {
        ImmichStubProtocol.reset()
        ImmichStubProtocol.handler = { request in
            request.url?.path == "/api/assets/immich-1"
                ? (200, Data(#"{"id":"immich-1"}"#.utf8))
                : (404, Data(#"{"message":"not found"}"#.utf8))
        }
        #expect(try await client().assetExists(id: "immich-1"))
        #expect(!(try await client().assetExists(id: "nope")))
    }

    @Test("an album is created once and reused, and assets are added by id list")
    func albums() async throws {
        ImmichStubProtocol.reset()
        ImmichStubProtocol.handler = { request in
            if request.httpMethod == "POST", request.url?.path == "/api/albums" {
                return (201, Data(#"{"id":"album-1"}"#.utf8))
            }
            if request.url?.path == "/api/albums/album-1" { return (200, Data(#"{"id":"album-1"}"#.utf8)) }
            return (200, Data("[]".utf8))
        }
        let created = try await client().ensureAlbum(named: "ComfyBox", cachedID: nil)
        #expect(created == "album-1")
        let reused = try await client().ensureAlbum(named: "ComfyBox", cachedID: "album-1")
        #expect(reused == "album-1")
        #expect(ImmichStubProtocol.seen.filter { $0.httpMethod == "POST" && $0.url?.path == "/api/albums" }.count == 1,
                "the album is created once")

        try await client().addToAlbum(albumID: "album-1", assetIDs: ["immich-1", "immich-2"])
        let put = try #require(ImmichStubProtocol.seen.last { $0.httpMethod == "PUT" })
        #expect(put.url?.path == "/api/albums/album-1/assets")
        let body = String(decoding: try #require(put.httpBody), as: UTF8.self)
        #expect(body.contains("\"ids\""))
        #expect(body.contains("immich-1") && body.contains("immich-2"))
    }

    @Test("a client with no key never reaches the network")
    func notConfigured() async throws {
        ImmichStubProtocol.reset()
        let bare = ImmichClient(baseURL: "http://10.0.100.232:2283", apiKey: "", session: ImmichStubProtocol.session())
        await #expect(throws: ImmichError.notConfigured) { _ = try await bare.assetExists(id: "x") }
        #expect(ImmichStubProtocol.seen.isEmpty)
    }
}
