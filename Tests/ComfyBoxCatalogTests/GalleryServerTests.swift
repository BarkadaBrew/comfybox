import XCTest
@testable import ComfyBoxCatalog

final class GalleryServerTests: XCTestCase {
    private var path: String!
    private var store: CatalogStore!

    override func setUp() async throws {
        try await super.setUp()
        path = NSTemporaryDirectory() + "srv-\(UUID().uuidString).sqlite3"
        store = try await CatalogStore.open(path: path)
        try await store.upsert(CatalogAsset(id: "k1", filename: "k1.png", absolutePath: "/tmp/k1.png",
                                            realm: .kira, prompt: "a tulip", contentMode: "neutral",
                                            lane: "still"), explicitCollectionIDs: [])
        try await store.upsert(CatalogAsset(id: "k2", filename: "k2.png", absolutePath: "/tmp/k2.png",
                                            realm: .kira, prompt: "a nightclub", contentMode: "avocado",
                                            lane: "kira"), explicitCollectionIDs: [])
        try await store.upsert(CatalogAsset(id: "s1", filename: "s1.png", absolutePath: "/tmp/s1.png",
                                            realm: .shared, prompt: "a tulip", contentMode: "neutral"),
                               explicitCollectionIDs: [])
    }

    override func tearDown() async throws {
        store = nil
        try? FileManager.default.removeItem(atPath: path)
        try? FileManager.default.removeItem(atPath: path + "-wal")
        try? FileManager.default.removeItem(atPath: path + "-shm")
        try await super.tearDown()
    }

    private func response(_ target: String, actor: String? = nil,
                          ceiling: String? = nil) async -> HTTPKit.Response {
        var headers: [String: String] = [:]
        if let actor { headers["x-catalog-actor"] = actor }
        if let ceiling { headers["x-catalog-ceiling"] = ceiling }
        let req = HTTPKit.Request(method: "GET", target: target, headers: headers, body: Data())
        return await GalleryServer.handle(request: req, store: store)
    }

    private func get(_ target: String, actor: String? = nil, ceiling: String? = nil) async throws -> [String: Any] {
        let res = await response(target, actor: actor, ceiling: ceiling)
        XCTAssertEqual(res.status, 200, "unexpected status for \(target)")
        return try XCTUnwrap(JSONSerialization.jsonObject(with: res.body) as? [String: Any])
    }

    func testHealthz() async throws {
        let body = try await get("/healthz")
        XCTAssertEqual(body["ok"] as? Bool, true)
    }

    func testSearchWithoutActorSeesEveryRealm() async throws {
        let body = try await get("/v1/catalog/search?limit=100")
        XCTAssertEqual(body["count"] as? Int, 3)
    }

    func testKiraActorHeaderScopesToHerRealm() async throws {
        let body = try await get("/v1/catalog/search?limit=100", actor: "kira")
        let items = try XCTUnwrap(body["items"] as? [[String: Any]])
        XCTAssertEqual(Set(items.compactMap { $0["id"] as? String }), ["k1", "k2"])
    }

    /// The lock cannot be widened from the wire, whatever the caller sends.
    func testRealmQueryParameterCannotOverrideTheActorHeader() async throws {
        for attempt in ["realm=shared", "realm=", "scope=shared", "realm=shared&realm=shared"] {
            let body = try await get("/v1/catalog/search?limit=100&\(attempt)", actor: "kira")
            let items = try XCTUnwrap(body["items"] as? [[String: Any]])
            XCTAssertTrue(items.allSatisfy { ($0["realm"] as? String) == "kira" },
                          "leaked with query \(attempt)")
        }
    }

    func testCeilingHeaderClampsTextAndPath() async throws {
        let body = try await get("/v1/catalog/search?limit=100", actor: "kira", ceiling: "apple")
        let items = try XCTUnwrap(body["items"] as? [[String: Any]])
        let avocado = try XCTUnwrap(items.first { ($0["id"] as? String) == "k2" })
        XCTAssertNil(avocado["prompt"])
        XCTAssertNil(avocado["path"])
        XCTAssertEqual(avocado["tier"] as? String, "avocado", "the label is metadata and survives")
    }

    func testFullTextSearch() async throws {
        let body = try await get("/v1/catalog/search?q=tulip&limit=100", actor: "kira")
        XCTAssertEqual(body["count"] as? Int, 1)
    }

    func testCollectionsAreRealmFiltered() async throws {
        let all = try await get("/v1/catalog/collections")
        let hers = try await get("/v1/catalog/collections", actor: "kira")
        let allSlugs = Set(try XCTUnwrap(all["items"] as? [[String: Any]]).compactMap { $0["slug"] as? String })
        let herSlugs = Set(try XCTUnwrap(hers["items"] as? [[String: Any]]).compactMap { $0["slug"] as? String })
        XCTAssertTrue(allSlugs.contains("decoupage"))
        XCTAssertTrue(herSlugs.contains("kira-still-life"))
        XCTAssertTrue(herSlugs.contains("decoupage"), "shared vocabulary is visible to her")
    }

    func testAssetDetailIncludesLocationsAndEdges() async throws {
        try await store.addLocation(assetID: "k1",
            AssetLocation(host: "mac", path: "/tmp/k1.png", mtime: Date()))
        try await store.addEdge(AssetEdge(fromAssetID: "k2", toAssetID: "k1", relation: .i2vSource))
        let body = try await get("/v1/catalog/asset/k1", actor: "kira")
        XCTAssertEqual((body["locations"] as? [[String: Any]])?.count, 1)
        XCTAssertEqual((body["edges"] as? [[String: Any]])?.count, 1)
    }

    func testKiraCannotFetchASharedAssetById() async throws {
        let req = HTTPKit.Request(method: "GET", target: "/v1/catalog/asset/s1",
                                  headers: ["x-catalog-actor": "kira"], body: Data())
        let res = await GalleryServer.handle(request: req, store: store)
        XCTAssertEqual(res.status, 404, "a shared row must be invisible to her, not merely filtered")
    }

    /// The detail route reaches the graph and the location table — both of which
    /// hold real on-disk paths. Neither may be walked out of the caller's realm.
    func testAssetDetailNeverRevealsASharedPathOrACrossRealmEdge() async throws {
        try await store.addLocation(assetID: "k1",
            AssetLocation(host: "mac", path: "/tmp/k1.png", mtime: Date()))
        try await store.addLocation(assetID: "s1",
            AssetLocation(host: "mac", path: "/tmp/s1.png", mtime: Date()))
        // A clip of hers animated from a SHARED still: the edge exists in the
        // table and the unscoped lookup would hand it to her.
        try await store.addEdge(AssetEdge(fromAssetID: "k1", toAssetID: "s1", relation: .i2vSource))

        let res = await response("/v1/catalog/asset/k1", actor: "kira")
        XCTAssertEqual(res.status, 200)
        let text = String(decoding: res.body, as: UTF8.self)
        XCTAssertFalse(text.contains("/tmp/s1.png"), "leaked a shared on-disk path")
        XCTAssertFalse(text.contains("\"s1\""), "leaked a shared asset id through the graph")

        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: res.body) as? [String: Any])
        XCTAssertEqual((body["edges"] as? [[String: Any]])?.count, 0)
        XCTAssertEqual((body["locations"] as? [[String: Any]])?.count, 1)

        // The same walk, unconfined, DOES see it — otherwise this test would
        // pass against a store that simply lost the edge.
        let open = try await get("/v1/catalog/asset/k1")
        XCTAssertEqual((open["edges"] as? [[String: Any]])?.count, 1)
    }

    /// The detail route must go through the same clamp `search` does. Fetching
    /// one row by id is not a way around the ceiling.
    func testAssetDetailHonoursTheCeiling() async throws {
        let body = try await get("/v1/catalog/asset/k2", actor: "kira", ceiling: "apple")
        XCTAssertNil(body["prompt"], "the clamp was bypassed on the detail route")
        XCTAssertNil(body["path"])
        XCTAssertEqual(body["tier"] as? String, "avocado")
    }

    /// SQLite reads a negative LIMIT as "no limit". A caller must not be able to
    /// turn the page cap off by asking for -1.
    func testNegativeLimitCannotDisableTheLimit() async throws {
        let body = try await get("/v1/catalog/search?limit=-1")
        XCTAssertEqual(body["count"] as? Int, 1)
    }

    func testUnknownRouteIs404() async throws {
        let req = HTTPKit.Request(method: "GET", target: "/nope", headers: [:], body: Data())
        let res = await GalleryServer.handle(request: req, store: store)
        XCTAssertEqual(res.status, 404)
    }

    func testQueryParsingHandlesPercentEncodingAndMissingValues() {
        let q = HTTPKit.queryParameters(of: "/v1/catalog/search?q=film%20noir&lane=&limit=10")
        XCTAssertEqual(q["q"], "film noir")
        XCTAssertEqual(q["lane"], "")
        XCTAssertEqual(q["limit"], "10")
    }

    /// A TCP read is not a message boundary. The parser must hold its peace
    /// until the headers are terminated and the declared body has arrived.
    /// End to end over a real socket. Every route test above calls `handle`
    /// directly, which cannot see whether the listener ever delivers a
    /// connection to it — and the first version of the CLI entry point bound the
    /// port, reported "listening", and then dropped every request on the floor
    /// because the `Server` had been deallocated and `newConnectionHandler`
    /// held it weakly. A bound port that answers nothing passes every health
    /// check, so the only test that catches it is one that actually asks.
    func testServerAnswersOverLoopback() async throws {
        let store = self.store!
        var started: HTTPKit.Server?
        var port: UInt16 = 0
        for candidate in UInt16(47_900)...UInt16(47_930) {
            let s = HTTPKit.Server(port: candidate) { await GalleryServer.handle(request: $0, store: store) }
            if (try? s.start()) != nil { started = s; port = candidate; break }
        }
        let server = try XCTUnwrap(started, "could not bind any loopback port")
        defer { server.stop() }

        let url = URL(string: "http://127.0.0.1:\(port)/v1/catalog/search?limit=100")!
        var request = URLRequest(url: url, timeoutInterval: 2)
        request.setValue("kira", forHTTPHeaderField: "X-Catalog-Actor")

        var data: Data?
        var status = 0
        // The listener needs a moment to reach .ready.
        for _ in 0..<8 {
            if let (d, r) = try? await URLSession.shared.data(for: request),
               let http = r as? HTTPURLResponse {
                data = d; status = http.statusCode; break
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(status, 200, "the socket accepted but never answered")
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: try XCTUnwrap(data)) as? [String: Any])
        // The realm lock holds over the wire, not just in the function.
        XCTAssertEqual(body["count"] as? Int, 2)
    }

    /// The bug above, isolated: a STARTED server must keep serving after the
    /// caller drops its reference, because that is precisely what
    /// `runCLIEntryPoint` does — it builds the server inside a Task whose body
    /// then ends. With a weakly-captured `newConnectionHandler` this request
    /// gets a bound socket and no answer.
    ///
    /// Nothing here can call `stop()` — having a reference would defeat the
    /// test — so this leaks one listener for the lifetime of the test process.
    /// That is the cost of testing the ownership rule rather than asserting it.
    func testAStartedServerKeepsServingAfterTheCallerDropsIt() async throws {
        let store = self.store!
        func startAndForget() -> UInt16? {
            for candidate in UInt16(47_940)...UInt16(47_970) {
                let s = HTTPKit.Server(port: candidate) {
                    await GalleryServer.handle(request: $0, store: store)
                }
                if (try? s.start()) != nil { return candidate }
            }
            return nil
        }
        let port = try XCTUnwrap(startAndForget(), "could not bind any loopback port")

        let request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/healthz")!,
                                 timeoutInterval: 2)
        var status = 0
        for _ in 0..<8 {
            if let (_, r) = try? await URLSession.shared.data(for: request),
               let http = r as? HTTPURLResponse {
                status = http.statusCode; break
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(status, 200, "the server was deallocated out from under its own listener")
    }

    func testParserWaitsForASplitRequest() throws {
        let whole = Data("POST /v1/catalog/file HTTP/1.1\r\nHost: x\r\nContent-Length: 9\r\n\r\n{\"a\":\"b\"}".utf8)
        for cut in 1..<whole.count {
            XCTAssertNil(HTTPKit.parseComplete(whole.prefix(cut)),
                         "parsed a truncated request at \(cut) bytes")
        }
        let req = try XCTUnwrap(HTTPKit.parseComplete(whole))
        XCTAssertEqual(req.method, "POST")
        XCTAssertEqual(req.headers["content-length"], "9")
        XCTAssertEqual(req.body.count, 9)
    }
}
