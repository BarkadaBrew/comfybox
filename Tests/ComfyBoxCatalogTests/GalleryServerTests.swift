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

    // MARK: - Who is asking

    /// Absent stays unscoped: the desktop app and the backfill tooling send no
    /// actor, and a caller who claims nothing is not claiming to be Kira.
    func testAnAbsentActorHeaderIsUnscoped() async throws {
        let body = try await get("/v1/catalog/search?limit=100")
        XCTAssertEqual(body["count"] as? Int, 3)
    }

    /// PRESENT-but-unrecognised is the dangerous case, and it used to fail OPEN:
    /// anything that was not exactly "kira" fell through to full cross-realm
    /// access, raw prompts and on-disk paths included. A daemon typo, a renamed
    /// tool or a `kira-worker` silently became an unconfined caller.
    func testAnUnrecognisedActorIsRejectedRatherThanWidened() async throws {
        for imposter in ["kira-worker", "kira2", "todd", "shared", "", "  ", "KIRA-WORKER"] {
            let res = await response("/v1/catalog/search?limit=100", actor: imposter)
            XCTAssertEqual(res.status, 400, "actor '\(imposter)' was not rejected")
            let text = String(decoding: res.body, as: UTF8.self)
            XCTAssertFalse(text.contains("/tmp/"), "a rejected actor still saw paths")
            XCTAssertFalse(text.contains("tulip"), "a rejected actor still saw prompts")
        }
    }

    /// Case and surrounding whitespace must not turn her into a stranger — that
    /// direction would 400 a legitimate caller, and the normalisation can only
    /// ever ADD callers to the confined set.
    func testActorMatchingIgnoresCaseAndWhitespace() async throws {
        for spelling in ["kira", "Kira", "  KIRA  "] {
            let body = try await get("/v1/catalog/search?limit=100", actor: spelling)
            let items = try XCTUnwrap(body["items"] as? [[String: Any]])
            XCTAssertTrue(items.allSatisfy { ($0["realm"] as? String) == "kira" },
                          "'\(spelling)' did not land in her realm")
        }
    }

    /// Nothing here is authenticated, so a web page on any local origin could
    /// read the whole catalog if the browser let it. Today the only thing
    /// stopping it is that we happen not to send Access-Control-Allow-Origin.
    func testBrowserOriginatedRequestsAreRefused() async throws {
        for origin in ["http://localhost:3000", "https://evil.example", "null"] {
            let req = HTTPKit.Request(method: "GET", target: "/v1/catalog/search?limit=100",
                                      headers: ["origin": origin], body: Data())
            let res = await GalleryServer.handle(request: req, store: store)
            XCTAssertEqual(res.status, 403, "origin '\(origin)' was served")
        }
        for site in ["cross-site", "same-site"] {
            let req = HTTPKit.Request(method: "GET", target: "/healthz",
                                      headers: ["sec-fetch-site": site], body: Data())
            let res = await GalleryServer.handle(request: req, store: store)
            XCTAssertEqual(res.status, 403, "sec-fetch-site '\(site)' was served")
        }
        // Direct navigation and a page this service itself served stay open, so
        // a future local UI is not pre-emptively broken.
        for site in ["none", "same-origin"] {
            let req = HTTPKit.Request(method: "GET", target: "/healthz",
                                      headers: ["sec-fetch-site": site], body: Data())
            let res = await GalleryServer.handle(request: req, store: store)
            XCTAssertEqual(res.status, 200, "sec-fetch-site '\(site)' should be allowed")
        }
    }

    /// SQLite's errmsg names tables and columns and quotes the failing SQL, and
    /// `error.localizedDescription` put all of it on the wire.
    ///
    /// This tests the formatter the route's `catch` calls rather than trying to
    /// provoke SQLite through a query string. I could not find an input that
    /// makes a healthy database fail — which is good news, and also means an
    /// end-to-end version of this test would assert nothing (an earlier draft
    /// silently skipped). The catch block exists for disk-full and corruption,
    /// so the error is injected directly.
    func testInternalErrorsDoNotLeakSchemaDetail() throws {
        let leaky = CatalogError.prepareFailed(
            "no such column: a.promt in \"SELECT a.id, a.prompt FROM assets a WHERE a.realm = ?1\"")
        let req = HTTPKit.Request(method: "GET", target: "/v1/catalog/search", headers: [:], body: Data())
        let res = GalleryServer.internalError(leaky, for: req)

        XCTAssertEqual(res.status, 500)
        let text = String(decoding: res.body, as: UTF8.self).lowercased()
        XCTAssertFalse(text.contains("assets"), "leaked a table name")
        XCTAssertFalse(text.contains("select"), "leaked SQL")
        XCTAssertFalse(text.contains("realm"), "leaked a column name")
        XCTAssertEqual(try JSONSerialization.jsonObject(with: res.body) as? [String: String],
                       ["error": "internal error"])
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

    // MARK: - Over a real socket

    /// Every route test above calls `handle` directly, which cannot see whether
    /// the listener ever delivers a connection to it — the first version of the
    /// CLI entry point bound the port, reported "listening", and then dropped
    /// every request on the floor.
    ///
    /// Port 0 rather than a fixed one: the kernel picks a free port, so two
    /// copies of this test binary running at once cannot collide, and — more to
    /// the point — cannot silently be answered by EACH OTHER. With a fixed port
    /// and a fixture this deterministic, a second instance whose bind failed
    /// would get a 200 and the right `count` from the FIRST instance's server
    /// and pass without ever having served anything.
    func testServerAnswersOverLoopback() async throws {
        let store = self.store!
        let server = HTTPKit.Server(port: 0) { await GalleryServer.handle(request: $0, store: store) }
        try await server.start()
        defer { server.stop() }
        let port = try XCTUnwrap(server.boundPort, "started but reported no bound port")
        XCTAssertNotEqual(port, 0, "the kernel-assigned port was never read back")

        let url = URL(string: "http://127.0.0.1:\(port)/v1/catalog/search?limit=100")!
        var request = URLRequest(url: url, timeoutInterval: 5)
        request.setValue("kira", forHTTPHeaderField: "X-Catalog-Actor")

        // No retry loop: `start()` does not return until the listener is .ready.
        let (data, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200,
                       "the socket accepted but never answered")
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        // The realm lock holds over the wire, not just in the function.
        XCTAssertEqual(body["count"] as? Int, 2)
    }

    /// A STARTED server must keep serving after the caller drops its reference,
    /// because that is precisely what `runCLIEntryPoint` does — it builds the
    /// server inside a Task whose body then ends. With a weakly-captured
    /// `newConnectionHandler` this request gets a bound socket and no answer.
    ///
    /// Nothing here can call `stop()` — holding a reference would defeat the
    /// test — so this leaks one listener for the lifetime of the test process.
    /// On a kernel-assigned port that leak is inert; on a fixed one it was a
    /// booby trap for the next run.
    func testAStartedServerKeepsServingAfterTheCallerDropsIt() async throws {
        let store = self.store!
        func startAndForget() async throws -> UInt16? {
            let s = HTTPKit.Server(port: 0) {
                await GalleryServer.handle(request: $0, store: store)
            }
            try await s.start()
            return s.boundPort
        }
        let bound = try await startAndForget()
        let port = try XCTUnwrap(bound, "could not bind a loopback port")

        let request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/healthz")!,
                                 timeoutInterval: 5)
        let (_, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200,
                       "the server was deallocated out from under its own listener")
    }

    /// A listener that cannot bind must say so. `NWListener.start(queue:)` is
    /// non-throwing and reports failure only through `stateUpdateHandler`, so
    /// without one this returned success and the caller went on to announce a
    /// service that was never listening — and, under launchd, to block forever
    /// rather than exit and be recycled.
    func testStartFailsWhenThePortIsAlreadyTaken() async throws {
        let store = self.store!
        let first = HTTPKit.Server(port: 0) { await GalleryServer.handle(request: $0, store: store) }
        try await first.start()
        defer { first.stop() }
        let taken = try XCTUnwrap(first.boundPort)

        let second = HTTPKit.Server(port: taken) { _ in .json(["ok": true]) }
        do {
            try await second.start(timeout: 3)
            second.stop()
            XCTFail("bound a port already in use and reported success")
        } catch {
            XCTAssertNil(second.boundPort, "a failed start must not leave a listener behind")
        }
    }

    func testASecondStartIsRejected() async throws {
        let store = self.store!
        let server = HTTPKit.Server(port: 0) { await GalleryServer.handle(request: $0, store: store) }
        try await server.start()
        defer { server.stop() }
        let port = server.boundPort

        do {
            try await server.start()
            XCTFail("a second start must not silently replace the first listener")
        } catch {
            XCTAssertEqual(server.boundPort, port, "the original listener was disturbed")
        }
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
