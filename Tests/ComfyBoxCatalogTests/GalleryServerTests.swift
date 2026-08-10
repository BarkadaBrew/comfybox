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
        // k2 carries one of EVERY text-bearing field, so the clamp assertions
        // below fail loudly rather than pass because the field was empty. `theme`
        // in particular: it looks like a facet and holds a free-text intent line.
        try await store.upsert(CatalogAsset(id: "k2", filename: "k2.png", absolutePath: "/tmp/k2.png",
                                            realm: .kira, prompt: "a nightclub",
                                            negativePrompt: "NEGTEXT", promptRaw: "RAWTEXT",
                                            promptInjected: "INJECTEDTEXT", caption: "CAPTIONTEXT",
                                            captionSource: "CAPSOURCE",
                                            preset: "PRESETTEXT", loras: "[\"LORATEXT\"]",
                                            renderID: "RENDERIDTEXT",
                                            contentMode: "avocado",
                                            lane: "kira", arc: "ARCTEXT", theme: "THEMETEXT",
                                            family: "FAMILYTEXT", style: "STYLETEXT"),
                               explicitCollectionIDs: [])
        try await store.addLocation(assetID: "k2", AssetLocation(
            host: "kira", path: "/Volumes/todd/.kira/studio/video/avocado/LOCATIONTEXT.mp4",
            mtime: Date(timeIntervalSince1970: 1)))
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
    ///
    /// This used to assert `prompt` and `path` and nothing else, so it passed
    /// while `theme` — a free-text intent line read out of render-journal.jsonl
    /// — sailed straight through the clamp. Every withheld field is named here,
    /// AND the whole response body is swept for each marker string, so a field
    /// added to `CatalogAsset` and carried through the clamp by accident is
    /// caught by the sweep even though no assertion knows its name.
    func testAssetDetailHonoursTheCeiling() async throws {
        let res = await response("/v1/catalog/asset/k2", actor: "kira", ceiling: "apple")
        XCTAssertEqual(res.status, 200)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: res.body) as? [String: Any])

        // The tier LABEL survives — that is the point of a clamp rather than a
        // 404 — and so do the facets and the numbers.
        XCTAssertEqual(body["tier"] as? String, "avocado")
        XCTAssertEqual(body["realm"] as? String, "kira")
        XCTAssertEqual(body["lane"] as? String, "kira")
        XCTAssertEqual(body["id"] as? String, "k2")

        for field in ["prompt", "prompt_raw", "caption", "theme", "arc",
                      "preset", "family", "style", "path", "filename"] {
            XCTAssertNil(body[field], "\(field) survived the clamp")
        }

        // Nothing withheld may appear ANYWHERE in the body, under any key.
        let text = String(decoding: res.body, as: UTF8.self)
        for marker in ["nightclub", "NEGTEXT", "RAWTEXT", "INJECTEDTEXT", "CAPTIONTEXT",
                       "CAPSOURCE", "PRESETTEXT", "LORATEXT", "RENDERIDTEXT",
                       "ARCTEXT", "THEMETEXT", "FAMILYTEXT", "STYLETEXT",
                       "k2.png", "/tmp/k2.png"] {
            XCTAssertFalse(text.contains(marker), "clamped response leaked \(marker)")
        }
    }

    /// `locations` is realm-scoped in the store but knows nothing about the
    /// ceiling, so the detail route omitted `path` from the clamped row and then
    /// returned the very same absolute path two lines later under `locations`.
    /// A live fetch at ceiling `neutral` came back `path: false, prompt: false`
    /// beside a full /Volumes/todd/…/avocado/… filename — and in this library
    /// the filename IS the prompt.
    func testLocationsAreWithheldAboveTheCeiling() async throws {
        let res = await response("/v1/catalog/asset/k2", actor: "kira", ceiling: "neutral")
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: res.body) as? [String: Any])
        XCTAssertEqual((body["locations"] as? [[String: Any]])?.count, 0,
                       "on-disk paths leaked past the ceiling through locations")
        XCTAssertFalse(String(decoding: res.body, as: UTF8.self).contains("LOCATIONTEXT"))

        // The same fetch UNDER the ceiling still has them — otherwise this test
        // would pass against a route that simply lost the locations.
        let open = try await get("/v1/catalog/asset/k2", actor: "kira", ceiling: "avocado")
        XCTAssertEqual((open["locations"] as? [[String: Any]])?.count, 1)
    }

    /// The clamp's only public entry point. `tierRank`/`ceilingRank` are
    /// internal so that a cross-module caller cannot spell the comparison the
    /// fail-open way (`tierRank(asset) > tierRank(ceiling)`, in which an
    /// unrecognised ceiling withholds nothing).
    func testIsWithheldIsTheOneDecision() {
        XCTAssertFalse(isWithheld(tier: "avocado", ceiling: nil), "no ceiling, no clamp")
        XCTAssertTrue(isWithheld(tier: "avocado", ceiling: "neutral"))
        XCTAssertFalse(isWithheld(tier: "neutral", ceiling: "avocado"))
        XCTAssertTrue(isWithheld(tier: "banana", ceiling: "never-heard-of-it"),
                      "an unrecognised ceiling must admit the least, not the most")
        XCTAssertTrue(isWithheld(tier: "never-heard-of-it", ceiling: "avocado"),
                      "an unrecognised tier must be withheld from every ceiling")
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

    // MARK: - POST /v1/catalog/file (the only write route)

    private func post(_ target: String, json: String,
                      actor: String? = nil,
                      extraHeaders: [String: String] = [:]) async -> HTTPKit.Response {
        var headers = extraHeaders
        if let actor { headers["x-catalog-actor"] = actor }
        headers["content-type"] = "application/json"
        let req = HTTPKit.Request(method: "POST", target: target,
                                  headers: headers, body: Data(json.utf8))
        return await GalleryServer.handle(request: req, store: store)
    }

    /// Every collection the asset is in. The fixtures arrive already DERIVED-filed
    /// by `upsert`, so these tests assert against a baseline rather than against
    /// emptiness — a test that files an asset into the collection its lane
    /// already implies would pass without the route existing.
    private func memberships(of assetID: String) async throws -> Set<String> {
        var out: Set<String> = []
        for c in try await store.collections(visibleTo: nil) {
            let q = CatalogQuery(scope: nil, collectionID: c.id, limit: 500)
            if try await store.search(q).contains(where: { $0.id == assetID }) { out.insert(c.id) }
        }
        return out
    }

    func testFileRouteFilesAnAsset() async throws {
        let before = try await memberships(of: "k1")
        XCTAssertFalse(before.contains("col-kira-dreams-memories"), "fixture already filed there")

        let res = await post("/v1/catalog/file",
                             json: #"{"asset_id":"k1","collection_id":"col-kira-dreams-memories"}"#)
        XCTAssertEqual(res.status, 200)
        XCTAssertEqual(try JSONSerialization.jsonObject(with: res.body) as? [String: Bool],
                       ["ok": true])
        let ids = try await memberships(of: "k1")
        XCTAssertTrue(ids.contains("col-kira-dreams-memories"),
                      "the route returned ok but filed nothing")
    }

    /// Her own asset into her own collection, with the realm lock engaged. This
    /// is the case the whole route exists for — "she curates her own realm".
    func testKiraMayFileHerOwnAssetIntoHerOwnCollection() async throws {
        let before = try await memberships(of: "k2")
        XCTAssertFalse(before.contains("col-kira-nightlife"),
                       "fixture already filed there — this test would prove nothing")
        let res = await post("/v1/catalog/file",
                             json: #"{"asset_id":"k2","collection_id":"col-kira-nightlife"}"#,
                             actor: "kira")
        XCTAssertEqual(res.status, 200)
        let after = try await memberships(of: "k2")
        XCTAssertTrue(after.contains("col-kira-nightlife"))
    }

    /// A shared asset can never enter a kira collection — a property of the DATA,
    /// so it holds however the caller identifies itself. If the route ever stops
    /// passing `by: scope` this is one of the two tests that notices.
    func testKiraCannotFileASharedAssetIntoHerRealm() async throws {
        let before = try await memberships(of: "s1")
        let res = await post("/v1/catalog/file",
                             json: #"{"asset_id":"s1","collection_id":"col-kira-everyday"}"#,
                             actor: "kira")
        XCTAssertEqual(res.status, 403)
        let after = try await memberships(of: "s1")
        XCTAssertEqual(after, before, "refused and filed it anyway")
    }

    /// The other one. `by: nil` would let her file her own work into a SHARED
    /// collection's structure and reach rows outside her realm; with the scope
    /// passed, an asset that is not hers is not hers to file.
    func testKiraCannotFileAnAssetOutsideHerRealmAtAll() async throws {
        // Not a realm-crossing COLLECTION this time — a shared collection she is
        // otherwise allowed to contribute to. The asset is what is out of reach.
        let res = await post("/v1/catalog/file",
                             json: #"{"asset_id":"s1","collection_id":"col-adult"}"#,
                             actor: "kira")
        XCTAssertEqual(res.status, 403)
        let after = try await memberships(of: "s1")
        XCTAssertFalse(after.contains("col-adult"))
    }

    /// Contributing to a shared collection is allowed; restructuring one is not.
    /// The store draws that line and the route must not blunt it.
    func testKiraMayContributeToASharedCollection() async throws {
        let before = try await memberships(of: "k1")
        XCTAssertFalse(before.contains("col-photography"),
                       "fixture already filed there — this test would prove nothing")
        let res = await post("/v1/catalog/file",
                             json: #"{"asset_id":"k1","collection_id":"col-photography"}"#,
                             actor: "kira")
        XCTAssertEqual(res.status, 200)
        let after = try await memberships(of: "k1")
        XCTAssertTrue(after.contains("col-photography"))
    }

    /// The store's refusal names the actor, the collection and its realm. That
    /// is an oracle for collections a confined caller cannot otherwise see.
    func testAFileRefusalDoesNotPutTheStoresWordsOnTheWire() async throws {
        let res = await post("/v1/catalog/file",
                             json: #"{"asset_id":"s1","collection_id":"col-kira-everyday"}"#,
                             actor: "kira")
        XCTAssertEqual(res.status, 403)
        XCTAssertEqual(try JSONSerialization.jsonObject(with: res.body) as? [String: String],
                       ["error": "not permitted"])
        let text = String(decoding: res.body, as: UTF8.self).lowercased()
        for leak in ["kira", "realm", "collection", "shared", "everyday"] {
            XCTAssertFalse(text.contains(leak), "leaked '\(leak)' from the store's message")
        }
    }

    /// A collection that does not exist is refused the same way one she may not
    /// touch is: the route must not become a way to enumerate collection ids.
    func testAnUnknownCollectionIsRefusedIdenticallyToAForbiddenOne() async throws {
        let unknown = await post("/v1/catalog/file",
                                 json: #"{"asset_id":"k1","collection_id":"col-does-not-exist"}"#,
                                 actor: "kira")
        let forbidden = await post("/v1/catalog/file",
                                   json: #"{"asset_id":"s1","collection_id":"col-kira-everyday"}"#,
                                   actor: "kira")
        XCTAssertEqual(unknown.status, 403)
        XCTAssertEqual(unknown.status, forbidden.status)
        XCTAssertEqual(unknown.body, forbidden.body)
    }

    func testMalformedFileBodiesAre400() async throws {
        let bodies = [
            "",                                             // no body at all
            "not json",
            "[]",                                           // an array, not an object
            "null",
            "{}",                                           // neither key
            #"{"asset_id":"k1"}"#,                          // incomplete
            #"{"collection_id":"col-kira-still-life"}"#,    // incomplete
            #"{"asset_id":"","collection_id":"col-kira-still-life"}"#,
            #"{"asset_id":"  ","collection_id":"col-kira-still-life"}"#,
            #"{"asset_id":"k1","collection_id":null}"#,
            #"{"asset_id":7,"collection_id":"col-kira-still-life"}"#,
            #"{"asset_id":"k1","collection_id":"col-kira-still-life""#,   // truncated JSON
        ]
        let before = try await memberships(of: "k1")
        for body in bodies {
            let res = await post("/v1/catalog/file", json: body, actor: "kira")
            XCTAssertEqual(res.status, 400, "body \(body.isEmpty ? "<empty>" : body) was not rejected")
        }
        let after = try await memberships(of: "k1")
        XCTAssertEqual(after, before, "a malformed body still filed something")
    }

    /// The write route is behind the SAME two gates as every read route, and
    /// both must fire BEFORE the body is looked at.
    func testTheWriteRouteIsBehindTheActorAndOriginGates() async throws {
        let good = #"{"asset_id":"k1","collection_id":"col-kira-dreams-memories"}"#
        for imposter in ["kira-worker", "todd", "", "  "] {
            let res = await post("/v1/catalog/file", json: good, actor: imposter)
            XCTAssertEqual(res.status, 400, "actor '\(imposter)' was allowed to file")
        }
        let browser = await post("/v1/catalog/file", json: good, actor: "kira",
                                 extraHeaders: ["origin": "http://localhost:3000"])
        XCTAssertEqual(browser.status, 403, "a page could file into the catalog")
        let sameSite = await post("/v1/catalog/file", json: good, actor: "kira",
                                  extraHeaders: ["sec-fetch-site": "cross-site"])
        XCTAssertEqual(sameSite.status, 403)
        let after = try await memberships(of: "k1")
        XCTAssertFalse(after.contains("col-kira-dreams-memories"),
                       "a request that never passed the gates still wrote a row")
    }

    /// The route parses the body, so `Content-Length` is now a message BOUNDARY
    /// rather than a minimum. A peer whose bytes arrive with anything appended
    /// used to hand the JSON parser the trailing garbage and turn a valid
    /// request into an unexplainable 400.
    func testAnOvershootingBodyIsTruncatedToContentLength() throws {
        let json = #"{"asset_id":"k1","collection_id":"col-kira-still-life"}"#
        let raw = "POST /v1/catalog/file HTTP/1.1\r\nContent-Length: \(json.utf8.count)\r\n\r\n"
            + json + "GET /healthz HTTP/1.1\r\n\r\n"
        let req = try XCTUnwrap(HTTPKit.parseComplete(Data(raw.utf8)))
        XCTAssertEqual(req.body.count, json.utf8.count)
        XCTAssertEqual(String(decoding: req.body, as: UTF8.self), json)
        XCTAssertNotNil(GalleryServer.fileRequestBody(req.body),
                        "the trailing request was parsed as part of the body")
    }

    /// `Content-Length: -1` declares nothing, and a naive truncation on it is a
    /// `prefix(-1)` precondition failure — one header, process gone.
    func testANegativeContentLengthDoesNotTrapOrBlock() throws {
        let raw = "POST /v1/catalog/file HTTP/1.1\r\nContent-Length: -1\r\n\r\n{}"
        let req = try XCTUnwrap(HTTPKit.parseComplete(Data(raw.utf8)))
        XCTAssertEqual(String(decoding: req.body, as: UTF8.self), "{}")
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

    // MARK: - CLI argument parsing (the SERVE path)

    /// The serve path CREATES the database, so a vault `--db` would write a
    /// catalog of raw prompt text INTO the vault. Same flag, same hazard as the
    /// backfill path, one function up.
    func testServeRefusesAVaultDBPath() {
        let result = GalleryServer.parseServeArgs(
            ["--db", "/Users/x/Documents/Vaults/BarkadaAI/x.sqlite3"])
        XCTAssertEqual(result, .failure(.vaultPath("/Users/x/Documents/Vaults/BarkadaAI/x.sqlite3")))
    }

    /// A typo'd flag that silently does nothing is the failure mode this task
    /// exists to rule out.
    func testServeRejectsAnUnknownFlagRatherThanIgnoringIt() {
        XCTAssertEqual(GalleryServer.parseServeArgs(["--prot", "7871"]),
                       .failure(.unknownArgument("--prot")))
    }

    func testServeParsesPortAndDB() throws {
        let opts = try XCTUnwrap(try? GalleryServer.parseServeArgs(
            ["--port", "7999", "--db", "/tmp/x.sqlite3"]).get())
        XCTAssertEqual(opts.port, 7999)
        XCTAssertEqual(opts.dbPath, "/tmp/x.sqlite3")
        XCTAssertFalse(opts.showHelp)
    }

    /// The refusal is on the PATH, not on the flag, so it holds wherever a vault
    /// path is named.
    func testVaultRefusalAppliesToAnyPathNotJustDB() {
        XCTAssertEqual(GalleryServer.vaultRefusal(in: [nil, "/tmp/ok", "/x/Vaults/y"]),
                       .vaultPath("/x/Vaults/y"))
        XCTAssertNil(GalleryServer.vaultRefusal(in: [nil, "/tmp/ok"]))
    }

    /// A value-taking flag given no value used to fall through to the default —
    /// a silent no-op inside the one function whose purpose is never to ignore a
    /// flag.
    func testATrailingValueFlagIsAnErrorNotADefault() {
        XCTAssertEqual(GalleryServer.parseServeArgs(["--db"]), .failure(.missingValue("--db")))
        XCTAssertEqual(GalleryServer.parseServeArgs(["--port"]), .failure(.missingValue("--port")))
        XCTAssertEqual(GalleryServer.parseBackfillArgs(["--kira-studio"]),
                       .failure(.missingValue("--kira-studio")))
    }

    /// The message must name the FLAG. `--port abc` reporting
    /// `unknown argument: abc` sends the reader hunting for a flag they never
    /// typed.
    func testABadPortNamesTheFlagRatherThanTheStrayToken() {
        XCTAssertEqual(GalleryServer.parseServeArgs(["--port", "abc"]),
                       .failure(.badValue(flag: "--port", value: "abc")))
        XCTAssertEqual(GalleryServer.parseServeArgs(["--port", "abc"]).failureMessage,
                       "--port: invalid value abc")
    }

    // MARK: - CLI argument parsing (the BACKFILL path)

    /// Previously reachable only through an inline `exit()`, so demonstrable in a
    /// transcript but never asserted.
    func testBackfillRejectsAnUnknownFlagRatherThanDroppingAnArchive() {
        XCTAssertEqual(GalleryServer.parseBackfillArgs(["--kira-studo", "/tmp/x"]),
                       .failure(.unknownArgument("--kira-studo")))
    }

    func testBackfillRefusesAVaultPathOnEveryRoot() {
        XCTAssertEqual(GalleryServer.parseBackfillArgs(["--db", "/x/Vaults/y.sqlite3"]),
                       .failure(.vaultPath("/x/Vaults/y.sqlite3")))
        XCTAssertEqual(GalleryServer.parseBackfillArgs(["--home", "/x/Vaults/pics"]),
                       .failure(.vaultPath("/x/Vaults/pics")))
        XCTAssertEqual(GalleryServer.parseBackfillArgs(["--kira-history", "/x/Vaults/h.json"]),
                       .failure(.vaultPath("/x/Vaults/h.json")))
    }

    /// The subdivision that makes i2v edges and journal rows resolve at all: each
    /// studio becomes TWO roots and each root carries its OWN remote prefix.
    /// Handing both the studio-level prefix would translate
    /// `…/studio/gallery/x.png` into `…/gallery/gallery/x.png`.
    func testBackfillSubdividesTheRemotePrefixInLockstepWithTheMediaRoot() throws {
        let opts = try XCTUnwrap(try? GalleryServer.parseBackfillArgs(
            ["--kira-studio", "/Volumes/todd/.kira/studio",
             "--kira-remote-prefix", "/home/todd/.kira/studio"]).get())

        let kira = opts.trees.first { $0.id == "kira" }
        let kiraVideo = opts.trees.first { $0.id == "kira-video" }
        XCTAssertEqual(kira?.mediaRoot, "/Volumes/todd/.kira/studio/gallery")
        XCTAssertEqual(kira?.remotePrefix, "/home/todd/.kira/studio/gallery")
        XCTAssertEqual(kiraVideo?.mediaRoot, "/Volumes/todd/.kira/studio/video")
        XCTAssertEqual(kiraVideo?.remotePrefix, "/home/todd/.kira/studio/video")
        // Both roots share one metadata tree, and the home tree has none at all.
        XCTAssertEqual(kira?.metadataRoot, "/Volumes/todd/.kira/studio/metadata")
        XCTAssertEqual(kiraVideo?.metadataRoot, "/Volumes/todd/.kira/studio/metadata")
        XCTAssertNil(opts.trees.first { $0.id == "home" }?.metadataRoot)
    }
}

private extension Result where Failure == GalleryServer.CLIFailure {
    var failureMessage: String? {
        if case .failure(let f) = self { return f.message }
        return nil
    }
}
