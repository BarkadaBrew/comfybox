// GalleryServer.swift — the gallery service's routes.
//
// THE REALM LOCK LIVES HERE, at the boundary: `scope` comes from the
// X-Catalog-Actor header the daemon sets per tool, never from a query
// parameter. A client cannot widen its own scope by asking.
//
// Every store call on these routes that TAKES a scope is GIVEN one. `edges`,
// `locations` and the by-id fetch no longer DEFAULT that parameter, so an
// omission is a compile error rather than a silent leak — the first draft of
// this file omitted two of the three and still built.

import Foundation

public enum GalleryServer {

    public static let defaultPort: UInt16 = 7871

    /// Never let a client page the whole catalog out in one request.
    static let maxLimit = 500

    enum ActorScope {
        case resolved(CatalogRealm?)
        case unrecognised
    }

    /// Resolve `X-Catalog-Actor` into a realm scope, failing CLOSED on anything
    /// unfamiliar.
    ///
    /// The distinction that matters is ABSENT versus PRESENT-BUT-UNKNOWN.
    /// Absent is unscoped: the desktop app and the backfill tooling depend on
    /// that, and a caller who sends no actor is not claiming to be one.
    /// Present-but-unknown is an ERROR — a daemon typo, a `kira-worker`, a
    /// renamed tool. Treating those as unscoped is how a confined caller gets
    /// silently promoted to full cross-realm access, raw prompts and on-disk
    /// paths included, by nothing more than a misspelling.
    ///
    /// The value is trimmed and lowercased first, so `Kira ` still lands in her
    /// realm rather than 400ing.
    static func actorScope(_ raw: String?) -> ActorScope {
        guard let raw else { return .resolved(nil) }   // absent: unscoped
        let name = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch name {
        case "kira": return .resolved(.kira)
        default: return .unrecognised   // includes an empty value
        }
    }

    public static func handle(request: HTTPKit.Request, store: CatalogStore) async -> HTTPKit.Response {
        // A browser must not be able to read the catalog from a page. Nothing
        // here is authenticated — it is a loopback service the daemon calls —
        // so today the only thing standing between a local web page and every
        // raw prompt in the database is the accident that we send no
        // Access-Control-Allow-Origin. Make it deliberate instead.
        //
        // curl and the daemon send neither header; a browser fetch always sends
        // one or the other. `sec-fetch-site: none` is direct navigation and
        // `same-origin` is a page this service itself served, so both are left
        // open for a future local UI.
        if request.headers["origin"] != nil {
            return .error(403, "cross-origin requests are not accepted")
        }
        if let site = request.headers["sec-fetch-site"]?.lowercased(),
           site == "cross-site" || site == "same-site" {
            return .error(403, "cross-origin requests are not accepted")
        }

        // Scope and ceiling are HEADERS, set by the trusted daemon per caller.
        let scope: CatalogRealm?
        switch actorScope(request.headers["x-catalog-actor"]) {
        case let .resolved(realm): scope = realm
        case .unrecognised: return .error(400, "unrecognised X-Catalog-Actor")
        }
        let ceiling = request.headers["x-catalog-ceiling"]

        do {
            switch (request.method, request.path) {
            case ("GET", "/healthz"):
                return .json(["ok": true])

            case ("GET", "/v1/catalog/search"):
                let q = query(from: request, scope: scope, ceiling: ceiling)
                let rows = try await store.search(q)
                return .json(["count": rows.count, "items": rows.map(dict(for:))])

            case ("GET", "/v1/catalog/facets"):
                let f = try await store.facets(scope: scope)
                return .json([
                    "lane": f.lane, "tier": f.tier, "character": f.character,
                    "source": f.source, "stock": f.stock, "genre": f.genre,
                    "kind": f.kind, "mode": f.mode, "collection": f.collection,
                ])

            case ("GET", "/v1/catalog/collections"):
                let cols = try await store.collections(visibleTo: scope)
                let counts = try await store.facets(scope: scope).collection
                return .json(["items": cols.map { c in
                    [
                        "id": c.id, "slug": c.slug, "name": c.name,
                        "parent_id": c.parentID as Any,
                        "realm": c.realm?.rawValue as Any,
                        "description": c.description as Any,
                        "count": counts[c.id] ?? 0,
                    ] as [String: Any]
                }])

            case ("GET", let p) where p.hasPrefix("/v1/catalog/asset/"):
                let id = String(p.dropFirst("/v1/catalog/asset/".count))
                // Scoped AND clamped by the store, on the primary key. (A
                // `search` capped at some large limit would answer this by
                // scanning, and would 404 a real row that fell off the end.)
                guard let row = try await store.asset(id: id, visibleTo: scope, ceiling: ceiling) else {
                    // 404 rather than 403: a row outside the caller's realm does
                    // not exist as far as that caller is concerned.
                    return .error(404, "no such asset")
                }
                var body = dict(for: row)
                // `scope` on both: these are the two lookups that reach OUT of
                // the row — to real on-disk paths, and along the graph to other
                // assets' ids. Unscoped, a confined caller walks an i2v edge
                // from her own clip to a shared still and reads a realm she
                // cannot search.
                body["locations"] = try await store.locations(of: id, scope: scope).map {
                    ["host": $0.host, "path": $0.path, "mtime": $0.mtime.timeIntervalSince1970]
                }
                body["edges"] = try await store.edges(for: id, scope: scope).map {
                    ["from": $0.fromAssetID, "to": $0.toAssetID, "relation": $0.relation.rawValue]
                }
                return .json(body)

            default:
                return .error(404, "no such route")
            }
        } catch {
            return internalError(error, for: request)
        }
    }

    /// SQLite's errmsg carries table and column names and fragments of the
    /// failing statement — `error.localizedDescription` put all of it on the
    /// wire. That detail is exactly what an operator reading logs needs and
    /// exactly what someone probing this service wants, so it goes to stderr
    /// and a fixed, uninformative body goes to the caller.
    static func internalError(_ error: Error, for request: HTTPKit.Request) -> HTTPKit.Response {
        FileHandle.standardError.write(
            Data("gallery: \(request.method) \(request.path) failed: \(error)\n".utf8))
        return .error(500, "internal error")
    }

    private static func query(from request: HTTPKit.Request,
                              scope: CatalogRealm?, ceiling: String?) -> CatalogQuery {
        let p = request.query
        func s(_ k: String) -> String? {
            guard let v = p[k], !v.isEmpty else { return nil }
            return v
        }
        func i(_ k: String) -> Int? { s(k).flatMap(Int.init) }
        func d(_ k: String) -> Date? { s(k).flatMap(Double.init).map(Date.init(timeIntervalSince1970:)) }

        var q = CatalogQuery(scope: scope, ceiling: ceiling)
        // NOTE: `realm` is deliberately NOT read from the query string.
        q.text = s("q")
        q.collectionID = s("collection")
        q.lane = s("lane"); q.tier = s("tier"); q.character = s("character")
        q.source = s("source"); q.stock = s("stock"); q.genre = s("genre")
        q.arc = s("arc"); q.kind = s("kind"); q.mode = s("mode")
        q.minDurationMs = i("min_duration"); q.maxDurationMs = i("max_duration")
        q.minRating = i("min_rating")
        q.since = d("since"); q.until = d("until")
        q.orderBy = CatalogOrder(rawValue: s("order") ?? "newest") ?? .newest
        // `max(1, ...)` is not tidiness: SQLite reads a NEGATIVE limit as "no
        // limit at all", so `?limit=-1` would page the entire catalog out in one
        // response — the cap turned off by asking for it.
        q.limit = max(1, min(i("limit") ?? 50, maxLimit))
        q.offset = max(0, i("offset") ?? 0)
        return q
    }

    private static func dict(for a: CatalogAsset) -> [String: Any] {
        var out: [String: Any] = [
            "id": a.id, "kind": a.kind, "realm": a.realm.rawValue,
            "created_at": a.createdAt.timeIntervalSince1970,
        ]
        // A clamped row has an empty path and no text; omit rather than send "".
        if !a.absolutePath.isEmpty { out["path"] = a.absolutePath }
        if !a.filename.isEmpty { out["filename"] = a.filename }
        func put(_ k: String, _ v: Any?) { if let v { out[k] = v } }
        put("prompt", a.prompt); put("prompt_raw", a.promptRaw); put("caption", a.caption)
        put("tier", a.contentMode); put("character", a.characterName); put("source", a.source)
        put("lane", a.lane); put("arc", a.arc); put("theme", a.theme)
        put("stock", a.stock); put("genre", a.genre); put("family", a.family); put("style", a.style)
        put("preset", a.preset); put("model", a.modelFamily); put("seed", a.seed)
        put("mode", a.mode); put("duration_ms", a.durationMs); put("fps", a.fps)
        put("frames", a.frames); put("resolution", a.resolution); put("aspect_ratio", a.aspectRatio)
        put("width", a.width); put("height", a.height)
        out["rating"] = a.rating
        out["favorite"] = a.favorite
        out["sealed"] = a.sealed
        return out
    }

    // MARK: - CLI

    public static func runCLIEntryPoint(args: [String]) {
        if args.first == "backfill" {
            runBackfillCLI(args: Array(args.dropFirst()))
            return
        }
        var port = defaultPort
        var dbPath: String? = nil
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--port": if i + 1 < args.count, let p = UInt16(args[i + 1]) { port = p; i += 1 }
            case "--db": if i + 1 < args.count { dbPath = args[i + 1]; i += 1 }
            case "--help", "-h":
                print("Usage: ComfyBoxGallery [--port \(defaultPort)] [--db PATH]")
                exit(0)
            default: break
            }
            i += 1
        }

        // The store and the server outlive the Task that builds them. Without
        // somewhere to put them the Task's locals die the moment its body ends,
        // leaving a bound socket with nothing behind it — a process that passes
        // every liveness check and answers no request.
        let live = LiveServer()
        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                let store = try await CatalogStore.open(path: dbPath)
                let server = HTTPKit.Server(port: port) { req in
                    await handle(request: req, store: store)
                }
                live.store = store
                live.server = server
                // Throws if the listener never reaches .ready — a port already
                // held by an overlapping launchd restart used to leave this
                // process logging "listening" and then blocking on `sem.wait()`
                // forever, which is a process KeepAlive will never recycle
                // because it never exits.
                try await server.start()
                let bound = server.boundPort ?? port
                FileHandle.standardError.write(Data("gallery listening on 127.0.0.1:\(bound)\n".utf8))
            } catch {
                FileHandle.standardError.write(Data("gallery failed to start: \(error)\n".utf8))
                exit(1)
            }
        }
        sem.wait()   // run forever; launchd owns the lifecycle
        withExtendedLifetime(live) {}
    }

    /// One-shot backfill. Every tree is passed explicitly so no path is ever
    /// implied — in particular, no vault path can be reached by default.
    ///
    /// The studio trees are SMB mounts of the server's filesystem, so each one
    /// is TWO roots (stills and clips) and each root needs its OWN
    /// `remotePrefix`. `BackfillTree.localPath(for:)` strips the prefix and
    /// appends the remainder to `mediaRoot`, so handing both roots the studio
    /// prefix `/home/todd/.kira/studio` would translate
    /// `…/studio/gallery/x.png` into `…/studio/gallery/gallery/x.png` — a path
    /// that exists nowhere. Every i2v edge and every journal row would miss, and
    /// the sweep would report success while covering nothing. Hence the prefix
    /// is subdivided in lockstep with the media root, from a single studio-level
    /// flag so the two cannot drift apart.
    static func runBackfillCLI(args: [String]) {
        var home = NSString(string: "~/Pictures/ComfyBox").expandingTildeInPath
        var kiraRoot: String? = nil
        var breeRoot: String? = nil
        var kiraRemote = "/home/todd/.kira/studio"
        var breeRemote = "/home/todd/.bree/studio"
        var renderJournal: String? = nil
        var kiraHistory: String? = nil
        var breeHistory: String? = nil
        var dbPath: String? = nil
        var refile = false

        var i = 0
        while i < args.count {
            func value() -> String? {
                guard i + 1 < args.count else { return nil }
                i += 1
                return args[i]
            }
            switch args[i] {
            case "--refile": refile = true
            case "--home": if let v = value() { home = v }
            case "--kira-studio": if let v = value() { kiraRoot = v }
            case "--bree-studio": if let v = value() { breeRoot = v }
            case "--kira-remote-prefix": if let v = value() { kiraRemote = v }
            case "--bree-remote-prefix": if let v = value() { breeRemote = v }
            case "--render-journal": if let v = value() { renderJournal = v }
            case "--kira-history": if let v = value() { kiraHistory = v }
            case "--bree-history": if let v = value() { breeHistory = v }
            case "--db": if let v = value() { dbPath = v }
            case "--help", "-h":
                print("""
                    Usage: ComfyBoxGallery backfill [options]
                      --home PATH                 local gallery tree (default ~/Pictures/ComfyBox)
                      --kira-studio PATH          local mount of Kira's studio
                      --bree-studio PATH          local mount of Bree's studio
                      --kira-remote-prefix PATH   studio path as the SERVER spells it
                      --bree-remote-prefix PATH   studio path as the SERVER spells it
                      --render-journal PATH       render-journal.jsonl (Kira)
                      --kira-history PATH         Kira's history.json
                      --bree-history PATH         Bree's history.json
                      --db PATH                   catalog database (default ~/.comfybox/dam.sqlite3)
                      --refile                    re-run derived filing over every existing row
                    """)
                exit(0)
            default:
                // Never ignore an unknown flag. A typo'd `--kira-studo` would
                // otherwise drop an entire archive from the sweep and still exit
                // 0 with a plausible-looking report — the silent-success failure
                // this whole task exists to rule out.
                FileHandle.standardError.write(Data("unknown argument: \(args[i])\n".utf8))
                exit(2)
            }
            i += 1
        }

        /// Stills and clips for one studio. The journal and history are attached
        /// to BOTH roots: each is indexed per tree and keyed by the local path
        /// its own prefix produces, so a journal seen only by the stills tree
        /// would leave every clip's lane unknown.
        func studio(id: String, realm: CatalogRealm, host: String,
                    root: String, remote: String,
                    journal: String?, history: String?) -> [BackfillTree] {
            [("", "gallery"), ("-video", "video")].map { suffix, leaf in
                BackfillTree(id: id + suffix, realm: realm, host: host,
                             mediaRoot: (root as NSString).appendingPathComponent(leaf),
                             metadataRoot: (root as NSString).appendingPathComponent("metadata"),
                             remotePrefix: (remote as NSString).appendingPathComponent(leaf),
                             journalPath: journal, historyPath: history)
            }
        }

        var trees: [BackfillTree] = [
            BackfillTree(id: "home", realm: nil, host: "mac", mediaRoot: home, metadataRoot: nil)
        ]
        if let k = kiraRoot {
            trees += studio(id: "kira", realm: .kira, host: "kira", root: k,
                            remote: kiraRemote, journal: renderJournal, history: kiraHistory)
        }
        if let b = breeRoot {
            trees += studio(id: "bree", realm: .shared, host: "bree", root: b,
                            remote: breeRemote, journal: nil, history: breeHistory)
        }

        // Belt and braces: CatalogBackfill.run refuses these too, but a CLI that
        // can NAME a vault path is one edit away from reading one.
        //
        // `dbPath` is in this list because the read guards do not cover it: the
        // catalog is a WRITE target full of raw prompt text, so
        // `--db ~/Documents/Vaults/…/x.sqlite3` would CREATE a database inside
        // the vault rather than read one out of it.
        for p in trees.flatMap({ [$0.mediaRoot, $0.metadataRoot, $0.journalPath, $0.historyPath] })
                      .compactMap({ $0 }) + [dbPath].compactMap({ $0 })
        where p.contains("Vaults") {
            FileHandle.standardError.write(Data("refusing a vault path: \(p)\n".utf8))
            exit(2)
        }

        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                let store = try await CatalogStore.open(path: dbPath)
                let started = Date()
                let report = try await CatalogBackfill.run(store: store, trees: trees)
                // AFTER the sweep, so rows it just re-realmed are filed under the
                // realm they ended up with rather than the one they started in.
                var refiled: Int? = nil
                if refile { refiled = try await store.refileAll() }
                let elapsed = Date().timeIntervalSince(started)
                // Every counter, including the three that report ABSENCE.
                // `edgesUnresolved` and `assetsUnfiled` are the difference
                // between a sweep that did the work and one that walked an empty
                // or mistranslated tree, and neither shows up in the others.
                print("""
                    scanned:    \(report.filesScanned)
                    indexed:    \(report.assetsIndexed)
                    merged:     \(report.duplicatesMerged)
                    sidecars:   \(report.sidecarsRead)
                    journal:    \(report.journalEntriesRead)
                    edges:      \(report.edgesCreated)
                    unresolved: \(report.edgesUnresolved)
                    unfiled:    \(report.assetsUnfiled)
                    skipped:    \(report.skipped)
                    elapsed:    \(String(format: "%.1fs", elapsed))
                    """)
                if let refiled { print("refiled:    \(refiled) assets now in ≥1 collection") }
            } catch {
                FileHandle.standardError.write(Data("backfill failed: \(error)\n".utf8))
                exit(1)
            }
            sem.signal()
        }
        sem.wait()
    }

    /// Process-lifetime ownership for the pieces the entry point starts.
    /// Written to once, from one Task, before anything can read it.
    private final class LiveServer: @unchecked Sendable {
        var server: HTTPKit.Server?
        var store: CatalogStore?
    }
}
