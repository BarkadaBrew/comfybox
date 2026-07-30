// CatalogBackfill.swift — reconstruct the catalog from what is on disk.
//
// Order matters and is not arbitrary:
//   pass 1  index every media file in every tree, deduping by sha256 so an
//           asset copied to a server tree becomes ONE row with several
//           locations rather than two rows.
//   pass 2  resolve `source_image` from video sidecars into i2v_source edges.
//           This is a second pass because the still may be indexed after the
//           clip that references it.
//
// Within ONE file the sources are collected STRONGEST FIRST —
//   JSON sidecar  →  embedded EXIF  →  render journal  →  history  →  container probe
// — and folded in that order, each filling only what the ones before it left
// unknown. The chain is per-file on purpose: a weak source applied while
// sweeping an EARLIER tree would permanently block the authoritative sidecar
// found in a later one, because the merge rule ACROSS trees is "first non-nil
// wins, forever".
//
// Realm is recovered from which tree holds the copy: an asset with a twin under
// a Kira tree is hers; anything else is shared. It cannot come from the Mac
// path, because every realm renders through the same output directory.
//
// NOTHING here may point at ~/Documents/Vaults/BarkadaAI. The vault is out of
// scope and is never read — checked at the roots AND at every individual path,
// because /Volumes/todd is a real mount of the server home on this machine and
// the vault sits underneath it.

import CryptoKit
import Foundation

public struct BackfillTree: Sendable {
    public let id: String
    /// nil = realm is decided by whether a twin exists elsewhere.
    public let realm: CatalogRealm?
    public let host: String
    public let mediaRoot: String
    public let metadataRoot: String?
    /// What `mediaRoot` is called ON ITS OWN HOST — e.g. `/home/todd/.kira/studio`
    /// for a `mediaRoot` of `/Volumes/todd/.kira/studio`. Sidecars and journals
    /// record server-absolute paths, which exist nowhere on this Mac, so without
    /// this every cross-reference (`source_image`, journal keys) silently misses.
    public let remotePrefix: String?
    /// `render-journal.jsonl` for this tree, if it has one.
    public let journalPath: String?
    /// `history.json` for this tree, if it has one.
    public let historyPath: String?

    public init(id: String, realm: CatalogRealm?, host: String,
                mediaRoot: String, metadataRoot: String?,
                remotePrefix: String? = nil,
                journalPath: String? = nil, historyPath: String? = nil) {
        self.id = id; self.realm = realm; self.host = host
        self.mediaRoot = mediaRoot; self.metadataRoot = metadataRoot
        self.remotePrefix = remotePrefix
        self.journalPath = journalPath; self.historyPath = historyPath
    }

    /// Translate a path recorded on this tree's own host into the local
    /// spelling. nil when the path does not belong to this tree.
    func localPath(for remote: String) -> String? {
        guard let prefix = remotePrefix, remote.hasPrefix(prefix) else { return nil }
        let rel = String(remote.dropFirst(prefix.count))
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard !rel.isEmpty else { return nil }
        return (mediaRoot as NSString).appendingPathComponent(rel)
    }
}

public enum CatalogBackfillError: Error, LocalizedError {
    case vaultPathRefused(String)

    public var errorDescription: String? {
        switch self {
        case let .vaultPathRefused(p): return "refusing to read a vault path: \(p)"
        }
    }
}

public struct BackfillReport: Sendable, Equatable {
    public var filesScanned = 0
    public var assetsIndexed = 0
    public var duplicatesMerged = 0
    public var edgesCreated = 0
    /// `source_image` values that named no asset this catalog knows. Counted
    /// rather than swallowed: a silent `continue` here is how "i2v edges never
    /// resolve in production" would look from the outside — like success.
    public var edgesUnresolved = 0
    public var sidecarsRead = 0
    public var journalEntriesRead = 0
    public var skipped = 0
    /// Assets in NO collection at all, catalog-wide, after the sweep. A bounded
    /// result must be logged, not read as complete coverage.
    public var assetsUnfiled = 0

    public init() {}
}

public enum CatalogBackfill {

    static let imageExtensions: Set<String> = ["png", "jpg", "jpeg", "webp", "tiff", "heic"]
    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v"]

    /// The realm a well-known tree id carries. `nil` means "undecided": the home
    /// gallery holds every realm's output, so a copy there proves nothing and the
    /// realm has to come from a twin in a realm-bearing tree (or default to
    /// shared). Unknown ids are undecided for the same reason.
    public static func realm(forTreeID id: String) -> CatalogRealm? {
        switch id {
        case "kira": return .kira
        case "bree", "studio", "shared": return .shared
        default: return nil
        }
    }

    /// The identity of a file on disk. Everything here is a property of the
    /// BYTES or of where they live — never of the metadata.
    struct FileFacts {
        let id: String
        let kind: String
        let filename: String
        let absolutePath: String
        let sha256: String?
        let fileSize: Int64
        let createdAt: Date
        let realm: CatalogRealm
    }

    public static func run(store: CatalogStore, trees: [BackfillTree]) async throws -> BackfillReport {
        // Bree's vault is out of scope, enforced here rather than only at the
        // CLI, so no caller can reach it by constructing trees directly. Checked
        // for EVERY tree before a single byte is read, so one bad tree in the
        // list cannot be reached by the sweep getting to it late.
        for t in trees {
            for p in [t.mediaRoot, t.metadataRoot, t.journalPath, t.historyPath].compactMap({ $0 })
            where isVaultPath(p) {
                throw CatalogBackfillError.vaultPathRefused(p)
            }
        }
        var report = BackfillReport()

        // Pass 1 — index, dedup, file.
        // sha256 → asset id, so the second sighting of the same bytes adds a
        // location instead of a row.
        var bySHA: [String: String] = [:]
        // Deferred i2v links — resolved once every still is indexed.
        var pendingEdges: [PendingEdge] = []

        for tree in trees {
            let journals = journalIndex(for: tree)
            report.journalEntriesRead += journals.count

            for path in mediaFiles(under: tree.mediaRoot) {
                report.filesScanned += 1
                // The root check above is not enough: a tree rooted at
                // /Volumes/todd is clean by its own name and still CONTAINS
                // Documents/Vaults/BarkadaAI. Refuse per path, before opening it.
                guard !isVaultPath(path) else {
                    report.skipped += 1
                    continue
                }
                let ext = (path as NSString).pathExtension.lowercased()
                let kind = videoExtensions.contains(ext) ? "video" : "image"

                guard let data = FileManager.default.contents(atPath: path), !data.isEmpty else {
                    // A zero-byte file is a truncated render, not an asset — and
                    // every one of them hashes to the SAME sha256, so indexing
                    // them would collapse the lot into a single bogus row with
                    // one location per failure.
                    report.skipped += 1
                    continue
                }
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()

                // The sources, STRONGEST FIRST. They are folded one at a time
                // into the row (see `fold`), each filling only what the ones
                // before it left unknown, so precedence is the array's order and
                // nothing else.
                var sources: [FileMetadata] = []
                if let mroot = tree.metadataRoot {
                    // The metadata tree mirrors the media tree EXCEPT for the
                    // content-tier directory, so try the mirror first and the
                    // tier-flattened spelling second. First readable one wins;
                    // one sidecar is still counted at most once.
                    for sidecar in MetadataReader.sidecarCandidates(
                        forMedia: path, galleryRoot: tree.mediaRoot, metadataRoot: mroot) {
                        guard !isVaultPath(sidecar),
                              let sdata = FileManager.default.contents(atPath: sidecar),
                              let smeta = MetadataReader.readSidecar(jsonData: sdata) else { continue }
                        report.sidecarsRead += 1
                        sources.append(smeta)
                        break
                    }
                }
                // Embedded EXIF — images only, a container carries none.
                if kind == "image", let embedded = MetadataReader.readEmbedded(path: path) {
                    sources.append(embedded)
                }
                // Journal AND history, in that order. They carry DISJOINT
                // fields — the journal knows lane/tier/theme/stock, history
                // knows character/provider/prompt/seed — so letting one stand in
                // for the other drops the filing key of whichever lost.
                sources.append(contentsOf: journals.sources(for: path))
                // The container is the only trustworthy source of duration —
                // real sidecars record `"duration": null` — and the only source
                // at all of fps/frames.
                if kind == "video", let info = MetadataReader.probeContainer(path: path) {
                    var probe = FileMetadata()
                    probe.durationMs = info.durationMs
                    probe.fps = info.fps
                    probe.frames = info.frames
                    probe.width = info.width
                    probe.height = info.height
                    sources.append(probe)
                }
                // `source_image` is the one fact with no column on the asset, so
                // it is picked off the sources directly, same precedence.
                let sourceImage = sources.compactMap(\.sourceImagePath).first

                let attrs = try? FileManager.default.attributesOfItem(atPath: path)
                let mtime = (attrs?[.modificationDate] as? Date) ?? Date()
                let size = (attrs?[.size] as? NSNumber)?.int64Value ?? Int64(data.count)

                // `??` cannot carry the await, so the store is only asked when
                // this run has not already seen the bytes.
                var seenID = bySHA[sha]
                if seenID == nil { seenID = try await store.assetID(forSHA256: sha) }
                if let existingID = seenID {
                    // Same bytes seen before — a downstream copy.
                    report.duplicatesMerged += 1
                    bySHA[sha] = existingID
                    try await store.addLocation(assetID: existingID,
                        AssetLocation(host: tree.host, path: path, mtime: mtime))

                    var existing = try await store.asset(id: existingID)
                    // A twin in a realm-bearing tree settles the realm — but only
                    // ever TOWARD kira. Her renders are mirrored onto shared hosts
                    // as well, so a copy in a shared tree is not evidence of shared
                    // ownership, and letting it demote would make an asset's realm
                    // depend on the order the trees happen to be swept in.
                    if let realm = tree.realm, existing?.realm != .kira, existing?.realm != realm {
                        try await store.setRealm(realm, forAssetID: existingID)
                        existing = try await store.asset(id: existingID)
                    }
                    // Fold in anything this copy knows and the row does not. The
                    // Mac gallery has no sidecars, so for every asset that also
                    // lives in a server tree the sidecar is the ONLY place the
                    // character / lane / tier facets exist — dropping them here
                    // would leave the common case permanently unfiled. Identity
                    // stays the row's own, so a copy never steals the primary path.
                    if let old = existing {
                        let filled = fold(identity: facts(of: old), existing: old, sources: sources)
                        if filled != old { try await store.upsert(filled, explicitCollectionIDs: []) }
                    }
                    if kind == "video", let src = sourceImage {
                        pendingEdges.append(PendingEdge(clipPath: path, sourceImage: src,
                                                        scope: tree.realm))
                    }
                    continue
                }

                // New bytes. If this exact path is ALREADY indexed, its content
                // changed under us (SidecarService embeds with
                // `-overwrite_original`; a re-render reuses the filename), so the
                // row is updated in place. Minting a new id here would collide
                // with `absolute_path NOT NULL UNIQUE` — which `ON CONFLICT(id)`
                // does not absorb — and the throw would abort the whole sweep,
                // leaving a half-written catalog.
                //
                // The lookup MUST be by `absolute_path` alone.
                // `assetID(forPath:)` is a UNION with `asset_locations`, and a
                // compound UNION under LIMIT 1 can return a row that merely has
                // a stale LOCATION here rather than the path's owner — which
                // reintroduces exactly the collision above, non-deterministically.
                var reuse: CatalogAsset?
                if let owner = try await store.assetID(owningPath: path) {
                    reuse = try await store.asset(id: owner)
                }
                let identity = FileFacts(
                    id: reuse?.id ?? UUID().uuidString,
                    kind: kind,
                    filename: (path as NSString).lastPathComponent,
                    absolutePath: path,
                    sha256: sha, fileSize: size,
                    createdAt: (attrs?[.creationDate] as? Date) ?? mtime,
                    // Re-indexing must not demote a realm a twin already settled.
                    realm: reuse?.realm == .kira ? .kira : (tree.realm ?? .shared))
                let asset = fold(identity: identity, existing: reuse, sources: sources)

                try await store.upsert(asset, explicitCollectionIDs: [])
                try await store.addLocation(assetID: asset.id,
                    AssetLocation(host: tree.host, path: path, mtime: mtime))
                bySHA[sha] = asset.id
                if reuse == nil { report.assetsIndexed += 1 }

                if kind == "video", let src = sourceImage {
                    pendingEdges.append(PendingEdge(clipPath: path, sourceImage: src,
                                                    scope: tree.realm))
                }
            }
        }

        // Pass 2 — i2v edges, now that every still is indexed.
        for pending in pendingEdges {
            guard let clipID = try await store.assetID(forPath: pending.clipPath),
                  let stillID = try await resolveSource(pending.sourceImage, scope: pending.scope,
                                                        trees: trees, store: store),
                  stillID != clipID else {
                report.edgesUnresolved += 1
                continue
            }
            let edge = AssetEdge(fromAssetID: clipID, toAssetID: stillID, relation: .i2vSource)
            // Count edges INSERTED, not attempted: addEdge is INSERT OR IGNORE,
            // so a re-sweep would otherwise report a fresh edge every time.
            let known = try await store.edges(for: clipID, scope: nil).contains(edge)
            guard !known else { continue }
            try await store.addEdge(edge)
            report.edgesCreated += 1
        }

        report.assetsUnfiled = try await store.unfiledAssetCount()
        return report
    }

    /// One clip waiting for its still, deferred until every still is indexed.
    struct PendingEdge {
        let clipPath: String
        /// As the sidecar recorded it — usually a path on the SERVER.
        let sourceImage: String
        /// The realm of the tree the CLIP came from, so a guessed match cannot
        /// reach across the boundary.
        let scope: CatalogRealm?
    }

    /// Resolve a sidecar's `source_image` — recorded on the SERVER — to a local
    /// asset id.
    static func resolveSource(_ remote: String, scope: CatalogRealm?,
                              trees: [BackfillTree],
                              store: CatalogStore) async throws -> String? {
        guard !isVaultPath(remote) else { return nil }
        var candidates = [remote]
        for t in trees {
            if let local = t.localPath(for: remote) { candidates.append(local) }
        }
        for candidate in candidates {
            if let id = try await store.assetID(forPath: candidate) { return id }
        }
        // Last resort: the basename, and ONLY when it names exactly one asset.
        // Two assets sharing a filename is normal (every tree has its own
        // `still.png`), and a wrong i2v edge is worse than a missing one.
        //
        // Scoped to the clip's own realm: this is the one lookup here that
        // GUESSES, and an unscoped guess could link her clip to a shared still
        // (or the reverse) purely because two files happen to share a name.
        let base = (remote as NSString).lastPathComponent
        guard !base.isEmpty else { return nil }
        let ids = try await store.assetIDs(forFilename: base, scope: scope)
        return ids.count == 1 ? ids[0] : nil
    }

    /// One tree's journal and history, kept APART and keyed by local path.
    ///
    /// Apart, because the two files know disjoint things: the journal has
    /// lane / tier / theme / stock / style / genre / family, history has
    /// prompt / character / contentMode / seed / provider / renderID. An output
    /// named in both is the normal case, and merging them into one dictionary
    /// meant whichever file was read first silently deleted the other's facts —
    /// including `character`, which is her filing key, and `provider`, which is
    /// the shared realm's. Two entries in the source list costs nothing: that is
    /// exactly what `fold` is for.
    struct JournalSources {
        var journal: [String: FileMetadata] = [:]
        var history: [String: FileMetadata] = [:]

        var count: Int { journal.count + history.count }

        /// Journal ahead of history — the journal is the more specific record of
        /// what was rendered, history the general one.
        func sources(for path: String) -> [FileMetadata] {
            [journal[path], history[path]].compactMap { $0 }
        }
    }

    /// These are the lowest-precedence sources and, for most of the fleet, the
    /// only ones that know the lane: it appears in 131/400 image sidecars and
    /// 3/200 video ones, so without them roughly two thirds of the catalog files
    /// into no collection at all.
    static func journalIndex(for tree: BackfillTree) -> JournalSources {
        var out = JournalSources()
        func absorb(_ entries: [MetadataReader.JournalEntry],
                    into index: inout [String: FileMetadata]) {
            for e in entries {
                guard !isVaultPath(e.path) else { continue }
                let local = tree.localPath(for: e.path) ?? e.path
                // Several lines in ONE file can describe one output (a retry, a
                // re-render). The first wins, the same rule the rest of the
                // sweep uses. (Not "first wins per FIELD": that would need a
                // second copy of the field list, which is the bug class this
                // whole design is trying to close.)
                if index[local] == nil { index[local] = e.meta }
            }
        }
        let fm = FileManager.default
        if let p = tree.journalPath, !isVaultPath(p), let d = fm.contents(atPath: p) {
            absorb(MetadataReader.readRenderJournal(jsonlData: d), into: &out.journal)
        }
        if let p = tree.historyPath, !isVaultPath(p), let d = fm.contents(atPath: p) {
            absorb(MetadataReader.readHistory(jsonData: d), into: &out.history)
        }
        return out
    }

    /// Which application produced this, across every source at once.
    ///
    /// `provider` — from the sidecar or history — is a key CollectionRules
    /// matches on ("krita", "tile-engine"). `EXIF:Software` is a human display
    /// string ("CoffeeShop Desktop (ComfyBox)") that matches no rule. So ANY
    /// source's provider beats ANY source's software: an image with embedded
    /// software, no sidecar and a history record naming krita must still file as
    /// krita, since `source` is the only filing input a shared asset has.
    static func sourceLabel(_ sources: [FileMetadata]) -> String? {
        let provider = sources.compactMap(\.provider).first
        let software = sources.compactMap(\.software).first
        return (provider ?? software)?.lowercased()
    }

    /// Fold every source into one row, strongest first.
    ///
    /// `row` already means "existing wins, meta fills the gaps", so applying the
    /// sources in order gives exactly the precedence their order describes, with
    /// no second merge function and therefore no second field list to forget a
    /// field in. Four struct constructions per file is the price; it is nothing
    /// next to the exiftool call standing beside it.
    static func fold(identity: FileFacts, existing: CatalogAsset?,
                     sources: [FileMetadata]) -> CatalogAsset {
        // `source` is resolved across ALL sources FIRST, not folded per source.
        // provider and software never appear in the same source — only
        // readEmbedded sets software, only the sidecar and history set provider
        // — so folding it would hand the column to whichever source came first
        // regardless of which field it carried, and an EXIF display string in
        // `source` unfiles every shared-realm asset.
        let label = sourceLabel(sources)
        var acc = row(file: identity, existing: existing,
                      meta: sources.first ?? FileMetadata(), source: label)
        for source in sources.dropFirst() {
            acc = row(file: identity, existing: acc, meta: source, source: label)
        }
        return acc
    }

    /// The ONE place a `CatalogAsset` is built.
    ///
    /// Identity comes from `file`; every fact already on `existing` survives;
    /// `meta` fills only what is still unknown. That ordering is what makes a
    /// re-sweep idempotent and stops a later tree overwriting an earlier one.
    /// `sealed` is the single exception — it ORs, so any source that says a row
    /// is sealed seals it, and `CatalogAsset.init` then drops the text.
    static func row(file: FileFacts, existing: CatalogAsset?, meta: FileMetadata,
                    source: String?) -> CatalogAsset {
        CatalogAsset(
            id: file.id, kind: file.kind,
            filename: file.filename, absolutePath: file.absolutePath,
            sha256: file.sha256, fileSize: file.fileSize,
            width: existing?.width ?? meta.width, height: existing?.height ?? meta.height,
            createdAt: existing?.createdAt ?? file.createdAt,
            realm: file.realm,
            source: existing?.source ?? source,
            sealed: (existing?.sealed ?? false) || meta.sealed,
            prompt: existing?.prompt ?? meta.prompt,
            negativePrompt: existing?.negativePrompt ?? meta.negativePrompt,
            promptRaw: existing?.promptRaw ?? meta.promptRaw,
            promptInjected: existing?.promptInjected ?? meta.promptInjected,
            caption: existing?.caption, captionSource: existing?.captionSource,
            seed: existing?.seed ?? meta.seed,
            steps: existing?.steps ?? meta.steps,
            guidance: existing?.guidance ?? meta.guidance,
            modelFamily: existing?.modelFamily ?? meta.modelFamily,
            preset: existing?.preset ?? meta.preset,
            loras: existing?.loras ?? meta.loras,
            renderID: existing?.renderID ?? meta.renderID,
            contentMode: existing?.contentMode ?? meta.contentMode,
            characterName: existing?.characterName ?? meta.characterName,
            lane: existing?.lane ?? meta.lane,
            arc: existing?.arc ?? meta.arc,
            theme: existing?.theme ?? meta.theme,
            stock: existing?.stock ?? meta.stock,
            genre: existing?.genre ?? meta.genre,
            family: existing?.family ?? meta.family,
            style: existing?.style ?? meta.style,
            mode: existing?.mode ?? meta.mode,
            durationMs: existing?.durationMs ?? meta.durationMs,
            fps: existing?.fps ?? meta.fps,
            frames: existing?.frames ?? meta.frames,
            resolution: existing?.resolution ?? meta.resolution,
            aspectRatio: existing?.aspectRatio ?? meta.aspectRatio,
            rating: existing?.rating ?? 0, favorite: existing?.favorite ?? false)
    }

    /// A row's own identity, for re-writing it without moving it.
    static func facts(of a: CatalogAsset) -> FileFacts {
        FileFacts(id: a.id, kind: a.kind, filename: a.filename, absolutePath: a.absolutePath,
                  sha256: a.sha256, fileSize: a.fileSize, createdAt: a.createdAt, realm: a.realm)
    }

    /// A path inside an Obsidian vault, in any of the forms a caller might hand
    /// us: as written, tilde-expanded, or with symlinks resolved.
    static func isVaultPath(_ path: String) -> Bool {
        let expanded = (path as NSString).expandingTildeInPath
        let forms = [path, expanded, (expanded as NSString).resolvingSymlinksInPath]
        return forms.contains { $0.contains("Vaults") }
    }

    /// Every image and video under `root`, named the way the CALLER spelled the
    /// root.
    ///
    /// The URL enumerator is deliberately not used: it hands back
    /// symlink-resolved paths (`/private/var/...` for a `/var/...` root,
    /// `/private/tmp` for `/tmp`), which is a different spelling from the tree
    /// the caller configured. That silently breaks the sidecar lookup — it is a
    /// `media.hasPrefix(galleryRoot)` test — and leaves every stored path in a
    /// namespace no other tool uses, so `assetID(forPath:)` and `source_image`
    /// resolution miss too. The path enumerator yields ROOT-RELATIVE paths, so
    /// re-rooting them keeps the caller's spelling exactly.
    ///
    /// It also does NOT descend into symlinked directories, which is what stops
    /// a symlink inside a swept tree from walking into the vault. Do not
    /// "optimise" this back to `enumerator(at:)`: that would reopen both holes.
    static func mediaFiles(under root: String) -> [String] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: root), let en = fm.enumerator(atPath: root) else { return [] }
        var out: [String] = []
        for case let rel as String in en {
            let ext = (rel as NSString).pathExtension.lowercased()
            guard imageExtensions.contains(ext) || videoExtensions.contains(ext) else { continue }
            // enumerator(atPath:) has no skipsHiddenFiles, so dotfiles and
            // anything inside a dot-directory are dropped by hand.
            guard !rel.split(separator: "/").contains(where: { $0.hasPrefix(".") }) else { continue }
            out.append((root as NSString).appendingPathComponent(rel))
        }
        return out.sorted()
    }
}
