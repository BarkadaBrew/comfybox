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
// Within ONE file, metadata is resolved strongest-first:
//   embedded EXIF  →  JSON sidecar  →  journal / history  →  container probe
// and every step only fills what is still nil. The chain is per-file on purpose:
// a weak source applied while sweeping an EARLIER tree would permanently block
// the authoritative sidecar found in a later one, because the merge rule across
// trees is "first non-nil wins, forever".
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
        // Deferred i2v links: (clip path, source image path as the sidecar
        // recorded it — i.e. usually a path on the SERVER).
        var pendingEdges: [(String, String)] = []

        for tree in trees {
            let journal = journalIndex(for: tree)
            report.journalEntriesRead += journal.count

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

                // Strongest first: embedded EXIF (images only — a container
                // carries none), then the sidecar, then the journal, then the
                // probe. Each step only fills what is still nil.
                var meta = kind == "image" ? (MetadataReader.readEmbedded(path: path) ?? FileMetadata())
                                           : FileMetadata()
                if let mroot = tree.metadataRoot,
                   let sidecar = MetadataReader.sidecarPath(forMedia: path,
                                                            galleryRoot: tree.mediaRoot,
                                                            metadataRoot: mroot),
                   let sdata = FileManager.default.contents(atPath: sidecar),
                   let smeta = MetadataReader.readSidecar(jsonData: sdata) {
                    report.sidecarsRead += 1
                    meta = smeta.fillingNils(from: meta)
                }
                if let jmeta = journal[path] {
                    meta = meta.fillingNils(from: jmeta)
                }
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
                    meta = meta.fillingNils(from: probe)
                }

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
                    if meta != FileMetadata(), let old = existing {
                        let filled = row(file: facts(of: old), existing: old, meta: meta)
                        if filled != old { try await store.upsert(filled, explicitCollectionIDs: []) }
                    }
                    if kind == "video", let src = meta.sourceImagePath {
                        pendingEdges.append((path, src))
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
                var reuse: CatalogAsset?
                if let candidate = try await store.assetID(forPath: path),
                   let candidateRow = try await store.asset(id: candidate),
                   candidateRow.absolutePath == path {
                    reuse = candidateRow
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
                let asset = row(file: identity, existing: reuse, meta: meta)

                try await store.upsert(asset, explicitCollectionIDs: [])
                try await store.addLocation(assetID: asset.id,
                    AssetLocation(host: tree.host, path: path, mtime: mtime))
                bySHA[sha] = asset.id
                if reuse == nil { report.assetsIndexed += 1 }

                if kind == "video", let src = meta.sourceImagePath {
                    pendingEdges.append((path, src))
                }
            }
        }

        // Pass 2 — i2v edges, now that every still is indexed.
        for (clipPath, sourcePath) in pendingEdges {
            guard let clipID = try await store.assetID(forPath: clipPath),
                  let stillID = try await resolveSource(sourcePath, trees: trees, store: store),
                  stillID != clipID else {
                report.edgesUnresolved += 1
                continue
            }
            let edge = AssetEdge(fromAssetID: clipID, toAssetID: stillID, relation: .i2vSource)
            // Count edges INSERTED, not attempted: addEdge is INSERT OR IGNORE,
            // so a re-sweep would otherwise report a fresh edge every time.
            let known = try await store.edges(for: clipID).contains(edge)
            guard !known else { continue }
            try await store.addEdge(edge)
            report.edgesCreated += 1
        }

        report.assetsUnfiled = try await store.unfiledAssetCount()
        return report
    }

    /// Resolve a sidecar's `source_image` — recorded on the SERVER — to a local
    /// asset id.
    static func resolveSource(_ remote: String, trees: [BackfillTree],
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
        let base = (remote as NSString).lastPathComponent
        guard !base.isEmpty else { return nil }
        let ids = try await store.assetIDs(forFilename: base)
        return ids.count == 1 ? ids[0] : nil
    }

    /// Journal + history facts for one tree, keyed by LOCAL path.
    ///
    /// These are the lowest-precedence source and, for most of the fleet, the
    /// only one that knows the lane: it appears in 131/400 image sidecars and
    /// 3/200 video ones, so without this roughly two thirds of the catalog files
    /// into no collection at all.
    static func journalIndex(for tree: BackfillTree) -> [String: FileMetadata] {
        var out: [String: FileMetadata] = [:]
        func absorb(_ entries: [MetadataReader.JournalEntry]) {
            for e in entries {
                guard !isVaultPath(e.path) else { continue }
                let local = tree.localPath(for: e.path) ?? e.path
                // Several journal lines can describe one output (a retry, a
                // re-render). The first wins and the rest fill its gaps, which
                // is the same rule the rest of the sweep uses.
                out[local] = out[local].map { $0.fillingNils(from: e.meta) } ?? e.meta
            }
        }
        let fm = FileManager.default
        if let p = tree.journalPath, !isVaultPath(p), let d = fm.contents(atPath: p) {
            absorb(MetadataReader.readRenderJournal(jsonlData: d))
        }
        if let p = tree.historyPath, !isVaultPath(p), let d = fm.contents(atPath: p) {
            absorb(MetadataReader.readHistory(jsonData: d))
        }
        return out
    }

    /// Which application produced this. The SIDECAR is authoritative — `provider`
    /// is a key CollectionRules matches on ("krita", "tile-engine") — while
    /// `EXIF:Software` is a human display string ("CoffeeShop Desktop
    /// (ComfyBox)") that matches no rule. Taking software first made `source` a
    /// display name and unfiled every shared-realm asset, since `source` is the
    /// only filing input a shared asset has.
    static func sourceLabel(_ meta: FileMetadata) -> String? {
        (meta.provider ?? meta.software)?.lowercased()
    }

    /// The ONE place a `CatalogAsset` is built.
    ///
    /// Identity comes from `file`; every fact already on `existing` survives;
    /// `meta` fills only what is still unknown. That ordering is what makes a
    /// re-sweep idempotent and stops a later tree overwriting an earlier one.
    /// `sealed` is the single exception — it ORs, so any source that says a row
    /// is sealed seals it, and `CatalogAsset.init` then drops the text.
    static func row(file: FileFacts, existing: CatalogAsset?, meta: FileMetadata) -> CatalogAsset {
        CatalogAsset(
            id: file.id, kind: file.kind,
            filename: file.filename, absolutePath: file.absolutePath,
            sha256: file.sha256, fileSize: file.fileSize,
            width: existing?.width ?? meta.width, height: existing?.height ?? meta.height,
            createdAt: existing?.createdAt ?? file.createdAt,
            realm: file.realm,
            source: existing?.source ?? sourceLabel(meta),
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
