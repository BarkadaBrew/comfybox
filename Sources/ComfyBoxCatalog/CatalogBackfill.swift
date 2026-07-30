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
// Realm is recovered from which tree holds the copy: an asset with a twin under
// a Kira tree is hers; anything else is shared. It cannot come from the Mac
// path, because every realm renders through the same output directory.
//
// NOTHING here may point at ~/Documents/Vaults/BarkadaAI. The vault is out of
// scope and is never read.

import CryptoKit
import Foundation

public struct BackfillTree: Sendable {
    public let id: String
    /// nil = realm is decided by whether a twin exists elsewhere.
    public let realm: CatalogRealm?
    public let host: String
    public let mediaRoot: String
    public let metadataRoot: String?

    public init(id: String, realm: CatalogRealm?, host: String,
                mediaRoot: String, metadataRoot: String?) {
        self.id = id; self.realm = realm; self.host = host
        self.mediaRoot = mediaRoot; self.metadataRoot = metadataRoot
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
    public var sidecarsRead = 0
    public var skipped = 0

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

    public static func run(store: CatalogStore, trees: [BackfillTree]) async throws -> BackfillReport {
        // Bree's vault is out of scope, enforced here rather than only at the
        // CLI, so no caller can reach it by constructing trees directly. Checked
        // for EVERY tree before a single byte is read, so one bad tree in the
        // list cannot be reached by the sweep getting to it late.
        for t in trees {
            for p in [t.mediaRoot, t.metadataRoot].compactMap({ $0 }) where isVaultPath(p) {
                throw CatalogBackfillError.vaultPathRefused(p)
            }
        }
        var report = BackfillReport()

        // Pass 1 — index, dedup, file.
        // sha256 → asset id, so the second sighting of the same bytes adds a
        // location instead of a row.
        var bySHA: [String: String] = [:]
        // Deferred i2v links: (clip path, source image path).
        var pendingEdges: [(String, String)] = []

        for tree in trees {
            for path in mediaFiles(under: tree.mediaRoot) {
                report.filesScanned += 1
                let ext = (path as NSString).pathExtension.lowercased()
                let kind = videoExtensions.contains(ext) ? "video" : "image"

                guard let data = FileManager.default.contents(atPath: path) else {
                    report.skipped += 1
                    continue
                }
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()

                // Metadata: embedded for images (video containers carry none),
                // sidecar for both — the ONLY source for video.
                var meta = kind == "image" ? (MetadataReader.readEmbedded(path: path) ?? FileMetadata())
                                           : FileMetadata()
                if let mroot = tree.metadataRoot,
                   let sidecar = MetadataReader.sidecarPath(forMedia: path,
                                                            galleryRoot: tree.mediaRoot,
                                                            metadataRoot: mroot),
                   let sdata = FileManager.default.contents(atPath: sidecar),
                   let smeta = MetadataReader.readSidecar(jsonData: sdata) {
                    report.sidecarsRead += 1
                    meta = merge(embedded: meta, sidecar: smeta)
                }
                // The container is the only trustworthy source of duration — real
                // sidecars record `"duration": null` — and the only source at all
                // of fps/frames. It never overrides what the sidecar did say.
                var fps: Double?
                var frames: Int?
                if kind == "video", let info = MetadataReader.probeContainer(path: path) {
                    meta.durationMs = info.durationMs ?? meta.durationMs
                    meta.width = meta.width ?? info.width
                    meta.height = meta.height ?? info.height
                    fps = info.fps
                    frames = info.frames
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
                    // would leave the common case permanently unfiled.
                    if meta != FileMetadata(), let row = existing {
                        let filled = enrich(row, with: meta, fps: fps, frames: frames)
                        if filled != row { try await store.upsert(filled, explicitCollectionIDs: []) }
                    }
                    if kind == "video", let src = meta.sourceImagePath {
                        pendingEdges.append((path, src))
                    }
                    continue
                }

                let asset = CatalogAsset(
                    kind: kind,
                    filename: (path as NSString).lastPathComponent,
                    absolutePath: path,
                    sha256: sha, fileSize: size,
                    width: meta.width, height: meta.height,
                    createdAt: (attrs?[.creationDate] as? Date) ?? mtime,
                    realm: tree.realm ?? .shared,
                    source: sourceLabel(meta),
                    sealed: meta.sealed,
                    prompt: meta.prompt, negativePrompt: meta.negativePrompt,
                    promptRaw: meta.promptRaw,
                    seed: meta.seed, steps: meta.steps, guidance: meta.guidance,
                    modelFamily: meta.modelFamily, preset: meta.preset, loras: meta.loras,
                    contentMode: meta.contentMode, characterName: meta.characterName,
                    lane: meta.lane,
                    mode: meta.mode, durationMs: meta.durationMs, fps: fps, frames: frames,
                    resolution: meta.resolution, aspectRatio: meta.aspectRatio)

                try await store.upsert(asset, explicitCollectionIDs: [])
                try await store.addLocation(assetID: asset.id,
                    AssetLocation(host: tree.host, path: path, mtime: mtime))
                bySHA[sha] = asset.id
                report.assetsIndexed += 1

                if kind == "video", let src = meta.sourceImagePath {
                    pendingEdges.append((path, src))
                }
            }
        }

        // Pass 2 — i2v edges, now that every still is indexed.
        for (clipPath, sourcePath) in pendingEdges {
            guard let clipID = try await store.assetID(forPath: clipPath),
                  let stillID = try await store.assetID(forPath: sourcePath) else { continue }
            try await store.addEdge(AssetEdge(fromAssetID: clipID, toAssetID: stillID,
                                              relation: .i2vSource))
            report.edgesCreated += 1
        }

        return report
    }

    /// Which application produced this. Images recover it from `EXIF:Software`;
    /// a video carries no embedded metadata at all, so for video the sidecar's
    /// `provider` is the only source and the fallback is not optional.
    static func sourceLabel(_ meta: FileMetadata) -> String? {
        (meta.software ?? meta.provider)?.lowercased()
    }

    /// The sidecar is authoritative where it speaks; embedded fills the gaps.
    static func merge(embedded: FileMetadata, sidecar: FileMetadata) -> FileMetadata {
        var m = embedded
        m.prompt = sidecar.prompt ?? m.prompt
        m.promptRaw = sidecar.promptRaw ?? m.promptRaw
        m.negativePrompt = sidecar.negativePrompt ?? m.negativePrompt
        m.seed = sidecar.seed ?? m.seed
        m.steps = sidecar.steps ?? m.steps
        m.guidance = sidecar.guidance ?? m.guidance
        m.width = sidecar.width ?? m.width
        m.height = sidecar.height ?? m.height
        m.modelFamily = sidecar.modelFamily ?? m.modelFamily
        m.preset = sidecar.preset ?? m.preset
        m.loras = sidecar.loras ?? m.loras
        m.characterName = sidecar.characterName ?? m.characterName
        m.contentMode = sidecar.contentMode ?? m.contentMode
        m.lane = sidecar.lane ?? m.lane
        m.mode = sidecar.mode ?? m.mode
        m.resolution = sidecar.resolution ?? m.resolution
        m.aspectRatio = sidecar.aspectRatio ?? m.aspectRatio
        m.sourceImagePath = sidecar.sourceImagePath ?? m.sourceImagePath
        m.provider = sidecar.provider ?? m.provider
        m.sealed = sidecar.sealed || m.sealed
        return m
    }

    /// Fill an already-indexed row's gaps from a copy's metadata. Existing values
    /// always win — this only ever turns nil into a fact — with the one exception
    /// of `sealed`, which can only ever be set: if any copy says the row is
    /// sealed, the row is sealed, and `CatalogAsset.init` drops its text.
    static func enrich(_ existing: CatalogAsset, with meta: FileMetadata,
                       fps: Double?, frames: Int?) -> CatalogAsset {
        CatalogAsset(
            id: existing.id, kind: existing.kind,
            filename: existing.filename, absolutePath: existing.absolutePath,
            sha256: existing.sha256, fileSize: existing.fileSize,
            width: existing.width ?? meta.width, height: existing.height ?? meta.height,
            createdAt: existing.createdAt,
            realm: existing.realm,
            source: existing.source ?? sourceLabel(meta),
            sealed: existing.sealed || meta.sealed,
            prompt: existing.prompt ?? meta.prompt,
            negativePrompt: existing.negativePrompt ?? meta.negativePrompt,
            promptRaw: existing.promptRaw ?? meta.promptRaw,
            caption: existing.caption, captionSource: existing.captionSource,
            seed: existing.seed ?? meta.seed, steps: existing.steps ?? meta.steps,
            guidance: existing.guidance ?? meta.guidance,
            modelFamily: existing.modelFamily ?? meta.modelFamily,
            preset: existing.preset ?? meta.preset,
            loras: existing.loras ?? meta.loras,
            renderID: existing.renderID,
            contentMode: existing.contentMode ?? meta.contentMode,
            characterName: existing.characterName ?? meta.characterName,
            lane: existing.lane ?? meta.lane, arc: existing.arc, theme: existing.theme,
            stock: existing.stock, genre: existing.genre,
            family: existing.family, style: existing.style,
            mode: existing.mode ?? meta.mode,
            durationMs: existing.durationMs ?? meta.durationMs,
            fps: existing.fps ?? fps, frames: existing.frames ?? frames,
            resolution: existing.resolution ?? meta.resolution,
            aspectRatio: existing.aspectRatio ?? meta.aspectRatio,
            rating: existing.rating, favorite: existing.favorite)
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
