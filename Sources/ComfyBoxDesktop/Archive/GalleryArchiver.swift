// GalleryArchiver.swift — .cbarchive archive engine (archive operation)
//
// Moves DAM assets out of the live gallery into a self-contained, browsable
// bundle on disk: copy → verify → commit → delete source. Nothing is
// destroyed before the commit point (INCOMPLETE.json removal at step 8 of
// §5.1 of the FDD task brief), so a crash or thrown error at any earlier
// point leaves the live gallery completely untouched.
//
// Restore (`restore`, `RestoreRequest`) reverses the process: stream
// `entries.jsonl`, resolve the four id-collision cases against a live
// snapshot, copy files back atomically, and re-file folders. It is
// idempotent — re-running a restore reports everything as `skipped`.
// `resumePendingRemovals` (crash-recovery for an interrupted archive) is a
// later task (T7) and is not implemented here.

import Foundation

// MARK: - Errors

public enum GalleryArchiverError: Error, LocalizedError, Equatable {
    /// A second `archive`/`restore` call was made while one was already running.
    case alreadyRunning
    /// `entries.jsonl` could not be created for writing.
    case cannotCreateEntriesFile(String)
    /// A restore entry's file could not be copied back to the destination.
    case restoreCopyFailed(String)

    public var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "An archive operation is already in progress."
        case .cannotCreateEntriesFile(let path):
            return "Could not create \(path) for writing."
        case .restoreCopyFailed(let path):
            return "Could not restore \(path)."
        }
    }
}

// MARK: - GalleryArchiver

@Observable
@MainActor
public final class GalleryArchiver {
    public enum Phase: Sendable {
        case idle, copying, writingManifest, removingSources, restoring
    }

    public private(set) var phase: Phase = .idle
    public private(set) var progress: (done: Int, total: Int)?
    public var lastError: String?
    public var isRunning: Bool { phase != .idle }

    private let store: DAMStore
    private let ingestor: AssetIngestor

    public init(store: DAMStore, ingestor: AssetIngestor) {
        self.store = store
        self.ingestor = ingestor
    }

    // MARK: - Archive

    public struct ArchiveRequest: Sendable {
        public var name: String
        public var destinationRoot: String     // e.g. ~/.comfybox/archives
        public var assets: [DAMAsset]
        public var folder: DAMFolder?          // set for "Archive Folder"; deleted on success

        public init(name: String, destinationRoot: String, assets: [DAMAsset], folder: DAMFolder? = nil) {
            self.name = name
            self.destinationRoot = destinationRoot
            self.assets = assets
            self.folder = folder
        }
    }

    public struct ArchiveResult: Sendable {
        public var bundlePath: String
        public var archived: Int
        public var skippedSecured: Int
        public var failed: [String]            // filenames
        public var totalBytes: Int64

        public init(bundlePath: String, archived: Int, skippedSecured: Int, failed: [String], totalBytes: Int64) {
            self.bundlePath = bundlePath
            self.archived = archived
            self.skippedSecured = skippedSecured
            self.failed = failed
            self.totalBytes = totalBytes
        }
    }

    /// Executes the FDD §5.1 archive sequence: secured-asset subtraction,
    /// one-shot folder-assignment snapshot, INCOMPLETE.json marker, batched
    /// copy/sidecar/thumbnail/verify/JSONL-append, atomic manifest write,
    /// PENDING_REMOVAL.json, INCOMPLETE removal as the commit point, source
    /// removal via `ingestor.deleteAsset` (Trash semantics),
    /// PENDING_REMOVAL removal, conditional folder deletion, result.
    @discardableResult
    public func archive(
        _ request: ArchiveRequest,
        progress: (@MainActor (_ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> ArchiveResult {
        guard !isRunning else {
            throw GalleryArchiverError.alreadyRunning
        }
        lastError = nil
        phase = .copying
        self.progress = nil
        defer {
            phase = .idle
            self.progress = nil
        }

        do {
            return try await performArchive(request, progress: progress)
        } catch {
            lastError = error.localizedDescription
            throw error
        }
    }

    private func performArchive(
        _ request: ArchiveRequest,
        progress: (@MainActor (_ done: Int, _ total: Int) -> Void)?
    ) async throws -> ArchiveResult {
        // Step 1: subtract secured assets.
        let securedIds = try await store.securedAssetIds()
        let workingAssets = request.assets.filter { !securedIds.contains($0.id) }
        let skippedSecured = request.assets.count - workingAssets.count

        // Step 2: one-shot snapshots (not one query per asset).
        let assignments = try await store.folderAssignments()
        let allFolders = try await store.listFolders()

        // Step 3: bundle layout.
        let bundleName = Self.bundleDirectoryName(for: request.name, at: Date())
        let bundleRoot = (request.destinationRoot as NSString).appendingPathComponent(bundleName)
        let assetsDir = (bundleRoot as NSString).appendingPathComponent("assets")
        try FileManager.default.createDirectory(atPath: assetsDir, withIntermediateDirectories: true)

        let incompletePath = (bundleRoot as NSString).appendingPathComponent("INCOMPLETE.json")
        let manifestPath = (bundleRoot as NSString).appendingPathComponent("manifest.json")
        let entriesPath = (bundleRoot as NSString).appendingPathComponent("entries.jsonl")
        let pendingRemovalPath = (bundleRoot as NSString).appendingPathComponent("PENDING_REMOVAL.json")

        // Step 4: INCOMPLETE.json marker — its absence later is the commit marker.
        let incompleteMarker = IncompleteMarker(
            startedAt: Date().timeIntervalSince1970,
            name: request.name,
            assetCount: workingAssets.count
        )
        try JSONEncoder().encode(incompleteMarker)
            .write(to: URL(fileURLWithPath: incompletePath), options: .atomic)

        // Step 5: batched copy / sidecar / thumbnail / verify / JSONL append.
        guard FileManager.default.createFile(atPath: entriesPath, contents: nil) else {
            throw GalleryArchiverError.cannotCreateEntriesFile(entriesPath)
        }
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: entriesPath))

        var committed: [(asset: DAMAsset, entry: ArchivedAsset, bytes: Int64)] = []
        var failedFilenames: [String] = []
        let total = workingAssets.count
        var done = 0

        do {
            var batchStart = 0
            while batchStart < workingAssets.count {
                let batchEnd = min(batchStart + 50, workingAssets.count)
                for asset in workingAssets[batchStart..<batchEnd] {
                    let assetDir = (assetsDir as NSString).appendingPathComponent(asset.id)
                    let thumbnailSourcePath = ingestor.thumbnailPath(for: asset.id)

                    let outcome = await Task.detached(priority: .utility) {
                        Self.copyAssetFiles(
                            asset: asset,
                            assetDir: assetDir,
                            thumbnailSourcePath: thumbnailSourcePath
                        )
                    }.value

                    if outcome.success {
                        var entry = ArchivedAsset(
                            from: asset,
                            folderId: assignments[asset.id],
                            relativeRoot: "assets/\(asset.id)"
                        )
                        if !outcome.sidecarCopied { entry.sidecarRelativePath = nil }
                        if !outcome.thumbnailCopied { entry.thumbnailRelativePath = nil }
                        try handle.write(contentsOf: ArchiveJSONL.encodeLine(entry))
                        committed.append((asset, entry, outcome.byteSize))
                    } else {
                        failedFilenames.append(asset.filename)
                    }

                    done += 1
                    progress?(done, total)
                    self.progress = (done, total)
                }
                try? handle.synchronize()
                batchStart = batchEnd
            }
        } catch {
            try? handle.close()
            throw error
        }

        // Step 6: close handle, write manifest atomically.
        phase = .writingManifest
        try? handle.close()

        let referencedFolderIds = Set(committed.compactMap { $0.entry.folderId })
        let manifestFolders = allFolders
            .filter { referencedFolderIds.contains($0.id) }
            .map { ArchivedFolder(id: $0.id, name: $0.name, createdAt: $0.createdAt.timeIntervalSince1970) }
        let totalBytes = committed.reduce(Int64(0)) { $0 + $1.bytes }

        let manifest = ArchiveManifest(
            archiveId: UUID().uuidString,
            name: request.name,
            createdAt: Date().timeIntervalSince1970,
            producer: "ComfyBox Desktop",
            assetCount: committed.count,
            totalBytes: totalBytes,
            sourceOutputDirectory: ingestor.watchDirectory,
            folders: manifestFolders
        )
        try JSONEncoder().encode(manifest)
            .write(to: URL(fileURLWithPath: manifestPath), options: .atomic)

        // Step 7: PENDING_REMOVAL.json.
        let committedIds = committed.map { $0.asset.id }
        let pendingRemoval = PendingRemovalMarker(assetIds: committedIds)
        try JSONEncoder().encode(pendingRemoval)
            .write(to: URL(fileURLWithPath: pendingRemovalPath), options: .atomic)

        // Step 8 — COMMIT POINT. From here the bundle is valid and self-sufficient.
        try FileManager.default.removeItem(atPath: incompletePath)

        // Step 9: remove sources via the existing Trash-semantics path.
        phase = .removingSources
        for (asset, _, _) in committed {
            try await ingestor.deleteAsset(asset)
        }

        // Step 10: PENDING_REMOVAL.json is no longer needed — removal finished cleanly.
        try? FileManager.default.removeItem(atPath: pendingRemovalPath)

        // Step 11: conditional folder deletion — only if every asset that was
        // filed in the folder (per the step-2 snapshot) was archived.
        if let folder = request.folder {
            let folderAssetIds = Set(assignments.filter { $0.value == folder.id }.keys)
            let committedIdSet = Set(committedIds)
            if folderAssetIds.isSubset(of: committedIdSet) {
                try await store.deleteFolder(id: folder.id)
            }
        }

        // Step 12: result.
        return ArchiveResult(
            bundlePath: bundleRoot,
            archived: committed.count,
            skippedSecured: skippedSecured,
            failed: failedFilenames,
            totalBytes: totalBytes
        )
    }

    // MARK: - Restore

    public struct RestoreRequest: Sendable {
        public var bundlePath: String
        public var assetIds: Set<String>?          // nil = restore all
        public var restoreToOriginalLocations: Bool
        public var overwriteExistingMetadata: Bool

        public init(
            bundlePath: String,
            assetIds: Set<String>? = nil,
            restoreToOriginalLocations: Bool = false,
            overwriteExistingMetadata: Bool = false
        ) {
            self.bundlePath = bundlePath
            self.assetIds = assetIds
            self.restoreToOriginalLocations = restoreToOriginalLocations
            self.overwriteExistingMetadata = overwriteExistingMetadata
        }
    }

    public struct RestoreResult: Sendable {
        public var restored: Int
        public var skipped: Int            // already present, untouched
        public var reIdentified: Int       // id was taken by a different asset
        public var renamed: Int            // filename collided at the destination
        public var failed: [String]

        public init(restored: Int = 0, skipped: Int = 0, reIdentified: Int = 0, renamed: Int = 0, failed: [String] = []) {
            self.restored = restored
            self.skipped = skipped
            self.reIdentified = reIdentified
            self.renamed = renamed
            self.failed = failed
        }
    }

    /// Executes the FDD §5.2 restore sequence: version-checked manifest read,
    /// streamed (id-filtered) `entries.jsonl` read, per-entry id-collision
    /// resolution against a live snapshot, atomic copy-back, thumbnail
    /// restore/regenerate, and — after the asset loop — folder re-creation
    /// with one batched `assignAssets` per folder.
    @discardableResult
    public func restore(
        _ request: RestoreRequest,
        progress: (@MainActor (_ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> RestoreResult {
        guard !isRunning else {
            throw GalleryArchiverError.alreadyRunning
        }
        lastError = nil
        phase = .restoring
        self.progress = nil
        defer {
            phase = .idle
            self.progress = nil
        }

        do {
            return try await performRestore(request, progress: progress)
        } catch {
            lastError = error.localizedDescription
            throw error
        }
    }

    private func performRestore(
        _ request: RestoreRequest,
        progress: (@MainActor (_ done: Int, _ total: Int) -> Void)?
    ) async throws -> RestoreResult {
        // Version policy is enforced before any work happens.
        let manifestPath = (request.bundlePath as NSString).appendingPathComponent("manifest.json")
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try ArchiveManifest.decode(manifestData)

        let bundleRoot = URL(fileURLWithPath: request.bundlePath)
        let entriesPath = (request.bundlePath as NSString).appendingPathComponent("entries.jsonl")

        // Stream entries.jsonl, materializing only the selected subset — a
        // partial restore never holds the other entries in memory.
        var selected: [ArchivedAsset] = []
        _ = try ArchiveJSONL.read(at: entriesPath) { entry in
            if let ids = request.assetIds, !ids.contains(entry.id) { return }
            selected.append(entry)
        }

        let total = selected.count
        var done = 0
        var result = RestoreResult()

        // One-shot live snapshot (keyed by id) — never re-queried per entry.
        let liveCount = try await store.assetCount()
        let liveAssets = try await store.fetchAssets(limit: liveCount, offset: 0)
        let liveById = Dictionary(uniqueKeysWithValues: liveAssets.map { ($0.id, $0) })

        let watchDirectory = ingestor.watchDirectory
        var folderMembers: [String: [String]] = [:]   // archived folderId -> resolved asset ids

        var batchStart = 0
        while batchStart < selected.count {
            let batchEnd = min(batchStart + 25, selected.count)
            for entry in selected[batchStart..<batchEnd] {
                do {
                    if let outcome = try await restoreOne(
                        entry: entry,
                        bundleRoot: bundleRoot,
                        liveById: liveById,
                        watchDirectory: watchDirectory,
                        request: request
                    ) {
                        switch outcome.kind {
                        case .restored: result.restored += 1
                        case .reIdentified: result.reIdentified += 1
                        }
                        if outcome.renamed { result.renamed += 1 }
                        if let folderId = entry.folderId {
                            folderMembers[folderId, default: []].append(outcome.resolvedId)
                        }
                    } else {
                        result.skipped += 1
                    }
                } catch {
                    result.failed.append(entry.filename)
                }
                done += 1
                progress?(done, total)
                self.progress = (done, total)
            }
            batchStart = batchEnd
        }

        // Folder re-creation, after the asset loop, one batched assignAssets
        // call per distinct archived folder.
        if !folderMembers.isEmpty {
            var liveFolders = try await store.listFolders()
            for archivedFolder in manifest.folders {
                guard let ids = folderMembers[archivedFolder.id], !ids.isEmpty else { continue }
                let resolvedFolderId = try await resolveFolderId(archived: archivedFolder, liveFolders: &liveFolders)
                try await store.assignAssets(ids: ids, toFolder: resolvedFolderId)
            }
        }

        return result
    }

    private struct RestoreOutcome {
        enum Kind { case restored, reIdentified }
        var kind: Kind
        var resolvedId: String
        var renamed: Bool
    }

    /// Resolves the id-collision table from §5.2 for a single entry and, for
    /// every outcome except the "skip entirely" case, performs the copy /
    /// insert / thumbnail sequence. Returns `nil` for the skip case (no copy,
    /// no DB write).
    private func restoreOne(
        entry: ArchivedAsset,
        bundleRoot: URL,
        liveById: [String: DAMAsset],
        watchDirectory: String,
        request: RestoreRequest
    ) async throws -> RestoreOutcome? {
        let destDir = Self.destinationDirectory(
            for: entry, watchDirectory: watchDirectory, useOriginal: request.restoreToOriginalLocations
        )
        let destPathCandidate = (destDir as NSString).appendingPathComponent(entry.filename)

        if let liveAsset = liveById[entry.id] {
            let sameLocation = liveAsset.absolutePath == destPathCandidate
            if sameLocation && FileManager.default.fileExists(atPath: liveAsset.absolutePath) {
                // Same id, same location, and the file is genuinely still
                // there — this is the live asset itself.
                guard request.overwriteExistingMetadata else {
                    return nil
                }
                let stored = try await store.insertAsset(
                    entry.toDAMAsset(absolutePath: liveAsset.absolutePath, id: entry.id)
                )
                await handleThumbnail(entry: entry, stored: stored, bundleRoot: bundleRoot)
                return RestoreOutcome(kind: .restored, resolvedId: stored.id, renamed: false)
            } else if sameLocation {
                // Same id, same recorded path, but the file is gone — an
                // orphaned row (pruneOrphans() exists precisely because this
                // happens). The archived id is a dead reference to the same
                // asset and is legitimately reclaimable: restore it for
                // real rather than reporting a false "skipped", regardless
                // of overwriteExistingMetadata.
                return try await performCopyAndInsert(
                    entry: entry, bundleRoot: bundleRoot, destDir: destDir, id: entry.id, kind: .restored
                )
            } else {
                // The id is taken by an unrelated asset — mint a fresh id so
                // both rows survive.
                return try await performCopyAndInsert(
                    entry: entry, bundleRoot: bundleRoot, destDir: destDir, id: UUID().uuidString, kind: .reIdentified
                )
            }
        } else {
            return try await performCopyAndInsert(
                entry: entry, bundleRoot: bundleRoot, destDir: destDir, id: entry.id, kind: .restored
            )
        }
    }

    /// Resolves the destination filename collision, copies the entry's file
    /// (unless it's already there under a different name and content-equal),
    /// inserts the DAM row under `id`, and restores/regenerates the
    /// thumbnail. Shared by the free, orphaned-same-path, and re-ID cases —
    /// they differ only in which id to insert under and how to count it.
    private func performCopyAndInsert(
        entry: ArchivedAsset, bundleRoot: URL, destDir: String, id: String, kind: RestoreOutcome.Kind
    ) async throws -> RestoreOutcome {
        let (finalPath, skipCopy, renamed) = try await resolveDestinationPath(destDir: destDir, entry: entry)
        if !skipCopy {
            try await copyEntry(entry: entry, bundleRoot: bundleRoot, destPath: finalPath)
        }
        let stored = try await store.insertAsset(
            entry.toDAMAsset(absolutePath: finalPath, id: id)
        )
        await handleThumbnail(entry: entry, stored: stored, bundleRoot: bundleRoot)
        return RestoreOutcome(kind: kind, resolvedId: stored.id, renamed: renamed)
    }

    /// `restoreToOriginalLocations` targets the archived original path's
    /// directory when it still exists and is writable, falling back to the
    /// watch directory per-asset otherwise (§5.2 destination policy).
    private static func destinationDirectory(for entry: ArchivedAsset, watchDirectory: String, useOriginal: Bool) -> String {
        guard useOriginal, let original = entry.originalPath else { return watchDirectory }
        let expanded = (original as NSString).expandingTildeInPath
        let dir = (expanded as NSString).deletingLastPathComponent
        var isDir: ObjCBool = false
        let fm = FileManager.default
        guard fm.fileExists(atPath: dir, isDirectory: &isDir), isDir.boolValue, fm.isWritableFile(atPath: dir) else {
            return watchDirectory
        }
        return dir
    }

    /// Resolves a filename collision at the destination: same size (and,
    /// when both are known, same sha256) is treated as the same file and the
    /// copy is skipped in favor of the existing path; otherwise `" (restored)"`,
    /// `" (restored 2)"`, … is appended until a free name is found.
    private func resolveDestinationPath(
        destDir: String, entry: ArchivedAsset
    ) async throws -> (path: String, skipCopy: Bool, renamed: Bool) {
        let fm = FileManager.default
        let candidatePath = (destDir as NSString).appendingPathComponent(entry.filename)
        guard fm.fileExists(atPath: candidatePath) else {
            return (candidatePath, false, false)
        }

        if let attrs = try? fm.attributesOfItem(atPath: candidatePath),
           let existingSize = attrs[.size] as? Int64,
           existingSize == entry.fileSize {
            var treatAsSameFile = true
            if let entrySha = entry.sha256,
               let existingAsset = try? await store.fetchAsset(byPath: candidatePath),
               let existingSha = existingAsset.sha256 {
                treatAsSameFile = existingSha == entrySha
            }
            if treatAsSameFile {
                return (candidatePath, true, false)
            }
        }

        let base = (entry.filename as NSString).deletingPathExtension
        let ext = (entry.filename as NSString).pathExtension
        var suffixIndex = 1
        var renamedPath: String
        repeat {
            let label = suffixIndex == 1 ? " (restored)" : " (restored \(suffixIndex))"
            let candidateName = ext.isEmpty ? "\(base)\(label)" : "\(base)\(label).\(ext)"
            renamedPath = (destDir as NSString).appendingPathComponent(candidateName)
            suffixIndex += 1
        } while fm.fileExists(atPath: renamedPath)
        return (renamedPath, false, true)
    }

    /// Copies the entry's source file (and sidecar, if present) from the
    /// bundle to `destPath`, atomically: `<destPath>.partial` then a rename.
    /// A stale `.partial` left by a crashed prior restore is deleted on
    /// sight before the new copy starts.
    private func copyEntry(entry: ArchivedAsset, bundleRoot: URL, destPath: String) async throws {
        let sourceURL = try ArchivePaths.resolveEntryPath(entry.relativePath, in: bundleRoot)
        var sidecarPath: String?
        if let sidecarRel = entry.sidecarRelativePath {
            let sidecarURL = try ArchivePaths.resolveEntryPath(sidecarRel, in: bundleRoot)
            if FileManager.default.fileExists(atPath: sidecarURL.path) {
                sidecarPath = sidecarURL.path
            }
        }
        let sourcePath = sourceURL.path
        let success = await Task.detached(priority: .utility) {
            Self.copyEntryFiles(sourceFilePath: sourcePath, sourceSidecarPath: sidecarPath, destPath: destPath)
        }.value
        guard success else {
            throw GalleryArchiverError.restoreCopyFailed(destPath)
        }
    }

    private nonisolated static func copyEntryFiles(
        sourceFilePath: String, sourceSidecarPath: String?, destPath: String
    ) -> Bool {
        let fm = FileManager.default
        let partialPath = destPath + ".partial"
        if fm.fileExists(atPath: partialPath) {
            try? fm.removeItem(atPath: partialPath)
        }
        do {
            try fm.copyItem(atPath: sourceFilePath, toPath: partialPath)
            try fm.moveItem(atPath: partialPath, toPath: destPath)
        } catch {
            try? fm.removeItem(atPath: partialPath)
            return false
        }
        if let sidecarSrc = sourceSidecarPath {
            let destSidecar = ((destPath as NSString).deletingPathExtension) + ".json"
            try? fm.copyItem(atPath: sidecarSrc, toPath: destSidecar)
        }
        return true
    }

    /// §5.2 step 5: copy the bundle's cached thumbnail into place when one
    /// exists and is non-empty; otherwise fall back to the existing backfill
    /// path. Always keyed off `stored.id` — never the id read from the
    /// manifest — per §0.3(a) (`insertAsset` may hand back a different id).
    private func handleThumbnail(entry: ArchivedAsset, stored: DAMAsset, bundleRoot: URL) async {
        if let thumbRel = entry.thumbnailRelativePath,
           let thumbURL = try? ArchivePaths.resolveEntryPath(thumbRel, in: bundleRoot),
           let attrs = try? FileManager.default.attributesOfItem(atPath: thumbURL.path),
           let size = attrs[.size] as? Int64, size > 0 {
            let destThumbPath = ingestor.thumbnailPath(for: stored.id)
            // generateThumbnail refuses to overwrite a non-empty file
            // (§0.3(b)); we copy directly, so clear any stale thumbnail first.
            try? FileManager.default.removeItem(atPath: destThumbPath)
            try? FileManager.default.copyItem(atPath: thumbURL.path, toPath: destThumbPath)
        } else {
            await ingestor.regenerateMissingThumbnails(for: [stored])
        }
    }

    /// Folder re-creation order (§5.2): same id → reuse; same name
    /// case-insensitively → reuse; else create with the archived id; if that
    /// id is unexpectedly taken by a different-named folder, fall back to a
    /// fresh id.
    private func resolveFolderId(archived: ArchivedFolder, liveFolders: inout [DAMFolder]) async throws -> String {
        if let match = liveFolders.first(where: { $0.id == archived.id }) {
            return match.id
        }
        if let match = liveFolders.first(where: { $0.name.caseInsensitiveCompare(archived.name) == .orderedSame }) {
            return match.id
        }
        do {
            let created = try await store.createFolder(name: archived.name, id: archived.id)
            liveFolders.append(created)
            return created.id
        } catch {
            let created = try await store.createFolder(name: archived.name)
            liveFolders.append(created)
            return created.id
        }
    }

    // MARK: - Off-main per-asset file work

    private struct CopyOutcome: Sendable {
        var success: Bool
        var sidecarCopied: Bool
        var thumbnailCopied: Bool
        var byteSize: Int64
    }

    /// Copies the source file, its sidecar (if any), and its cached
    /// thumbnail (if any and non-empty) into `assetDir`, then verifies the
    /// destination's on-disk size matches the source's on-disk size (falling
    /// back to the DB-recorded `fileSize` only when the source can no longer
    /// be stat'd). Runs off-main via `Task.detached`.
    private nonisolated static func copyAssetFiles(
        asset: DAMAsset,
        assetDir: String,
        thumbnailSourcePath: String
    ) -> CopyOutcome {
        let fm = FileManager.default
        let failure = CopyOutcome(success: false, sidecarCopied: false, thumbnailCopied: false, byteSize: 0)

        do {
            try fm.createDirectory(atPath: assetDir, withIntermediateDirectories: true)
            let destPath = (assetDir as NSString).appendingPathComponent(asset.filename)
            try fm.copyItem(atPath: asset.absolutePath, toPath: destPath)

            var sidecarCopied = false
            let sourceSidecarPath = ((asset.absolutePath as NSString).deletingPathExtension) + ".json"
            if fm.fileExists(atPath: sourceSidecarPath) {
                let baseName = (asset.filename as NSString).deletingPathExtension
                let destSidecarPath = (assetDir as NSString).appendingPathComponent("\(baseName).json")
                try fm.copyItem(atPath: sourceSidecarPath, toPath: destSidecarPath)
                sidecarCopied = true
            }

            var thumbnailCopied = false
            if let thumbAttrs = try? fm.attributesOfItem(atPath: thumbnailSourcePath),
               let thumbSize = thumbAttrs[.size] as? Int64, thumbSize > 0 {
                let destThumbPath = (assetDir as NSString).appendingPathComponent("thumb.jpg")
                try fm.copyItem(atPath: thumbnailSourcePath, toPath: destThumbPath)
                thumbnailCopied = true
            }

            let expectedSize: Int64
            if let sourceAttrs = try? fm.attributesOfItem(atPath: asset.absolutePath),
               let sourceSize = sourceAttrs[.size] as? Int64 {
                expectedSize = sourceSize
            } else {
                expectedSize = asset.fileSize
            }

            guard let destAttrs = try? fm.attributesOfItem(atPath: destPath),
                  let destSize = destAttrs[.size] as? Int64,
                  destSize == expectedSize
            else {
                return failure
            }

            return CopyOutcome(success: true, sidecarCopied: sidecarCopied, thumbnailCopied: thumbnailCopied, byteSize: destSize)
        } catch {
            return failure
        }
    }

    // MARK: - Bundle naming (§3.1)

    /// `<slug>-<yyyyMMdd-HHmmss>.cbarchive`. `slug` collapses any run of
    /// characters outside `[A-Za-z0-9-_]` to a single `-`, trims leading and
    /// trailing `-`, truncates to 60 characters, and defaults to `Archive`
    /// when the result is empty.
    static func bundleDirectoryName(for name: String, at date: Date) -> String {
        "\(slug(for: name))-\(timestampSuffix(for: date)).cbarchive"
    }

    private static func slug(for name: String) -> String {
        let collapsed = name.replacingOccurrences(
            of: "[^A-Za-z0-9_-]+", with: "-", options: .regularExpression
        )
        let trimmed = collapsed.trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        let truncated = String(trimmed.prefix(60))
        return truncated.isEmpty ? "Archive" : truncated
    }

    private static let timestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = .current
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return formatter
    }()

    private static func timestampSuffix(for date: Date) -> String {
        timestampFormatter.string(from: date)
    }
}

// MARK: - On-disk marker files

private struct IncompleteMarker: Codable {
    var startedAt: Double
    var name: String
    var assetCount: Int
}

private struct PendingRemovalMarker: Codable {
    var assetIds: [String]
}
