// GalleryArchiver.swift — .cbarchive archive engine (archive operation)
//
// Moves DAM assets out of the live gallery into a self-contained, browsable
// bundle on disk: copy → verify → commit → delete source. Nothing is
// destroyed before the commit point (INCOMPLETE.json removal at step 8 of
// §5.1 of the FDD task brief), so a crash or thrown error at any earlier
// point leaves the live gallery completely untouched.
//
// Restore (`restore`, `RestoreRequest`, `resumePendingRemovals`) is a later
// task — this file implements `archive` only. The `Phase` enum already
// carries `.restoring` so that work can be added without reshaping this type.

import Foundation

// MARK: - Errors

public enum GalleryArchiverError: Error, LocalizedError, Equatable {
    /// A second `archive`/`restore` call was made while one was already running.
    case alreadyRunning
    /// `entries.jsonl` could not be created for writing.
    case cannotCreateEntriesFile(String)

    public var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "An archive operation is already in progress."
        case .cannotCreateEntriesFile(let path):
            return "Could not create \(path) for writing."
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
