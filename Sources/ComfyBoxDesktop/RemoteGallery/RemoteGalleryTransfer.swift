// RemoteGalleryTransfer.swift — sending an asset to a remote gallery.
//
// Todd 2026-09-17 chose MOVE: "the remote becomes the only copy." That makes
// the order the whole design (FDD-remote-galleries §3.3):
//
//   1. journal `copying`
//   2. copy the media as `<name>.part`, plus sidecar and thumbnail
//   3. re-read it and compare digests — a mismatch aborts with the local copy
//      untouched
//   4. rename into place, index it, journal `verified`
//   5. relocate the catalog row (never delete it), journal `relocated`
//   6. only now delete the local media, sidecar and thumbnail, and clear the
//      journal
//
// A crash at any point is recoverable from the journal, and never between
// "the local file is gone" and "the catalog knows where it went".

import Foundation
import CryptoKit
import ComfyBoxCatalog

@MainActor
public final class RemoteGalleryTransfer {

    public struct Failure: Sendable, Equatable {
        public let assetID: String
        public let reason: String
    }

    public struct Outcome: Sendable, Equatable {
        public var sent: [String] = []
        public var failed: [Failure] = []
    }

    private let store: DAMStore
    private let catalog: CatalogStore
    private let ingestor: AssetIngestor

    public init(store: DAMStore, catalog: CatalogStore, ingestor: AssetIngestor) {
        self.store = store
        self.catalog = catalog
        self.ingestor = ingestor
    }

    // MARK: - Sending

    /// Move `assets` to `remote`, one at a time. A failure is recorded against
    /// that asset and the rest continue; a failed asset keeps its local copy.
    public func send(assets: [DAMAsset],
                     to remote: RemoteGalleryConfig,
                     progress: ((Int, Int) -> Void)? = nil) async -> Outcome {
        var outcome = Outcome()
        guard remote.kind == .folder, let galleryRoot = remote.galleryRoot else {
            return Outcome(sent: [], failed: assets.map {
                Failure(assetID: $0.id, reason: "This remote kind cannot receive sends yet")
            })
        }

        var index: FolderGalleryIndex
        do {
            index = try FolderGalleryIndex.create(at: galleryRoot, name: remote.name)
        } catch {
            return Outcome(sent: [], failed: assets.map {
                Failure(assetID: $0.id, reason: "Could not open the gallery on \(remote.name): \(error.localizedDescription)")
            })
        }

        for (offset, asset) in assets.enumerated() {
            progress?(offset, assets.count)
            do {
                try await sendOne(asset, to: remote, index: &index)
                outcome.sent.append(asset.id)
            } catch {
                outcome.failed.append(Failure(assetID: asset.id, reason: error.localizedDescription))
            }
        }
        progress?(assets.count, assets.count)
        return outcome
    }

    private func sendOne(_ asset: DAMAsset,
                         to remote: RemoteGalleryConfig,
                         index: inout FolderGalleryIndex) async throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: asset.absolutePath) else {
            throw TransferError.localFileMissing(asset.filename)
        }

        let sentAt = Date()
        let mediaPath = FolderGalleryIndex.mediaRelativePath(filename: asset.filename,
                                                            sentAt: sentAt,
                                                            taken: index.takenMediaPaths)
        let sidecarPath = Self.sidecarPath(forMedia: mediaPath)
        let thumbnailPath = "\(FolderGalleryIndex.thumbnailDirectory)/\(asset.id).jpg"

        let sourceDigest = try Self.sha256(ofFileAt: asset.absolutePath)
        try await catalog.beginTransfer(assetID: asset.id, remoteID: remote.id,
                                        remotePath: mediaPath, sha256: sourceDigest)

        let mediaDestination = index.absolutePath(for: mediaPath)
        let partPath = mediaDestination + ".part"
        let secured = (try? await store.securedAssetIds().contains(asset.id)) ?? false

        do {
            // 2. Copy the bytes and everything that describes them.
            try fm.createDirectory(atPath: (mediaDestination as NSString).deletingLastPathComponent,
                                   withIntermediateDirectories: true)
            if fm.fileExists(atPath: partPath) { try fm.removeItem(atPath: partPath) }
            try fm.copyItem(atPath: asset.absolutePath, toPath: partPath)

            // 3. Verify before anything is deleted.
            let writtenDigest = try Self.sha256(ofFileAt: partPath)
            guard writtenDigest == sourceDigest else { throw TransferError.digestMismatch(asset.filename) }

            let sidecarDestination = index.absolutePath(for: sidecarPath)
            try fm.createDirectory(atPath: (sidecarDestination as NSString).deletingLastPathComponent,
                                   withIntermediateDirectories: true)
            let sidecar = RemoteSidecar(asset: asset, sha256: sourceDigest, sentAt: sentAt, sensitive: secured)
            try sidecar.encoded().write(to: URL(fileURLWithPath: sidecarDestination), options: .atomic)

            var thumbnailRelative: String?
            let localThumbnail = ingestor.thumbnailPath(for: asset.id)
            if fm.fileExists(atPath: localThumbnail) {
                let destination = index.absolutePath(for: thumbnailPath)
                try? fm.createDirectory(atPath: (destination as NSString).deletingLastPathComponent,
                                        withIntermediateDirectories: true)
                if fm.fileExists(atPath: destination) { try? fm.removeItem(atPath: destination) }
                try? fm.copyItem(atPath: localThumbnail, toPath: destination)
                thumbnailRelative = thumbnailPath
            }

            // 4. Into place, indexed.
            if fm.fileExists(atPath: mediaDestination) { try fm.removeItem(atPath: mediaDestination) }
            try fm.moveItem(atPath: partPath, toPath: mediaDestination)
            try index.append(FolderGalleryIndex.Entry(
                assetID: asset.id, mediaPath: mediaPath, sidecarPath: sidecarPath,
                thumbnailPath: thumbnailRelative, kind: asset.kind, filename: asset.filename,
                fileSize: asset.fileSize, sha256: sourceDigest, sentAt: sentAt))
            try await catalog.setTransferState(assetID: asset.id, state: .verified)
        } catch {
            try? fm.removeItem(atPath: partPath)
            try? await catalog.setTransferState(assetID: asset.id, state: .failed)
            throw error
        }

        // 5 and 6 — from here the remote copy is authoritative.
        try await finishMove(assetID: asset.id, localPath: asset.absolutePath,
                             remote: remote, mediaPath: mediaPath, wasSecured: secured)
    }

    /// The tail every path shares: relocate, then delete the local copy.
    /// Re-runnable, which is what makes recovery safe.
    private func finishMove(assetID: String, localPath: String,
                            remote: RemoteGalleryConfig, mediaPath: String,
                            wasSecured: Bool) async throws {
        try await catalog.relocateAsset(id: assetID, host: remote.locationHost, path: mediaPath)
        try await catalog.setTransferState(assetID: assetID, state: .relocated)

        let fm = FileManager.default
        try? fm.removeItem(atPath: localPath)
        let localSidecar = ((localPath as NSString).deletingPathExtension) + ".json"
        try? fm.removeItem(atPath: localSidecar)
        try? fm.removeItem(atPath: ingestor.thumbnailPath(for: assetID))
        ingestor.forgetKnownPath(localPath)

        if wasSecured {
            // It is out of the vault now; leaving the row would keep it hidden
            // on the remote too (Codex review, finding 3).
            _ = try? await store.unsecureAsset(id: assetID)
        }
        try await catalog.finishTransfer(assetID: assetID)
    }

    // MARK: - Recovery

    /// Finish or undo transfers interrupted by a crash or a pulled drive.
    /// Called at launch. Anything whose remote is not configured or not
    /// present is left for next time.
    public func recoverPending(remotes: [RemoteGalleryConfig]) async {
        let pending = (try? await catalog.pendingTransfers()) ?? []
        guard !pending.isEmpty else { return }
        let byID = Dictionary(uniqueKeysWithValues: remotes.map { ($0.id, $0) })

        for transfer in pending {
            guard let remote = byID[transfer.remoteID], let galleryRoot = remote.galleryRoot,
                  FileManager.default.fileExists(atPath: galleryRoot) else { continue }

            switch transfer.state {
            case .copying:
                // The copy never completed: the local file is authoritative.
                try? FileManager.default.removeItem(atPath: (galleryRoot as NSString)
                    .appendingPathComponent(transfer.remotePath) + ".part")
                try? await catalog.finishTransfer(assetID: transfer.assetID)
            case .verified, .relocated:
                // The remote copy exists and was verified: finish the move.
                let asset = try? await store.fetchAsset(id: transfer.assetID)
                let localPath = asset?.absolutePath ?? ""
                let secured = (try? await store.securedAssetIds().contains(transfer.assetID)) ?? false
                try? await finishMove(assetID: transfer.assetID, localPath: localPath,
                                      remote: remote, mediaPath: transfer.remotePath, wasSecured: secured)
            case .failed:
                try? await catalog.finishTransfer(assetID: transfer.assetID)
            }
        }
    }

    // MARK: - Helpers

    static func sidecarPath(forMedia mediaPath: String) -> String {
        let withoutPrefix = mediaPath.hasPrefix(FolderGalleryIndex.mediaDirectory + "/")
            ? String(mediaPath.dropFirst(FolderGalleryIndex.mediaDirectory.count + 1))
            : mediaPath
        return "\(FolderGalleryIndex.sidecarDirectory)/\(withoutPrefix).json"
    }

    /// Streamed, so a 2 GB video does not become 2 GB of memory.
    static func sha256(ofFileAt path: String) throws -> String {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    enum TransferError: LocalizedError {
        case localFileMissing(String)
        case digestMismatch(String)

        var errorDescription: String? {
            switch self {
            case .localFileMissing(let name):
                return "\(name) is not on this Mac any more"
            case .digestMismatch(let name):
                return "\(name) did not arrive intact — the local copy was kept"
            }
        }
    }
}
