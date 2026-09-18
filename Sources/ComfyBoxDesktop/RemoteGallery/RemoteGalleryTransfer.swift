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
        /// For an Immich remote: the album the send used, so the caller can
        /// cache it back into the config and skip the lookup next time.
        public var albumID: String?
    }

    private let store: DAMStore
    private let catalog: CatalogStore
    private let ingestor: AssetIngestor
    /// How an Immich remote's client is built. Injectable so tests can talk to
    /// a stub instead of a server.
    private let immichClient: (RemoteGalleryConfig) -> ImmichClient?

    public init(store: DAMStore, catalog: CatalogStore, ingestor: AssetIngestor,
                immichClient: ((RemoteGalleryConfig) -> ImmichClient?)? = nil) {
        self.store = store
        self.catalog = catalog
        self.ingestor = ingestor
        self.immichClient = immichClient ?? { remote in
            guard let base = remote.normalizedBaseURL, let key = Keychain.get(remote.keychainAccount) else { return nil }
            return ImmichClient(baseURL: base, apiKey: key)
        }
    }

    // MARK: - Sending

    /// Move `assets` to `remote`, one at a time. A failure is recorded against
    /// that asset and the rest continue; a failed asset keeps its local copy.
    public func send(assets: [DAMAsset],
                     to remote: RemoteGalleryConfig,
                     progress: ((Int, Int) -> Void)? = nil) async -> Outcome {
        var outcome = Outcome()
        if remote.kind == .immich {
            return await sendToImmich(assets: assets, remote: remote, progress: progress)
        }
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

        // Claim the path for the length of the move so the ingestor's poller
        // cannot re-ingest it mid-transfer.
        ingestor.reservePath(asset.absolutePath)
        defer { ingestor.releasePath(asset.absolutePath) }

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

    // MARK: - Immich

    /// Upload, confirm it is really there, then move the catalog row and
    /// delete the local copy — the same order as a folder send
    /// (FDD-remote-galleries §3.4).
    private func sendToImmich(assets: [DAMAsset],
                              remote: RemoteGalleryConfig,
                              progress: ((Int, Int) -> Void)?) async -> Outcome {
        var outcome = Outcome()
        guard let client = immichClient(remote) else {
            return Outcome(sent: [], failed: assets.map {
                Failure(assetID: $0.id, reason: ImmichError.notConfigured.localizedDescription)
            })
        }
        do {
            try await client.assertSupportedVersion()
        } catch {
            return Outcome(sent: [], failed: assets.map {
                Failure(assetID: $0.id, reason: error.localizedDescription)
            })
        }

        var albumID: String?
        if let albumName = remote.albumName, !albumName.isEmpty {
            albumID = try? await client.ensureAlbum(named: albumName, cachedID: remote.albumID)
        }
        outcome.albumID = albumID

        for (offset, asset) in assets.enumerated() {
            progress?(offset, assets.count)
            do {
                try await sendOneToImmich(asset, remote: remote, client: client, albumID: albumID)
                outcome.sent.append(asset.id)
            } catch {
                outcome.failed.append(Failure(assetID: asset.id, reason: error.localizedDescription))
            }
        }
        progress?(assets.count, assets.count)
        return outcome
    }

    private func sendOneToImmich(_ asset: DAMAsset,
                                 remote: RemoteGalleryConfig,
                                 client: ImmichClient,
                                 albumID: String?) async throws {
        let fm = FileManager.default
        guard fm.fileExists(atPath: asset.absolutePath) else {
            throw TransferError.localFileMissing(asset.filename)
        }
        ingestor.reservePath(asset.absolutePath)
        defer { ingestor.releasePath(asset.absolutePath) }

        let sentAt = Date()
        let secured = (try? await store.securedAssetIds().contains(asset.id)) ?? false
        let digest = try Self.sha256(ofFileAt: asset.absolutePath)
        try await catalog.beginTransfer(assetID: asset.id, remoteID: remote.id,
                                        remotePath: "immich://pending", sha256: digest)

        let uploaded: ImmichAsset
        do {
            uploaded = try await client.upload(fileAt: asset.absolutePath, assetID: asset.id,
                                               createdAt: asset.createdAt, modifiedAt: asset.modifiedAt)
            // Confirm by CHECKSUM, not merely by id: the local copy is about
            // to be deleted (Codex review of the implementation).
            let sha1 = try ImmichClient.sha1Base64(ofFileAt: asset.absolutePath)
            guard uploaded.isPresent, try await client.assetMatches(id: uploaded.id, sha1Base64: sha1) else {
                throw TransferError.notConfirmedOnServer(asset.filename)
            }
            if let albumID { try? await client.addToAlbum(albumID: albumID, assetIDs: [uploaded.id]) }
            // The recipe travels in the description (Immich takes no JSON
            // sidecar). Best effort: a send is not failed over it.
            if let sidecar = try? RemoteSidecar(asset: asset, sha256: digest, sentAt: sentAt, sensitive: secured).encoded(),
               let text = String(data: sidecar, encoding: .utf8) {
                try? await client.setDescription(assetID: uploaded.id, text: text)
            }
        } catch {
            try? await catalog.setTransferState(assetID: asset.id, state: .failed)
            throw error
        }

        let remotePath = "immich://\(uploaded.id)"
        try await catalog.setTransferState(assetID: asset.id, state: .verified, remotePath: remotePath)
        try await finishMove(assetID: asset.id, localPath: asset.absolutePath,
                             remote: remote, mediaPath: remotePath, wasSecured: secured)
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
            guard let remote = byID[transfer.remoteID] else { continue }
            let galleryRoot = remote.galleryRoot ?? ""
            if remote.kind == .folder, !FileManager.default.fileExists(atPath: galleryRoot) { continue }

            switch transfer.state {
            case .copying:
                // The copy never completed: the local file is authoritative.
                if !transfer.remotePath.hasPrefix("immich://") {
                    try? FileManager.default.removeItem(atPath: (galleryRoot as NSString)
                        .appendingPathComponent(transfer.remotePath) + ".part")
                }
                try? await catalog.finishTransfer(assetID: transfer.assetID)
            case .verified, .relocated:
                // A journal row is not proof forever: the drive may have been
                // written to, or the Immich asset removed, since (Codex review
                // of the implementation). Re-check the remote copy BEFORE the
                // local one is deleted.
                guard await remoteCopyStillGood(transfer, remote: remote, galleryRoot: galleryRoot) else {
                    // The remote copy is gone or changed: keep the local file
                    // and drop the journal row. The asset is local again.
                    if let asset = try? await store.fetchAsset(id: transfer.assetID),
                       FileManager.default.fileExists(atPath: asset.absolutePath) {
                        try? await catalog.relocateAsset(id: transfer.assetID, host: "mac", path: asset.absolutePath)
                    }
                    try? await catalog.finishTransfer(assetID: transfer.assetID)
                    continue
                }
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

    /// Is the remote copy this journal row claims still there, and still the
    /// same bytes? Checked on recovery, before any local delete.
    private func remoteCopyStillGood(_ transfer: CatalogStore.PendingTransfer,
                                     remote: RemoteGalleryConfig,
                                     galleryRoot: String) async -> Bool {
        if transfer.remotePath.hasPrefix("immich://") {
            let immichID = String(transfer.remotePath.dropFirst("immich://".count))
            guard !immichID.isEmpty, immichID != "pending", let client = immichClient(remote) else { return false }
            guard let asset = try? await store.fetchAsset(id: transfer.assetID),
                  FileManager.default.fileExists(atPath: asset.absolutePath) else {
                // Nothing local to protect; trust the id alone.
                return (try? await client.assetExists(id: immichID)) ?? false
            }
            guard let sha1 = try? ImmichClient.sha1Base64(ofFileAt: asset.absolutePath) else { return false }
            return (try? await client.assetMatches(id: immichID, sha1Base64: sha1)) ?? false
        }

        let path = (galleryRoot as NSString).appendingPathComponent(transfer.remotePath)
        guard FileManager.default.fileExists(atPath: path) else { return false }
        guard let expected = transfer.sha256 else { return true }
        guard let actual = try? Self.sha256(ofFileAt: path) else { return false }
        return actual == expected
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
        case notConfirmedOnServer(String)

        var errorDescription: String? {
            switch self {
            case .localFileMissing(let name):
                return "\(name) is not on this Mac any more"
            case .digestMismatch(let name):
                return "\(name) did not arrive intact — the local copy was kept"
            case .notConfirmedOnServer(let name):
                return "\(name) could not be confirmed on the server — the local copy was kept"
            }
        }
    }
}
