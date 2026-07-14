// GalleryArchiver.swift — moves the ENTIRE local gallery's files into a new
// named remote gallery (see GalleryStore.swift server-side) and purges the
// local DAM index. A bulk "declutter" action: the media itself isn't
// deleted, just relocated out of the local Gallery into a gallery that's
// browsable (and later compressible/deletable) via Remote Galleries.
//
// Only works when the connected server is on this same Mac: the destination
// directory is computed from the server's own ~/.comfybox/galleries/<id>
// convention (GalleryStore.directory(for:) server-side) and files are moved
// there with a plain FileManager call, which only resolves correctly when
// Desktop and the server share a home directory.

import Foundation

public enum GalleryArchiverError: LocalizedError {
    case remoteServerNotSupported
    case nothingToArchive

    public var errorDescription: String? {
        switch self {
        case .remoteServerNotSupported:
            return "Archiving only works when connected to the local ComfyBox server — files can't be moved onto a remote machine's disk from here."
        case .nothingToArchive:
            return "The gallery is already empty."
        }
    }
}

public enum GalleryArchiver {
    /// Move every asset's file + JSON sidecar into a freshly created named
    /// gallery, then purge their DAM rows. Thumbnails are deleted, not
    /// moved — they're cheap to regenerate and Remote Galleries doesn't use
    /// them (it streams full files). Returns the number of assets archived.
    @discardableResult
    public static func archiveAll(
        store: DAMStore,
        ingestor: AssetIngestor,
        hub: GalleryHubService,
        name: String
    ) async throws -> Int {
        guard await hub.isLocalServer else {
            throw GalleryArchiverError.remoteServerNotSupported
        }
        let assets = try await store.fetchAssets(limit: 100_000)
        guard !assets.isEmpty else { throw GalleryArchiverError.nothingToArchive }

        let entry = try await hub.create(name: name, hidden: true, password: nil)
        let galleryDir = (("~/.comfybox/galleries/" + entry.id) as NSString).expandingTildeInPath
        let fm = FileManager.default
        try fm.createDirectory(atPath: galleryDir, withIntermediateDirectories: true)

        var movedIds: [String] = []
        for asset in assets {
            let destPath = Self.uniqueDestination((galleryDir as NSString).appendingPathComponent(asset.filename), fm: fm)
            if fm.fileExists(atPath: asset.absolutePath) {
                try? fm.moveItem(atPath: asset.absolutePath, toPath: destPath)
            }
            let sidecarSrc = ((asset.absolutePath as NSString).deletingPathExtension) + ".json"
            if fm.fileExists(atPath: sidecarSrc) {
                let sidecarDest = ((destPath as NSString).deletingPathExtension) + ".json"
                try? fm.moveItem(atPath: sidecarSrc, toPath: sidecarDest)
            }
            try? fm.removeItem(atPath: await ingestor.thumbnailPath(for: asset.id))
            movedIds.append(asset.id)
        }

        try await store.deleteAssets(ids: movedIds)
        return movedIds.count
    }

    /// Appends "-2", "-3", … before the extension if `path` already exists —
    /// two assets can't collide in the fresh archive directory just because
    /// they happened to share a filename.
    private static func uniqueDestination(_ path: String, fm: FileManager) -> String {
        guard fm.fileExists(atPath: path) else { return path }
        let ns = path as NSString
        let ext = ns.pathExtension
        let base = ns.deletingPathExtension
        var n = 2
        var candidate = ext.isEmpty ? "\(base)-\(n)" : "\(base)-\(n).\(ext)"
        while fm.fileExists(atPath: candidate) {
            n += 1
            candidate = ext.isEmpty ? "\(base)-\(n)" : "\(base)-\(n).\(ext)"
        }
        return candidate
    }
}
