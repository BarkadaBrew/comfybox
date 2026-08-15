// GalleryMaintenance.swift — Gallery maintenance backend
//
// Orphan-thumbnail scan/purge: a thumbnail is orphaned when its cached
// `.jpg` file (named `{assetId}.jpg`) no longer has a matching row in
// DAMStore — e.g. the asset's file was deleted and pruneOrphans() removed
// the row but a stray thumbnail write raced it, or an older bug left one
// behind. This mirrors pruneOrphans()'s shape, inverted: pruneOrphans
// finds rows without files; scanOrphanThumbnails finds thumbnail files
// without rows.

import Foundation

/// Result of scanning the thumbnail cache for files with no matching
/// DAMStore row.
public struct ThumbnailOrphanReport: Sendable {
    public var scanned: Int
    public var orphanPaths: [String]
    public var reclaimableBytes: Int64
    public var orphanCount: Int { orphanPaths.count }
}

@MainActor
public struct GalleryMaintenance {
    private let store: DAMStore
    private let ingestor: AssetIngestor

    public init(store: DAMStore, ingestor: AssetIngestor) {
        self.store = store
        self.ingestor = ingestor
    }

    /// Scan the thumbnail directory (non-recursive) for `.jpg` files whose
    /// derived asset id has no matching row in DAMStore.
    public func scanOrphanThumbnails() async throws -> ThumbnailOrphanReport {
        let ids = try await store.allAssetIds()
        let fm = FileManager.default
        let dir = ingestor.thumbnailDirectory

        let entries = (try? fm.contentsOfDirectory(atPath: dir)) ?? []

        var scanned = 0
        var orphanPaths: [String] = []
        var reclaimableBytes: Int64 = 0

        for name in entries {
            guard (name as NSString).pathExtension.lowercased() == "jpg" else { continue }
            let path = (dir as NSString).appendingPathComponent(name)

            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: path, isDirectory: &isDirectory), !isDirectory.boolValue else {
                continue
            }

            scanned += 1
            let assetId = (name as NSString).deletingPathExtension
            guard !ids.contains(assetId) else { continue }

            orphanPaths.append(path)
            let attrs = try? fm.attributesOfItem(atPath: path)
            let size = (attrs?[.size] as? Int64) ?? 0
            reclaimableBytes += size
        }

        return ThumbnailOrphanReport(
            scanned: scanned,
            orphanPaths: orphanPaths,
            reclaimableBytes: reclaimableBytes
        )
    }

    /// Delete exactly the paths in `report` — no re-derivation, so the user
    /// deletes what the confirmation dialog told them about. Re-checks each
    /// path is still inside `thumbnailDirectory` and still ends `.jpg`
    /// before unlinking, and uses `removeItem` (not Trash): a regenerable
    /// ~30KB cache file does not deserve Trash ceremony, and thousands of
    /// them would swamp it.
    @discardableResult
    public func purgeOrphanThumbnails(_ report: ThumbnailOrphanReport) async -> (deleted: Int, bytesFreed: Int64) {
        let fm = FileManager.default
        let dir = URL(fileURLWithPath: ingestor.thumbnailDirectory, isDirectory: true)
            .standardizedFileURL.path

        var deleted = 0
        var bytesFreed: Int64 = 0

        for path in report.orphanPaths {
            guard (path as NSString).pathExtension.lowercased() == "jpg" else { continue }
            let standardizedPath = URL(fileURLWithPath: path).standardizedFileURL.path
            guard standardizedPath.hasPrefix(dir + "/") else { continue }

            let attrs = try? fm.attributesOfItem(atPath: standardizedPath)
            let size = (attrs?[.size] as? Int64) ?? 0

            guard (try? fm.removeItem(atPath: standardizedPath)) != nil else { continue }
            deleted += 1
            bytesFreed += size
        }

        return (deleted, bytesFreed)
    }
}
