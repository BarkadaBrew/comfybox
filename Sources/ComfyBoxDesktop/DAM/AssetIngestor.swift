// AssetIngestor.swift — Watches output directory for new images
//
// Polls the ComfyBox output directory for new image files. When a
// new .png/.jpg appears, reads its JSON sidecar metadata, creates
// a DAMAsset, generates a 256px thumbnail, and inserts into DAMStore.
//
// Uses a simple polling approach (every 5 seconds) rather than
// complex DispatchSource file watching.

import Foundation
import ImageIO
import UniformTypeIdentifiers
import AVFoundation

/// Watches an output directory and ingests new images into DAMStore.
/// MainActor-isolated so its @Observable state is only mutated on the
/// main thread; thumbnail generation runs off-main via a detached task.
@Observable
@MainActor
public final class AssetIngestor {
    // MARK: - Published State

    public var isWatching: Bool = false
    public var lastIngestedFile: String?
    public var ingestedCount: Int = 0
    public var lastError: String?

    // MARK: - Configuration

    public var watchDirectory: String
    public var thumbnailDirectory: String

    // MARK: - Private

    private var pollTask: Task<Void, Never>?
    private var knownPaths: Set<String> = []
    private let store: DAMStore
    private let pollInterval: TimeInterval = 5.0
    private let thumbnailMaxDimension: CGFloat = 256
    private let thumbnailJPEGQuality: CGFloat = 0.8

    public init(store: DAMStore, watchDirectory: String? = nil, thumbnailDirectory: String? = nil) {
        self.store = store
        let comfyboxDir = NSString(string: "~/.comfybox").expandingTildeInPath
        self.watchDirectory = watchDirectory ?? (comfyboxDir as NSString).appendingPathComponent("output")
        self.thumbnailDirectory = thumbnailDirectory ?? (comfyboxDir as NSString).appendingPathComponent("thumbnails")
    }

    // MARK: - Start / Stop

    /// Begin watching the output directory for new images.
    public func startWatching() async {
        guard !isWatching else { return }

        // Ensure directories exist.
        let fm = FileManager.default
        try? fm.createDirectory(atPath: watchDirectory, withIntermediateDirectories: true)
        try? fm.createDirectory(atPath: thumbnailDirectory, withIntermediateDirectories: true)

        // Seed known paths from DAMStore so we don't re-ingest existing assets.
        do {
            let existing = try await store.allAssetPaths()
            knownPaths = Set(existing)
        } catch {
            // If we fail to read existing paths, start fresh — may re-ingest some.
            knownPaths = []
        }

        // Also add any files already in the directory to avoid initial flood.
        if let contents = try? fm.contentsOfDirectory(atPath: watchDirectory) {
            for filename in contents where isImageFile(filename) {
                let path = (watchDirectory as NSString).appendingPathComponent(filename)
                knownPaths.insert(path)
            }
        }

        isWatching = true
        lastError = nil

        pollTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.scanForNewFiles()
                try? await Task.sleep(for: .seconds(self?.pollInterval ?? 5))
            }
        }
    }

    /// Stop watching.
    public func stopWatching() {
        pollTask?.cancel()
        pollTask = nil
        isWatching = false
    }

    /// Manually ingest a single file at the given path. Returns the stored
    /// asset (which keeps its original id if the path was already tracked).
    @discardableResult
    public func ingestFile(at path: String) async throws -> DAMAsset {
        let asset = try buildAsset(from: path)
        let stored = try await store.insertAsset(asset)
        let thumbPath = thumbnailPath(for: stored.id)
        let maxDimension = thumbnailMaxDimension
        let quality = thumbnailJPEGQuality
        await Task.detached(priority: .utility) {
            Self.generateThumbnail(
                from: path, to: thumbPath,
                maxDimension: maxDimension, jpegQuality: quality
            )
        }.value
        knownPaths.insert(path)
        ingestedCount += 1
        lastIngestedFile = stored.filename
        return stored
    }

    /// Regenerate thumbnails for already-known assets whose cached thumbnail
    /// is missing or empty — e.g. left over from a write that was
    /// interrupted before the thumbnail-generation fix below existed.
    /// Cheap to call on every gallery load: most assets already have a
    /// valid thumbnail, so this is just a file-size stat per asset.
    public func regenerateMissingThumbnails(for assets: [DAMAsset]) async {
        let maxDimension = thumbnailMaxDimension
        let quality = thumbnailJPEGQuality
        for asset in assets {
            let thumbPath = thumbnailPath(for: asset.id)
            let attrs = try? FileManager.default.attributesOfItem(atPath: thumbPath)
            let size = (attrs?[.size] as? Int) ?? 0
            if size > 0 { continue }
            let sourcePath = asset.absolutePath
            await Task.detached(priority: .utility) {
                Self.generateThumbnail(
                    from: sourcePath, to: thumbPath,
                    maxDimension: maxDimension, jpegQuality: quality
                )
            }.value
        }
    }

    /// Remove DAM rows whose file was deleted out from under us, dropping
    /// their cached thumbnails too. Returns how many were pruned.
    @discardableResult
    public func pruneOrphans() async throws -> Int {
        let removedIds = try await store.pruneOrphans()
        for id in removedIds {
            try? FileManager.default.removeItem(atPath: thumbnailPath(for: id))
        }
        return removedIds.count
    }

    // MARK: - Folder import

    /// Outcome of importing a folder.
    public struct ImportSummary: Sendable {
        public var total: Int      // images found
        public var imported: Int   // newly ingested
        public var skipped: Int    // already tracked
        public var failed: Int
    }

    /// Add an existing folder to the gallery (Photo Mechanic style): scan it
    /// recursively for images, ingest each one — generating a thumbnail and
    /// merging any sidecar / embedded metadata exactly as a local render — and
    /// report progress. Already-tracked files are skipped, so re-importing is
    /// cheap. Files are read in place; nothing is moved or copied.
    @discardableResult
    public func importFolder(
        at folderPath: String,
        progress: (@MainActor (_ done: Int, _ total: Int) -> Void)? = nil
    ) async throws -> ImportSummary {
        let fm = FileManager.default
        let root = URL(fileURLWithPath: folderPath, isDirectory: true)
        guard let enumerator = fm.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles])
        else {
            throw NSError(domain: "AssetIngestor", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot read folder: \(folderPath)"])
        }

        let paths = enumerator.compactMap { $0 as? URL }
            .filter { isImageFile($0.lastPathComponent) }
            .map(\.path)
            .sorted()

        var summary = ImportSummary(total: paths.count, imported: 0, skipped: 0, failed: 0)
        for (index, path) in paths.enumerated() {
            let alreadyTracked: Bool
            if knownPaths.contains(path) {
                alreadyTracked = true
            } else {
                alreadyTracked = ((try? await store.fetchAsset(byPath: path)) ?? nil) != nil
            }
            if alreadyTracked {
                summary.skipped += 1
            } else {
                do {
                    try await ingestFile(at: path)
                    summary.imported += 1
                } catch {
                    summary.failed += 1
                    lastError = "Failed to import \((path as NSString).lastPathComponent): \(error.localizedDescription)"
                }
            }
            progress?(index + 1, paths.count)
        }
        return summary
    }

    // MARK: - Asset security

    /// Vault for secured assets. The .noindex suffix keeps Spotlight out;
    /// the directory is created owner-only (0700).
    public var secureDirectory: String = {
        let comfybox = NSString(string: "~/.comfybox").expandingTildeInPath
        return (comfybox as NSString).appendingPathComponent("secure.noindex")
    }()

    /// Secure a sensitive asset: move the image and its sidecar into the
    /// vault, destroy the cached thumbnail (a thumbnail of a secured image
    /// defeats the point), and record the original location so unsecuring
    /// can put everything back. Returns the updated asset.
    public func secureAsset(_ asset: DAMAsset) async throws -> DAMAsset {
        let fm = FileManager.default
        try fm.createDirectory(
            atPath: secureDirectory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])

        let destination = (secureDirectory as NSString).appendingPathComponent(asset.filename)
        try fm.moveItem(atPath: asset.absolutePath, toPath: destination)

        let sidecar = ((asset.absolutePath as NSString).deletingPathExtension) + ".json"
        if fm.fileExists(atPath: sidecar) {
            let sidecarDest = ((destination as NSString).deletingPathExtension) + ".json"
            try? fm.moveItem(atPath: sidecar, toPath: sidecarDest)
        }

        try? fm.removeItem(atPath: thumbnailPath(for: asset.id))
        try await store.secureAsset(id: asset.id, securedPath: destination, originalPath: asset.absolutePath)
        // Stop tracking the original path so the poller doesn't treat a
        // future file with the same name as already ingested.
        knownPaths.remove(asset.absolutePath)

        return asset.withLocation(path: destination)
    }

    /// Restore a secured asset to its original location and regenerate its
    /// thumbnail. Returns the updated asset.
    public func unsecureAsset(_ asset: DAMAsset) async throws -> DAMAsset {
        guard let originalPath = try await store.unsecureAsset(id: asset.id) else {
            return asset
        }
        let fm = FileManager.default
        try fm.moveItem(atPath: asset.absolutePath, toPath: originalPath)

        let sidecar = ((asset.absolutePath as NSString).deletingPathExtension) + ".json"
        if fm.fileExists(atPath: sidecar) {
            let sidecarDest = ((originalPath as NSString).deletingPathExtension) + ".json"
            try? fm.moveItem(atPath: sidecar, toPath: sidecarDest)
        }

        let thumbPath = thumbnailPath(for: asset.id)
        let maxDimension = thumbnailMaxDimension
        let quality = thumbnailJPEGQuality
        await Task.detached(priority: .utility) {
            Self.generateThumbnail(
                from: originalPath, to: thumbPath,
                maxDimension: maxDimension, jpegQuality: quality
            )
        }.value
        knownPaths.insert(originalPath)

        return asset.withLocation(path: originalPath)
    }

    /// Delete an asset: moves the image and its JSON sidecar to the Trash
    /// (falling back to permanent removal where trashing is unavailable),
    /// removes the cached thumbnail, and deletes the database row. The path
    /// is un-tracked so a future file with the same name is re-ingested.
    public func deleteAsset(_ asset: DAMAsset) async throws {
        // The media file FIRST, and its failure aborts the whole deletion. The
        // catalog-backed grid now lists rows whose bytes live on the server
        // (roughly 1,300 of 2,994 in the live database, under an smbfs mount at
        // /Volumes/todd). `trashItem` does not work on smbfs, and the old
        // fallback answered that by calling `removeItem` — a permanent,
        // unrecoverable delete of production media, from a dialog whose button
        // says "Move to Trash". Leaving the row in place when the file survives
        // is the only honest outcome.
        try Self.trashOrRemove(atPath: asset.absolutePath)
        let sidecarPath = ((asset.absolutePath as NSString).deletingPathExtension) + ".json"
        try Self.trashOrRemove(atPath: sidecarPath)
        try? FileManager.default.removeItem(atPath: thumbnailPath(for: asset.id))

        try await store.deleteAsset(id: asset.id)
        knownPaths.remove(asset.absolutePath)
    }

    /// Why a deletion stopped without touching the file.
    public enum DeletionRefusal: Error, LocalizedError, Equatable {
        /// The file is not on this Mac and could not be trashed. Escalating to a
        /// permanent removal would destroy another host's copy for good.
        case notOnThisMac(String)

        public var errorDescription: String? {
            switch self {
            case .notOnThisMac(let path):
                return "\(path) is on the server, not this Mac. macOS cannot move it "
                    + "to the Trash, and it will not be deleted permanently. Nothing was removed."
            }
        }
    }

    /// Move a file to the Trash. Missing files are ignored.
    ///
    /// A failed trash may be escalated to a permanent `removeItem` ONLY for a
    /// file on a local volume — that is the sandboxed-test / no-Trash-here case
    /// the fallback was written for. On a network volume a failed trash is the
    /// NORMAL outcome, so the same fallback there is an unattended permanent
    /// delete of a file this Mac does not own. Refuse instead.
    ///
    /// The filesystem operations are injectable so both branches are testable
    /// without an smbfs mount.
    nonisolated static func trashOrRemove(
        atPath path: String,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) },
        trash: (String) throws -> Void = {
            try FileManager.default.trashItem(at: URL(fileURLWithPath: $0), resultingItemURL: nil)
        },
        remove: (String) throws -> Void = { try FileManager.default.removeItem(atPath: $0) },
        isLocal: (String) -> Bool = { AssetIngestor.isOnThisMac($0) }
    ) throws {
        guard fileExists(path) else { return }
        do {
            try trash(path)
        } catch {
            guard isLocal(path) else { throw DeletionRefusal.notOnThisMac(path) }
            try? remove(path)
        }
    }

    /// Whether a path's bytes live on a volume attached to this Mac.
    ///
    /// Fails CLOSED: when the volume cannot be interrogated at all (an
    /// unmounted share, a revoked permission) the answer is "not this Mac", so
    /// the refusal above stands rather than a hard delete proceeding on a guess.
    nonisolated public static func isOnThisMac(_ path: String) -> Bool {
        let url = URL(fileURLWithPath: path)
        if let values = try? url.resourceValues(forKeys: [.volumeIsLocalKey]),
           let local = values.volumeIsLocal {
            return local
        }
        // No answer from the volume. Anything under a mount point other than the
        // root filesystem is assumed to be someone else's disk.
        if let values = try? url.resourceValues(forKeys: [.volumeURLKey]),
           let volume = values.volume {
            return volume.path == "/"
        }
        return !path.hasPrefix("/Volumes/")
    }

    // MARK: - Polling

    private func scanForNewFiles() async {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(atPath: watchDirectory) else {
            return
        }

        let imageFiles = contents.filter { isImageFile($0) }

        for filename in imageFiles {
            let path = (watchDirectory as NSString).appendingPathComponent(filename)

            guard !knownPaths.contains(path) else { continue }

            // Check file is fully written (size stable).
            guard isFileStable(at: path) else { continue }

            do {
                try await ingestFile(at: path)
            } catch {
                lastError = "Failed to ingest \(filename): \(error.localizedDescription)"
            }
        }
    }

    // MARK: - Asset Building

    static let videoExtensions: Set<String> = ["mp4", "mov", "m4v", "webm"]

    private func buildAsset(from path: String) throws -> DAMAsset {
        let fm = FileManager.default
        let filename = (path as NSString).lastPathComponent
        let ext = (path as NSString).pathExtension.lowercased()
        let isVideo = Self.videoExtensions.contains(ext)

        // File attributes.
        let attrs = try fm.attributesOfItem(atPath: path)
        let fileSize = (attrs[.size] as? Int64) ?? 0
        let createdAt = (attrs[.creationDate] as? Date) ?? Date()
        let modifiedAt = (attrs[.modificationDate] as? Date) ?? Date()

        // Dimensions: ImageIO for images, AVAsset for video.
        var width: Int?
        var height: Int?
        if isVideo {
            if let track = AVURLAsset(url: URL(fileURLWithPath: path)).tracks(withMediaType: .video).first {
                let size = track.naturalSize.applying(track.preferredTransform)
                width = Int(abs(size.width)); height = Int(abs(size.height))
            }
        } else if let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) {
            if let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any] {
                width = props[kCGImagePropertyPixelWidth as String] as? Int
                height = props[kCGImagePropertyPixelHeight as String] as? Int
            }
        }

        // Read JSON sidecar metadata if present.
        let sidecar = readSidecar(for: path)

        return DAMAsset(
            kind: isVideo ? "video" : "image",
            filename: filename,
            absolutePath: path,
            fileSize: fileSize,
            width: width,
            height: height,
            createdAt: createdAt,
            modifiedAt: modifiedAt,
            ingestedAt: Date(),
            prompt: sidecar?.prompt,
            negativePrompt: sidecar?.negativePrompt,
            seed: sidecar?.seed,
            steps: sidecar?.steps,
            guidance: sidecar?.guidance,
            modelFamily: sidecar?.modelFamily,
            contentMode: sidecar?.contentMode,
            characterName: sidecar?.characterName,
            source: sidecar?.source
        )
    }

    // MARK: - Sidecar Metadata

    /// LoRAs recorded in an image's metadata — read on demand for display.
    /// Checks the `.json` sidecar first, then the embedded EXIF UserComment JSON.
    /// Returns "name @scale" strings (empty if none were used).
    static func embeddedLoras(imagePath: String) -> [String] {
        func loras(from params: [String: Any]) -> [String] {
            guard let arr = params["loras"] as? [[String: Any]] else { return [] }
            return arr.compactMap { l in
                guard let name = l["name"] as? String else { return nil }
                let scale = (l["scale"] as? Double) ?? (l["scale"] as? Int).map(Double.init) ?? 1.0
                return "\(name) @\(String(format: "%g", scale))"
            }
        }
        let jsonPath = ((imagePath as NSString).deletingPathExtension) + ".json"
        if let data = FileManager.default.contents(atPath: jsonPath),
           let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            let l = loras(from: j); if !l.isEmpty { return l }
        }
        if let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: imagePath) as CFURL, nil),
           let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
           let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any],
           let uc = exif[kCGImagePropertyExifUserComment] as? String,
           let d = uc.data(using: .utf8),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            return loras(from: j)
        }
        return []
    }

    private struct SidecarMetadata {
        let prompt: String?
        let negativePrompt: String?
        let seed: Int?
        let steps: Int?
        let guidance: Double?
        let modelFamily: String?
        let contentMode: String?
        let characterName: String?
        var source: String? = nil
    }

    /// Read generation metadata embedded in the image's standard EXIF/IPTC/TIFF
    /// fields (fast, in-process via ImageIO) — for images that carry it instead
    /// of a .json sidecar. Prompt comes from the params JSON (UserComment) or the
    /// image description; other fields from the params JSON.
    private func readEmbeddedMetadata(imagePath: String) -> SidecarMetadata? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: imagePath) as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any]
        else { return nil }
        let tiff = props[kCGImagePropertyTIFFDictionary] as? [CFString: Any]
        let exif = props[kCGImagePropertyExifDictionary] as? [CFString: Any]
        let iptc = props[kCGImagePropertyIPTCDictionary] as? [CFString: Any]
        let png = props[kCGImagePropertyPNGDictionary] as? [CFString: Any]

        let description = (tiff?[kCGImagePropertyTIFFImageDescription] as? String)
            ?? (iptc?[kCGImagePropertyIPTCCaptionAbstract] as? String)
            ?? (png?[kCGImagePropertyPNGDescription] as? String)

        var params: [String: Any] = [:]
        if let uc = exif?[kCGImagePropertyExifUserComment] as? String,
           let d = uc.data(using: .utf8),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any] {
            params = j
        }
        guard description != nil || !params.isEmpty else { return nil }
        return SidecarMetadata(
            prompt: (params["prompt"] as? String) ?? description,
            negativePrompt: params["negative_prompt"] as? String,
            seed: params["seed"] as? Int,
            steps: params["steps"] as? Int,
            guidance: (params["guidance"] as? Double) ?? (params["guidance"] as? Int).map(Double.init),
            modelFamily: params["model"] as? String,
            contentMode: nil,
            characterName: nil,
            source: params["source"] as? String ?? params["generated_by"] as? String
        )
    }

    private func readSidecar(for imagePath: String) -> SidecarMetadata? {
        // ComfyBox writes {filename}.json next to each output image.
        let basePath = (imagePath as NSString).deletingPathExtension
        let jsonPath = basePath + ".json"

        guard FileManager.default.fileExists(atPath: jsonPath),
              let data = FileManager.default.contents(atPath: jsonPath),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            // No .json sidecar — fall back to metadata embedded in the image itself
            // (our exiftool embed, or mflux/A1111-style params).
            return readEmbeddedMetadata(imagePath: imagePath)
        }

        return SidecarMetadata(
            prompt: json["prompt"] as? String,
            negativePrompt: json["negative_prompt"] as? String
                ?? json["negativePrompt"] as? String,
            seed: json["seed"] as? Int,
            steps: json["steps"] as? Int,
            guidance: json["guidance"] as? Double
                ?? (json["guidance"] as? Int).map(Double.init),
            modelFamily: json["model"] as? String
                ?? json["modelFamily"] as? String
                ?? json["model_family"] as? String,
            contentMode: json["contentMode"] as? String
                ?? json["content_mode"] as? String,
            characterName: json["characterName"] as? String
                ?? json["character_name"] as? String
                ?? json["character"] as? String,
            source: json["source"] as? String
                ?? json["generated_by"] as? String
                ?? json["generatedBy"] as? String
        )
    }

    // MARK: - Thumbnail Generation

    private nonisolated static func generateThumbnail(
        from imagePath: String,
        to thumbPath: String,
        maxDimension: CGFloat,
        jpegQuality: CGFloat
    ) {
        // Skip if a real (non-empty) thumbnail already exists. A 0-byte file
        // means a prior write was interrupted or failed — treat it as missing
        // so generation retries instead of leaving a permanently blank cell.
        if let existingSize = try? FileManager.default.attributesOfItem(atPath: thumbPath)[.size] as? Int,
           existingSize > 0 {
            return
        }

        // Video: grab a representative frame with AVAssetImageGenerator.
        let ext = (imagePath as NSString).pathExtension.lowercased()
        if videoExtensions.contains(ext) {
            generateVideoThumbnail(from: imagePath, to: thumbPath,
                                   maxDimension: maxDimension, jpegQuality: jpegQuality)
            return
        }

        guard let source = CGImageSourceCreateWithURL(
            URL(fileURLWithPath: imagePath) as CFURL, nil
        ) else { return }

        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceThumbnailMaxPixelSize: Int(maxDimension),
            kCGImageSourceCreateThumbnailWithTransform: true,
        ]

        guard let thumbnail = CGImageSourceCreateThumbnailAtIndex(
            source, 0, options as CFDictionary
        ) else { return }

        // Write as JPEG.
        guard let dest = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: thumbPath) as CFURL,
            UTType.jpeg.identifier as CFString,
            1, nil
        ) else { return }

        let destOptions: [CFString: Any] = [
            kCGImageDestinationLossyCompressionQuality: jpegQuality,
        ]
        CGImageDestinationAddImage(dest, thumbnail, destOptions as CFDictionary)
        if !CGImageDestinationFinalize(dest) {
            // Don't leave a corrupt/partial file behind masquerading as a
            // completed thumbnail — the next ingest attempt should retry.
            try? FileManager.default.removeItem(atPath: thumbPath)
        }
    }

    private nonisolated static func generateVideoThumbnail(
        from videoPath: String, to thumbPath: String,
        maxDimension: CGFloat, jpegQuality: CGFloat
    ) {
        let asset = AVURLAsset(url: URL(fileURLWithPath: videoPath))
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.maximumSize = CGSize(width: maxDimension, height: maxDimension)
        // A frame ~1s in (or the first frame for very short clips).
        let dur = asset.duration.seconds
        let time = CMTime(seconds: min(1.0, max(0, dur / 2)), preferredTimescale: 600)
        guard let cg = try? gen.copyCGImage(at: time, actualTime: nil) else { return }
        guard let dest = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: thumbPath) as CFURL, UTType.jpeg.identifier as CFString, 1, nil
        ) else { return }
        CGImageDestinationAddImage(dest, cg, [kCGImageDestinationLossyCompressionQuality: jpegQuality] as CFDictionary)
        if !CGImageDestinationFinalize(dest) {
            try? FileManager.default.removeItem(atPath: thumbPath)
        }
    }

    /// Returns the thumbnail file path for a given asset ID.
    public func thumbnailPath(for assetId: String) -> String {
        (thumbnailDirectory as NSString).appendingPathComponent("\(assetId).jpg")
    }

    // MARK: - Helpers

    private func isImageFile(_ filename: String) -> Bool {
        let ext = (filename as NSString).pathExtension.lowercased()
        return ext == "png" || ext == "jpg" || ext == "jpeg" || Self.videoExtensions.contains(ext)
    }

    private func isFileStable(at path: String) -> Bool {
        // Simple check: file must be at least 1 second old (modification time).
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let modified = attrs[.modificationDate] as? Date
        else {
            return false
        }
        return Date().timeIntervalSince(modified) > 1.0
    }
}
