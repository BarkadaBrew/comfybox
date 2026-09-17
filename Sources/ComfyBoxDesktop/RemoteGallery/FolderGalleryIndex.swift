// FolderGalleryIndex.swift — the gallery that lives on the drive.
//
// Todd 2026-09-17: "a gallery that I can send images and videos to but not as
// an archive." An archive (.cbarchive) is a bundle that must be restored to be
// useful. This is the opposite: plain media under their own names, a JSON
// sidecar beside each one, and an index at the root, so the drive is readable
// on any machine — with or without ComfyBox.
//
//   <galleryRoot>/
//     gallery.json
//     media/YYYY/MM/<filename>
//     sidecars/YYYY/MM/<filename>.json
//     thumbnails/<assetID>.jpg
//
// The index is written atomically (temp + replace), so a drive pulled
// mid-write leaves the previous index intact (FDD-remote-galleries §3.2).

import Foundation

public struct FolderGalleryIndex: Codable, Sendable {

    public struct Entry: Codable, Sendable, Equatable {
        public let assetID: String
        /// Relative to the gallery root, e.g. `media/2026/09/kira-01.png`.
        public var mediaPath: String
        public var sidecarPath: String
        public var thumbnailPath: String?
        public var kind: String
        public var filename: String
        public var fileSize: Int64
        public var sha256: String?
        public var sentAt: Date

        public init(assetID: String, mediaPath: String, sidecarPath: String, thumbnailPath: String?,
                    kind: String, filename: String, fileSize: Int64, sha256: String?, sentAt: Date) {
            self.assetID = assetID
            self.mediaPath = mediaPath
            self.sidecarPath = sidecarPath
            self.thumbnailPath = thumbnailPath
            self.kind = kind
            self.filename = filename
            self.fileSize = fileSize
            self.sha256 = sha256
            self.sentAt = sentAt
        }
    }

    public static let schemaVersion = 1
    public static let indexFilename = "gallery.json"
    public static let mediaDirectory = "media"
    public static let sidecarDirectory = "sidecars"
    public static let thumbnailDirectory = "thumbnails"

    public var schemaVersion: Int
    public let id: String
    public var name: String
    public var createdAt: Date
    public private(set) var entries: [Entry]

    /// Not written to disk; where this gallery was loaded from.
    private var galleryRoot: String = ""

    private enum CodingKeys: String, CodingKey { case schemaVersion, id, name, createdAt, entries }

    // MARK: - Paths

    public static func indexPath(inGalleryRoot root: String) -> String {
        (root as NSString).appendingPathComponent(indexFilename)
    }

    public static func exists(atGalleryRoot root: String) -> Bool {
        FileManager.default.fileExists(atPath: indexPath(inGalleryRoot: root))
    }

    /// `media/YYYY/MM/<filename>`, with a numeric suffix when `taken` already
    /// holds that path. Case-insensitive, because the drive may or may not be.
    public static func mediaRelativePath(filename: String, sentAt: Date, taken: Set<String>) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = .current
        let parts = calendar.dateComponents([.year, .month], from: sentAt)
        let folder = String(format: "%@/%04d/%02d", mediaDirectory, parts.year ?? 1970, parts.month ?? 1)
        let base = (filename as NSString).deletingPathExtension
        let ext = (filename as NSString).pathExtension
        let lowered = Set(taken.map { $0.lowercased() })

        func candidate(_ index: Int) -> String {
            let name = index == 0 ? base : "\(base)-\(index)"
            return ext.isEmpty ? "\(folder)/\(name)" : "\(folder)/\(name).\(ext)"
        }
        var index = 0
        while lowered.contains(candidate(index).lowercased()) { index += 1 }
        return candidate(index)
    }

    // MARK: - Lifecycle

    /// Open the gallery at `root`, creating its layout and index when absent.
    /// An existing gallery is opened as it stands: `name` is only used for a
    /// gallery being created, so re-creating never wipes one.
    @discardableResult
    public static func create(at root: String, name: String) throws -> FolderGalleryIndex {
        if exists(atGalleryRoot: root) { return try load(fromGalleryRoot: root) }
        let fm = FileManager.default
        for sub in [mediaDirectory, sidecarDirectory, thumbnailDirectory] {
            try fm.createDirectory(atPath: (root as NSString).appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        var index = FolderGalleryIndex(schemaVersion: schemaVersion, id: UUID().uuidString,
                                       name: name, createdAt: Date(), entries: [])
        index.galleryRoot = root
        try index.write()
        return index
    }

    public static func load(fromGalleryRoot root: String) throws -> FolderGalleryIndex {
        let data = try Data(contentsOf: URL(fileURLWithPath: indexPath(inGalleryRoot: root)))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var index = try decoder.decode(FolderGalleryIndex.self, from: data)
        index.galleryRoot = root
        return index
    }

    /// Record an asset. Appending the same asset twice replaces its entry, so
    /// a resumed transfer cannot list it twice.
    public mutating func append(_ entry: Entry) throws {
        if let existing = entries.firstIndex(where: { $0.assetID == entry.assetID }) {
            entries[existing] = entry
        } else {
            entries.append(entry)
        }
        try write()
    }

    public mutating func remove(assetID: String) throws {
        entries.removeAll { $0.assetID == assetID }
        try write()
    }

    /// Every media path in use, for collision checks.
    public var takenMediaPaths: Set<String> { Set(entries.map(\.mediaPath)) }

    /// Atomic: write a temp file beside the index, then replace. A drive
    /// pulled mid-write leaves the previous index readable.
    public func write() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(self)
        let final = Self.indexPath(inGalleryRoot: galleryRoot)
        let temp = final + ".tmp"
        try data.write(to: URL(fileURLWithPath: temp), options: .atomic)
        _ = try FileManager.default.replaceItemAt(URL(fileURLWithPath: final), withItemAt: URL(fileURLWithPath: temp))
    }

    /// Absolute path of a relative entry path inside this gallery.
    public func absolutePath(for relative: String) -> String {
        (galleryRoot as NSString).appendingPathComponent(relative)
    }

    public var root: String { galleryRoot }
}
