// ArchiveManifest.swift — .cbarchive manifest types + JSONL codec
//
// Types for the ComfyBox Gallery archive bundle format:
//   <archiveRoot>/<slug>-<yyyyMMdd-HHmmss>.cbarchive/
//     manifest.json   — ArchiveManifest header (schemaVersion, counts, folders)
//     entries.jsonl    — one ArchivedAsset JSON object per line, no wrapper array
//     assets/<id>/…    — original file + optional sidecar + optional thumbnail
//
// All timestamps are Double epoch seconds end-to-end (matches SQLite REAL exactly;
// no ISO-8601 formatting drift, no JSONEncoder date-strategy to get wrong).

import Foundation

// MARK: - Errors

/// Errors surfaced while reading an archive bundle. Extended by later tasks
/// as the reader/writer/restore pipeline grows.
public enum ArchiveError: Error, LocalizedError, Equatable {
    /// `schemaVersion` is missing or `< 1` — the manifest cannot be trusted.
    case unreadableManifest
    /// `schemaVersion > currentSchemaVersion` — written by a newer app version.
    case unsupportedSchemaVersion(Int)
    /// A relative path in the manifest is absolute, contains `..`, or resolves
    /// outside the bundle root. Manifests are untrusted input the moment
    /// archives are shareable.
    case pathTraversal(String)

    public var errorDescription: String? {
        switch self {
        case .unreadableManifest:
            return "This archive's manifest could not be read."
        case .unsupportedSchemaVersion(let version):
            return "This archive was written by a newer version of ComfyBox Desktop " +
                "(format v\(version)). Update the app to restore it."
        case .pathTraversal(let path):
            return "Archive entry path escapes the bundle: \(path)"
        }
    }
}

// MARK: - Manifest types

public struct ArchiveManifest: Codable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var archiveId: String
    public var name: String
    public var createdAt: Double          // epoch seconds — matches SQLite REAL exactly
    public var producer: String
    public var producerVersion: String?
    public var assetCount: Int
    public var totalBytes: Int64
    public var sourceMachine: String?
    public var sourceOutputDirectory: String?
    public var folders: [ArchivedFolder]

    public init(
        schemaVersion: Int = ArchiveManifest.currentSchemaVersion,
        archiveId: String,
        name: String,
        createdAt: Double,
        producer: String,
        producerVersion: String? = nil,
        assetCount: Int,
        totalBytes: Int64,
        sourceMachine: String? = nil,
        sourceOutputDirectory: String? = nil,
        folders: [ArchivedFolder]
    ) {
        self.schemaVersion = schemaVersion
        self.archiveId = archiveId
        self.name = name
        self.createdAt = createdAt
        self.producer = producer
        self.producerVersion = producerVersion
        self.assetCount = assetCount
        self.totalBytes = totalBytes
        self.sourceMachine = sourceMachine
        self.sourceOutputDirectory = sourceOutputDirectory
        self.folders = folders
    }

    /// Decodes a `manifest.json` payload and enforces the version policy:
    /// missing/`< 1` → `.unreadableManifest`, `> currentSchemaVersion` →
    /// `.unsupportedSchemaVersion`, `== 1` → accepted.
    public static func decode(_ data: Data) throws -> ArchiveManifest {
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let version = obj["schemaVersion"] as? Int,
              version >= 1
        else {
            throw ArchiveError.unreadableManifest
        }
        guard version <= currentSchemaVersion else {
            throw ArchiveError.unsupportedSchemaVersion(version)
        }
        do {
            return try JSONDecoder().decode(ArchiveManifest.self, from: data)
        } catch {
            throw ArchiveError.unreadableManifest
        }
    }
}

public struct ArchivedFolder: Codable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var createdAt: Double

    public init(id: String, name: String, createdAt: Double) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
    }
}

public struct ArchivedAsset: Codable, Sendable, Identifiable {
    // Locators — bundle-relative, portable
    public var id: String
    public var relativePath: String
    public var sidecarRelativePath: String?
    public var thumbnailRelativePath: String?
    public var originalPath: String?          // tilde-abbreviated hint, never authoritative

    // Every `assets` column
    public var kind: String
    public var filename: String
    public var fileSize: Int64
    public var sha256: String?
    public var width: Int?
    public var height: Int?
    public var createdAt: Double
    public var modifiedAt: Double
    public var ingestedAt: Double
    public var orphaned: Bool
    public var prompt: String?
    public var negativePrompt: String?
    public var seed: Int?
    public var steps: Int?
    public var guidance: Double?
    public var modelFamily: String?
    public var rating: Int
    public var favorite: Bool
    public var contentMode: String?
    public var characterName: String?
    public var source: String?

    // Membership + bookkeeping
    public var folderId: String?
    public var archivedAt: Double

    /// Builds an archive entry for `asset`. `relativeRoot` is this asset's
    /// bundle-relative directory (`assets/<id>`); locators are derived from
    /// it. `sidecarRelativePath`/`thumbnailRelativePath` are populated with
    /// their conventional paths — the archive writer clears whichever one
    /// doesn't actually exist on disk before persisting.
    public init(from asset: DAMAsset, folderId: String?, relativeRoot: String) {
        self.id = asset.id
        self.relativePath = "\(relativeRoot)/\(asset.filename)"
        let baseName = (asset.filename as NSString).deletingPathExtension
        self.sidecarRelativePath = "\(relativeRoot)/\(baseName).json"
        self.thumbnailRelativePath = "\(relativeRoot)/thumb.jpg"
        self.originalPath = (asset.absolutePath as NSString).abbreviatingWithTildeInPath

        self.kind = asset.kind
        self.filename = asset.filename
        self.fileSize = asset.fileSize
        self.sha256 = asset.sha256
        self.width = asset.width
        self.height = asset.height
        self.createdAt = asset.createdAt.timeIntervalSince1970
        self.modifiedAt = asset.modifiedAt.timeIntervalSince1970
        self.ingestedAt = asset.ingestedAt.timeIntervalSince1970
        self.orphaned = asset.orphaned
        self.prompt = asset.prompt
        self.negativePrompt = asset.negativePrompt
        self.seed = asset.seed
        self.steps = asset.steps
        self.guidance = asset.guidance
        self.modelFamily = asset.modelFamily
        self.rating = asset.rating
        self.favorite = asset.favorite
        self.contentMode = asset.contentMode
        self.characterName = asset.characterName
        self.source = asset.source

        self.folderId = folderId
        self.archivedAt = Date().timeIntervalSince1970
    }

    /// Rebuilds a `DAMAsset` from this entry. `absolutePath` is supplied by
    /// the caller (resolved against the opened bundle's root, never from
    /// `originalPath`); `overrideId` lets restore mint a fresh id when the
    /// original would collide with an existing asset.
    public func toDAMAsset(absolutePath: String, id overrideId: String? = nil) -> DAMAsset {
        DAMAsset(
            id: overrideId ?? id,
            kind: kind,
            filename: filename,
            absolutePath: absolutePath,
            fileSize: fileSize,
            sha256: sha256,
            width: width,
            height: height,
            createdAt: Date(timeIntervalSince1970: createdAt),
            modifiedAt: Date(timeIntervalSince1970: modifiedAt),
            ingestedAt: Date(timeIntervalSince1970: ingestedAt),
            orphaned: orphaned,
            prompt: prompt,
            negativePrompt: negativePrompt,
            seed: seed,
            steps: steps,
            guidance: guidance,
            modelFamily: modelFamily,
            rating: rating,
            favorite: favorite,
            contentMode: contentMode,
            characterName: characterName,
            source: source
        )
    }
}

// MARK: - JSONL codec

public enum ArchiveJSONL {
    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = []
        encoder.keyEncodingStrategy = .useDefaultKeys
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .useDefaultKeys
        return decoder
    }()

    /// One compact JSON object + `"\n"`. Appended with a `FileHandle` in batches.
    public static func encodeLine(_ entry: ArchivedAsset) throws -> Data {
        var data = try encoder.encode(entry)
        data.append(0x0A) // "\n"
        return data
    }

    /// Streams a `.jsonl` file, decoding one entry per line. Malformed lines
    /// are collected into `skipped` rather than aborting the read; errors
    /// thrown by `onEntry` itself propagate.
    public static func read(
        at path: String,
        onEntry: (ArchivedAsset) throws -> Void
    ) throws -> (count: Int, skipped: Int) {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }

        var count = 0
        var skipped = 0
        var buffer = Data()
        let newline = UInt8(ascii: "\n")
        let chunkSize = 1 << 16

        func processLine(_ lineData: Data) throws {
            let trimmed = lineData.last == UInt8(ascii: "\r") ? lineData.dropLast() : lineData[...]
            guard !trimmed.isEmpty else { return }
            guard let entry = try? decoder.decode(ArchivedAsset.self, from: trimmed) else {
                skipped += 1
                return
            }
            try onEntry(entry)
            count += 1
        }

        while true {
            let chunk = handle.readData(ofLength: chunkSize)
            if chunk.isEmpty { break }
            buffer.append(chunk)
            while let newlineIndex = buffer.firstIndex(of: newline) {
                let lineData = buffer[buffer.startIndex..<newlineIndex]
                try processLine(Data(lineData))
                buffer.removeSubrange(buffer.startIndex...newlineIndex)
            }
        }
        if !buffer.isEmpty {
            try processLine(buffer)
        }
        return (count, skipped)
    }
}

// MARK: - Path traversal guard

/// Resolves archive-relative paths (`relativePath`, `sidecarRelativePath`,
/// `thumbnailRelativePath`) against an opened bundle's root, rejecting
/// anything that could escape it. The bundle root always comes from the URL
/// the browser opened — never from anything inside the manifest — so a
/// shared/tampered archive can only point at paths inside itself.
public enum ArchivePaths {
    public static func resolveEntryPath(_ relativePath: String, in bundleRoot: URL) throws -> URL {
        guard !relativePath.hasPrefix("/") else {
            throw ArchiveError.pathTraversal(relativePath)
        }
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.contains("..") else {
            throw ArchiveError.pathTraversal(relativePath)
        }

        let root = bundleRoot.standardizedFileURL
        let candidate = root.appendingPathComponent(relativePath).standardizedFileURL

        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard candidate.path == root.path || candidate.path.hasPrefix(rootPath) else {
            throw ArchiveError.pathTraversal(relativePath)
        }
        return candidate
    }
}
