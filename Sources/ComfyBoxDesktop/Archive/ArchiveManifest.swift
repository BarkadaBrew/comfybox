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
///
/// Rule (issue #264): no code path may join a manifest-supplied relative
/// path onto a bundle root except through `resolveEntryPath` — manifests are
/// untrusted input the moment archives are shareable.
public enum ArchivePaths {
    /// Rejects a manifest-supplied relative path before it is ever joined
    /// onto `bundleRoot`, then re-checks containment after symlinks are
    /// resolved — and returns the *resolved* URL, not the lexical one.
    /// Three defenses, in order:
    ///
    /// 1. **Lexical**: an absolute path, or any `..` path component, is
    ///    rejected outright — this alone stops the common case and matches
    ///    what `standardizedFileURL` would collapse anyway.
    /// 2. **Percent-encoding**: `relativePath` is never percent-decoded to
    ///    build the candidate URL (`appendingPathComponent` treats it as a
    ///    literal path component, so `%2e%2e` on its own is inert today),
    ///    but a manifest could still carry an encoded — even doubly-encoded
    ///    (`%252e%252e`) — traversal hoping a future or alternate call site
    ///    decodes it first. `fullyPercentDecoded` decodes repeatedly until
    ///    the string stops changing (capped, so a manifest can't force
    ///    unbounded work) and the lexical check re-runs on the result,
    ///    closing that off without changing behavior for any legitimately
    ///    named file — a stray literal `%` that isn't valid
    ///    percent-encoding just makes `removingPercentEncoding` return
    ///    `nil`, which skips this extra check rather than rejecting the
    ///    file.
    /// 3. **Symlink escape**: `standardizedFileURL` only collapses `.` and
    ///    `..` lexically — it does not follow symlinks. A bundle (e.g.
    ///    extracted from a shared zip) could contain a symlink whose target
    ///    lives outside the bundle; a `relativePath` through it would pass
    ///    the lexical check yet resolve outside the bundle at open time.
    ///    Resolving symlinks on both `root` and `candidate` and re-checking
    ///    containment catches that.
    ///
    /// The function returns the **resolved** URL, not the lexical
    /// `candidate` — returning the unresolved path would mean every caller
    /// opens a different path than the one just checked, and a symlink
    /// swapped into place between the check and the open (plausible on a
    /// bundle living in a synced folder) would be followed anyway (TOCTOU).
    /// Returning the resolved URL means callers open exactly what was
    /// checked. `resolvingSymlinksInPath()` leaves the classic BSD
    /// compatibility symlinks (`/tmp`, `/var`, `/etc`) as-is rather than
    /// rewriting them to `/private/...`, so this doesn't surprise callers
    /// or tests that build bundle roots under `/tmp`.
    ///
    /// Backslash is not treated as a path separator here — deliberately:
    /// on APFS (this app's only supported filesystem) `\` is a legal
    /// filename character, not a separator, so `assets\..\etc` is one
    /// literal, harmless path component, never rejected.
    public static func resolveEntryPath(_ relativePath: String, in bundleRoot: URL) throws -> URL {
        try rejectTraversal(in: relativePath)

        let root = bundleRoot.standardizedFileURL
        let candidate = root.appendingPathComponent(relativePath).standardizedFileURL
        guard isContained(candidate, in: root) else {
            throw ArchiveError.pathTraversal(relativePath)
        }

        let resolvedRoot = root.resolvingSymlinksInPath()
        let resolvedCandidate = candidate.resolvingSymlinksInPath()
        guard isContained(resolvedCandidate, in: resolvedRoot) else {
            throw ArchiveError.pathTraversal(relativePath)
        }

        return resolvedCandidate
    }

    /// Validates a manifest-supplied bare `filename` before it is joined
    /// onto a *live* destination directory during restore (the write-side
    /// counterpart to `resolveEntryPath`'s bundle-root reads). Unlike a
    /// `relativePath`, a `filename` must never contain a path separator at
    /// all — it names one file directly inside the destination directory a
    /// caller already chose (the watch directory, or a validated original
    /// location), never a path relative to anything. A crafted
    /// `entries.jsonl` line could otherwise set `filename` to something like
    /// `"../../../Library/LaunchAgents/evil.plist"` and have restore's
    /// destination-path join write outside the gallery entirely.
    ///
    /// No symlink resolution here — a bare filename never has intermediate
    /// path components for a symlink to hide in, and it names a file that
    /// generally doesn't exist yet at the destination.
    @discardableResult
    public static func validateFilename(_ filename: String) throws -> String {
        guard isSafeFilename(filename) else {
            throw ArchiveError.pathTraversal(filename)
        }
        if let decoded = fullyPercentDecoded(filename), decoded != filename {
            guard isSafeFilename(decoded) else {
                throw ArchiveError.pathTraversal(filename)
            }
        }
        return filename
    }

    private static func isSafeFilename(_ filename: String) -> Bool {
        !filename.isEmpty && !filename.contains("/") && filename != "." && filename != ".."
    }

    /// Lexical check shared by `resolveEntryPath`'s primary pass and its
    /// percent-decoded re-check: absolute, or any path component exactly
    /// `".."`.
    private static func rejectTraversal(in relativePath: String) throws {
        guard !relativePath.hasPrefix("/") else {
            throw ArchiveError.pathTraversal(relativePath)
        }
        let components = relativePath.split(separator: "/", omittingEmptySubsequences: false)
        guard !components.contains("..") else {
            throw ArchiveError.pathTraversal(relativePath)
        }
        // Defense-in-depth: a manifest could carry a percent-encoded, or
        // even doubly-encoded (`%252e%252e`), traversal hoping some call
        // site decodes it before use. `fullyPercentDecoded` returns `nil`
        // when nothing decodes (e.g. a filename with a stray literal `%`),
        // which harmlessly skips this extra check rather than rejecting a
        // legitimately named file.
        if let decoded = fullyPercentDecoded(relativePath), decoded != relativePath {
            guard !decoded.hasPrefix("/") else {
                throw ArchiveError.pathTraversal(relativePath)
            }
            let decodedComponents = decoded.split(separator: "/", omittingEmptySubsequences: false)
            guard !decodedComponents.contains("..") else {
                throw ArchiveError.pathTraversal(relativePath)
            }
        }
    }

    /// Percent-decodes `value` repeatedly until it stops changing (a single
    /// pass only unwraps one layer of encoding, so `%252e%252e` — encoded
    /// `%2e%2e`, itself encoded `..` — would otherwise slip past a
    /// single-pass check), capped at 3 iterations so a manifest can't force
    /// unbounded decode work. Returns `nil` if no iteration actually
    /// changed the string (nothing to re-check) — including when
    /// `removingPercentEncoding` itself returns `nil` because `value` isn't
    /// valid percent-encoding at all.
    private static func fullyPercentDecoded(_ value: String, maxIterations: Int = 3) -> String? {
        var current = value
        var decodedAtLeastOnce = false
        for _ in 0..<maxIterations {
            guard let decoded = current.removingPercentEncoding, decoded != current else { break }
            current = decoded
            decodedAtLeastOnce = true
        }
        return decodedAtLeastOnce ? current : nil
    }

    private static func isContained(_ candidate: URL, in root: URL) -> Bool {
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        return candidate.path == root.path || candidate.path.hasPrefix(rootPath)
    }
}
