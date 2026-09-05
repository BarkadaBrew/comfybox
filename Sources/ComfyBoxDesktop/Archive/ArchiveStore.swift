// ArchiveStore.swift — lists .cbarchive bundles for the archive browser
//
// Scans DesktopSettings.archiveRoots (or the single default root) one level
// deep for `*.cbarchive` directories and decodes just each bundle's
// `manifest.json` header — never `entries.jsonl`, which can be large and
// isn't needed for a listing (assetCount/totalBytes already live in the
// header). Filesystem work runs off-main via Task.detached, matching the
// pattern in AssetIngestor/GalleryArchiver; published state is only ever
// mutated back on the main actor.

import Foundation

@Observable
@MainActor
public final class ArchiveStore {
    /// One row in the archive browser.
    public struct Summary: Identifiable, Sendable {
        public var id: String            // bundlePath
        public var bundlePath: String
        public var manifest: ArchiveManifest
        public var isIncomplete: Bool
        public var hasPendingRemoval: Bool
        /// True for a `<name>.cbarchive.zip` FILE produced by `compress(_:)`
        /// (#223 (b)) — the uncompressed `.cbarchive` DIRECTORY it replaced no
        /// longer exists. Restore/Export/Compress-again are not offered for
        /// these rows (decompress-then-restore is a follow-up; see the PR body).
        public var isCompressed: Bool
        /// True when the `.zip` AND the `.cbarchive` directory it should have
        /// replaced BOTH still exist — the verified rename succeeded but the
        /// final directory removal didn't (review round 2: a permissions
        /// failure, a file still open, …). Recoverable, not a silent orphan:
        /// `compress(_:)` on this row re-verifies the zip and finishes the
        /// removal (`ArchiveStore.finishPendingCleanup`) instead of refusing
        /// with "already compressed". Always false unless `isCompressed` is
        /// also true.
        public var compressionCleanupPending: Bool

        public init(
            id: String, bundlePath: String, manifest: ArchiveManifest,
            isIncomplete: Bool, hasPendingRemoval: Bool, isCompressed: Bool = false,
            compressionCleanupPending: Bool = false
        ) {
            self.id = id
            self.bundlePath = bundlePath
            self.manifest = manifest
            self.isIncomplete = isIncomplete
            self.hasPendingRemoval = hasPendingRemoval
            self.isCompressed = isCompressed
            self.compressionCleanupPending = compressionCleanupPending
        }
    }

    public private(set) var archives: [Summary] = []
    public private(set) var isLoading = false
    public var error: String?
    public var roots: [String]

    /// `roots` defaults to `DesktopSettings.archiveRoots`, falling back to
    /// the single default archive directory when unset.
    public init(roots: [String]? = nil) {
        self.roots = roots ?? (DesktopSettings.load().archiveRoots ?? [DesktopSettings.defaultArchiveRoot])
    }

    /// Rescans `roots` and replaces `archives`. Bundles whose `manifest.json`
    /// is missing/unreadable are skipped and noted in `error` — unless the
    /// bundle carries `INCOMPLETE.json`, in which case a minimal `Summary` is
    /// synthesized so the browser can still offer to discard it (the crash
    /// that left the marker behind may have happened before the manifest was
    /// ever written).
    public func reload() async {
        isLoading = true
        error = nil
        let roots = self.roots
        let outcome = await Task.detached(priority: .utility) {
            Self.scan(roots: roots)
        }.value
        archives = outcome.summaries
        error = outcome.errors.isEmpty ? nil : outcome.errors.joined(separator: "\n")
        isLoading = false
    }

    // MARK: - Off-main scan

    private struct ScanOutcome: Sendable {
        var summaries: [Summary]
        var errors: [String]
    }

    private nonisolated static func scan(roots: [String]) -> ScanOutcome {
        let fm = FileManager.default
        var summaries: [Summary] = []
        var errors: [String] = []

        for root in roots {
            guard let entries = try? fm.contentsOfDirectory(atPath: root) else { continue }
            for entry in entries.sorted() {
                let fullPath = (root as NSString).appendingPathComponent(entry)
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: fullPath, isDirectory: &isDir) else { continue }

                if entry.hasSuffix(".cbarchive"), isDir.boolValue {
                    // Review round 2: if `compress(_:)` verified-and-renamed
                    // the zip but then failed to remove THIS directory, both
                    // exist — that pair is represented ONCE, by the
                    // `.cbarchive.zip` branch below (with
                    // `compressionCleanupPending`), not twice.
                    guard !fm.fileExists(atPath: fullPath + ".zip") else { continue }

                    let manifestPath = (fullPath as NSString).appendingPathComponent("manifest.json")
                    let incompletePath = (fullPath as NSString).appendingPathComponent("INCOMPLETE.json")
                    let pendingRemovalPath = (fullPath as NSString).appendingPathComponent("PENDING_REMOVAL.json")

                    let isIncomplete = fm.fileExists(atPath: incompletePath)
                    let hasPendingRemoval = fm.fileExists(atPath: pendingRemovalPath)

                    if let data = fm.contents(atPath: manifestPath),
                       let manifest = try? ArchiveManifest.decode(data) {
                        summaries.append(Summary(
                            id: fullPath, bundlePath: fullPath, manifest: manifest,
                            isIncomplete: isIncomplete, hasPendingRemoval: hasPendingRemoval
                        ))
                    } else if isIncomplete {
                        // Marker present, no readable manifest yet (or ever) —
                        // still list it so the browser can offer "Discard".
                        summaries.append(Summary(
                            id: fullPath, bundlePath: fullPath,
                            manifest: synthesizedManifest(bundleName: entry, bundlePath: fullPath),
                            isIncomplete: true, hasPendingRemoval: hasPendingRemoval
                        ))
                    } else {
                        errors.append("Could not read manifest for \(entry)")
                    }
                } else if isCompressedArchivePath(entry), !isDir.boolValue {
                    // A `compress(_:)` (#223 (b)) output. Usually the
                    // directory bundle it replaced is already gone, so there
                    // is no manifest.json to read off disk directly — read
                    // it out of the zip itself either way (also re-validates
                    // the zip is actually readable on every scan, not just
                    // once at compress time).
                    let bundleDirName = (entry as NSString).deletingPathExtension
                    let siblingDirPath = (root as NSString).appendingPathComponent(bundleDirName)
                    var siblingIsDir: ObjCBool = false
                    let cleanupPending = fm.fileExists(atPath: siblingDirPath, isDirectory: &siblingIsDir)
                        && siblingIsDir.boolValue

                    if let manifest = try? readManifestFromZip(zipPath: fullPath, bundleDirName: bundleDirName) {
                        summaries.append(Summary(
                            id: fullPath, bundlePath: fullPath, manifest: manifest,
                            isIncomplete: false, hasPendingRemoval: false, isCompressed: true,
                            compressionCleanupPending: cleanupPending
                        ))
                    } else {
                        errors.append("Could not read manifest for \(entry)")
                    }
                }
            }
        }

        summaries.sort { $0.manifest.createdAt > $1.manifest.createdAt }
        return ScanOutcome(summaries: summaries, errors: errors)
    }

    /// Minimal placeholder header for a bundle whose real `manifest.json`
    /// doesn't exist (yet). `createdAt` falls back to the bundle directory's
    /// filesystem creation date so newest-first ordering still behaves.
    private nonisolated static func synthesizedManifest(bundleName: String, bundlePath: String) -> ArchiveManifest {
        let attrs = try? FileManager.default.attributesOfItem(atPath: bundlePath)
        let createdAt = (attrs?[.creationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let name = (bundleName as NSString).deletingPathExtension
        return ArchiveManifest(
            archiveId: bundleName,
            name: name,
            createdAt: createdAt,
            producer: "",
            assetCount: 0,
            totalBytes: 0,
            folders: []
        )
    }

    // MARK: - Export as Zip (T12)

    public enum ExportError: Error, LocalizedError {
        case failed(Int32, String)

        public var errorDescription: String? {
            switch self {
            case .failed(let code, let output):
                let tail = output.split(separator: "\n").last.map(String.init) ?? ""
                return "ditto failed (\(code))\(tail.isEmpty ? "" : ": \(tail)")"
            }
        }
    }

    /// `ditto -c -k --sequesterRsrc --keepParent <bundlePath> <destination>` —
    /// pure argument builder, unit-tested against the exact vector.
    nonisolated static func dittoArguments(bundlePath: String, destination: String) -> [String] {
        ["-c", "-k", "--sequesterRsrc", "--keepParent", bundlePath, destination]
    }

    /// Shells `/usr/bin/ditto` to zip `bundlePath` to `destination` (a full
    /// `.zip` path). Runs off the main actor; throws `ExportError.failed` on
    /// a non-zero exit.
    public func exportAsZip(bundlePath: String, destination: String) async throws {
        try await Task.detached(priority: .utility) {
            try Self.runDitto(bundlePath: bundlePath, destination: destination)
        }.value
    }

    private nonisolated static func runDitto(bundlePath: String, destination: String) throws {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        proc.arguments = dittoArguments(bundlePath: bundlePath, destination: destination)
        let pipe = Pipe()
        proc.standardOutput = pipe
        proc.standardError = pipe
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else {
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            let output = String(data: data, encoding: .utf8) ?? ""
            throw ExportError.failed(proc.terminationStatus, output)
        }
    }

    // MARK: - Compress (#223 (b))
    //
    // "Delete exists, compress does not" — an archived gallery bundle is a
    // directory of raw copied files (GalleryArchiver's writer never
    // compresses anything; that was never its job). Compress turns an
    // eligible bundle into cold storage: a single verified `.zip` replacing
    // the directory, written atomically (`.part` then rename) and NEVER
    // deleting the uncompressed source until the zip has been reopened and
    // proven readable. Reuses the exact `ditto`/`dittoArguments` writer
    // `exportAsZip` (T12) already shells — the difference is atomicity,
    // verify-before-delete, and writing IN PLACE rather than to a
    // caller-chosen export destination that leaves the original untouched.
    //
    // Restore/Export/Compress-again are not offered for a compressed row in
    // ArchiveBrowserView (`Summary.isCompressed`) — decompressing back to a
    // browsable bundle is a follow-up, not part of this change (see the PR
    // body); the zip is fully valid cold storage on its own (`ditto -x` or
    // Archive Utility opens it) even without that follow-up.

    public struct CompressResult: Sendable, Equatable {
        public var zipPath: String
        public var originalBytes: Int64
        public var compressedBytes: Int64

        public init(zipPath: String, originalBytes: Int64, compressedBytes: Int64) {
            self.zipPath = zipPath
            self.originalBytes = originalBytes
            self.compressedBytes = compressedBytes
        }
    }

    public enum CompressError: Error, LocalizedError, Equatable {
        /// The bundle isn't in a state that's safe to compress (incomplete,
        /// mid-removal, already compressed, or a `.zip` already sits at the
        /// destination path).
        case notEligible(String)
        /// The bundle's `entries.jsonl` points outside its own root — the
        /// traversal guard (#264) tripped, so this bundle is refused rather
        /// than compressed (and thereby preserved/propagated) as-is.
        case unsafeEntry(String)
        /// The `.part` zip `ditto` produced could not be reopened and
        /// decoded back to the same manifest — nothing was deleted.
        case verifyFailed(String)

        public var errorDescription: String? {
            switch self {
            case .notEligible(let reason): return reason
            case .unsafeEntry(let path): return "Archive entry path escapes the bundle: \(path)"
            case .verifyFailed(let reason): return reason
            }
        }
    }

    /// Compresses `bundle`'s directory into a `.zip` in place, deleting the
    /// directory only once the zip is verified readable. Throws
    /// `CompressError.notEligible` up front for a bundle that shouldn't be
    /// touched (mirrors the same safety class `deleteBundle` already checks
    /// in `ArchiveBrowserView` — incomplete / mid-removal — plus "already
    /// compressed").
    @discardableResult
    /// A bundle carrying `compressionCleanupPending` (both the directory AND
    /// its `.zip` exist — the orphan pair review round 2 asked for recovery
    /// from) routes to `finishPendingCleanup` instead of the ineligibility
    /// guards below: it IS eligible, its zip already verified readable at
    /// `scan()` time, and the only remaining work is the directory removal
    /// that didn't finish last time. `finishPendingCleanup` is itself
    /// idempotent (re-verifies, then removes only if the directory is still
    /// there), so calling this again after a second failure is always safe.
    public func compress(_ bundle: Summary) async throws -> CompressResult {
        if bundle.isCompressed {
            guard bundle.compressionCleanupPending else {
                throw CompressError.notEligible("This archive is already compressed.")
            }
            let zipPath = bundle.bundlePath
            return try await Task.detached(priority: .utility) {
                try Self.finishPendingCleanup(zipPath: zipPath)
            }.value
        }
        guard !bundle.isIncomplete else {
            throw CompressError.notEligible("This archive never finished writing — delete it instead of compressing it.")
        }
        guard !bundle.hasPendingRemoval else {
            throw CompressError.notEligible("This archive is still finishing a prior operation — try again shortly.")
        }
        let bundlePath = bundle.bundlePath
        return try await Task.detached(priority: .utility) {
            try Self.performCompress(bundlePath: bundlePath)
        }.value
    }

    /// `removeBundleDirectory` is injectable (default: the real
    /// `FileManager.removeItem`) purely so a test can simulate the final
    /// removal failing — the exact case review round 2 asked to be
    /// recoverable rather than left as a silent orphan pair. `internal`
    /// (not `private`) so `ArchiveStoreTests` can call it directly with a
    /// failing closure without going through the public `compress(_:)`.
    nonisolated static func performCompress(
        bundlePath: String,
        removeBundleDirectory: (String) throws -> Void = { try FileManager.default.removeItem(atPath: $0) }
    ) throws -> CompressResult {
        // Respect the traversal guard (#264) before anything else: refuse a
        // bundle whose entries.jsonl resolves outside its own root rather
        // than compressing (and thereby legitimizing) it.
        try validateEntriesForCompress(bundlePath: bundlePath)

        let fm = FileManager.default
        let zipPath = compressedZipPath(for: bundlePath)
        guard !fm.fileExists(atPath: zipPath) else {
            throw CompressError.notEligible("A compressed archive already exists at \(zipPath).")
        }
        let originalBytes = directorySize(atPath: bundlePath)

        // Atomic: write to `.part`, verify, THEN rename into place.
        let partialPath = zipPath + ".part"
        if fm.fileExists(atPath: partialPath) {
            try? fm.removeItem(atPath: partialPath)
        }
        try runDitto(bundlePath: bundlePath, destination: partialPath)

        // Verify readable BEFORE the source is touched: reopen the `.part`
        // zip and decode its manifest.json back out.
        let bundleDirName = (bundlePath as NSString).lastPathComponent
        guard (try? readManifestFromZip(zipPath: partialPath, bundleDirName: bundleDirName)) != nil else {
            try? fm.removeItem(atPath: partialPath)
            throw CompressError.verifyFailed(
                "The compressed archive could not be verified readable — nothing was deleted."
            )
        }

        try fm.moveItem(atPath: partialPath, toPath: zipPath)

        // Only now — the destination is a verified-readable zip — remove the
        // uncompressed source. If THIS throws (permissions, a file still
        // open, …), both the zip and the directory are left on disk: a
        // recoverable pair, not a silent orphan — `scan()` lists it as one
        // `compressionCleanupPending` row, and `compress(_:)` on that row
        // re-enters through `finishPendingCleanup` rather than
        // `notEligible`. Nothing here catches the throw; it propagates so
        // the caller's `errorMessage` names the real failure.
        try removeBundleDirectory(bundlePath)

        let compressedBytes = (try? fm.attributesOfItem(atPath: zipPath))?[.size] as? Int64 ?? 0
        return CompressResult(zipPath: zipPath, originalBytes: originalBytes, compressedBytes: compressedBytes)
    }

    /// Finishes a `compressionCleanupPending` row: re-verifies the zip is
    /// still readable (never trust a previous verification blindly — the
    /// zip is exactly the file a second, unrelated failure could also have
    /// touched), then removes the leftover directory if it's still there.
    /// Idempotent by construction: a directory already gone (this ran
    /// successfully before, or something else cleaned it up) is simply
    /// skipped, not an error. `removeBundleDirectory` is the same
    /// injection point `performCompress` uses, for the same testing reason.
    nonisolated static func finishPendingCleanup(
        zipPath: String,
        removeBundleDirectory: (String) throws -> Void = { try FileManager.default.removeItem(atPath: $0) }
    ) throws -> CompressResult {
        let fm = FileManager.default
        // `compressedZipPath(for:)` only ever appends ".zip", so stripping
        // the last path extension is exactly its inverse.
        let bundlePath = (zipPath as NSString).deletingPathExtension
        let bundleDirName = (bundlePath as NSString).lastPathComponent

        guard (try? readManifestFromZip(zipPath: zipPath, bundleDirName: bundleDirName)) != nil else {
            throw CompressError.verifyFailed(
                "The compressed archive could not be verified readable — nothing was deleted."
            )
        }

        let stillPresent = fm.fileExists(atPath: bundlePath)
        let originalBytes = stillPresent ? directorySize(atPath: bundlePath) : 0
        if stillPresent {
            try removeBundleDirectory(bundlePath)
        }

        let compressedBytes = (try? fm.attributesOfItem(atPath: zipPath))?[.size] as? Int64 ?? 0
        return CompressResult(zipPath: zipPath, originalBytes: originalBytes, compressedBytes: compressedBytes)
    }

    /// `<bundlePath>.zip` — kept alongside the (about-to-be-removed)
    /// directory bundle rather than in a caller-chosen destination, so
    /// `ArchiveStore.scan()` picks it back up as the same archive under
    /// `isCompressedArchivePath`.
    nonisolated static func compressedZipPath(for bundlePath: String) -> String {
        bundlePath + ".zip"
    }

    /// True for a `<name>.cbarchive.zip` path — a compressed archive's own
    /// on-disk spelling, distinct from a plain `.zip` a user happened to
    /// place in an archive root (which is left alone) and from the directory
    /// `.cbarchive` bundle format.
    nonisolated static func isCompressedArchivePath(_ path: String) -> Bool {
        path.hasSuffix(".cbarchive.zip")
    }

    /// Reads every entry in `bundlePath`'s `entries.jsonl` (when present —
    /// hand-built test fixtures often omit it, matching the existing
    /// header-only `scan()` behavior) purely to run each locator through
    /// `ArchivePaths.resolveEntryPath`, the traversal guard from #264.
    /// Throws `CompressError.unsafeEntry` for the first entry that fails it;
    /// never writes anything.
    nonisolated static func validateEntriesForCompress(bundlePath: String) throws {
        let entriesPath = (bundlePath as NSString).appendingPathComponent("entries.jsonl")
        guard FileManager.default.fileExists(atPath: entriesPath) else { return }
        let bundleRoot = URL(fileURLWithPath: bundlePath)
        do {
            _ = try ArchiveJSONL.read(at: entriesPath) { entry in
                _ = try ArchivePaths.resolveEntryPath(entry.relativePath, in: bundleRoot)
                if let sidecar = entry.sidecarRelativePath {
                    _ = try ArchivePaths.resolveEntryPath(sidecar, in: bundleRoot)
                }
                if let thumb = entry.thumbnailRelativePath {
                    _ = try ArchivePaths.resolveEntryPath(thumb, in: bundleRoot)
                }
            }
        } catch let error as ArchiveError {
            if case .pathTraversal(let path) = error {
                throw CompressError.unsafeEntry(path)
            }
            throw error
        }
    }

    /// Shells `unzip -p <zipPath> "<bundleDirName>/manifest.json"` and decodes
    /// the result — both the verify-before-delete step in `performCompress`
    /// and how `scan()` lists a compressed row without a directory bundle
    /// left on disk to read `manifest.json` from directly. `bundleDirName` is
    /// the entry prefix `ditto --keepParent` wrote every path under.
    nonisolated static func readManifestFromZip(zipPath: String, bundleDirName: String) throws -> ArchiveManifest {
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
        proc.arguments = ["-p", zipPath, "\(bundleDirName)/manifest.json"]
        let outPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = Pipe()   // discard — never let stderr chatter into the captured bytes
        try proc.run()
        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0, !data.isEmpty else {
            throw CompressError.verifyFailed("Could not read manifest.json from the compressed archive.")
        }
        return try ArchiveManifest.decode(data)
    }

    /// Sum of every regular file's on-disk size under `path`, recursively.
    /// Used only for `CompressResult.originalBytes` — a summary figure, not a
    /// safety check, so a stat failure on one file just doesn't count it
    /// rather than aborting the whole compress.
    nonisolated static func directorySize(atPath path: String) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(atPath: path) else { return 0 }
        var total: Int64 = 0
        for case let relative as String in enumerator {
            let full = (path as NSString).appendingPathComponent(relative)
            if let size = (try? fm.attributesOfItem(atPath: full))?[.size] as? Int64 {
                total += size
            }
        }
        return total
    }
}
