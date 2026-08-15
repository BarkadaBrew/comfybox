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

        public init(id: String, bundlePath: String, manifest: ArchiveManifest, isIncomplete: Bool, hasPendingRemoval: Bool) {
            self.id = id
            self.bundlePath = bundlePath
            self.manifest = manifest
            self.isIncomplete = isIncomplete
            self.hasPendingRemoval = hasPendingRemoval
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
            for entry in entries.sorted() where entry.hasSuffix(".cbarchive") {
                let bundlePath = (root as NSString).appendingPathComponent(entry)
                var isDir: ObjCBool = false
                guard fm.fileExists(atPath: bundlePath, isDirectory: &isDir), isDir.boolValue else { continue }

                let manifestPath = (bundlePath as NSString).appendingPathComponent("manifest.json")
                let incompletePath = (bundlePath as NSString).appendingPathComponent("INCOMPLETE.json")
                let pendingRemovalPath = (bundlePath as NSString).appendingPathComponent("PENDING_REMOVAL.json")

                let isIncomplete = fm.fileExists(atPath: incompletePath)
                let hasPendingRemoval = fm.fileExists(atPath: pendingRemovalPath)

                if let data = fm.contents(atPath: manifestPath),
                   let manifest = try? ArchiveManifest.decode(data) {
                    summaries.append(Summary(
                        id: bundlePath, bundlePath: bundlePath, manifest: manifest,
                        isIncomplete: isIncomplete, hasPendingRemoval: hasPendingRemoval
                    ))
                } else if isIncomplete {
                    // Marker present, no readable manifest yet (or ever) —
                    // still list it so the browser can offer "Discard".
                    summaries.append(Summary(
                        id: bundlePath, bundlePath: bundlePath,
                        manifest: synthesizedManifest(bundleName: entry, bundlePath: bundlePath),
                        isIncomplete: true, hasPendingRemoval: hasPendingRemoval
                    ))
                } else {
                    errors.append("Could not read manifest for \(entry)")
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
}
