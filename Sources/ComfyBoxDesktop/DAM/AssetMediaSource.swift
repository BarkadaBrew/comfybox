// AssetMediaSource.swift — where a full-size surface gets one asset's bytes.
//
// The converged gallery lists 2,994 assets and 1,278 of them have no copy in
// this Mac's own library: their path is on the server, under the smbfs mount at
// /Volumes/todd, or nowhere reachable at all. The grid already coped
// (GalleryCellView falls back to streaming the thumbnail from the engine), but
// the detail pane and the lightbox opened `absolutePath` directly, so a row
// whose file wasn't there opened EMPTY — no image, no reason, nothing.
//
// One caveat worth knowing before trusting the remote leg: the engine validates
// `?path=` against its allowed output directory, so it will serve a row under
// ~/Pictures/ComfyBox and refuse one under /Volumes/todd with a 400. That is
// the engine's rule and it is deliberately unchanged here. What changed is the
// failure: a refusal now renders the cached thumbnail or a plain sentence,
// never a blank pane.
//
// This is the one place that answers "local, server, or neither", so the grid,
// the detail pane and the lightbox cannot disagree about a row. The inputs are
// the same ones the grid used: the local path CatalogBrowser resolved for the
// page, and the engine stream URL it built (`/v1/gallery/file?path=`, the route
// the spec deliberately left unchanged).
//
// THE GATE COMES FIRST. `resolve` returns `.gated` before it looks at anything
// else, so while the app is Rated G no surface reads a file OR issues an HTTP
// request for asset bytes. The streaming path added here must not become a way
// around AppContentGate, and the only way to guarantee that is to decide it
// here, above both the disk read and the network read.

import Foundation
import CryptoKit

/// Where the gallery decided a row's bytes live. Built once per surface from
/// `CatalogBrowser.localPath(forID:)` / `resolvedStreamURL(forID:)`.
public struct AssetMediaLocation: Equatable, Sendable {
    /// A path on this Mac to try first (may not exist — it is checked).
    public var localPath: String?
    /// The engine URL for a row whose bytes are only on a server.
    public var remoteURL: URL?

    public init(localPath: String?, remoteURL: URL?) {
        self.localPath = localPath
        self.remoteURL = remoteURL
    }
}

/// The resolved answer a view acts on.
public enum AssetMediaSource: Equatable, Sendable {
    /// Readable file on this Mac — open it directly, no HTTP.
    case local(String)
    /// Only on a server — fetch through the engine.
    case remote(URL)
    /// Not here and nothing claims it elsewhere.
    case missing
    /// The app is Rated G. No bytes are read at all until the gate is revealed.
    case gated

    public var isLocal: Bool { if case .local = self { return true }; return false }
    public var isRemote: Bool { if case .remote = self { return true }; return false }

    /// Resolve where to read from.
    ///
    /// Order matters: gate, then disk, then the server. A row with BOTH a local
    /// file and a server twin reads from disk — a local file is never streamed,
    /// matching `CatalogBrowser.resolvedStreamURL`, which returns nil for one.
    public static func resolve(
        _ location: AssetMediaLocation,
        gateRevealed: Bool,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) -> AssetMediaSource {
        guard gateRevealed else { return .gated }
        if let path = location.localPath, !path.isEmpty, fileExists(path) { return .local(path) }
        if let url = location.remoteURL { return .remote(url) }
        return .missing
    }

    /// Why an operation that needs a real file on this disk can't run, or nil
    /// when it can.
    ///
    /// Copy, Reveal in Finder and the Finder colour labels all write to or read
    /// from the file itself. For a server-side row they would fail silently —
    /// `NSWorkspace.selectFile` on a path that isn't there just does nothing —
    /// so the controls are disabled with this as their reason instead.
    public var localOnlyReason: String? {
        switch self {
        case .local: return nil
        case .remote: return "This asset is on the server. Use “Save to this Mac” in the gallery first."
        case .missing: return "This asset's file isn't on this Mac."
        case .gated: return "Hidden while the app is Rated G."
        }
    }
}

/// Server-side video, pulled once to a temp file so AVPlayer can play it.
///
/// The engine's `/v1/gallery/file` writes the whole body in one response and
/// advertises no Range support, and AVFoundation needs byte ranges to play an
/// HTTP asset progressively — handing AVPlayer that URL yields a player that
/// never becomes ready and a lightbox that spins forever. 381 of the catalog's
/// assets are video, so "degrade honestly" here means: fetch it, then play the
/// copy, and say so while it's fetching.
public enum RemoteMediaCache {

    /// Where copies live. Under the temp directory on purpose: this is a cache,
    /// not a library, and "Save to this Mac" remains the way to keep a file.
    public static var directory: URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("comfybox-remote-media", isDirectory: true)
    }

    /// A stable path for one remote URL. Keyed by the full URL (which carries
    /// the server-side path), so two files with the same basename on different
    /// hosts never collide, and the extension is preserved because AVFoundation
    /// sniffs the container from it.
    public static func cacheURL(for remote: URL, filename: String, in directory: URL) -> URL {
        let digest = SHA256.hash(data: Data(remote.absoluteString.utf8))
        let key = digest.compactMap { String(format: "%02x", $0) }.joined().prefix(16)
        let ext = (filename as NSString).pathExtension
        let name = ext.isEmpty ? String(key) : "\(key).\(ext)"
        return directory.appendingPathComponent(name)
    }

    /// A remote fetch failure, classified — never a raw HTTP status number
    /// surfaced to the browse view (#223 (a)). `.unauthorized` is the one a
    /// caller can act on: it means whatever session/password the request
    /// carried is no good, and the fix is to re-open the unlock prompt, not
    /// to show a dead-end error banner.
    public enum FetchError: Error, LocalizedError, Equatable {
        case unauthorized
        case notFound
        case server(Int)

        /// Classifies an HTTP status code. A pure, directly unit-testable
        /// mapping — the one place that decides which codes mean "the
        /// unlock prompt should reopen".
        public static func classify(statusCode: Int) -> FetchError {
            switch statusCode {
            case 401, 403: return .unauthorized
            case 404: return .notFound
            default: return .server(statusCode)
            }
        }

        public var errorDescription: String? {
            switch self {
            case .unauthorized: return "This server rejected the session — enter the password again."
            case .notFound: return "The server no longer has this file."
            case .server(let code): return "The server couldn't provide this file (code \(code))."
            }
        }
    }

    /// Fetch `remote` into the cache (or reuse the copy already there).
    public static func localCopy(of remote: URL, filename: String) async throws -> URL {
        let dir = directory
        let dest = cacheURL(for: remote, filename: filename, in: dir)
        let fm = FileManager.default
        if fm.fileExists(atPath: dest.path) { return dest }
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let (tmp, response) = try await URLSession.shared.download(from: remote)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            try? fm.removeItem(at: tmp)
            throw FetchError.classify(statusCode: http.statusCode)
        }
        // Another view may have won the race while this download ran.
        if fm.fileExists(atPath: dest.path) {
            try? fm.removeItem(at: tmp)
            return dest
        }
        try fm.moveItem(at: tmp, to: dest)
        return dest
    }
}
