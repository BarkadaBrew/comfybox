// AssetMediaSourceTests.swift — the local/remote decision the detail pane, the
// lightbox and the grid all share.
//
// The bug this exists to prevent: 1,278 of the catalog's 2,994 rows have no
// file on this Mac, and the detail view opened `absolutePath` regardless — an
// empty pane for every one of them. There is no ViewInspector in this repo, so
// what is testable is the decision itself; the views are wired to it by hand.

import XCTest
@testable import ComfyBoxDesktop

final class AssetMediaSourceTests: XCTestCase {

    private let engineURL = URL(string: "http://127.0.0.1:7870/v1/gallery/file?path=/home/todd/.kira/r.png")!

    // MARK: - The gate

    /// THE gate check. While the app is Rated G nothing is resolved to bytes —
    /// not a local file, not an HTTP fetch. This is what stops the streaming
    /// path from becoming a way around AppContentGate: the network URL is never
    /// even handed to the view, so there is nothing for it to request.
    func testNothingResolvesToBytesWhileTheAppIsRatedG() {
        let onServer = AssetMediaLocation(localPath: "/nope.png", remoteURL: engineURL)
        XCTAssertEqual(AssetMediaSource.resolve(onServer, gateRevealed: false, fileExists: { _ in false }),
                       .gated)

        let onDisk = AssetMediaLocation(localPath: "/here.png", remoteURL: nil)
        XCTAssertEqual(AssetMediaSource.resolve(onDisk, gateRevealed: false, fileExists: { _ in true }),
                       .gated, "a local file is withheld by the gate too")
    }

    /// A fresh launch is G-rated, so the default posture is "read nothing".
    @MainActor
    func testAFreshGateResolvesToGated() {
        let location = AssetMediaLocation(localPath: "/here.png", remoteURL: engineURL)
        XCTAssertEqual(
            AssetMediaSource.resolve(location, gateRevealed: AppContentGate().revealed,
                                     fileExists: { _ in true }),
            .gated)
    }

    // MARK: - Where the bytes are

    func testAFileOnThisMacIsOpenedFromDiskNotStreamed() {
        let both = AssetMediaLocation(localPath: "/here.png", remoteURL: engineURL)
        XCTAssertEqual(AssetMediaSource.resolve(both, gateRevealed: true, fileExists: { _ in true }),
                       .local("/here.png"),
                       "a row with a local file must never go over HTTP for it")
    }

    func testARowOnlyOnTheServerStreamsFromTheEngine() {
        let remote = AssetMediaLocation(localPath: "/Volumes/todd/gone.png", remoteURL: engineURL)
        XCTAssertEqual(AssetMediaSource.resolve(remote, gateRevealed: true, fileExists: { _ in false }),
                       .remote(engineURL))
    }

    /// The catalog's opinion loses to the filesystem: a `mac` location whose
    /// file has since been deleted falls through to streaming, exactly as
    /// `CatalogBrowser.isRemote` does, rather than rendering an empty pane.
    func testADeletedLocalCopyFallsThroughToTheServer() {
        let stale = AssetMediaLocation(localPath: "/Users/todd/deleted.png", remoteURL: engineURL)
        XCTAssertTrue(AssetMediaSource.resolve(stale, gateRevealed: true, fileExists: { _ in false }).isRemote)
    }

    func testNeitherHereNorAnywhereIsMissingNotABlankPane() {
        let nothing = AssetMediaLocation(localPath: "/gone.png", remoteURL: nil)
        XCTAssertEqual(AssetMediaSource.resolve(nothing, gateRevealed: true, fileExists: { _ in false }),
                       .missing)
    }

    func testAnEmptyPathIsNotTreatedAsALocalFile() {
        let empty = AssetMediaLocation(localPath: "", remoteURL: engineURL)
        XCTAssertEqual(AssetMediaSource.resolve(empty, gateRevealed: true, fileExists: { _ in true }),
                       .remote(engineURL))
    }

    // MARK: - What a remote row cannot do

    /// Copy, Reveal in Finder and the Finder colour labels all need the real
    /// file. For a server-side row they would fail silently — so they are
    /// disabled with a reason instead.
    func testLocalOnlyOperationsAreRefusedWithAReasonNotSilently() {
        XCTAssertNil(AssetMediaSource.local("/here.png").localOnlyReason)

        let remote = try? XCTUnwrap(AssetMediaSource.remote(engineURL).localOnlyReason)
        XCTAssertTrue((remote ?? "").contains("Save to this Mac"),
                      "the reason must point at the way to fix it")
        XCTAssertNotNil(AssetMediaSource.missing.localOnlyReason)
        XCTAssertNotNil(AssetMediaSource.gated.localOnlyReason)
    }

    // MARK: - Remote video

    /// 381 catalog assets are video. The engine's /v1/gallery/file serves whole
    /// bodies with no Range support and AVPlayer needs ranges over HTTP, so a
    /// remote video is fetched to a cache file and played from there. The cache
    /// name has to be stable (same URL, same file) and unique per URL.
    func testTheVideoCacheNameIsStablePerURLAndKeepsTheExtension() {
        let dir = URL(fileURLWithPath: "/tmp/comfybox-remote-media")
        let a = URL(string: "http://127.0.0.1:7870/v1/gallery/file?path=/srv/a/clip.mp4")!
        let b = URL(string: "http://127.0.0.1:7870/v1/gallery/file?path=/srv/b/clip.mp4")!

        let first = RemoteMediaCache.cacheURL(for: a, filename: "clip.mp4", in: dir)
        XCTAssertEqual(first, RemoteMediaCache.cacheURL(for: a, filename: "clip.mp4", in: dir),
                       "same URL must reuse the copy already fetched")
        XCTAssertEqual(first.pathExtension, "mp4",
                       "AVFoundation sniffs the container from the extension")
        XCTAssertNotEqual(first, RemoteMediaCache.cacheURL(for: b, filename: "clip.mp4", in: dir),
                          "two hosts can hold the same basename; they must not collide")
        XCTAssertEqual(first.deletingLastPathComponent().standardizedFileURL.path, dir.path)
    }

    func testTheCacheLivesUnderTempNotInTheLibrary() {
        XCTAssertTrue(RemoteMediaCache.directory.path.hasPrefix(FileManager.default.temporaryDirectory.path),
                      "this is a cache; keeping a file is what “Save to this Mac” is for")
    }

    // MARK: - FetchError classification (#223 (a))

    /// 401 and 403 are BOTH "the session/credential this request carried is
    /// no good" from the caller's point of view — neither should ever render
    /// as a raw status number in the browse view.
    func testClassifyMapsAuthCodesToUnauthorized() {
        XCTAssertEqual(RemoteMediaCache.FetchError.classify(statusCode: 401), .unauthorized)
        XCTAssertEqual(RemoteMediaCache.FetchError.classify(statusCode: 403), .unauthorized)
    }

    func testClassifyMapsNotFoundAndFallsThroughToServerForEverythingElse() {
        XCTAssertEqual(RemoteMediaCache.FetchError.classify(statusCode: 404), .notFound)
        XCTAssertEqual(RemoteMediaCache.FetchError.classify(statusCode: 500), .server(500))
        XCTAssertEqual(RemoteMediaCache.FetchError.classify(statusCode: 400), .server(400))
    }

    func testFetchErrorDescriptionsNeverLeakARawStatusCodeAlone() {
        // Every description is a sentence, not a bare number — "Server
        // returned 401" (the bug this exists to fix) is the one shape these
        // must never take again.
        XCTAssertFalse((RemoteMediaCache.FetchError.unauthorized.errorDescription ?? "").isEmpty)
        XCTAssertNotEqual(RemoteMediaCache.FetchError.unauthorized.errorDescription, "401")
        XCTAssertNotEqual(RemoteMediaCache.FetchError.server(401).errorDescription, "Server returned 401")
    }
}
