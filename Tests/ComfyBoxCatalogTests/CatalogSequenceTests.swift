import XCTest

@testable import ComfyBoxCatalog

/// A Director render in the catalog (FDD-ltx-director-tab §4.9.2, WP13).
///
/// The clip's `kind` stays `"video"` — it IS one, and the play badge, lightbox,
/// container probe and i2v edge all key off that. What makes it a sequence is a
/// `.sequence.json` beside it, so `sequence_id` is a second axis, the way `mode`
/// ("i2v" | "t2v") already is.
final class CatalogSequenceTests: XCTestCase {

    private var directory: String!

    override func setUpWithError() throws {
        directory = NSTemporaryDirectory() + "catalog-sequence-\(UUID().uuidString)"
        try FileManager.default.createDirectory(
            atPath: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(atPath: directory)
    }

    private func write(_ json: String, besides clip: String) throws -> String {
        let media = directory + "/" + clip
        FileManager.default.createFile(atPath: media, contents: Data("mp4".utf8))
        let sidecar = MetadataReader.sequenceSidecarPath(forMedia: media)
        try Data(json.utf8).write(to: URL(fileURLWithPath: sidecar))
        return media
    }

    private var valid: String {
        """
        {"schema":"comfybox.sequence","version":1,"id":"seq-7","kind":"director",
         "name":"morning monologue","created_at":"2026-09-18T04:00:00Z",
         "library_item_ids":[],
         "timeline":{"version":1,"settings":{"width":576,"height":896,"length_frames":289},
                     "global_prompt":"x","keyframes":[],"prompt_segments":[],"audio_clips":[]},
         "chunks":[{"index":0,"start_frame":0,"frames":145},
                   {"index":1,"start_frame":144,"frames":145}],
         "assets":[],"outputs":[],"engine":{}}
        """
    }

    // MARK: the sidecar is found and read

    func testTheSidecarSitsBesideTheClipByName() {
        XCTAssertEqual(
            MetadataReader.sequenceSidecarPath(forMedia: "/a/b/clip.mp4"),
            "/a/b/clip.sequence.json")
    }

    func testAValidSidecarYieldsTheFactsThatMakeItFindable() throws {
        let media = try write(valid, besides: "clip.mp4")
        let meta = try XCTUnwrap(MetadataReader.readSequence(forMedia: media))
        XCTAssertEqual(meta.sequenceID, "seq-7")
        XCTAssertEqual(meta.sequenceName, "morning monologue")
        XCTAssertEqual(meta.sequenceChunks, 2, "chunk count comes from the array's length")
    }

    func testAnOrdinaryRenderHasNoSequence() {
        let media = directory + "/plain.mp4"
        FileManager.default.createFile(atPath: media, contents: Data("mp4".utf8))
        XCTAssertNil(MetadataReader.readSequence(forMedia: media))
    }

    // MARK: an unreadable sidecar must not lose the clip

    func testAnUnparseableSidecarLeavesTheClipAnOrdinaryVideo() throws {
        let media = try write("{ this is not json", besides: "clip.mp4")
        XCTAssertNil(
            MetadataReader.readSequence(forMedia: media),
            "a broken sidecar degrades to a plain video row, it never fails the file")
    }

    func testAForeignSidecarIsNotReadAsOneOfOurs() throws {
        // Some other tool's `.sequence.json` must not turn a clip into a
        // Director render the desktop then offers to reopen.
        let media = try write(#"{"id":"x","name":"y","chunks":[]}"#, besides: "clip.mp4")
        XCTAssertNil(MetadataReader.readSequence(forMedia: media), "no schema, not ours")
    }

    func testASidecarWithoutAnIdIsRefused() throws {
        let media = try write(
            #"{"schema":"comfybox.sequence","id":"","name":"y","chunks":[]}"#,
            besides: "clip.mp4")
        XCTAssertNil(MetadataReader.readSequence(forMedia: media))
    }

    func testAnEmptySidecarIsRefused() throws {
        let media = try write("", besides: "clip.mp4")
        XCTAssertNil(MetadataReader.readSequence(forMedia: media))
    }

    // MARK: the row keeps its kind

    func testTheRowStaysAVideoAndCarriesTheSequence() {
        var meta = FileMetadata()
        meta.sequenceID = "seq-7"
        meta.sequenceName = "morning monologue"
        meta.sequenceChunks = 2
        let file = CatalogBackfill.FileFacts(
            id: "a1", kind: "video", filename: "clip.mp4", absolutePath: "/a/clip.mp4",
            sha256: "s", fileSize: 10, createdAt: Date(), realm: .shared)

        let row = CatalogBackfill.row(file: file, existing: nil, meta: meta, source: nil)

        XCTAssertEqual(row.kind, "video", "a Director render is still a video")
        XCTAssertEqual(row.sequenceID, "seq-7")
        XCTAssertEqual(row.sequenceChunks, 2)
    }

    func testARerenderOntoTheSamePathAdoptsTheNewSequence() {
        // Unlike the generation facts, the sidecar wins over the stored row:
        // re-rendering a sequence onto the same path makes the NEW sidecar the
        // truth about what that clip is.
        var meta = FileMetadata()
        meta.sequenceID = "seq-new"
        let file = CatalogBackfill.FileFacts(
            id: "a1", kind: "video", filename: "clip.mp4", absolutePath: "/a/clip.mp4",
            sha256: "s", fileSize: 10, createdAt: Date(), realm: .shared)
        let existing = CatalogAsset(
            id: "a1", kind: "video", filename: "clip.mp4", absolutePath: "/a/clip.mp4",
            sequenceID: "seq-old")

        let row = CatalogBackfill.row(file: file, existing: existing, meta: meta, source: nil)

        XCTAssertEqual(row.sequenceID, "seq-new")
    }

    func testASealedRowDropsTheAuthoredNameButKeepsTheId() {
        // The name carries the timeline's title, so it follows the same rule as
        // the prompts; the id and chunk count are facets.
        let row = CatalogAsset(
            id: "a1", kind: "video", filename: "c.mp4", absolutePath: "/a/c.mp4",
            sealed: true, sequenceID: "seq-7", sequenceName: "morning monologue",
            sequenceChunks: 2)
        XCTAssertNil(row.sequenceName)
        XCTAssertEqual(row.sequenceID, "seq-7")
        XCTAssertEqual(row.sequenceChunks, 2)
    }
}
