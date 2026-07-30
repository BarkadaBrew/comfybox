import XCTest
@testable import ComfyBoxCatalog

final class MetadataReaderTests: XCTestCase {

    /// Verbatim shape of a real EXIF:UserComment written by the engine.
    func testParsesEngineUserCommentJSON() throws {
        let json = """
        {"width":896,"prompt":"a woman by a window","loras":[{"name":"KNPV4.1_pre","scale":1},\
        {"scale":0.35,"name":"Filipina_Pinay_Women"}],"seed":2090500631,"steps":9,\
        "height":1664,"guidance":0,"model":"krea-2-turbo"}
        """
        let m = MetadataReader.parseUserComment(json)
        XCTAssertEqual(m.prompt, "a woman by a window")
        XCTAssertEqual(m.seed, 2090500631)
        XCTAssertEqual(m.steps, 9)
        XCTAssertEqual(m.width, 896)
        XCTAssertEqual(m.height, 1664)
        XCTAssertEqual(m.guidance, 0)
        XCTAssertEqual(m.modelFamily, "krea-2-turbo")
        XCTAssertNotNil(m.loras)
        XCTAssertTrue(m.loras!.contains("KNPV4.1_pre"))
    }

    func testGarbageUserCommentYieldsEmptyMetadataNotACrash() {
        let m = MetadataReader.parseUserComment("not json at all {{{")
        XCTAssertNil(m.prompt)
        XCTAssertNil(m.seed)
    }

    /// Verbatim shape of a real IMAGE sidecar from ~/.kira/studio/metadata.
    func testParsesImageSidecar() throws {
        let json = """
        {"character":"kira","category":"generated","provider":"comfybox","tier":"standard",
         "generated_at":"2026-07-22T05:35:45.468Z","durationMs":84944,"width":576,"height":1024,
         "seed":1387857967,"content_mode":"avocado","model":"krea2","preset":"krea-kira",
         "guidance":0,"sealed":false,
         "loras":[{"path":"KNPV4.1_pre.safetensors","scale":1},{"path":"Filipina_Pinay_Women.safetensors","scale":0.35}]}
        """
        let m = try XCTUnwrap(MetadataReader.readSidecar(jsonData: Data(json.utf8)))
        XCTAssertEqual(m.characterName, "kira")
        XCTAssertEqual(m.contentMode, "avocado")
        XCTAssertEqual(m.preset, "krea-kira")
        XCTAssertEqual(m.modelFamily, "krea2")
        XCTAssertEqual(m.seed, 1387857967)
        XCTAssertFalse(m.sealed)
        XCTAssertEqual(m.provider, "comfybox")
        XCTAssertNotNil(m.loras, "the path-shaped loras array must round-trip too, not just the name-shaped one")
        XCTAssertTrue(m.loras!.contains("KNPV4.1_pre.safetensors"))
        XCTAssertTrue(m.loras!.contains("Filipina_Pinay_Women.safetensors"))
    }

    /// Verbatim shape of a real VIDEO sidecar — note source_image and prompt_raw.
    func testParsesVideoSidecarIncludingSourceImage() throws {
        let json = """
        {"character":"bree","mode":"i2v","duration":null,"provider":"comfybox","model":"ltx",
         "resolution":"480p","aspect_ratio":"9:16","content_mode":"avocado",
         "generated_at":"2026-07-11T12:17:40.506Z","prompt":"optimized text","prompt_raw":"original text",
         "source_image":"/home/todd/.kira/studio/gallery/Bree/generated/1783770983068_x.png"}
        """
        let m = try XCTUnwrap(MetadataReader.readSidecar(jsonData: Data(json.utf8)))
        XCTAssertEqual(m.mode, "i2v")
        XCTAssertEqual(m.resolution, "480p")
        XCTAssertEqual(m.aspectRatio, "9:16")
        XCTAssertEqual(m.prompt, "optimized text")
        XCTAssertEqual(m.promptRaw, "original text")
        XCTAssertEqual(m.sourceImagePath,
                       "/home/todd/.kira/studio/gallery/Bree/generated/1783770983068_x.png")
        XCTAssertEqual(m.provider, "comfybox",
                       "video has no embedded metadata at all — the sidecar's provider is the ONLY source of `source` for video")
        XCTAssertNil(m.durationMs, "sidecar duration is null — must NOT be invented")
    }

    /// Same asymmetry as above, but proves the field is IGNORED rather than
    /// merely absent: even when the sidecar supplies a real, non-null
    /// duration, readSidecar must not surface it. Duration must always come
    /// from `probeContainer` (the container itself), because real sidecars
    /// are unreliable for it — this is the load-bearing assertion for that
    /// design decision; without it, a regression that started trusting the
    /// sidecar's duration would go undetected by the null-duration test alone.
    func testSidecarDurationIsIgnoredEvenWhenNotNull() throws {
        let json = """
        {"character":"bree","mode":"i2v","duration":12.5,"provider":"comfybox","model":"ltx",
         "resolution":"480p","aspect_ratio":"9:16","content_mode":"avocado"}
        """
        let m = try XCTUnwrap(MetadataReader.readSidecar(jsonData: Data(json.utf8)))
        XCTAssertNil(m.durationMs,
                      "sidecar duration must be ignored even when present — container probe is the only trusted source")
    }

    func testSealedSidecarIsFlagged() throws {
        let json = #"{"character":"bree","sealed":true,"content_mode":"apple"}"#
        let m = try XCTUnwrap(MetadataReader.readSidecar(jsonData: Data(json.utf8)))
        XCTAssertTrue(m.sealed)
    }

    func testUnreadableSidecarReturnsNilRatherThanThrowing() {
        XCTAssertNil(MetadataReader.readSidecar(jsonData: Data("]]not json[[".utf8)))
        XCTAssertNil(MetadataReader.readSidecar(jsonData: Data()))
    }

    func testSidecarPathMirrorsMediaPath() {
        XCTAssertEqual(
            MetadataReader.sidecarPath(
                forMedia: "/home/todd/.kira/studio/gallery/Kira/generated/x.png",
                galleryRoot: "/home/todd/.kira/studio/gallery",
                metadataRoot: "/home/todd/.kira/studio/metadata"),
            "/home/todd/.kira/studio/metadata/Kira/generated/x.json")
    }

    /// A backfill runs `runTool` over several thousand files in sequence; one
    /// hung child process (corrupt file, stalled filesystem, network-mounted
    /// media) must not stall the whole run. Uses an injected short timeout
    /// (not the production 20s constant) so the suite stays fast, against a
    /// deliberately slow command that would otherwise still be sleeping when
    /// this assertion runs.
    func testRunToolTimesOutRatherThanHangingForever() {
        let start = Date()
        let result = MetadataReader.runTool("/bin/sh", ["-c", "sleep 5; echo done"], timeout: 0.2)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertNil(result, "a hung/slow process must degrade to nil, never block forever")
        XCTAssertLessThan(elapsed, 2.0,
                           "the timeout must terminate the process well before its natural 5s completion")
    }
}
