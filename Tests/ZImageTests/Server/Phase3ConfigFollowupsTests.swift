// Phase3ConfigFollowupsTests.swift — comfybox#324 (Phase 3 config follow-ups
// from the ad8d01b adversarial review, F3/F4). F5/F6 are covered elsewhere
// (F6 in ComfyBoxServerConfigTests.swift; F5 has no Swift-server-side unit —
// see PR body).
//
// F3: the config-defaults `??` chain at the real generate call sites
// (`makePipelineRequest`/`makeImg2ImgRequest`, `runKrea2Generate`, LTX-2's
// `prepareLocalVideo`) was previously covered only by tests that RE-IMPLEMENTED
// the chain inline (see `ServerConfigStoreTests.
// testMigrationIsVideoEngineNeutralDespiteDesktopValues`'s local
// `let reqWidth: Int? = nil, ...` shadows) — a future edit to the real chain
// could drift from those tests without either one failing. This file drives
// the REAL call sites instead: `makePipelineRequest`/`makeImg2ImgRequest`
// directly (they need no live model), and the pure merge functions extracted
// from `runKrea2Generate`/`prepareLocalVideo` (which DO need a loaded
// pipeline past this point, so cannot run end-to-end in a unit test).
//
// F4: `WarmServer.swift` must read config through `ServerConfigStore`
// (the one authoritative in-memory document PUT/PATCH write through), never
// via a direct `ComfyBoxServerConfig.loadOrMigrate()` disk read — a source
// scan (mirrors `RouteTaskExecutorTests`' render-call-chain scan) so a future
// PR can't silently reintroduce a bypass.

import XCTest
@testable import ZImage

final class Phase3ConfigFollowupsTests: XCTestCase {

  // MARK: - F4: WarmServer.swift must not read config around the store

  private static let repoRoot: URL = {
    // <root>/Tests/ZImageTests/Server/Phase3ConfigFollowupsTests.swift
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }()

  func testWarmServerNeverReadsConfigDirectlyBypassingTheStore() throws {
    let url = ControlSurfaceParser.warmServerSource(repoRoot: Self.repoRoot)
    let source = try String(contentsOf: url, encoding: .utf8)
    XCTAssertFalse(
      source.contains("ComfyBoxServerConfig.loadOrMigrate()"),
      "WarmServer.swift must read config via ServerConfigStore.shared.current().config, "
        + "never a direct ComfyBoxServerConfig.loadOrMigrate() disk read (comfybox#324, F4) — "
        + "a direct read can be clobbered by / clobber a concurrent PATCH's out-of-band save")
  }

  // MARK: - F3: makePipelineRequest / makeImg2ImgRequest (flux1) — the real call site

  private func configuration() -> WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  func testMakePipelineRequestAppliesInjectedConfigDefaultsWhenRequestOmitsThem() throws {
    let payload = GeneratePayload(prompt: "x")
    let configDefaults = RenderDefaultValues(width: 640, height: 960, steps: 15, guidance: 2.5)

    let request = try payload.makePipelineRequest(
      configuration: configuration(), activeLoRAs: [], configDefaults: configDefaults)

    XCTAssertEqual(request.width, 640, "config width must apply when the request omits it")
    XCTAssertEqual(request.height, 960)
    XCTAssertEqual(request.steps, 15)
    XCTAssertEqual(request.guidanceScale, 2.5, accuracy: 1e-6)
  }

  func testMakePipelineRequestRequestValuesStillWinOverConfigDefaults() throws {
    let payload = GeneratePayload(prompt: "x", width: 1536, height: 1536, steps: 30, guidance: 4.0)
    let configDefaults = RenderDefaultValues(width: 640, height: 960, steps: 15, guidance: 2.5)

    let request = try payload.makePipelineRequest(
      configuration: configuration(), activeLoRAs: [], configDefaults: configDefaults)

    XCTAssertEqual(request.width, 1536, "an explicit request value must win over the config default")
    XCTAssertEqual(request.height, 1536)
    XCTAssertEqual(request.steps, 30)
    XCTAssertEqual(request.guidanceScale, 4.0, accuracy: 1e-6)
  }

  func testMakePipelineRequestFallsBackToEngineConstantWhenBothAreAbsent() throws {
    let payload = GeneratePayload(prompt: "x")

    let request = try payload.makePipelineRequest(
      configuration: configuration(), activeLoRAs: [], configDefaults: RenderDefaultValues())

    XCTAssertEqual(request.width, ZImageModelMetadata.recommendedWidth)
    XCTAssertEqual(request.height, ZImageModelMetadata.recommendedHeight)
    XCTAssertEqual(request.steps, ZImageModelMetadata.recommendedInferenceSteps)
    XCTAssertEqual(request.guidanceScale, ZImageModelMetadata.recommendedGuidanceScale, accuracy: 1e-6)
  }

  func testMakeImg2ImgRequestAppliesInjectedConfigStepsAndGuidance() throws {
    let payload = GeneratePayload(
      prompt: "x", imagePath: "/tmp/source.png", imageStrength: 0.5)
    let configDefaults = RenderDefaultValues(steps: 22, guidance: 3.5)

    let request = try payload.makeImg2ImgRequest(
      configuration: configuration(), activeLoRAs: [], configDefaults: configDefaults)

    XCTAssertEqual(request.steps, 22)
    XCTAssertEqual(request.guidanceScale, 3.5, accuracy: 1e-6)
  }

  // MARK: - F3: runKrea2Generate's config merge — the real (extracted) chain

  func testMergedKrea2RenderDefaultsAppliesConfigWhenRequestOmitsIt() {
    let merged = mergedKrea2RenderDefaults(
      requestWidth: nil, requestHeight: nil, requestSteps: nil, requestGuidance: nil,
      configDefaults: RenderDefaultValues(width: 1280, height: 720, steps: 18, guidance: 1.5))

    XCTAssertEqual(merged.width, 1280)
    XCTAssertEqual(merged.height, 720)
    XCTAssertEqual(merged.steps, 18)
    XCTAssertEqual(merged.guidance, 1.5)
  }

  func testMergedKrea2RenderDefaultsRequestWinsOverConfig() {
    let merged = mergedKrea2RenderDefaults(
      requestWidth: 512, requestHeight: 512, requestSteps: 9, requestGuidance: 1.0,
      configDefaults: RenderDefaultValues(width: 1280, height: 720, steps: 18, guidance: 1.5))

    XCTAssertEqual(merged.width, 512)
    XCTAssertEqual(merged.height, 512)
    XCTAssertEqual(merged.steps, 9)
    XCTAssertEqual(merged.guidance, 1.0)
  }

  func testMergedKrea2RenderDefaultsFallsBackTo1024WhenBothDimsAreAbsent() {
    let merged = mergedKrea2RenderDefaults(
      requestWidth: nil, requestHeight: nil, requestSteps: nil, requestGuidance: nil,
      configDefaults: RenderDefaultValues())

    XCTAssertEqual(merged.width, 1024)
    XCTAssertEqual(merged.height, 1024)
    XCTAssertNil(merged.steps, "krea2 steps are variant-dependent — nil must reach Krea2Variant.resolvedSteps unchanged")
    XCTAssertNil(merged.guidance)
  }

  // MARK: - F3: LTX-2 prep (prepareLocalVideo) — the real (extracted) chain

  func testResolvedLTX2RequestDimsAppliesConfigWhenEverythingElseIsAbsent() {
    let dims = WarmServer.resolvedLTX2RequestDims(
      requestWidth: nil, requestHeight: nil,
      namedWidth: nil, namedHeight: nil,
      presetWidth: nil, presetHeight: nil,
      videoConfigDefaults: VideoDefaultValues(width: 960, height: 544))

    XCTAssertEqual(dims.width, 960)
    XCTAssertEqual(dims.height, 544)
  }

  func testResolvedLTX2RequestDimsPriorityRequestOverNamedOverPresetOverConfig() {
    // Request wins outright.
    let requestWins = WarmServer.resolvedLTX2RequestDims(
      requestWidth: 1280, requestHeight: 720,
      namedWidth: 704, namedHeight: 480,
      presetWidth: 960, presetHeight: 544,
      videoConfigDefaults: VideoDefaultValues(width: 640, height: 360))
    XCTAssertEqual(requestWins.width, 1280)
    XCTAssertEqual(requestWins.height, 720)

    // No request -> named resolution wins over preset/config.
    let namedWins = WarmServer.resolvedLTX2RequestDims(
      requestWidth: nil, requestHeight: nil,
      namedWidth: 704, namedHeight: 480,
      presetWidth: 960, presetHeight: 544,
      videoConfigDefaults: VideoDefaultValues(width: 640, height: 360))
    XCTAssertEqual(namedWins.width, 704)
    XCTAssertEqual(namedWins.height, 480)

    // No request/named -> preset wins over config.
    let presetWins = WarmServer.resolvedLTX2RequestDims(
      requestWidth: nil, requestHeight: nil,
      namedWidth: nil, namedHeight: nil,
      presetWidth: 960, presetHeight: 544,
      videoConfigDefaults: VideoDefaultValues(width: 640, height: 360))
    XCTAssertEqual(presetWins.width, 960)
    XCTAssertEqual(presetWins.height, 544)
  }

  func testResolvedLTX2RequestDimsFallsBackToEngineDefaultWhenAllAbsent() {
    let dims = WarmServer.resolvedLTX2RequestDims(
      requestWidth: nil, requestHeight: nil,
      namedWidth: nil, namedHeight: nil,
      presetWidth: nil, presetHeight: nil,
      videoConfigDefaults: VideoDefaultValues())

    XCTAssertEqual(dims.width, 704, "704x448 is the LTX-2 engine constant, bit-identical to pre-migration")
    XCTAssertEqual(dims.height, 448)
  }

  func testResolvedLTX2FramesAppliesConfigWhenRequestOmitsIt() {
    XCTAssertEqual(
      WarmServer.resolvedLTX2Frames(requestFrames: nil, videoConfigDefaults: VideoDefaultValues(frames: 121)),
      121)
  }

  func testResolvedLTX2FramesRequestWinsOverConfig() {
    XCTAssertEqual(
      WarmServer.resolvedLTX2Frames(requestFrames: 49, videoConfigDefaults: VideoDefaultValues(frames: 121)),
      49)
  }

  func testResolvedLTX2FramesFallsBackTo97WhenBothAreAbsent() {
    XCTAssertEqual(WarmServer.resolvedLTX2Frames(requestFrames: nil, videoConfigDefaults: VideoDefaultValues()), 97)
  }
}
