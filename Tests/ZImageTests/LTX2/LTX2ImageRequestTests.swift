import XCTest
@testable import ZImage

final class LTX2ImageRequestTests: XCTestCase {
  private func decode(_ json: String) throws -> GeneratePayload {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  private func makeStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("ltx2-image-presets-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  private var configuration: WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  private func tierPreset(
    _ tier: String, sampler: String, loras: [LoraReference]
  ) -> ImagePreset {
    ImagePreset(
      id: "ltx-image-\(tier)", name: "LTX Image — \(tier.capitalized)",
      mediaKind: "image", provider: "local", engine: "ltx2",
      negativePrompt: tier == "avocado" ? "low quality" : nil,
      contentMode: tier, steps: 8, guidance: 1, seed: 42,
      width: 1280, height: 704, loras: loras,
      scheduler: sampler, sampler: sampler, sigmaSchedule: "flow")
  }

  private func expand(_ json: String, store: PresetStore) throws -> GeneratePayload {
    try WarmServer.decodedGeneratePayload(
      from: Data(json.utf8), store: store, configuration: configuration,
      loraExists: { _ in true })
  }

  func testLTXEngineDecodesAndValidatesNativeDefaults() throws {
    let payload = try decode(#"{"prompt":"a butterfly","engine":"ltx2"}"#)
    XCTAssertTrue(payload.usesLTX2ImageEngine)
    XCTAssertNoThrow(try payload.validateEngine())
    XCTAssertNoThrow(try payload.validateLTX2ImageFields())
  }

  func testDefaultEngineDoesNotSelectLTX() throws {
    let payload = try decode(#"{"prompt":"a butterfly","engine":"default"}"#)
    XCTAssertFalse(payload.usesLTX2ImageEngine)
    XCTAssertNoThrow(try payload.validateEngine())
  }

  func testLTXImageRejectsInvalidSpatialGrid() throws {
    let payload = try decode(
      #"{"prompt":"a butterfly","engine":"ltx2","width":1279,"height":704}"#)
    XCTAssertThrowsError(try payload.validateLTX2ImageFields())
  }

  func testLTXImageRejectsImg2ImgFieldsUntilNativePathIsWired() throws {
    let payload = try decode(
      #"{"prompt":"change it","engine":"ltx2","image_path":"/tmp/source.png"}"#)
    XCTAssertThrowsError(try payload.validateLTX2ImageFields())
  }

  func testServerFramePolicyPreservesTheNativeSingleFrameRequest() {
    XCTAssertEqual(
      WarmServer.resolvedLTX2Frames(
        requestFrames: 1, videoConfigDefaults: VideoDefaultValues(frames: 121)),
      1)
  }

  func testLTXImageAcceptsNamedPresetForRouteExpansion() throws {
    let payload = try decode(
      #"{"prompt":"a butterfly","engine":"ltx2","preset":"portrait"}"#)
    XCTAssertNoThrow(try payload.validateLTX2ImageFields())
  }

  func testEveryImplementedLTXImageSamplerValidates() throws {
    for sampler in LTX2ImageRecipe.samplerNames {
      let payload = try decode(
        #"{"prompt":"a butterfly","engine":"ltx2","sampler":"\#(sampler)","sigma_schedule":"flow"}"#)
      XCTAssertNoThrow(try payload.validateLTX2ImageFields(), sampler)
    }
  }

  func testLTXImageRejectsSamplerThatTheDenoisingLoopDoesNotCall() throws {
    let payload = try decode(
      #"{"prompt":"a butterfly","engine":"ltx2","sampler":"res_2s"}"#)
    XCTAssertThrowsError(try payload.validateLTX2ImageFields())
  }

  func testFourTierPresetsExpandThroughTheGenerateRoute() throws {
    let store = try makeStore()
    let reasoning = LoraReference(filename: "ltx-reasoning.safetensors", scale: 1)
    let sensual = LoraReference(filename: "ltx-sensual.safetensors", scale: 0.7)
    let fixtures: [(String, String, [LoraReference])] = [
      ("neutral", "euler", []),
      ("apple", "euler", [reasoning]),
      ("banana", "euler_ancestral", [reasoning, sensual]),
      ("avocado", "euler_ancestral_cfg_pp", [reasoning, sensual]),
    ]

    for (tier, sampler, loras) in fixtures {
      try store.upsert(tierPreset(tier, sampler: sampler, loras: loras))
      let payload = try expand(
        #"{"prompt":"portrait","engine":"ltx2","preset":"ltx-image-\#(tier)"}"#,
        store: store)
      XCTAssertEqual(payload.contentMode, tier)
      XCTAssertEqual(payload.width, 1280)
      XCTAssertEqual(payload.height, 704)
      XCTAssertEqual(payload.steps, 8)
      XCTAssertEqual(payload.guidance, 1)
      XCTAssertEqual(payload.seed, 42)
      XCTAssertEqual(payload.scheduler, sampler)
      XCTAssertEqual(payload.sigmaSchedule, "flow")
      XCTAssertEqual(payload.loras?.map(\.path), loras.map(\.filename))
      XCTAssertEqual(payload.presetStackApplied, true)
      XCTAssertEqual(payload.presetRecipeApplied, ["scheduler", "sigma_schedule"])
      XCTAssertNil(payload.presetUnresolved)
    }
  }

  func testExplicitLTXFieldsAndLoRAStackOverridePreset() throws {
    let store = try makeStore()
    try store.upsert(tierPreset(
      "banana", sampler: "euler_ancestral",
      loras: [LoraReference(filename: "preset.safetensors", scale: 0.7)]))

    let payload = try expand(#"""
      {"prompt":"portrait","engine":"ltx2","preset":"ltx-image-banana",
       "content_mode":"neutral","width":1024,"height":768,"steps":12,"guidance":2,
       "seed":99,"sampler":"euler_cfg_pp","sigma_schedule":"flow",
       "loras":[{"path":"request.safetensors","scale":0.5}]}
      """#, store: store)

    XCTAssertEqual(payload.contentMode, "neutral")
    XCTAssertEqual(payload.width, 1024)
    XCTAssertEqual(payload.height, 768)
    XCTAssertEqual(payload.steps, 12)
    XCTAssertEqual(payload.guidance, 2)
    XCTAssertEqual(payload.seed, 99)
    XCTAssertEqual(payload.scheduler, "euler_cfg_pp")
    XCTAssertEqual(payload.loras?.map(\.path), ["request.safetensors"])
    XCTAssertEqual(payload.presetStackMismatch, true)
    XCTAssertNil(payload.presetStackApplied)
    XCTAssertNil(payload.presetRecipeApplied)
  }

  func testExpandedLTXPresetIsFrozenForQueueReplayIncludingEmptyStack() throws {
    let store = try makeStore()
    try store.upsert(tierPreset("neutral", sampler: "euler", loras: []))
    let original = Data(
      #"{"prompt":"portrait","engine":"ltx2","preset":"ltx-image-neutral"}"#.utf8)
    let accepted = try WarmServer.decodedGeneratePayload(
      from: original, store: store, configuration: configuration,
      loraExists: { _ in true })
    let persisted = WarmServer.rawBody(original, expandedWith: accepted)
    let object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: persisted) as? [String: Any])

    XCTAssertEqual(object["width"] as? Int, 1280)
    XCTAssertEqual(object["height"] as? Int, 704)
    XCTAssertEqual(object["steps"] as? Int, 8)
    XCTAssertEqual(object["guidance"] as? Double, 1)
    XCTAssertEqual((object["seed"] as? NSNumber)?.uint64Value, 42)
    XCTAssertEqual(object["content_mode"] as? String, "neutral")
    XCTAssertEqual(object["scheduler"] as? String, "euler")
    XCTAssertEqual(object["sigma_schedule"] as? String, "flow")
    XCTAssertEqual((object["loras"] as? [[String: Any]])?.count, 0)
  }

  func testLTXPresetRoundTripsContentModeAndResolvesLoRAFamily() throws {
    let preset = tierPreset(
      "apple", sampler: "euler", loras: [LoraReference(filename: "reasoning.safetensors", scale: 1)])
    let data = try JSONEncoder().encode(preset)
    let decoded = try JSONDecoder().decode(ImagePreset.self, from: data)
    XCTAssertEqual(decoded.contentMode, "apple")
    XCTAssertEqual(PresetStore.resolvedLoRAFamily(for: decoded), "ltx")
  }

  func testLTXPresetValidationUsesItsOwnSamplerCatalog() throws {
    let store = try makeStore()
    XCTAssertThrowsError(try store.upsert(tierPreset("neutral", sampler: "res_2s", loras: [])))
    XCTAssertNoThrow(try store.upsert(tierPreset(
      "neutral", sampler: "euler_cfg_pp", loras: [])))
  }

  func testUnknownEngineFailsLoud() throws {
    let payload = try decode(#"{"prompt":"a butterfly","engine":"something-else"}"#)
    XCTAssertThrowsError(try payload.validateEngine())
  }
}
