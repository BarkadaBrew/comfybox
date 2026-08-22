import XCTest
@testable import ZImage

final class PresetStoreTests: XCTestCase {

  private func makeTempPath() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-preset-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir.appendingPathComponent("presets.json")
  }

  private func sample(id: String = "portrait", name: String = "Portrait") -> ImagePreset {
    ImagePreset(
      id: id,
      name: name,
      description: "A portrait preset",
      mediaKind: "image",
      provider: "local",
      engine: "zimage",
      mode: "z-image-turbo",
      model: "z-image-turbo",
      prompt: "a person",
      negativePrompt: "cropped",
      injectedKeywords: ["cinematic"],
      steps: 9,
      guidance: 1.0,
      seed: 42,
      width: 1280,
      height: 1280,
      loras: [LoraReference(filename: "moody.safetensors", scale: 0.8)],
      scheduler: "euler"
    )
  }

  // MARK: - Seeding / defaults

  func testFirstRunSeedsDefaultPresetsAndPersists() throws {
    let path = try makeTempPath()
    let store = PresetStore(path: path)
    let ids = store.list().map(\.id)
    XCTAssertEqual(ids, ["zimage-chat", "schnell-hq"])
    // The seed was written to disk, so a fresh store over the same file sees the same data.
    let reopened = PresetStore(path: path)
    XCTAssertEqual(reopened.list().map(\.id), ids)
  }

  func testSeedDefaultsCanBeDisabled() throws {
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)
    XCTAssertTrue(store.list().isEmpty)
  }

  // MARK: - CRUD

  func testUpsertInsertAndReplace() throws {
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)

    let inserted = try store.upsert(sample())
    XCTAssertEqual(inserted.id, "portrait")
    XCTAssertEqual(store.list().count, 1)
    XCTAssertEqual(store.get("portrait")?.name, "Portrait")

    // Same id → replace, not append.
    var edited = sample()
    edited.name = "Portrait v2"
    edited.steps = 12
    try store.upsert(edited)
    XCTAssertEqual(store.list().count, 1)
    XCTAssertEqual(store.get("portrait")?.name, "Portrait v2")
    XCTAssertEqual(store.get("portrait")?.steps, 12)
  }

  func testGetMissingReturnsNil() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    XCTAssertNil(store.get("nope"))
  }

  func testDeleteRemovesAndReportsChange() throws {
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)
    try store.upsert(sample())

    XCTAssertTrue(try store.delete("portrait"))
    XCTAssertNil(store.get("portrait"))
    XCTAssertFalse(try store.delete("portrait")) // second delete is a no-op

    // Deletion persisted.
    XCTAssertTrue(PresetStore(path: path, seedDefaults: false).list().isEmpty)
  }

  // MARK: - Validation

  func testUpsertRejectsEmptyId() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    XCTAssertThrowsError(try store.upsert(ImagePreset(id: "  ", name: "X"))) { error in
      XCTAssertEqual(error as? PresetStoreError, .validation(#"required field "id" is missing or empty"#))
    }
  }

  func testUpsertRejectsNonPositiveDimension() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    var bad = sample()
    bad.width = 0
    XCTAssertThrowsError(try store.upsert(bad))
  }

  func testUpsertRejectsEmptyLoraFilename() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    var bad = sample()
    bad.loras = [LoraReference(filename: "", scale: 0.5)]
    XCTAssertThrowsError(try store.upsert(bad))
  }

  // MARK: - Resolve / merge

  func testResolveMergesPresetOntoDefaults() throws {
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)
    // Sparse preset: only a couple of fields set; the rest must come from defaults.
    try store.upsert(ImagePreset(id: "sparse", name: "Sparse", engine: "mflux", steps: 20))

    let resolved = try store.resolve("sparse")
    XCTAssertEqual(resolved.steps, 20)          // from preset
    XCTAssertEqual(resolved.width, 512)         // from defaults
    XCTAssertEqual(resolved.height, 512)        // from defaults
    XCTAssertEqual(resolved.provider, "local")  // from defaults
    XCTAssertEqual(resolved.engine, "mflux")    // from preset
    XCTAssertEqual(resolved.mediaKind, "image") // from defaults
    XCTAssertEqual(resolved.injectedKeywords, [])
    XCTAssertNil(resolved.seed)
  }

  func testResolvePrefersPresetValuesOverDefaults() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    try store.upsert(sample())
    let resolved = try store.resolve("portrait")
    XCTAssertEqual(resolved.steps, 9)
    XCTAssertEqual(resolved.width, 1280)
    XCTAssertEqual(resolved.height, 1280)
    XCTAssertEqual(resolved.guidance, 1.0)
    XCTAssertEqual(resolved.seed, 42)
    XCTAssertEqual(resolved.model, "z-image-turbo")
    XCTAssertEqual(resolved.injectedKeywords, ["cinematic"])
    XCTAssertEqual(resolved.loras, [LoraReference(filename: "moody.safetensors", scale: 0.8)])
    XCTAssertEqual(resolved.scheduler, "euler")
  }

  func testResolveHonorsCustomDefaults() throws {
    let path = try makeTempPath()
    let custom = PresetDefaults(provider: "replicate", engine: "mflux", steps: 30, width: 1024, height: 1536, guidance: 3.5)
    let store = PresetStore(path: path, defaults: custom, seedDefaults: false)
    try store.upsert(ImagePreset(id: "bare", name: "Bare"))
    let resolved = try store.resolve("bare")
    XCTAssertEqual(resolved.steps, 30)
    XCTAssertEqual(resolved.width, 1024)
    XCTAssertEqual(resolved.height, 1536)
    XCTAssertEqual(resolved.provider, "replicate")
    XCTAssertEqual(resolved.guidance, 3.5)
  }

  func testResolveMissingThrowsNotFound() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    XCTAssertThrowsError(try store.resolve("ghost")) { error in
      XCTAssertEqual(error as? PresetStoreError, .notFound("ghost"))
    }
  }

  // MARK: - Round-trip / tolerant decode

  func testPresetCodableRoundTrip() throws {
    let original = sample()
    let data = try JSONEncoder().encode(original)
    let decoded = try JSONDecoder().decode(ImagePreset.self, from: data)
    XCTAssertEqual(original, decoded)
  }

  func testStoreRoundTripThroughDisk() throws {
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)
    try store.upsert(sample(id: "a", name: "A"))
    try store.upsert(sample(id: "b", name: "B"))

    let reopened = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(reopened.list(), store.list())
  }

  func testTolerantDecodeOfPartialPreset() throws {
    // Only id + name present; everything else must default without throwing.
    let json = Data(#"{ "id": "min", "name": "Minimal" }"#.utf8)
    let preset = try JSONDecoder().decode(ImagePreset.self, from: json)
    XCTAssertEqual(preset.id, "min")
    XCTAssertEqual(preset.description, "")
    XCTAssertNil(preset.steps)
    XCTAssertEqual(preset.loras, [])
    // Resolving a bare preset yields the full default parameter set.
    let resolved = ResolvedPreset(preset: preset)
    XCTAssertEqual(resolved.steps, 4)
    XCTAssertEqual(resolved.width, 512)
  }

  func testDecodeAcceptsBareArrayEnvelope() throws {
    let path = try makeTempPath()
    // Write a legacy bare-array file (no { presets: ... } wrapper).
    let bare = try JSONEncoder().encode([sample(id: "legacy", name: "Legacy")])
    try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try bare.write(to: path)

    let store = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(store.list().map(\.id), ["legacy"])
  }

  func testDecodeGarbageFallsBackToEmpty() {
    XCTAssertTrue(PresetStore.decode(Data("not json".utf8)).isEmpty)
  }

  // MARK: - Legacy image-service import

  private func writeLegacyPreset(_ json: String, named name: String, in dir: URL) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data(json.utf8).write(to: dir.appendingPathComponent(name))
  }

  func testImportLegacyImageServicePresets() throws {
    let path = try makeTempPath()
    let legacyDir = path.deletingLastPathComponent().appendingPathComponent("legacy", isDirectory: true)
    // Two legacy presets: one with LoRAs (path/scale), one bare.
    try writeLegacyPreset(#"""
    {
      "id": "cs-nsfw", "name": "CoffeeShop NSFW",
      "description": "Explicit content preset with character LoRAs.",
      "model": "Tongyi-MAI/Z-Image-Turbo-BF16",
      "steps": 16, "guidance": 3.5, "width": 1024, "height": 1024,
      "loras": [{"path": "cs-bree.safetensors", "scale": 0.85}],
      "injectedKeywords": "cinematic, detailed",
      "negativePrompt": "blurry, child"
    }
    """#, named: "cs-nsfw.json", in: legacyDir)
    try writeLegacyPreset(#"""
    {
      "id": "cs-control", "name": "Control Test", "description": "Bare ZIT.",
      "model": "z-image-turbo-bf16", "steps": 12, "guidance": 5.0,
      "width": 1024, "height": 1024, "loras": [], "injectedKeywords": "", "negativePrompt": ""
    }
    """#, named: "cs-control.json", in: legacyDir)

    let store = PresetStore(path: path, seedDefaults: false)
    let imported = store.importLegacyImageService(from: legacyDir)
    XCTAssertEqual(imported, 2)

    // Prefixed id avoids clobbering built-ins; fields mapped correctly.
    let nsfw = store.get("imported-cs-nsfw")
    XCTAssertEqual(nsfw?.name, "CoffeeShop NSFW")
    XCTAssertEqual(nsfw?.model, "Tongyi-MAI/Z-Image-Turbo-BF16")
    XCTAssertEqual(nsfw?.steps, 16)
    XCTAssertEqual(nsfw?.width, 1024)
    XCTAssertEqual(nsfw?.loras.first?.filename, "cs-bree.safetensors")  // path -> filename
    XCTAssertEqual(nsfw?.loras.first?.scale, 0.85)
    XCTAssertEqual(nsfw?.injectedKeywords, ["cinematic", "detailed"])   // string -> [keywords]
    XCTAssertEqual(nsfw?.negativePrompt, "blurry, child")
    XCTAssertEqual(nsfw?.engine, "zimage")

    // Empty injectedKeywords/negativePrompt become nil/empty, not [""] / "".
    let control = store.get("imported-cs-control")
    XCTAssertEqual(control?.loras.count, 0)
    XCTAssertTrue((control?.injectedKeywords ?? []).isEmpty)

    // Idempotent: a second import changes nothing (already present).
    XCTAssertEqual(store.importLegacyImageService(from: legacyDir), 0)
    XCTAssertEqual(store.list().count, 2)
  }

  func testImportLegacyMissingDirectoryIsNoOp() throws {
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(store.importLegacyImageService(
      from: URL(fileURLWithPath: "/nope/presets")), 0)
    XCTAssertTrue(store.list().isEmpty)
  }
  // MARK: - videoTuning round-trip (2026-08-07)

  /// `videoTuning` existed on the struct but was MISSING from CodingKeys, so
  /// the custom decoder and the synthesized encoder both dropped it: every
  /// preset-level Tier-A tuning write since task #9 Phase 2 silently vanished
  /// on the JSON/API path. The desktop tuning UI wrote values nothing read.
  func testVideoTuningSurvivesJSONRoundTrip() throws {
    let json = """
    {"id":"t","name":"T","videoTuning":{"imgCompression":22,"condFps":18}}
    """.data(using: .utf8)!
    let decoded = try JSONDecoder().decode(ImagePreset.self, from: json)
    XCTAssertEqual(decoded.videoTuning?.imgCompression, 22, "decode must carry videoTuning")
    XCTAssertEqual(decoded.videoTuning?.condFps, 18)
    let encoded = try JSONEncoder().encode(decoded)
    let redecoded = try JSONDecoder().decode(ImagePreset.self, from: encoded)
    XCTAssertEqual(redecoded.videoTuning?.imgCompression, 22, "encode must carry videoTuning")
  }

}

// MARK: - WP-E9 (FDD §3.9, AC-58 slice): `vae` on ImagePreset at every site

extension PresetStoreTests {

  /// The `videoTuning` regression class: a field must survive the custom
  /// decoder, the synthesized encoder, the memberwise init, the store and
  /// `ResolvedPreset`. WP-E20 widens this to the nine new fields.
  func testVAEFieldRoundTrip() throws {
    let wan = "/Users/toddwalderman/LocalModels/vae/Wan2_1_VAE_fp32.safetensors"
    // Memberwise init.
    let preset = ImagePreset(id: "ref", name: "Reference", vae: wan)
    XCTAssertEqual(preset.vae, wan)
    // JSON round trip (decoder + encoder).
    let data = try JSONEncoder().encode(preset)
    XCTAssertTrue(String(data: data, encoding: .utf8)!.contains("\"vae\""), "encoder dropped vae")
    let decoded = try JSONDecoder().decode(ImagePreset.self, from: data)
    XCTAssertEqual(decoded.vae, wan)
    XCTAssertEqual(decoded, preset)
    // Wire decode from a hand-written document.
    let fromJSON = try JSONDecoder().decode(
      ImagePreset.self, from: Data(#"{"id":"r","name":"R","vae":"\#(wan)"}"#.utf8))
    XCTAssertEqual(fromJSON.vae, wan)
    // Absent stays nil — Wan is never ambient (D16).
    XCTAssertNil(try JSONDecoder().decode(ImagePreset.self, from: Data(#"{"id":"r","name":"R"}"#.utf8)).vae)
    // Store + resolve.
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    try store.upsert(preset)
    let resolved = try store.resolve("ref")
    XCTAssertEqual(resolved.vae, wan)
    XCTAssertNil(ResolvedPreset(preset: ImagePreset(id: "p", name: "P")).vae)
  }
}
