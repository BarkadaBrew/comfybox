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
}
