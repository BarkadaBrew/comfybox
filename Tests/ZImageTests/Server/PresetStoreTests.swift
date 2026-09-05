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

  /// The Krea-2 r256 distill is an accelerator by declaration, not by its
  /// spelling. The canonical preset store must retain that declaration across
  /// both persistence and resolve; otherwise the daemon sees an ordinary
  /// style LoRA and refuses the raw-turbo recipe before ComfyBox is called.
  func testKrea2R256AcceleratorRoleSurvivesStoreAndResolve() throws {
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)
    let r256 = LoraReference(
      filename: "krea2_turbo_distill_r256.safetensors",
      scale: 0.6,
      role: "accel")
    let preset = ImagePreset(
      id: "krea-kira",
      name: "Krea Kira",
      mediaKind: "image",
      engine: "zimage",
      model: "krea2-raw",
      steps: 12,
      guidance: 1,
      width: 1024,
      height: 1024,
      loras: [r256],
      kroma: KromaPolicy(
        strength: 0.6,
        file: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors"))

    try store.upsert(preset)

    // Todd 2026-09-04: `kroma` is deprecated — `upsert` folds it into
    // `loras[]` as a regular, role-tagged entry (`ImagePreset.
    // migratingKromaDeprecation`). Review r2 (I4): inserted at the FRONT,
    // not appended — render parity with the pre-deprecation prepend
    // behavior for a preset migrating for the first time. The accel role
    // must survive that migration unchanged regardless of position.
    let migratedKroma = LoraReference(
      filename: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors", scale: 0.6, role: "kroma")
    let reopened = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(reopened.get("krea-kira")?.loras, [migratedKroma, r256])
    XCTAssertEqual(try reopened.resolve("krea-kira").loras, [migratedKroma, r256])

    let root = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
    let presets = try XCTUnwrap(root["presets"] as? [[String: Any]])
    let loras = try XCTUnwrap(presets.first?["loras"] as? [[String: Any]])
    XCTAssertEqual(loras.last?["role"] as? String, "accel")
  }

  func testPresetRejectsUnknownLoraRole() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    var bad = sample()
    bad.loras = [LoraReference(filename: "r256.safetensors", scale: 0.6, role: "turboish")]

    XCTAssertThrowsError(try store.upsert(bad)) { error in
      guard case PresetStoreError.validation(let message) = error else {
        return XCTFail("unexpected error: \(error)")
      }
      XCTAssertTrue(message.contains("loras[0].role"), message)
      XCTAssertTrue(message.contains("accel"), message)
    }
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

  func testRES4LYFKnobsRoundTripAndResolve() throws {
    let preset = ImagePreset(
      id: "res4lyf", name: "RES4LYF",
      noiseType: "fractal", noiseAlpha: -0.7, implicitSteps: 4, c2: 0.4)

    let data = try JSONEncoder().encode(preset)
    let text = try XCTUnwrap(String(data: data, encoding: .utf8))
    for key in ["noiseType", "noiseAlpha", "implicitSteps", "c2"] {
      XCTAssertTrue(text.contains("\"\(key)\""), "encoder dropped \(key): \(text)")
    }

    let decoded = try JSONDecoder().decode(ImagePreset.self, from: data)
    XCTAssertEqual(decoded.noiseType, "fractal")
    XCTAssertEqual(decoded.noiseAlpha, -0.7)
    XCTAssertEqual(decoded.implicitSteps, 4)
    XCTAssertEqual(decoded.c2, 0.4)

    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    try store.upsert(decoded)
    let resolved = try store.resolve("res4lyf")
    XCTAssertEqual(resolved.noiseType, "fractal")
    XCTAssertEqual(resolved.noiseAlpha, -0.7)
    XCTAssertEqual(resolved.implicitSteps, 4)
    XCTAssertEqual(resolved.c2, 0.4)

    let bare = ResolvedPreset(preset: ImagePreset(id: "bare", name: "Bare"))
    XCTAssertNil(bare.noiseType)
    XCTAssertNil(bare.noiseAlpha)
    XCTAssertNil(bare.implicitSteps)
    XCTAssertNil(bare.c2)
  }
}

// MARK: - WP-E20 (FDD §3.15, D14, AC-44b/44c, AC-58): the nine new ImagePreset fields + O4a on the engine

extension PresetStoreTests {

  /// AC-58 — all nine new fields (`checkpointFamily`, `kroma`, `vae`, `sampler`,
  /// `sigmaSchedule`, `shift`, `eta`, `bongmath`, `stage2`) survive memberwise
  /// init → encode → decode → store → `ResolvedPreset`, asserted field by field.
  /// The `videoTuning` regression class: one missing `CodingKeys` entry or one
  /// missing `decodeIfPresent` line and the field silently vanishes.
  func testNewImageFieldsRoundTrip() throws {
    let wan = "/Users/toddwalderman/LocalModels/vae/Wan2_1_VAE_fp32.safetensors"
    let stage2 = PresetStage(
      sampler: "dpmpp_2m", sigmaSchedule: "karras", steps: 2, denoise: 0.2, eta: 0.5, bongmath: false)
    let preset = ImagePreset(
      id: "ref", name: "Reference",
      model: "krea2-raw",
      steps: 6, guidance: 1.0,
      loras: [LoraReference(filename: "krea2_turbo_lora_rank_64_bf16.safetensors", scale: 0.6)],
      vae: wan,
      checkpointFamily: "raw-accel",
      kroma: KromaPolicy(strength: 0.3, file: "kroma-v0.2-base-lora-rank-384-fro-0985.safetensors"),
      sampler: "res_2s", sigmaSchedule: "beta", shift: 1.15, eta: 0.5, bongmath: false,
      stage2: stage2)

    // Memberwise init.
    XCTAssertEqual(preset.checkpointFamily, "raw-accel")
    XCTAssertEqual(preset.kroma, KromaPolicy(strength: 0.3, file: "kroma-v0.2-base-lora-rank-384-fro-0985.safetensors"))
    XCTAssertEqual(preset.vae, wan)
    XCTAssertEqual(preset.sampler, "res_2s")
    XCTAssertEqual(preset.sigmaSchedule, "beta")
    XCTAssertEqual(preset.shift, 1.15)
    XCTAssertEqual(preset.eta, 0.5)
    XCTAssertEqual(preset.bongmath, false)
    XCTAssertEqual(preset.stage2, stage2)

    // Encoder carries every key (CodingKeys site).
    let data = try JSONEncoder().encode(preset)
    let text = try XCTUnwrap(String(data: data, encoding: .utf8))
    for key in ["checkpointFamily", "kroma", "vae", "sampler", "sigmaSchedule", "shift", "eta", "bongmath", "stage2"] {
      XCTAssertTrue(text.contains("\"\(key)\""), "encoder dropped \(key): \(text)")
    }

    // Decoder carries every field (custom init(from:) site).
    let decoded = try JSONDecoder().decode(ImagePreset.self, from: data)
    XCTAssertEqual(decoded, preset)

    // Wire decode from a hand-written document (the §3.15 shape).
    let wire = try JSONDecoder().decode(ImagePreset.self, from: Data(#"""
    {"id":"w","name":"W","model":"krea2-raw","checkpointFamily":"raw-accel",
     "kroma":{"strength":0},"vae":"\#(wan)",
     "sampler":"res_2s","sigmaSchedule":"beta","shift":1.15,"eta":0.5,"bongmath":true,
     "stage2":{"sampler":"dpmpp_2m","sigmaSchedule":"karras","steps":2,"denoise":0.2,"eta":0.5,"bongmath":true}}
    """#.utf8))
    XCTAssertEqual(wire.checkpointFamily, "raw-accel")
    XCTAssertEqual(wire.kroma, KromaPolicy(strength: 0, file: nil))
    XCTAssertEqual(wire.vae, wan)
    XCTAssertEqual(wire.sampler, "res_2s")
    XCTAssertEqual(wire.sigmaSchedule, "beta")
    XCTAssertEqual(wire.shift, 1.15)
    XCTAssertEqual(wire.eta, 0.5)
    XCTAssertEqual(wire.bongmath, true)
    XCTAssertEqual(wire.stage2?.sampler, "dpmpp_2m")
    XCTAssertEqual(wire.stage2?.sigmaSchedule, "karras")
    XCTAssertEqual(wire.stage2?.steps, 2)
    XCTAssertEqual(wire.stage2?.denoise, 0.2)
    XCTAssertEqual(wire.stage2?.eta, 0.5)
    XCTAssertEqual(wire.stage2?.bongmath, true)

    // Absent stays nil — every default is visible in the record, never ambient.
    let bare = try JSONDecoder().decode(ImagePreset.self, from: Data(#"{"id":"r","name":"R"}"#.utf8))
    XCTAssertNil(bare.checkpointFamily); XCTAssertNil(bare.kroma); XCTAssertNil(bare.vae)
    XCTAssertNil(bare.sampler); XCTAssertNil(bare.sigmaSchedule); XCTAssertNil(bare.shift)
    XCTAssertNil(bare.eta); XCTAssertNil(bare.bongmath); XCTAssertNil(bare.stage2)

    // Store (validate + persist + reopen) and ResolvedPreset site.
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)
    try store.upsert(preset)
    let reopened = PresetStore(path: path, seedDefaults: false)
    // Todd 2026-09-04: `upsert` migrates the structured `kroma` onto
    // `loras[]` (`ImagePreset.migratingKromaDeprecation`) — the stored
    // preset differs from the one passed in by exactly that fold, plus the
    // now-true `kromaDeprecated` marker.
    // Review r2 (I4): the migrated entry is inserted at the FRONT, not
    // appended — render parity with the pre-deprecation prepend behavior.
    var migrated = preset
    migrated.loras.insert(LoraReference(
      filename: "kroma-v0.2-base-lora-rank-384-fro-0985.safetensors", scale: 0.3, role: "kroma"), at: 0)
    migrated.kromaDeprecated = true
    XCTAssertEqual(reopened.get("ref"), migrated)
    let resolved = try reopened.resolve("ref")
    XCTAssertEqual(resolved.checkpointFamily, "raw-accel")
    XCTAssertEqual(resolved.kroma, preset.kroma)
    XCTAssertEqual(resolved.vae, wan)
    XCTAssertEqual(resolved.sampler, "res_2s")
    XCTAssertEqual(resolved.sigmaSchedule, "beta")
    XCTAssertEqual(resolved.shift, 1.15)
    XCTAssertEqual(resolved.eta, 0.5)
    XCTAssertEqual(resolved.bongmath, false)
    XCTAssertEqual(resolved.stage2, stage2)
    let bareResolved = ResolvedPreset(preset: ImagePreset(id: "p", name: "P"))
    XCTAssertNil(bareResolved.checkpointFamily); XCTAssertNil(bareResolved.kroma)
    XCTAssertNil(bareResolved.sampler); XCTAssertNil(bareResolved.sigmaSchedule)
    XCTAssertNil(bareResolved.shift); XCTAssertNil(bareResolved.eta)
    XCTAssertNil(bareResolved.bongmath); XCTAssertNil(bareResolved.stage2)
  }

  /// Todd 2026-09-04: O4a (a krea2-family image preset must declare `kroma`)
  /// is RETIRED — kroma is a regular LoRA now, not an independent
  /// declaration, so its absence is exactly as legal as any other specific
  /// adapter's absence. (Was: "AC-44b/AC-44c — a krea2-family preset with no
  /// kroma is refused on save / flagged invalid on load".) The kroma FIELD's
  /// own range checks (finite non-negative strength, non-empty file) still
  /// apply when a client does send one — this is the compatibility shim,
  /// not a green light for garbage.
  func testKrea2ImagePresetNoLongerRequiresKroma() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)

    // A krea2-family preset with no `kroma` at all now saves cleanly.
    let noKroma = ImagePreset(id: "krea-kira", name: "Kira", engine: "zimage", model: "krea2", steps: 12)
    XCTAssertNoThrow(try store.upsert(noKroma))
    XCTAssertNil(store.validationError(for: "krea-kira"))
    XCTAssertEqual(store.get("krea-kira")?.kroma, nil)

    // Every family shape that used to be gated: turbo aliases, the
    // declared-table specs, and an explicit `checkpointFamily` — all save.
    for model in ["krea2", "krea-2", "krea-2-turbo", "krea/krea-2-turbo", "krea2-raw", "kroma-v0.2-turbo"] {
      var p = noKroma; p.id = "m-\(model)"; p.model = model
      XCTAssertNoThrow(try store.upsert(p), "\(model) no longer requires kroma")
    }
    var declaredFamily = noKroma
    declaredFamily.id = "declared-raw"
    declaredFamily.model = "some-custom-raw-install"
    declaredFamily.checkpointFamily = "raw-stock"
    XCTAssertNoThrow(try store.upsert(declaredFamily))

    // Kroma strength must still be a finite, non-negative number; a declared
    // file must still be non-empty — the FIELD's own validity, independent
    // of whether it is required.
    var nan = noKroma; nan.id = "nan"; nan.kroma = KromaPolicy(strength: .nan)
    XCTAssertThrowsError(try store.upsert(nan))
    var negative = noKroma; negative.id = "neg"; negative.kroma = KromaPolicy(strength: -0.1)
    XCTAssertThrowsError(try store.upsert(negative))
    var emptyFile = noKroma; emptyFile.id = "empty"; emptyFile.kroma = KromaPolicy(strength: 0.6, file: "  ")
    XCTAssertThrowsError(try store.upsert(emptyFile))

    // A preset already on disk in the old shape (krea2 family, no kroma) is
    // no longer flagged invalid — it is exactly as valid as it looks.
    let path = try makeTempPath()
    let onDisk = """
    {"presets":[
      {"id":"krea-film-apple","name":"Apple","engine":"zimage","model":"krea2","steps":8},
      {"id":"krea-bree","name":"Bree","engine":"zimage","model":"kroma-v0.2-turbo","kroma":{"strength":0}},
      {"id":"imported-cs-vector","name":"Vector","mediaKind":"image","engine":"zimage",
       "model":"Tongyi-MAI/Z-Image-Turbo-BF16","steps":16,"guidance":4.0}
    ]}
    """
    try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(onDisk.utf8).write(to: path)
    let loaded = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(loaded.list().map(\.id), ["krea-film-apple", "krea-bree", "imported-cs-vector"])
    XCTAssertNil(loaded.validationError(for: "krea-film-apple"))
    XCTAssertNil(loaded.validationError(for: "krea-bree"))
    XCTAssertNil(loaded.validationError(for: "imported-cs-vector"))
    XCTAssertTrue(loaded.invalidPresetIds.isEmpty)
    let listing = loaded.listing()
    XCTAssertEqual(listing.map(\.invalid), [false, false, false])
    XCTAssertNoThrow(try loaded.resolve("krea-film-apple"))
    XCTAssertNoThrow(try loaded.resolve("krea-bree"))
    // `krea-bree`'s `kroma: {strength: 0}` (an explicit "off") migrates to
    // no loras entry and no derived view — there is nothing to be a LoRA of.
    XCTAssertEqual(loaded.get("krea-bree")?.loras, [])
    XCTAssertNil(loaded.get("krea-bree")?.kroma)
    XCTAssertNil(loaded.get("krea-bree")?.kromaDeprecated)
  }

  /// PR #365 review r1 → redirect (Todd 2026-09-04): the migration itself,
  /// in isolation. A preset with ONLY a structured `kroma` (no matching
  /// `loras[]` entry yet) migrates to exactly one `loras[]` entry.
  func testMigrationFoldsStructuredKromaIntoExactlyOneLorasEntry() {
    let preset = ImagePreset(
      id: "only-structured", name: "x", model: "krea2-raw",
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base.safetensors"))
    let migrated = ImagePreset.migratingKromaDeprecation(preset)
    XCTAssertEqual(migrated.loras, [
      LoraReference(filename: "kroma-v0.3-base.safetensors", scale: 0.6, role: "kroma"),
    ])
    XCTAssertEqual(migrated.kroma, KromaPolicy(strength: 0.6, file: "kroma-v0.3-base.safetensors"))
    XCTAssertEqual(migrated.kromaDeprecated, true)
  }

  /// A preset with BOTH a structured `kroma` AND an already-present matching
  /// `loras[]` entry (the shape a client mid-migration, or an already-fixed
  /// preset, would send) migrates to exactly ONE entry — no duplicate.
  func testMigrationDoesNotDuplicateAnAlreadyPresentMirror() {
    let preset = ImagePreset(
      id: "already-mirrored", name: "x", model: "krea2-raw",
      loras: [
        LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel"),
        LoraReference(filename: "kroma-v0.3-base.safetensors", scale: 0.6, role: "kroma"),
      ],
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base.safetensors"))
    let migrated = ImagePreset.migratingKromaDeprecation(preset)
    XCTAssertEqual(migrated.loras.count, 2, "no duplicate kroma entry")
    XCTAssertEqual(migrated.loras.filter { $0.filename == "kroma-v0.3-base.safetensors" }.count, 1)
  }

  /// Idempotent: migrating an already-migrated preset again is a no-op —
  /// running it twice (once on load, once on the next save) must not drift.
  func testMigrationIsIdempotent() {
    let preset = ImagePreset(
      id: "idempotent", name: "x", model: "krea2-raw",
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base.safetensors"))
    let once = ImagePreset.migratingKromaDeprecation(preset)
    let twice = ImagePreset.migratingKromaDeprecation(once)
    XCTAssertEqual(once, twice)
  }

  /// The derived `kroma` view round-trips: saved, reopened, and resolved, the
  /// value is byte-identical to what a client that only ever reads `.kroma`
  /// (never `loras[]`) would have seen before the deprecation.
  func testDerivedKromaViewRoundTrips() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let preset = ImagePreset(
      id: "derived-view", name: "x", model: "krea2-raw",
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base.safetensors"))
    try store.upsert(preset)
    let expected = KromaPolicy(strength: 0.6, file: "kroma-v0.3-base.safetensors")
    XCTAssertEqual(store.get("derived-view")?.kroma, expected)
    XCTAssertEqual(try store.resolve("derived-view").kroma, expected)
    // A second save (the desktop editor round-tripping the GET response
    // verbatim) does not drift or duplicate.
    try store.upsert(try XCTUnwrap(store.get("derived-view")))
    XCTAssertEqual(store.get("derived-view")?.kroma, expected)
    XCTAssertEqual(store.get("derived-view")?.loras.count, 1)
  }

  /// Review r2, C1 (Critical): a client that deletes the kroma row and
  /// re-saves — sending `loras[]` without it and NO `kroma` field at all
  /// (the fixed desktop's shape, `ServerPreset.encode` never emits `kroma`)
  /// — must not have it resurrected. Nothing to migrate: the deletion wins.
  func testDeleteRowRoundTrip() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let original = ImagePreset(
      id: "delete-me", name: "x", model: "krea2-raw",
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-base.safetensors"))
    try store.upsert(original)
    XCTAssertEqual(store.get("delete-me")?.loras.count, 1, "sanity: it migrated in")

    // The client re-saves with the kroma row removed from `loras[]` and no
    // `kroma` field sent at all.
    let edited = ImagePreset(id: "delete-me", name: "x", model: "krea2-raw", loras: [])
    try store.upsert(edited)
    XCTAssertEqual(store.get("delete-me")?.loras, [])
    XCTAssertNil(store.get("delete-me")?.kroma)
    XCTAssertNil(store.get("delete-me")?.kromaDeprecated)
  }

  /// Review r2, C1: swapping which file is tagged `role: "kroma"` — the
  /// preset already has a NEW kroma-role entry in `loras[]`, but still
  /// carries the OLD structured `kroma.file` (a stale echo a naive
  /// round-trip could resend) — must not resurrect the old file as a
  /// second entry. Exactly one `role: "kroma"` row survives, and it is
  /// the new one.
  func testSwapFileRoundTripYieldsOneRow() {
    let preset = ImagePreset(
      id: "swap", name: "x", model: "krea2-raw",
      loras: [LoraReference(filename: "kroma-v0.4-new.safetensors", scale: 0.5, role: "kroma")],
      kroma: KromaPolicy(strength: 0.6, file: "kroma-v0.3-old.safetensors"))
    let migrated = ImagePreset.migratingKromaDeprecation(preset)
    XCTAssertEqual(migrated.loras.filter { ($0.role ?? "").lowercased() == "kroma" }.count, 1,
                   "exactly one kroma-role row")
    XCTAssertEqual(migrated.loras.map(\.filename), ["kroma-v0.4-new.safetensors"])
    XCTAssertEqual(migrated.kroma, KromaPolicy(strength: 0.5, file: "kroma-v0.4-new.safetensors"),
                   "the derived view reflects the NEW row, not the stale structured field")
  }

  /// Review r2, I3: the warning is actually logged, not just recorded in
  /// `migrationNotes`.
  func testMigrationLogsAWarningWhenKromaHasNoFile() {
    let preset = ImagePreset(id: "no-file", name: "x", model: "krea2-raw", kroma: KromaPolicy(strength: 0.6))
    var logged: [String] = []
    let migrated = ImagePreset.migratingKromaDeprecation(preset, log: { logged.append($0) })
    XCTAssertEqual(migrated.migrationNotes, ["kroma_dropped_no_file"])
    XCTAssertTrue(logged.contains { $0.contains("kroma_dropped_no_file") }, "\(logged)")
  }

  /// Review r2, I2: `ResolvedPreset` must decode a payload from an
  /// older engine that predates `kromaDeprecated`/`migrationNotes` entirely
  /// — a non-optional `Bool` would fail this decode outright.
  func testResolvedPresetDecodesWithoutKromaDeprecatedKey() throws {
    let json = #"""
    {"id":"old","name":"Old","description":"","mediaKind":"image","provider":"local","engine":"zimage",
     "steps":8,"width":512,"height":512,"loras":[],"injectedKeywords":[]}
    """#
    let decoded = try JSONDecoder().decode(ResolvedPreset.self, from: Data(json.utf8))
    XCTAssertNil(decoded.kromaDeprecated)
    XCTAssertNil(decoded.migrationNotes)
  }

  /// Review r2, I2: an ordinary preset with no kroma at all never emits
  /// `"kromaDeprecated": false` on the wire — the key is simply absent.
  func testKromaDeprecatedIsOmittedWhenFalse() throws {
    let preset = ImagePreset(id: "plain", name: "Plain", model: "z-image-turbo")
    let data = try JSONEncoder().encode(preset)
    let text = try XCTUnwrap(String(data: data, encoding: .utf8))
    XCTAssertFalse(text.contains("kromaDeprecated"), text)
    XCTAssertFalse(text.contains("migrationNotes"), text)
  }

  /// The recipe fields are validated with the same resolver `/v1/generate`
  /// uses (WP-E4): a preset naming a sampler/schedule the engine does not have
  /// is refused naming the value, and numeric knobs are range-checked. A
  /// `checkpointFamily` outside the five declared labels is refused naming them.
  func testNewFieldsAreValidatedOnSave() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    let base = ImagePreset(id: "p", name: "P", engine: "zimage", model: "krea2-raw",
                           checkpointFamily: "raw-accel", kroma: KromaPolicy(strength: 0))
    XCTAssertNoThrow(try store.upsert(base))

    func expectRefusal(_ mutate: (inout ImagePreset) -> Void, naming needle: String,
                       file: StaticString = #filePath, line: UInt = #line) {
      var p = base; p.id = "bad"; mutate(&p)
      XCTAssertThrowsError(try store.upsert(p), "expected refusal naming \(needle)", file: file, line: line) { error in
        guard case .validation(let message)? = error as? PresetStoreError else {
          return XCTFail("expected .validation, got \(error)", file: file, line: line)
        }
        XCTAssertTrue(message.contains(needle), "\(message) should name \(needle)", file: file, line: line)
      }
      XCTAssertNil(store.get("bad"), file: file, line: line)
    }

    expectRefusal({ $0.sampler = "uni_pc" }, naming: "uni_pc")
    expectRefusal({ $0.sigmaSchedule = "ays" }, naming: "ays")
    expectRefusal({ $0.stage2 = PresetStage(sampler: "uni_pc") }, naming: "uni_pc")
    expectRefusal({ $0.stage2 = PresetStage(sigmaSchedule: "ays") }, naming: "ays")
    expectRefusal({ $0.stage2 = PresetStage(steps: 0) }, naming: "stage2.steps")
    expectRefusal({ $0.stage2 = PresetStage(denoise: 0) }, naming: "stage2.denoise")
    expectRefusal({ $0.stage2 = PresetStage(denoise: 1.5) }, naming: "stage2.denoise")
    expectRefusal({ $0.stage2 = PresetStage(eta: -1) }, naming: "stage2.eta")
    expectRefusal({ $0.shift = 0 }, naming: "shift")
    expectRefusal({ $0.shift = .infinity }, naming: "shift")
    expectRefusal({ $0.eta = -0.5 }, naming: "eta")
    expectRefusal({ $0.noiseType = "perlin" }, naming: "noise_type")
    expectRefusal({ $0.noiseAlpha = .infinity }, naming: "noise_alpha")
    expectRefusal({ $0.implicitSteps = -1 }, naming: "implicit_steps")
    expectRefusal({ $0.implicitSteps = 9 }, naming: "implicit_steps")
    expectRefusal({ $0.c2 = 0 }, naming: "c2")
    expectRefusal({ $0.c2 = 1.01 }, naming: "c2")
    expectRefusal({ $0.c2 = 2.0 / 3.0 }, naming: "2/3")
    expectRefusal({ $0.checkpointFamily = "raw" }, naming: "raw-accel")
    expectRefusal({ $0.vae = "" }, naming: "vae")

    // Aliases the generate path accepts are accepted here too (D22/D25).
    var aliased = base; aliased.id = "aliased"
    aliased.sampler = "exponential/res_2s"; aliased.sigmaSchedule = "normal"
    aliased.stage2 = PresetStage(sampler: "dpmpp_2m", sigmaSchedule: "sgm_uniform", steps: 2, denoise: 0.2)
    XCTAssertNoThrow(try store.upsert(aliased))
  }

  /// A malformed entry on disk (here: `kroma` present but without `strength`)
  /// must not take the whole store down — the other entries load, and the
  /// broken one is kept as a flagged placeholder naming the decode failure.
  func testUndecodableEntryIsFlaggedNotDropped() throws {
    let path = try makeTempPath()
    let onDisk = """
    {"presets":[
      {"id":"ok","name":"OK","engine":"zimage","model":"z-image-turbo"},
      {"id":"broken","name":"Broken","engine":"zimage","model":"krea2","kroma":{}},
      {"id":"also-ok","name":"Also","mediaKind":"video"}
    ]}
    """
    try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(onDisk.utf8).write(to: path)
    let store = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(store.list().map(\.id), ["ok", "broken", "also-ok"])
    XCTAssertEqual(store.invalidPresetIds, ["broken"])
    let reason = try XCTUnwrap(store.validationError(for: "broken"))
    XCTAssertTrue(reason.contains("broken"), reason)
    XCTAssertTrue(reason.contains("kroma"), reason)
    XCTAssertThrowsError(try store.resolve("broken"))
    XCTAssertNoThrow(try store.resolve("ok"))
  }
}

// MARK: - WP-E20: the live store loads whole (fixture-gated on ~/.comfybox/presets.json)

extension PresetStoreTests {

  /// Deploy-day evidence (FDD §7.3 ordering: E20 → presets.json edit → C3/C6).
  /// A COPY of the live store is loaded through the new per-entry decoder:
  /// every entry is listed (none dropped). Todd 2026-09-04: O4a is retired,
  /// so — unlike before — a krea2-family preset with no `kroma` is no longer
  /// flagged; nothing in the live store is expected to be flagged for kroma
  /// reasons any more, regardless of family.
  func testLiveStoreLoadsWholeAndFlagsNothingForKroma() throws {
    let live = PresetStore.defaultPath()
    try XCTSkipUnless(FileManager.default.fileExists(atPath: live.path), "no live presets.json")
    let path = try makeTempPath()
    try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: live, to: path)

    let raw = try XCTUnwrap(
      (try JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])?["presets"] as? [[String: Any]])
    let store = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(store.list().count, raw.count, "every live entry must be listed — none silently dropped")
    XCTAssertEqual(store.list().map(\.id), raw.map { $0["id"] as? String ?? "" })

    for preset in store.list() {
      let flagged = store.validationError(for: preset.id)
      XCTAssertFalse(flagged?.contains("kroma") ?? false, "\(preset.id) flagged for kroma — O4a is retired: \(flagged ?? "")")
    }
  }
}
