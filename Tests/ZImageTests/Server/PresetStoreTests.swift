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

    let reopened = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(reopened.get("krea-kira")?.loras, [r256])
    XCTAssertEqual(try reopened.resolve("krea-kira").loras, [r256])

    let root = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(contentsOf: path)) as? [String: Any])
    let presets = try XCTUnwrap(root["presets"] as? [[String: Any]])
    let loras = try XCTUnwrap(presets.first?["loras"] as? [[String: Any]])
    XCTAssertEqual(loras.first?["role"] as? String, "accel")
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
    XCTAssertEqual(reopened.get("ref"), preset)
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

  /// AC-44b (store half) + AC-44c — O4a is not a daemon-only rule. A
  /// krea2-family image preset with no `kroma` is refused on save (naming
  /// the preset and the field), is flagged `invalid` on load (logged, kept in
  /// the list so nothing is silently dropped, un-resolvable), and a
  /// `zimage-*` preset is untouched (D14).
  func testKrea2ImagePresetRequiresKroma() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)

    // Save: refused, naming preset + field.
    let missing = ImagePreset(id: "krea-kira", name: "Kira", engine: "zimage", model: "krea2", steps: 12)
    XCTAssertThrowsError(try store.upsert(missing)) { error in
      guard case .validation(let message)? = error as? PresetStoreError else {
        return XCTFail("expected .validation, got \(error)")
      }
      XCTAssertTrue(message.contains("krea-kira"), message)
      XCTAssertTrue(message.contains("kroma"), message)
    }
    XCTAssertNil(store.get("krea-kira"), "a refused preset must not be stored")

    // The family can be declared by `checkpointFamily` even when the model is a
    // spec the alias table does not know.
    var declared = missing
    declared.id = "declared-raw"
    declared.model = "some-custom-raw-install"
    declared.checkpointFamily = "raw-stock"
    XCTAssertThrowsError(try store.upsert(declared))

    // The four turbo aliases and the declared-table specs all count as krea2.
    for model in ["krea2", "krea-2", "krea-2-turbo", "krea/krea-2-turbo", "krea2-raw", "kroma-v0.2-turbo"] {
      var p = missing; p.id = "m"; p.model = model
      XCTAssertThrowsError(try store.upsert(p), "\(model) must require kroma")
    }

    // `{strength: 0}` validates (explicit zero is a declaration, not an absence).
    var zero = missing
    zero.kroma = KromaPolicy(strength: 0)
    XCTAssertNoThrow(try store.upsert(zero))
    XCTAssertEqual(store.get("krea-kira")?.kroma, KromaPolicy(strength: 0))
    XCTAssertNil(store.validationError(for: "krea-kira"))

    // A Z-Image preset (the five imported-cs-* entries) needs no kroma.
    let zimage = ImagePreset(id: "imported-cs-neutral", name: "Neutral", mediaKind: "image",
                             engine: "zimage", model: "Tongyi-MAI/Z-Image-Turbo-BF16", steps: 9, guidance: 3.5)
    XCTAssertNoThrow(try store.upsert(zimage))
    var zimageFamily = zimage; zimageFamily.id = "zf"; zimageFamily.checkpointFamily = "zimage-turbo"
    XCTAssertNoThrow(try store.upsert(zimageFamily))

    // A video preset is never subject to the rule.
    XCTAssertNoThrow(try store.upsert(ImagePreset(id: "kira-video-neutral", name: "V", mediaKind: "video")))

    // Kroma strength must be a finite, non-negative number; a declared file must be non-empty.
    var nan = zero; nan.kroma = KromaPolicy(strength: .nan)
    XCTAssertThrowsError(try store.upsert(nan))
    var negative = zero; negative.kroma = KromaPolicy(strength: -0.1)
    XCTAssertThrowsError(try store.upsert(negative))
    var emptyFile = zero; emptyFile.kroma = KromaPolicy(strength: 0.6, file: "  ")
    XCTAssertThrowsError(try store.upsert(emptyFile))

    // Load: an invalid preset already on disk is flagged, not dropped, and cannot resolve.
    let path = try makeTempPath()
    let onDisk = """
    {"presets":[
      {"id":"krea-film-apple","name":"Apple","engine":"zimage","model":"krea2","steps":8,
       "loras":[{"filename":"kroma-v0.1.safetensors","scale":1.0}]},
      {"id":"krea-bree","name":"Bree","engine":"zimage","model":"kroma-v0.2-turbo","kroma":{"strength":0}},
      {"id":"imported-cs-vector","name":"Vector","mediaKind":"image","engine":"zimage",
       "model":"Tongyi-MAI/Z-Image-Turbo-BF16","steps":16,"guidance":4.0}
    ]}
    """
    try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(onDisk.utf8).write(to: path)
    let loaded = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(loaded.list().map(\.id), ["krea-film-apple", "krea-bree", "imported-cs-vector"],
                   "an invalid entry stays in the list — flagged, never silently dropped")
    let reason = try XCTUnwrap(loaded.validationError(for: "krea-film-apple"))
    XCTAssertTrue(reason.contains("krea-film-apple"), reason)
    XCTAssertTrue(reason.contains("kroma"), reason)
    XCTAssertNil(loaded.validationError(for: "krea-bree"))
    XCTAssertNil(loaded.validationError(for: "imported-cs-vector"))
    XCTAssertEqual(loaded.invalidPresetIds, ["krea-film-apple"])
    // The listing the API serves carries the flag.
    let listing = loaded.listing()
    XCTAssertEqual(listing.map(\.invalid), [true, false, false])
    XCTAssertEqual(listing[0].invalidReason, reason)
    XCTAssertNil(listing[1].invalidReason)
    // Nothing downstream can select it.
    XCTAssertThrowsError(try loaded.resolve("krea-film-apple")) { error in
      XCTAssertEqual(error as? PresetStoreError, .invalid(id: "krea-film-apple", reason: reason))
    }
    XCTAssertNoThrow(try loaded.resolve("krea-bree"))
    // Fixing it through the store clears the flag (the desktop app's edit path).
    var fixed = try XCTUnwrap(loaded.get("krea-film-apple"))
    fixed.kroma = KromaPolicy(strength: 1.0, file: "kroma-v0.1.safetensors")
    fixed.loras = fixed.loras.filter { !$0.filename.hasPrefix("kroma") }
    try loaded.upsert(fixed)
    XCTAssertNil(loaded.validationError(for: "krea-film-apple"))
    XCTAssertTrue(loaded.invalidPresetIds.isEmpty)
    XCTAssertNoThrow(try loaded.resolve("krea-film-apple"))
    // Deleting an invalid preset clears it as well.
    try Data(onDisk.utf8).write(to: path)
    let again = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(again.invalidPresetIds, ["krea-film-apple"])
    try again.delete("krea-film-apple")
    XCTAssertTrue(again.invalidPresetIds.isEmpty)
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
  /// every entry is listed (none dropped), no video or Z-Image preset is
  /// flagged, and anything flagged is a krea2-family image preset — before
  /// the migration that is the eight Krea entries without `kroma`; after it,
  /// nothing. The assertion holds on both sides of the migration.
  func testLiveStoreLoadsWholeAndFlagsOnlyKrea2Presets() throws {
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
      let krea2 = PresetStore.resolvesToKrea2Family(preset)
      if preset.mediaKind == "video" || !krea2 {
        XCTAssertNil(flagged, "\(preset.id) is not a krea2 image preset and must not be flagged: \(flagged ?? "")")
      }
      if let flagged {
        XCTAssertTrue(krea2, "\(preset.id) flagged but not krea2-family: \(flagged)")
        XCTAssertTrue(flagged.contains("kroma"), flagged)
        XCTAssertNil(preset.kroma, "\(preset.id) declares kroma yet is flagged: \(flagged)")
      } else if krea2, preset.mediaKind != "video" {
        XCTAssertNotNil(preset.kroma, "\(preset.id) is krea2-family, unflagged, yet declares no kroma")
      }
    }
  }
}
