import XCTest

@testable import ZImage

/// WP-E5 — `Krea2Variant` and fail-closed model-directory resolution (FDD
/// §3.5, AC-33 / AC-34 / AC-34a). F3's bug: `detect(at:)` required
/// `turbo.safetensors` by name and `resolve(spec:)` fell through to the HF
/// Krea-2-Turbo snapshot with no error and no log, so pointing the engine at a
/// Raw directory silently rendered Turbo. Every test here is weight-free: a
/// Krea-2 model root is detected by file *presence*, so placeholder files in
/// a temp dir are a faithful fixture for resolution.
final class Krea2VariantDetectionTests: XCTestCase {

  private var scratch: URL!

  override func setUpWithError() throws {
    scratch = FileManager.default.temporaryDirectory
      .appending(path: "krea2-variant-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: scratch)
  }

  /// Build a model root with the given transformer filenames plus the Qwen
  /// text-encoder / VAE files a Krea-2 root must carry.
  @discardableResult
  private func makeRoot(
    _ name: String, transformers: [String], textEncoder: Bool = true, vae: Bool = true,
    modelIndex: String? = nil
  ) throws -> URL {
    let root = scratch.appending(path: name)
    let fm = FileManager.default
    try fm.createDirectory(at: root.appending(path: "text_encoder"), withIntermediateDirectories: true)
    try fm.createDirectory(at: root.appending(path: "vae"), withIntermediateDirectories: true)
    for t in transformers {
      fm.createFile(atPath: root.appending(path: t).path, contents: Data([0]))
    }
    if textEncoder {
      fm.createFile(atPath: root.appending(path: "text_encoder/model.safetensors").path, contents: Data([0]))
    }
    if vae {
      fm.createFile(atPath: root.appending(path: "vae/diffusion_pytorch_model.safetensors").path, contents: Data([0]))
    }
    if let modelIndex {
      try modelIndex.write(to: root.appending(path: "model_index.json"), atomically: true, encoding: .utf8)
    }
    return root
  }

  /// A stand-in for the HF Krea-2-Turbo snapshot so the alias branch is
  /// observable without touching ~/.cache.
  private func fakeSnapshot(_ calls: UnsafeMutablePointer<Int>) -> () throws -> Krea2ModelPaths {
    let root = scratch.appending(path: "hf-snapshot")
    return {
      calls.pointee += 1
      return Krea2ModelPaths(root: root, variant: .turbo)
    }
  }

  // MARK: AC-33 — detection names the variant and the file

  func testDetectsRawDirectory() throws {
    let root = try makeRoot("raw", transformers: ["raw.safetensors"])
    let paths = try Krea2ModelDetection.detect(at: root)
    XCTAssertEqual(paths.variant, .raw)
    XCTAssertEqual(paths.transformerFile.lastPathComponent, "raw.safetensors")
    XCTAssertEqual(paths.root, root)
  }

  func testDetectsTurboDirectory() throws {
    let root = try makeRoot("turbo", transformers: ["turbo.safetensors"])
    let paths = try Krea2ModelDetection.detect(at: root)
    XCTAssertEqual(paths.variant, .turbo)
    XCTAssertEqual(paths.transformerFile.lastPathComponent, "turbo.safetensors")
  }

  /// AC-33 on the real install (the pipeline-load half is the integration
  /// test `Krea2RawLoadTests`). Skips when the Raw checkpoint is not on disk.
  func testDetectsLiveRawInstall() throws {
    let root = URL(fileURLWithPath: NSString(string: "~/LocalModels/krea2-raw").expandingTildeInPath)
    try XCTSkipUnless(
      FileManager.default.fileExists(atPath: root.appending(path: "raw.safetensors").path),
      "krea2-raw not installed on this machine")
    let paths = try Krea2ModelDetection.detect(at: root)
    XCTAssertEqual(paths.variant, .raw)
    XCTAssertEqual(paths.transformerFile.lastPathComponent, "raw.safetensors")
  }

  func testModelIndexEscapeHatchNamesAThirdFilename() throws {
    let root = try makeRoot(
      "comfy-named", transformers: ["krea2_raw_bf16.safetensors"],
      modelIndex: #"{"krea2_variant": "raw", "transformer_file": "krea2_raw_bf16.safetensors"}"#)
    let paths = try Krea2ModelDetection.detect(at: root)
    XCTAssertEqual(paths.variant, .raw)
    XCTAssertEqual(paths.transformerFile.lastPathComponent, "krea2_raw_bf16.safetensors")
  }

  func testModelIndexWithUnknownVariantThrows() throws {
    let root = try makeRoot(
      "bad-index", transformers: ["x.safetensors"],
      modelIndex: #"{"krea2_variant": "ultra", "transformer_file": "x.safetensors"}"#)
    XCTAssertThrowsError(try Krea2ModelDetection.detect(at: root)) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(_, let reason) = error,
            case .invalidModelIndex = reason
      else { return XCTFail("expected notAKrea2ModelDirectory(.invalidModelIndex), got \(error)") }
    }
  }

  // MARK: AC-34 — fail closed

  func testFailsClosed() throws {
    // An existing directory with no recognisable DiT throws the named error —
    // it does NOT return the HF-cache turbo snapshot. Asserted on the error
    // type, not on "not turbo".
    let empty = try makeRoot("no-dit", transformers: [])
    var snapshotCalls = 0
    XCTAssertThrowsError(
      try Krea2ModelDetection.resolve(
        spec: empty.path, specDirectories: [:], turboSnapshot: fakeSnapshot(&snapshotCalls))
    ) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(let path, let reason) = error else {
        return XCTFail("expected notAKrea2ModelDirectory, got \(error)")
      }
      XCTAssertEqual(path, empty.path)
      XCTAssertEqual(reason, .noTransformer)
    }
    XCTAssertEqual(snapshotCalls, 0, "a directory that is not a Krea-2 root must never fall through to the HF snapshot")

    // A dir with both files throws ambiguousVariant. Never guess.
    let both = try makeRoot("both", transformers: ["raw.safetensors", "turbo.safetensors"])
    XCTAssertThrowsError(try Krea2ModelDetection.detect(at: both)) { error in
      guard case Krea2ModelPathsError.ambiguousVariant(let dir) = error else {
        return XCTFail("expected ambiguousVariant, got \(error)")
      }
      XCTAssertEqual(dir, both)
    }

    // A DiT without its encoder / decoder is not a model root either.
    let noTE = try makeRoot("no-te", transformers: ["raw.safetensors"], textEncoder: false)
    XCTAssertThrowsError(try Krea2ModelDetection.detect(at: noTE)) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(_, let reason) = error else {
        return XCTFail("expected notAKrea2ModelDirectory, got \(error)")
      }
      XCTAssertEqual(reason, .missingTextEncoder)
    }
    let noVAE = try makeRoot("no-vae", transformers: ["raw.safetensors"], vae: false)
    XCTAssertThrowsError(try Krea2ModelDetection.detect(at: noVAE)) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(_, let reason) = error else {
        return XCTFail("expected notAKrea2ModelDirectory, got \(error)")
      }
      XCTAssertEqual(reason, .missingVAE)
    }
  }

  func testNotADirectoryThrows() throws {
    let file = scratch.appending(path: "file.txt")
    try "x".write(to: file, atomically: true, encoding: .utf8)
    XCTAssertThrowsError(try Krea2ModelDetection.detect(at: file)) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(_, let reason) = error else {
        return XCTFail("expected notAKrea2ModelDirectory, got \(error)")
      }
      XCTAssertEqual(reason, .notADirectory)
    }
  }

  func testIsKrea2ModelDirectoryProbeIsNonThrowing() throws {
    let raw = try makeRoot("probe-raw", transformers: ["raw.safetensors"])
    let none = try makeRoot("probe-none", transformers: [])
    XCTAssertTrue(Krea2ModelDetection.isKrea2ModelDirectory(raw))
    XCTAssertFalse(Krea2ModelDetection.isKrea2ModelDirectory(none))
  }

  // MARK: AC-34a — spec resolution: alias table, the only fallback, unmapped throws

  func testSpecResolution() throws {
    let rawRoot = try makeRoot("alias-raw", transformers: ["raw.safetensors"])
    var snapshotCalls = 0
    let snapshot = fakeSnapshot(&snapshotCalls)

    // 1. An unmapped alias throws — it does not become Turbo.
    XCTAssertThrowsError(
      try Krea2ModelDetection.resolve(spec: "krea2-raw", specDirectories: [:], turboSnapshot: snapshot)
    ) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(let spec, let reason) = error else {
        return XCTFail("expected notAKrea2ModelDirectory(.unmappedSpec), got \(error)")
      }
      XCTAssertEqual(spec, "krea2-raw")
      XCTAssertEqual(reason, .unmappedSpec)
    }
    XCTAssertEqual(snapshotCalls, 0)

    // 2. With the entry present it returns .raw and raw.safetensors.
    let table = ["krea2-raw": rawRoot.path]
    let resolved = try Krea2ModelDetection.resolve(
      spec: "krea2-raw", specDirectories: table, turboSnapshot: snapshot)
    XCTAssertEqual(resolved.variant, .raw)
    XCTAssertEqual(resolved.transformerFile.lastPathComponent, "raw.safetensors")
    XCTAssertEqual(resolved.root.standardizedFileURL.path, rawRoot.standardizedFileURL.path)
    XCTAssertEqual(snapshotCalls, 0)

    // 3. The four turbo aliases — and only those four — reach the HF snapshot.
    for alias in ["krea2", "krea-2", "krea-2-turbo", "krea/krea-2-turbo", "Krea2", "KREA/Krea-2-Turbo"] {
      let before = snapshotCalls
      let paths = try Krea2ModelDetection.resolve(spec: alias, specDirectories: [:], turboSnapshot: snapshot)
      XCTAssertEqual(snapshotCalls, before + 1, "alias \(alias) must reach the snapshot")
      XCTAssertEqual(paths.variant, .turbo)
    }
    XCTAssertEqual(Krea2ModelDetection.turboAliases, ["krea2", "krea-2", "krea-2-turbo", "krea/krea-2-turbo"])

    // 4. A fifth invented alias throws.
    let before = snapshotCalls
    XCTAssertThrowsError(
      try Krea2ModelDetection.resolve(spec: "krea-2-ultra", specDirectories: [:], turboSnapshot: snapshot)
    ) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(_, let reason) = error else {
        return XCTFail("expected notAKrea2ModelDirectory, got \(error)")
      }
      XCTAssertEqual(reason, .unmappedSpec)
    }
    XCTAssertEqual(snapshotCalls, before, "an invented alias must not reach the snapshot")

    // 5. An explicit existing path wins over the table and the aliases.
    let explicit = try Krea2ModelDetection.resolve(
      spec: rawRoot.path, specDirectories: ["krea2-raw": "/nonexistent"], turboSnapshot: snapshot)
    XCTAssertEqual(explicit.variant, .raw)
    XCTAssertEqual(snapshotCalls, before)

    // 6. A table entry whose directory is missing fails loud, naming the dir.
    XCTAssertThrowsError(
      try Krea2ModelDetection.resolve(
        spec: "krea2-raw", specDirectories: ["krea2-raw": scratch.appending(path: "missing").path],
        turboSnapshot: snapshot)
    ) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(_, let reason) = error else {
        return XCTFail("expected notAKrea2ModelDirectory, got \(error)")
      }
      XCTAssertEqual(reason, .notADirectory)
    }
  }

  // MARK: comfybox#359 — detectVariant (GET /v1/model/family), no HF fallback

  func testDetectVariantResolvesTurboAliasesWithoutTouchingTheTable() {
    for alias in ["krea2", "krea-2", "krea-2-turbo", "krea/krea-2-turbo", "Krea2"] {
      XCTAssertEqual(Krea2ModelDetection.detectVariant(spec: alias, specDirectories: [:]), .turbo, alias)
    }
  }

  func testDetectVariantResolvesADeclaredAliasFromItsDirectoryContents() throws {
    let rawRoot = try makeRoot("detect-alias-raw", transformers: ["raw.safetensors"])
    let table = ["krea2-raw": rawRoot.path]
    XCTAssertEqual(Krea2ModelDetection.detectVariant(spec: "krea2-raw", specDirectories: table), .raw)
    XCTAssertEqual(Krea2ModelDetection.detectVariant(spec: "KREA2-RAW", specDirectories: table), .raw,
      "case-insensitive, like specDirectory")

    let turboRoot = try makeRoot("detect-alias-turbo", transformers: ["turbo.safetensors"])
    let kromaTable = ["kroma-v0.2-turbo": turboRoot.path]
    XCTAssertEqual(Krea2ModelDetection.detectVariant(spec: "kroma-v0.2-turbo", specDirectories: kromaTable), .turbo)
  }

  func testDetectVariantResolvesAnExplicitExistingPath() throws {
    let root = try makeRoot("detect-explicit-raw", transformers: ["raw.safetensors"])
    // Not in the table at all — an on-disk path is detected from its own
    // contents, exactly like `resolve(spec:)`'s first branch.
    XCTAssertEqual(Krea2ModelDetection.detectVariant(spec: root.path, specDirectories: [:]), .raw)
  }

  func testDetectVariantIsNilForAnUnmappedAliasOrUndetectableDirectory() throws {
    // An alias with no table entry never falls through to the HF snapshot
    // (there is no snapshot parameter here at all) — it is simply unknown.
    XCTAssertNil(Krea2ModelDetection.detectVariant(spec: "krea-2-ultra", specDirectories: [:]))
    XCTAssertNil(Krea2ModelDetection.detectVariant(spec: "z-image-turbo", specDirectories: [:]))

    // An existing directory with no recognisable DiT is not a silent Turbo —
    // it is nil, same fail-closed posture as `detect(at:)` throwing.
    let empty = try makeRoot("detect-no-dit", transformers: [])
    XCTAssertNil(Krea2ModelDetection.detectVariant(spec: empty.path, specDirectories: [:]))

    XCTAssertNil(Krea2ModelDetection.detectVariant(spec: "/nonexistent/path", specDirectories: [:]))
  }

  func testDefaultSpecDirectoryTableCoversRawAndKroma() {
    // The single table WarmServer.parseModelSpec consults (no second table).
    let raw = Krea2ModelDetection.specDirectory("krea2-raw")
    XCTAssertTrue(raw?.path.hasSuffix("/LocalModels/krea2-raw") ?? false, "\(String(describing: raw))")
    let kroma = Krea2ModelDetection.specDirectory("kroma-v0.2-turbo")
    XCTAssertTrue(kroma?.path.hasSuffix("/LocalModels/kroma-v0.2") ?? false, "\(String(describing: kroma))")
    XCTAssertNil(Krea2ModelDetection.specDirectory("krea-2-ultra"))
  }

  func testConfiguredSpecDirectoriesMergeOverDefaults() {
    let original = Krea2ModelDetection.specDirectories
    defer { Krea2ModelDetection.configureSpecDirectories(original, replace: true) }

    Krea2ModelDetection.configureSpecDirectories(["krea2-raw": "/Volumes/Bolt/krea2-raw", "my-krea": "~/x"])
    XCTAssertEqual(Krea2ModelDetection.specDirectory("krea2-raw")?.path, "/Volumes/Bolt/krea2-raw")
    XCTAssertEqual(Krea2ModelDetection.specDirectory("my-krea")?.path, NSString(string: "~/x").expandingTildeInPath)
    XCTAssertNotNil(Krea2ModelDetection.specDirectory("kroma-v0.2-turbo"), "defaults survive a partial override")
    XCTAssertTrue(Krea2ModelDetection.isKnownKrea2Model("my-krea"), "a declared alias is a krea2 family hint")
  }

  func testParseModelSpecConsultsTheSpecDirectoryTable() {
    let raw = WarmServer.parseModelSpec(from: "krea2-raw")
    XCTAssertTrue(raw.hasSuffix("/LocalModels/krea2-raw"), raw)
    XCTAssertTrue(raw.hasPrefix("/"), "tilde must be expanded: \(raw)")
    // The q8 suffix the pool-style ids carry is stripped before the lookup.
    XCTAssertEqual(WarmServer.parseModelSpec(from: "krea2-raw-q8"), raw)
  }

  func testIsKnownKrea2ModelIncludesRaw() {
    XCTAssertTrue(Krea2ModelDetection.isKnownKrea2Model("krea2-raw"))
    XCTAssertTrue(Krea2ModelDetection.isKnownKrea2Model("krea-2-raw"))
    XCTAssertTrue(Krea2ModelDetection.isKnownKrea2Model("Krea2-Raw"))
    XCTAssertTrue(Krea2ModelDetection.isKnownKrea2Model("krea2"))
    XCTAssertTrue(Krea2ModelDetection.isKnownKrea2Model("krea/Krea-2-Turbo"))
    XCTAssertFalse(Krea2ModelDetection.isKnownKrea2Model("z-image-turbo"))
  }

  func testExplicitModelDirGoesThroughDetection() throws {
    // The CLI's --model-dir path: a non-Krea-2 directory is refused, not
    // constructed blind.
    let none = try makeRoot("cli-none", transformers: [])
    XCTAssertThrowsError(try Krea2ModelPaths.resolve(modelDir: none.path))
    let raw = try makeRoot("cli-raw", transformers: ["raw.safetensors"])
    XCTAssertEqual(try Krea2ModelPaths.resolve(modelDir: raw.path).variant, .raw)
  }
}
