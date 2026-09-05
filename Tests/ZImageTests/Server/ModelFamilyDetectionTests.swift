// ModelFamilyDetectionTests.swift — GET /v1/model/family (comfybox#359).

import XCTest
@testable import ZImage

final class ModelFamilyDetectionTests: XCTestCase {

  private var scratch: URL!

  override func setUpWithError() throws {
    scratch = FileManager.default.temporaryDirectory
      .appending(path: "model-family-detect-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: scratch)
  }

  @discardableResult
  private func makeKrea2Root(_ name: String, transformer: String) throws -> URL {
    let root = scratch.appending(path: name)
    let fm = FileManager.default
    try fm.createDirectory(at: root.appending(path: "text_encoder"), withIntermediateDirectories: true)
    try fm.createDirectory(at: root.appending(path: "vae"), withIntermediateDirectories: true)
    fm.createFile(atPath: root.appending(path: transformer).path, contents: Data([0]))
    fm.createFile(atPath: root.appending(path: "text_encoder/model.safetensors").path, contents: Data([0]))
    fm.createFile(atPath: root.appending(path: "vae/diffusion_pytorch_model.safetensors").path, contents: Data([0]))
    return root
  }

  // MARK: krea2 — turbo alias, declared alias, on-disk path (the real
  // desktop shape: `custom_model_path` pointing at ~/LocalModels/krea2-raw)

  func testTurboAliasIsKrea2Turbo() {
    let result = ModelFamilyDetector.detect(spec: "krea2")
    XCTAssertEqual(result.model, "krea2")
    XCTAssertEqual(result.family, "krea2")
    XCTAssertEqual(result.variant, "turbo")
  }

  func testDeclaredRawAliasResolvesFromItsDirectory() throws {
    let root = try makeKrea2Root("raw", transformer: "raw.safetensors")
    let original = Krea2ModelDetection.specDirectories
    defer { Krea2ModelDetection.configureSpecDirectories(original, replace: true) }
    Krea2ModelDetection.configureSpecDirectories(["krea2-raw": root.path])

    let result = ModelFamilyDetector.detect(spec: "krea2-raw")
    XCTAssertEqual(result.family, "krea2")
    XCTAssertEqual(result.variant, "raw")
  }

  func testOnDiskCustomModelPathResolvesLikeADeclaredAlias() throws {
    // The production shape of the 26 desktop presets: `custom_model_path`
    // is a literal filesystem path, not a declared alias — but it IS an
    // existing, detectable Krea-2 directory.
    let root = try makeKrea2Root("custom-raw", transformer: "raw.safetensors")
    let result = ModelFamilyDetector.detect(spec: root.path)
    XCTAssertEqual(result.family, "krea2")
    XCTAssertEqual(result.variant, "raw")
  }

  func testUndetectableKrea2LikePathHasNilVariant() throws {
    // isKnownKrea2Model doesn't fire for an arbitrary path, and it isn't a
    // krea2 directory either — this is just unclassifiable, not z-image.
    let empty = scratch.appending(path: "empty")
    try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
    let result = ModelFamilyDetector.detect(spec: empty.path)
    XCTAssertNil(result.family)
    XCTAssertNil(result.variant)
  }

  // MARK: z-image

  func testZImageTurboAliases() {
    for spec in ["z-image", "z-image-turbo", "zimage-turbo"] {
      let result = ModelFamilyDetector.detect(spec: spec)
      XCTAssertEqual(result.family, "z-image", spec)
      XCTAssertEqual(result.variant, "turbo", spec)
    }
  }

  func testZImageBaseAliases() {
    for spec in ["z-image-base", "zimage-base"] {
      let result = ModelFamilyDetector.detect(spec: spec)
      XCTAssertEqual(result.family, "z-image", spec)
      XCTAssertEqual(result.variant, "base", spec)
    }
  }

  func testQuantizationSuffixesStillResolveToTheirAlias() {
    // Not a text guess: `-q4`/`-q8`/`-bf16` are the quantization suffixes
    // `WarmServer.parseModelSpec` already strips, and what is left is an
    // EXACT alias.
    XCTAssertEqual(ModelFamilyDetector.detect(spec: "z-image-turbo-bf16").variant, "turbo")
    XCTAssertEqual(ModelFamilyDetector.detect(spec: "z-image-base-bf16").variant, "base")
    XCTAssertEqual(ModelFamilyDetector.detect(spec: "z-image-turbo-q8").variant, "turbo")
  }

  // MARK: - Round 2, ruling 5: NEVER guess the z-image variant from text
  //
  // `cyberrealisticZImage_v50.safetensors` is served as BASE
  // (kira-model-is-zimage-base), and its filename contains neither "base" nor
  // "turbo" — the old "not base ⇒ turbo" fallback labelled it `zimage-turbo`,
  // which is the wrong recipe under the right name. Unknown variant now means
  // NO label; `model` is still written, which is what makes a preset
  // expandable anyway.

  func testCyberrealisticPathIsZImageWithNoVariantGuess() {
    let spec = "/Users/todd/Models-working/cyberrealistic-z-image/cyberrealisticZImage_v50.safetensors"
    let result = ModelFamilyDetector.detect(spec: spec)
    XCTAssertEqual(result.family, "z-image", "the spec does name z-image")
    XCTAssertNil(result.variant, "served as BASE — a filename must never decide turbo vs base")
  }

  func testAHuggingFaceIdIsNotAVariantDeclaration() {
    let result = ModelFamilyDetector.detect(spec: "Tongyi-MAI/Z-Image-Turbo-BF16")
    XCTAssertEqual(result.family, "z-image")
    XCTAssertNil(result.variant, "a repo name is text, not a declared alias")
  }

  func testAnArbitraryZImageCheckpointNameYieldsNoVariant() {
    for spec in ["some-zimage-merge-v3", "moodyPornMix_zit_v7", "/models/z-image/mix.safetensors"] {
      XCTAssertNil(ModelFamilyDetector.detect(spec: spec).variant, spec)
    }
  }

  // MARK: unclassifiable

  func testUnknownSpecIsNilFamilyAndVariant() {
    let result = ModelFamilyDetector.detect(spec: "some-other-checkpoint")
    XCTAssertEqual(result.model, "some-other-checkpoint")
    XCTAssertNil(result.family)
    XCTAssertNil(result.variant)
  }

  func testWhitespaceIsTrimmedForDetectionButModelIsEchoedVerbatim() {
    let result = ModelFamilyDetector.detect(spec: "  krea2  ")
    XCTAssertEqual(result.model, "  krea2  ")
    XCTAssertEqual(result.family, "krea2")
    XCTAssertEqual(result.variant, "turbo")
  }

  // MARK: - spec / loadable (fix round 1)
  //
  // The whole point of the route: `checkpoint_family` alone changes NOTHING
  // for a preset that names no `model` — `PresetLoRAStack.decide` returns
  // `no_model` before it is ever read. The desktop has to write `model`, so
  // the route must hand back a spec the engine would actually accept.

  func testDeclaredAliasDirectoryComesBackAsTheAliasNotThePath() throws {
    let root = try makeKrea2Root("aliased-raw", transformer: "raw.safetensors")
    let original = Krea2ModelDetection.specDirectories
    defer { Krea2ModelDetection.configureSpecDirectories(original, replace: true) }
    Krea2ModelDetection.configureSpecDirectories(["krea2-raw": root.path], replace: true)

    // The desktop stores the literal path; the canonical engine spec for it
    // is the declared alias.
    let result = ModelFamilyDetector.detect(spec: root.path)
    XCTAssertEqual(result.spec, "krea2-raw")
    XCTAssertTrue(result.loadable)
    XCTAssertNil(result.reason)
  }

  func testTurboAliasIsLoadableAndCanonicalizesToItself() {
    let result = ModelFamilyDetector.detect(spec: "krea-2-turbo")
    XCTAssertEqual(result.spec, "krea-2-turbo")
    XCTAssertTrue(result.loadable)
  }

  func testUndeclaredKrea2DirectoryComesBackAsItsOwnAbsolutePath() throws {
    let root = try makeKrea2Root("undeclared-raw", transformer: "raw.safetensors")
    let original = Krea2ModelDetection.specDirectories
    defer { Krea2ModelDetection.configureSpecDirectories(original, replace: true) }
    Krea2ModelDetection.configureSpecDirectories([:], replace: true)

    let result = ModelFamilyDetector.detect(spec: root.path)
    XCTAssertEqual(result.spec, root.standardizedFileURL.path)
    XCTAssertTrue(result.loadable, "an existing krea2 directory is exactly what Krea2ModelDetection.resolve accepts")
  }

  func testPathSpecIsStandardizedForTheCanonicalSpec() throws {
    // ModelResolution.resolve does NOT expand `~`, and neither does
    // parseModelSpec for a path — a tilde spec written into `model` would
    // throw modelNotFound, so the canonical spec is always tilde-expanded,
    // standardized and free of a trailing slash.
    let root = try makeKrea2Root("tilde-raw", transformer: "raw.safetensors")
    let original = Krea2ModelDetection.specDirectories
    defer { Krea2ModelDetection.configureSpecDirectories(original, replace: true) }
    Krea2ModelDetection.configureSpecDirectories([:], replace: true)

    let result = ModelFamilyDetector.detect(spec: root.path + "/")
    XCTAssertEqual(result.spec, root.standardizedFileURL.path, "no trailing slash, standardized")
    XCTAssertTrue(result.loadable)
  }

  func testNonexistentPathIsNotLoadableAndSaysWhy() {
    let missing = scratch.appending(path: "does-not-exist").path
    let result = ModelFamilyDetector.detect(spec: missing)
    XCTAssertFalse(result.loadable)
    XCTAssertNotNil(result.reason)
    XCTAssertTrue(result.reason?.contains("does not exist") == true, result.reason ?? "nil")
  }

  func testDeclaredAliasWithAMissingDirectoryIsNotLoadable() {
    let original = Krea2ModelDetection.specDirectories
    defer { Krea2ModelDetection.configureSpecDirectories(original, replace: true) }
    Krea2ModelDetection.configureSpecDirectories(
      ["krea2-ghost": scratch.appending(path: "ghost").path], replace: true)

    let result = ModelFamilyDetector.detect(spec: "krea2-ghost")
    XCTAssertFalse(result.loadable, "Krea2ModelDetection.resolve would throw for this alias")
    XCTAssertNotNil(result.reason)
  }

  func testExistingDirectoryThatIsNotKrea2IsStillLoadableAsASnapshot() throws {
    // ModelResolution.resolve returns any existing path as-is, and
    // ModelPool.detectFamily then falls through to flux1 (Z-Image). The
    // engine accepts it; the route says so without pretending to know the
    // family.
    let dir = scratch.appending(path: "plain-snapshot")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let result = ModelFamilyDetector.detect(spec: dir.path)
    XCTAssertNil(result.family)
    XCTAssertTrue(result.loadable)
    XCTAssertEqual(result.spec, dir.standardizedFileURL.path)
  }

  func testExistingSafetensorsFileIsLoadable() throws {
    let file = scratch.appending(path: "checkpoint.safetensors")
    FileManager.default.createFile(atPath: file.path, contents: Data([0]))
    let result = ModelFamilyDetector.detect(spec: file.path)
    XCTAssertTrue(result.loadable)
    XCTAssertEqual(result.spec, file.standardizedFileURL.path)
  }

  func testExistingNonModelFileIsNotLoadable() throws {
    let file = scratch.appending(path: "notes.txt")
    FileManager.default.createFile(atPath: file.path, contents: Data([0]))
    let result = ModelFamilyDetector.detect(spec: file.path)
    XCTAssertFalse(result.loadable)
    XCTAssertNotNil(result.reason)
  }

  func testKnownEngineSpecsAreLoadable() {
    for spec in ["z-image-turbo", "z-image-base", "briaai/FIBO", "klein-4b", "chroma-8.9b"] {
      let result = ModelFamilyDetector.detect(spec: spec)
      XCTAssertTrue(result.loadable, spec)
      XCTAssertEqual(result.spec, spec, spec)
    }
  }

  func testAnUnknownBareSpecIsNotLoadable() {
    let result = ModelFamilyDetector.detect(spec: "some-other-checkpoint")
    XCTAssertFalse(result.loadable)
    XCTAssertNotNil(result.reason)
  }

  func testWhitespaceIsTrimmedOutOfTheCanonicalSpec() {
    let result = ModelFamilyDetector.detect(spec: "  krea2  ")
    XCTAssertEqual(result.model, "  krea2  ", "echoed verbatim")
    XCTAssertEqual(result.spec, "krea2")
    XCTAssertTrue(result.loadable)
  }
}
