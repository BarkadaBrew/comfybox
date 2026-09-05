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

  func testZImageCatalogAndHuggingFaceIdsReadVariantFromText() {
    XCTAssertEqual(ModelFamilyDetector.detect(spec: "z-image-turbo-bf16").variant, "turbo")
    XCTAssertEqual(ModelFamilyDetector.detect(spec: "z-image-base-bf16").variant, "base")
    XCTAssertEqual(ModelFamilyDetector.detect(spec: "Tongyi-MAI/Z-Image-Turbo-BF16").variant, "turbo")
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
}
