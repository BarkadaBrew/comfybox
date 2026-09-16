import XCTest

@testable import ZImage

/// Kroma v0.2 (lodestones) registration — a Krea-2 fine-tune served from a
/// local model root (~/LocalModels/kroma-v0.2) with TE/VAE/tokenizer
/// symlinked from the Krea-2 snapshot.
final class KromaModelTests: XCTestCase {

  func testParseModelSpecResolvesKromaToLocalRoot() {
    let spec = WarmServer.parseModelSpec(from: "kroma-v0.2-turbo")
    XCTAssertTrue(spec.hasSuffix("/LocalModels/kroma-v0.2"), spec)
    XCTAssertTrue(spec.hasPrefix("/"), "tilde must be expanded: \(spec)")
  }

  func testRegistryCarriesKromaAsKrea2Family() throws {
    let model = try XCTUnwrap(ComfyBoxModelRegistry.models["kroma-v0.2-turbo"])
    XCTAssertEqual(model.family, .krea2)
    XCTAssertEqual(model.huggingFaceId, "lodestones/Kroma")
    XCTAssertFalse(model.supportsGuidance, "turbo distill — no CFG")
    XCTAssertTrue(model.supportsLoRA)
  }

  /// Kroma v0.3 BASE (Todd 2026-09-15 "just support it"): the undistilled
  /// fine-tune as a resident base — declared alias, registry entry, and a
  /// `model_index.json`-named transformer that detects as the `raw` variant.
  func testKromaV03BaseIsADeclaredRawVariantBase() throws {
    let spec = WarmServer.parseModelSpec(from: "kroma-v0.3-base")
    XCTAssertTrue(spec.hasSuffix("/LocalModels/kroma-v0.3-base"), spec)
    XCTAssertTrue(spec.hasPrefix("/"), "tilde must be expanded: \(spec)")
    XCTAssertTrue(Krea2ModelDetection.isKnownKrea2Model("kroma-v0.3-base"))
    XCTAssertEqual(Krea2ModelDetection.alias(forSpec: "~/LocalModels/kroma-v0.3-base"), "kroma-v0.3-base")

    let model = try XCTUnwrap(ComfyBoxModelRegistry.models["kroma-v0.3-base"])
    XCTAssertEqual(model.family, .krea2)
    XCTAssertEqual(model.huggingFaceId, "lodestones/Kroma")
    XCTAssertTrue(model.supportsGuidance, "an undistilled base honours CFG like Raw")
    XCTAssertTrue(model.supportsLoRA)

    // Layout contract, on a scratch dir so the test needs no 26GB file: the
    // transformer keeps its real filename and model_index.json declares it.
    let scratch = FileManager.default.temporaryDirectory.appending(path: "kroma-v03-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch.appending(path: "text_encoder"), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: scratch.appending(path: "vae"), withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: scratch) }
    FileManager.default.createFile(atPath: scratch.appending(path: "text_encoder/model.safetensors").path, contents: Data())
    FileManager.default.createFile(atPath: scratch.appending(path: "vae/diffusion_pytorch_model.safetensors").path, contents: Data())
    FileManager.default.createFile(atPath: scratch.appending(path: "kroma-v0.3-base.safetensors").path, contents: Data())
    try Data(#"{"krea2_variant":"raw","transformer_file":"kroma-v0.3-base.safetensors"}"#.utf8)
      .write(to: scratch.appending(path: "model_index.json"))
    let paths = try Krea2ModelDetection.detect(at: scratch)
    XCTAssertEqual(paths.variant, .raw, "Kroma v0.3 base is undistilled — a raw variant, never turbo")
    XCTAssertEqual(paths.transformerFile.lastPathComponent, "kroma-v0.3-base.safetensors")
    XCTAssertEqual(Krea2ModelDetection.detectVariant(spec: "kv03", specDirectories: ["kv03": scratch.path]), .raw)
  }

  func testKromaRootDetectsAsKrea2WhenAssembled() throws {
    // Layout contract: an explicit dir with turbo.safetensors + TE/VAE files
    // is a Krea-2 model root. Skip when the checkpoint isn't downloaded yet.
    let root = URL(fileURLWithPath: NSString(string: "~/LocalModels/kroma-v0.2").expandingTildeInPath)
    guard FileManager.default.fileExists(atPath: root.appending(path: "turbo.safetensors").path) else {
      throw XCTSkip("kroma-v0.2 checkpoint not downloaded on this machine")
    }
    let paths = try Krea2ModelDetection.detect(at: root)
    XCTAssertEqual(paths.variant, .turbo, "Kroma ships as a full turbo checkpoint")
    XCTAssertEqual(paths.transformerFile.lastPathComponent, "turbo.safetensors")
  }
}

/// Winner-action basename matching must tolerate the daemon's temp prefix
/// (sidecars recorded "1786475197556_ltx2-….mp4" — 2026-08-11 extend 404).
final class TimestampPrefixTests: XCTestCase {
  func testStripsEpochPrefix() {
    XCTAssertEqual(
      WarmServer.stripTimestampPrefix("1786475197556_ltx2-BCC184B2.mp4"),
      "ltx2-BCC184B2.mp4")
  }
  func testLeavesCleanNamesAndShortPrefixesAlone() {
    XCTAssertEqual(WarmServer.stripTimestampPrefix("ltx2-ABC.mp4"), "ltx2-ABC.mp4")
    XCTAssertEqual(WarmServer.stripTimestampPrefix("97_clip.mp4"), "97_clip.mp4")
    XCTAssertEqual(WarmServer.stripTimestampPrefix("comfybox-kroma-v0.2-avocado.png"),
                   "comfybox-kroma-v0.2-avocado.png")
  }
}
