import XCTest
import MLX
@testable import ZImage

/// WP-E9 — VAE selection (FDD §3.9, D16, D17; AC-56 / AC-57 / AC-59).
/// Precedence: `payload.vae` → model dir (`model_index.json` `"vae_file"`,
/// else `vae/diffusion_pytorch_model.safetensors`). A named VAE that is not
/// on disk FAILS the render with a path-naming error — never a fallback. The
/// decoder is reloaded in place on one `Krea2VAE` instance (never pool-keyed).
final class Krea2VAESelectionTests: XCTestCase {

  private var scratch: URL!

  override func setUpWithError() throws {
    scratch = FileManager.default.temporaryDirectory
      .appending(path: "krea2-vae-sel-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: scratch)
  }

  /// A Krea-2 model root by file presence (the Krea2VariantDetectionTests fixture).
  private func makeRoot(_ name: String, modelIndex: String? = nil, extraVAE: String? = nil) throws -> URL {
    let root = scratch.appending(path: name)
    let fm = FileManager.default
    try fm.createDirectory(at: root.appending(path: "text_encoder"), withIntermediateDirectories: true)
    try fm.createDirectory(at: root.appending(path: "vae"), withIntermediateDirectories: true)
    fm.createFile(atPath: root.appending(path: "raw.safetensors").path, contents: Data([0]))
    fm.createFile(atPath: root.appending(path: "text_encoder/model.safetensors").path, contents: Data([0]))
    fm.createFile(atPath: root.appending(path: "vae/diffusion_pytorch_model.safetensors").path, contents: Data([0]))
    if let extraVAE {
      fm.createFile(atPath: root.appending(path: extraVAE).path, contents: Data([0]))
    }
    if let modelIndex {
      try modelIndex.write(to: root.appending(path: "model_index.json"), atomically: true, encoding: .utf8)
    }
    return root
  }

  // MARK: - Precedence (weight-free)

  func testModelDirDefaultIsTheDiffusersFile() throws {
    let root = try makeRoot("plain")
    let paths = try Krea2ModelDetection.detect(at: root)
    let sel = try Krea2VAESelector.resolve(requested: nil, paths: paths)
    XCTAssertEqual(sel.file.standardizedFileURL, root.appending(path: "vae/diffusion_pytorch_model.safetensors").standardizedFileURL)
    XCTAssertEqual(sel.source, .modelDir)
  }

  func testModelIndexVaeFileOverridesTheDefault() throws {
    let root = try makeRoot(
      "indexed",
      modelIndex: #"{"krea2_variant":"raw","transformer_file":"raw.safetensors","vae_file":"vae/wan.safetensors"}"#,
      extraVAE: "vae/wan.safetensors")
    let paths = try Krea2ModelDetection.detect(at: root)
    XCTAssertEqual(paths.vaeFile.standardizedFileURL, root.appending(path: "vae/wan.safetensors").standardizedFileURL)
    let sel = try Krea2VAESelector.resolve(requested: nil, paths: paths)
    XCTAssertEqual(sel.file.standardizedFileURL, root.appending(path: "vae/wan.safetensors").standardizedFileURL)
    XCTAssertEqual(sel.source, .modelDir)
  }

  func testModelIndexVaeFileThatDoesNotExistFailsDetection() throws {
    let root = try makeRoot(
      "indexed-missing",
      modelIndex: #"{"krea2_variant":"raw","transformer_file":"raw.safetensors","vae_file":"vae/nope.safetensors"}"#)
    XCTAssertThrowsError(try Krea2ModelDetection.detect(at: root)) { error in
      guard case Krea2ModelPathsError.notAKrea2ModelDirectory(_, let reason) = error,
            case .invalidModelIndex(let detail) = reason else {
        return XCTFail("expected invalidModelIndex, got \(error)")
      }
      XCTAssertTrue(detail.contains("vae/nope.safetensors"), detail)
    }
  }

  func testPayloadVAEWinsOverModelDir() throws {
    let root = try makeRoot("payload", extraVAE: "vae/other.safetensors")
    let paths = try Krea2ModelDetection.detect(at: root)
    let requested = root.appending(path: "vae/other.safetensors").path
    let sel = try Krea2VAESelector.resolve(requested: requested, paths: paths)
    XCTAssertEqual(sel.file.standardizedFileURL, URL(fileURLWithPath: requested).standardizedFileURL)
    XCTAssertEqual(sel.source, .payload)
  }

  /// AC-56: a VAE not on disk fails with a path-naming error — never the model dir's VAE.
  func testMissingVAEFailsTheRenderNamingThePath() throws {
    let root = try makeRoot("missing")
    let paths = try Krea2ModelDetection.detect(at: root)
    let missing = root.appending(path: "vae/Wan2_1_VAE_fp32.safetensors").path
    XCTAssertThrowsError(try Krea2VAESelector.resolve(requested: missing, paths: paths)) { error in
      guard case Krea2VAESelectionError.vaeNotFound(let path, let source) = error else {
        return XCTFail("expected vaeNotFound, got \(error)")
      }
      XCTAssertEqual(path, missing)
      XCTAssertEqual(source, .payload)
      XCTAssertTrue(error.localizedDescription.contains(missing), error.localizedDescription)
    }
    // Tilde-expanded, so "~/nope.safetensors" names the expanded path, not a literal tilde.
    XCTAssertThrowsError(try Krea2VAESelector.resolve(requested: "~/definitely-not-here-\(UUID()).safetensors", paths: paths)) { error in
      guard case Krea2VAESelectionError.vaeNotFound(let path, _) = error else { return XCTFail("\(error)") }
      XCTAssertFalse(path.hasPrefix("~"))
    }
    // An empty string is not "no selection" — it is a bad selection; so is a directory.
    XCTAssertThrowsError(try Krea2VAESelector.resolve(requested: "", paths: paths))
    XCTAssertThrowsError(try Krea2VAESelector.resolve(requested: "   ", paths: paths))
    XCTAssertThrowsError(try Krea2VAESelector.resolve(requested: root.appending(path: "vae").path, paths: paths)) { error in
      guard case Krea2VAESelectionError.vaeNotFound(let path, _) = error else { return XCTFail("\(error)") }
      XCTAssertEqual(path, root.appending(path: "vae").path)
    }
  }

  // MARK: - The slot: one instance, in-place reload, fail-closed (weight-free)

  func testSlotRefusesMissingFileAndLeavesResidentUntouched() throws {
    let resident = scratch.appending(path: "resident.safetensors")
    let slot = Krea2VAESlot(unloaded: Krea2VAE(), file: resident, layout: .qwenDiffusers, source: .modelDir)
    let before = slot.vae
    let missing = scratch.appending(path: "nope.safetensors")
    XCTAssertThrowsError(try slot.ensure(file: missing, source: .payload)) { error in
      guard case Krea2VAESelectionError.vaeNotFound(let path, _) = error else { return XCTFail("\(error)") }
      XCTAssertEqual(path, missing.path)
    }
    XCTAssertTrue(slot.vae === before, "the resident instance is untouched")
    XCTAssertEqual(slot.reloadCount, 0)
    XCTAssertEqual(slot.current.file.standardizedFileURL, resident.standardizedFileURL)
    XCTAssertEqual(slot.current.source, .modelDir)
  }

  func testSlotRefusesUnrecognisedLayoutAndLeavesResidentUntouched() throws {
    let resident = scratch.appending(path: "resident.safetensors")
    let slot = Krea2VAESlot(unloaded: Krea2VAE(), file: resident, layout: .qwenDiffusers, source: .modelDir)
    let third = scratch.appending(path: "third.safetensors")
    try Krea2VAEFixtures.writeSafetensors(keys: ["decoder.blocks.0.weight": [4, 4]], to: third)
    XCTAssertThrowsError(try slot.ensure(file: third, source: .payload)) { error in
      guard case Krea2VAEKeyMapError.unrecognizedVAELayout(let file) = error else { return XCTFail("\(error)") }
      XCTAssertEqual(file, third.path)
    }
    XCTAssertEqual(slot.reloadCount, 0)
    XCTAssertEqual(slot.current.layout, .qwenDiffusers)
    XCTAssertEqual(slot.current.file.standardizedFileURL, resident.standardizedFileURL)
  }

  /// AC-59 (first half): requesting the resident VAE does not reload.
  func testRequestingTheResidentVAEDoesNotReload() throws {
    let resident = scratch.appending(path: "resident.safetensors")
    FileManager.default.createFile(atPath: resident.path, contents: Data([0]))
    let slot = Krea2VAESlot(unloaded: Krea2VAE(), file: resident, layout: .qwenDiffusers, source: .modelDir)
    let reloaded = try slot.ensure(file: resident, source: .payload)
    XCTAssertFalse(reloaded)
    XCTAssertEqual(slot.reloadCount, 0)
    // The record names what decoded and how it was selected.
    XCTAssertEqual(slot.current.source, .payload)
    XCTAssertEqual(slot.current.file.standardizedFileURL, resident.standardizedFileURL)
  }

  // MARK: - Real files: in-place swap and single instance

  /// AC-59 (second half) + AC-57: Wan while Qwen is resident increments the
  /// reload counter, the record names the Wan file, the `Krea2VAE` instance
  /// is the same object before and after (so encode and decode can never
  /// disagree), and the weights actually changed.
  func testSwapReloadsInPlaceOnTheSameInstance() throws {
    try Krea2VAEFixtures.requireBoth()
    let slot = try Krea2VAESlot(loading: Krea2VAEFixtures.qwen, source: .modelDir)
    XCTAssertEqual(slot.current.layout, .qwenDiffusers)
    let instance = slot.vae
    // Snapshot by VALUE: the in-place reload mutates the parameter's storage,
    // so a lazy view of it would follow the swap and prove nothing.
    let qwenProbe = instance.decoder.convIn.weight.asArray(Float.self)

    XCTAssertTrue(try slot.ensure(file: Krea2VAEFixtures.wan, source: .payload))
    XCTAssertEqual(slot.reloadCount, 1)
    XCTAssertEqual(slot.current.file.standardizedFileURL, Krea2VAEFixtures.wan.standardizedFileURL)
    XCTAssertEqual(slot.current.layout, .wanNative)
    XCTAssertEqual(slot.current.source, .payload)
    XCTAssertTrue(slot.vae === instance, "in-place reload — the pool never sees a new pipeline")
    let wanProbe = instance.decoder.convIn.weight.asArray(Float.self)
    XCTAssertEqual(wanProbe.count, qwenProbe.count)
    let maxDiff = zip(wanProbe, qwenProbe).map { abs($0 - $1) }.max() ?? 0
    XCTAssertGreaterThan(maxDiff, 0, "weights did not change")

    // Asking for Wan again is a no-op; asking for Qwen reloads again.
    XCTAssertFalse(try slot.ensure(file: Krea2VAEFixtures.wan, source: .payload))
    XCTAssertEqual(slot.reloadCount, 1)
    XCTAssertTrue(try slot.ensure(file: Krea2VAEFixtures.qwen, source: .modelDir))
    XCTAssertEqual(slot.reloadCount, 2)
    XCTAssertEqual(slot.current.layout, .qwenDiffusers)
  }

  /// AC-57: img2img's encoder IS the decoder — one `Krea2VAE` instance serves
  /// both, asserted by identity through the public pipeline surface.
  func testSingleInstance() throws {
    try Krea2VAEFixtures.requireBoth()
    let slot = try Krea2VAESlot(loading: Krea2VAEFixtures.qwen, source: .modelDir)
    _ = try slot.ensure(file: Krea2VAEFixtures.wan, source: .payload)
    // The encode path and the decode path are the same object.
    XCTAssertTrue(slot.encoder === slot.decoder)
    XCTAssertTrue(slot.encoder === slot.vae)
    // And they agree on a round trip: encode → decode of a small image
    // reproduces it through the Wan weights (no NaN, recognisable).
    let size = 64
    var px = [Float](repeating: 0, count: size * size * 3)
    for i in 0..<px.count { px[i] = Float((i / 3) % size) / Float(size - 1) * 2 - 1 }
    let source = MLXArray(px, [1, size, size, 3])
    let latents = slot.encoder.encode(source)
    let decoded = slot.decoder.decode(latents)
    MLX.eval(decoded)
    XCTAssertEqual(decoded.shape, [1, size, size, 3])
    XCTAssertFalse(MLX.any(MLX.isNaN(decoded)).item(Bool.self))
    let mae = MLX.abs(decoded - (source + 1) * 0.5).mean().item(Float.self)
    XCTAssertLessThan(mae, 0.15, "Wan round trip is not recognisable (mae=\(mae))")
  }
}
