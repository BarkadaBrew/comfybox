import XCTest
import MLX
@testable import ZImage

/// WP-E10 "E9b" (FDD Addendum A.2, E9 review MAJOR): a VAE file holding only a
/// SUBSET of the decoder's expected keys used to load a MIXED decoder silently
/// — the subset overwrote its targets and every other parameter kept whatever
/// was resident. That is a silent substitution of the worst kind (half one
/// VAE, half another, named as the new file). The loader now checks key-set
/// completeness against the module's own parameter set BEFORE the first
/// weight is written and throws `vaeIncomplete` naming the missing keys.
final class Krea2VAECompletenessTests: XCTestCase {

  private var scratch: URL!

  override func setUpWithError() throws {
    scratch = FileManager.default.temporaryDirectory
      .appending(path: "krea2-vae-complete-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: scratch)
  }

  /// A Qwen-layout file carrying exactly two (shape-correct) gamma tensors.
  private func writeSubsetFile() throws -> (file: URL, expectedCount: Int) {
    let probe = Krea2VAE()
    let resident = Dictionary(uniqueKeysWithValues: probe.parameters().flattened())
    let normOut = try XCTUnwrap(resident["decoder.norm_out.gamma"])
    let norm1 = try XCTUnwrap(resident["decoder.up_blocks.0.resnets.0.norm1.gamma"])
    let file = scratch.appending(path: "subset.safetensors")
    // Checkpoint gammas are [C,1,1,1]; the loader flattens them to [C].
    try Krea2VAEFixtures.writeSafetensors(
      keys: [
        "decoder.norm_out.gamma": [normOut.dim(0), 1, 1, 1],
        "decoder.up_blocks.0.resnets.0.norm1.gamma": [norm1.dim(0), 1, 1, 1],
      ],
      to: file)
    return (file, resident.count)
  }

  func testSubsetFileIsRefusedAsIncompleteNamingMissingKeys() throws {
    let (file, expectedCount) = try writeSubsetFile()
    let vae = Krea2VAE()
    XCTAssertThrowsError(try Krea2WeightLoader.loadVAE(vae, from: file)) { error in
      guard case Krea2VAEKeyMapError.vaeIncomplete(let named, let missing, let expected) = error else {
        return XCTFail("expected vaeIncomplete, got \(error)")
      }
      XCTAssertEqual(named, file.path)
      XCTAssertEqual(expected, expectedCount)
      XCTAssertEqual(missing.count, expectedCount - 2, "every module parameter the file did not carry")
      XCTAssertFalse(missing.contains("decoder.norm_out.gamma"))
      XCTAssertTrue(missing.contains("decoder.conv_out.weight"))
      XCTAssertEqual(missing, missing.sorted(), "missing keys are reported sorted")
      let text = error.localizedDescription
      XCTAssertTrue(text.contains(file.path), text)
      XCTAssertTrue(text.contains("\(missing.count)"), text)
    }
  }

  /// Fail-closed: the resident decoder is untouched by a refused subset file —
  /// the two gammas the file DID carry (zeros) must not have landed.
  func testRefusedSubsetLeavesResidentUntouched() throws {
    let (file, _) = try writeSubsetFile()
    let vae = Krea2VAE()
    let before = Dictionary(uniqueKeysWithValues: vae.parameters().flattened())["decoder.norm_out.gamma"]!
      .asArray(Float.self)
    XCTAssertThrowsError(try Krea2WeightLoader.loadVAE(vae, from: file))
    let after = Dictionary(uniqueKeysWithValues: vae.parameters().flattened())["decoder.norm_out.gamma"]!
      .asArray(Float.self)
    XCTAssertEqual(before, after, "a refused load must not write even the keys it had")
  }

  /// The slot path (what `Krea2Pipeline.ensureVAE` goes through) refuses too,
  /// with no reload counted and the record still naming the resident file.
  func testSlotRefusesIncompleteFile() throws {
    let (file, _) = try writeSubsetFile()
    let resident = scratch.appending(path: "resident.safetensors")
    let slot = Krea2VAESlot(unloaded: Krea2VAE(), file: resident, layout: .qwenDiffusers, source: .modelDir)
    XCTAssertThrowsError(try slot.ensure(file: file, source: .payload)) { error in
      guard case Krea2VAEKeyMapError.vaeIncomplete = error else { return XCTFail("\(error)") }
    }
    XCTAssertEqual(slot.reloadCount, 0)
    XCTAssertEqual(slot.current.file.standardizedFileURL, resident.standardizedFileURL)
    XCTAssertEqual(slot.current.source, .modelDir)
  }

  /// The real files are complete — the check must not refuse a good VAE.
  func testRealFilesAreComplete() throws {
    try Krea2VAEFixtures.requireBoth()
    XCTAssertEqual(try Krea2WeightLoader.loadVAE(Krea2VAE(), from: Krea2VAEFixtures.qwen), .qwenDiffusers)
    XCTAssertEqual(try Krea2WeightLoader.loadVAE(Krea2VAE(), from: Krea2VAEFixtures.wan), .wanNative)
  }
}
