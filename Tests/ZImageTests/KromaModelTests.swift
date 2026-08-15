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

  func testKromaRootDetectsAsKrea2WhenAssembled() throws {
    // Layout contract: an explicit dir with turbo.safetensors + TE/VAE files
    // is a Krea-2 model root. Skip when the checkpoint isn't downloaded yet.
    let root = URL(fileURLWithPath: NSString(string: "~/LocalModels/kroma-v0.2").expandingTildeInPath)
    guard FileManager.default.fileExists(atPath: root.appending(path: "turbo.safetensors").path) else {
      throw XCTSkip("kroma-v0.2 checkpoint not downloaded on this machine")
    }
    XCTAssertNotNil(Krea2ModelDetection.detect(at: root), "assembled root must detect as Krea-2")
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
