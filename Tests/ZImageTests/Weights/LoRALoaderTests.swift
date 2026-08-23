import MLX
import XCTest
@testable import ZImage

final class LoRALoaderTests: XCTestCase {

  func testLoRAUnetPrefixRemoval() {
    let input = "lora_unet_transformer_blocks.0.attn.to_q"
    let expected = "layers.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testDiffusionModelPrefixRemoval() {
    let input = "diffusion_model.layers.0.attention.to_q"
    let expected = "layers.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testNoPrefixUnchanged() {
    let input = "layers.0.attention.to_q.weight"
    let expected = "layers.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testFFLayerMapping_Net0() {
    let input = "transformer_blocks.5.ff.net.0.proj"
    let expected = "layers.5.feed_forward.w1.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testFFLayerMapping_Net2() {
    let input = "transformer_blocks.5.ff.net.2"
    let expected = "layers.5.feed_forward.w2.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testAttentionKeyMapping() {
    let testCases = [
      ("transformer_blocks.0.attn.to_q", "layers.0.attention.to_q.weight"),
      ("transformer_blocks.0.attn.to_k", "layers.0.attention.to_k.weight"),
      ("transformer_blocks.0.attn.to_v", "layers.0.attention.to_v.weight"),
      ("transformer_blocks.0.attn.to_out.0", "layers.0.attention.to_out.0.weight"),
    ]

    for (input, expected) in testCases {
      let result = LoRAKeyMapper.mapToZImageKey(input)
      XCTAssertEqual(result, expected, "Failed for input: \(input)")
    }
  }

  func testLoRAKeyWithLoraunnetPrefix() {
    let testCases = [
      ("lora_unet_transformer_blocks.0.attn.to_q", "layers.0.attention.to_q.weight"),
      ("lora_unet_transformer_blocks.5.ff.net.0.proj", "layers.5.feed_forward.w1.weight"),
    ]

    for (input, expected) in testCases {
      let result = LoRAKeyMapper.mapToZImageKey(input)
      XCTAssertEqual(result, expected, "Failed for input: \(input)")
    }
  }

  func testNoiseRefinerKeys() {
    let input = "noise_refiner.0.attn.to_q"
    let expected = "noise_refiner.0.attention.to_q.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testMfluxCheckpointStyleKeysMapDirectly() {
    let input = "diffusion_model.layers.12.feed_forward.w3"
    let expected = "layers.12.feed_forward.w3.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
  }

  func testAdaLNTargetsAreRecognized() {
    let input = "diffusion_model.layers.0.adaLN_modulation.0"
    let expected = "layers.0.adaLN_modulation.0.weight"

    let result = LoRAKeyMapper.mapToZImageKey(input)
    XCTAssertEqual(result, expected)
    XCTAssertTrue(LoRAKeyMapper.isValidTarget(expected))
  }

  func testValidTargetPaths() {
    XCTAssertTrue(LoRAKeyMapper.isValidTarget("layers.0.attention.to_q.weight"))
    XCTAssertTrue(LoRAKeyMapper.isValidTarget("layers.0.feed_forward.w1.weight"))
    XCTAssertTrue(LoRAKeyMapper.isValidTarget("noise_refiner.0.attention.to_q.weight"))
    XCTAssertTrue(LoRAKeyMapper.isValidTarget("context_refiner.0.attention.to_k.weight"))
  }

  func testInvalidTargetPaths() {
    XCTAssertFalse(LoRAKeyMapper.isValidTarget("invalid.path.weight"))
    XCTAssertFalse(LoRAKeyMapper.isValidTarget("layers.99.attention.to_q.weight"))
  }

  func testLoRAErrorFileNotFound() {
    let error = LoRAError.fileNotFound("/nonexistent/path")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("/nonexistent/path"))
  }

  func testLoRAErrorInvalidFormat() {
    let error = LoRAError.invalidFormat("missing keys")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("missing keys"))
  }

  func testLoRAErrorIncompatibleWeights() {
    let error = LoRAError.incompatibleWeights("Shape mismatch")
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("Shape mismatch"))
  }

  func testLoRAErrorNoSafetensorsFound() {
    let url = URL(fileURLWithPath: "/some/path")
    let error = LoRAError.noSafetensorsFound(url)
    XCTAssertNotNil(error.errorDescription)
    XCTAssertTrue(error.errorDescription!.contains("/some/path"))
  }

  func testSupportedTargetPathsCount() {

    let paths = LoRAKeyMapper.supportedTargetPaths
    XCTAssertEqual(paths.count, 272)
  }

  func testLoRAConfigurationDoesNotClampScale() {
    XCTAssertEqual(LoRAConfiguration.local("/tmp/test.safetensors", scale: -2).scale, -2)
    XCTAssertEqual(LoRAConfiguration.local("/tmp/test.safetensors", scale: 1.5).scale, 1.5)
  }
}

// MARK: - WP-E6 B4a: fail-loud preflight on a JSON error page named .safetensors

/// FDD §3.6 / D10 / AC-49. `scratchpad/fetch.log:3` records a 99-byte civitai
/// auth/early-access JSON body saved as `krea2filterbypass_2vector.safetensors`.
/// Such a file must throw `notASafetensorsFile` quoting the payload — never a
/// `SafeTensorsReader` internal error, never a silent skip — and the guard is
/// NEVER a size floor: the real 1,040-byte bypass file must still load.
extension LoRALoaderTests {

  private func writeErrorPage(_ body: String, name: String = "krea2filterbypass_2vector.safetensors") throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("lora-preflight-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let url = dir.appendingPathComponent(name)
    try body.data(using: .utf8)!.write(to: url)
    return url
  }

  func testJSONErrorPageRejected() throws {
    let body = #"{"error":"Early Access: this model version requires purchase"}"#
    let url = try writeErrorPage(body)
    defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }

    for (label, loader) in [
      ("loadForKrea2", { try LoRAWeightLoader.loadForKrea2(from: url) }),
      ("load", { try LoRAWeightLoader.load(from: url) }),
      ("loadForFlux2", { try LoRAWeightLoader.loadForFlux2(from: url) }),
    ] as [(String, () throws -> LoRAWeights)] {
      XCTAssertThrowsError(try loader(), label) { error in
        guard case LoRAError.notASafetensorsFile(let path, let firstBytes) = error else {
          return XCTFail("\(label): expected notASafetensorsFile, got \(error)")
        }
        XCTAssertEqual(path, url.path, label)
        XCTAssertTrue(firstBytes.contains("Early Access"), "\(label): must quote the payload: \(firstBytes)")
        XCTAssertTrue(error.localizedDescription.contains("Early Access"), error.localizedDescription)
      }
    }
  }

  /// Leading whitespace / a JSON array body are still JSON; a random binary
  /// blob is NOT re-labelled as JSON (the reader's own error surfaces).
  func testPreflightSniffsJSONNotArbitraryGarbage() throws {
    let arrayPage = try writeErrorPage("  \n[{\"message\":\"unauthorized\"}]")
    defer { try? FileManager.default.removeItem(at: arrayPage.deletingLastPathComponent()) }
    XCTAssertThrowsError(try LoRAWeightLoader.loadForKrea2(from: arrayPage)) { error in
      guard case LoRAError.notASafetensorsFile = error else {
        return XCTFail("expected notASafetensorsFile, got \(error)")
      }
    }

    let garbage = try writeErrorPage("\u{01}\u{02}garbage-not-json-and-not-safetensors")
    defer { try? FileManager.default.removeItem(at: garbage.deletingLastPathComponent()) }
    XCTAssertThrowsError(try LoRAWeightLoader.loadForKrea2(from: garbage)) { error in
      if case LoRAError.notASafetensorsFile = error {
        XCTFail("binary garbage is not a JSON error page — must not be mislabelled")
      }
    }

    // A corrupt safetensors whose header length is 123 starts with 0x7B — `{`.
    // The zero bytes 1…7 mark it as a binary length prefix, not JSON.
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("lora-preflight-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let corrupt = dir.appendingPathComponent("corrupt.safetensors")
    try Data([0x7B, 0, 0, 0, 0, 0, 0, 0] + [UInt8](repeating: 0xFF, count: 40)).write(to: corrupt)
    XCTAssertThrowsError(try LoRAWeightLoader.loadForKrea2(from: corrupt)) { error in
      if case LoRAError.notASafetensorsFile = error {
        XCTFail("a binary length prefix beginning 0x7B is not JSON — must surface the reader's own error")
      }
    }
  }

  /// AC-49 second clause (and AC-47's Fedor half): the real 1,040-byte bypass
  /// artifact loads — 0 pairs, 1 delta on `txtfusion.projector`, F32 [1, 12].
  func testRealBypassFileStillLoads() throws {
    let url = URL(fileURLWithPath: NSString(
      string: "~/comfybox-models/loras/vault/krea2_filter_bypass_fedor.safetensors").expandingTildeInPath)
    try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path),
                      "krea2_filter_bypass_fedor.safetensors not in the vault on this machine")

    let size = try FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int
    XCTAssertEqual(size, 1040, "the artifact on disk is the 1,040-byte Fedor file")

    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(weights.weights.count, 0)
    XCTAssertEqual(weights.deltas.count, 1)
    guard case .diff(let t) = try XCTUnwrap(weights.deltas["txtfusion.projector"]) else {
      return XCTFail("expected a .diff delta on txtfusion.projector, got \(weights.deltas.keys.sorted())")
    }
    XCTAssertEqual(t.shape, [1, 12])
    XCTAssertEqual(t.dtype, .float32)
  }
}
