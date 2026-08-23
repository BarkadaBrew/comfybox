import XCTest
import MLX
@testable import ZImage

/// Real-file fixtures for the O7 decoder tests (FDD §5.1: header-only reads,
/// gated with `XCTSkipUnless(fileExists)` — the `RealVAEExactTests` pattern).
enum Krea2VAEFixtures {
  static let wan = URL(fileURLWithPath:
    ("~/LocalModels/vae/Wan2_1_VAE_fp32.safetensors" as NSString).expandingTildeInPath)
  static let qwen = URL(fileURLWithPath:
    ("~/LocalModels/kroma-v0.2/vae/diffusion_pytorch_model.safetensors" as NSString).expandingTildeInPath)
  static let qwenConfig = URL(fileURLWithPath:
    ("~/LocalModels/kroma-v0.2/vae/config.json" as NSString).expandingTildeInPath)

  static func requireBoth() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: wan.path), "Wan 2.1 FP32 VAE not on disk")
    try XCTSkipUnless(FileManager.default.fileExists(atPath: qwen.path), "kroma-v0.2 Qwen VAE not on disk")
  }

  /// Write a minimal, valid safetensors file holding the named tensors (all
  /// F32 zeros) so a "third layout" can be sniffed without a real model.
  static func writeSafetensors(keys: [String: [Int]], to url: URL) throws {
    var header: [String: Any] = [:]
    var offset = 0
    for (key, shape) in keys.sorted(by: { $0.key < $1.key }) {
      let bytes = shape.reduce(1, *) * 4
      header[key] = ["dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]]
      offset += bytes
    }
    let headerData = try JSONSerialization.data(withJSONObject: header)
    var data = Data()
    var len = UInt64(headerData.count).littleEndian
    data.append(Data(bytes: &len, count: 8))
    data.append(headerData)
    data.append(Data(count: offset))
    try data.write(to: url)
  }
}

/// WP-E9 — `Krea2VAEKeyMap` (FDD §3.9, AC-52 / AC-53). The Wan 2.1 FP32 file
/// and the Qwen-Image VAE share zero key names; the map is a bijection over
/// 194 keys, asserted here against BOTH real headers in BOTH directions.
/// Detection is by key sniff, never by filename.
final class Krea2VAEKeyMapTests: XCTestCase {

  // MARK: - Rule table (weight-free)

  func testRuleTable() {
    let cases: [(String, String)] = [
      ("conv1.weight", "quant_conv.weight"),
      ("conv2.bias", "post_quant_conv.bias"),
      ("encoder.conv1.weight", "encoder.conv_in.weight"),
      ("decoder.conv1.bias", "decoder.conv_in.bias"),
      ("encoder.head.0.gamma", "encoder.norm_out.gamma"),
      ("decoder.head.0.gamma", "decoder.norm_out.gamma"),
      ("encoder.head.2.weight", "encoder.conv_out.weight"),
      ("decoder.head.2.bias", "decoder.conv_out.bias"),
      ("encoder.middle.0.residual.0.gamma", "encoder.mid_block.resnets.0.norm1.gamma"),
      ("encoder.middle.0.residual.2.weight", "encoder.mid_block.resnets.0.conv1.weight"),
      ("decoder.middle.2.residual.3.gamma", "decoder.mid_block.resnets.1.norm2.gamma"),
      ("decoder.middle.2.residual.6.bias", "decoder.mid_block.resnets.1.conv2.bias"),
      ("encoder.middle.1.norm.gamma", "encoder.mid_block.attentions.0.norm.gamma"),
      ("encoder.middle.1.to_qkv.weight", "encoder.mid_block.attentions.0.to_qkv.weight"),
      ("decoder.middle.1.proj.bias", "decoder.mid_block.attentions.0.proj.bias"),
      ("encoder.downsamples.0.residual.0.gamma", "encoder.down_blocks.0.norm1.gamma"),
      ("encoder.downsamples.3.shortcut.weight", "encoder.down_blocks.3.conv_shortcut.weight"),
      ("encoder.downsamples.2.resample.1.weight", "encoder.down_blocks.2.resample.1.weight"),
      ("encoder.downsamples.5.time_conv.bias", "encoder.down_blocks.5.time_conv.bias"),
      ("encoder.downsamples.10.residual.6.weight", "encoder.down_blocks.10.conv2.weight"),
      ("decoder.upsamples.0.residual.0.gamma", "decoder.up_blocks.0.resnets.0.norm1.gamma"),
      ("decoder.upsamples.2.residual.6.bias", "decoder.up_blocks.0.resnets.2.conv2.bias"),
      ("decoder.upsamples.3.resample.1.weight", "decoder.up_blocks.0.upsamplers.0.resample.1.weight"),
      ("decoder.upsamples.3.time_conv.weight", "decoder.up_blocks.0.upsamplers.0.time_conv.weight"),
      ("decoder.upsamples.4.shortcut.weight", "decoder.up_blocks.1.resnets.0.conv_shortcut.weight"),
      ("decoder.upsamples.7.resample.1.bias", "decoder.up_blocks.1.upsamplers.0.resample.1.bias"),
      ("decoder.upsamples.11.resample.1.weight", "decoder.up_blocks.2.upsamplers.0.resample.1.weight"),
      ("decoder.upsamples.12.residual.2.weight", "decoder.up_blocks.3.resnets.0.conv1.weight"),
      ("decoder.upsamples.14.residual.3.gamma", "decoder.up_blocks.3.resnets.2.norm2.gamma"),
    ]
    for (wan, qwen) in cases {
      XCTAssertEqual(Krea2VAEKeyMap.canonicalize(wan), qwen, wan)
    }
  }

  func testCanonicalizeIsIdentityOnQwenKeys() {
    let qwen = [
      "quant_conv.weight", "post_quant_conv.bias",
      "encoder.conv_in.weight", "encoder.conv_out.bias", "encoder.norm_out.gamma",
      "encoder.down_blocks.3.conv_shortcut.weight", "encoder.mid_block.attentions.0.to_qkv.weight",
      "decoder.conv_in.weight", "decoder.up_blocks.1.upsamplers.0.resample.1.weight",
      "decoder.up_blocks.3.resnets.2.norm2.gamma", "decoder.mid_block.resnets.1.conv2.bias",
    ]
    for key in qwen {
      XCTAssertEqual(Krea2VAEKeyMap.canonicalize(key), key, key)
    }
  }

  func testUnmappedKeysAreNil() {
    // Beyond the decoder's 15 flat entries.
    XCTAssertNil(Krea2VAEKeyMap.canonicalize("decoder.upsamples.15.residual.0.gamma"))
    // Slot 3 of a decoder block is the upsampler; a residual there is not a Krea2VAE path.
    XCTAssertNil(Krea2VAEKeyMap.canonicalize("decoder.upsamples.3.residual.0.gamma"))
    // residual.1 / residual.4 / residual.5 are the SiLU / dropout slots — no weights.
    XCTAssertNil(Krea2VAEKeyMap.canonicalize("encoder.middle.0.residual.1.weight"))
    XCTAssertNil(Krea2VAEKeyMap.canonicalize("decoder.middle.1.residual.0.gamma"))
    XCTAssertNil(Krea2VAEKeyMap.canonicalize("foo.weight"))
    XCTAssertNil(Krea2VAEKeyMap.canonicalize(""))
  }

  // MARK: - AC-53: detection by key sniff, never by filename

  func testLayoutSniff() throws {
    try Krea2VAEFixtures.requireBoth()
    XCTAssertEqual(try Krea2VAEKeyMap.detectLayout(file: Krea2VAEFixtures.wan), .wanNative)
    XCTAssertEqual(try Krea2VAEKeyMap.detectLayout(file: Krea2VAEFixtures.qwen), .qwenDiffusers)

    // A third file — valid safetensors, neither key space — throws, naming the file.
    let third = FileManager.default.temporaryDirectory
      .appending(path: "third-layout-\(UUID().uuidString).safetensors")
    defer { try? FileManager.default.removeItem(at: third) }
    try Krea2VAEFixtures.writeSafetensors(
      keys: ["decoder.blocks.0.weight": [4, 4], "encoder.stem.weight": [4, 4]], to: third)
    XCTAssertThrowsError(try Krea2VAEKeyMap.detectLayout(file: third)) { error in
      guard case Krea2VAEKeyMapError.unrecognizedVAELayout(let file) = error else {
        return XCTFail("expected unrecognizedVAELayout, got \(error)")
      }
      XCTAssertEqual(file, third.path)
    }
  }

  func testLayoutSniffFromKeysIsPureAndFilenameBlind() throws {
    XCTAssertEqual(Krea2VAEKeyMap.detectLayout(keys: ["decoder.upsamples.0.residual.0.gamma"]), .wanNative)
    XCTAssertEqual(Krea2VAEKeyMap.detectLayout(keys: ["decoder.up_blocks.0.resnets.0.norm1.gamma"]), .qwenDiffusers)
    XCTAssertNil(Krea2VAEKeyMap.detectLayout(keys: ["decoder.blocks.0.weight"]))
    // Both key spaces in one file is not a layout either.
    XCTAssertNil(Krea2VAEKeyMap.detectLayout(
      keys: ["decoder.upsamples.0.residual.0.gamma", "decoder.up_blocks.0.resnets.0.norm1.gamma"]))
  }

  // MARK: - AC-52: the 194/194 bijection against both real files

  func testWanKeysMapOntoQwenKeysWithExactShapeEquality() throws {
    try Krea2VAEFixtures.requireBoth()
    let wan = try SafeTensorsReader(fileURL: Krea2VAEFixtures.wan)
    let qwen = try SafeTensorsReader(fileURL: Krea2VAEFixtures.qwen)
    let qwenShapes = Dictionary(uniqueKeysWithValues: qwen.allMetadata().map { ($0.name, $0.shape) })
    XCTAssertEqual(wan.tensorNames.count, 194)
    XCTAssertEqual(qwenShapes.count, 194)

    var mapped = Set<String>()
    for meta in wan.allMetadata() {
      guard let canonical = Krea2VAEKeyMap.canonicalize(meta.name) else {
        return XCTFail("Wan key \(meta.name) has no Krea2VAE path")
      }
      guard let qwenShape = qwenShapes[canonical] else {
        return XCTFail("Wan key \(meta.name) → \(canonical) is not a Qwen-Image VAE key")
      }
      XCTAssertEqual(meta.shape, qwenShape, "\(meta.name) → \(canonical)")
      XCTAssertTrue(mapped.insert(canonical).inserted, "two Wan keys collapsed onto \(canonical)")
    }
    XCTAssertEqual(mapped.count, 194, "194/194 Wan keys mapped")
    XCTAssertEqual(mapped, Set(qwenShapes.keys), "194/194 Qwen keys covered — the map is a bijection")
  }

  /// Both directions land on real `Krea2VAE` module parameters (after the
  /// loader's `.resample.1.` → `.resample.conv.` rename); `time_conv` is the
  /// one deliberate skip (dead code for images — Krea2VAE.swift).
  func testEveryMappedKeyIsAKrea2VAEParameter() throws {
    try Krea2VAEFixtures.requireBoth()
    let modulePaths = Set(Krea2VAE().parameters().flattened().map { $0.0 })
    for file in [Krea2VAEFixtures.wan, Krea2VAEFixtures.qwen] {
      let reader = try SafeTensorsReader(fileURL: file)
      for name in reader.tensorNames {
        let canonical = try XCTUnwrap(Krea2VAEKeyMap.canonicalize(name), name)
        if canonical.contains(".time_conv.") { continue }
        let path = Krea2WeightLoader.vaeModulePath(forCanonicalKey: canonical)
        XCTAssertTrue(modulePaths.contains(path), "\(name) → \(path) is not a Krea2VAE parameter")
      }
    }
  }
}
