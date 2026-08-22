// Krea2SamplerParityTests.swift — the WP-E3 merge gate, re-runnable.
//
// FDD-krea2-raw-recipe §4 criteria 1 and 2: after the denoise loop moved onto
// `ZImageScheduler`, a default Krea 2 render must be BYTE-identical to the one
// the pre-change engine produced — not "close enough". The fixtures in
// `Tests/ZImageIntegrationTests/Fixtures/` pin the exact request and a digest
// of the image the pre-change engine produced for it.
//
// **What is compared, and why it is not the whole file.** The criterion says
// "PNG SHA-256 equals a fixture hash". Taken literally that check can never
// pass, and not because of the pixels: `QwenImageIO.ImageMetadata.generation`
// builds its parameter JSON from a Swift `Dictionary`, whose key order differs
// from process to process, and `saveImage` embeds that JSON in the file's eXIf
// and iTXt chunks. Two runs of the SAME binary on the SAME request produce
// different whole-file hashes. Measured 2026-08-22: the differing bytes are
// confined to file offsets 233..1474 — exactly the eXIf and iTXt chunks — and
// everything from the first IDAT chunk to EOF is identical. So the comparison
// here is the digest of the IHDR chunk data plus every IDAT chunk's data: the
// raster geometry and the compressed pixels, and nothing else. That is the
// byte-identity the criterion is about, and it IS reproducible.
//
// Skipped, with the reason named, when the model directory is not on this
// machine.

import XCTest
import MLX
import CoreGraphics
import CryptoKit
@testable import ZImage

final class Krea2SamplerParityTests: XCTestCase {

  // MARK: - Fixtures

  static let fixturesDirectory = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures")

  struct Oracle: Decodable {
    struct Request: Decodable {
      let prompt: String
      let seed: UInt64
      let steps: Int
      let guidance: Float
      let width: Int
      let height: Int
      /// img2img only.
      let strength: Float?
      /// img2img only: the fixed source PNG, pinned by path and by hash.
      let sourceImage: String?
      let sourceImageSha256: String?
    }
    let engineModel: String
    let request: Request
    /// IHDR + IDAT digest — see the file comment.
    let imageSha256: String
  }

  func oracle(_ name: String) throws -> Oracle {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(
      Oracle.self, from: try Data(contentsOf: Self.fixturesDirectory.appendingPathComponent(name)))
  }

  // MARK: - Digests

  static func hex(_ digest: SHA256.Digest) -> String {
    digest.map { String(format: "%02x", $0) }.joined()
  }

  func fileSHA256(_ url: URL) throws -> String {
    Self.hex(SHA256.hash(data: try Data(contentsOf: url)))
  }

  /// SHA-256 over the PNG's IHDR chunk data followed by every IDAT chunk's
  /// data — the image, without the metadata chunks whose byte order is
  /// process-dependent.
  ///
  /// A PNG is an 8-byte signature then a chunk sequence, each chunk a 4-byte
  /// big-endian length, a 4-byte type, that many data bytes, and a 4-byte CRC.
  func imageSHA256(ofPNGAt url: URL) throws -> String {
    let data = try Data(contentsOf: url)
    let signature: [UInt8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    guard data.count > 8, Array(data.prefix(8)) == signature else {
      throw XCTSkip("not a PNG: \(url.path)")
    }
    var hasher = SHA256()
    var offset = 8
    var sawIHDR = false, idatChunks = 0
    while offset + 8 <= data.count {
      let length = data[offset..<(offset + 4)].reduce(0) { $0 << 8 | Int($1) }
      let type = String(decoding: data[(offset + 4)..<(offset + 8)], as: UTF8.self)
      let body = (offset + 8)..<(offset + 8 + length)
      guard body.upperBound <= data.count else { break }
      switch type {
      case "IHDR": hasher.update(data: data[body]); sawIHDR = true
      case "IDAT": hasher.update(data: data[body]); idatChunks += 1
      case "IEND": offset = data.count
      default: break
      }
      if type == "IEND" { break }
      offset += 12 + length
    }
    XCTAssertTrue(sawIHDR, "PNG had no IHDR: \(url.path)")
    XCTAssertGreaterThan(idatChunks, 0, "PNG had no IDAT: \(url.path)")
    return Self.hex(hasher.finalize())
  }

  // MARK: - The pipeline the server builds

  /// q8 transformer (`WarmServer` loads Krea 2 with `quantizeTransformer: 8`),
  /// the model directory's own VAE, no LoRAs — the oracle's configuration.
  func loadPipeline(_ oracle: Oracle) throws -> Krea2Pipeline {
    let root = URL(fileURLWithPath: (oracle.engineModel as NSString).expandingTildeInPath)
    guard FileManager.default.fileExists(atPath: root.path) else {
      throw XCTSkip("Krea 2 oracle model directory is not on this machine: \(root.path)")
    }
    return try Krea2Pipeline(paths: try Krea2ModelDetection.detect(at: root), quantizeTransformer: 8)
  }

  /// Write the render the way `runKrea2Generate` does (same transpose). The
  /// metadata is deliberately omitted: it is not part of what is compared, and
  /// leaving it out keeps the written file itself deterministic.
  func writePNG(_ image: MLXArray, to url: URL) throws {
    try QwenImageIO.saveImage(array: image.transposed(2, 0, 1), to: url)
  }

  func outputURL(_ name: String) -> URL {
    URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent(name)
  }

  // MARK: - AC-1

  /// Criterion 1: the default Krea 2 t2i render (kroma-v0.2 q8, 1024², 9
  /// steps, guidance 1.0, seed 44821, no LoRAs, no sampler field) is
  /// byte-identical to the pre-refactor engine's.
  func testDefaultT2IByteIdentical() throws {
    let oracle = try self.oracle("krea2-turbo-seed44821-oracle.json")
    let pipeline = try loadPipeline(oracle)

    let (image, trace) = try pipeline.generateWithRecipe(
      .init(
        prompt: oracle.request.prompt, guidance: oracle.request.guidance,
        width: oracle.request.width, height: oracle.request.height,
        steps: oracle.request.steps, seed: oracle.request.seed))

    // The defaults ARE the pre-change recipe, and the loop reports it itself.
    XCTAssertEqual(trace.sampler, .euler)
    XCTAssertEqual(trace.sigmaSchedule, .krea2)
    XCTAssertEqual(trace.shiftSource, "dynamic")
    XCTAssertEqual(trace.stepsRun, oracle.request.steps)
    XCTAssertEqual(trace.modelEvals, oracle.request.steps, "guidance 1.0 is one forward per step")
    XCTAssertEqual(trace.startIndex, 0)

    let out = outputURL("krea2-e3-parity-t2i.png")
    try writePNG(image, to: out)
    XCTAssertEqual(
      try imageSHA256(ofPNGAt: out), oracle.imageSha256,
      "AC-1: the default t2i render moved. Do not relax this — find the arithmetic "
        + "difference (dtype promotion, eval order, dt casting) and fix the driver.")
  }

  // MARK: - AC-2

  /// Criterion 2: the default img2img render at strength 0.3 is byte-identical.
  /// §3.3's trap lives on this path — the noise/source mix must run in float32
  /// (`Krea2Sampling.mixSourceLatent` takes the sigma as a float32 `MLXArray`,
  /// never `.item(Float.self)`), or every img2img render moves.
  func testDefaultImg2ImgByteIdentical() throws {
    let oracle = try self.oracle("krea2-turbo-seed44821-img2img-oracle.json")
    guard let sourcePath = oracle.request.sourceImage,
          let strength = oracle.request.strength else {
      return XCTFail("the img2img fixture must pin source_image and strength")
    }
    let sourceURL = URL(fileURLWithPath: (sourcePath as NSString).expandingTildeInPath)
    guard FileManager.default.fileExists(atPath: sourceURL.path) else {
      throw XCTSkip("the pinned img2img source image is not on this machine: \(sourceURL.path)")
    }
    XCTAssertEqual(
      try fileSHA256(sourceURL), oracle.request.sourceImageSha256,
      "the pinned source image changed — the oracle is meaningless against a different input")

    let pipeline = try loadPipeline(oracle)

    // The server's own load path: resize → normalize to [-1,1] NCHW → NHWC.
    let cg = try InpaintUtilities.loadCGImage(from: try Data(contentsOf: sourceURL))
    let pixNCHW = try QwenImageIO.resizedPixelArray(
      from: cg, width: oracle.request.width, height: oracle.request.height)
    let sourceNHWC = QwenImageIO.normalizeForEncoder(pixNCHW).transposed(0, 2, 3, 1)

    let (image, trace) = try pipeline.generateImg2ImgWithRecipe(
      .init(
        prompt: oracle.request.prompt, guidance: oracle.request.guidance,
        sourceImage: sourceNHWC, width: oracle.request.width, height: oracle.request.height,
        steps: oracle.request.steps, seed: oracle.request.seed, strength: strength))

    XCTAssertEqual(trace.sampler, .euler)
    XCTAssertEqual(trace.sigmaSchedule, .krea2)
    XCTAssertEqual(trace.denoise, 1.0 - strength, accuracy: 1e-6)
    XCTAssertEqual(trace.startIndex, 2, "strength 0.3 → denoise 0.7 → start at grid index 2")
    XCTAssertEqual(trace.stepsRun, trace.stepsEffective - trace.startIndex)

    let out = outputURL("krea2-e3-parity-img2img.png")
    try writePNG(image, to: out)
    XCTAssertEqual(
      try imageSHA256(ofPNGAt: out), oracle.imageSha256,
      "AC-2: the default img2img render moved — check the float32 mix first (§3.3).")
  }
}
