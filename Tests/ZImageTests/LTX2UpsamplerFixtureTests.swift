// LTX2UpsamplerFixtureTests.swift — end-to-end numeric check of the Swift
// latent upsampler against a ComfyUI reference input/output pair.
//
// Fixtures (see docs/HANDOFF-ltx-quality-2026-08-01.md §3):
//   /private/tmp/up_in.npy  — (1,128,3,8,12) latent, torch.manual_seed(0)
//   /private/tmp/up_out.npy — (1,128,3,16,24) reference LatentUpsampler output
//
// The reference output has UNIFORM per-row energy (0.427–0.477 across all 16
// rows). The refine pass in production truncates the bottom ~41% of every
// frame regardless of which weights are bound, so this test decides where the
// fault lives: if our per-row energy collapses after ~row 9 the upsampler
// module itself is wrong; if it stays uniform the bug is downstream in the
// refine plumbing (LTX2Pipeline stats/re-noise).
//
// Skips cleanly when fixtures or weights are absent. Override paths with
// LTX2_UPSAMPLER_FIXTURE_IN / _OUT / LTX2_UPSAMPLER_PATH.

import XCTest
import MLX
import MLXNN
@testable import ZImage

final class LTX2UpsamplerFixtureTests: XCTestCase {

  private func firstExisting(_ paths: [String]) -> String? {
    paths.first { FileManager.default.fileExists(atPath: $0) }
  }

  /// Mean squared value per output H row, reduced over (B, C, F, W).
  private func rowEnergy(_ x: MLXArray) -> [Float] {
    // x: (B, C, F, H, W)
    let e = MLX.mean(x.asType(.float32) * x.asType(.float32), axes: [0, 1, 2, 4])
    return e.asArray(Float.self)
  }

  func testFixturePairEndToEnd() throws {
    let env = ProcessInfo.processInfo.environment
    let home = NSHomeDirectory()

    guard
      let inPath = firstExisting([
        env["LTX2_UPSAMPLER_FIXTURE_IN"] ?? "", "/private/tmp/up_in.npy",
      ]),
      let outPath = firstExisting([
        env["LTX2_UPSAMPLER_FIXTURE_OUT"] ?? "", "/private/tmp/up_out.npy",
      ]),
      let ckptPath = firstExisting([
        env["LTX2_UPSAMPLER_PATH"] ?? "",
        "\(home)/LocalModels/ltx2-upsampler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "\(home)/LocalModels/ltx2-upsampler/ltx-2.3-spatial-upscaler-x2-1.1-official.safetensors",
      ])
    else {
      throw XCTSkip("upsampler fixtures or checkpoint not present")
    }

    // Load + remap weights exactly as LTX2VideoGenerator does.
    let up = LTX2LatentUpsampler()
    let w = try MLX.loadArrays(url: URL(fileURLWithPath: ckptPath))
    let remapped: [(String, MLXArray)] = w.map { (rawKey, v) in
      let key = rawKey.hasPrefix("upsampler.0.")
        ? "upsampler.conv." + rawKey.dropFirst("upsampler.0.".count)
        : rawKey
      if key.hasSuffix(".weight") {
        if v.ndim == 5 { return (key, v.transposed(0, 2, 3, 4, 1)) }
        if v.ndim == 4 { return (key, v.transposed(0, 2, 3, 1)) }
      }
      return (key, v)
    }
    let expected = Set(up.parameters().flattened().map { $0.0 })
    let bound = remapped.filter { expected.contains($0.0) }.count
    XCTAssertEqual(bound, expected.count, "upsampler weights failed to bind")
    try up.update(parameters: ModuleParameters.unflattened(remapped), verify: [.shapeMismatch])
    MLX.eval(up.parameters())

    let x = try MLX.loadArray(url: URL(fileURLWithPath: inPath)).asType(.float32)
    let ref = try MLX.loadArray(url: URL(fileURLWithPath: outPath)).asType(.float32)
    XCTAssertEqual(x.shape, [1, 128, 3, 8, 12], "unexpected fixture input shape")
    XCTAssertEqual(ref.shape, [1, 128, 3, 16, 24], "unexpected fixture output shape")

    let ours = up(x).asType(.float32)
    MLX.eval(ours)
    XCTAssertEqual(ours.shape, ref.shape)

    let oursRows = rowEnergy(ours)
    let refRows = rowEnergy(ref)
    print("ROWS ours: \(oursRows.map { String(format: "%.4f", $0) }.joined(separator: " "))")
    print("ROWS ref : \(refRows.map { String(format: "%.4f", $0) }.joined(separator: " "))")

    let diff = MLX.abs(ours - ref)
    let maxAbs = diff.max().item(Float.self)
    let meanAbs = diff.mean().item(Float.self)
    let refScale = MLX.abs(ref).max().item(Float.self)
    print(String(
      format: "DIFF maxAbs %.5f meanAbs %.5f refMaxAbs %.5f relErr %.5f",
      maxAbs, meanAbs, refScale, maxAbs / max(refScale, 1e-6)))

    // The decision this test exists to make: per-row energy must not collapse.
    let minRow = oursRows.min() ?? 0
    let maxRow = oursRows.max() ?? 0
    XCTAssertGreaterThan(
      minRow, 0.5 * maxRow,
      "per-row energy collapses (\(minRow) vs \(maxRow)) — fault is inside the upsampler module")

    // Numeric parity with the reference output (bf16 weights → loose-ish bound).
    XCTAssertLessThan(maxAbs / max(refScale, 1e-6), 2e-2, "upsampler diverges from reference output")
  }
}
