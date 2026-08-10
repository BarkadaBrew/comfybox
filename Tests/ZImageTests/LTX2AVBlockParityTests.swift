import Foundation
import MLX
import XCTest

@testable import ZImage

/// Task #21 wire 1: the joint dual-stream block vs the pinned reference
/// fixture (av_block0_goldens.safetensors — real v16b block-0 weights, all
/// 14 conditioning inputs, seed 31337, pe=None v1).
///
/// EXPECTED RED: the target API this encodes does not exist yet. The
/// current callDualStream lacks the four cross-AdaLN timestep inputs,
/// gated attention (to_gate_logits), and reference row semantics. Green
/// means the joint block math is reference-parity.
final class LTX2AVBlockParityTests: XCTestCase {

  static let fixtureURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures/ltx2-audio/av_block0_goldens.safetensors")

  /// Block-0-only extract (773MB; the 43GB monolith cannot stage locally —
  /// home volume headroom). Produced by the same prefix-strip the oracle
  /// exporter uses, so keys match the fixture's weight source exactly.
  static let weightsPath = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".comfybox/reference/v16b_block0.safetensors").path

  private func corr(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.reshaped([-1]).asType(.float32), y = b.reshaped([-1]).asType(.float32)
    let n = ((x - x.mean()) * (y - y.mean())).mean()
    let d = MLX.sqrt(((x - x.mean()) * (x - x.mean())).mean()) * MLX.sqrt(((y - y.mean()) * (y - y.mean())).mean())
    return (n / d).item(Float.self)
  }

  func testDualBlockMatchesReferenceFixture() throws {
    let weights = Self.weightsPath
    try XCTSkipUnless(FileManager.default.fileExists(atPath: weights),
      "v16b block-0 extract not present in ~/.comfybox/reference")
    let g = try MLX.loadArrays(url: Self.fixtureURL)

    // Target loader API: block 0 with the full A/V + cross + gated layout.
    let block = try LTX2TransformerBlock.loadAVBlock(
      blockExtractPath: weights)

    // Target forward contract — mirrors BasicAVTransformerBlock.forward.
    let (vxOut, axOut) = block.callDualStream(
      video: g["vx"]!.asType(.float32),
      audio: g["ax"]!.asType(.float32),
      context: g["v_context"]!.asType(.float32),
      audioContext: g["a_context"]!.asType(.float32),
      timestep: g["v_timestep"]!.asType(.float32),
      audioTimestep: g["a_timestep"]!.asType(.float32),
      pe: nil, audioPE: nil, crossPE: nil, audioCrossPE: nil,
      crossScaleShiftTimestep: g["v_cross_scale_shift_timestep"]!.asType(.float32),
      audioCrossScaleShiftTimestep: g["a_cross_scale_shift_timestep"]!.asType(.float32),
      crossGateTimestep: g["v_cross_gate_timestep"]!.asType(.float32),
      audioCrossGateTimestep: g["a_cross_gate_timestep"]!.asType(.float32),
      promptTimestep: g["v_prompt_timestep"]!.asType(.float32),
      audioPromptTimestep: g["a_prompt_timestep"]!.asType(.float32))

    XCTAssertEqual(vxOut.shape, g["vx_out"]!.shape)
    XCTAssertEqual(axOut.shape, g["ax_out"]!.shape)
    XCTAssertGreaterThan(corr(vxOut, g["vx_out"]!), 0.999, "video stream parity")
    XCTAssertGreaterThan(corr(axOut, g["ax_out"]!), 0.999, "audio stream parity")
  }
}
