import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

/// Task #21 wire 1 (top-level plumbing) vs pinned goldens
/// (av_toplevel_goldens.safetensors, exporter scripts/export_av_toplevel_goldens.py,
/// real v16b weights, CPU fp32, seed 7331, v_sigma=0.7 / a_sigma=0.35 —
/// DISTINCT so crossed gate wiring (a2v gate <- audio sigma, v2a gate <-
/// video sigma) cannot pass swapped).
///
/// Covers: audio adaln (9-coeff), audio prompt adaln, four av_ca adaln
/// singles with the 1/1000 av_ca factor, audio patchify+proj input path,
/// real-time cross-modal RoPE, audio self RoPE, audio output processing.
final class LTX2AVToplevelParityTests: XCTestCase {

  static let fixtureURL = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures/ltx2-audio/av_toplevel_goldens.safetensors")

  static let weightsPath = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".comfybox/reference/v16b_av_toplevel.safetensors").path

  private func goldens() throws -> [String: MLXArray] {
    try MLX.loadArrays(url: Self.fixtureURL)
  }

  /// Full-size hasAudio transformer, lazily allocated (48 blocks stay
  /// unevaluated zeros); only the small top-level audio modules get real
  /// weights from the extract and only they are evaluated here.
  private func plumbing() throws -> LTX2Transformer {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath),
      "v16b top-level extract not on this machine")
    let t = LTX2Transformer(
      numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
      numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
      normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
      positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
      useMiddleIndicesGrid: true, ropeMode: .split, doublePrecisionRoPE: true,
      hasAudio: true, audioInnerDim: 2048, audioInChannels: 128)
    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: Self.weightsPath))
    var weights: [String: MLXArray] = [:]
    for (key, value) in raw {
      var k = key
      k = k.replacingOccurrences(of: ".linear_1.", with: ".linear1.")
      k = k.replacingOccurrences(of: ".linear_2.", with: ".linear2.")
      weights[k] = value.asType(.float32)
    }
    let params = ModuleParameters.unflattened(weights.map { ($0.key, $0.value) })
    try t.update(parameters: params, verify: [.shapeMismatch])
    return t
  }

  private func corr(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.reshaped([-1]).asType(.float32), y = b.reshaped([-1]).asType(.float32)
    let n = ((x - x.mean()) * (y - y.mean())).mean()
    let d = MLX.sqrt(((x - x.mean()) * (x - x.mean())).mean()) * MLX.sqrt(((y - y.mean()) * (y - y.mean())).mean())
    return (n / d).item(Float.self)
  }

  func testConditioningMatchesGoldens() throws {
    let g = try goldens()
    let p = try plumbing()
    let c = p.prepareAVConditioning(videoSigma: 0.7, audioSigma: 0.35, batchSize: 1)

    XCTAssertEqual(c.audioTimestep.shape, [1, 1, 9 * 2048])
    XCTAssertGreaterThan(corr(c.audioTimestep, g["a_timestep_emb"]!), 0.999, "audio 9-coeff adaln")
    XCTAssertGreaterThan(corr(c.audioEmbeddedTimestep, g["a_embedded_timestep"]!), 0.999, "audio embedded ts")
    XCTAssertGreaterThan(corr(c.audioPromptTimestep, g["a_prompt_timestep"]!), 0.999, "audio prompt adaln")
    XCTAssertGreaterThan(corr(c.crossScaleShiftTimestep, g["av_ca_video_ss_timestep"]!), 0.999, "av_ca video ss")
    XCTAssertGreaterThan(corr(c.audioCrossScaleShiftTimestep, g["av_ca_audio_ss_timestep"]!), 0.999, "av_ca audio ss")
    XCTAssertGreaterThan(corr(c.crossGateTimestep, g["av_ca_a2v_gate_timestep"]!), 0.999,
      "a2v gate must be driven by the AUDIO sigma (crossed)")
    XCTAssertGreaterThan(corr(c.audioCrossGateTimestep, g["av_ca_v2a_gate_timestep"]!), 0.999,
      "v2a gate must be driven by the VIDEO sigma (crossed)")
  }

  func testAudioTokenProjectionMatchesGoldens() throws {
    let g = try goldens()
    let p = try plumbing()
    let (tokens, coords) = p.projectAudioTokens(g["ax_latents"]!)
    XCTAssertEqual(tokens.shape, [1, 12, 2048])
    XCTAssertGreaterThan(corr(tokens, g["a_tokens_projected"]!), 0.999, "patchify + audio_patchify_proj")
    let coordDiff = MLX.abs(coords - g["a_latent_coords"]!).max().item(Float.self)
    XCTAssertLessThan(coordDiff, 1e-5, "real-time start/end coords")
  }

  func testAudioOutputProcessingMatchesGoldens() throws {
    let g = try goldens()
    let p = try plumbing()
    let out = p.processAudioOutput(g["ax_hidden"]!, embeddedTimestep: g["a_embedded_timestep"]!)
    XCTAssertEqual(out.shape, [1, 8, 12, 16])
    XCTAssertGreaterThan(corr(out, g["ax_out_latents"]!), 0.999,
      "norm -> table+embedded modulation -> proj_out -> unpatchify")
  }

  /// Swift RoPE returns (B, H, T, half); the fixture stores the reference's
  /// pre-rotation layout (B, T, H*half). Realign before comparing.
  private func flatPE(_ x: MLXArray) -> MLXArray {
    let t = x.transposed(0, 2, 1, 3)  // (B, T, H, half)
    return t.reshaped([t.dim(0), t.dim(1), -1])
  }

  func testCrossModalRoPEMatchesGoldens() throws {
    let g = try goldens()
    // Video side: time axis converted to seconds (already in fixture input).
    var vSecs = g["v_pixel_coords"]!.asType(.float32)
    let timeRow = vSecs[0..., 0..<1] * (1.0 / 25.0)
    vSecs = MLX.concatenated([timeRow, vSecs[0..., 1...]], axis: 1)

    let crossV = ltx2PrecomputeFreqsCIS(
      indicesGrid: vSecs[0..., 0..<1], dim: 2048, maxPos: [20],
      useMiddleIndicesGrid: true, numAttentionHeads: 32, ropeMode: .split,
      doublePrecision: true)
    XCTAssertGreaterThan(corr(flatPE(crossV.cos), g["cross_v_pe_cos"]!), 0.9999, "cross video PE cos")
    XCTAssertGreaterThan(corr(flatPE(crossV.sin), g["cross_v_pe_sin"]!), 0.9999, "cross video PE sin")

    let aCoords = g["a_latent_coords"]!.asType(.float32)
    let crossA = ltx2PrecomputeFreqsCIS(
      indicesGrid: aCoords[0..., 0..<1], dim: 2048, maxPos: [20],
      useMiddleIndicesGrid: true, numAttentionHeads: 32, ropeMode: .split,
      doublePrecision: true)
    XCTAssertGreaterThan(corr(flatPE(crossA.cos), g["cross_a_pe_cos"]!), 0.9999, "cross audio PE cos")
    XCTAssertGreaterThan(corr(flatPE(crossA.sin), g["cross_a_pe_sin"]!), 0.9999, "cross audio PE sin")
  }

  func testAudioSelfRoPEMatchesGoldens() throws {
    let g = try goldens()
    let aCoords = g["a_latent_coords"]!.asType(.float32)
    for (flag, key) in [(true, "a_pe_mid"), (false, "a_pe_nomid")] {
      let pe = ltx2PrecomputeFreqsCIS(
        indicesGrid: aCoords, dim: 2048, maxPos: [20],
        useMiddleIndicesGrid: flag, numAttentionHeads: 32, ropeMode: .split,
        doublePrecision: true)
      XCTAssertGreaterThan(corr(flatPE(pe.cos), g["\(key)_cos"]!), 0.9999, "\(key) cos")
      XCTAssertGreaterThan(corr(flatPE(pe.sin), g["\(key)_sin"]!), 0.9999, "\(key) sin")
    }
  }
}
