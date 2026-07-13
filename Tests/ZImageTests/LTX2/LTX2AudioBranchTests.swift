import XCTest
import Logging
import MLX
import MLXNN
@testable import ZImage

/// Unit tests for the JoyAI-Echo audio DiT branch (Phase 2, step 4).
///
/// Covers the 2729 audio-branch tensors of the monolith `model.diffusion_model.`
/// prefix — 41 top-level (audio_*, av_ca_*) + 56 per-block × 48 blocks — loading
/// onto real parameters of the `hasAudio: true` transformer (anti-noise guard).
/// The audio_embeddings_connector (text-encoder concern) is excluded.
final class LTX2AudioBranchTests: XCTestCase {

  override func setUpWithError() throws {
    do {
      try MLX.withError {
        let probe = MLXArray([1 as Float, 2], [2]) + MLXArray([3 as Float, 4], [2])
        MLX.eval(probe)
      }
    } catch {
      throw XCTSkip("MLX evaluation is unavailable in this test runner: \(error)")
    }
  }

  /// Top-level audio tensors (name relative to model.diffusion_model., shape).
  private static let topFixture: [(String, [Int])] = [
    ("audio_adaln_single.emb.timestep_embedder.linear_1.bias", [2048]),
    ("audio_adaln_single.emb.timestep_embedder.linear_1.weight", [2048,256]),
    ("audio_adaln_single.emb.timestep_embedder.linear_2.bias", [2048]),
    ("audio_adaln_single.emb.timestep_embedder.linear_2.weight", [2048,2048]),
    ("audio_adaln_single.linear.bias", [18432]),
    ("audio_adaln_single.linear.weight", [18432,2048]),
    ("audio_patchify_proj.bias", [2048]),
    ("audio_patchify_proj.weight", [2048,128]),
    ("audio_proj_out.bias", [128]),
    ("audio_proj_out.weight", [128,2048]),
    ("audio_prompt_adaln_single.emb.timestep_embedder.linear_1.bias", [2048]),
    ("audio_prompt_adaln_single.emb.timestep_embedder.linear_1.weight", [2048,256]),
    ("audio_prompt_adaln_single.emb.timestep_embedder.linear_2.bias", [2048]),
    ("audio_prompt_adaln_single.emb.timestep_embedder.linear_2.weight", [2048,2048]),
    ("audio_prompt_adaln_single.linear.bias", [4096]),
    ("audio_prompt_adaln_single.linear.weight", [4096,2048]),
    ("audio_scale_shift_table", [2,2048]),
    ("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.bias", [4096]),
    ("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight", [4096,256]),
    ("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_2.bias", [4096]),
    ("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_2.weight", [4096,4096]),
    ("av_ca_a2v_gate_adaln_single.linear.bias", [4096]),
    ("av_ca_a2v_gate_adaln_single.linear.weight", [4096,4096]),
    ("av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_1.bias", [2048]),
    ("av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_1.weight", [2048,256]),
    ("av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_2.bias", [2048]),
    ("av_ca_audio_scale_shift_adaln_single.emb.timestep_embedder.linear_2.weight", [2048,2048]),
    ("av_ca_audio_scale_shift_adaln_single.linear.bias", [8192]),
    ("av_ca_audio_scale_shift_adaln_single.linear.weight", [8192,2048]),
    ("av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_1.bias", [2048]),
    ("av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_1.weight", [2048,256]),
    ("av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_2.bias", [2048]),
    ("av_ca_v2a_gate_adaln_single.emb.timestep_embedder.linear_2.weight", [2048,2048]),
    ("av_ca_v2a_gate_adaln_single.linear.bias", [2048]),
    ("av_ca_v2a_gate_adaln_single.linear.weight", [2048,2048]),
    ("av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_1.bias", [4096]),
    ("av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_1.weight", [4096,256]),
    ("av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_2.bias", [4096]),
    ("av_ca_video_scale_shift_adaln_single.emb.timestep_embedder.linear_2.weight", [4096,4096]),
    ("av_ca_video_scale_shift_adaln_single.linear.bias", [16384]),
    ("av_ca_video_scale_shift_adaln_single.linear.weight", [16384,4096]),
  ]
  /// Per-block audio tensors (name relative to the block, shape) — replicated ×48.
  private static let blockFixture: [(String, [Int])] = [
    ("audio_attn1.k_norm.weight", [2048]),
    ("audio_attn1.q_norm.weight", [2048]),
    ("audio_attn1.to_gate_logits.bias", [32]),
    ("audio_attn1.to_gate_logits.weight", [32,2048]),
    ("audio_attn1.to_k.bias", [2048]),
    ("audio_attn1.to_k.weight", [2048,2048]),
    ("audio_attn1.to_out.0.bias", [2048]),
    ("audio_attn1.to_out.0.weight", [2048,2048]),
    ("audio_attn1.to_q.bias", [2048]),
    ("audio_attn1.to_q.weight", [2048,2048]),
    ("audio_attn1.to_v.bias", [2048]),
    ("audio_attn1.to_v.weight", [2048,2048]),
    ("audio_attn2.k_norm.weight", [2048]),
    ("audio_attn2.q_norm.weight", [2048]),
    ("audio_attn2.to_gate_logits.bias", [32]),
    ("audio_attn2.to_gate_logits.weight", [32,2048]),
    ("audio_attn2.to_k.bias", [2048]),
    ("audio_attn2.to_k.weight", [2048,2048]),
    ("audio_attn2.to_out.0.bias", [2048]),
    ("audio_attn2.to_out.0.weight", [2048,2048]),
    ("audio_attn2.to_q.bias", [2048]),
    ("audio_attn2.to_q.weight", [2048,2048]),
    ("audio_attn2.to_v.bias", [2048]),
    ("audio_attn2.to_v.weight", [2048,2048]),
    ("audio_ff.net.0.proj.bias", [8192]),
    ("audio_ff.net.0.proj.weight", [8192,2048]),
    ("audio_ff.net.2.bias", [2048]),
    ("audio_ff.net.2.weight", [2048,8192]),
    ("audio_prompt_scale_shift_table", [2,2048]),
    ("audio_scale_shift_table", [9,2048]),
    ("audio_to_video_attn.k_norm.weight", [2048]),
    ("audio_to_video_attn.q_norm.weight", [2048]),
    ("audio_to_video_attn.to_gate_logits.bias", [32]),
    ("audio_to_video_attn.to_gate_logits.weight", [32,4096]),
    ("audio_to_video_attn.to_k.bias", [2048]),
    ("audio_to_video_attn.to_k.weight", [2048,2048]),
    ("audio_to_video_attn.to_out.0.bias", [4096]),
    ("audio_to_video_attn.to_out.0.weight", [4096,2048]),
    ("audio_to_video_attn.to_q.bias", [2048]),
    ("audio_to_video_attn.to_q.weight", [2048,4096]),
    ("audio_to_video_attn.to_v.bias", [2048]),
    ("audio_to_video_attn.to_v.weight", [2048,2048]),
    ("scale_shift_table_a2v_ca_audio", [5,2048]),
    ("scale_shift_table_a2v_ca_video", [5,4096]),
    ("video_to_audio_attn.k_norm.weight", [2048]),
    ("video_to_audio_attn.q_norm.weight", [2048]),
    ("video_to_audio_attn.to_gate_logits.bias", [32]),
    ("video_to_audio_attn.to_gate_logits.weight", [32,2048]),
    ("video_to_audio_attn.to_k.bias", [2048]),
    ("video_to_audio_attn.to_k.weight", [2048,4096]),
    ("video_to_audio_attn.to_out.0.bias", [2048]),
    ("video_to_audio_attn.to_out.0.weight", [2048,2048]),
    ("video_to_audio_attn.to_q.bias", [2048]),
    ("video_to_audio_attn.to_q.weight", [2048,2048]),
    ("video_to_audio_attn.to_v.bias", [2048]),
    ("video_to_audio_attn.to_v.weight", [2048,4096]),
  ]

  /// Echo dual-stream transformer, lazily allocated (no eval → no real memory).
  private func makeAudioTransformer() -> LTX2Transformer {
    LTX2Transformer(
      numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
      numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
      normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
      positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
      useMiddleIndicesGrid: true, ropeMode: .split, doublePrecisionRoPE: true,
      hasAudio: true, audioInnerDim: 2048, audioInChannels: 128)
  }

  /// Synthetic monolith audio-branch subset (all 2729 tensors, lazy zeros).
  private func syntheticAudioCheckpoint() -> [String: MLXArray] {
    var w: [String: MLXArray] = [:]
    let p = "model.diffusion_model."
    for (k, shape) in Self.topFixture { w[p + k] = MLXArray.zeros(shape) }
    for b in 0..<48 {
      for (k, shape) in Self.blockFixture {
        w["\(p)transformer_blocks.\(b).\(k)"] = MLXArray.zeros(shape)
      }
    }
    return w
  }

  private func isAudioKey(_ k: String) -> Bool {
    k.contains("audio_") || k.contains("av_ca_")
      || k.contains("scale_shift_table_a2v")
      || k.contains("audio_to_video") || k.contains("video_to_audio")
  }

  func testFixtureCounts() {
    XCTAssertEqual(Self.topFixture.count, 41)
    XCTAssertEqual(Self.blockFixture.count, 56)
    XCTAssertEqual(syntheticAudioCheckpoint().count, 2729)
  }

  /// Video-only transformer holds NO audio params (audio strictly gated off).
  func testVideoOnlyTransformerHasNoAudioParams() {
    let t = LTX2Transformer(
      numHeads: 32, headDim: 128, numLayers: 48, crossAttentionDim: 4096,
      captionChannels: 3840, hasPromptAdaLN: true, ropeMode: .split)
    let keys = t.parameters().flattened().map { $0.0 }
    XCTAssertFalse(keys.contains { isAudioKey($0) }, "audio params leaked into video-only transformer")
  }

  /// Anti-noise guard: every one of the 2729 audio-branch tensors maps (after
  /// audio-inclusive sanitize) to a real parameter of the hasAudio transformer.
  func testAllAudioBranchKeysMapToModuleParameters() {
    let t = makeAudioTransformer()
    let moduleKeys = Set(t.parameters().flattened().map { $0.0 })
    let sanitized = LTX2Transformer.sanitizeWeightsWithAudio(syntheticAudioCheckpoint())
    let audioSanitized = sanitized.filter { isAudioKey($0.key) }
    XCTAssertEqual(audioSanitized.count, 2729, "sanitize changed audio-key count")
    var matched = 0
    for k in audioSanitized.keys {
      if moduleKeys.contains(k) { matched += 1 }
      else { XCTFail("audio key has no module param: \(k)") }
    }
    XCTAssertEqual(matched, 2729, "not all audio-branch tensors matched")
  }

  /// The FF / to_out name remaps land on the module's proj_in/proj_out/to_out.
  func testAudioAndVideoNameRemaps() {
    let s = LTX2Transformer.sanitizeWeightsWithAudio(syntheticAudioCheckpoint())
    XCTAssertNotNil(s["transformer_blocks.0.audio_ff.proj_in.weight"])
    XCTAssertNotNil(s["transformer_blocks.0.audio_ff.proj_out.weight"])
    XCTAssertNil(s["transformer_blocks.0.audio_ff.net.0.proj.weight"])
    XCTAssertNotNil(s["transformer_blocks.0.audio_attn1.to_out.weight"])
    XCTAssertNil(s["transformer_blocks.0.audio_attn1.to_out.0.weight"])
    XCTAssertNotNil(s["audio_patchify_proj.weight"])
    XCTAssertNotNil(s["av_ca_video_scale_shift_adaln_single.linear.weight"])
  }

  /// Weights apply with shape verification on (validates every audio channel
  /// count) — lazy, no forward eval.
  func testAudioWeightsApplyWithShapeVerification() throws {
    let t = makeAudioTransformer()
    let sanitized = LTX2Transformer.sanitizeWeightsWithAudio(syntheticAudioCheckpoint())
    let params = ModuleParameters.unflattened(sanitized.map { ($0.key, $0.value) })
    XCTAssertNoThrow(try t.update(parameters: params, verify: [.shapeMismatch]))
  }

  /// Cross-modal / dual-stream block forward runs and preserves both stream
  /// shapes (tiny dims to keep it cheap; the real-dim forward is deferred to the
  /// generateAV pipeline + validation render).
  func testDualStreamBlockForwardShapes() {
    let block = LTX2TransformerBlock(
      dim: 64, contextDim: 64, heads: 2, dimHead: 32,
      hasPromptAdaLN: true, hasAudio: true, audioDim: 32, audioHeads: 2, audioDimHead: 16)
    let b = 1, tv = 8, ta = 6, s = 4
    let video = MLXRandom.normal([b, tv, 64])
    let audio = MLXRandom.normal([b, ta, 32])
    let ctx = MLXRandom.normal([b, s, 64])
    let actx = MLXRandom.normal([b, s, 32])
    let ts = MLXRandom.normal([b, 1, 9 * 64])
    let ats = MLXRandom.normal([b, 1, 9 * 32])
    let (vo, ao) = block.callDualStream(
      video: video, audio: audio, context: ctx, audioContext: actx,
      timestep: ts, audioTimestep: ats)
    MLX.eval(vo, ao)
    XCTAssertEqual(vo.shape, [b, tv, 64])
    XCTAssertEqual(ao.shape, [b, ta, 32])
    let m = MLX.max(MLX.abs(vo)).item(Float.self)
    XCTAssertTrue(m.isFinite, "dual-stream produced non-finite video")
  }
}
