// LTX2NAGBlockTests.swift — NAG threaded through the transformer block's
// cross-attention (attn2).
//
// The reference wires NAG as a MODEL patch fed by its own negative
// conditioning (`LTX2_NAG {nag_cond_video, nag_cond_audio}`, v1.8 workflow node
// 342), independent of the CFG negative — which is inert in that recipe because
// CFG runs at 1.0. So the block must accept an optional negative context and,
// when present, run cross-attention twice and combine with ltx2ApplyNAG.

import XCTest
import MLX
import MLXRandom
@testable import ZImage

final class LTX2NAGBlockTests: XCTestCase {

  private func makeBlock(dim: Int, contextDim: Int, heads: Int) -> LTX2TransformerBlock {
    MLXRandom.seed(3)
    let block = LTX2TransformerBlock(
      dim: dim, contextDim: contextDim, heads: heads, dimHead: dim / heads)
    // Randomize so attention isn't degenerate; zeros would make every path equal.
    for (_, p) in block.parameters().flattened() {
      p[0..., .ellipsis] = MLXRandom.normal(p.shape) * 0.05
    }
    MLX.eval(block.parameters())
    return block
  }

  private func makeAVBlock() -> LTX2TransformerBlock {
    MLXRandom.seed(23)
    let block = LTX2TransformerBlock(
      dim: 64, contextDim: 64, heads: 4, dimHead: 16,
      hasPromptAdaLN: true, hasAudio: true,
      audioDim: 32, audioHeads: 2, audioDimHead: 16)
    for (_, p) in block.parameters().flattened() {
      p[0..., .ellipsis] = MLXRandom.normal(p.shape) * 0.05
    }
    MLX.eval(block.parameters())
    return block
  }

  /// Without a negative context the block must behave exactly as before —
  /// NAG is opt-in and existing renders stay byte-identical.
  func testNoNegativeContextLeavesBlockUnchanged() {
    let block = makeBlock(dim: 64, contextDim: 32, heads: 4)
    MLXRandom.seed(11)
    let x = MLXRandom.normal([1, 8, 64])
    let ctx = MLXRandom.normal([1, 5, 32])
    let ts = MLXRandom.normal([1, 1, 6 * 64])

    let baseline = block(x, context: ctx, timestep: ts)
    let withNilNeg = block(
      x, context: ctx, timestep: ts,
      negativeContext: nil, nag: .reference)
    MLX.eval(baseline, withNilNeg)

    XCTAssertLessThan(MLX.abs(baseline - withNilNeg).max().item(Float.self), 1e-5,
                      "passing a nil negative context must not perturb the block")
  }

  /// With NAG disabled (scale 1) a negative context must also be inert, so the
  /// config is a real off switch rather than a code path that always fires.
  func testDisabledNAGIgnoresNegativeContext() {
    let block = makeBlock(dim: 64, contextDim: 32, heads: 4)
    MLXRandom.seed(13)
    let x = MLXRandom.normal([1, 8, 64])
    let ctx = MLXRandom.normal([1, 5, 32])
    let neg = MLXRandom.normal([1, 5, 32])
    let ts = MLXRandom.normal([1, 1, 6 * 64])

    let baseline = block(x, context: ctx, timestep: ts)
    let withDisabled = block(
      x, context: ctx, timestep: ts,
      negativeContext: neg, nag: .disabled)
    MLX.eval(baseline, withDisabled)

    XCTAssertLessThan(MLX.abs(baseline - withDisabled).max().item(Float.self), 1e-5)
  }

  /// The point of the feature: an enabled NAG with a DIFFERENT negative context
  /// must actually change the output.
  func testEnabledNAGChangesOutput() {
    let block = makeBlock(dim: 64, contextDim: 32, heads: 4)
    MLXRandom.seed(17)
    let x = MLXRandom.normal([1, 8, 64])
    let ctx = MLXRandom.normal([1, 5, 32])
    let neg = MLXRandom.normal([1, 5, 32])
    let ts = MLXRandom.normal([1, 1, 6 * 64])

    let baseline = block(x, context: ctx, timestep: ts)
    let guided = block(
      x, context: ctx, timestep: ts,
      negativeContext: neg, nag: .reference)
    MLX.eval(baseline, guided)

    XCTAssertGreaterThan(MLX.abs(baseline - guided).max().item(Float.self), 1e-4,
                         "NAG with a distinct negative context must alter the output")
  }

  /// Identical positive and negative contexts must collapse to the baseline —
  /// the block-level mirror of the operator's identity property, and a check
  /// that the negative pass really is the same computation as the positive one.
  func testIdenticalContextsCollapseToBaseline() {
    let block = makeBlock(dim: 64, contextDim: 32, heads: 4)
    MLXRandom.seed(19)
    let x = MLXRandom.normal([1, 8, 64])
    let ctx = MLXRandom.normal([1, 5, 32])
    let ts = MLXRandom.normal([1, 1, 6 * 64])

    let baseline = block(x, context: ctx, timestep: ts)
    let guided = block(
      x, context: ctx, timestep: ts,
      negativeContext: ctx, nag: .reference)
    MLX.eval(baseline, guided)

    XCTAssertLessThan(MLX.abs(baseline - guided).max().item(Float.self), 1e-4,
                      "pos == neg must be a no-op through the block")
  }


  /// Joint audio/video inference used a separate block entry point and was
  /// silently bypassing the video-side NAG patch. The negative pass belongs
  /// only on video text cross-attention; audio text/cross-modal structure is
  /// otherwise unchanged.
  func testEnabledNAGChangesJointAVVideoOutput() {
    let block = makeAVBlock()
    MLXRandom.seed(29)
    let video = MLXRandom.normal([1, 8, 64])
    let audio = MLXRandom.normal([1, 5, 32])
    let context = MLXRandom.normal([1, 5, 64])
    let negative = MLXRandom.normal([1, 5, 64])
    let audioContext = MLXRandom.normal([1, 5, 32])
    let timestep = MLXRandom.normal([1, 1, 9 * 64])
    let audioTimestep = MLXRandom.normal([1, 1, 9 * 32])
    let promptTimestep = MLXRandom.normal([1, 1, 2 * 64])
    let audioPromptTimestep = MLXRandom.normal([1, 1, 2 * 32])

    let baseline = block.callDualStream(
      video: video, audio: audio, context: context, audioContext: audioContext,
      timestep: timestep, audioTimestep: audioTimestep,
      promptTimestep: promptTimestep, audioPromptTimestep: audioPromptTimestep)
    let guided = block.callDualStream(
      video: video, audio: audio, context: context, audioContext: audioContext,
      timestep: timestep, audioTimestep: audioTimestep,
      promptTimestep: promptTimestep, audioPromptTimestep: audioPromptTimestep,
      negativeContext: negative, nag: .reference)
    MLX.eval(baseline.video, guided.video)

    XCTAssertGreaterThan(
      MLX.abs(baseline.video - guided.video).max().item(Float.self), 1e-4,
      "joint A/V inference must apply NAG to the video text cross-attention")
  }

  func testDisabledNAGLeavesJointAVOutputUnchanged() {
    let block = makeAVBlock()
    MLXRandom.seed(31)
    let video = MLXRandom.normal([1, 8, 64])
    let audio = MLXRandom.normal([1, 5, 32])
    let context = MLXRandom.normal([1, 5, 64])
    let negative = MLXRandom.normal([1, 5, 64])
    let audioContext = MLXRandom.normal([1, 5, 32])
    let timestep = MLXRandom.normal([1, 1, 9 * 64])
    let audioTimestep = MLXRandom.normal([1, 1, 9 * 32])
    let promptTimestep = MLXRandom.normal([1, 1, 2 * 64])
    let audioPromptTimestep = MLXRandom.normal([1, 1, 2 * 32])

    let baseline = block.callDualStream(
      video: video, audio: audio, context: context, audioContext: audioContext,
      timestep: timestep, audioTimestep: audioTimestep,
      promptTimestep: promptTimestep, audioPromptTimestep: audioPromptTimestep)
    let disabled = block.callDualStream(
      video: video, audio: audio, context: context, audioContext: audioContext,
      timestep: timestep, audioTimestep: audioTimestep,
      promptTimestep: promptTimestep, audioPromptTimestep: audioPromptTimestep,
      negativeContext: negative, nag: .disabled)
    MLX.eval(baseline.video, disabled.video, baseline.audio, disabled.audio)

    XCTAssertLessThan(MLX.abs(baseline.video - disabled.video).max().item(Float.self), 1e-5)
    XCTAssertLessThan(MLX.abs(baseline.audio - disabled.audio).max().item(Float.self), 1e-5)
  }
}
