// LTX2BeatScheduleBlockTests.swift — temporal beat scheduling (comfybox#310)
// threaded through the transformer block's text cross-attention (attn2), at
// the same tiny synthetic shape as LTX2NAGBlockTests.

import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import ZImage

final class LTX2BeatScheduleBlockTests: XCTestCase {

  private func makeBlock(dim: Int, contextDim: Int, heads: Int) -> LTX2TransformerBlock {
    MLXRandom.seed(29)
    let block = LTX2TransformerBlock(
      dim: dim, contextDim: contextDim, heads: heads, dimHead: dim / heads)
    // Randomize so attention isn't degenerate; zeros would make every path equal.
    for (_, p) in block.parameters().flattened() {
      p[0..., .ellipsis] = MLXRandom.normal(p.shape) * 0.05
    }
    MLX.eval(block.parameters())
    return block
  }

  /// (a) The bias must change the block's output when it DIFFERENTIATES
  /// between key columns (some text tokens penalized, others not) — and
  /// must be a no-op when it's merely a UNIFORM additive constant across
  /// every key, because softmax is shift-invariant to a per-query constant
  /// added identically to every key. This is the sharpest check that the
  /// bias lands INSIDE the softmax (on the biased columns specifically),
  /// not as some coarser after-the-fact adjustment.
  func testBeatBiasChangesOutputOnlyForDifferentiatedColumns() {
    let block = makeBlock(dim: 64, contextDim: 32, heads: 4)
    MLXRandom.seed(31)
    let x = MLXRandom.normal([1, 8, 64])
    let ctx = MLXRandom.normal([1, 5, 32])
    let ts = MLXRandom.normal([1, 1, 6 * 64])

    let baseline = block(x, context: ctx, timestep: ts)

    // Uniform bias: same constant added to every key column of every query
    // row. Softmax(logits + c) == Softmax(logits) for a query-wise constant
    // c, so this must be indistinguishable from no bias at all.
    let uniformBias = MLXArray([Float](repeating: -7.0, count: 8 * 5), [1, 1, 8, 5])
    let withUniform = block(x, context: ctx, timestep: ts, beatBias: uniformBias)
    MLX.eval(baseline, withUniform)
    XCTAssertLessThan(MLX.abs(baseline - withUniform).max().item(Float.self), 1e-4,
                      "a uniform (non-differentiating) bias must be a no-op through softmax")

    // Differentiated bias: strongly penalize only columns 0-1, leaving 2-4
    // untouched — this actually changes which keys the query attends to.
    var differentiated = [Float](repeating: 0, count: 8 * 5)
    for q in 0..<8 { differentiated[q * 5 + 0] = -50; differentiated[q * 5 + 1] = -50 }
    let columnBias = MLXArray(differentiated, [1, 1, 8, 5])
    let withColumns = block(x, context: ctx, timestep: ts, beatBias: columnBias)
    MLX.eval(withColumns)
    XCTAssertGreaterThan(MLX.abs(baseline - withColumns).max().item(Float.self), 1e-4,
                         "a bias that differentiates between key columns must change the output")
  }

  /// (b) The bias must compose with NAG rather than being canceled or
  /// ignored by it — bias alone and bias+NAG must give DIFFERENT outputs,
  /// since NAG only ever touches the block's own negative-context pass
  /// (never the biased positive pass) and its ltx2ApplyNAG extrapolation
  /// runs on top of whatever the positive cross-attn produced.
  func testBeatBiasComposesWithNAG() {
    let block = makeBlock(dim: 64, contextDim: 32, heads: 4)
    MLXRandom.seed(37)
    let x = MLXRandom.normal([1, 8, 64])
    let ctx = MLXRandom.normal([1, 5, 32])
    let neg = MLXRandom.normal([1, 5, 32])
    let ts = MLXRandom.normal([1, 1, 6 * 64])

    var differentiated = [Float](repeating: 0, count: 8 * 5)
    for q in 0..<8 { differentiated[q * 5 + 0] = -50; differentiated[q * 5 + 1] = -50 }
    let bias = MLXArray(differentiated, [1, 1, 8, 5])

    let biasOnly = block(x, context: ctx, timestep: ts, beatBias: bias)
    let biasPlusNAG = block(
      x, context: ctx, timestep: ts,
      negativeContext: neg, nag: .reference, beatBias: bias)
    MLX.eval(biasOnly, biasPlusNAG)

    XCTAssertGreaterThan(MLX.abs(biasOnly - biasPlusNAG).max().item(Float.self), 1e-4,
                         "bias + NAG must differ from bias alone — NAG must not be swallowed by the bias")
  }

  /// (c) An EXPLICIT all-zero bias tensor must be numerically identical to
  /// passing no bias at all (nil) — the additive-bias math has no other
  /// side effect (e.g. shape coercion, dtype promotion) that a zero tensor
  /// could silently introduce.
  func testZeroBiasTensorMatchesNilBiasIdentically() {
    let block = makeBlock(dim: 64, contextDim: 32, heads: 4)
    MLXRandom.seed(41)
    let x = MLXRandom.normal([1, 8, 64])
    let ctx = MLXRandom.normal([1, 5, 32])
    let ts = MLXRandom.normal([1, 1, 6 * 64])

    let withNil = block(x, context: ctx, timestep: ts, beatBias: nil)
    let zeroBias = MLXArray.zeros([1, 1, 8, 5])
    let withZero = block(x, context: ctx, timestep: ts, beatBias: zeroBias)
    MLX.eval(withNil, withZero)

    XCTAssertLessThan(MLX.abs(withNil - withZero).max().item(Float.self), 1e-6,
                      "an explicit zero-bias tensor must be a true no-op, identical to nil")
  }

  /// Adversarial review F12(a): the PRODUCTION dtype path. The pipeline runs
  /// the model at bf16 and (post-F1) converts the bias to bf16 before it
  /// reaches SDPA — this drives WEIGHTS, inputs AND bias through the block
  /// at .bfloat16 end to end (the fixture's randomized weights are cast to
  /// bf16 so nothing silently promotes back to fp32), asserting the pass
  /// completes (no SDPA dtype abort), stays bf16, and the differentiated
  /// bias still changes the output, mirroring
  /// testBeatBiasChangesOutputOnlyForDifferentiatedColumns at fp32.
  func testBF16InputsAndBF16BiasChangeOutputOnBiasedColumns() throws {
    let block = makeBlock(dim: 64, contextDim: 32, heads: 4)
    // Cast every parameter to bf16 — production weights load as bf16, and
    // fp32 fixture weights would promote the whole forward back to fp32.
    let bf16Params = ModuleParameters.unflattened(
      block.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) })
    try block.update(parameters: bf16Params, verify: [.shapeMismatch])
    MLX.eval(block.parameters())

    MLXRandom.seed(43)
    let x = MLXRandom.normal([1, 8, 64]).asType(.bfloat16)
    let ctx = MLXRandom.normal([1, 5, 32]).asType(.bfloat16)
    let ts = MLXRandom.normal([1, 1, 6 * 64]).asType(.bfloat16)

    let baseline = block(x, context: ctx, timestep: ts)
    XCTAssertEqual(baseline.dtype, .bfloat16, "bf16 weights + inputs must keep the block on the bf16 path")

    var differentiated = [Float](repeating: 0, count: 8 * 5)
    for q in 0..<8 { differentiated[q * 5 + 0] = -50; differentiated[q * 5 + 1] = -50 }
    let bias = MLXArray(differentiated, [1, 1, 8, 5]).asType(.bfloat16)

    let biased = block(x, context: ctx, timestep: ts, beatBias: bias)
    MLX.eval(baseline, biased)
    XCTAssertEqual(biased.dtype, .bfloat16, "the bf16 bias must not promote the output")

    XCTAssertGreaterThan(
      MLX.abs(baseline.asType(.float32) - biased.asType(.float32)).max().item(Float.self), 1e-4,
      "a differentiating bf16 bias through a bf16 block must change the output")
  }

  /// Adversarial review F12(c): the AV branch composes an existing
  /// contextMask WITH the beat bias — combine must sum both when non-nil,
  /// broadcasting the (B,1,1,S) key-axis mask over the (B,1,Q,S) bias rows.
  func testCombineSumsContextMaskAndBeatBiasWithBroadcast() {
    // Key-axis mask like the transformer's float contextMask: (1, 1, 1, 5),
    // last key column masked out at -1e9.
    var maskVals = [Float](repeating: 0, count: 5)
    maskVals[4] = -1e9
    let contextMask = MLXArray(maskVals, [1, 1, 1, 5])

    // Beat bias differentiated per query row: (1, 1, 8, 5).
    var biasVals = [Float](repeating: 0, count: 8 * 5)
    for q in 0..<8 { biasVals[q * 5 + 0] = Float(-(q + 1)) }
    let beatBias = MLXArray(biasVals, [1, 1, 8, 5])

    guard let combined = LTX2BeatScheduleBuilder.combine(contextMask, beatBias) else {
      XCTFail("combine of two non-nil masks must be non-nil")
      return
    }
    XCTAssertEqual(combined.shape, [1, 1, 8, 5], "mask must broadcast over the bias's query rows")

    let vals = combined.reshaped([8 * 5]).asArray(Float.self)
    for q in 0..<8 {
      XCTAssertEqual(vals[q * 5 + 0], Float(-(q + 1)), accuracy: 1e-3, "bias-only column")
      XCTAssertEqual(vals[q * 5 + 4], -1e9, accuracy: 1e3, "masked key column survives the sum in every row")
      XCTAssertEqual(vals[q * 5 + 2], 0, accuracy: 1e-6, "untouched column stays zero")
    }
  }
}
