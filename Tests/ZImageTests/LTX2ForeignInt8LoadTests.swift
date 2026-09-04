// LTX2ForeignInt8LoadTests.swift — comfybox#256.
//
// PinkCherry v1.7's int8 export uses a DIFFERENT quantization layout than
// `LTX2Quantizer`'s own output. Per docs/HANDOFF-ltx-quality-2026-08-02.md
// §5: "His int8 is `I8` + `F32 .weight_scale` (torch-style, ... symmetric,
// per-output-channel). Ours is MLX affine group-wise with `.scales`/
// `.biases` at group 64 — `LTX2Quantizer.applyQuantizedLayout` would not
// recognise his."
//
// `applyQuantizedLayout` (Sources/ZImage/LTX2/LTX2Quantizer.swift:191)
// only ever looks for a `.scales` sibling key. A `.weight` tensor that
// carries `.weight_scale` instead is invisible to it: the Linear layer is
// never converted to QuantizedLinear, and the raw int8 byte values get
// loaded directly into a float weight parameter by
// `transformer.update(parameters:)` in `LTX2VideoGenerator.load()` — the
// small integers (-127...127) get treated as if they were already
// dequantized float weights, which is why every render comes out as
// uniform noise (issue #256).
//
// These tests pin that root cause on synthetic tensors (no model weights
// loaded) and drive the fix: recognising the foreign layout and
// dequantizing it correctly at load time, or failing loudly by name when
// an int8 tensor has no recognised scale sidecar at all.

import MLX
import MLXNN
import XCTest

@testable import ZImage

final class LTX2ForeignInt8LoadTests: XCTestCase {

  // MARK: - Fixtures

  /// A tiny 2x4 "dense" reference weight, exactly representable by
  /// symmetric per-row int8 quantization so the round-trip is exact (no
  /// rounding slack to hide a bug behind).
  ///   row 0: scale 0.01, int8 [-127, -64, -10, 64]  -> [-1.27, -0.64, -0.10, 0.64]
  ///   row 1: scale 0.02, int8 [10, 20, 30, 40]      -> [0.2, 0.4, 0.6, 0.8]
  /// (deliberately no zero entries, so a naive "int8 == float" comparison
  /// can't accidentally match by coincidence.)
  private static let denseReference: [Float] = [
    -1.27, -0.64, -0.10, 0.64,
    0.2, 0.4, 0.6, 0.8,
  ]
  private static let int8Rows: [Int8] = [
    -127, -64, -10, 64,
    10, 20, 30, 40,
  ]
  private static let perChannelScale: [Float] = [0.01, 0.02]

  private func foreignInt8Weight() -> MLXArray {
    MLXArray(Self.int8Rows, [2, 4])
  }

  /// Torch-style per-output-channel scale, shape [out, 1] — the common
  /// export shape (broadcasts over the input dimension).
  private func foreignScale() -> MLXArray {
    MLXArray(Self.perChannelScale, [2, 1])
  }

  private func floats(_ a: MLXArray) -> [Float] {
    let flat = a.asType(.float32).flattened()
    eval(flat)
    return flat.asArray(Float.self)
  }

  // MARK: - Root cause: the loader's own recognition code ignores this layout

  /// `applyQuantizedLayout` only recognises its own MLX affine output
  /// (`.scales` sibling). A `.weight_scale` sidecar — PinkCherry's layout —
  /// converts zero layers, so the Linear stays plain and the raw int8
  /// tensor is what ends up loaded as its "weight".
  func testApplyQuantizedLayoutDoesNotRecognizeForeignWeightScaleSidecar() {
    let toy = ToyLinearHolder(weight: MLXArray.zeros([2, 4]))
    let foreignDict: [String: MLXArray] = [
      "to_q.weight": foreignInt8Weight(),
      "to_q.weight_scale": foreignScale(),
    ]

    let converted = LTX2Quantizer.applyQuantizedLayout(to: toy, sanitizedWeights: foreignDict)

    XCTAssertEqual(
      converted, 0,
      "a foreign .weight_scale sidecar must not be mistaken for LTX2Quantizer's own .scales layout")
  }

  /// Sanity check on the positive case: LTX2Quantizer's own layout (packed
  /// uint32 `.weight` + `.scales`) IS recognised and converts the layer.
  func testApplyQuantizedLayoutRecognizesOwnScalesLayout() {
    // groupSize must be one of MLX's supported sizes (32/64/128), so this
    // uses a wider toy weight than the [2,4] foreign-layout fixture above.
    let toy = ToyLinearHolder(weight: MLXArray.zeros([2, 32]))
    let dense = MLXRandom.normal([2, 32])
    let (wq, scales, biases) = MLX.quantized(dense, groupSize: 32, bits: 8, mode: .affine)
    var ownDict: [String: MLXArray] = [
      "to_q.weight": wq,
      "to_q.scales": scales,
    ]
    if let biases { ownDict["to_q.biases"] = biases }

    let converted = LTX2Quantizer.applyQuantizedLayout(to: toy, sanitizedWeights: ownDict)

    XCTAssertEqual(converted, 1, "LTX2Quantizer's own .scales layout must still be recognised")
  }

  // MARK: - Pinning the noise bug

  /// This is the bug: treating the raw int8 bytes as if they were already
  /// the dequantized float weight (what happens today once the layer isn't
  /// converted to QuantizedLinear and the int8 array is assigned straight
  /// into a float parameter) does NOT reconstruct the reference weight —
  /// it's wildly wrong in scale (off by ~100x, the reciprocal of the
  /// quantization scale), which is exactly "confetti noise" territory.
  func testNaiveInt8AsFloatDoesNotReconstructReference() {
    let naive = floats(foreignInt8Weight().asType(.float32))
    let reference = Self.denseReference

    // Values like -127 vs -1.27 — not remotely close.
    for (n, r) in zip(naive, reference) {
      XCTAssertGreaterThan(abs(n - r), 1.0, "raw int8 values must NOT already equal the dequantized reference")
    }
  }

  // MARK: - The fix: recognise + correctly dequantize the foreign layout

  func testDequantizeForeignInt8WeightsReconstructsReference() throws {
    let weights: [String: MLXArray] = [
      "transformer_blocks.0.attn1.to_q.weight": foreignInt8Weight(),
      "transformer_blocks.0.attn1.to_q.weight_scale": foreignScale(),
    ]

    let out = try LTX2Quantizer.dequantizeForeignInt8Weights(weights)

    guard let result = out["transformer_blocks.0.attn1.to_q.weight"] else {
      return XCTFail("dequantized weight missing from output")
    }
    XCTAssertEqual(result.dtype, .bfloat16, "foreign int8 weights are dequantized to plain bf16, not repacked into MLX affine form")
    XCTAssertNil(
      out["transformer_blocks.0.attn1.to_q.weight_scale"],
      "the consumed sidecar should not leak into the sanitized dict the loader hands to update(parameters:)")

    let reconstructed = floats(result)
    for (got, want) in zip(reconstructed, Self.denseReference) {
      XCTAssertEqual(got, want, accuracy: 0.01, "dequantized weight must match the reference within bf16 tolerance")
    }
  }

  /// A checkpoint with mixed foreign-quantized and plain bf16 tensors —
  /// only the int8 ones should be touched.
  func testDequantizeForeignInt8WeightsPassesThroughNonInt8Tensors() throws {
    let passthrough = MLXArray.zeros([3], dtype: .bfloat16)
    let weights: [String: MLXArray] = [
      "transformer_blocks.0.attn1.to_q.weight": foreignInt8Weight(),
      "transformer_blocks.0.attn1.to_q.weight_scale": foreignScale(),
      "transformer_blocks.0.attn1.norm_q.weight": passthrough,
    ]

    let out = try LTX2Quantizer.dequantizeForeignInt8Weights(weights)

    XCTAssertEqual(out["transformer_blocks.0.attn1.norm_q.weight"]?.dtype, .bfloat16)
    XCTAssertEqual(out.count, weights.count - 1, "the consumed .weight_scale sidecar is dropped, nothing else changes count")
  }

  // MARK: - Fail loud instead of rendering noise

  /// An int8 `.weight` with NO recognised scale sidecar at all must throw a
  /// named error identifying the offending key, rather than silently
  /// loading raw bytes as float (the original bug).
  func testInt8WeightWithoutScaleSidecarThrowsNamedError() {
    let weights: [String: MLXArray] = [
      "transformer_blocks.3.ff.proj_in.weight": foreignInt8Weight()
    ]

    XCTAssertThrowsError(try LTX2Quantizer.dequantizeForeignInt8Weights(weights)) { error in
      guard case LTX2Quantizer.ForeignQuantError.missingScaleSidecar(let key) = error else {
        return XCTFail("expected .missingScaleSidecar, got \(error)")
      }
      XCTAssertEqual(key, "transformer_blocks.3.ff.proj_in.weight")
    }
  }

  // MARK: - Toy module

  /// A minimal `Module` with one `Linear` submodule at path "to_q", keyed
  /// the same way `applyQuantizedLayout`'s `MLXNN.quantize(model:filter:)`
  /// walk addresses real transformer projections — without paying for a
  /// full 48-layer `LTX2Transformer`.
  private final class ToyLinearHolder: Module {
    @ModuleInfo(key: "to_q") var to_q: Linear
    init(weight: MLXArray) {
      self._to_q.wrappedValue = Linear(weight: weight, bias: nil)
      super.init()
    }
  }
}
