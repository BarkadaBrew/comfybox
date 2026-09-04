// LTX2ForeignInt8LoadTests.swift — comfybox#256.
//
// Round 1 fix history: the first pass assumed PinkCherry v1.7's int8
// checkpoint was plain per-channel `int8 × scale` (an `F32 .weight_scale`
// sidecar IS present and correctly named — see
// docs/HANDOFF-ltx-quality-2026-08-02.md §5) and dequantized it that way.
// Codex review opened the real checkpoint and ComfyUI's `comfy/ops.py` and
// found 1,344 `.comfy_quant` sidecar records declaring ComfyUI's
// `int8_tensorwise` format with `convrot: true`, `convrot_groupsize: 256`
// — a rotated coordinate basis. Plain `int8 × scale` against that format
// measured cosine similarity ~0.008 against the matching bf16 checkpoint:
// nowhere close to correct, and NOT fixable by adjusting a scale multiply.
//
// ConvRot-aware dequantization is a real feature (tracked as a #256
// follow-up), not a load-path fix. This round instead makes the loader
// FAIL LOUD on any int8 layout it does not implement — including
// ComfyUI's `.comfy_quant`/ConvRot format — naming the exact tensor and
// format, before any weight is materialised, rather than silently
// reconstructing garbage (round 1's regression) or the original bug
// (loading raw int8 bytes as if they were already float, comfybox#256's
// original report).
//
// `applyQuantizedLayout` (Sources/ZImage/LTX2/LTX2Quantizer.swift, the
// `scalesKeys = ... filter { $0.hasSuffix(".scales") }` scan) only ever
// recognises `LTX2Quantizer`'s own MLX affine group-wise output (packed
// `uint32` + `.scales`/`.biases`, which is never `int8`-dtype). Any
// `int8`-dtype `.weight` tensor is therefore, by construction, NOT that
// format — `LTX2Quantizer.rejectUnsupportedInt8Weights` is the guard that
// catches it before `LTX2VideoGenerator.load()` reaches
// `transformer.update(parameters:)`.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

final class LTX2ForeignInt8LoadTests: XCTestCase {

  // MARK: - Fixtures

  private func int8Weight() -> MLXArray {
    MLXArray([Int8]([-127, -64, -10, 64, 10, 20, 30, 40]), [2, 4])
  }

  private func weightScaleSidecar() -> MLXArray {
    MLXArray([Float(0.01), Float(0.02)], [2, 1])
  }

  /// Builds a ComfyUI `.comfy_quant` sidecar exactly as `comfy/ops.py`
  /// writes it: raw UTF-8 JSON bytes in a `uint8` tensor
  /// (`torch.tensor(list(json.dumps(quant_conf).encode("utf-8")),
  /// dtype=torch.uint8)`).
  private func comfyQuantRecord(
    format: String = "int8_tensorwise", convrot: Bool = true, groupSize: Int = 256
  ) -> MLXArray {
    let json: [String: Any] = ["format": format, "convrot": convrot, "convrot_groupsize": groupSize]
    let data = try! JSONSerialization.data(withJSONObject: json)
    return MLXArray([UInt8](data))
  }

  private func floats(_ a: MLXArray) -> [Float] {
    let flat = a.asType(.float32).flattened()
    eval(flat)
    return flat.asArray(Float.self)
  }

  // MARK: - Root cause: the loader's own recognition code ignores int8 layouts

  /// `applyQuantizedLayout` only recognises its own MLX affine output
  /// (`.scales` sibling, packed `uint32`). Neither a foreign
  /// `.weight_scale` sidecar nor a ComfyUI `.comfy_quant` record is
  /// visible to it — proving why an unguarded load would silently leave
  /// the Linear plain and let raw int8 bytes reach `update(parameters:)`.
  func testApplyQuantizedLayoutDoesNotRecognizeForeignInt8Layouts() {
    let toy = ToyLinearHolder(weight: MLXArray.zeros([2, 4]))
    let comfyDict: [String: MLXArray] = [
      "to_q.weight": int8Weight(),
      "to_q.comfy_quant": comfyQuantRecord(),
    ]
    let scaleOnlyDict: [String: MLXArray] = [
      "to_q.weight": int8Weight(),
      "to_q.weight_scale": weightScaleSidecar(),
    ]

    XCTAssertEqual(LTX2Quantizer.applyQuantizedLayout(to: toy, sanitizedWeights: comfyDict), 0)
    XCTAssertEqual(LTX2Quantizer.applyQuantizedLayout(to: toy, sanitizedWeights: scaleOnlyDict), 0)
  }

  /// Sanity check on the positive case: LTX2Quantizer's own layout (packed
  /// uint32 `.weight` + `.scales`) IS recognised and converts the layer —
  /// (a) from the fix-round test matrix: native packed layout accepted.
  func testApplyQuantizedLayoutRecognizesOwnScalesLayout() {
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

  /// (a), the negative complement: a plain-layout dict has no int8 tensors
  /// at all, so `rejectUnsupportedInt8Weights` must not throw for it.
  func testNativePackedLayoutIsAccepted() throws {
    let dense = MLXRandom.normal([2, 32])
    let (wq, scales, biases) = MLX.quantized(dense, groupSize: 32, bits: 8, mode: .affine)
    var weights: [String: MLXArray] = [
      "transformer_blocks.0.attn1.to_q.weight": wq,
      "transformer_blocks.0.attn1.to_q.scales": scales,
      "transformer_blocks.0.attn1.norm_q.weight": MLXArray.zeros([32], dtype: .bfloat16),
    ]
    if let biases { weights["transformer_blocks.0.attn1.to_q.biases"] = biases }

    XCTAssertNoThrow(try LTX2Quantizer.rejectUnsupportedInt8Weights(weights))
  }

  // MARK: - Pinning the noise bug (why plain int8×scale is not a fix)

  /// Round 1's mistake, still worth pinning: raw int8 bytes are nowhere
  /// near a real dequantized weight if simply cast to float. This is why
  /// the loader must never let a `.weight` reach `update(parameters:)`
  /// while still `int8`.
  func testNaiveInt8AsFloatIsNotAReasonableWeight() {
    let naive = floats(int8Weight().asType(.float32))
    for v in naive {
      XCTAssertGreaterThan(abs(v), 1.0, "raw int8 values are integers in roughly [-127, 127], not plausible weight magnitudes")
    }
  }

  // MARK: - (b) int8 + .comfy_quant -> typed error naming ConvRot

  /// Pins the round-1 regression: BEFORE this round's fix, an int8 tensor
  /// with a `.comfy_quant` ConvRot record was silently accepted by the
  /// (now-removed) plain-scale dequantizer, reconstructing garbage
  /// (cosine similarity ~0.008 against the real weights per Codex's
  /// review). After the fix, it must be rejected loudly, naming ConvRot.
  func testComfyQuantConvRotFormatThrowsNamedError() {
    let weights: [String: MLXArray] = [
      "transformer_blocks.0.attn1.to_k.weight": int8Weight(),
      "transformer_blocks.0.attn1.to_k.weight_scale": weightScaleSidecar(),
      "transformer_blocks.0.attn1.to_k.comfy_quant": comfyQuantRecord(
        format: "int8_tensorwise", convrot: true, groupSize: 256),
    ]

    XCTAssertThrowsError(try LTX2Quantizer.rejectUnsupportedInt8Weights(weights)) { error in
      guard case LTX2Quantizer.UnsupportedInt8FormatError.comfyQuantFormat(let key, let format, let convrot, let groupSize) = error else {
        return XCTFail("expected .comfyQuantFormat, got \(error)")
      }
      XCTAssertEqual(key, "transformer_blocks.0.attn1.to_k.weight")
      XCTAssertEqual(format, "int8_tensorwise")
      XCTAssertTrue(convrot)
      XCTAssertEqual(groupSize, 256)
      XCTAssertTrue(
        (error as? LocalizedError)?.errorDescription?.contains("convrot=true") ?? false,
        "the surfaced message must name convrot explicitly")
      XCTAssertTrue(
        (error as? LocalizedError)?.errorDescription?.contains("group=256") ?? false,
        "the surfaced message must name the ConvRot group size")
    }
  }

  /// A `.comfy_quant` record without `convrot` set (some other ComfyUI
  /// int8 format) is still refused — parsed fields just report `false`.
  func testComfyQuantNonConvRotFormatStillThrows() {
    let weights: [String: MLXArray] = [
      "transformer_blocks.1.ff.proj_in.weight": int8Weight(),
      "transformer_blocks.1.ff.proj_in.comfy_quant": comfyQuantRecord(
        format: "int8_tensorwise", convrot: false, groupSize: 64),
    ]

    XCTAssertThrowsError(try LTX2Quantizer.rejectUnsupportedInt8Weights(weights)) { error in
      guard case LTX2Quantizer.UnsupportedInt8FormatError.comfyQuantFormat(_, let format, let convrot, let groupSize) = error else {
        return XCTFail("expected .comfyQuantFormat, got \(error)")
      }
      XCTAssertEqual(format, "int8_tensorwise")
      XCTAssertFalse(convrot)
      XCTAssertEqual(groupSize, 64)
    }
  }

  // MARK: - (c) int8 + .weight_scale only -> typed error naming the sidecar

  func testWeightScaleOnlySidecarThrowsNamedError() {
    let weights: [String: MLXArray] = [
      "transformer_blocks.2.attn1.to_v.weight": int8Weight(),
      "transformer_blocks.2.attn1.to_v.weight_scale": weightScaleSidecar(),
    ]

    XCTAssertThrowsError(try LTX2Quantizer.rejectUnsupportedInt8Weights(weights)) { error in
      guard case LTX2Quantizer.UnsupportedInt8FormatError.unidentifiedWeightScaleSidecar(let key, let sidecarKey) = error else {
        return XCTFail("expected .unidentifiedWeightScaleSidecar, got \(error)")
      }
      XCTAssertEqual(key, "transformer_blocks.2.attn1.to_v.weight")
      XCTAssertEqual(sidecarKey, "transformer_blocks.2.attn1.to_v.weight_scale")
    }
  }

  /// An int8 `.weight` with NEITHER a `.comfy_quant` record NOR a
  /// `.weight_scale` sidecar — no layout at all applies.
  func testInt8WeightWithNoSidecarAtAllThrowsNamedError() {
    let weights: [String: MLXArray] = [
      "transformer_blocks.3.attn1.to_out.weight": int8Weight()
    ]

    XCTAssertThrowsError(try LTX2Quantizer.rejectUnsupportedInt8Weights(weights)) { error in
      guard case LTX2Quantizer.UnsupportedInt8FormatError.noRecognisedSidecar(let key) = error else {
        return XCTFail("expected .noRecognisedSidecar, got \(error)")
      }
      XCTAssertEqual(key, "transformer_blocks.3.attn1.to_out.weight")
    }
  }

  // MARK: - (d) a non-.weight int8 tensor names the tensor's role

  func testNonWeightInt8TensorNamesItsRoleNotMissingSidecar() {
    let weights: [String: MLXArray] = [
      "transformer_blocks.4.ff.proj_out.weight_codebook": MLXArray([Int8]([1, 2, 3]))
    ]

    XCTAssertThrowsError(try LTX2Quantizer.rejectUnsupportedInt8Weights(weights)) { error in
      guard case LTX2Quantizer.UnsupportedInt8FormatError.unsupportedInt8TensorRole(let key) = error else {
        return XCTFail("expected .unsupportedInt8TensorRole, got \(error)")
      }
      XCTAssertEqual(key, "transformer_blocks.4.ff.proj_out.weight_codebook")
      let message = (error as? LocalizedError)?.errorDescription ?? ""
      XCTAssertTrue(message.contains(key), "message must name the tensor's own key/role")
      XCTAssertFalse(
        message.contains("missing sidecar"),
        "a non-.weight tensor is an unsupported role, not literally a 'missing sidecar' — different failure, different wording")
    }
  }

  // MARK: - Round-1 regression guard: no dequantized weight is ever produced

  /// The round-1 dequantize path is gone entirely — this locks that in.
  /// If it (or anything like it) comes back, this fails to compile, which
  /// is the point: no plain int8×scale reconstruction is an acceptable
  /// "fix" for a format we haven't verified isn't rotated (comfybox#256).
  func testNoPlainDequantizationPathExists() {
    // `LTX2Quantizer` must not expose a function that returns dequantized
    // weights for a foreign int8 layout — only the loud rejection above.
    XCTAssertThrowsError(try LTX2Quantizer.rejectUnsupportedInt8Weights([
      "x.weight": int8Weight(),
      "x.weight_scale": weightScaleSidecar(),
    ])) { _ in }
  }

  // MARK: - The sanitizer preserves .comfy_quant markers (not silently dropped)

  /// `LTX2Transformer.sanitizeWeights` renames keys by whole-path substring
  /// replacement (prefix strip + a few fixed substring renames) and does
  /// not filter on suffix, so a `.comfy_quant` sidecar survives the same
  /// as `.scales`/`.biases` do. That matters here: it's what lets
  /// `rejectUnsupportedInt8Weights` see the record at all (the loud error
  /// fires before anything downstream has a chance to drop it — the
  /// coverage guard in `LTX2VideoGenerator.load()` is never reached).
  func testSanitizePreservesComfyQuantSidecar() {
    let record = comfyQuantRecord()
    let weights: [String: MLXArray] = [
      "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight": int8Weight(),
      "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.comfy_quant": record,
    ]

    let sanitized = LTX2Transformer.sanitizeWeights(weights)

    XCTAssertNotNil(sanitized["transformer_blocks.0.attn1.to_out.weight"])
    XCTAssertNotNil(
      sanitized["transformer_blocks.0.attn1.to_out.comfy_quant"],
      "sanitizeWeights must not silently drop the .comfy_quant marker — rejectUnsupportedInt8Weights relies on seeing it")
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
