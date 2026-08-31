// Krea2LoKrDensifierTests.swift — comfybox#329: full-matrix LoKr as a
// transactional `.diff` delta on the Krea-2 per-render dynamic path.
//
// The feature under test: ``LoKrDensifier`` materializes each provable
// full-matrix LoKr layer into ΔW = kron(w1, w2) · lokrAlphaScale and routes
// it through ``LoRAPatchSession`` — the mechanism whose first-write-wins
// snapshots (exact packed q8 weight/scales/biases tuple included) give the
// apply/clear cycle an exact restore. After conversion `lokrLayerCount == 0`
// and the K-FIX-1/C1 guard (``Krea2AdapterSupport/checkTransactional``)
// passes; the guard itself is unchanged and stays the fail-closed backstop
// (pinned below). `Krea2LoKrRefusalTests` keeps pinning the guard's own
// behaviour; these tests pin the conversion.
//
// All offline: toy modules, synthetic safetensors fixtures, no checkpoint,
// no warm server.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

final class Krea2LoKrDensifierTests: XCTestCase {

  private var tmp: URL!

  override func setUpWithError() throws {
    tmp = URL(fileURLWithPath: NSTemporaryDirectory())
      .appendingPathComponent("krea2-lokr-densifier-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tmp)
  }

  private func write(_ arrays: [String: MLXArray], as name: String) throws -> URL {
    let url = tmp.appendingPathComponent(name)
    try MLX.save(arrays: arrays, url: url)
    return url
  }

  /// Deterministic, non-uniform matrix — uniform fixtures hide transpose
  /// and scale-ordering faults.
  private func ramp(_ rows: Int, _ cols: Int, step: Float) -> MLXArray {
    MLXArray((0..<(rows * cols)).map { Float($0) * step }).reshaped(rows, cols)
  }

  // MARK: - Toys

  private final class QToy: Module {
    @ModuleInfo(key: "qlin") var qlin: QuantizedLinear
    init(dim: Int) {
      let lin = Linear(dim, dim, bias: false)
      self._qlin.wrappedValue = QuantizedLinear(
        weight: lin.weight, bias: nil, groupSize: 64, bits: 8)
      super.init()
    }
  }

  private final class LinToy: Module {
    @ModuleInfo(key: "lin") var lin: Linear
    init(dim: Int) {
      self._lin.wrappedValue = Linear(dim, dim, bias: false)
      super.init()
    }
  }

  private func packedTuple(_ toy: QToy) -> (weight: [UInt32], scales: [Float], biases: [Float]?) {
    (toy.qlin.weight.asArray(UInt32.self),
     toy.qlin.scales.asArray(Float.self),
     toy.qlin.biases?.asArray(Float.self))
  }

  // MARK: - (a) THE transactional AC: quantized apply → clear is byte-exact

  /// The C1 complaint was that in-place LoKr on q8 destroys the original
  /// packed bytes unrecoverably. The densified path must not: apply changes
  /// the packed representation, clear restores the EXACT packed tuple —
  /// compared as stored (packed uint32 words + scales/biases), never as
  /// roundtripped floats.
  func testQuantizedLoKrAsDiffAppliesAndClearsByteIdentical() throws {
    let toy = QToy(dim: 64)
    let before = packedTuple(toy)

    // 8x8 ⊗ 8x8 = 64x64, non-uniform halves.
    let url = try write([
      "diffusion_model.qlin.lokr_w1": ramp(8, 8, step: 0.002),
      "diffusion_model.qlin.lokr_w2": ramp(8, 8, step: 0.003),
    ], as: "full-matrix.safetensors")

    let loaded = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(loaded.lokrLayerCount, 1, "the loader still loads LoKr")

    let weights = try LoKrDensifier.densify(loaded, for: toy, name: url.lastPathComponent)
    XCTAssertEqual(weights.lokrLayerCount, 0, "every layer densified")
    XCTAssertNotNil(weights.deltas["qlin.weight"], "…into a diff on the target weight")
    XCTAssertNoThrow(
      try Krea2AdapterSupport.checkTransactional(
        lokrLayerCount: weights.lokrLayerCount, lora: url.lastPathComponent),
      "the C1 guard passes untouched")

    let session = LoRAPatchSession(module: toy)
    // Preflight must be quantized-aware: the delta is (64, 64) float while
    // the stored packed weight is (64, 16) uint32.
    XCTAssertEqual(try session.preflight(weights: weights), 1)

    XCTAssertEqual(try session.apply(weights: weights, scale: 0.8), 1)
    XCTAssertNotEqual(
      toy.qlin.weight.asArray(UInt32.self), before.weight,
      "apply must actually change the packed representation")

    session.clear()
    let after = packedTuple(toy)
    XCTAssertEqual(after.weight, before.weight, "packed weight bytes restored exactly")
    XCTAssertEqual(after.scales, before.scales, "scales restored exactly")
    XCTAssertEqual(after.biases, before.biases, "biases restored exactly")
  }

  // MARK: - (b) Math equivalence with the in-place path

  /// On an unquantized layer the densified `.diff` must land on the same
  /// weights as the existing in-place `applyLoKr` — same w1/w2/alpha/userScale
  /// fixture, including a real (non-1.0) alpha/dim folding.
  func testDenseDeltaMatchesInPlaceApplyLoKr() throws {
    let dim = 16
    let toyInPlace = LinToy(dim: dim)
    let toyDiff = LinToy(dim: dim)
    // Same starting weights, detached (Linear inits randomly).
    toyDiff.update(parameters: ModuleParameters.unflattened([
      ("lin.weight", toyInPlace.lin.weight + MLXArray(Float(0)))
    ]))

    // 4x4 ⊗ 4x4 = 16x16; alpha 2 over dim min(4,4) → alphaScale 0.5.
    let url = try write([
      "diffusion_model.lin.lokr_w1": ramp(4, 4, step: 0.01),
      "diffusion_model.lin.lokr_w2": ramp(4, 4, step: 0.02),
      "diffusion_model.lin.alpha": MLXArray(Float(2.0)),
    ], as: "equivalence.safetensors")
    let loaded = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(loaded.lokrWeights["lin"]?.alpha, 2.0)

    let userScale: Float = 0.7
    LoRAApplicator.applyLoKr(to: toyInPlace, loraWeights: loaded, scale: userScale)

    let weights = try LoKrDensifier.densify(loaded, for: toyDiff, name: nil)
    let session = LoRAPatchSession(module: toyDiff)
    try session.apply(weights: weights, scale: userScale)

    let expected = toyInPlace.lin.weight.asArray(Float.self)
    let actual = toyDiff.lin.weight.asArray(Float.self)
    XCTAssertEqual(expected.count, actual.count)
    for i in expected.indices {
      XCTAssertEqual(actual[i], expected[i], accuracy: 1e-4,
        "densified delta diverges from in-place applyLoKr at flat index \(i)")
    }
  }

  // MARK: - (c) snofs-shaped fixture: scale folding + accounting

  /// w1 [4,4] with a scalar alpha, the shape of the snofs adapter. alpha 1
  /// over w2 [2,2] → dim 2 → alphaScale 0.5, folded into the stored delta
  /// (LoRAPatchSession applies userScale alone, by design).
  func testSnofsShapedFixtureFoldsAlphaScaleAndZeroesLoKrCount() throws {
    let toy = LinToy(dim: 8)
    let base = toy.lin.weight.asArray(Float.self)

    // Uniform ON PURPOSE here: makes the expected per-element delta exact.
    let url = try write([
      "diffusion_model.lin.lokr_w1": MLXArray.ones([4, 4]) * Float(0.2),
      "diffusion_model.lin.lokr_w2": MLXArray.ones([2, 2]) * Float(0.1),
      "diffusion_model.lin.alpha": MLXArray(Float(1.0)),
    ], as: "snofs-shaped.safetensors")
    let loaded = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(loaded.lokrLayerCount, 1)

    let weights = try LoKrDensifier.densify(loaded, for: toy, name: nil)
    XCTAssertEqual(weights.lokrLayerCount, 0)
    XCTAssertEqual(weights.deltas.count, 1)
    XCTAssertNoThrow(try Krea2AdapterSupport.checkTransactional(
      lokrLayerCount: weights.lokrLayerCount, lora: "snofs-shaped.safetensors"))

    let session = LoRAPatchSession(module: toy)
    try session.apply(weights: weights, scale: 1.0)
    let applied = toy.lin.weight.asArray(Float.self)
    // kron element = 0.2 × 0.1 = 0.02; × alphaScale 0.5 = 0.01.
    for i in applied.indices {
      XCTAssertEqual(applied[i] - base[i], 0.01, accuracy: 1e-5,
        "alpha/dim folding wrong at flat index \(i)")
    }
    session.clear()
    XCTAssertEqual(toy.lin.weight.asArray(Float.self), base)
  }

  /// The ai-toolkit ~1e10 alpha sentinel must keep falling back to a neutral
  /// 1.0 on the densified path, exactly as `lokrAlphaScale` documents.
  func testAiToolkitAlphaSentinelFallsBackToNeutralScale() throws {
    let toy = LinToy(dim: 8)
    let w1 = MLXArray.ones([4, 4]) * Float(0.2)
    let w2 = MLXArray.ones([2, 2]) * Float(0.1)
    let weights = LoRAWeights(
      weights: [String: (down: MLXArray, up: MLXArray)](),
      lokrWeights: ["lin": LoKrWeights(w1: w1, w2: w2, alpha: 1e10)],
      rank: 0)

    let densified = try LoKrDensifier.densify(weights, for: toy, name: nil)
    let delta = try XCTUnwrap(densified.deltas["lin.weight"]).tensor.asArray(Float.self)
    for value in delta {
      XCTAssertEqual(value, 0.02, accuracy: 1e-6,
        "sentinel alpha must not scale the delta (fallback 1.0)")
    }
  }

  // MARK: - (d) Factored / Tucker LoKr stays refused

  /// Factored LoKr (`w1_a/w1_b`, `w2_a/w2_b`) never reaches the densifier:
  /// `loadForKrea2` refuses the key spellings outright (fail-loud
  /// `unknownKeys`, before any weight is touched). Pinned so a loader change
  /// can't silently start half-loading factored adapters.
  func testFactoredLoKrFileIsStillRefusedByTheLoader() throws {
    let url = try write([
      "diffusion_model.lin.lokr_w1_a": MLXArray.ones([8, 2]),
      "diffusion_model.lin.lokr_w1_b": MLXArray.ones([2, 8]),
      "diffusion_model.lin.lokr_w2_a": MLXArray.ones([8, 2]),
      "diffusion_model.lin.lokr_w2_b": MLXArray.ones([2, 8]),
    ], as: "factored.safetensors")
    XCTAssertThrowsError(try LoRAWeightLoader.loadForKrea2(from: url)) { error in
      guard case LoRAError.unknownKeys(let keys) = error else {
        return XCTFail("expected unknownKeys, got \(error)")
      }
      XCTAssertEqual(keys.count, 4, "every factored half named, none loaded")
    }
  }

  /// Tucker LoKr (`lokr_t1`/`lokr_t2`) likewise.
  func testTuckerLoKrFileIsStillRefusedByTheLoader() throws {
    let url = try write([
      "diffusion_model.lin.lokr_w1": MLXArray.ones([4, 4]),
      "diffusion_model.lin.lokr_w2": MLXArray.ones([2, 2]),
      "diffusion_model.lin.lokr_t2": MLXArray.ones([2, 2, 2]),
    ], as: "tucker.safetensors")
    XCTAssertThrowsError(try LoRAWeightLoader.loadForKrea2(from: url)) { error in
      guard case LoRAError.unknownKeys(let keys) = error else {
        return XCTFail("expected unknownKeys, got \(error)")
      }
      XCTAssertEqual(keys, ["diffusion_model.lin.lokr_t2"])
    }
  }

  /// Belt and braces at the densifier itself: a non-2-D half constructed in
  /// memory (bypassing the loader) is refused with the path's established
  /// `unsupportedAdapter` taxonomy — only full-matrix LoKr is in scope.
  func testNonFullMatrixLayerIsRefusedWithUnsupportedAdapter() throws {
    let toy = LinToy(dim: 8)
    let weights = LoRAWeights(
      weights: [String: (down: MLXArray, up: MLXArray)](),
      lokrWeights: ["lin": LoKrWeights(w1: MLXArray.ones([2, 2, 2]), w2: MLXArray.ones([2, 2]))],
      rank: 0)
    XCTAssertThrowsError(try LoKrDensifier.densify(weights, for: toy, name: "tucker.safetensors")) { error in
      guard case LoRAError.unsupportedAdapter(let lora, let format, let reason) = error else {
        return XCTFail("expected unsupportedAdapter, got \(error)")
      }
      XCTAssertEqual(lora, "tucker.safetensors")
      XCTAssertEqual(format, "LoKr")
      XCTAssertTrue(reason.contains("full-matrix"), reason)
    }
  }

  // MARK: - (e) apply → clear → apply across two simulated renders

  /// The compounding bug C1 closed must stay closed on the new path: two
  /// full render cycles land on identical packed bytes, and clearing after
  /// either returns to the identical base.
  func testApplyClearApplyIsIdempotentAcrossSimulatedRenders() throws {
    let toy = QToy(dim: 64)
    let base = packedTuple(toy)

    let url = try write([
      "diffusion_model.qlin.lokr_w1": ramp(8, 8, step: 0.002),
      "diffusion_model.qlin.lokr_w2": ramp(8, 8, step: 0.003),
    ], as: "renders.safetensors")
    let weights = try LoKrDensifier.densify(
      try LoRAWeightLoader.loadForKrea2(from: url), for: toy, name: nil)

    let session = LoRAPatchSession(module: toy)

    // Render 1.
    try session.apply(weights: weights, scale: 1.0)
    let render1 = packedTuple(toy)
    XCTAssertNotEqual(render1.weight, base.weight)
    session.clear()
    XCTAssertEqual(packedTuple(toy).weight, base.weight)

    // Render 2 — same file, fresh apply, as loadLoRAs does per render.
    try session.apply(weights: weights, scale: 1.0)
    let render2 = packedTuple(toy)
    XCTAssertEqual(render2.weight, render1.weight, "no accumulation across renders")
    XCTAssertEqual(render2.scales, render1.scales)
    session.clear()
    let restored = packedTuple(toy)
    XCTAssertEqual(restored.weight, base.weight)
    XCTAssertEqual(restored.scales, base.scales)
    XCTAssertEqual(restored.biases, base.biases)
  }

  // MARK: - Fail-closed backstop and refusals

  /// A LoKr key no bindable module answers to stays LoKr-shaped, so the C1
  /// guard still refuses the file whole — the densifier must never let an
  /// unprovable layer slip through as "converted".
  func testUnboundLoKrKeyLeavesGuardRefusingTheFile() throws {
    let toy = LinToy(dim: 8)
    let url = try write([
      "diffusion_model.ghost.lokr_w1": MLXArray.ones([4, 4]),
      "diffusion_model.ghost.lokr_w2": MLXArray.ones([2, 2]),
    ], as: "ghost.safetensors")
    let weights = try LoKrDensifier.densify(
      try LoRAWeightLoader.loadForKrea2(from: url), for: toy, name: "ghost.safetensors")
    XCTAssertEqual(weights.lokrLayerCount, 1, "unprovable layer NOT converted")
    XCTAssertThrowsError(try Krea2AdapterSupport.checkTransactional(
      lokrLayerCount: weights.lokrLayerCount, lora: "ghost.safetensors")) { error in
      guard case LoRAError.unsupportedAdapter = error else {
        return XCTFail("expected unsupportedAdapter, got \(error)")
      }
    }
  }

  /// A kron that names a real target it cannot fit is a loud shape refusal
  /// naming both shapes — never a silent skip on this strict path.
  func testKronShapeMismatchOnRealTargetThrowsIncompatibleWeights() throws {
    let toy = LinToy(dim: 16)  // target 16x16
    let url = try write([
      "diffusion_model.lin.lokr_w1": MLXArray.ones([4, 4]),
      "diffusion_model.lin.lokr_w2": MLXArray.ones([2, 2]),  // kron 8x8
    ], as: "mismatch.safetensors")
    let loaded = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertThrowsError(
      try LoKrDensifier.densify(loaded, for: toy, name: "mismatch.safetensors")
    ) { error in
      guard case LoRAError.incompatibleWeights(let message) = error else {
        return XCTFail("expected incompatibleWeights, got \(error)")
      }
      XCTAssertTrue(message.contains("8x8"), message)
      XCTAssertTrue(message.contains("16x16"), message)
    }
  }

  /// One target carrying both a LoKr layer and a bare patch is refused
  /// rather than guessing an application order (same posture as
  /// `LoRABareParameterPairs`).
  func testLoKrPlusBarePatchOnSameTargetIsRefused() throws {
    let toy = LinToy(dim: 8)
    let weights = LoRAWeights(
      weights: [String: (down: MLXArray, up: MLXArray)](),
      lokrWeights: ["lin": LoKrWeights(w1: MLXArray.ones([4, 4]), w2: MLXArray.ones([2, 2]))],
      rank: 0,
      deltas: ["lin.weight": .diff(MLXArray.ones([8, 8]))])
    XCTAssertThrowsError(try LoKrDensifier.densify(weights, for: toy, name: nil)) { error in
      guard case LoRAError.incompatibleWeights(let message) = error else {
        return XCTFail("expected incompatibleWeights, got \(error)")
      }
      XCTAssertTrue(message.contains("bare patch"), message)
    }
  }

  /// A file with no LoKr at all passes through untouched — the densifier is
  /// inert for every adapter loading today.
  func testNonLoKrWeightsPassThroughUnchanged() throws {
    let toy = LinToy(dim: 8)
    let pair = (down: MLXArray.ones([2, 8]), up: MLXArray.ones([8, 2]))
    let weights = LoRAWeights(
      weights: ["lin.weight": pair], rank: 2, alpha: 2.0,
      deltas: ["lin.weight2": .diff(MLXArray.ones([8, 8]))])
    let out = try LoKrDensifier.densify(weights, for: toy, name: nil)
    XCTAssertEqual(out.weights.count, 1)
    XCTAssertEqual(out.deltas.count, 1)
    XCTAssertEqual(out.lokrLayerCount, 0)
    XCTAssertEqual(out.alpha, 2.0)
    XCTAssertEqual(out.rank, 2)
  }
}
