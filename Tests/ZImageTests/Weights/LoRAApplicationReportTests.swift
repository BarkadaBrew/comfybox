import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

/// WP-E6 (FDD §3.6, D9, AC-42 / AC-42a): `LoRAApplicator.applyDynamically`
/// returns a `LoRAApplicationReport` and, under `strict: true`, refuses to
/// bind a LoRA partially. Every test here is weight-free: a toy module whose
/// Linear children are keyed like the Krea-2 DiT (`blocks.N.attn.w{q,k,v,o}`)
/// exercises the same module walk the real transformer does.
///
/// The detection-site correction from D9 is what these tests pin:
/// `unbound` is **offered keys minus consumed keys**, never a list of the
/// modules that happened to have no adapter.
final class LoRAApplicationReportTests: XCTestCase {

  // MARK: - Toy module (Krea-2-shaped paths)

  private final class ToyAttn: Module {
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(dim: Int) {
      // Deterministic base weights so "nothing mutated" is a byte check.
      func lin(_ seed: Float) -> Linear {
        let w = MLXArray.ones([dim, dim]).asType(.float32) * seed
        return Linear(weight: w, bias: nil)
      }
      self._wq.wrappedValue = lin(0.1)
      self._wk.wrappedValue = lin(0.2)
      self._wv.wrappedValue = lin(0.3)
      self._wo.wrappedValue = lin(0.4)
      super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
      wo(wv(wk(wq(x))))
    }
  }

  private final class ToyBlock: Module {
    @ModuleInfo(key: "attn") var attn: ToyAttn
    init(dim: Int) {
      self._attn.wrappedValue = ToyAttn(dim: dim)
      super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { attn(x) }
  }

  private final class ToyDiT: Module {
    @ModuleInfo(key: "blocks") var blocks: [ToyBlock]
    init(blocks n: Int, dim: Int) {
      self._blocks.wrappedValue = (0..<n).map { _ in ToyBlock(dim: dim) }
      super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
      var h = x
      for b in blocks { h = b(h) }
      return h
    }
  }

  private let dim = 4
  private let blockCount = 2

  /// The eight Linear targets of a 2-block toy, as `.weight` keys.
  private var allTargetKeys: [String] {
    (0..<blockCount).flatMap { i in
      ["wq", "wk", "wv", "wo"].map { "blocks.\(i).attn.\($0).weight" }
    }
  }

  /// A rank-1 pair that binds a `dim × dim` Linear.
  private func pair(rank: Int = 1, inFeatures: Int? = nil) -> (down: MLXArray, up: MLXArray) {
    let inF = inFeatures ?? dim
    return (
      down: MLXArray.ones([rank, inF]).asType(.float32),
      up: (MLXArray.ones([dim, rank]) * Float(0.01)).asType(.float32)
    )
  }

  private func fullLoRA(extraKeys: [String: (down: MLXArray, up: MLXArray)] = [:]) -> LoRAWeights {
    var w: [String: (down: MLXArray, up: MLXArray)] = [:]
    for k in allTargetKeys { w[k] = pair() }
    for (k, v) in extraKeys { w[k] = v }
    return LoRAWeights(weights: w, rank: 1)
  }

  private func probe(_ model: ToyDiT) -> [Float] {
    let x = MLXArray([Float](repeating: 1.0, count: dim), [1, dim])
    let y = model(x)
    eval(y)
    return y.asArray(Float.self)
  }

  // MARK: - Report shape

  func testFullBindReportsBoundEqualsOffered() throws {
    let model = ToyDiT(blocks: blockCount, dim: dim)
    let report = try LoRAApplicator.applyDynamically(
      to: model, loraWeights: fullLoRA(), scale: 1.0, strict: true)

    XCTAssertEqual(report.offered, 8)
    XCTAssertEqual(report.bound, 8)
    XCTAssertEqual(report.quantizedBound, 0)
    XCTAssertEqual(report.deltasApplied, 0, "applyDynamically never applies deltas itself")
    XCTAssertEqual(report.shapeRejected, 0)
    XCTAssertTrue(report.unbound.isEmpty)
    XCTAssertTrue(report.isComplete)
  }

  /// AC-42a — `unbound` names OFFERED keys that bound nothing. A LoRA whose
  /// keys include one that targets no module reports `bound == offered - 1`
  /// and `unbound == [that key]` — not thousands of entries for the modules
  /// that had no adapter, and not an empty list.
  func testUnboundIsOfferedMinusConsumed() throws {
    let phantom = "blocks.0.attn.phantom.weight"
    let model = ToyDiT(blocks: blockCount, dim: dim)
    let lora = fullLoRA(extraKeys: [phantom: pair()])

    let report = try LoRAApplicator.applyDynamically(
      to: model, loraWeights: lora, scale: 1.0, strict: false)

    XCTAssertEqual(report.offered, 9)
    XCTAssertEqual(report.bound, 8)
    XCTAssertEqual(report.unbound, [phantom])
    XCTAssertEqual(report.shapeRejected, 0)
    XCTAssertFalse(report.isComplete)
  }

  /// AC-42a (second half) — a pair that matches a module but fails
  /// `normalizeLoRAPair` increments `shapeRejected` and appears in neither
  /// `bound` nor `unbound`.
  func testShapeRejectedIsCountedSeparately() throws {
    let model = ToyDiT(blocks: blockCount, dim: dim)
    var lora = fullLoRA()
    // in-features 5 can never bind a 4×4 Linear.
    var w = lora.weights
    w["blocks.1.attn.wo.weight"] = pair(inFeatures: dim + 1)
    lora = LoRAWeights(weights: w, rank: 1)

    let report = try LoRAApplicator.applyDynamically(
      to: model, loraWeights: lora, scale: 1.0, strict: false)

    XCTAssertEqual(report.offered, 8)
    XCTAssertEqual(report.bound, 7)
    XCTAssertEqual(report.shapeRejected, 1)
    XCTAssertTrue(report.unbound.isEmpty, "a shape-rejected key is not an unbound key")
  }

  // MARK: - Strict

  /// AC-42a (strict) — the phantom key throws `partialApplication` naming
  /// the key, and NOTHING is mutated: the base forward pass is unchanged.
  func testStrictThrowsPartialApplicationNamingTheKeyWithNothingMutated() throws {
    let phantom = "blocks.1.attn.phantom.weight"
    let model = ToyDiT(blocks: blockCount, dim: dim)
    let before = probe(model)
    let lora = fullLoRA(extraKeys: [phantom: pair()])

    XCTAssertThrowsError(
      try LoRAApplicator.applyDynamically(
        to: model, loraWeights: lora, scale: 1.0, strict: true, name: "toy-lora")
    ) { error in
      guard case LoRAError.partialApplication(let name, let unbound) = error else {
        return XCTFail("expected partialApplication, got \(error)")
      }
      XCTAssertEqual(name, "toy-lora")
      XCTAssertEqual(unbound, [phantom])
      XCTAssertTrue("\(error.localizedDescription)".contains(phantom),
                    "the error must name the key: \(error.localizedDescription)")
    }

    XCTAssertEqual(probe(model), before, "strict refusal must leave the base untouched")
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: model))
  }

  /// Under strict, a shape-rejected pair is also a refusal — otherwise a
  /// strict apply could "succeed" with `bound < offered`, which is the
  /// silent partial bind AC-42 forbids.
  func testStrictThrowsOnShapeRejectedPair() throws {
    let model = ToyDiT(blocks: blockCount, dim: dim)
    let before = probe(model)
    var w = fullLoRA().weights
    w["blocks.0.attn.wq.weight"] = pair(inFeatures: dim + 1)
    let lora = LoRAWeights(weights: w, rank: 1)

    XCTAssertThrowsError(
      try LoRAApplicator.applyDynamically(to: model, loraWeights: lora, scale: 1.0, strict: true)
    ) { error in
      guard case LoRAError.incompatibleWeights(let message) = error else {
        return XCTFail("expected incompatibleWeights, got \(error)")
      }
      XCTAssertTrue(message.contains("blocks.0.attn.wq.weight"), message)
    }
    XCTAssertEqual(probe(model), before)
  }

  /// AC-42 — `strict: false` (Z-Image / Flux2 / Chroma) never throws; the
  /// report still tells the truth.
  func testStrictFalseDoesNotThrowOnPartialBind() throws {
    let model = ToyDiT(blocks: blockCount, dim: dim)
    let lora = fullLoRA(extraKeys: ["nowhere.weight": pair()])
    let report = try LoRAApplicator.applyDynamically(
      to: model, loraWeights: lora, scale: 1.0)  // strict defaults to false
    XCTAssertEqual(report.unbound, ["nowhere.weight"])
    XCTAssertTrue(LoRAApplicator.hasDynamicLoRA(in: model), "non-strict still applies what bound")
  }

  /// AC-42 — a four-deep stack either applies all four with
  /// `bound == offered` each, or throws at the truncated one; after the
  /// caller's rollback the base is restored.
  func testFourDeepStackAllBindOrThrowAtTruncatedThird() throws {
    // All four clean.
    do {
      let model = ToyDiT(blocks: blockCount, dim: dim)
      for i in 0..<4 {
        let report = try LoRAApplicator.applyDynamically(
          to: model, loraWeights: fullLoRA(), scale: 0.5, strict: true, name: "lora-\(i)")
        XCTAssertEqual(report.bound, report.offered, "lora-\(i)")
        XCTAssertTrue(report.unbound.isEmpty, "lora-\(i)")
      }
    }

    // Third one truncated (carries a key that binds nothing).
    do {
      let model = ToyDiT(blocks: blockCount, dim: dim)
      let base = probe(model)
      let stack: [LoRAWeights] = [
        fullLoRA(), fullLoRA(),
        fullLoRA(extraKeys: ["blocks.0.attn.ghost.weight": pair()]),
        fullLoRA(),
      ]
      var thrownAt: Int?
      do {
        for (i, lora) in stack.enumerated() {
          try LoRAApplicator.applyDynamically(
            to: model, loraWeights: lora, scale: 1.0, strict: true, name: "lora-\(i)")
        }
      } catch {
        guard case LoRAError.partialApplication(let name, let unbound) = error else {
          return XCTFail("expected partialApplication, got \(error)")
        }
        XCTAssertEqual(name, "lora-2")
        XCTAssertEqual(unbound, ["blocks.0.attn.ghost.weight"])
        thrownAt = 2
        // The pipeline's rollback (Krea2Pipeline.loadLoRAs catch block).
        LoRAApplicator.clearDynamicLoRA(from: model)
      }
      XCTAssertEqual(thrownAt, 2, "the stack must stop at the truncated adapter")
      XCTAssertEqual(probe(model), base, "base restored after rollback")
    }
  }

  /// D9 — the return is `@discardableResult`; the Z-Image/Flux2/Chroma call
  /// sites ignore it and must compile warning-free. (Compile-time check.)
  func testResultIsDiscardable() throws {
    let model = ToyDiT(blocks: blockCount, dim: dim)
    try LoRAApplicator.applyDynamically(to: model, loraWeights: fullLoRA(), scale: 1.0)
    XCTAssertTrue(LoRAApplicator.hasDynamicLoRA(in: model))
  }
}
