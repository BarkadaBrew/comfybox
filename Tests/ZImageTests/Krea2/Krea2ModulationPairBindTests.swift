// Krea2ModulationPairBindTests.swift — the turbo distills' ONE bare-parameter
// pair, and the fail-closed guarantees that must survive accepting it.
//
// `krea2_turbo_distill_r256.safetensors` offers 531 keys. 530 name Linear
// modules and bound fine; the 531st names `last.modulation.lin`, which is NOT
// a Linear — `Krea2SimpleModulation.lin` is a bare (2, features) MLXArray
// ADDED to the timestep vector, stored in the checkpoint as a plain tensor.
// `LoRAApplicator.applyDynamically` binds by walking `namedModules()` and
// wrapping Linears, so it could never reach that key, and the strict Krea-2
// apply refused the WHOLE file over it:
//
//   LoRA 'krea2_turbo_distill_r256.safetensors' did not bind completely
//   — 1 key(s) matched nothing: last.modulation.lin.weight
//
// Kroma v0.3 patches the SAME parameter — `diffusion_model.last.modulation
// .lin.diff`, shape [2, 6144] — and has always worked, because bare-parameter
// patches go through `LoRAPatchSession`, which indexes PARAMETERS and owns an
// exact restore. A rank-2 pair on that parameter is arithmetically the same
// object (`up @ down`), so it takes the same route.
//
// What these tests pin, in order: the divert happens and is exact; the value
// follows the alpha/rank convention and userScale; clear() restores byte-for-
// byte; a Linear target is NEVER diverted (the no-regression property); and a
// key naming something the architecture does not have STILL refuses loudly.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

final class Krea2ModulationPairBindTests: XCTestCase {

  // MARK: - Fixtures

  /// The real last layer, at toy width. Same types, same `@ModuleInfo` keys,
  /// same parameter paths as production — `norm.scale` [8], `linear.weight`
  /// [16, 8], `linear.bias` [16], `modulation.lin` [2, 8].
  private func lastLayer() -> Krea2LastLayer {
    Krea2LastLayer(features: 8, patch: 2, channels: 4)
  }

  private func ramp(_ shape: [Int], _ step: Float = 0.01) -> MLXArray {
    let n = shape.reduce(1, *)
    return MLXArray((0..<n).map { Float($0) * step }).reshaped(shape)
  }

  private func floats(_ a: MLXArray) -> [Float] {
    let flat = a.asType(.float32).flattened()
    eval(flat)
    return flat.asArray(Float.self)
  }

  private func param(_ module: Module, _ path: String) throws -> MLXArray {
    let index = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
    return try XCTUnwrap(index[path], "no parameter '\(path)'")
  }

  /// down [2, 8] (rank 2, in 8) / up [2, 2] (out 2, rank 2) — the r256's own
  /// orientation for `last.modulation.lin`, scaled down to the toy width.
  private func modulationPair() -> (down: MLXArray, up: MLXArray) {
    (down: ramp([2, 8], 0.01), up: ramp([2, 2], 0.1))
  }

  /// down [4, 8] / up [16, 4] — an ordinary Linear target.
  private func linearPair() -> (down: MLXArray, up: MLXArray) {
    (down: ramp([4, 8], 0.003), up: ramp([16, 4], 0.002))
  }

  private func expectedDelta(_ pair: (down: MLXArray, up: MLXArray)) -> [Float] {
    floats(MLX.matmul(pair.up.asType(.float32), pair.down.asType(.float32)))
  }

  // MARK: - The divert

  func testBareParameterPairMovesToDeltas() throws {
    let layer = lastLayer()
    let mod = modulationPair()
    let lin = linearPair()
    let loaded = LoRAWeights(
      weights: ["modulation.lin.weight": mod, "linear.weight": lin], rank: 2)

    let split = try LoRABareParameterPairs.split(loaded, for: layer)

    XCTAssertEqual(Set(split.weights.keys), ["linear.weight"],
                   "the Linear pair stays a runtime adapter")
    XCTAssertEqual(Set(split.deltas.keys), ["modulation.lin"],
                   "the bare-parameter pair becomes a delta on the REAL parameter path")
    let delta = try XCTUnwrap(split.deltas["modulation.lin"])
    XCTAssertEqual(delta.tensor.shape, [2, 8])
    XCTAssertEqual(floats(delta.tensor), expectedDelta(mod), "delta == up @ down")
    if case .diff = delta {} else { XCTFail("must be an additive .diff, never .set_weight") }
  }

  /// The whole point: with the divert in place the strict apply — the thing
  /// that refused the r256 — now succeeds, and the modulation contribution is
  /// really in the parameter afterwards.
  func testStrictApplyBindsEverythingAndPatchLands() throws {
    let layer = lastLayer()
    let mod = modulationPair()
    let before = floats(try param(layer, "modulation.lin"))

    let split = try LoRABareParameterPairs.split(
      LoRAWeights(weights: ["modulation.lin.weight": mod, "linear.weight": linearPair()],
                  rank: 2),
      for: layer)

    let report = try LoRAApplicator.applyDynamically(
      to: layer, loraWeights: split, scale: 0.6, strict: true, name: "toy-distill")
    XCTAssertEqual(report.offered, 1)
    XCTAssertEqual(report.bound, 1)
    XCTAssertTrue(report.unbound.isEmpty)

    let session = LoRAPatchSession(module: layer)
    XCTAssertEqual(try session.apply(weights: split, scale: 0.6), 1,
                   "the modulation delta is APPLIED, not skipped")

    let after = floats(try param(layer, "modulation.lin"))
    let delta = expectedDelta(mod)
    for i in 0..<after.count {
      XCTAssertEqual(after[i], before[i] + 0.6 * delta[i], accuracy: 1e-5,
                     "element \(i): base + userScale × (up @ down)")
    }
    XCTAssertNotEqual(after, before, "the layer's contribution is not a no-op")
  }

  /// Transactional, like every other Krea-2 adapter: rollback restores the
  /// parameter exactly, so a later failure in the stack cannot leave the
  /// modulation half-patched.
  func testClearRestoresTheParameterExactly() throws {
    let layer = lastLayer()
    let before = floats(try param(layer, "modulation.lin"))
    let split = try LoRABareParameterPairs.split(
      LoRAWeights(weights: ["modulation.lin.weight": modulationPair()], rank: 2), for: layer)

    let session = LoRAPatchSession(module: layer)
    XCTAssertEqual(try session.apply(weights: split, scale: 0.6), 1)
    XCTAssertNotEqual(floats(try param(layer, "modulation.lin")), before)

    session.clear()
    XCTAssertEqual(floats(try param(layer, "modulation.lin")), before,
                   "exact restore, element for element")
  }

  /// Alpha/rank parity: a diverted pair is scaled exactly as the applicator
  /// would have scaled it on a Linear — `userScale × alpha/rank × (up @ down)`.
  func testDeltaFollowsAlphaOverRankConvention() throws {
    let layer = lastLayer()
    let mod = modulationPair()
    // rank of this layer's pair is min(2, 8) == 2; alpha 4 → 2.0.
    let split = try LoRABareParameterPairs.split(
      LoRAWeights(weights: ["modulation.lin.weight": mod], rank: 2, alpha: 4), for: layer)

    let baked = floats(try XCTUnwrap(split.deltas["modulation.lin"]).tensor)
    let raw = expectedDelta(mod)
    for i in 0..<baked.count {
      XCTAssertEqual(baked[i], 2.0 * raw[i], accuracy: 1e-5)
    }
  }

  // MARK: - No regression

  /// A pair whose target IS a Linear is never diverted — the module walk wins
  /// unconditionally, which is what makes this change inert for every adapter
  /// that already loads (`krea2_turbo_lora_rank_64_bf16` and friends).
  func testLinearTargetsAreNeverDiverted() throws {
    let layer = lastLayer()
    let loaded = LoRAWeights(weights: ["linear.weight": linearPair()], rank: 4)
    let split = try LoRABareParameterPairs.split(loaded, for: layer)

    XCTAssertEqual(Set(split.weights.keys), ["linear.weight"])
    XCTAssertTrue(split.deltas.isEmpty, "a Linear's .weight is a parameter too — and stays a pair")
  }

  /// `linear.bias` is a bare parameter of the Linear. A pair keyed at the
  /// MODULE (`linear.weight`) must not be re-pointed at it, and the Linear
  /// module path must keep winning even though `linear.weight` also appears
  /// in the parameter index.
  func testSplitIsIdentityWhenNothingMoves() throws {
    let layer = lastLayer()
    let loaded = LoRAWeights(weights: ["linear.weight": linearPair()], rank: 4, alpha: 8)
    let split = try LoRABareParameterPairs.split(loaded, for: layer)
    XCTAssertEqual(split.effectiveScale(forLayer: "linear.weight"),
                   loaded.effectiveScale(forLayer: "linear.weight"),
                   "an untouched split preserves the alpha metadata exactly")
    XCTAssertEqual(split.rank, loaded.rank)
  }

  // MARK: - Still fail-closed

  /// The guarantee that must NOT be weakened: a key naming a module the
  /// architecture does not have is not diverted, not dropped, and still
  /// refuses under strict.
  func testUnknownTargetStillRefusesLoudly() throws {
    let layer = lastLayer()
    let split = try LoRABareParameterPairs.split(
      LoRAWeights(weights: ["blocks.99.attn.wq.weight": linearPair()], rank: 4), for: layer)
    XCTAssertEqual(Set(split.weights.keys), ["blocks.99.attn.wq.weight"],
                   "left where the strict apply will see it")
    XCTAssertTrue(split.deltas.isEmpty)

    XCTAssertThrowsError(try LoRAApplicator.applyDynamically(
      to: layer, loraWeights: split, scale: 1.0, strict: true, name: "bogus")) { error in
      guard case LoRAError.partialApplication(_, let unbound) = error else {
        return XCTFail("expected partialApplication, got \(error)")
      }
      XCTAssertEqual(unbound, ["blocks.99.attn.wq.weight"])
    }
  }

  /// A 1-D bare parameter (`Krea2DoubleSharedModulation.lin`, the per-block
  /// `mod.lin` Kroma patches as a [36864] `.diff`) has no low-rank structure
  /// to reconstruct. Refuse — never divert a delta that would not fit.
  func testOneDimensionalBareParameterRefuses() throws {
    let mod = Krea2DoubleSharedModulation(8)  // lin: [48]
    let loaded = LoRAWeights(weights: ["lin.weight": modulationPair()], rank: 2)
    XCTAssertThrowsError(try LoRABareParameterPairs.split(loaded, for: mod)) { error in
      guard case LoRAError.incompatibleWeights(let message) = error else {
        return XCTFail("expected incompatibleWeights, got \(error)")
      }
      XCTAssertTrue(message.contains("lin"), message)
    }
  }

  /// Right target, wrong shape: refuse rather than leave it to be reported as
  /// "matched nothing", and never coerce it.
  func testShapeThatCannotFitTheParameterRefuses() throws {
    let layer = lastLayer()
    let bad = (down: ramp([2, 7], 0.01), up: ramp([2, 2], 0.1))  // in=7, target in=8
    XCTAssertThrowsError(try LoRABareParameterPairs.split(
      LoRAWeights(weights: ["modulation.lin.weight": bad], rank: 2), for: layer)) { error in
      guard case LoRAError.incompatibleWeights(let message) = error else {
        return XCTFail("expected incompatibleWeights, got \(error)")
      }
      XCTAssertTrue(message.contains("modulation.lin"), message)
    }
  }

  /// A file carrying BOTH a bare patch and a pair on the same parameter has no
  /// defined order. Refuse instead of picking one.
  func testPairAndBarePatchOnTheSameTargetRefuses() throws {
    let layer = lastLayer()
    let loaded = LoRAWeights(
      weights: ["modulation.lin.weight": modulationPair()],
      rank: 2,
      deltas: ["modulation.lin": .diff(ramp([2, 8], 0.5))])
    XCTAssertThrowsError(try LoRABareParameterPairs.split(loaded, for: layer)) { error in
      guard case LoRAError.incompatibleWeights = error else {
        return XCTFail("expected incompatibleWeights, got \(error)")
      }
    }
  }

  // MARK: - The real artifact

  private static let vault = URL(
    fileURLWithPath: NSString(string: "~/comfybox-models/loras/vault").expandingTildeInPath,
    isDirectory: true)

  /// Header-only (no 1.8 GB read): the r256 and r128 distills each carry
  /// exactly ONE `modulation` pair, at the shape the production last layer's
  /// bare parameter has — so the divert this file adds is the only mechanism
  /// those two keys need.
  func testDistillArtifactsCarryOneModulationPairAtParameterShape() throws {
    let production = Krea2LastLayer(features: 6144, patch: 2, channels: 16)
    let modulation = try param(production, "modulation.lin")
    XCTAssertEqual(modulation.shape, [2, 6144])
    XCTAssertNil(LoRAApplicator.linearDims(for: production.modulation),
                 "Krea2SimpleModulation is not a Linear — no module walk can bind it")

    for name in ["krea2_turbo_distill_r256.safetensors", "krea2_turbo_distill_r128.safetensors"] {
      let url = Self.vault.appending(path: name)
      try XCTSkipUnless(FileManager.default.fileExists(atPath: url.path),
                        "distill absent: \(url.path)")
      let reader = try LoRAWeightLoader.openSafetensors(url)
      let names = reader.tensorNames.filter { $0.contains("modulation") }
      XCTAssertEqual(Set(names), [
        "diffusion_model.last.modulation.lin.lora_A.weight",
        "diffusion_model.last.modulation.lin.lora_B.weight",
      ], "\(name)")
      XCTAssertEqual(try reader.tensor(named: names.first { $0.hasSuffix("lora_A.weight") }!).shape,
                     [2, 6144], "\(name): lora_A is [rank=2, in=6144]")
      XCTAssertEqual(try reader.tensor(named: names.first { $0.hasSuffix("lora_B.weight") }!).shape,
                     [2, 2], "\(name): lora_B is [out=2, rank=2]")
    }
  }
}
