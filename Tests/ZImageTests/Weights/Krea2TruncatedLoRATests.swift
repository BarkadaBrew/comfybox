// Krea2TruncatedLoRATests.swift — WP-E7 / Addendum A.2 (the E6 MAJOR
// reassigned to E7): AC-42's "an artificially truncated LoRA throws
// `partialApplication` with the base restored" was satisfied in E6 by a
// SUPERSET dictionary — a hand-built `LoRAWeights` carrying one extra ghost
// key. That proves the applicator's detection site; it does not prove what a
// really-truncated FILE does, which is the failure the acceptance criterion
// is about.
//
// These tests write real `.safetensors` files (Krea-2 / ComfyUI key spelling,
// `diffusion_model.blocks.N.attn.w*.lora_A|lora_B.weight`), damage them the
// three ways a truncated file is actually damaged, and put them through the
// real `LoRAWeightLoader.loadForKrea2` → `LoRAApplicator.applyDynamically`
// path against a toy DiT keyed exactly like `Krea2SingleStreamDiT`.
//
// FINDING (reported, not fixed): a dropped `lora_B` never reaches the
// applicator at all — `loadForKrea2` refuses it as an orphan pair half, so
// the observed error is `LoRAError.invalidFormat`, NOT
// `LoRAError.partialApplication`. That is *stricter* than AC-42 asked for
// (refused before a single tensor is read into the module tree) and it is
// pinned below so a future loosening of the loader cannot silently turn a
// truncated file into a partial bind.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

final class Krea2TruncatedLoRATests: XCTestCase {

  // MARK: - Toy DiT (Krea-2 module paths)

  private final class ToyAttn: Module {
    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    init(dim: Int) {
      func lin(_ seed: Float) -> Linear {
        Linear(weight: (MLXArray.ones([dim, dim]) * seed).asType(.float32), bias: nil)
      }
      self._wq.wrappedValue = lin(0.1)
      self._wk.wrappedValue = lin(0.2)
      self._wv.wrappedValue = lin(0.3)
      self._wo.wrappedValue = lin(0.4)
      super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { wo(wv(wk(wq(x)))) }
  }

  private final class ToyBlock: Module {
    @ModuleInfo(key: "attn") var attn: ToyAttn
    init(dim: Int) { self._attn.wrappedValue = ToyAttn(dim: dim); super.init() }
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
  private let rank = 2
  private let blockCount = 2
  private var scratch: URL!

  override func setUpWithError() throws {
    scratch = FileManager.default.temporaryDirectory
      .appending(path: "krea2-truncated-lora-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: scratch)
  }

  // MARK: - File builders

  /// The ComfyUI spelling every real Krea-2 LoRA on disk uses (verified
  /// against `kroma-v0.2-base-lora-rank-384-fro-0985.safetensors`).
  private func downKey(_ block: Int, _ proj: String) -> String {
    "diffusion_model.blocks.\(block).attn.\(proj).lora_A.weight"
  }
  private func upKey(_ block: Int, _ proj: String) -> String {
    "diffusion_model.blocks.\(block).attn.\(proj).lora_B.weight"
  }

  /// A complete, well-formed Krea-2 LoRA over `blocks` × {wq,wk,wv,wo}.
  private func completeArrays(blocks: Int) -> [String: MLXArray] {
    var arrays: [String: MLXArray] = [:]
    for b in 0..<blocks {
      for proj in ["wq", "wk", "wv", "wo"] {
        arrays[downKey(b, proj)] = (MLXArray.ones([rank, dim]) * Float(0.5)).asType(.float32)
        arrays[upKey(b, proj)] = (MLXArray.ones([dim, rank]) * Float(0.01)).asType(.float32)
      }
    }
    return arrays
  }

  private func write(_ arrays: [String: MLXArray], as name: String) throws -> URL {
    let url = scratch.appending(path: name)
    try MLX.save(arrays: arrays, url: url)
    return url
  }

  private func probe(_ model: ToyDiT) -> [Float] {
    let x = MLXArray([Float](repeating: 1.0, count: dim), [1, dim])
    let y = model(x)
    eval(y)
    return y.asArray(Float.self)
  }

  // MARK: - Baseline: the undamaged file binds completely

  func testCompleteFileBindsEveryOfferedPairUnderStrict() throws {
    let url = try write(completeArrays(blocks: blockCount), as: "complete.safetensors")
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(weights.weights.count, blockCount * 4)
    XCTAssertTrue(weights.deltas.isEmpty)
    // The loader remaps `diffusion_model.` off and lands on the DiT's paths.
    XCTAssertNotNil(weights.weights["blocks.0.attn.wq.weight"])

    let model = ToyDiT(blocks: blockCount, dim: dim)
    let report = try LoRAApplicator.applyDynamically(
      to: model, loraWeights: weights, scale: 1.0, strict: true, name: "complete")
    XCTAssertEqual(report.offered, blockCount * 4)
    XCTAssertEqual(report.bound, report.offered)
    XCTAssertEqual(report.shapeRejected, 0)
    XCTAssertTrue(report.unbound.isEmpty)
    XCTAssertTrue(report.isComplete)
  }

  // MARK: - Truncation 1: a dropped `lora_B` (Addendum A.2, literally)

  /// Drop one `lora_B` of a bound pair from a real file. The refusal is
  /// REAL and comes EARLIER than AC-42 predicted: `loadForKrea2` throws
  /// `invalidFormat` naming the orphaned half, so the applicator is never
  /// reached and nothing can be mutated. Pinned so the loader can never be
  /// relaxed into offering a half-pair.
  func testDroppingOneLoraBIsRefusedByTheLoaderBeforeAnyApply() throws {
    var arrays = completeArrays(blocks: blockCount)
    let dropped = upKey(1, "wv")
    arrays.removeValue(forKey: dropped)
    let url = try write(arrays, as: "truncated-missing-up.safetensors")

    let model = ToyDiT(blocks: blockCount, dim: dim)
    let before = probe(model)

    XCTAssertThrowsError(try LoRAWeightLoader.loadForKrea2(from: url)) { error in
      guard case LoRAError.invalidFormat(let message) = error else {
        return XCTFail(
          "a file missing one lora_B must be refused at load; got \(error). "
            + "If this ever becomes partialApplication, the loader started offering half-pairs.")
      }
      XCTAssertTrue(message.contains("orphan"), message)
      // The message names the SURVIVING half, which is what is on disk.
      XCTAssertTrue(message.contains(downKey(1, "wv")), message)
    }

    XCTAssertEqual(probe(model), before, "a truncated file must leave the base untouched")
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: model))
  }

  /// The same truncation through the same guard the pipeline uses first:
  /// `validate(at:)` must not call the file usable.
  func testValidateRejectsTheTruncatedFile() throws {
    var arrays = completeArrays(blocks: blockCount)
    arrays.removeValue(forKey: upKey(0, "wq"))
    let url = try write(arrays, as: "truncated-validate.safetensors")
    // The generic loader is the one `validate` runs; it must refuse too.
    XCTAssertThrowsError(try LoRAWeightLoader.load(from: url)) { error in
      guard case LoRAError.invalidFormat(let message) = error else {
        return XCTFail("expected invalidFormat, got \(error)")
      }
      XCTAssertTrue(message.contains("orphan"), message)
    }
  }

  // MARK: - Truncation 2: a truncated `lora_B` TENSOR

  /// A file cut short mid-write can also carry a pair whose up half has
  /// fewer rows than the target Linear. That one DOES reach the applicator:
  /// it is `shapeRejected` (never silently bound, never counted as
  /// `unbound`), and strict refuses with `incompatibleWeights` naming the key
  /// while the base stays byte-identical.
  func testTruncatedUpTensorIsShapeRejectedAndStrictRefusesWithNothingMutated() throws {
    var arrays = completeArrays(blocks: blockCount)
    // out-features 3 can never bind a 4×4 Linear in any orientation.
    arrays[upKey(1, "wo")] = (MLXArray.ones([dim - 1, rank]) * Float(0.01)).asType(.float32)
    let url = try write(arrays, as: "truncated-up-tensor.safetensors")
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(weights.weights.count, blockCount * 4, "the pair still loads — the damage is the shape")

    // Non-strict: counted, not hidden, and not confused with `unbound`.
    let loose = ToyDiT(blocks: blockCount, dim: dim)
    let report = try LoRAApplicator.applyDynamically(
      to: loose, loraWeights: weights, scale: 1.0, strict: false, name: "truncated-tensor")
    XCTAssertEqual(report.offered, blockCount * 4)
    XCTAssertEqual(report.bound, blockCount * 4 - 1)
    XCTAssertEqual(report.shapeRejected, 1)
    XCTAssertTrue(report.unbound.isEmpty, "a shape-rejected key is not an unbound key")
    XCTAssertFalse(report.isComplete)

    // Strict (Krea-2): refuse, naming the key, with the base untouched.
    let strictModel = ToyDiT(blocks: blockCount, dim: dim)
    let before = probe(strictModel)
    XCTAssertThrowsError(
      try LoRAApplicator.applyDynamically(
        to: strictModel, loraWeights: weights, scale: 1.0, strict: true, name: "truncated-tensor")
    ) { error in
      guard case LoRAError.incompatibleWeights(let message) = error else {
        return XCTFail("expected incompatibleWeights, got \(error)")
      }
      XCTAssertTrue(message.contains("blocks.1.attn.wo.weight"), message)
    }
    XCTAssertEqual(probe(strictModel), before)
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: strictModel))
  }

  // MARK: - Truncation 3: a pair the model has no module for (AC-42's
  // `partialApplication`, driven from a REAL file rather than a dictionary)

  /// The one damage mode that produces `partialApplication`: an offered key
  /// that reaches the applicator and matches no module. Written to disk and
  /// loaded through `loadForKrea2` so the whole path — key remap included —
  /// is the production one.
  func testOfferedKeyWithNoModuleIsUnboundAndStrictThrowsPartialApplication() throws {
    var arrays = completeArrays(blocks: blockCount)
    // A block the (2-block) DiT does not have — the shape a stale or
    // mis-targeted extraction really takes.
    arrays[downKey(9, "wq")] = (MLXArray.ones([rank, dim]) * Float(0.5)).asType(.float32)
    arrays[upKey(9, "wq")] = (MLXArray.ones([dim, rank]) * Float(0.01)).asType(.float32)
    let url = try write(arrays, as: "extra-target.safetensors")
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    let orphanTarget = "blocks.9.attn.wq.weight"
    XCTAssertNotNil(weights.weights[orphanTarget])

    // Non-strict (Z-Image / Flux2 posture): reported in `unbound`, no throw.
    let loose = ToyDiT(blocks: blockCount, dim: dim)
    let report = try LoRAApplicator.applyDynamically(
      to: loose, loraWeights: weights, scale: 1.0, strict: false, name: "extra-target")
    XCTAssertEqual(report.offered, blockCount * 4 + 1)
    XCTAssertEqual(report.bound, blockCount * 4)
    XCTAssertEqual(report.unbound, [orphanTarget])
    XCTAssertEqual(report.shapeRejected, 0)

    // Strict (Krea-2): throw, naming the key, base untouched.
    let strictModel = ToyDiT(blocks: blockCount, dim: dim)
    let before = probe(strictModel)
    XCTAssertThrowsError(
      try LoRAApplicator.applyDynamically(
        to: strictModel, loraWeights: weights, scale: 1.0, strict: true, name: "extra-target")
    ) { error in
      guard case LoRAError.partialApplication(let name, let unbound) = error else {
        return XCTFail("expected partialApplication, got \(error)")
      }
      XCTAssertEqual(name, "extra-target")
      XCTAssertEqual(unbound, [orphanTarget])
      XCTAssertTrue(error.localizedDescription.contains(orphanTarget), error.localizedDescription)
    }
    XCTAssertEqual(probe(strictModel), before)
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: strictModel))
  }

  // MARK: - Orphan LoKr half (K-FIX-1 / I3 — the E7 tripwire, now flipped)

  /// Was the E7 report's KNOWN GAP: a truncated **LyCORIS LoKr** file was the
  /// one damage mode neither refused nor reported — `loadForKrea2` paired the
  /// halves with `guard let w2 = lokrW2[key] else { continue }`, so a
  /// `lokr_w1` whose `lokr_w2` was dropped was consumed (hence not an
  /// `unknownKeys` refusal) and then silently discarded, and LoKr modules are
  /// outside `LoRAApplicationReport.offered` entirely, so strict apply could
  /// not see it either.
  ///
  /// Codex engine review I3 / ledger AC-42: an orphan LoKr half is now
  /// refused as `invalidFormat` naming the key — the same posture the plain
  /// `lora_A`/`lora_B` pair has had all along (see
  /// `testDroppedLoRABIsRefusedAsAnOrphanPairHalf`). A complete file still
  /// loads both entries.
  func testOrphanLoKrHalfIsRefusedAsInvalidFormat() throws {
    var arrays: [String: MLXArray] = [:]
    for proj in ["wq", "wk"] {
      arrays["diffusion_model.blocks.0.attn.\(proj).lokr_w1"] =
        (MLXArray.ones([dim, rank]) * Float(0.1)).asType(.float32)
      arrays["diffusion_model.blocks.0.attn.\(proj).lokr_w2"] =
        (MLXArray.ones([rank, dim]) * Float(0.1)).asType(.float32)
    }
    let complete = try write(arrays, as: "lokr-complete.safetensors")
    XCTAssertEqual(try LoRAWeightLoader.loadForKrea2(from: complete).lokrWeights.count, 2)

    // w2 dropped — the orphan is `lokr_w1`.
    var missingW2 = arrays
    missingW2.removeValue(forKey: "diffusion_model.blocks.0.attn.wk.lokr_w2")
    let truncatedW2 = try write(missingW2, as: "lokr-truncated-w2.safetensors")
    XCTAssertThrowsError(try LoRAWeightLoader.loadForKrea2(from: truncatedW2)) { error in
      guard case LoRAError.invalidFormat(let message) = error else {
        return XCTFail("expected invalidFormat, got \(error)")
      }
      XCTAssertTrue(message.contains("lokr_w2"), message)
      XCTAssertTrue(message.contains("blocks.0.attn.wk"), message)
    }

    // …and the mirror case: w1 dropped, the orphan is `lokr_w2`. Without this
    // half of the guard a file that lost its w1 would still load a half-LoKr
    // dictionary entry-free and report a clean apply.
    var missingW1 = arrays
    missingW1.removeValue(forKey: "diffusion_model.blocks.0.attn.wk.lokr_w1")
    let truncatedW1 = try write(missingW1, as: "lokr-truncated-w1.safetensors")
    XCTAssertThrowsError(try LoRAWeightLoader.loadForKrea2(from: truncatedW1)) { error in
      guard case LoRAError.invalidFormat(let message) = error else {
        return XCTFail("expected invalidFormat, got \(error)")
      }
      XCTAssertTrue(message.contains("lokr_w1"), message)
      XCTAssertTrue(message.contains("blocks.0.attn.wk"), message)
    }
  }

  /// An `.alpha` beside a LoKr pair is metadata, not a half: a module that
  /// carries alpha but no w1/w2 at all must NOT be reported as an orphan
  /// (that would refuse files the loader has always accepted).
  func testAlphaWithoutLoKrHalvesIsNotAnOrphan() throws {
    var arrays: [String: MLXArray] = [:]
    arrays["diffusion_model.blocks.0.attn.wq.lokr_w1"] =
      (MLXArray.ones([dim, rank]) * Float(0.1)).asType(.float32)
    arrays["diffusion_model.blocks.0.attn.wq.lokr_w2"] =
      (MLXArray.ones([rank, dim]) * Float(0.1)).asType(.float32)
    arrays["diffusion_model.blocks.0.attn.wq.alpha"] = MLXArray(Float(4.0))
    let url = try write(arrays, as: "lokr-with-alpha.safetensors")
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(weights.lokrWeights.count, 1)
    XCTAssertEqual(weights.lokrWeights.values.first?.alpha, 4.0)
  }
}
