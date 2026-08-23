// Krea2LoKrRefusalTests.swift — K-FIX-1 / Codex engine review C1.
//
// LoKr application is NOT transactional: `LoRAApplicator.applyLoKr` replaces
// base `Linear` weights outright and DEQUANTIZES→requantizes `QuantizedLinear`
// ones, while `clearDynamicLoRA` only empties `LoRALinear`/`LoRAQuantizedLinear`
// adapters. So a LoKr adapter applied on the Krea 2 path could never be
// cleared: `loadLoRAs([])` reported an empty stack over a still-mutated
// checkpoint, a re-apply compounded it, and every control on/off toggle
// compounded it again. On q8 the first application also destroys the original
// packed bytes, so even an exact subtraction cannot restore them.
//
// Controller ruling (ledger, "Codex engine review" section): the Krea 2 path
// REFUSES LoKr adapters, fail-loud, BEFORE any weight is touched, until LoKr
// is made transactional under its own ticket. The Z-Image path is unchanged.
//
// These tests pin the guard itself (pure, no checkpoint), the refusal on a
// REAL LoKr `.safetensors` file walked through `loadForKrea2`, that a
// refused file leaves a model bit-identical, and the 400 mapping.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

final class Krea2LoKrRefusalTests: XCTestCase {

  private let dim = 8
  private let rank = 4

  private var tmp: URL!

  override func setUpWithError() throws {
    tmp = URL(fileURLWithPath: NSTemporaryDirectory())
      .appendingPathComponent("krea2-lokr-refusal-\(UUID().uuidString)")
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

  // MARK: - The guard, pure

  func testGuardPassesWhenTheFileCarriesNoLoKr() throws {
    XCTAssertNoThrow(try Krea2AdapterSupport.checkTransactional(lokrLayerCount: 0, lora: "plain.safetensors"))
  }

  func testGuardRefusesAnyLoKrLayerNamingTheFileAndTheReason() throws {
    var caught: Error?
    do {
      try Krea2AdapterSupport.checkTransactional(lokrLayerCount: 3, lora: "realism-v2.safetensors")
    } catch {
      caught = error
    }
    let error = try XCTUnwrap(caught)
    guard case LoRAError.unsupportedAdapter(let lora, let format, let reason) = error else {
      return XCTFail("expected unsupportedAdapter, got \(error)")
    }
    XCTAssertEqual(lora, "realism-v2.safetensors")
    XCTAssertEqual(format, "LoKr")
    XCTAssertTrue(reason.contains("not transactional"), reason)
    // The message a caller reads must name BOTH: which file, and why.
    let described = error.localizedDescription
    XCTAssertTrue(described.contains("realism-v2.safetensors"), described)
    XCTAssertTrue(described.contains("LoKr"), described)
    XCTAssertTrue(described.contains("not transactional"), described)
  }

  /// The refusal is the caller's error, not a server fault (AC-15 posture).
  func testRefusalIsAn400() throws {
    let error = LoRAError.unsupportedAdapter(
      lora: "realism-v2.safetensors", format: "LoKr", reason: "LoKr is not transactional on this path")
    let response = WarmServer.errorResponse(for: error)
    XCTAssertEqual(response.status, 400)
    let body = String(decoding: response.body, as: UTF8.self)
    XCTAssertTrue(body.contains("realism-v2.safetensors"), body)
    XCTAssertTrue(body.contains("LoKr"), body)
  }

  // MARK: - A real LoKr file, through the real loader

  /// End-to-end over the path `Krea2Pipeline.loadLoRAs` walks: resolve →
  /// `loadForKrea2` → guard. The guard fires on the loaded weights, so no
  /// LoKr file can reach `applyDynamically`/`applyLoKr` on this path.
  func testRealLoKrFileIsRefusedAfterLoadAndBeforeAnyApply() throws {
    var arrays: [String: MLXArray] = [:]
    for proj in ["wq", "wk"] {
      arrays["diffusion_model.blocks.0.attn.\(proj).lokr_w1"] =
        (MLXArray.ones([dim, rank]) * Float(0.1)).asType(.float32)
      arrays["diffusion_model.blocks.0.attn.\(proj).lokr_w2"] =
        (MLXArray.ones([rank, dim]) * Float(0.1)).asType(.float32)
    }
    let url = try write(arrays, as: "pure-lokr.safetensors")

    // The loader itself still loads it — the refusal is a Krea 2 POLICY, not
    // a parse failure, and the Z-Image path keeps loading the same file.
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(weights.lokrLayerCount, 2)
    XCTAssertTrue(weights.hasLoKr)

    XCTAssertThrowsError(
      try Krea2AdapterSupport.checkTransactional(
        lokrLayerCount: weights.lokrLayerCount, lora: url.lastPathComponent)
    ) { error in
      guard case LoRAError.unsupportedAdapter(let lora, _, _) = error else {
        return XCTFail("expected unsupportedAdapter, got \(error)")
      }
      XCTAssertEqual(lora, "pure-lokr.safetensors")
    }
  }

  /// A MIXED file (ordinary pairs + LoKr) is refused whole. Refusing only the
  /// LoKr half would apply a fraction of the adapter and report it complete —
  /// the same silent-partial the strict preflight exists to prevent.
  func testMixedLoKrAndPlainPairsIsRefusedWhole() throws {
    var arrays: [String: MLXArray] = [:]
    arrays["diffusion_model.blocks.0.attn.wq.lora_A.weight"] =
      (MLXArray.ones([rank, dim]) * Float(0.1)).asType(.float32)
    arrays["diffusion_model.blocks.0.attn.wq.lora_B.weight"] =
      (MLXArray.ones([dim, rank]) * Float(0.1)).asType(.float32)
    arrays["diffusion_model.blocks.0.attn.wk.lokr_w1"] =
      (MLXArray.ones([dim, rank]) * Float(0.1)).asType(.float32)
    arrays["diffusion_model.blocks.0.attn.wk.lokr_w2"] =
      (MLXArray.ones([rank, dim]) * Float(0.1)).asType(.float32)
    let url = try write(arrays, as: "mixed.safetensors")

    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(weights.weights.count, 1, "the plain pair loaded")
    XCTAssertEqual(weights.lokrLayerCount, 1, "and so did the LoKr layer")
    XCTAssertThrowsError(
      try Krea2AdapterSupport.checkTransactional(
        lokrLayerCount: weights.lokrLayerCount, lora: url.lastPathComponent))
  }

  // MARK: - Nothing is mutated by a refused file

  /// The reason the guard is a REFUSAL and not a rollback: there is nothing
  /// to roll back to. Asserted as a parameter fingerprint over a toy model
  /// keyed like `Krea2SingleStreamDiT` — the guard inspects counts only.
  func testARefusedFileLeavesTheModelBitIdentical() throws {
    final class ToyAttn: Module {
      @ModuleInfo(key: "wq") var wq: Linear
      @ModuleInfo(key: "wk") var wk: Linear
      init(dim: Int) {
        self._wq.wrappedValue = Linear(dim, dim, bias: false)
        self._wk.wrappedValue = Linear(dim, dim, bias: false)
      }
    }
    final class ToyBlock: Module {
      @ModuleInfo(key: "attn") var attn: ToyAttn
      init(dim: Int) { self._attn.wrappedValue = ToyAttn(dim: dim) }
    }
    final class ToyDiT: Module {
      @ModuleInfo(key: "blocks") var blocks: [ToyBlock]
      init(dim: Int) { self._blocks.wrappedValue = [ToyBlock(dim: dim)] }
    }

    func fingerprint(_ model: ToyDiT) -> [Float] {
      var out: [Float] = []
      for (_, module) in model.namedModules() {
        guard let lin = module as? Linear else { continue }
        out += lin.weight.asType(.float32).flattened().asArray(Float.self)
      }
      return out
    }

    let model = ToyDiT(dim: dim)
    let before = fingerprint(model)

    // A REAL, well-formed LoKr: kron(w1, w2) must have the target's exact
    // shape (2x2 ⊗ 4x4 = 8x8), or `applyLoKr` skips the layer and the
    // counter-proof below would prove nothing.
    var arrays: [String: MLXArray] = [:]
    arrays["diffusion_model.blocks.0.attn.wq.lokr_w1"] =
      (MLXArray.ones([2, 2]) * Float(0.5)).asType(.float32)
    arrays["diffusion_model.blocks.0.attn.wq.lokr_w2"] =
      (MLXArray.ones([dim / 2, dim / 2]) * Float(0.5)).asType(.float32)
    let url = try write(arrays, as: "refused.safetensors")
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)

    XCTAssertThrowsError(
      try Krea2AdapterSupport.checkTransactional(
        lokrLayerCount: weights.lokrLayerCount, lora: url.lastPathComponent))
    XCTAssertEqual(fingerprint(model), before)
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: model))

    // …and the counter-proof that the refusal is load-bearing: applying the
    // same file the OLD way mutates the base, and `clearDynamicLoRA` — the
    // "rollback" `loadLoRAs` relies on — does not put it back.
    LoRAApplicator.applyLoKr(to: model, loraWeights: weights, scale: 1.0)
    XCTAssertNotEqual(fingerprint(model), before, "applyLoKr mutates the base weights")
    LoRAApplicator.clearDynamicLoRA(from: model)
    XCTAssertNotEqual(
      fingerprint(model), before,
      "clearDynamicLoRA cannot restore LoKr-mutated base weights — this is C1")
  }
}
