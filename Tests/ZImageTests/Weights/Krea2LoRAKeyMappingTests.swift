import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

/// WP-E6 (FDD §3.6, AC-39): the Comfy-Org rank-64 turbo LoRA binds the Raw
/// DiT completely. Fixture-gated on the vault file (the
/// `RealVAEExactTests.swift` pattern); header + tensor reads only, no
/// checkpoint. The "applied to Raw" half runs against a lazily-initialised
/// `Krea2SingleStreamDiT` — module paths and Linear shapes are metadata, so the
/// bind walk is exact without materialising 26 GB of weights.
final class Krea2LoRAKeyMappingTests: XCTestCase {

  private static let turboLoRA = URL(fileURLWithPath: NSString(
    string: "~/comfybox-models/loras/vault/krea2_turbo_lora_rank_64_bf16.safetensors").expandingTildeInPath)

  private func requireTurboLoRA() throws -> URL {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.turboLoRA.path),
                      "krea2_turbo_lora_rank_64_bf16.safetensors not in the vault on this machine")
    return Self.turboLoRA
  }

  /// AC-39 — `loadForKrea2(krea2_turbo_lora_rank_64_bf16)` → 264 pairs,
  /// 7 deltas, rank 64, no throw.
  func testTurboLoRALoads264PairsAnd7Deltas() throws {
    let url = try requireTurboLoRA()
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)

    XCTAssertEqual(weights.weights.count, 264)
    XCTAssertEqual(weights.deltas.count, 7)
    XCTAssertEqual(weights.rank, 64)
    XCTAssertEqual(weights.effectiveScale(forLayer: "blocks.0.attn.wq.weight"), 1.0,
                   "no alpha ⇒ the applied scale passes through verbatim")

    // The seven `.diff_b` deltas land on the REAL bias paths after the
    // numeric-index remap (tmlp.0 → tmlp.lin0, txtmlp.1 → txtmlp.lin1, …).
    let deltaKeys = weights.deltas.keys.sorted()
    XCTAssertEqual(deltaKeys, [
      "first.bias", "last.linear.bias", "tmlp.lin0.bias", "tmlp.lin2.bias",
      "tproj.lin1.bias", "txtmlp.lin1.bias", "txtmlp.lin3.bias",
    ], "\(deltaKeys)")
  }

  /// AC-39 — applied to the (Raw-shaped) DiT under `strict: true`:
  /// `report.bound == report.offered == 264`, `unbound.isEmpty`, and the
  /// seven deltas preflight against real parameter paths.
  func testTurboLoRABindsCompletelyOnTheDiT() throws {
    let url = try requireTurboLoRA()
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)

    let dit = Krea2SingleStreamDiT(cfg: Krea2Config())
    let report = try LoRAApplicator.applyDynamically(
      to: dit, loraWeights: weights, scale: 0.6, strict: true, name: "krea2_turbo_lora_rank_64_bf16")

    XCTAssertEqual(report.offered, 264)
    XCTAssertEqual(report.bound, 264)
    XCTAssertTrue(report.unbound.isEmpty, "\(report.unbound)")
    XCTAssertEqual(report.shapeRejected, 0)
    XCTAssertTrue(report.isComplete)

    let session = LoRAPatchSession(module: dit)
    XCTAssertEqual(try session.preflight(weights: weights), 7)
  }
}
