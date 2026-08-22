import XCTest

@testable import ZImage

/// WP-E6 (FDD §3.6): `LoRAScanner.detectCompatibilityFromKeys` gains a Krea-2
/// branch. Every Krea-2 LoRA in the vault was classified `unknown` before this
/// (library.json, read 2026-08-22: krea2_turbo_lora_rank_64_bf16,
/// kroma-lora-v0.3, kroma-v0.2-base-lora-rank-384-fro-0985 and
/// krea2_filter_bypass_fedor all `["unknown"]`). Key shapes below are copied
/// from those files' headers.
final class LoRAScannerCompatibilityTests: XCTestCase {

  /// krea2_turbo_lora_rank_64_bf16: kohya-form pairs on `blocks.N.attn.w*`
  /// plus `.diff_b` deltas — no `txtfusion.` key at all.
  func testTurboLoRAKeysClassifyAsKrea2() {
    let keys = [
      "diffusion_model.blocks.0.attn.gate.lora_down.weight",
      "diffusion_model.blocks.0.attn.gate.lora_up.weight",
      "diffusion_model.blocks.0.attn.wk.lora_down.weight",
      "diffusion_model.blocks.0.attn.wk.lora_up.weight",
      "diffusion_model.blocks.0.attn.wq.lora_down.weight",
      "diffusion_model.blocks.0.attn.wq.lora_up.weight",
      "diffusion_model.blocks.0.mlp.down.lora_down.weight",
      "diffusion_model.first.diff_b",
      "diffusion_model.tmlp.0.diff_b",
    ]
    XCTAssertEqual(LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: keys), ["krea2"])
  }

  /// kroma-lora-v0.3: suffix-form `lora_A`/`lora_B` (no trailing `.weight`)
  /// plus `.diff` on qknorm scales.
  func testKromaV03KeysClassifyAsKrea2() {
    let keys = [
      "diffusion_model.blocks.0.attn.gate.lora_A",
      "diffusion_model.blocks.0.attn.gate.lora_B",
      "diffusion_model.blocks.0.attn.qknorm.knorm.scale.diff",
      "diffusion_model.blocks.0.attn.wk.lora_A",
      "diffusion_model.blocks.0.attn.wk.lora_B",
      "diffusion_model.blocks.0.attn.wo.lora_A",
      "diffusion_model.blocks.0.attn.wo.lora_B",
    ]
    XCTAssertEqual(LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: keys), ["krea2"])
  }

  /// kroma-v0.2-base-lora-rank-384: diffusers-dotted `lora_A.weight` form.
  func testKromaV02BaseKeysClassifyAsKrea2() {
    let keys = [
      "diffusion_model.blocks.15.mlp.up.lora_A.weight",
      "diffusion_model.blocks.15.mlp.up.lora_B.weight",
      "diffusion_model.blocks.0.attn.wv.lora_A.weight",
      "diffusion_model.blocks.0.attn.wv.lora_B.weight",
    ]
    XCTAssertEqual(LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: keys), ["krea2"])
  }

  /// krea2_filter_bypass_fedor: ONE tensor, `txtfusion.projector.diff` — no
  /// block key at all. `txtfusion.` alone is a Krea-2 signature.
  func testBypassLoRATxtfusionOnlyClassifiesAsKrea2() {
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: ["diffusion_model.txtfusion.projector.diff"]),
      ["krea2"])
  }

  /// Metadata still wins, and other families are untouched by the new branch.
  func testOtherFamiliesUnaffected() {
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(
        metadata: nil,
        sampleKeys: ["diffusion_model.layers.0.attention.to_q.lora_down.weight",
                     "diffusion_model.context_refiner.0.attention.to_q.lora_down.weight"]),
      ["z-image"])
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(
        metadata: nil,
        sampleKeys: ["transformer_blocks.0.attn1.to_q.lora_A.weight",
                     "transformer_blocks.0.audio_attn1.to_q.lora_A.weight"]),
      ["ltx"])
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(metadata: ["ss_base_model_version": "krea2"], sampleKeys: []),
      ["krea2"])
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: ["something.else.lora_A.weight"]),
      ["unknown"])
  }
}
