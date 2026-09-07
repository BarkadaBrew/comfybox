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

  /// #313: production LTX-2.3 LoRAs trained with sd-scripts' `networks.lora_ltx2`
  /// (`ic-lora-union-control-ref0.5`, `phool-realism-ltx-2.3-v1.0`,
  /// `LTX2.3_reasoning_Sulphur-2_I2V_V4` — all read straight from
  /// ~/Models/loras/library.json as `unknown` before this fix) carry no
  /// `ss_base_model_version`/`modelspec.architecture` metadata at all and use
  /// the diffusers-style `diffusion_model.transformer_blocks.N.attn1/attn2/ff`
  /// layout — no `layers.` substring, no audio/video branch. LTX-2's real
  /// module tree (LTX2TransformerBlock.swift) is the only architecture with
  /// BOTH `attn1` (self) and `attn2` (cross) under one block — Z-Image has a
  /// single `attention`, never a second `attn2` — so requiring both is a safe
  /// signature distinct from any generic diffusers Z-Image export.
  func testLTX2ComfyExportTransformerBlocksClassifyAsLTX() {
    let keys = [
      "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
      "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight",
      "diffusion_model.transformer_blocks.0.attn1.to_v.lora_A.weight",
      "diffusion_model.transformer_blocks.0.attn1.to_out.0.lora_A.weight",
      "diffusion_model.transformer_blocks.0.attn2.to_q.lora_A.weight",
      "diffusion_model.transformer_blocks.0.attn2.to_k.lora_A.weight",
      "diffusion_model.transformer_blocks.0.attn2.to_v.lora_A.weight",
      "diffusion_model.transformer_blocks.0.attn2.to_out.0.lora_A.weight",
      "diffusion_model.transformer_blocks.0.ff.net.0.proj.lora_A.weight",
      "diffusion_model.transformer_blocks.0.ff.net.2.lora_A.weight",
    ]
    XCTAssertEqual(LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: keys), ["ltx"])
  }

  /// A `transformer_blocks` LoRA with only `attn1` (single attention, no
  /// cross-attn) is NOT LTX-2's signature — must not be swept into ["ltx"].
  func testTransformerBlocksSingleAttentionDoesNotClassifyAsLTX() {
    let keys = [
      "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
      "diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight",
    ]
    XCTAssertNotEqual(LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: keys), ["ltx"])
  }

  /// #313: `Anneliese_Zbase3.safetensors` (real vault file, `unknown` before
  /// this fix) is a kohya/ComfyUI "comfy-export" LoRA: `lora_unet_` prefix,
  /// `layers_N_...` — underscores throughout, no dots at all. `double_blocks_`
  /// / `single_blocks_` already had an underscore fallback; `layers_` did not.
  func testZImageComfyExportUnderscoreKeysClassifyAsZImage() {
    let keys = [
      "lora_unet_layers_0_attention_out.lora_down.weight",
      "lora_unet_layers_0_attention_out.lora_up.weight",
      "lora_unet_layers_0_attention_qkv.lora_down.weight",
      "lora_unet_layers_0_attention_qkv.lora_up.weight",
      "lora_unet_layers_0_feed_forward_w1.lora_down.weight",
      "lora_unet_layers_0_feed_forward_w1.lora_up.weight",
    ]
    XCTAssertEqual(LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: keys), ["z-image"])
  }

  /// #313: same file's real metadata — `ss_base_model_version: "z_image"`
  /// (underscore, not the `"zimage"`/`"z-image"` the switch recognized) and
  /// `modelspec.architecture: "Z-Image/lora"` (the architecture branch had no
  /// z-image case at all, only klein/chroma/ltx/flux) — both metadata paths
  /// fell through to key heuristics and, before the key fix above, to
  /// `unknown`.
  func testZImageMetadataVariantsClassifyAsZImage() {
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(metadata: ["ss_base_model_version": "z_image"], sampleKeys: []),
      ["z-image"])
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(metadata: ["modelspec.architecture": "Z-Image/lora"], sampleKeys: []),
      ["z-image"])
  }

  /// #313: synthetic — no real "control_" fixture exists in the vault, so
  /// this is built straight from `ZImageControlTransformer2D`'s own module
  /// keys (control_all_x_embedder / control_noise_refiner / control_layers).
  func testZImageControlNetKeysClassifyAsZImage() {
    let keys = [
      "diffusion_model.control_layers.0.attention.to_q.lora_down.weight",
      "diffusion_model.control_layers.0.attention.to_q.lora_up.weight",
      "diffusion_model.control_noise_refiner.0.attention.to_q.lora_down.weight",
      "diffusion_model.control_noise_refiner.0.attention.to_q.lora_up.weight",
    ]
    XCTAssertEqual(LoRAScanner.detectCompatibility(metadata: nil, sampleKeys: keys), ["z-image"])
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

// MARK: - #402 fix round 1 (Critical 2): the flux1/klein-9b mistag

extension LoRAScannerCompatibilityTests {

  /// The real vault file this bug was found on: `ss_base_model_version:
  /// "flux2_klein_9b"` (the "2" this switch was missing) with
  /// `modelspec.architecture: "flux-2/lora"` — before this fix, neither
  /// metadata field was recognized and it fell all the way through to a
  /// confident (wrong) `["flux1"]` via the generic `contains("flux")`
  /// fallback. Fixed at the SOURCE: the `ss_base_model_version` switch now
  /// accepts the "2" spelling directly.
  func testFlux2Klein9bSpellingClassifiesAsKlein9b() {
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(
        metadata: ["ss_base_model_version": "flux2_klein_9b", "modelspec.architecture": "flux-2/lora"],
        sampleKeys: []),
      ["klein-9b"])
  }

  func testFlux2Klein4bSpellingClassifiesAsKlein4b() {
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(metadata: ["ss_base_model_version": "flux2_klein_4b"], sampleKeys: []),
      ["klein-4b"])
  }

  /// Defense in depth: even with NO `ss_base_model_version` at all, a bare
  /// `modelspec.architecture: "flux-2/lora"` (no klein_9b/klein_4b marker)
  /// must not fall into the generic flux1 branch — it falls through to key
  /// heuristics, which land on "unknown" for a key shape they don't
  /// recognize (never a guessed "flux1").
  func testModelspecFlux2WithNoKleinMarkerNeverConfidentlyFlux1() {
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(
        metadata: ["modelspec.architecture": "flux-2/lora"], sampleKeys: ["something.else.lora_A.weight"]),
      ["unknown"])
  }

  /// An EXPLICIT Flux.1 marker (real Flux.1-dev/schnell) still classifies
  /// confidently — the fallback was narrowed, not removed.
  func testModelspecExplicitFlux1MarkerStillClassifiesAsFlux1() {
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(
        metadata: ["modelspec.architecture": "flux-1-dev/lora"], sampleKeys: []),
      ["flux1"])
  }

  /// `ss_base_model_version: "flux1"` (the direct, explicit metadata form —
  /// unaffected by this fix, which only touched the GENERIC `contains("flux")`
  /// fallback) still classifies confidently, real vault files
  /// (chroma-unlocked-v47…, sweet-asian-flux) both use this exact form.
  func testExplicitSSBaseModelVersionFlux1StillConfident() {
    XCTAssertEqual(
      LoRAScanner.detectCompatibility(metadata: ["ss_base_model_version": "flux1"], sampleKeys: []),
      ["flux1"])
  }
}
