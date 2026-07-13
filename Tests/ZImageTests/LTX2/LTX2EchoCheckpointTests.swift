import XCTest
import MLX
@testable import ZImage

/// Unit tests for the JoyAI-Echo monolithic-checkpoint load path (Phase 1).
///
/// These exercise the pure prefix-filter / key-normalization logic that lets the
/// existing per-component LTX-2 loaders consume Echo's single 46 GB file. They
/// build tiny in-memory tensor dicts (scalar MLXArrays) so they run without the
/// real checkpoint.
final class LTX2EchoCheckpointTests: XCTestCase {

    override func setUpWithError() throws {
        // Match the rest of the MLX suite: skip (don't crash) where the Metal
        // library can't be located in the test runner.
        do {
            try MLX.withError {
                let probe = MLXArray([1 as Float, 2], [2]) + MLXArray([3 as Float, 4], [2])
                MLX.eval(probe)
            }
        } catch {
            throw XCTSkip("MLX evaluation is unavailable in this test runner: \(error)")
        }
    }

    /// A tiny scalar stand-in tensor.
    private func t(_ v: Float = 1.0) -> MLXArray { MLXArray([v]) }

    /// The transformer built exactly as `LTX2VideoGenerator.load()` builds it
    /// (LTX-2.3 / JoyAI-Echo video variant).
    private func makeVideoTransformer() -> LTX2Transformer {
        LTX2Transformer(
            numHeads: 32, headDim: 128, inChannels: 128, outChannels: 128,
            numLayers: 48, crossAttentionDim: 4096, captionChannels: 3840,
            normEps: 1e-6, hasPromptAdaLN: true, timestepScaleMultiplier: 1000,
            positionalEmbeddingTheta: 10000, positionalEmbeddingMaxPos: [20, 2048, 2048],
            useMiddleIndicesGrid: true, ropeMode: .split, doublePrecisionRoPE: true
        )
    }

    /// Representative slice of Echo monolith keys across every top-level prefix.
    private func echoMonolithKeys() -> [String: MLXArray] {
        var w: [String: MLXArray] = [:]
        // --- transformer (video DiT) ---
        w["model.diffusion_model.patchify_proj.weight"] = t()
        w["model.diffusion_model.proj_out.weight"] = t()
        w["model.diffusion_model.adaln_single.linear.weight"] = t()
        w["model.diffusion_model.scale_shift_table"] = t()
        w["model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.ff.net.2.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.scale_shift_table"] = t()
        // --- audio DiT branch (must be excluded on the video-only path) ---
        w["model.diffusion_model.audio_patchify_proj.weight"] = t()
        w["model.diffusion_model.audio_proj_out.weight"] = t()
        w["model.diffusion_model.audio_adaln_single.linear.weight"] = t()
        w["model.diffusion_model.av_ca_a2v_gate_adaln_single.linear.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.audio_attn1.to_q.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.audio_ff.net.0.proj.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.audio_to_video_attn.to_out.0.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.video_to_audio_attn.to_q.weight"] = t()
        w["model.diffusion_model.transformer_blocks.0.scale_shift_table_a2v_ca_video"] = t()
        w["model.diffusion_model.transformer_blocks.0.scale_shift_table_a2v_ca_audio"] = t()
        // --- connectors (handled by the text encoder, not the DiT) ---
        w["model.diffusion_model.video_embeddings_connector.learnable_registers"] = t()
        w["model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"] = t()
        w["model.diffusion_model.audio_embeddings_connector.learnable_registers"] = t()
        w["model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"] = t()
        // --- text embedding projection (aggregate embeds, bare prefix) ---
        w["text_embedding_projection.video_aggregate_embed.weight"] = t()
        w["text_embedding_projection.video_aggregate_embed.bias"] = t()
        w["text_embedding_projection.audio_aggregate_embed.weight"] = t()
        w["text_embedding_projection.audio_aggregate_embed.bias"] = t()
        // --- video VAE ---
        w["vae.encoder.conv_in.conv.weight"] = t()
        w["vae.decoder.conv_out.conv.weight"] = t()
        w["vae.per_channel_statistics.mean-of-means"] = t(0.5)
        w["vae.per_channel_statistics.std-of-means"] = t(2.0)
        // --- audio VAE + vocoder (Phase 2, must never leak into Phase 1) ---
        w["audio_vae.encoder.conv_in.conv.weight"] = t()
        w["audio_vae.per_channel_statistics.mean-of-means"] = t()
        w["vocoder.vocoder.conv_pre.weight"] = t()
        w["vocoder.mel_stft.mel_basis"] = t()
        return w
    }

    // MARK: - Monolith detection

    func testDetectsEchoMonolithLayout() {
        XCTAssertTrue(LTX2EchoCheckpoint.isMonolithLayout(echoMonolithKeys().keys))
    }

    func testRejectsSeparateTransformerFileAsMonolith() {
        // The distilled transformer-only file uses the `transformer.` prefix and
        // carries no VAE / audio_vae / vocoder / projection keys.
        let separate: [String: MLXArray] = [
            "transformer.patchify_proj.weight": t(),
            "transformer.transformer_blocks.0.attn1.to_q.weight": t(),
        ]
        XCTAssertFalse(LTX2EchoCheckpoint.isMonolithLayout(separate.keys))
    }

    func testRejectsDitOnlyModelPrefixAsMonolith() {
        // model.diffusion_model.* without the VAE/extra prefixes is not a monolith.
        let ditOnly: [String: MLXArray] = [
            "model.diffusion_model.patchify_proj.weight": t(),
            "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": t(),
        ]
        XCTAssertFalse(LTX2EchoCheckpoint.isMonolithLayout(ditOnly.keys))
    }

    // MARK: - Transformer subset (sanitizeWeights, video-only)

    func testSanitizeSelectsVideoDitAndRemapsNames() {
        let out = LTX2Transformer.sanitizeWeights(echoMonolithKeys())
        // Core video keys present with prefix stripped.
        XCTAssertNotNil(out["patchify_proj.weight"])
        XCTAssertNotNil(out["proj_out.weight"])
        XCTAssertNotNil(out["adaln_single.linear.weight"])
        XCTAssertNotNil(out["transformer_blocks.0.attn1.to_q.weight"])
        // Name remaps.
        XCTAssertNotNil(out["transformer_blocks.0.attn1.to_out.weight"])       // .to_out.0. -> .to_out.
        XCTAssertNil(out["transformer_blocks.0.attn1.to_out.0.weight"])
        XCTAssertNotNil(out["transformer_blocks.0.ff.proj_in.weight"])          // .ff.net.0.proj. -> .ff.proj_in.
        XCTAssertNotNil(out["transformer_blocks.0.ff.proj_out.weight"])         // .ff.net.2. -> .ff.proj_out.
    }

    func testSanitizeExcludesAudioBranchOnVideoOnlyPath() {
        let out = LTX2Transformer.sanitizeWeights(echoMonolithKeys())
        // Top-level audio keys.
        XCTAssertNil(out["audio_patchify_proj.weight"])
        XCTAssertNil(out["audio_proj_out.weight"])
        XCTAssertNil(out["audio_adaln_single.linear.weight"])
        XCTAssertNil(out["av_ca_a2v_gate_adaln_single.linear.weight"])
        // Per-block cross-modal / audio keys.
        XCTAssertNil(out["transformer_blocks.0.audio_attn1.to_q.weight"])
        XCTAssertNil(out["transformer_blocks.0.audio_ff.proj_in.weight"])
        XCTAssertNil(out["transformer_blocks.0.audio_to_video_attn.to_out.weight"])
        XCTAssertNil(out["transformer_blocks.0.video_to_audio_attn.to_q.weight"])
        XCTAssertNil(out["transformer_blocks.0.scale_shift_table_a2v_ca_video"])
        XCTAssertNil(out["transformer_blocks.0.scale_shift_table_a2v_ca_audio"])
        // No key mentioning audio / av_ca should survive.
        for k in out.keys {
            XCTAssertFalse(k.contains("audio_"), "audio key leaked into video DiT: \(k)")
            XCTAssertFalse(k.contains("av_ca_"), "av_ca key leaked into video DiT: \(k)")
            XCTAssertFalse(k.contains("scale_shift_table_a2v"), "a2v table leaked: \(k)")
        }
    }

    /// The anti-noise guard: every remapped video key must land on a real
    /// parameter of the LTX2Transformer module. `update(verify:[.shapeMismatch])`
    /// silently DROPS unmatched names — a wrong remap loads 0 weights and renders
    /// pure noise while reporting success. This asserts the remap targets actually
    /// exist in the module namespace (i.e. matched-key count > 0 and total).
    func testSanitizedVideoKeysAllMatchTransformerModuleParameters() {
        let transformer = makeVideoTransformer()
        let moduleKeys = Set(transformer.parameters().flattened().map { $0.0 })
        XCTAssertFalse(moduleKeys.isEmpty)

        let sanitized = LTX2Transformer.sanitizeWeights(echoMonolithKeys())
        XCTAssertFalse(sanitized.isEmpty, "remap produced no video keys")

        var matched = 0
        for key in sanitized.keys {
            if moduleKeys.contains(key) {
                matched += 1
            } else {
                XCTFail("remapped key not found in transformer module params: \(key)")
            }
        }
        // Full match — no silent drops on the video path.
        XCTAssertEqual(matched, sanitized.count)
    }

    func testSanitizeExcludesVaeConnectorsAndProjection() {
        let out = LTX2Transformer.sanitizeWeights(echoMonolithKeys())
        for k in out.keys {
            XCTAssertFalse(k.hasPrefix("vae."), "vae key leaked into DiT: \(k)")
            XCTAssertFalse(k.hasPrefix("audio_vae."), "audio_vae key leaked: \(k)")
            XCTAssertFalse(k.hasPrefix("vocoder."), "vocoder key leaked: \(k)")
            XCTAssertFalse(k.contains("embeddings_connector"), "connector leaked into DiT: \(k)")
            XCTAssertFalse(k.contains("aggregate_embed"), "projection leaked into DiT: \(k)")
        }
    }

    // MARK: - Video VAE subset

    func testVideoVAETensorsSelectsOnlyVaePrefix() {
        let vae = LTX2EchoCheckpoint.videoVAETensors(from: echoMonolithKeys())
        XCTAssertNotNil(vae["vae.encoder.conv_in.conv.weight"])
        XCTAssertNotNil(vae["vae.decoder.conv_out.conv.weight"])
        XCTAssertNotNil(vae["vae.per_channel_statistics.mean-of-means"])
        // audio_vae must not be mistaken for vae.
        for k in vae.keys {
            XCTAssertFalse(k.hasPrefix("audio_vae."), "audio_vae leaked into video VAE: \(k)")
            XCTAssertFalse(k.hasPrefix("vocoder."), "vocoder leaked into video VAE: \(k)")
        }
    }

    func testVideoVAETensorsInjectsDecoderStats() {
        // Echo stores stats only at the top level; the decoder module also needs
        // them, so the adapter must mirror them under vae.decoder.*.
        let vae = LTX2EchoCheckpoint.videoVAETensors(from: echoMonolithKeys())
        XCTAssertNotNil(vae["vae.decoder.per_channel_statistics.mean"])
        XCTAssertNotNil(vae["vae.decoder.per_channel_statistics.std"])
    }

    // MARK: - Audio VAE subset (Phase 2)

    func testAudioVAETensorsSelectsOnlyAudioVaePrefix() {
        var keys = echoMonolithKeys()
        keys["audio_vae.decoder.conv_out.conv.weight"] = t()
        keys["audio_vae.per_channel_statistics.std-of-means"] = t(2.0)
        let avae = LTX2EchoCheckpoint.audioVAETensors(from: keys)
        XCTAssertNotNil(avae["audio_vae.encoder.conv_in.conv.weight"])
        XCTAssertNotNil(avae["audio_vae.decoder.conv_out.conv.weight"])
        XCTAssertNotNil(avae["audio_vae.per_channel_statistics.mean-of-means"])
        // The video VAE (`vae.`) and vocoder must never leak into the audio VAE.
        for k in avae.keys {
            XCTAssertTrue(k.hasPrefix("audio_vae."), "non-audio_vae key leaked: \(k)")
            XCTAssertFalse(k.hasPrefix("vocoder."), "vocoder leaked into audio VAE: \(k)")
        }
    }

    func testAudioVAETensorsInjectsDecoderStats() {
        var keys = echoMonolithKeys()
        keys["audio_vae.per_channel_statistics.std-of-means"] = t(2.0)
        let avae = LTX2EchoCheckpoint.audioVAETensors(from: keys)
        XCTAssertNotNil(avae["audio_vae.decoder.per_channel_statistics.mean"])
        XCTAssertNotNil(avae["audio_vae.decoder.per_channel_statistics.std"])
    }

    func testVideoAndAudioVaeSelectorsAreDisjoint() {
        let keys = echoMonolithKeys()
        let vae = LTX2EchoCheckpoint.videoVAETensors(from: keys)
        let avae = LTX2EchoCheckpoint.audioVAETensors(from: keys)
        // No source (non-injected) key should appear in both selectors.
        for k in vae.keys where k.hasPrefix("vae.") {
            XCTAssertNil(avae[k], "video-VAE key also captured by audio-VAE selector: \(k)")
        }
        for k in avae.keys where k.hasPrefix("audio_vae.") {
            XCTAssertFalse(vae.keys.contains(k), "audio-VAE key also captured by video-VAE selector: \(k)")
        }
    }

    // MARK: - Vocoder subset (Phase 2)

    func testVocoderTensorsSelectsAllVocoderSubgenerators() {
        var keys = echoMonolithKeys()
        keys["vocoder.vocoder.resblocks.0.acts1.0.act.alpha"] = t()
        keys["vocoder.bwe_generator.conv_pre.weight"] = t()
        let voc = LTX2EchoCheckpoint.vocoderTensors(from: keys)
        XCTAssertNotNil(voc["vocoder.vocoder.conv_pre.weight"])
        XCTAssertNotNil(voc["vocoder.vocoder.resblocks.0.acts1.0.act.alpha"])
        XCTAssertNotNil(voc["vocoder.bwe_generator.conv_pre.weight"])
        XCTAssertNotNil(voc["vocoder.mel_stft.mel_basis"])
        // audio_vae / video VAE must not leak into the vocoder subset.
        for k in voc.keys {
            XCTAssertTrue(k.hasPrefix("vocoder."), "non-vocoder key leaked: \(k)")
        }
    }

    // MARK: - Connector / projection normalization for the text encoder

    func testNormalizeMonolithProjectionKeysStripsPrefixes() {
        let proj = LTX2TextEncoder.normalizeMonolithProjectionKeys(echoMonolithKeys())
        // Connectors: model.diffusion_model. stripped -> bare connector name.
        XCTAssertNotNil(proj["video_embeddings_connector.learnable_registers"])
        XCTAssertNotNil(proj["video_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.weight"])
        XCTAssertNotNil(proj["audio_embeddings_connector.learnable_registers"])
        // Aggregate embeds: text_embedding_projection. stripped -> bare.
        XCTAssertNotNil(proj["video_aggregate_embed.weight"])
        XCTAssertNotNil(proj["video_aggregate_embed.bias"])
        XCTAssertNotNil(proj["audio_aggregate_embed.weight"])
    }

    func testNormalizeMonolithProjectionKeysExcludesDitAndVae() {
        let proj = LTX2TextEncoder.normalizeMonolithProjectionKeys(echoMonolithKeys())
        for k in proj.keys {
            XCTAssertFalse(k.hasPrefix("vae."), "vae leaked into projection: \(k)")
            XCTAssertFalse(k.hasPrefix("patchify_proj"), "DiT leaked into projection: \(k)")
            XCTAssertFalse(k.contains("transformer_blocks."), "DiT block leaked into projection: \(k)")
            XCTAssertFalse(k.hasPrefix("model.diffusion_model."), "prefix not stripped: \(k)")
            XCTAssertFalse(k.hasPrefix("text_embedding_projection."), "prefix not stripped: \(k)")
        }
    }
}
