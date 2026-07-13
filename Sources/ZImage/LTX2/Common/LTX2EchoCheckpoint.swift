// LTX2EchoCheckpoint.swift — JoyAI-Echo monolithic-checkpoint load adapter
// Phase 1 of the JoyAI-Echo → MLX/Swift port
//
// JoyAI-Echo ships ONE monolithic ~46 GB safetensors file whose keys are
// namespaced by top-level prefix:
//
//   model.diffusion_model.*        — the dual-stream DiT (video + audio halves)
//   vae.*                          — the spatial/temporal video VAE
//   audio_vae.*                    — the 2D mel VAE            (Phase 2)
//   vocoder.*                      — BigVGAN v2 vocoder        (Phase 2)
//   text_embedding_projection.*    — aggregate embeds (video/audio, bare prefix)
//
// ComfyBox today loads per-component files (transformer-distilled.safetensors,
// connector.safetensors, vae_encoder/decoder.safetensors). Rather than split the
// monolith into new 46 GB files, this adapter reads the single file once and
// hands prefix-filtered *subsets* to the existing loaders:
//
//   • transformer → LTX2Transformer.sanitizeWeights  (already prefix-filters the
//                    `model.diffusion_model.` branch and, as of Phase 1, skips the
//                    audio branch explicitly)
//   • video VAE   → LTX2WeightLoader.loadVAEWeightsFromTensors (expects `vae.` keys)
//   • connectors + aggregate embeds → LTX2TextEncoder.loadWeightsFromMonolith
//
// Because MLX safetensors arrays are materialized lazily on first eval, the
// audio-branch / audio_vae / vocoder tensors referenced by the full dict are
// never realized on the video-only Phase-1 path — only the subsets we actually
// eval hit RAM.
//
// The functions here are pure and unit-tested without the real checkpoint.

import Foundation
import MLX

public enum LTX2EchoCheckpoint {

    /// Detects the JoyAI-Echo monolithic layout from a checkpoint's key set.
    ///
    /// True only when the file carries *all three* signatures at once: the DiT
    /// (`model.diffusion_model.`), the video VAE (`vae.`), and at least one
    /// Echo-exclusive top-level group (`audio_vae.`, `vocoder.`, or
    /// `text_embedding_projection.`). This distinguishes the monolith from the
    /// separate `transformer-distilled.safetensors` (which uses the
    /// `transformer.` prefix and bundles no VAE) and from a bare DiT-only file.
    public static func isMonolithLayout<Keys: Sequence>(_ keys: Keys) -> Bool
    where Keys.Element == String {
        var hasDiT = false
        var hasVideoVAE = false
        var hasEchoExtra = false
        for key in keys {
            if key.hasPrefix("model.diffusion_model.") {
                hasDiT = true
            } else if key.hasPrefix("vae.") {
                hasVideoVAE = true
            } else if key.hasPrefix("audio_vae.")
                || key.hasPrefix("vocoder.")
                || key.hasPrefix("text_embedding_projection.") {
                hasEchoExtra = true
            }
            if hasDiT && hasVideoVAE && hasEchoExtra { return true }
        }
        return hasDiT && hasVideoVAE && hasEchoExtra
    }

    /// Extracts the video-VAE subset (`vae.*`) from a monolith tensor dict, in
    /// the `vae.encoder.*` / `vae.decoder.*` / `vae.per_channel_statistics.*`
    /// layout `LTX2WeightLoader.loadVAEWeightsFromTensors` already expects.
    ///
    /// Echo stores per-channel statistics only at the top level
    /// (`vae.per_channel_statistics.mean-of-means` / `std-of-means`). The encoder
    /// remap already picks those up, but the *decoder* module has its own
    /// `per_channel_statistics` submodule (used for un-normalize) and the decoder
    /// remap ignores the top-level stats. The separate-file layout carried a
    /// `vae.decoder.per_channel_statistics.*` copy; here we mirror the top-level
    /// stats into the decoder path so the decoder is normalized identically.
    public static func videoVAETensors(from monolith: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in monolith where key.hasPrefix("vae.") {
            out[key] = value
        }
        if let mean = out["vae.per_channel_statistics.mean-of-means"],
           out["vae.decoder.per_channel_statistics.mean"] == nil {
            out["vae.decoder.per_channel_statistics.mean"] = mean
        }
        if let std = out["vae.per_channel_statistics.std-of-means"],
           out["vae.decoder.per_channel_statistics.std"] == nil {
            out["vae.decoder.per_channel_statistics.std"] = std
        }
        return out
    }

    /// Extracts the audio-VAE subset (`audio_vae.*`) from a monolith tensor dict,
    /// in the `audio_vae.encoder.*` / `audio_vae.decoder.*` /
    /// `audio_vae.per_channel_statistics.*` layout `LTX2AudioVAE` expects.
    ///
    /// Mirrors `videoVAETensors`: the checkpoint stores per-channel statistics
    /// only at the top level (`audio_vae.per_channel_statistics.mean-of-means` /
    /// `std-of-means`); the decoder module needs its own copy for un-normalize, so
    /// we mirror the top-level stats into the decoder path.
    ///
    /// The `audio_vae.` prefix is *longer* than `vae.` and does not start with it,
    /// so the video-VAE selector never captures these keys and vice-versa —
    /// confirmed by unit tests on both sides.
    public static func audioVAETensors(from monolith: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in monolith where key.hasPrefix("audio_vae.") {
            out[key] = value
        }
        if let mean = out["audio_vae.per_channel_statistics.mean-of-means"],
           out["audio_vae.decoder.per_channel_statistics.mean"] == nil {
            out["audio_vae.decoder.per_channel_statistics.mean"] = mean
        }
        if let std = out["audio_vae.per_channel_statistics.std-of-means"],
           out["audio_vae.decoder.per_channel_statistics.std"] == nil {
            out["audio_vae.decoder.per_channel_statistics.std"] = std
        }
        return out
    }

    /// Extracts the vocoder subset (`vocoder.*`) from a monolith tensor dict.
    ///
    /// Covers all three vocoder sub-generators the checkpoint carries:
    /// `vocoder.vocoder.*` (main BigVGAN v2), `vocoder.bwe_generator.*`
    /// (16k→48k bandwidth extension), and `vocoder.mel_stft.*` (mel filterbank +
    /// STFT bases). No key rewriting here — the vocoder loader owns its own
    /// key-map; this is a pure prefix cut that keeps audio_vae / video-VAE keys
    /// out.
    public static func vocoderTensors(from monolith: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in monolith where key.hasPrefix("vocoder.") {
            out[key] = value
        }
        return out
    }
}
