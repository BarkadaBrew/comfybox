import Foundation
import MLX
import XCTest

@testable import ZImage

/// Task #21, ranked step 3: the Swift audio codec chain vs the PINNED
/// reference goldens (Tests/ZImageTests/Fixtures/ltx2-audio, produced by
/// scripts/export_audio_codec_goldens.py from ComfyUI's implementation,
/// CPU fp32, seed 4242; manifest pins checkpoint sha + comfy rev).
///
/// Stage contract (Codex findings #7/#8, confirmed by fixture shapes):
///   z_normalized [1,C,32,F] → denorm → causal decode → mel [1,2,125,64]
///   (125 = 4·32−3) → run_vocoder (base + BWE + 3× resample) → wav
///   [1,2,60000] = 48kHz stereo.
///
/// These tests are EXPECTED RED against the current Swift port: symmetric
/// (non-causal) conv padding, missing per-channel denormalization, missing
/// 4T−3 crop, and a synthesize() that stops at the 16kHz base generator.
final class LTX2AudioCodecParityTests: XCTestCase {

  static let fixtureDir = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .appendingPathComponent("Fixtures/ltx2-audio")

  /// Real audio-VAE weights: same file the reference used; required for
  /// parity (synthetic weights cannot prove operator semantics).
  static let weightsPath = FileManager.default.homeDirectoryForCurrentUser
    .appendingPathComponent(".comfybox/reference/LTX23_audio_vae_bf16.safetensors").path  // Bolt is TCC-invisible to xctest

  private func goldens() throws -> [String: MLXArray] {
    let url = Self.fixtureDir.appendingPathComponent("codec_goldens.safetensors")
    return try MLX.loadArrays(url: url)
  }

  func testFixturesPresentAndShapedAsPinned() throws {
    let g = try goldens()
    XCTAssertEqual(g["mel"]?.shape, [1, 2, 125, 64],
      "mel is 2ch x 64-bin with the causal 4T-3 crop (125 = 4*32-3)")
    XCTAssertEqual(g["wav_final"]?.shape, [1, 2, 60000],
      "final waveform is 48kHz stereo (125 frames x 480 samples)")
    XCTAssertEqual(g["z_normalized"]?.shape.first, 1)
  }

  func testDenormalizationMatchesReference() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath),
      "reference audio VAE weights not on this machine")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)

    let zDenorm = vae.denormalize(g["z_normalized"]!)
    let diff = MLX.abs(zDenorm - g["z_denorm"]!).max().item(Float.self)
    XCTAssertLessThan(diff, 1e-4, "per-channel latent denormalization must match reference")
  }

  func testCausalDecodeMatchesReferenceMel() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath),
      "reference audio VAE weights not on this machine")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)

    let mel = vae.decodeToMel(g["z_normalized"]!)
    XCTAssertEqual(mel.shape, [1, 2, 125, 64], "causal target shape 4T-3")
    let ref = g["mel"]!
    let denom = MLX.abs(ref).mean().item(Float.self)
    let diff = MLX.abs(mel - ref).mean().item(Float.self)
    XCTAssertLessThan(diff / max(denom, 1e-6), 0.02,
      "mel relative error must be small (fp32 vs bf16-loaded weights)")
  }

  private func corr(_ a: MLXArray, _ b: MLXArray) -> Float {
    let x = a.reshaped([-1]).asType(.float32), y = b.reshaped([-1]).asType(.float32)
    let n = ((x - x.mean()) * (y - y.mean())).mean()
    let d = MLX.sqrt(((x - x.mean()) * (x - x.mean())).mean()) * MLX.sqrt(((y - y.mean()) * (y - y.mean())).mean())
    return (n / d).item(Float.self)
  }

  func testMicro_GeneratorStages() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath), "weights absent")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)
    let gen = vae.vocoder!.vocoder

    let nlc = g["gen_in"]!.transposed(0, 2, 1)         // [B, T, 128]
    let g0 = gen.convPre(nlc)                          // [B, T, 1536]
    XCTAssertGreaterThan(corr(g0.transposed(0, 2, 1), g["gen_conv_pre"]!), 0.999, "conv_pre")

    let refG0 = g["gen_conv_pre"]!.transposed(0, 2, 1) // isolate up0 with ref input
    let g1 = gen.ups[0](refG0)
    XCTAssertEqual(g1.transposed(0, 2, 1).shape, g["gen_up0"]!.shape, "up0 shape")
    XCTAssertGreaterThan(corr(g1.transposed(0, 2, 1), g["gen_up0"]!), 0.999, "ups[0] transpose-conv")

    let refG1 = g["gen_up0"]!.transposed(0, 2, 1)      // isolate resblock group 0
    var xs = gen.resblocks[0](refG1)
    for j in 1..<gen.numKernels { xs = xs + gen.resblocks[j](refG1) }
    xs = xs / MLXArray(Float(gen.numKernels))
    // micro-micro: isolate activation vs dilated conv inside AMP block 0
    let act0 = gen.resblocks[0].acts1[0](refG1)
    XCTAssertGreaterThan(corr(act0.transposed(0, 2, 1), g["amp_act0"]!), 0.999, "anti-aliased snake (Activation1d)")
    let conv0 = gen.resblocks[0].convs1[0](g["amp_act0"]!.transposed(0, 2, 1))
    XCTAssertGreaterThan(corr(conv0.transposed(0, 2, 1), g["amp_conv0"]!), 0.999, "dilated conv1")

    XCTAssertGreaterThan(corr(xs.transposed(0, 2, 1), g["gen_res0"]!), 0.999, "resblock group 0 (AMP)")
  }

  func testBisect_BaseVocoderStage() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath), "weights absent")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)
    // Reference: vocoder_input = mel.transpose(2,3); stereo fold; base generator.
    let bft = g["mel"]!.transposed(0, 1, 3, 2)
    let folded = MLX.concatenated([bft[0..., 0], bft[0..., 1]], axis: 1)
    let base = vae.vocoder!.synthesize(folded)
    XCTAssertEqual(base.shape, g["wav16k_base"]!.shape, "base stage shape")
    XCTAssertGreaterThan(corr(base, g["wav16k_base"]!), 0.99, "BASE generator parity")
  }

  func testBisect_MelOfBaseStage() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath), "weights absent")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)
    let base = g["wav16k_base"]!  // reference base as input isolates THIS stage
    let b = base.dim(0), c = base.dim(1)
    let mel = vae.vocoder!.melStft.logMel(base.reshaped([b * c, -1]), hopLength: 80)
      .reshaped([b, c, 64, -1])
    XCTAssertEqual(mel.shape, g["mel_of_base"]!.shape, "mel-of-base shape")
    XCTAssertGreaterThan(corr(mel, g["mel_of_base"]!), 0.99, "causal STFT+mel parity")
  }

  func testBisect_ResampleSkipStage() throws {
    let g = try goldens()
    let skip = LTX2HannUpsampler.upsample(g["wav16k_base"]!, ratio: 3)
    XCTAssertEqual(skip.shape, g["resample_skip"]!.shape, "skip shape")
    XCTAssertGreaterThan(corr(skip, g["resample_skip"]!), 0.999, "hann resampler parity")
  }

  func testBisect_BWEResidualStage() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath), "weights absent")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)
    let melOfBase = g["mel_of_base"]!  // (B,2,64,frames) — reference input isolates BWE
    let folded = MLX.concatenated([melOfBase[0..., 0], melOfBase[0..., 1]], axis: 1)
    let nlc = folded.transposed(0, 2, 1)
    let residual = vae.vocoder!.bweGenerator(nlc).transposed(0, 2, 1)
    XCTAssertEqual(residual.shape, g["bwe_residual"]!.shape, "residual shape")
    XCTAssertGreaterThan(corr(residual, g["bwe_residual"]!), 0.99, "BWE generator parity")
  }

  func testFullChainProducesFortyEightKStereo() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath),
      "reference audio VAE weights not on this machine")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)

    let wav = vae.decodeToWaveform(g["z_normalized"]!)
    XCTAssertEqual(wav.shape, [1, 2, 60000],
      "run_vocoder chain = base 16k + BWE + 3x resample -> 48kHz stereo")
    // Waveform parity is looser (long accumulation): assert correlation, not
    // pointwise closeness.
    let ref = g["wav_final"]!
    let a = wav.reshaped([-1]).asType(.float32)
    let b = ref.reshaped([-1]).asType(.float32)
    let corr = ((a - a.mean()) * (b - b.mean())).mean()
      / (MLX.sqrt(((a - a.mean()) * (a - a.mean())).mean()) * MLX.sqrt(((b - b.mean()) * (b - b.mean())).mean()))
    XCTAssertGreaterThan(corr.item(Float.self), 0.99,
      "waveforms must be near-identical up to numeric noise")
  }

  /// Regression for the electrical buzz in non-voice regions: a near-zero
  /// normalized latent must be restored to the training log-mel silence floor
  /// before the matched BigVGAN+BWE chain sees it.
  func testSilentLatentDecodesToNearSilence() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath),
      "reference audio VAE weights not on this machine")
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)
    let latent = MLXArray.zeros([1, 8, 8, 16])
    let decodedMel = vae.decodeToMel(latent)
    let vocoderMel = vae.prepareMelForVocoder(decodedMel, normalizedLatent: latent)
    let wav = vae.decodeToWaveform(latent)
    MLX.eval(vocoderMel, wav)

    XCTAssertEqual(
      MLX.min(vocoderMel).item(Float.self), LTX2AudioVAE.logMelFloor,
      accuracy: 1e-5)
    XCTAssertEqual(
      MLX.max(vocoderMel).item(Float.self), LTX2AudioVAE.logMelFloor,
      accuracy: 1e-5)
    let rms = MLX.sqrt(MLX.mean(wav * wav)).item(Float.self)
    print("SILENT_LATENT_RMS=\(rms)")
    XCTAssertTrue(rms.isFinite, "silent decode produced NaN/Inf")
    XCTAssertLessThan(rms, 1e-4, "silent latent produced audible buzz (RMS \(rms))")
  }

  /// Exercise the monolith's actual `vocoder.vocoder.*` BigVGAN weights with
  /// a pinned, deterministic mel rather than random synthetic parameters.
  func testBundledBigVGANProducesFiniteNonDegenerateOutputFromKnownMel() throws {
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath),
      "reference audio VAE weights not on this machine")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)
    let wav = vae.vocoder!.synthesize(g["gen_in"]!)
    MLX.eval(wav)

    let rms = MLX.sqrt(MLX.mean(wav * wav)).item(Float.self)
    let maxAbs = MLX.max(MLX.abs(wav)).item(Float.self)
    XCTAssertTrue(rms.isFinite && maxAbs.isFinite, "bundled BigVGAN produced NaN/Inf")
    XCTAssertGreaterThan(rms, 1e-3, "bundled BigVGAN output is degenerate/silent")
    XCTAssertGreaterThan(maxAbs, rms, "bundled BigVGAN output has no dynamic range")
  }
}
