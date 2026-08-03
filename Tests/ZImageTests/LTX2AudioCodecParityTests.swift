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

  func testFullChainProducesFortyEightKStereo() throws {
    throw XCTSkip("BWE chain pending: UpSample1d (3x hann) + synthesizeFull not yet implemented — tests 1-3 gate the VAE half; this gates the vocoder half")
    try XCTSkipUnless(FileManager.default.fileExists(atPath: Self.weightsPath),
      "reference audio VAE weights not on this machine")
    let g = try goldens()
    let vae = try LTX2AudioVAE.load(path: Self.weightsPath)

    let wav = MLXArray.zeros([1, 2, 60000])  // placeholder past unreachable skip
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
}
