// LTX2VAERowBandTests.swift — does the video VAE round-trip stay clean at the
// refine pass's latent height (latH=16, 512px) where the two-stage output
// shows a flat wash below ~row 300?
//
// Context (docs/HANDOFF-ltx-quality-2026-08-01.md §3): the refine's bottom-band
// truncation is content-independent (identical with garbage or real upsampler
// output) and the upsampler itself is now proven numerically exact
// (LTX2UpsamplerFixtureTests). Remaining suspects: the refine denoise, or the
// plain VAE decode — which was only ever verified clean at 288 tokens/frame
// (latH 12), never at the refine's 384 tokens/frame (latH 16).
//
// This test encodes+decodes a synthetic structured clip at both heights with
// the REAL v2.3 video VAE weights and compares per-row variance profiles.
// A collapse in the bottom rows at 512 but not 384 convicts the VAE; clean
// profiles at both exonerate it and leave the refine denoise as the suspect.
//
// Needs ~/LocalModels/LTX23_video_vae_bf16.safetensors (bare decoder.*/
// encoder.* keys, Kijai LTX2.3_comfy bundle); skips if absent.

import XCTest
import Logging
import MLX
import MLXNN
@testable import ZImage

final class LTX2VAERowBandTests: XCTestCase {
  // Heavyweight band-bug investigation probe (task #18): allocates tens of GB
  // and crashes the runner beside the resident warm server. Off by default;
  // set ZIMAGE_HEAVY_DIAGNOSTICS=1 to run.
  override func setUpWithError() throws {
    try XCTSkipUnless(
      ProcessInfo.processInfo.environment["ZIMAGE_HEAVY_DIAGNOSTICS"] == "1",
      "heavy diagnostic — set ZIMAGE_HEAVY_DIAGNOSTICS=1 to run")
  }


  private static var vae: LTX2VAE?

  private func loadVAE() throws -> LTX2VAE {
    if let v = Self.vae { return v }
    let path = NSHomeDirectory() + "/LocalModels/LTX23_video_vae_bf16.safetensors"
    guard FileManager.default.fileExists(atPath: path) else {
      throw XCTSkip("standalone video VAE not present at \(path)")
    }
    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: path))
    // Bare decoder.*/encoder.* keys -> the vae.* layout the loader expects,
    // mirroring the top-level stats into the decoder (as videoVAETensors does).
    var tensors: [String: MLXArray] = [:]
    for (k, v) in raw { tensors["vae." + k] = v }
    if let m = tensors["vae.per_channel_statistics.mean-of-means"] {
      tensors["vae.decoder.per_channel_statistics.mean"] = m
    }
    if let s = tensors["vae.per_channel_statistics.std-of-means"] {
      tensors["vae.decoder.per_channel_statistics.std"] = s
    }
    let vae = LTX2VAE(config: .v23)
    try LTX2WeightLoader.loadVAEWeightsFromTensors(
      into: vae, tensors: tensors, logger: Logger(label: "vae-rowband-test"))
    MLX.eval(vae.parameters())
    Self.vae = vae
    return vae
  }

  /// Synthetic clip with structure in every row: diagonal sinusoid + vertical
  /// stripes, drifting per frame. (B, 3, F, H, W) in [-1, 1].
  private func syntheticClip(frames: Int, height: Int, width: Int) -> MLXArray {
    let y = MLXArray(0..<height).asType(.float32).reshaped(1, 1, 1, height, 1)
    let x = MLXArray(0..<width).asType(.float32).reshaped(1, 1, 1, 1, width)
    let f = MLXArray(0..<frames).asType(.float32).reshaped(1, 1, frames, 1, 1)
    let base = MLX.sin(x * 0.13 + y * 0.07 + f * 0.5)
    let stripes = MLX.sin(x * 0.55) * 0.4
    let rgbPhase = MLXArray([Float(0), 1.1, 2.2]).reshaped(1, 3, 1, 1, 1)
    let clip = MLX.sin(base + stripes + rgbPhase) * 0.8
    return MLX.broadcast(clip, to: [1, 3, frames, height, width])
  }

  /// Per-row variance across (C, W) of the middle output frame, averaged into
  /// `blocks` row-blocks for a readable profile.
  private func rowVarianceProfile(_ video: MLXArray, blocks: Int) -> [Float] {
    let h = video.dim(3)
    let midF = video.dim(2) / 2
    let frame = video[0..., 0..., midF..<(midF + 1), 0..., 0...].asType(.float32)
    let mean = frame.mean(axes: [1, 4], keepDims: true)
    let variance = ((frame - mean) * (frame - mean)).mean(axes: [0, 1, 2, 4])  // (H,)
    let per = variance.reshaped(blocks, h / blocks).mean(axis: 1)
    return per.asArray(Float.self)
  }

  private func runRoundTrip(height: Int, label: String) throws -> [Float] {
    let vae = try loadVAE()
    let clip = syntheticClip(frames: 9, height: height, width: 768).asType(.bfloat16)
    let latent = vae.encode(clip)
    MLX.eval(latent)
    XCTAssertEqual(latent.dim(3), height / 32, "unexpected latent height")
    let decoded = vae.decode(latent)
    MLX.eval(decoded)
    let profile = rowVarianceProfile(decoded, blocks: height / 32)
    print("ROWVAR \(label): " + profile.map { String(format: "%.4f", $0) }.joined(separator: " "))
    return profile
  }

  func testRoundTripCleanAtControlHeight384() throws {
    let profile = try runRoundTrip(height: 384, label: "h384-latH12")
    let minV = profile.min()!, maxV = profile.max()!
    XCTAssertGreaterThan(minV, 0.1 * maxV,
      "row variance collapses at latH=12 — VAE broken even at the proven-clean height??")
  }

  func testRoundTripCleanAtRefineHeight512() throws {
    let profile = try runRoundTrip(height: 512, label: "h512-latH16")
    let minV = profile.min()!, maxV = profile.max()!
    XCTAssertGreaterThan(minV, 0.1 * maxV,
      "row variance collapses at latH=16 (refine height) — the two-stage band is a VAE decode bug")
  }

  /// The 2026-08-01 refine-band root cause: PLAIN decode corrupts at latent
  /// shape (1,128,13,16,28) — frame 0 wrap-duplicated, mid pixel frames flat —
  /// while the same latent decodes fine when latF is small (the tests above)
  /// or latH is 12 (production 193f/289f single-chunk). Volume does not sort
  /// failures from passes, so LTX2_PLAIN_DECODE_MAX_VOL guards the wrong axis.
  /// Suspect: MLX Metal int32-offset overflow (mlx #3836) in a big kernel launch.
  private func frameStd(_ video: MLXArray, _ f: Int) -> Float {
    let frame = video[0..., 0..., f..<(f + 1), 0..., 0...].asType(.float32)
    let m = frame.mean()
    return MLX.sqrt(((frame - m) * (frame - m)).mean()).item(Float.self)
  }

  private func runFailingShape(streamed: Bool, label: String) throws -> (mid: Float, first: Float, last: Float) {
    let vae = try loadVAE()
    let clip = syntheticClip(frames: 97, height: 512, width: 896).asType(.bfloat16)
    let latent = vae.encode(clip)
    MLX.eval(latent)
    XCTAssertEqual(latent.shape, [1, 128, 13, 16, 28], "unexpected latent shape")
    let decoded = streamed ? try vae.decodeStreamed(latent) : vae.decode(latent)
    MLX.eval(decoded)
    let pixF = decoded.dim(2)
    let s = (mid: frameStd(decoded, pixF / 2), first: frameStd(decoded, 0), last: frameStd(decoded, pixF - 1))
    print("SHAPE97 \(label): frame0 std \(s.first) mid std \(s.mid) last std \(s.last)")
    return s
  }

  func testPlainDecodeAtFailingShape97f512x896() throws {
    let s = try runFailingShape(streamed: false, label: "plain")
    XCTAssertGreaterThan(s.mid, 0.05,
      "mid frame flat — plain decode corrupts at (1,128,13,16,28)")
    XCTAssertGreaterThan(s.last, 0.05,
      "last frame flat — plain decode corrupts at (1,128,13,16,28)")
  }

  func testStreamedDecodeAtFailingShape97f512x896() throws {
    let s = try runFailingShape(streamed: true, label: "streamed")
    XCTAssertGreaterThan(s.mid, 0.05, "mid frame flat — streamed decode ALSO corrupts at this shape")
    XCTAssertGreaterThan(s.last, 0.05, "last frame flat — streamed decode ALSO corrupts at this shape")
    XCTAssertGreaterThan(s.first, 0.05, "frame 0 flat/mosaic — streamed decode frame-0 issue at this shape")
  }

  /// Walk the decoder stage by stage at the failing shape and report where the
  /// mid-frame structure collapses — that stage's kernel launch is the int32
  /// overflow site. Frame-axis std of the middle frame vs the first frame:
  /// healthy stages keep them comparable; the corrupting stage flattens mid.
  func testTapDecoderStagesAtFailingShape() throws {
    let vae = try loadVAE()
    let clip = syntheticClip(frames: 97, height: 512, width: 896).asType(.bfloat16)
    let latent = vae.encode(clip)
    MLX.eval(latent)

    func midStd(_ x: MLXArray, _ name: String) {
      // x: (B, C, F, H, W) — the decoder's public channels-first layout
      let f = x.dim(2) / 2
      func stdOf(_ fr: Int) -> Float {
        let frame = x[0..., 0..., fr..<(fr + 1), 0..., 0...].asType(.float32)
        let m = frame.mean()
        return MLX.sqrt(((frame - m) * (frame - m)).mean()).item(Float.self)
      }
      print("TAP \(name) shape \(x.shape): f0 std \(stdOf(0)) mid std \(stdOf(f))")
    }

    let dec = vae.decoder
    // Mirror LTX2Decoder3D.callAsFunction for v23 (no timestep conditioning):
    // channels-first end to end; the conv wrappers transpose internally.
    var x = latent
    x = dec.perChannelStatistics.unNormalize(x)
    x = dec.convIn(x)
    MLX.eval(x); midStd(x, "conv_in")
    let sortedKeys = dec.upBlocks.keys.sorted { Int($0)! < Int($1)! }
    for key in sortedKeys {
      let block = dec.upBlocks[key]!
      if let resGroup = block as? LTX2DecoderResBlockGroup {
        x = resGroup(x, timestep: nil)
      } else if let upsample = block as? LTX2DepthToSpaceUpsample {
        x = upsample(x)
      }
      MLX.eval(x); midStd(x, "block\(key) (\(type(of: block)))")
    }

    // Tail: pixelNorm (replicated — private in the decoder) -> SiLU -> conv_out
    // -> unpatchify. If these stay clean too, the corruption only occurs when
    // the whole decode graph evaluates in ONE eval() — i.e. a fusion/scheduling
    // fault, not a single op — which is also why chunk-wise streamed decode
    // and this per-stage tap are both clean.
    x = x / MLX.sqrt(MLX.mean(x * x, axis: 1, keepDims: true) + 1e-8)
    MLX.eval(x); midStd(x, "pixelNorm")
    x = silu(x)
    x = dec.convOut(x)
    MLX.eval(x); midStd(x, "conv_out")
    x = LTX2Patchify.unpatchify(x, patchSizeHW: 4, patchSizeT: 1)
    MLX.eval(x); midStd(x, "unpatchify")

    let f = x.dim(2) / 2
    let frame = x[0..., 0..., f..<(f + 1), 0..., 0...].asType(.float32)
    let m = frame.mean()
    let sd = MLX.sqrt(((frame - m) * (frame - m)).mean()).item(Float.self)
    XCTAssertGreaterThan(
      sd, 0.05,
      "mid frame flat even with per-stage eval — corruption is in a tail op, not whole-graph scheduling")
  }
}
