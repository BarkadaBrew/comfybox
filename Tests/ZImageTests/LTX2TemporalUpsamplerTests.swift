// LTX2TemporalUpsamplerTests.swift — the ltx-2.3-temporal-upscaler-x2 port
// (2026-09-13 design: docs/superpowers/specs/2026-09-13-ltx-temporal-upscaler-design.md).
//
// 1. Temporal pixel-shuffle ordering (pure).
// 2. Checkpoint config -> mode/width (pure).
// 3. Loader binds every tensor of the real checkpoint; 5 latent frames -> 9.
// 4. Parity vs the PyTorch reference at every tap (fixture pinned in
//    Tests/ZImageTests/Fixtures, fp32, torch.manual_seed(0) randn(1,128,5,8,8)).
// 3/4 skip cleanly when the weights are absent.

import XCTest
import Logging
import MLX
import MLXNN
import MLXRandom
@testable import ZImage

final class LTX2TemporalUpsamplerTests: XCTestCase {

  static let checkpoint = "\(NSHomeDirectory())/LocalModels/ltx2-upsampler/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"
  /// PyTorch taps live next to the weights (4 MB; override with LTX2_TEMPORAL_UPSAMPLER_TAPS).
  static var fixture: URL {
    let env = ProcessInfo.processInfo.environment["LTX2_TEMPORAL_UPSAMPLER_TAPS"]
    return URL(fileURLWithPath: env ?? "\(NSHomeDirectory())/LocalModels/ltx2-upsampler/ltx2-temporal-upsampler-taps.safetensors")
  }

  func testTemporalShuffleInterleavesChannelPairsIntoFramePairs() {
    // (N=1, D=2, H=1, W=1, 2C=4): channel value = 10*d + ch, so the mapping is legible.
    let vals: [Float] = [0, 1, 2, 3, 10, 11, 12, 13]
    let x = MLXArray(vals).reshaped(1, 2, 1, 1, 4)
    let y = LTX2TemporalUpsampler2x.temporalShuffle(x)
    XCTAssertEqual(y.shape, [1, 4, 1, 1, 2])
    // einops "(c p)" is c-major: input channel 2c+p -> output frame 2d+p, channel c.
    // frame 0 <- d0,p0 = ch {0,2}; frame 1 <- d0,p1 = ch {1,3}; frame 2 <- d1,p0 = {10,12}; frame 3 <- {11,13}
    XCTAssertEqual(y.reshaped(8).asArray(Float.self), [0, 2, 1, 3, 10, 12, 11, 13])
  }

  func testCheckpointConfigSelectsTheVariant() {
    let temporal = LTX2LatentUpsampler.modeFromCheckpointConfig(
      #"{"_class_name": "LatentUpsampler", "in_channels": 128, "mid_channels": 512, "num_blocks_per_stage": 4, "dims": 3, "spatial_upsample": false, "temporal_upsample": true}"#)
    XCTAssertEqual(temporal?.mode, .temporal2x); XCTAssertEqual(temporal?.midChannels, 512)
    let spatial = LTX2LatentUpsampler.modeFromCheckpointConfig(#"{"spatial_upsample": true, "mid_channels": 1024}"#)
    XCTAssertEqual(spatial?.mode, .spatial2x); XCTAssertEqual(spatial?.midChannels, 1024)
    XCTAssertEqual(LTX2LatentUpsampler.modeFromCheckpointConfig(nil)?.mode, .spatial2x, "no metadata = the spatial checkpoint this loader always knew")
    XCTAssertNil(LTX2LatentUpsampler.modeFromCheckpointConfig(#"{"spatial_upsample": true, "temporal_upsample": true}"#), "spatiotemporal is not ported")
  }

  func testTemporalModuleDoublesFramesMinusOne() {
    let up = LTX2LatentUpsampler(midChannels: 64, mode: .temporal2x)
    let x = MLXRandom.normal([1, 128, 5, 4, 4])
    let y = up(x)
    XCTAssertEqual(y.shape, [1, 128, 9, 4, 4])
  }

  private func loadReal() throws -> LTX2LatentUpsampler {
    guard FileManager.default.fileExists(atPath: Self.checkpoint) else {
      throw XCTSkip("temporal upscaler weights absent: \(Self.checkpoint)")
    }
    let logger = Logger(label: "temporal-upsampler-test")
    guard let up = LTX2VideoGenerator.loadUpsampler(path: Self.checkpoint, logger: logger) else {
      XCTFail("loadUpsampler refused the real temporal checkpoint (a partial bind renders a mesh)")
      throw XCTSkip("unbound")
    }
    XCTAssertEqual(up.mode, .temporal2x); XCTAssertEqual(up.midChannels, 512)
    return up
  }

  func testRealCheckpointBindsAndUpsamplesTime() throws {
    let up = try loadReal()
    let x = MLXRandom.normal([1, 128, 5, 8, 8])
    let y = up(x)
    XCTAssertEqual(y.shape, [1, 128, 9, 8, 8])
    XCTAssertTrue(y.asType(.float32).abs().max().item(Float.self).isFinite)
  }

  func testParityAgainstTorchReferenceAtEveryTap() throws {
    let up = try loadReal()
    guard FileManager.default.fileExists(atPath: Self.fixture.path) else { throw XCTSkip("fixture absent") }
    let taps = try MLX.loadArrays(url: Self.fixture)
    func relErr(_ a: MLXArray, _ b: MLXArray) -> Float {
      let d = MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
      return d / max(MLX.abs(b.asType(.float32)).max().item(Float.self), 1e-6)
    }
    // Reference tensors are (B, C, F, H, W); walk the module channels-last exactly as callAsFunction does.
    var h = taps["x"]!.asType(.float32).transposed(0, 2, 3, 4, 1)
    h = silu(up.initialNorm(up.initialConv(h)))
    XCTAssertLessThan(relErr(h.transposed(0, 4, 1, 2, 3), taps["t0"]!), 2e-2, "t0 initial conv/norm/silu")
    for b in up.resBlocks { h = b(h) }
    XCTAssertLessThan(relErr(h.transposed(0, 4, 1, 2, 3), taps["t1"]!), 3e-2, "t1 res blocks")
    h = up.temporalStage!(h)
    XCTAssertEqual(h.shape[1], 9)
    XCTAssertLessThan(relErr(h.transposed(0, 4, 1, 2, 3), taps["t2"]!), 3e-2, "t2 temporal upsample + first-frame drop")
    for b in up.postResBlocks { h = b(h) }
    XCTAssertLessThan(relErr(h.transposed(0, 4, 1, 2, 3), taps["t3"]!), 4e-2, "t3 post res blocks")
    h = up.finalConv(h)
    XCTAssertLessThan(relErr(h.transposed(0, 4, 1, 2, 3), taps["out"]!), 4e-2, "final conv")
    // And the public entry point end to end.
    let out = up(taps["x"]!.asType(.float32))
    XCTAssertLessThan(relErr(out, taps["out"]!), 4e-2, "callAsFunction end to end")
  }
}

// MARK: - Codex review 2026-09-13: conditioning fps follows the request's generation fps

extension LTX2TemporalUpsamplerTests {
  func testConditioningFpsDefaultsToRequestFpsWithCondFpsAsOverride() {
    XCTAssertEqual(LTX2Pipeline.conditioningFps(latF: 19, condFps: nil, requestFps: 30, configFps: 24), 30, "a 30 fps request conditions at 30")
    XCTAssertEqual(LTX2Pipeline.conditioningFps(latF: 19, condFps: nil, requestFps: nil, configFps: 24), 24, "no request fps → pipeline default")
    XCTAssertEqual(LTX2Pipeline.conditioningFps(latF: 19, condFps: 12, requestFps: 30, configFps: 24), 12, "cond_fps stays the explicit motion dial")
    XCTAssertEqual(LTX2Pipeline.conditioningFps(latF: 1, condFps: 12, requestFps: 30, configFps: 24), 1, "a still is conditioned at 1")
  }
}
