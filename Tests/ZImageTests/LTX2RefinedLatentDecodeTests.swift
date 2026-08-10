// LTX2RefinedLatentDecodeTests.swift — decode the ACTUAL refined latent dumped
// from the failing two-stage render (2026-08-01) through the real video VAE,
// offline. The refine-band render produced: frame 0 vertically wrap-duplicated
// at ~60% height, mid frames entirely flat — while the refined latent itself
// is statistically healthy per frame. This test decides whether the fault is
// the VAE decode at this exact shape (1,128,13,16,28) or the post-decode
// pipeline (rescale/PostProcess/AVAssetWriter).
//
// Needs ~/LocalModels/LTX23_video_vae_bf16.safetensors and the dump at
// $LTX2_REFINED_DUMP (default: the session scratchpad refine-dump/refined.npy).

import XCTest
import Logging
import MLX
import MLXNN
@testable import ZImage

final class LTX2RefinedLatentDecodeTests: XCTestCase {

  func testDecodeDumpedRefinedLatent() throws {
    let env = ProcessInfo.processInfo.environment
    let home = NSHomeDirectory()
    let vaePath = home + "/LocalModels/LTX23_video_vae_bf16.safetensors"
    let dumpPath = env["LTX2_REFINED_DUMP"]
      ?? "/private/tmp/claude-501/-Users-toddwalderman-Projects-zimage-swift/818bd225-e43c-4387-9594-bbb87c47538f/scratchpad/refine-dump/refined.npy"
    guard FileManager.default.fileExists(atPath: vaePath),
          FileManager.default.fileExists(atPath: dumpPath) else {
      throw XCTSkip("VAE weights or refined-latent dump not present")
    }

    let raw = try MLX.loadArrays(url: URL(fileURLWithPath: vaePath))
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
      into: vae, tensors: tensors, logger: Logger(label: "refined-decode-test"))
    MLX.eval(vae.parameters())

    let latent = try MLX.loadArray(url: URL(fileURLWithPath: dumpPath)).asType(.bfloat16)
    print("DECODE input shape \(latent.shape)")
    let useStreamed = env["LTX2_TEST_STREAMED"] == "1"
    let decoded = (useStreamed ? vae.decodeStreamed(latent) : vae.decode(latent)).asType(.float32)
    MLX.eval(decoded)
    print("DECODE output shape \(decoded.shape) (\(useStreamed ? "streamed" : "plain"))")

    // Per-frame std over (C,H,W) for a set of probe pixel frames.
    let pixF = decoded.dim(2)
    let probes = [0, pixF / 2, pixF - 1]
    for f in probes {
      let frame = decoded[0..., 0..., f..<(f + 1), 0..., 0...]
      let m = frame.mean()
      let sd = MLX.sqrt(((frame - m) * (frame - m)).mean()).item(Float.self)
      // Row variance profile in 16 blocks
      let h = decoded.dim(3)
      let mean = frame.mean(axes: [1, 4], keepDims: true)
      let rowVar = ((frame - mean) * (frame - mean)).mean(axes: [0, 1, 2, 4])
      let blocks = rowVar.reshaped(16, h / 16).mean(axis: 1).asArray(Float.self)
      print("FRAME \(f): std \(String(format: "%.4f", sd)) rowvar "
        + blocks.map { String(format: "%.4f", $0) }.joined(separator: " "))
      XCTAssertGreaterThan(sd, 0.05, "decoded frame \(f) is flat — decode reproduces the corruption")
    }

    // Wrap detection on frame 0: correlation between top 60% and the rest.
    if let dir = env["LTX2_DECODE_TEST_OUT"] {
      try? MLX.save(array: decoded[0..., 0..., 0..<1, 0..., 0...],
                    url: URL(fileURLWithPath: dir).appendingPathComponent("decoded-f0.npy"))
      try? MLX.save(array: decoded[0..., 0..., (pixF / 2)..<(pixF / 2 + 1), 0..., 0...],
                    url: URL(fileURLWithPath: dir).appendingPathComponent("decoded-mid.npy"))
    }
  }
}
