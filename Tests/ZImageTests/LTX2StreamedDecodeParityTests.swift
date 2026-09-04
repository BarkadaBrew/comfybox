// LTX2StreamedDecodeParityTests.swift — the streamed (chunked-io) decode must
// be numerically IDENTICAL to a single full-tensor decode. This is the whole
// point of #36: every seam-based approximation (spatial tiles, blended
// temporal windows) measurably costs sharpness/stability, and plain decode of
// large tensors silently corrupts (MLX Metal int32-offset overflow, mlx
// #3836). Streaming carries per-conv temporal state between chunks instead.
//
// Uses the real v2.3 decoder architecture (non-causal convs, DepthToSpace
// temporal x2, PixelNorm res blocks) with random weights on a small latent —
// no model assets needed, runs everywhere.

import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import ZImage

final class LTX2StreamedDecodeParityTests: XCTestCase {

  private func makeRandomDecoder() -> LTX2Decoder3D {
    let decoder = LTX2Decoder3D(config: .v23)
    MLXRandom.seed(11)
    // Randomize conv weights (module init is zeros -> outputs would be
    // trivially equal). Small scale keeps activations bounded through the
    // PixelNorm stack.
    for m in decoder.modules() {
      if let conv = m as? CausalConv3d {
        conv.weight = MLXRandom.normal(conv.weight.shape) * 0.05
        conv.bias = MLXRandom.normal(conv.bias.shape) * 0.01
      }
    }
    MLX.eval(decoder.parameters())
    return decoder
  }

  func testStreamedDecodeMatchesPlainDecode() throws {
    let decoder = makeRandomDecoder()

    MLXRandom.seed(23)
    // 5 latent frames -> 3 chunks at chunk size 2 (2+2+1), exercising first
    // chunk (replicate pad), steady state, and the final flush.
    let latent = MLXRandom.normal([1, 128, 5, 4, 4]).asType(.float32)

    let plain = decoder(latent)
    MLX.eval(plain)
    let streamed = try decoder.decodeStreamed(latent)
    MLX.eval(streamed)

    XCTAssertEqual(plain.shape, streamed.shape,
                   "streamed decode changed the output shape")

    let diff = MLX.abs(plain - streamed).max().item(Float.self)
    let scale = MLX.abs(plain).max().item(Float.self)
    let rel = diff / max(scale, 1e-6)
    print("PARITY streamed-vs-plain: maxAbs \(diff) rel \(rel) shape \(plain.shape)")
    XCTAssertLessThan(rel, 1e-4, "streamed decode diverges from plain decode")
  }

  func testStreamedDecodeChunkSizeOne() throws {
    let decoder = makeRandomDecoder()
    MLXRandom.seed(29)
    let latent = MLXRandom.normal([1, 128, 4, 3, 3]).asType(.float32)

    let plain = decoder(latent)
    let streamed = try decoder.decodeStreamed(latent)
    MLX.eval(plain, streamed)

    XCTAssertEqual(plain.shape, streamed.shape)
    let rel = MLX.abs(plain - streamed).max().item(Float.self)
      / max(MLX.abs(plain).max().item(Float.self), 1e-6)
    print("PARITY chunk-1: rel \(rel)")
    XCTAssertLessThan(rel, 1e-4)
  }

  func testStreamStateFullyResetsBetweenDecodes() throws {
    let decoder = makeRandomDecoder()
    MLXRandom.seed(31)
    let latent = MLXRandom.normal([1, 128, 5, 4, 4]).asType(.float32)

    // Two consecutive streamed decodes of the same input must agree — stale
    // stream state (conv caches, drop-first flags, skip queues) would skew
    // the second run.
    let first = try decoder.decodeStreamed(latent)
    let second = try decoder.decodeStreamed(latent)
    MLX.eval(first, second)

    let rel = MLX.abs(first - second).max().item(Float.self)
      / max(MLX.abs(first).max().item(Float.self), 1e-6)
    XCTAssertLessThan(rel, 1e-6, "stream state leaked between decodes")
  }
}
