// LTX2ConvOutOverflowProbeTests.swift — minimal pure-MLX reproduction of the
// conv_out corruption behind the two-stage refine band (2026-08-01).
//
// The decoder's final conv (CausalConv3d 128 -> 48, k=3, non-causal) corrupts
// at input (1,128,97,128,224) BCTHW even though CausalConv3d already chunks
// the temporal axis at 64 frames to dodge the MLX Metal large-launch bug
// (ml-explore/mlx #3836/#3609/#3524) — 64-frame chunks are still too large at
// this spatial slab. Meanwhile the SAME-shaped 128->128 res convs in block8
// are clean, so chunk size, C_out, and dtype all need mapping.
//
// This probe calls convGeneral exactly the way the wrapper does (temporal
// padding pre-applied by frame replication, conv padding [0,1,1]) and
// compares single launches of varying temporal length against a chunk-8
// reference (exact by construction; chunk-8 with halo verified self-consistent
// at small sizes).

import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import ZImage

final class LTX2ConvOutOverflowProbeTests: XCTestCase {
  // Heavyweight band-bug investigation probe (task #18): allocates tens of GB
  // and crashes the runner beside the resident warm server. Off by default;
  // set ZIMAGE_HEAVY_DIAGNOSTICS=1 to run.
  override func setUpWithError() throws {
    try XCTSkipUnless(
      ProcessInfo.processInfo.environment["ZIMAGE_HEAVY_DIAGNOSTICS"] == "1",
      "heavy diagnostic — set ZIMAGE_HEAVY_DIAGNOSTICS=1 to run")
  }


  /// Reference: conv over [B, T, H, W, C] chunked along T by 8 with k-1 halo,
  /// conv padding [0, 1, 1]. Input must already carry its temporal padding.
  private func chunkedRef(_ x: MLXArray, _ w: MLXArray, chunk: Int = 8) -> MLXArray {
    let t = x.dim(1)
    let kt = w.dim(1)
    var parts: [MLXArray] = []
    var outStart = 0  // first output frame this chunk must produce
    let outTotal = t - (kt - 1)
    while outStart < outTotal {
      let outEnd = min(outStart + chunk, outTotal)
      // output frame i needs input frames [i, i+kt-1]
      let inLo = outStart
      let inHi = outEnd + (kt - 1)
      let slice = x[0..., inLo..<inHi, 0..., 0..., 0...]
      let y = convGeneral(slice, w, strides: IntOrArray([1, 1, 1]),
                          padding: IntOrArray([0, 1, 1]))
      parts.append(y)
      outStart = outEnd
    }
    return MLX.concatenated(parts, axis: 1)
  }

  private func fullLaunch(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
    convGeneral(x, w, strides: IntOrArray([1, 1, 1]), padding: IntOrArray([0, 1, 1]))
  }

  /// relDiff of a single full launch of `frames` output frames vs the chunk-8
  /// reference, at (h, w, cIn->cOut) in `dtype`.
  private func probe(frames: Int, h: Int, w: Int, cIn: Int, cOut: Int,
                     dtype: DType) -> Float {
    MLXRandom.seed(7)
    let kt = 3
    // frames output frames need frames + kt - 1 input frames
    let x = MLXRandom.normal([1, frames + kt - 1, h, w, cIn]).asType(dtype)
    let wt = (MLXRandom.normal([cOut, kt, 3, 3, cIn]) * 0.05).asType(dtype)
    let full = fullLaunch(x, wt)
    let ref = chunkedRef(x, wt)
    MLX.eval(full, ref)
    XCTAssertEqual(full.shape, ref.shape)
    let diff = MLX.abs(full.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
    let scale = MLX.abs(ref.asType(.float32)).max().item(Float.self)
    let rel = diff / max(scale, 1e-6)
    print(String(format: "PROBE T%d h%d w%d %d->%d %@: relDiff %.5f  (in %.0fM elems)",
                 frames, h, w, cIn, cOut, "\(dtype)" as NSString, rel,
                 Double((frames + 2) * h * w * cIn) / 1e6))
    return rel
  }

  func testConvLaunchEnvelope() throws {
    var rows: [(String, Float)] = []
    // Production conv_out slab (128x224, cin 128), bf16 — find the safe T.
    for t in [64, 48, 36, 32, 24, 16] {
      rows.append(("bf16 cout48 T\(t)", probe(frames: t, h: 128, w: 224, cIn: 128, cOut: 48, dtype: .bfloat16)))
    }
    // Same slab, cout 128 (block8 shape) and f32 — explain why block8 is clean.
    rows.append(("bf16 cout128 T64", probe(frames: 64, h: 128, w: 224, cIn: 128, cOut: 128, dtype: .bfloat16)))
    rows.append(("f32  cout48  T64", probe(frames: 64, h: 128, w: 224, cIn: 128, cOut: 48, dtype: .float32)))
    rows.append(("f32  cout128 T64", probe(frames: 64, h: 128, w: 224, cIn: 128, cOut: 128, dtype: .float32)))

    for (name, rel) in rows {
      print("RESULT \(name): \(rel < 1e-2 ? "OK" : "CORRUPT")")
    }
  }

  /// Same comparison but with inputs fully materialized BEFORE the conv and the
  /// full launch evaluated ALONE — rules scheduling/fusion in or out — plus a
  /// tighter T bisect between the known-good 36 and known-bad 48.
  private func probeIsolated(frames: Int, h: Int, w: Int, cIn: Int, cOut: Int,
                             dtype: DType) -> Float {
    MLXRandom.seed(7)
    let kt = 3
    let x = MLXRandom.normal([1, frames + kt - 1, h, w, cIn]).asType(dtype)
    let wt = (MLXRandom.normal([cOut, kt, 3, 3, cIn]) * 0.05).asType(dtype)
    MLX.eval(x, wt)                        // inputs materialized first
    let full = fullLaunch(x, wt)
    MLX.eval(full)                          // full launch alone in its own eval
    let ref = chunkedRef(x, wt)
    MLX.eval(ref)
    let diff = MLX.abs(full.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
    let scale = MLX.abs(ref.asType(.float32)).max().item(Float.self)
    let rel = diff / max(scale, 1e-6)
    print(String(format: "ISOPROBE T%d h%d w%d %d->%d %@: relDiff %.5f (in %.0fM)",
                 frames, h, w, cIn, cOut, "\(dtype)" as NSString, rel,
                 Double((frames + 2) * h * w * cIn) / 1e6))
    return rel
  }

  func testIsolatedLaunchBisect() throws {
    for t in [38, 40, 42, 44, 46, 48, 64] {
      _ = probeIsolated(frames: t, h: 128, w: 224, cIn: 128, cOut: 48, dtype: .bfloat16)
    }
    // Scale check: quarter slab — does the boundary track input elements?
    for t in [160, 180, 200] {
      _ = probeIsolated(frames: t, h: 64, w: 112, cIn: 128, cOut: 48, dtype: .bfloat16)
    }
  }

  /// Drive the ACTUAL production wrapper (CausalConv3d, non-causal v23 config)
  /// at the conv_out shape and compare against itself on a safely small input:
  /// splits the clip into halves, runs each through a fresh wrapper (replicate
  /// padding handled by feeding overlap frames and trimming), and diffs.
  func testProductionWrapperAtConvOutShape() throws {
    MLXRandom.seed(11)
    let t = 97, h = 128, w = 224, cIn = 128, cOut = 48
    let conv = CausalConv3d(
      inChannels: cIn, outChannels: cOut,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1),
      causalTemporal: false)
    conv.weight = (MLXRandom.normal(conv.weight.shape) * 0.05).asType(.bfloat16)
    conv.bias = (MLXRandom.normal(conv.bias.shape) * 0.01).asType(.bfloat16)
    let x = MLXRandom.normal([1, cIn, t, h, w]).asType(.bfloat16)
    MLX.eval(x, conv.weight, conv.bias)

    let full = conv(x)
    MLX.eval(full)

    // Reference: run the wrapper on windows of 20 frames with 2-frame overlap
    // on each side; interior windows get correct context, and we trim the
    // replicate-padded edges (1 frame each side of each window's output).
    var parts: [MLXArray] = []
    let win = 20
    var s = 0
    while s < t {
      let e = min(s + win, t)
      let lo = max(0, s - 1), hi = min(t, e + 1)
      let y = conv(x[0..., 0..., lo..<hi, 0..., 0...])
      let trimLo = s - lo, trimHi = y.dim(2) - (hi - e)
      parts.append(y[0..., 0..., trimLo..<trimHi, 0..., 0...])
      s = e
    }
    let ref = MLX.concatenated(parts, axis: 2)
    MLX.eval(ref)

    XCTAssertEqual(full.shape, ref.shape)
    let diff = MLX.abs(full.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
    let scale = MLX.abs(ref.asType(.float32)).max().item(Float.self)
    let rel = diff / max(scale, 1e-6)
    print("WRAPPER conv_out-shape relDiff \(rel)")
    XCTAssertLessThan(rel, 1e-2,
      "CausalConv3d's own 64-frame chunking still corrupts at the conv_out slab")
  }
}
