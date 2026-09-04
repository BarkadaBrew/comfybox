import Foundation
import MLX
import XCTest

@testable import ZImage

/// RES4LYF `fractal` / `pyramid` spatial noise generators
/// (`beta/noise_classes.py`, pinned `26036f64…`), as opt-in alternatives to the
/// gaussian SDE noise stream.
///
/// The load-bearing invariant is that `fractal` with `alpha == 0` is
/// byte-identical to the gaussian stream (flat spectrum ⇒ identity), so turning
/// the option on with the neutral parameter changes nothing. The spatial
/// transforms are then checked for the z-score invariants every draw carries and
/// for being genuinely different from a plain gaussian draw.
final class RES4LYFSpatialNoiseTests: XCTestCase {

  // Krea 2's SDE layout: (1, tokens, C·p·p) with C = 16, p = 2, an 8×8 token
  // grid ⇒ trailing = 64, tokens = 64, latH = latW = 16.
  private let layout = RES4LYFNoiseLayout.patchifiedTrailing(channels: 16)
  private let hTok = 8
  private let wTok = 8
  private let channels = 16
  private var sampleShape: [Int] { [1, hTok * wTok, channels * 2 * 2] }

  /// Per-channel `(mean, ddof=1 std)` for a patchified `(1, tokens, C·p²)` draw
  /// — the population upstream's `normalize_zscore(channelwise=True)` normalises.
  private func channelStats(_ x: MLXArray) -> (means: [Float], stds: [Float]) {
    let last = x.dim(x.ndim - 1)
    let per = last / channels
    let r = x.reshaped(1, x.dim(1), channels, per)
    let means = r.mean(axes: [1, 3]).asArray(Float.self)
    let stds = MLX.std(r, axes: [1, 3], ddof: 1).asArray(Float.self)
    return (means, stds)
  }

  private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a - b).max().item(Float.self)
  }

  // MARK: - Fractal

  func testFractalAlphaZeroEqualsGaussian() {
    let sample = MLXArray.zeros(sampleShape)
    let seed: UInt64 = 4243

    let gaussian = RES4LYFGaussianNoiseStream(seed: seed, layout: layout)
    let fractal = RES4LYFFractalNoiseStream(
      seed: seed, layout: layout, hTok: hTok, wTok: wTok, channels: channels, alpha: 0.0)

    let g = gaussian.next(like: sample)
    let f = fractal.next(like: sample)

    XCTAssertEqual(f.shape, g.shape)
    // Byte-identical short-circuit: same seed, same key chain, flat spectrum.
    XCTAssertEqual(maxAbsDiff(g, f), 0.0, "fractal(alpha=0) must equal the gaussian draw exactly")
  }

  func testFractalNonzeroAlphaDiffersAndKeepsZscoreInvariants() {
    let sample = MLXArray.zeros(sampleShape)
    let seed: UInt64 = 4243

    let gaussian = RES4LYFGaussianNoiseStream(seed: seed, layout: layout)
    let fractal = RES4LYFFractalNoiseStream(
      seed: seed, layout: layout, hTok: hTok, wTok: wTok, channels: channels, alpha: 1.0)

    let g = gaussian.next(like: sample)
    let f = fractal.next(like: sample)

    XCTAssertEqual(f.shape, sampleShape)
    XCTAssertGreaterThan(
      maxAbsDiff(g, f), 1e-3, "fractal(alpha=1) must differ from the gaussian draw")

    XCTAssertTrue(f.asArray(Float.self).allSatisfy { $0.isFinite }, "fractal output must be finite")

    let (means, stds) = channelStats(f)
    for m in means { XCTAssertEqual(m, 0.0, accuracy: 1e-4, "per-channel mean ≈ 0") }
    for sd in stds { XCTAssertEqual(sd, 1.0, accuracy: 1e-3, "per-channel ddof=1 std ≈ 1") }
  }

  // MARK: - Pyramid

  func testPyramidIsFiniteRightShapeZscoredAndNotGaussian() {
    let sample = MLXArray.zeros(sampleShape)
    let seed: UInt64 = 4243

    let pyramid = RES4LYFPyramidNoiseStream(
      seed: seed, layout: layout, hTok: hTok, wTok: wTok, channels: channels)
    let p = pyramid.next(like: sample)

    XCTAssertEqual(p.shape, sampleShape)
    XCTAssertTrue(p.asArray(Float.self).allSatisfy { $0.isFinite }, "pyramid output must be finite")

    let (means, stds) = channelStats(p)
    for m in means { XCTAssertEqual(m, 0.0, accuracy: 1e-4, "per-channel mean ≈ 0") }
    for sd in stds { XCTAssertEqual(sd, 1.0, accuracy: 1e-3, "per-channel ddof=1 std ≈ 1") }

    // The multi-octave sum is not a single gaussian draw.
    let gaussian = RES4LYFGaussianNoiseStream(seed: seed, layout: layout)
    let g = gaussian.next(like: sample)
    XCTAssertGreaterThan(
      maxAbsDiff(g, p), 1e-3, "pyramid must differ from a single gaussian draw")
  }

  // MARK: - nearest-exact downsample

  func testNearestExactDownsamplePicksIndices() {
    // 4×4 arange, factor r = 2 → picks source index i·r + r/2 = {1, 3} on each
    // axis: rows {1,3} × cols {1,3} of
    //   0  1  2  3
    //   4  5  6  7
    //   8  9 10 11
    //  12 13 14 15
    // ⇒ [[5, 7], [13, 15]].
    let base = MLXArray((0..<16).map { Float($0) }, [1, 1, 4, 4])
    let down = RES4LYFPyramidNoiseStream.nearestExactDownsample(base, toH: 2, toW: 2)

    XCTAssertEqual(down.shape, [1, 1, 2, 2])
    XCTAssertEqual(down.asArray(Float.self), [5, 7, 13, 15])
  }

  func testFftFreqMatchesTorch() {
    // torch.fft.fftfreq(n, 1/n): even n = 4 → [0, 1, -2, -1]; odd n = 5 → [0,1,2,-2,-1].
    XCTAssertEqual(RES4LYFFractalNoiseStream.fftFreq(4), [0, 1, -2, -1])
    XCTAssertEqual(RES4LYFFractalNoiseStream.fftFreq(5), [0, 1, 2, -2, -1])
  }
}
