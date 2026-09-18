import Foundation
import MLX
import XCTest

@testable import ZImage

/// `encodeToLatent` must be the exact inverse of `decodeToMel` (FDD §4.7, WP11).
///
/// Nothing called the encode side until audio-driven chunks did, and it had
/// drifted: it put FREQUENCY on the causal height axis where the decoder puts
/// TIME, and it returned an un-normalized posterior mean where the decoder
/// expects a normalized latent. The result patchified to a 1208-wide token, and
/// MLX answers a bad `addmm` with `fatalError` — the whole engine went down
/// mid-render rather than the one job failing.
///
/// These run on an unloaded module: the shapes are a property of the
/// architecture, not of the weights, and that is exactly what broke.
final class LTX2AudioVAERoundTripTests: XCTestCase {

  private func vae() -> LTX2AudioVAE { LTX2AudioVAE() }

  func testEncodeDecodeRoundTripKeepsLayout() {
    let latent = MLX.zeros([1, 8, 10, 16], dtype: .float32)
    let mel = vae().decodeToMel(latent)
    XCTAssertEqual(mel.shape, [1, 2, 37, 64], "(B, 2, 4T-3, melBins)")

    let back = vae().encodeToLatent(mel)
    XCTAssertEqual(back.dim(0), 1)
    XCTAssertEqual(back.dim(1), 8, "channels — the axis the crash proved was not being kept")
    XCTAssertEqual(back.dim(3), 16, "features")
    // Measured, not assumed: decode is 4T-3 and encode is floor(m/4), so a
    // round trip comes back one frame short — the causal encoder's warm-up.
    // Chunk windows are sized in LATENT frames (`latentFrames`) precisely so
    // this never shows up as a gap at the end of a chunk.
    XCTAssertEqual(back.dim(2), latent.dim(2) - 1)
  }

  func testTheMelWindowForAChunkLandsExactlyOnTheAudioStream() {
    // 145 frames at 24 fps is 6.04s -> ta = 152 latent frames. The window the
    // generator asks for must encode to exactly 152, not 151.
    let seconds = 145.0 / 24.0
    let ta = LTX2MelAnalysis.latentFrames(seconds: seconds)
    XCTAssertEqual(ta, 152)
    let samples = LTX2MelAnalysis.sampleCount(frames: 4 * ta)
    let melFrames = LTX2MelAnalysis.frameCount(samples: samples)
    let latent = vae().encodeToLatent(MLX.zeros([1, 2, melFrames, 64], dtype: .float32))
    XCTAssertEqual(
      latent.dim(2), ta,
      "the supplied voice must span the whole chunk, with no silence padded on the end")
  }

  func testTimeIsTheCausalAxisOnTheWayIn() {
    // A mel twice as long in TIME must produce a latent twice as long in
    // dim(2) — not dim(3). This is the assertion the old encode failed.
    let short = vae().encodeToLatent(MLX.zeros([1, 2, 37, 64], dtype: .float32))
    let long = vae().encodeToLatent(MLX.zeros([1, 2, 77, 64], dtype: .float32))
    XCTAssertEqual(short.dim(3), long.dim(3), "the feature axis is fixed by the mel bins")
    XCTAssertEqual(short.dim(1), 8)
    XCTAssertGreaterThan(long.dim(2), short.dim(2), "more audio is more frames, not more features")
  }

  func testTheEncodedLatentPatchifiesToTheWidthTheTransformerProjects() {
    // 8 channels x 16 features = the 128-wide audio token. The crash was this
    // number coming out as 1208.
    let latent = vae().encodeToLatent(MLX.zeros([1, 2, 37, 64], dtype: .float32))
    let (tokens, _) = LTX2AudioPatchifier.patchify(latent)
    XCTAssertEqual(tokens.dim(2), 128)
  }
}
