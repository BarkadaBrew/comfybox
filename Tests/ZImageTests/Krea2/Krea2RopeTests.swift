import XCTest
import MLX
@testable import ZImage

/// Krea-2 has its own RoPE, separate from ZImageRopeEmbedder, which is why DyPE
/// never applied to krea2 renders. These cover the per-axis NTK scaling that
/// closes that gap. Pure MLX — no model weights, so this is cheap to run.
final class Krea2RopeTests: XCTestCase {

    /// Krea2Config.axes for the shipping model: 3 axes summing to headDim.
    private let axes = [40, 24, 24]
    private let theta: Float = 1000.0

    /// One text token at the origin, one image token at (row 8, col 8) —
    /// the shape Krea2Sampling.buildPositions produces.
    private func positions() -> MLXArray {
        MLXArray([0, 0, 0, 0, 8, 8].map { Float($0) }, [2, 3])
    }

    func testDefaultScalesAreIdenticalToUnscaled() {
        let pos = positions()
        let (cosA, sinA) = Krea2Rope.make(pos: pos, axes: axes, theta: theta)
        let (cosB, sinB) = Krea2Rope.make(pos: pos, axes: axes, theta: theta, scales: [1, 1, 1])

        XCTAssertTrue(MLX.allClose(cosA, cosB, atol: 0).all().item(Bool.self),
                      "explicit unit scales must not perturb the existing path")
        XCTAssertTrue(MLX.allClose(sinA, sinB, atol: 0).all().item(Bool.self))
    }

    func testScalingAnAxisLowersItsFrequencies() {
        let pos = positions()
        let (cosPlain, _) = Krea2Rope.make(pos: pos, axes: axes, theta: theta)
        let (cosScaled, _) = Krea2Rope.make(pos: pos, axes: axes, theta: theta, scales: [1, 2, 2])

        // Axis 1 occupies columns [axes[0]/2 ..< axes[0]/2 + axes[1]/2], and
        // within it frequency j has omega = theta^(-2j/d). NTK widens theta, so
        // every j > 0 frequency shrinks; at image-token row 8 the rotation angle
        // shrinks with it, driving cos upward toward 1.
        let start = axes[0] / 2
        let plain = cosPlain[1, start + 3].item(Float.self)
        let scaled = cosScaled[1, start + 3].item(Float.self)

        XCTAssertGreaterThan(scaled, plain,
                             "NTK scaling must lower axis-1 frequencies")
    }

    func testLowestFrequencyIsThetaInvariantByConstruction() {
        // Frequency j=0 has exponent 0, so omega = theta^0 = 1 regardless of
        // theta. NTK cannot move it, and a test that probes this column would
        // pass whether or not scaling works. Pinned so nobody "fixes" the
        // implementation to chase it.
        let pos = positions()
        let (cosPlain, _) = Krea2Rope.make(pos: pos, axes: axes, theta: theta)
        let (cosScaled, _) = Krea2Rope.make(pos: pos, axes: axes, theta: theta, scales: [1, 2, 2])

        let start = axes[0] / 2
        XCTAssertEqual(cosPlain[1, start].item(Float.self),
                       cosScaled[1, start].item(Float.self),
                       accuracy: 1e-7,
                       "j=0 carries no theta dependence")
    }

    func testAxisZeroIsNeverScaled() {
        // Synthetic position with a non-zero axis-0 value. buildPositions never
        // emits this today, but the invariant is what protects text-image
        // alignment, so assert it directly rather than relying on the caller.
        let pos = MLXArray([Float(5), 0, 0], [1, 3])
        let (cosPlain, _) = Krea2Rope.make(pos: pos, axes: axes, theta: theta)
        let (cosScaled, _) = Krea2Rope.make(pos: pos, axes: axes, theta: theta, scales: [4, 4, 4])

        let axis0 = 0 ..< (axes[0] / 2)
        XCTAssertTrue(
            MLX.allClose(cosPlain[0..., axis0], cosScaled[0..., axis0], atol: 0)
                .all().item(Bool.self),
            "axis 0 must stay vanilla under any scale")
    }

    func testScaleAtOrBelowOneIsANoOp() {
        let pos = positions()
        let (cosPlain, _) = Krea2Rope.make(pos: pos, axes: axes, theta: theta)
        let (cosSmall, _) = Krea2Rope.make(pos: pos, axes: axes, theta: theta, scales: [1, 0.5, 1])

        XCTAssertTrue(MLX.allClose(cosPlain, cosSmall, atol: 0).all().item(Bool.self),
                      "downscaling must not rewrite frequencies")
    }
}
