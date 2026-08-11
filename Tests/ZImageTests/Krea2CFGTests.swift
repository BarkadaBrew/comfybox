import MLX
import XCTest

@testable import ZImage

/// Opt-in CFG for the krea2 family (Todd 2026-08-11): guidance > 1 runs a
/// second unconditioned pass and combines via classifier-free guidance,
/// making negative prompts real on krea2/Kroma. guidance 1.0 must remain
/// byte-identical to the distilled single-pass recipe.
final class Krea2CFGTests: XCTestCase {

  func testScaleOneReturnsCondExactly() {
    let cond = MLXArray([2.0, -3.0, 0.5] as [Float])
    let uncond = MLXArray([1.0, 1.0, 1.0] as [Float])
    let out = Krea2Sampling.applyCFG(cond: cond, uncond: uncond, scale: 1.0)
    XCTAssertEqual(out.asArray(Float.self), cond.asArray(Float.self))
  }

  func testCombineMathAtScale() {
    // uncond + s*(cond - uncond): 1 + 1.5*(2-1) = 2.5 ; 0 + 1.5*(-2-0) = -3
    let cond = MLXArray([2.0, -2.0] as [Float])
    let uncond = MLXArray([1.0, 0.0] as [Float])
    let out = Krea2Sampling.applyCFG(cond: cond, uncond: uncond, scale: 1.5)
    let values = out.asArray(Float.self)
    XCTAssertEqual(values[0], 2.5, accuracy: 1e-5)
    XCTAssertEqual(values[1], -3.0, accuracy: 1e-5)
  }

  func testRequestDefaultsKeepDistilledRecipe() {
    let request = Krea2Pipeline.Request(prompt: "x")
    XCTAssertEqual(request.guidance, 1.0, "default stays guidance-free")
    XCTAssertNil(request.negativePrompt)
  }
}
