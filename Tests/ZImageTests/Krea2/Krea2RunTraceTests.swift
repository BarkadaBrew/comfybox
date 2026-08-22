import XCTest
@testable import ZImage

/// WP-E10 — the pipeline's run trace (FDD §3.10): every provenance number
/// that describes the loop is COUNTED by the loop (`Krea2RunCounter`), never
/// recomputed from the request. The trace is a plain value so its accounting
/// is asserted here with no weights (AC-63 for the img2img start index; the
/// sigma head/tail and `steps_effective` for the record).
final class Krea2RunTraceTests: XCTestCase {

  private func trace(sigmas: [Float], stepsRequested: Int, startIndex: Int, cfg: Bool) -> Krea2RunTrace {
    let counter = Krea2RunCounter()
    // Walk the grid exactly as the loop does: one step per interval from
    // startIndex, one eval per transformer call (two under CFG).
    for _ in startIndex..<(sigmas.count - 1) {
      counter.eval()
      if cfg { counter.eval() }
      counter.step()
    }
    return Krea2RunTrace(
      width: 1024, height: 1024, seed: 44821,
      mu: 0.9062, shift: 2.475, shiftSource: "dynamic",
      sigmas: sigmas, stepsRequested: stepsRequested, startIndex: startIndex,
      guidance: cfg ? 2.0 : 1.0, denoise: startIndex == 0 ? 1.0 : 0.7,
      sampler: "euler", sigmaSchedule: "krea2", counter: counter)
  }

  func testTextToImageAccounting() {
    let sigmas = SigmaSchedule.krea2(numSteps: 9, mu: 0.9062)
    let t = trace(sigmas: sigmas, stepsRequested: 9, startIndex: 0, cfg: false)
    XCTAssertEqual(t.stepsRequested, 9)
    XCTAssertEqual(t.stepsEffective, 9)
    XCTAssertEqual(t.stepsRun, 9)
    XCTAssertEqual(t.modelEvals, 9)
    XCTAssertEqual(t.sigmaHead, Array(sigmas.prefix(3)))
    XCTAssertEqual(t.sigmaTail, Array(sigmas.suffix(3)))
    XCTAssertEqual(t.sigmaHead.first, 1.0)
    XCTAssertEqual(t.sigmaTail.last, 0.0)
  }

  /// AC-12's cost line: CFG doubles the evals, never the steps.
  func testCFGDoublesModelEvals() {
    let sigmas = SigmaSchedule.krea2(numSteps: 6, mu: 0.9062)
    let t = trace(sigmas: sigmas, stepsRequested: 6, startIndex: 0, cfg: true)
    XCTAssertEqual(t.stepsRun, 6)
    XCTAssertEqual(t.modelEvals, 12)
    XCTAssertEqual(t.guidance, 2.0)
  }

  /// AC-63: `strength 0.3` / 20 steps → `steps_requested 20`, `steps_run 14`,
  /// `model_evals 14` at guidance 1 — the start index is the loop's own
  /// convention (`denoise = 1 - strength`, `start = total - ceil(total·denoise)`).
  func testImg2ImgStepAccounting() {
    XCTAssertEqual(Krea2Sampling.img2imgStartIndex(total: 20, strength: 0.3), 6)
    XCTAssertEqual(Krea2Sampling.img2imgStartIndex(total: 9, strength: 0.3), 2)   // 9 - ceil(6.3) = 2
    XCTAssertEqual(Krea2Sampling.img2imgStartIndex(total: 9, strength: 0.99), 8)  // denoise 0.01 → ceil(0.09)=1 → 8
    XCTAssertEqual(Krea2Sampling.img2imgStartIndex(total: 9, strength: 0.01), 0)  // denoise 0.99 → ceil(8.91)=9 → 0
    let sigmas = SigmaSchedule.krea2(numSteps: 20, mu: 0.9062)
    let start = Krea2Sampling.img2imgStartIndex(total: 20, strength: 0.3)
    let t = trace(sigmas: sigmas, stepsRequested: 20, startIndex: start, cfg: false)
    XCTAssertEqual(t.stepsRequested, 20)
    XCTAssertEqual(t.stepsEffective, 20)
    XCTAssertEqual(t.stepsRun, 14)
    XCTAssertEqual(t.modelEvals, 14)
    XCTAssertEqual(t.startIndex, 6)
  }

  func testCounterStartsAtZeroAndIsIndependentPerRun() {
    let a = Krea2RunCounter(), b = Krea2RunCounter()
    a.eval(); a.eval(); a.step()
    XCTAssertEqual(a.steps, 1); XCTAssertEqual(a.evals, 2)
    XCTAssertEqual(b.steps, 0); XCTAssertEqual(b.evals, 0)
  }

  func testShortGridHeadAndTailDoNotOverlapBeyondTheGrid() {
    let sigmas: [Float] = [1.0, 0.5, 0.0]
    let t = trace(sigmas: sigmas, stepsRequested: 2, startIndex: 0, cfg: false)
    XCTAssertEqual(t.sigmaHead, [1.0, 0.5, 0.0])
    XCTAssertEqual(t.sigmaTail, [1.0, 0.5, 0.0])
    XCTAssertEqual(t.stepsEffective, 2)
  }
}
