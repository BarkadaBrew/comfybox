// SchedulerEtaConsumersTests.swift — #419: `SchedulerKind.readsAncestralEta`
// is pinned to what `SchedulerFactory.create` ACTUALLY does with `eta`, not
// to a hand-kept list: the same seed and inputs, eta 0 vs eta 1, must change
// the stepped sample exactly for the kinds that declare they read it.
// Mutation check: adding `.euler` to the set fails (its output is
// eta-independent); removing `.ddim` fails (its output changes).

import MLX
import XCTest

@testable import ZImage

final class SchedulerEtaConsumersTests: XCTestCase {

  private func stepped(kind: SchedulerKind, eta: Float) throws -> MLXArray? {
    // N-row tableau conformers trap when `step` is driven by a 1-row loop
    // (they belong to Krea2DenoiseLoop); multi-evaluation kinds (heun, the
    // 2S/RES ports) need an intermediate before `step`. Neither is handed
    // an ancestral eta by the factory; they are pinned by the explicit set
    // assertion below rather than behaviourally.
    guard !kind.isNRowTableau else { return nil }
    let config = FlowMatchSchedulerTests.makeConfig()
    var scheduler = try SchedulerFactory.create(
      kind: kind, sigmaSchedule: .flow, numInferenceSteps: 9, config: config, seed: 7, eta: eta)
    guard !scheduler.requiresIntermediateEvaluation else { return nil }
    let sample = MLXArray([0.3, -0.7, 1.1, 0.2] as [Float], [1, 1, 2, 2])
    let modelOutput = MLXArray([0.5, -0.5, 1.0, -1.0] as [Float], [1, 1, 2, 2])
    // Index 1, past the sigma≈1 guard at step 0.
    let out = scheduler.step(modelOutput: modelOutput, timestepIndex: 1, sample: sample)
    eval(out)
    return out
  }

  func testReadsAncestralEtaMatchesTheFactory() throws {
    var behaviourallyChecked = 0
    for kind in SchedulerKind.allCases {
      guard let a = try stepped(kind: kind, eta: 0.0), let b = try stepped(kind: kind, eta: 1.0) else {
        XCTAssertFalse(kind.readsAncestralEta, "\(kind.rawValue) is not stepped with eta by the factory")
        continue
      }
      let etaChangesOutput = !allClose(a, b).item(Bool.self)
      XCTAssertEqual(
        kind.readsAncestralEta, etaChangesOutput,
        "\(kind.rawValue): readsAncestralEta=\(kind.readsAncestralEta) but eta 0→1 \(etaChangesOutput ? "changed" : "did not change") the step")
      behaviourallyChecked += 1
    }
    XCTAssertGreaterThanOrEqual(behaviourallyChecked, 3, "euler / ddim / deis at least must be exercised behaviourally")
    // The declared set, spelled out, so a drift in either direction names itself.
    XCTAssertEqual(
      Set(SchedulerKind.allCases.filter(\.readsAncestralEta)), [.ddim, .dpmplusplus2sa])
  }
}
