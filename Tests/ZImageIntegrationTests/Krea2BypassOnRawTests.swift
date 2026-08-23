// Krea2BypassOnRawTests.swift — WP-E8 (FDD §3.8, O9 AC-47a/48) at PRODUCTION
// config: the real Raw checkpoint at the deployed q8, the real bypass files
// off disk, through `Krea2Pipeline.loadLoRAs`' strict apply.
//
// The unit half (`Krea2BypassLoRATests`) proves the arithmetic against a toy
// module. This is the half that proves it against the tensor that actually
// ships: `txtfusion.projector.weight` in `raw.safetensors`, F32 `[1, 12]`,
// excluded from quantization (`Krea2Pipeline` — `!path.contains("projector")`)
// so the target is full precision even at q8.
//
// Same skip discipline as `Krea2KromaOnRawTests`: a missing model directory
// skips by name; an init failure on an INSTALLED model FAILS.

import Foundation
import MLX
import XCTest
@testable import ZImage

final class Krea2BypassOnRawTests: XCTestCase {

  private static let vault = ("~/comfybox-models/loras/vault" as NSString).expandingTildeInPath

  private func artifact(_ name: String) throws -> URL {
    let url = URL(fileURLWithPath: Self.vault).appending(path: name)
    try XCTSkipUnless(
      FileManager.default.fileExists(atPath: url.path),
      "bypass artifact absent: \(url.path)")
    return url
  }

  // MARK: - The Raw pipeline (built once for the class)

  private static var sharedRaw: Krea2Pipeline?
  private static var rawSkip: XCTSkip?
  private static var rawInitFailure: Error?

  private func rawPipeline() throws -> Krea2Pipeline {
    if ProcessInfo.processInfo.environment["CI"] != nil { throw XCTSkip("GPU test skipped in CI") }
    if let existing = Self.sharedRaw { return existing }
    if let skip = Self.rawSkip { throw skip }
    if let failure = Self.rawInitFailure {
      XCTFail("Krea2Pipeline init already failed once on an INSTALLED krea2-raw: \(failure)")
      throw failure
    }
    let paths: Krea2ModelPaths
    do {
      paths = try Krea2ModelDetection.resolve(spec: "krea2-raw")
    } catch let error as Krea2ModelPathsError {
      let skip = XCTSkip("krea2-raw not installed (\(error)) — WP-E8 live check not runnable here")
      Self.rawSkip = skip
      throw skip
    }
    guard FileManager.default.fileExists(atPath: paths.transformerFile.path) else {
      let skip = XCTSkip("\(paths.transformerFile.path) absent — WP-E8 live check not runnable here")
      Self.rawSkip = skip
      throw skip
    }
    XCTAssertEqual(paths.variant, .raw)
    do {
      let pipeline = try Krea2Pipeline(paths: paths, quantizeTransformer: 8)
      Self.sharedRaw = pipeline
      return pipeline
    } catch {
      Self.rawInitFailure = error
      XCTFail("Krea2Pipeline(quantizeTransformer: 8) failed with krea2-raw INSTALLED — a "
        + "regression (quantization, shape or memory), not a missing model: \(error)")
      throw error
    }
  }

  @discardableResult
  private func applyStack(_ k2: Krea2Pipeline, _ configs: [LoRAConfiguration]) throws -> [LoRAApplicationReport] {
    let done = expectation(description: "loadLoRAs(\(configs.count))")
    var caught: Error?
    Task {
      do { try await k2.loadLoRAs(configs) } catch { caught = error }
      done.fulfill()
    }
    wait(for: [done], timeout: 900)
    if let caught { throw caught }
    return k2.loadedLoRAReports
  }

  /// A DETACHED, evaluated copy of the live projector weight — MLX
  /// parameters are references mutated in place, so holding the array itself
  /// would hold an alias and every "before" would equal every "after".
  private func projectorArray(_ k2: Krea2Pipeline) throws -> MLXArray {
    let flat = k2.transformer.parameters().flattened()
    let entry = try XCTUnwrap(
      flat.first { $0.0 == "txtfusion.projector.weight" },
      "txtfusion.projector.weight is not a parameter of the loaded DiT: "
        + "\(flat.map(\.0).filter { $0.contains("txtfusion") }.prefix(10))")
    let copy = entry.1 + MLXArray(0, dtype: entry.1.dtype)
    eval(copy)
    return copy.flattened()
  }

  private func floats(_ a: MLXArray) -> [Float] {
    let f = a.asType(.float32).flattened()
    eval(f)
    return f.asArray(Float.self)
  }

  private func projector(_ k2: Krea2Pipeline) throws -> [Float] {
    floats(try projectorArray(k2))
  }

  private func delta(of url: URL) throws -> MLXArray {
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertEqual(weights.deltas.count, 1)
    return try XCTUnwrap(weights.deltas["txtfusion.projector"]).tensor.flattened()
  }

  /// A minimal trace so the record can be BUILT — the render itself is not
  /// what this test measures (AC-50 covers that), the provenance shape is.
  private static func trace() -> Krea2RunTrace {
    let sigmas = SigmaSchedule.krea2(numSteps: 9, mu: 0.9062)
    return Krea2RunTrace(
      sampler: .euler, sigmaSchedule: .krea2, sigmaScheduleRequested: nil,
      mu: 0.9062, shift: exp(0.9062), shiftSource: "dynamic", sigmas: sigmas,
      stepsRequested: 9, stepsEffective: sigmas.count - 1, stepsRun: sigmas.count - 1,
      modelEvals: sigmas.count - 1, startIndex: 0, denoise: 1.0, guidance: 1.0,
      eta: 0, bongmath: false, seed: 44821, width: 1024, height: 1024,
      negativePromptApplied: nil)
  }

  // MARK: - The live check

  /// The whole of WP-E8's engine claim, on the real base: applying the
  /// workflow's bypass at strength `s` moves `txtfusion.projector.weight` by
  /// exactly `delta × s` on the touched columns, leaves the rest
  /// bit-unchanged, reports `deltasApplied == 1 / pairsBound == 0`, and lands
  /// in the record under `role: "bypass"`.
  func testBypassAppliesOnRawAsExactlyDeltaTimesStrength() throws {
    let k2 = try rawPipeline()

    for name in [Krea2BypassPolicy.workflowFile, Krea2BypassPolicy.fedorFile] {
      let url = try artifact(name)
      let d = try delta(of: url)
      for scale in [Float(1.0), 0.5] {
        let beforeArray = try projectorArray(k2)
        let before = floats(beforeArray)
        XCTAssertEqual(before.count, 12, "Raw's projector is [1, 12]")

        var config = LoRAConfiguration.local(url, scale: scale)
        config.role = "bypass"
        let reports = try applyStack(k2, [config])

        let report = try XCTUnwrap(reports.first)
        XCTAssertEqual(report.offered, 0, "\(name): the bypass offers no pairs")
        XCTAssertEqual(report.bound, 0)
        XCTAssertEqual(report.deltasApplied, 1)
        XCTAssertTrue(report.isComplete)

        let afterArray = try projectorArray(k2)
        let after = floats(afterArray)

        // The claim, at the dtype chain the production path actually uses
        // (§3.8): `LoRAPatchSession` casts the delta to the PARAMETER dtype
        // before scaling, and the Krea 2 transformer loads bf16 — so the
        // expectation is computed in that dtype and the comparison is
        // BIT-exact, not an epsilon. Computing it in F32 instead is what
        // makes the "~4e-5" wobble appear, and that wobble is the test's
        // arithmetic, not the engine's.
        let expected = floats(beforeArray + d.asType(beforeArray.dtype) * scale)
        XCTAssertEqual(after, expected,
                       "\(name) @ \(scale): the projector must move by exactly delta × strength")
        let dF = floats(d)
        for i in before.indices where dF[i] == 0 {
          XCTAssertEqual(after[i], before[i], "column \(i) must be bit-unchanged")
        }
        print("[WP-E8] \(name) @ \(scale) dtype=\(beforeArray.dtype):")
        print("        before=\(before)")
        print("        after =\(after)")

        // The doubling identity §3.8 states, at the workflow's strength —
        // EXACT in bf16, not approximate.
        if scale == 1.0 {
          for i in [8, 9] {
            XCTAssertEqual(after[i], 2 * before[i],
                           "column \(i) doubles exactly at strength 1.0 on Raw")
          }
        }

        // Provenance: the slot label survives into the record.
        let readBacks = try XCTUnwrap(RenderRecipe.loRAReadBacks(
          configs: k2.loadedLoRAConfigs, reports: k2.loadedLoRAReports,
          relativities: k2.loadedLoRARelativities))
        let recipe = RenderRecipe.krea2(.init(
          baseModel: "krea2-raw", variant: k2.variant,
          transformerFile: k2.paths.transformerFile,
          quantizationBits: k2.transformerQuantBits,
          vae: k2.currentVAE,
          textEncoderFile: k2.paths.textEncoderFile,
          loras: readBacks, control: nil,
          trace: Self.trace()))
        let applied = try XCTUnwrap(recipe.loras.first)
        XCTAssertEqual(applied.role, "bypass")
        XCTAssertEqual(applied.deltasApplied, 1)
        XCTAssertEqual(applied.pairsBound, 0)
        XCTAssertEqual(applied.pairsOffered, 0)
        XCTAssertTrue(applied.file.hasSuffix(name))
        XCTAssertNil(applied.relativeTo, "the bypass declares no relativity")

        // Clear and prove the base came back exactly.
        _ = try applyStack(k2, [])
        XCTAssertEqual(try projector(k2), before, "clearing restores the projector exactly")
      }
    }
  }

  /// The relativity half, live: the bypass is refused on NEITHER base, so it
  /// binds on the resident Raw with no `requiresBase` declared anywhere.
  func testBypassIsNotRefusedOnRaw() throws {
    let k2 = try rawPipeline()
    let url = try artifact(Krea2BypassPolicy.workflowFile)
    XCTAssertNil(Krea2LoRARelativity.seeded(forFilename: url.lastPathComponent))
    var config = LoRAConfiguration.local(url, scale: 1.0)
    config.role = "bypass"
    XCTAssertNoThrow(try applyStack(k2, [config]))
    XCTAssertEqual(k2.loadedLoRARelativities.first ?? nil, nil)
    _ = try applyStack(k2, [])
  }
}
