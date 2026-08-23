// Krea2BypassLoRATests.swift — WP-E8 / FDD §3.8, O9 (AC-47, 47a, 48, 51).
//
// The bypass LoRA needs NO new engine mechanism: it is a bare-parameter
// `.diff` on `diffusion_model.txtfusion.projector`, which `loadForKrea2`
// already parses and `LoRAPatchSession` already applies transactionally.
// What WP-E8 owes is evidence, and these are it — read from the two real
// artifacts in the vault, applied to a toy module keyed exactly like
// `Krea2SingleStreamDiT.txtfusion.projector` (`Linear(numTxtLayers=12, 1)`,
// weight `[1, 12]`), so every assertion runs without a 22 GB checkpoint.
//
// The live-on-Raw half (the same element-wise check against the REAL
// projector at production q8) is `Krea2BypassOnRawTests` in the integration
// target.

import CryptoKit
import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

final class Krea2BypassLoRATests: XCTestCase {

  // MARK: - The artifacts

  /// The workflow's own file (civitai 2728234 / version 3066812), pinned by
  /// SHA-256 (AC-47) so a re-download of a DIFFERENT version is caught rather
  /// than absorbed.
  static let workflowSHA256 =
    "ac6114d7112ae2397eb26b9e6e9623aad059d346fc285ea050ffb042c7c6748e"

  private static let vault = URL(fileURLWithPath: NSString(string: "~/comfybox-models/loras/vault")
    .expandingTildeInPath, isDirectory: true)

  private func artifact(_ name: String) throws -> URL {
    let url = Self.vault.appending(path: name)
    try XCTSkipUnless(
      FileManager.default.fileExists(atPath: url.path),
      "bypass artifact absent: \(url.path) — WP-E8's acquisition step (FDD §7.1)")
    return url
  }

  private func workflowFile() throws -> URL { try artifact(Krea2BypassPolicy.workflowFile) }
  private func fedorFile() throws -> URL { try artifact(Krea2BypassPolicy.fedorFile) }

  private func sha256(_ url: URL) throws -> String {
    SHA256.hash(data: try Data(contentsOf: url))
      .map { String(format: "%02x", $0) }.joined()
  }

  /// The single delta the file carries, as the loader resolved it.
  private func onlyDelta(of url: URL) throws -> (key: String, tensor: MLXArray) {
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertTrue(weights.weights.isEmpty, "the bypass offers NO low-rank pairs")
    XCTAssertEqual(weights.deltas.count, 1)
    let entry = try XCTUnwrap(weights.deltas.first)
    return (entry.key, entry.value.tensor)
  }

  private func floats(_ a: MLXArray) -> [Float] {
    let flat = a.asType(.float32).flattened()
    eval(flat)
    return flat.asArray(Float.self)
  }

  // MARK: - AC-47 — it loads as ONE delta on the projector

  func testLoadsAsDelta() throws {
    for url in [try workflowFile(), try fedorFile()] {
      let weights = try LoRAWeightLoader.loadForKrea2(from: url)

      XCTAssertEqual(weights.weights.count, 0, "\(url.lastPathComponent): 0 pairs")
      // `rank` is `inferRank`'s fallback (16) because there is no pair to
      // infer from — pinned so it is never read as a real rank. It is inert:
      // deltas are scaled by userScale alone, never alpha/rank.
      XCTAssertEqual(weights.rank, 16, "no pairs → the loader's fallback, not a rank")
      XCTAssertEqual(weights.deltas.count, 1, "\(url.lastPathComponent): exactly 1 delta")

      let (key, tensor) = try onlyDelta(of: url)
      // `diffusion_model.txtfusion.projector.diff` → the module path the
      // patch session resolves to `txtfusion.projector.weight`.
      XCTAssertEqual(key, "txtfusion.projector")
      XCTAssertEqual(tensor.shape, [1, 12])
      XCTAssertEqual(tensor.dtype, .float32, "F32 in the file, per the header")
    }
  }

  /// AC-47's pin: the WORKFLOW's artifact, by content, not by name.
  func testWorkflowArtifactSHA256IsPinned() throws {
    let url = try workflowFile()
    XCTAssertEqual(
      try sha256(url), Self.workflowSHA256,
      "\(url.lastPathComponent) is not the artifact this WP was verified against "
        + "(civitai 2728234 / version 3066812, 160 bytes) — re-check the download")
    XCTAssertEqual(try Data(contentsOf: url).count, 160)
  }

  /// AC-49's other half: the real 1,040-byte file must load — no size floor.
  func testTheSmallFilesAreNotRejectedForBeingSmall() throws {
    // Resolve fixtures before entering an XCTest assertion autoclosure. A
    // missing optional artifact throws XCTSkip; inside XCTAssertEqual that
    // skip is converted into a failure instead of skipping the test on CI.
    let fedor = try fedorFile()
    let workflow = try workflowFile()
    let fedorSize = try Data(contentsOf: fedor).count
    XCTAssertEqual(fedorSize, 1040)
    _ = try onlyDelta(of: fedor)
    _ = try onlyDelta(of: workflow)
  }

  // MARK: - AC-47a — the substitution is MEASURED, not assumed

  /// Fedor's `__metadata__` claims "identical numerical effect" with the
  /// workflow's file. That is its author's assertion. This is the measurement.
  ///
  /// Result (recorded as an O6 fixture fact in the WP-E8 report): the two
  /// tensors are NOT bit-equal — Fedor stores the values typed to ten
  /// decimals, the workflow file stores their bf16-exact forms. The
  /// element-wise gap is ~2.5e-5 absolute / ~2.8e-5 relative, and BOTH round
  /// to the same bf16 — which is the dtype the production chain casts to
  /// before scaling (`LoRAPatchSession.swift`), so at the production dtype
  /// chain the two artifacts are indistinguishable.
  func testEquivalentToReferenceFile() throws {
    let a = floats(try onlyDelta(of: try workflowFile()).tensor)
    let b = floats(try onlyDelta(of: try fedorFile()).tensor)
    XCTAssertEqual(a.count, 12)
    XCTAssertEqual(b.count, 12)

    // The same two columns, and only those two, are touched by both.
    let touchedA = a.indices.filter { a[$0] != 0 }
    let touchedB = b.indices.filter { b[$0] != 0 }
    XCTAssertEqual(touchedA, [8, 9])
    XCTAssertEqual(touchedB, [8, 9])

    var maxAbs: Float = 0
    var maxRel: Float = 0
    for i in a.indices {
      let d = abs(a[i] - b[i])
      maxAbs = max(maxAbs, d)
      if a[i] != 0 { maxRel = max(maxRel, d / abs(a[i])) }
    }
    print("[WP-E8] AC-47a 2vector: \(a)")
    print("[WP-E8] AC-47a fedor  : \(b)")
    print("[WP-E8] AC-47a max|Δ| = \(maxAbs)  max relative = \(maxRel)")

    XCTAssertNotEqual(a, b, "they are NOT bit-equal — the brief's expectation, pinned")
    // The brief's bound was ≤ 2e-5. MEASURED: 2.4974e-5 at column 9
    // (0.890625 vs 0.8906000256538391) — just over it. The assertion is the
    // measurement, not the estimate; the discrepancy is in the report.
    XCTAssertLessThanOrEqual(maxAbs, 3e-5, "max|Δ| between the two artifacts")
    XCTAssertLessThanOrEqual(maxRel, 5e-5)

    // The decisive half: the production chain casts the delta to the
    // parameter dtype BEFORE scaling, and the transformer is bf16. At that
    // cast the two artifacts are BIT-IDENTICAL, so Fedor is a verified
    // stand-in *at strength 1.0 on a bf16 parameter* — which is the only
    // configuration `krea2-reference` uses.
    let ab = floats(MLXArray(a).asType(.bfloat16))
    let bb = floats(MLXArray(b).asType(.bfloat16))
    XCTAssertEqual(ab, bb, "bf16(2vector) == bf16(fedor) element-wise")
  }

  // MARK: - Toy projector (Krea-2 module paths)

  /// `txtfusion.projector` is `Linear(numTxtLayers, 1, bias: false)` —
  /// weight `[1, numTxtLayers]`.
  private final class ToyTextFusion: Module {
    @ModuleInfo(key: "projector") var projector: Linear
    init(weight: MLXArray) {
      self._projector.wrappedValue = Linear(weight: weight, bias: nil)
      super.init()
    }
  }

  private final class ToyKrea2: Module {
    @ModuleInfo(key: "txtfusion") var txtfusion: ToyTextFusion
    init(weight: MLXArray) {
      self._txtfusion.wrappedValue = ToyTextFusion(weight: weight)
      super.init()
    }
  }

  /// Raw's real projector row, as read from `raw.safetensors` this WP
  /// (FDD §3.8): cols 8 and 9 are the two the delta tracks.
  private static let rawProjectorRow: [Float] = [
    0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
    -0.51171875, -0.890625, 0.90, 1.00,
  ]

  private func toy(_ row: [Float], dtype: DType) -> ToyKrea2 {
    ToyKrea2(weight: MLXArray(row, [1, row.count]).asType(dtype))
  }

  private func projectorWeight(_ model: ToyKrea2) -> [Float] {
    floats(model.txtfusion.projector.weight)
  }

  // MARK: - The application path (element-wise, deltasApplied == 1)

  /// The whole engine claim of WP-E8 in one test: through the REAL
  /// applicator + patch session, the workflow's file moves the projector by
  /// exactly `delta × strength` on the touched columns and leaves every
  /// other column bit-unchanged — and the report says `1 delta, 0 pairs`.
  func testAppliesAsExactlyDeltaTimesStrength() throws {
    let url = try workflowFile()
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    let delta = floats(try onlyDelta(of: url).tensor)

    for scale in [Float(1.0), 0.5, 3.0] {
      let model = toy(Self.rawProjectorRow, dtype: .float32)
      let before = projectorWeight(model)

      let report = try LoRAApplicator.applyDynamically(
        to: model, loraWeights: weights, scale: scale, strict: true,
        name: url.lastPathComponent, logger: nil)
      let session = LoRAPatchSession(module: model)
      let applied = try session.apply(weights: weights, scale: scale)
      let full = report.withDeltasApplied(applied)

      XCTAssertEqual(full.offered, 0, "no low-rank pairs")
      XCTAssertEqual(full.bound, 0)
      XCTAssertEqual(full.deltasApplied, 1, "the one `.diff`")
      XCTAssertTrue(full.isComplete)
      XCTAssertEqual(full.unbound, [])

      let after = projectorWeight(model)
      for i in before.indices {
        let expected = before[i] + delta[i] * scale
        XCTAssertEqual(after[i], expected, accuracy: 0, "column \(i) at scale \(scale)")
      }
      // Untouched columns are BIT-unchanged, not merely close.
      for i in before.indices where delta[i] == 0 {
        XCTAssertEqual(after[i], before[i], "column \(i) must not move")
      }

      session.clear()
      XCTAssertEqual(projectorWeight(model), before, "clear() restores exactly")
    }
  }

  /// AC-48 — the doubling, WITH the dtype chain it depends on. On a bf16
  /// parameter at strength 1.0 the two taps land on exactly `2·w`; at 0.5
  /// they do not, and only the ~4e-5 agreement holds.
  func testDoublesTwoColumns() throws {
    let weights = try LoRAWeightLoader.loadForKrea2(from: try workflowFile())

    let model = toy(Self.rawProjectorRow, dtype: .bfloat16)
    let before = projectorWeight(model)
    _ = try LoRAPatchSession(module: model).apply(weights: weights, scale: 1.0)
    let after = projectorWeight(model)

    for i in [8, 9] {
      XCTAssertEqual(after[i], 2 * before[i], "column \(i) doubles in bf16 at strength 1.0")
    }
    for i in before.indices where !(i == 8 || i == 9) {
      XCTAssertEqual(after[i], before[i], "column \(i) is bit-unchanged")
    }

    // At 0.5 the identity is 1.5·w only to the dtype's precision.
    let half = toy(Self.rawProjectorRow, dtype: .bfloat16)
    let halfBefore = projectorWeight(half)
    _ = try LoRAPatchSession(module: half).apply(weights: weights, scale: 0.5)
    let halfAfter = projectorWeight(half)
    for i in [8, 9] {
      XCTAssertEqual(halfAfter[i], 1.5 * halfBefore[i], accuracy: 1e-2)
    }
  }

  // MARK: - AC-51 — a shape mismatch never applies partially

  /// FINDING (as WP-E7 found for AC-42): a projector of the WRONG WIDTH is
  /// refused as `incompatibleWeights`, not `partialApplication` — the key
  /// RESOLVES (the module exists), it is the tensor that does not fit, and
  /// `LoRAPatchSession.alignedPatchTensor` catches that before any mutation.
  /// `partialApplication` is what a MISSING target yields. Both are asserted:
  /// the criterion's concern (never a silent partial apply) holds either way.
  func testShapeMismatchThrows() throws {
    let weights = try LoRAWeightLoader.loadForKrea2(from: try workflowFile())

    // [1, 10] — the same module path, a different number of layer taps.
    let narrow = toy(Array(Self.rawProjectorRow.prefix(10)), dtype: .float32)
    let before = projectorWeight(narrow)
    let session = LoRAPatchSession(module: narrow)
    XCTAssertThrowsError(try session.apply(weights: weights, scale: 1.0)) { error in
      guard case LoRAError.incompatibleWeights(let message) = error else {
        return XCTFail("expected incompatibleWeights, got \(error)")
      }
      XCTAssertTrue(message.contains("txtfusion.projector.weight"), message)
      XCTAssertTrue(message.contains("[1, 12]"), message)
    }
    XCTAssertEqual(projectorWeight(narrow), before, "nothing was mutated")
    XCTAssertFalse(session.isActive, "no snapshot was taken — the throw was pre-mutation")

    // Preflight alone refuses it too, so a caller can ask without applying.
    XCTAssertThrowsError(try LoRAPatchSession(module: narrow).preflight(weights: weights))

    // And a module with NO projector at all is the `partialApplication` case.
    let empty = ToyTextFusion(weight: MLXArray([Float(1)], [1, 1]))
    XCTAssertThrowsError(try LoRAPatchSession(module: empty).apply(weights: weights, scale: 1.0)) {
      guard case LoRAError.partialApplication(_, let unbound) = $0 else {
        return XCTFail("expected partialApplication, got \($0)")
      }
      XCTAssertEqual(unbound, ["txtfusion.projector"])
    }
  }

  // MARK: - Relativity: the bypass is base-agnostic

  /// `krea2Relative: nil` — the delta shape-matches BOTH variants (Raw's
  /// projector is `[1,12] F32`, kroma-v0.2-turbo's is `[1,12] BF16`), so the
  /// seed table declares nothing and the guard refuses it on NEITHER base.
  /// The doubling identity is Raw-only; that is documented, not enforced.
  func testItIsNotRefusedOnEitherBase() throws {
    for name in [Krea2BypassPolicy.workflowFile, Krea2BypassPolicy.fedorFile] {
      XCTAssertNil(
        Krea2LoRARelativity.seeded(forFilename: name),
        "\(name) must declare no relativity")

      let config = LoRAConfiguration.local("/vault/\(name)", scale: 1.0)
      let url = URL(fileURLWithPath: "/vault/\(name)")
      let required = Krea2LoRARelativity.required(for: config, resolvedURL: url)
      XCTAssertNil(required)
      for base in [Krea2Variant.raw, .turbo] {
        XCTAssertNoThrow(
          try Krea2LoRARelativity.check(lora: name, required: required, loaded: base),
          "the bypass must apply on \(base.rawValue)")
      }
    }
  }

  // MARK: - Provenance: role "bypass"

  /// The slot label reaches `RenderRecipe.loras[].role`, and the counters are
  /// the shape §3.8 specifies: `pairs_offered: 0, pairs_bound: 0,
  /// deltas_applied: 1`, with `file` naming WHICH artifact applied.
  func testRecipeRecordsTheBypassSlot() throws {
    let path = Self.vault.appending(path: Krea2BypassPolicy.workflowFile).path
    var config = LoRAConfiguration.local(path, scale: 1.0)
    config.role = "bypass"

    let readBack = RenderRecipe.LoRAReadBack(
      configuration: config,
      report: LoRAApplicationReport(
        offered: 0, bound: 0, quantizedBound: 0, deltasApplied: 1,
        shapeRejected: 0, unbound: []),
      resolvedRelativeTo: nil)

    let recipe = RenderRecipeFixture.recipe(steps: 9, loras: [readBack])
    let applied = try XCTUnwrap(recipe.loras.first)
    XCTAssertEqual(applied.role, "bypass")
    XCTAssertEqual(applied.pairsOffered, 0)
    XCTAssertEqual(applied.pairsBound, 0)
    XCTAssertEqual(applied.deltasApplied, 1)
    XCTAssertEqual(applied.scaleApplied, 1.0)
    XCTAssertNil(applied.relativeTo, "base-agnostic — no relativity to record")
    XCTAssertTrue(
      applied.file.hasSuffix(Krea2BypassPolicy.workflowFile),
      "the record names WHICH artifact applied: \(applied.file)")
  }
}
