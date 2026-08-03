import Foundation
import MLX
import MLXNN
import XCTest

@testable import ZImage

/// Task #16 rev 2: bare-delta LoRA keys (.diff / .diff_b / .set_weight).
///
/// Driver: Kroma v0.1 ships 264 lora_A/B pairs + 159 `.diff` tensors targeting
/// norm/modulation parameters. Spec: specs/lora-delta-keys-design.md (rev 2,
/// post-Codex). The contract under test:
///   - loader classifies every tensor key (bindable / metadata /
///     known-unsupported / unknown) instead of silently skipping
///   - deltas apply through a preflighted, transactional, instance-scoped
///     session with first-write-wins snapshots of detached copies
///   - clear() restores byte-identical state, including exact packed tuples
///     for quantized targets
final class LoRADeltaKeyTests: XCTestCase {

  private var tempDir: URL!

  override func setUpWithError() throws {
    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("lora-delta-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: tempDir)
  }

  private func writeFixture(_ arrays: [String: MLXArray], name: String = "fixture.safetensors") throws -> URL {
    let url = tempDir.appendingPathComponent(name)
    try MLX.save(arrays: arrays, url: url)
    return url
  }

  /// Toy module mirroring the three Kroma target kinds: a biased Linear,
  /// and a bare non-module leaf parameter (norm-style scale vector).
  private final class ToyBlock: Module {
    @ModuleInfo(key: "lin") var lin: Linear
    @ModuleInfo(key: "scale") var scale: MLXArray

    init(dim: Int = 8) {
      self._lin.wrappedValue = Linear(dim, dim, bias: true)
      self._scale.wrappedValue = MLXArray.ones([dim])
      super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
      lin(x) * scale
    }
  }

  // MARK: - Loader: parsing + classification

  func testLoaderParsesDiffKeysIntoDeltas() throws {
    let url = try writeFixture([
      "blocks.0.attn.q.lora_A.weight": MLXArray.zeros([4, 16]),
      "blocks.0.attn.q.lora_B.weight": MLXArray.zeros([16, 4]),
      "blocks.0.prenorm.scale.diff": MLXArray.ones([16]),
      "blocks.0.postnorm.scale.diff": MLXArray.ones([16]),
    ])

    let weights = try LoRAWeightLoader.load(from: url)

    XCTAssertEqual(weights.weights.count, 1, "adapter pair still loads")
    XCTAssertEqual(weights.deltas.count, 2, "both .diff tensors must load as deltas")
    guard case .diff = weights.deltas["blocks.0.prenorm.scale"] else {
      return XCTFail("prenorm delta missing or wrong kind")
    }
  }

  func testLoaderParsesSuffixFormPairKeys() throws {
    // Kroma (and other ComfyUI-ecosystem LoRAs) name pairs `...lora_A` /
    // `...lora_B` with NO trailing `.weight` — our patterns only matched the
    // diffusers dotted form `.lora_A.`, so the real Kroma file loaded ZERO
    // pairs. Caught by the classification guard on the real file.
    let url = try writeFixture([
      "diffusion_model.blocks.0.attn.wq.lora_A": MLXArray.zeros([4, 16]),
      "diffusion_model.blocks.0.attn.wq.lora_B": MLXArray.zeros([16, 4]),
    ])

    let weights = try LoRAWeightLoader.loadForKrea2(from: url)

    XCTAssertEqual(weights.weights.count, 1)
    XCTAssertNotNil(weights.weights["blocks.0.attn.wq.weight"],
      "suffix-form pair must map to the module's .weight target")
  }

  func testLoaderMapsDiffBiasToRealBiasPathAndParsesSetWeight() throws {
    let url = try writeFixture([
      "blocks.0.mod.lin.diff": MLXArray.zeros([8, 8]),
      "blocks.0.mod.lin.diff_b": MLXArray.zeros([8]),
      "blocks.0.out.set_weight": MLXArray.zeros([8, 8]),
    ])

    let weights = try LoRAWeightLoader.load(from: url)

    XCTAssertEqual(weights.deltas.count, 3)
    // ComfyUI maps .diff_b onto the target's REAL bias parameter (lora.py:79),
    // not an invented key.
    guard case .diffBias = weights.deltas["blocks.0.mod.lin.bias"] else {
      return XCTFail("diff_b must resolve to the real .bias parameter path")
    }
    guard case .setWeight = weights.deltas["blocks.0.out"] else {
      return XCTFail("set_weight must load as a replacement")
    }
  }

  func testAlphaTensorsAreMetadataNotErrorsAndNotDeltas() throws {
    // kohya-style per-layer alpha: consumed by the pair, never a delta,
    // never an unknown-key error (Codex finding 6).
    let url = try writeFixture([
      "blocks.0.attn.q.lora_down.weight": MLXArray.zeros([4, 16]),
      "blocks.0.attn.q.lora_up.weight": MLXArray.zeros([16, 4]),
      "blocks.0.attn.q.alpha": MLXArray(Float(2.0)),
    ])

    let weights = try LoRAWeightLoader.load(from: url)

    XCTAssertEqual(weights.weights.count, 1)
    XCTAssertTrue(weights.deltas.isEmpty)
    XCTAssertEqual(weights.layerAlphas["blocks.0.attn.q.weight"], 2.0)
  }

  func testKnownUnsupportedFeatureErrorNamesTheFeature() throws {
    let url = try writeFixture([
      "blocks.0.attn.q.lora_down.weight": MLXArray.zeros([4, 16]),
      "blocks.0.attn.q.lora_up.weight": MLXArray.zeros([16, 4]),
      "blocks.0.attn.q.dora_scale": MLXArray.zeros([16]),
    ])

    XCTAssertThrowsError(try LoRAWeightLoader.load(from: url)) { error in
      let message = String(describing: error).lowercased()
      XCTAssertTrue(message.contains("dora"), "must say WHICH unsupported feature: \(message)")
    }
  }

  func testUnknownKeyIsALoadErrorNamingTheKey() throws {
    let url = try writeFixture([
      "blocks.0.attn.q.lora_A.weight": MLXArray.zeros([4, 16]),
      "blocks.0.attn.q.lora_B.weight": MLXArray.zeros([16, 4]),
      "blocks.0.mystery_tensor": MLXArray.zeros([16]),
    ])

    XCTAssertThrowsError(try LoRAWeightLoader.load(from: url)) { error in
      XCTAssertTrue(String(describing: error).contains("mystery_tensor"))
    }
  }

  // MARK: - Session: apply / scale / clear lifecycle

  private func makeDeltaWeights(_ deltas: [String: DeltaPatch], rank: Int = 4, alpha: Float? = nil) -> LoRAWeights {
    LoRAWeights(weights: [:], rank: rank, alpha: alpha, deltas: deltas)
  }

  func testDiffChangesOutputAndHonorsUserScaleNotAlphaOverRank() throws {
    let toy = ToyBlock()
    let x = MLXArray.ones([1, 8])
    let before = toy(x)
    eval(before)

    // alpha (2) != rank (4): pairs would get 0.5x, deltas must get EXACTLY
    // userScale (Codex finding 8).
    let delta = MLXArray.ones([8]) * 3.0
    let weights = makeDeltaWeights(["scale": .diff(delta)], rank: 4, alpha: 2)

    let session = LoRAPatchSession(module: toy)
    try session.apply(weights: weights, scale: 0.5)

    let after = toy(x)
    eval(after)
    // scale param went 1.0 -> 1.0 + 0.5*3.0 = 2.5; output scales by 2.5.
    let ratio = (after / before).mean().item(Float.self)
    XCTAssertEqual(ratio, 2.5, accuracy: 1e-3,
      "delta must apply as userScale × delta with NO alpha/rank scaling")
  }

  func testDiffBiasPatchesTheRealBiasOfABiasedLinear() throws {
    let toy = ToyBlock()
    let biasBefore = toy.lin.bias!.asArray(Float.self)

    let weights = makeDeltaWeights(["lin.bias": .diffBias(MLXArray.ones([8]))])
    let session = LoRAPatchSession(module: toy)
    try session.apply(weights: weights, scale: 1.0)

    let biasAfter = toy.lin.bias!.asArray(Float.self)
    for i in 0..<8 {
      XCTAssertEqual(biasAfter[i], biasBefore[i] + 1.0, accuracy: 1e-5)
    }
  }

  func testSetWeightReplacesAndIgnoresUserScale() throws {
    let toy = ToyBlock()
    let replacement = MLXArray.ones([8]) * 7.0
    let weights = makeDeltaWeights(["scale": .setWeight(replacement)])

    let session = LoRAPatchSession(module: toy)
    try session.apply(weights: weights, scale: 0.0)  // scale must be IGNORED

    let scaleNow = toy.scale.asArray(Float.self)
    XCTAssertEqual(scaleNow[0], 7.0, accuracy: 1e-6,
      "set_weight replaces outright, even at userScale 0")
  }

  func testMissingTargetThrowsBeforeAnyMutation() throws {
    let toy = ToyBlock()
    let scaleBefore = toy.scale.asArray(Float.self)

    let weights = makeDeltaWeights([
      "scale": .diff(MLXArray.ones([8])),
      "nonexistent.param": .diff(MLXArray.ones([8])),
    ])

    let session = LoRAPatchSession(module: toy)
    XCTAssertThrowsError(try session.apply(weights: weights, scale: 1.0)) { error in
      XCTAssertTrue(String(describing: error).contains("nonexistent.param"),
        "preflight must name the missing key")
    }

    // Preflight failure means NOTHING mutated — including the resolvable key.
    let scaleAfter = toy.scale.asArray(Float.self)
    XCTAssertEqual(scaleBefore, scaleAfter, "no mutation before a failed preflight")
  }

  func testApplyClearApplyIsByteIdenticalAndStackedFirstWriteWins() throws {
    let toy = ToyBlock()
    let base = toy.scale.asArray(Float.self)

    let session = LoRAPatchSession(module: toy)
    // Two LoRAs patching the SAME parameter (Codex finding 3).
    try session.apply(weights: makeDeltaWeights(["scale": .diff(MLXArray.ones([8]))]), scale: 1.0)
    try session.apply(weights: makeDeltaWeights(["scale": .diff(MLXArray.ones([8]) * 2.0)]), scale: 1.0)

    let stacked = toy.scale.asArray(Float.self)
    XCTAssertEqual(stacked[0], base[0] + 3.0, accuracy: 1e-5, "deltas accumulate additively")

    session.clear()
    let restored = toy.scale.asArray(Float.self)
    XCTAssertEqual(restored, base,
      "clear must restore the TRUE base (first-write-wins), not an intermediate")

    // The full cycle must be repeatable.
    try session.apply(weights: makeDeltaWeights(["scale": .diff(MLXArray.ones([8]))]), scale: 1.0)
    session.clear()
    XCTAssertEqual(toy.scale.asArray(Float.self), base)
  }

  func testSnapshotIsDetachedNotAnAlias() throws {
    // MLXArray params are references mutated in place by Module.update —
    // a snapshot that aliases the parameter restores garbage (finding 3).
    let toy = ToyBlock()
    let base = toy.scale.asArray(Float.self)

    let session = LoRAPatchSession(module: toy)
    try session.apply(weights: makeDeltaWeights(["scale": .diff(MLXArray.ones([8]) * 5.0)]), scale: 1.0)
    // Mutate again THROUGH the module (as later pipeline stages might).
    try session.apply(weights: makeDeltaWeights(["scale": .diff(MLXArray.ones([8]) * 5.0)]), scale: 1.0)

    session.clear()
    XCTAssertEqual(toy.scale.asArray(Float.self), base)
  }

  func testQuantizedTargetRestoresExactPackedTuple() throws {
    let lin = Linear(64, 64, bias: false)
    let quantized = QuantizedLinear(weight: lin.weight, bias: nil, groupSize: 64, bits: 8)
    final class QToy: Module {
      @ModuleInfo(key: "qlin") var qlin: QuantizedLinear
      init(_ q: QuantizedLinear) { self._qlin.wrappedValue = q; super.init() }
    }
    let toy = QToy(quantized)

    let packedBefore = toy.qlin.weight.asArray(UInt32.self)
    let scalesBefore = toy.qlin.scales.asArray(Float16.self)

    let session = LoRAPatchSession(module: toy)
    let delta = MLXArray.ones([64, 64]) * 0.01
    try session.apply(weights: makeDeltaWeights(["qlin.weight": .diff(delta)]), scale: 1.0)

    // Applying must actually change the packed representation…
    XCTAssertNotEqual(toy.qlin.weight.asArray(UInt32.self), packedBefore)

    // …and clear must restore weight AND scales EXACTLY, not via requantize
    // (finding 5: requantizing a snapshot drifts across cycles).
    session.clear()
    XCTAssertEqual(toy.qlin.weight.asArray(UInt32.self), packedBefore)
    XCTAssertEqual(toy.qlin.scales.asArray(Float16.self), scalesBefore)
  }

  // MARK: - Integration (real Kroma, skipped when Bolt is absent)

  /// The end-to-end guarantee for task #16: EVERY tensor in the real Kroma
  /// file either binds or the load/preflight throws. 687 tensors = 264
  /// lora_A + 264 lora_B (→ pairs) + 159 .diff (→ deltas). Preflight runs
  /// against a lazily-initialized full-size Krea2 DiT — shape metadata only,
  /// nothing is evaluated.
  func testRealKromaLoadsCompletelyAndPreflightsAgainstKrea2DiT() throws {
    // Kroma's production home (the Bolt copy is unreadable from the xctest
    // sandbox — TCC Removable Volumes grants don't extend to test runners).
    let kromaURL = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".comfybox/loras/kroma-v0.1.safetensors")
    try XCTSkipUnless(
      FileManager.default.fileExists(atPath: kromaURL.path),
      "Kroma file not present on this machine")

    let weights = try LoRAWeightLoader.loadForKrea2(from: kromaURL)

    XCTAssertEqual(weights.weights.count, 264, "all adapter pairs load")
    XCTAssertEqual(weights.deltas.count, 159, "all bare deltas load — the 38% ComfyBox used to drop")

    let dit = Krea2SingleStreamDiT(cfg: Krea2Config())
    let session = LoRAPatchSession(module: dit)
    let resolved = try session.preflight(weights: weights)
    XCTAssertEqual(resolved, 159, "every delta resolves to a real Krea2 parameter path")
  }

  func testTransposedTwoDimensionalDeltaAutoCorrects() throws {
    final class RectToy: Module {
      @ModuleInfo(key: "lin") var lin: Linear
      override init() { self._lin.wrappedValue = Linear(4, 8, bias: false); super.init() }
    }
    let toy = RectToy()
    let before = toy.lin.weight.asArray(Float.self)

    // Weight is [8,4]; deliver the delta as [4,8] (mlx-chroma defensive case).
    let transposed = MLXArray.ones([4, 8]) * 0.5
    let session = LoRAPatchSession(module: toy)
    try session.apply(weights: makeDeltaWeights(["lin.weight": .diff(transposed)]), scale: 1.0)

    let after = toy.lin.weight.asArray(Float.self)
    for i in 0..<before.count {
      XCTAssertEqual(after[i], before[i] + 0.5, accuracy: 1e-5)
    }
  }
}
