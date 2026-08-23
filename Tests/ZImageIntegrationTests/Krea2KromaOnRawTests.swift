// Krea2KromaOnRawTests.swift — WP-E7 (FDD §3.7, AC-40 / AC-41 / AC-42) at
// PRODUCTION config: the real Raw checkpoint at the deployed q8, the real
// kroma files off disk, `Krea2Pipeline.loadLoRAs`' strict apply.
//
// §3.7 is "zero new code": the claim is that a Raw-RELATIVE kroma extraction
// binds completely on Raw — every offered pair lands on a module, nothing is
// shape-rejected, and `deltas_applied: 0` records D15's fidelity gap on the
// face of the render. This is the test that proves it against the files
// rather than against a spreadsheet.
//
// Three things are pinned here:
//   1. AC-40 — each Raw-relative kroma file binds `bound == offered`, with
//      `deltasApplied` equal to the number of bare-delta keys the FILE
//      carries (0 for v0.2; > 0 for the v0.3-base extraction, which ships
//      `.diff`/`.diff_b` alongside its pairs). Read from the safetensors
//      header, never hard-coded, and skipped by name when a file is absent.
//   2. AC-41 (negative) — the TURBO-relative `kroma-lora-v0.3` is refused on
//      Raw with `incompatibleBase`, BEFORE any weight is touched.
//   3. AC-42 at the pipeline — a really-truncated kroma subset is refused and
//      rolled back; the base is byte-identical afterwards.
//
// Loading Raw is ~22 GB; the pipeline is built ONCE for the class. Run by
// hand (FDD §5.3), skipped with a named message when the weights are absent.

import Foundation
import MLX
import XCTest
@testable import ZImage

final class Krea2KromaOnRawTests: XCTestCase {

  // MARK: - Files

  private static let loraRoots = ["~/comfybox-models/loras/vault", "~/comfybox-models/loras"]
    .map { ($0 as NSString).expandingTildeInPath }

  /// The Raw-relative kroma extractions, by the name their relativity is
  /// seeded under (`Krea2LoRARelativity.seeded`). The v0.3-base file is
  /// produced by an offline extraction and may not be on disk yet — its case
  /// skips by name rather than being silently dropped.
  private static let rawRelativeKroma = [
    "kroma-v0.2-base-lora-rank-384-fro-0985.safetensors",
    "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors",
  ]

  /// The Turbo-relative kroma — must keep being REFUSED on Raw.
  private static let turboRelativeKroma = "kroma-lora-v0.3.safetensors"

  private static func locate(_ filename: String) -> URL? {
    for root in loraRoots {
      let candidate = URL(fileURLWithPath: root).appending(path: filename)
      if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
    }
    return nil
  }

  // MARK: - The Raw pipeline (built once)

  private static var sharedRaw: Krea2Pipeline?
  /// The model directory is genuinely absent → every test in the class skips.
  private static var rawSkip: XCTSkip?
  /// The directory is THERE and `Krea2Pipeline` init threw anyway → every
  /// test in the class FAILS. A quantization fault, a shape mismatch or an
  /// OOM is a regression this batch exists to catch; it must never read as
  /// "the model is not installed".
  private static var rawInitFailure: Error?

  private func rawPipeline() throws -> Krea2Pipeline {
    if ProcessInfo.processInfo.environment["CI"] != nil { throw XCTSkip("GPU test skipped in CI") }
    if let existing = Self.sharedRaw { return existing }
    if let skip = Self.rawSkip { throw skip }
    if let failure = Self.rawInitFailure {
      XCTFail("Krea2Pipeline init already failed once on an INSTALLED krea2-raw — real regression: \(failure)")
      throw failure
    }

    // 1. Resolution only. A `Krea2ModelPathsError` means the directory is
    //    absent / unusable, which is the ONLY skippable condition here.
    let paths: Krea2ModelPaths
    do {
      paths = try Krea2ModelDetection.resolve(spec: "krea2-raw")
    } catch let error as Krea2ModelPathsError {
      let skip = XCTSkip("krea2-raw not installed (\(error)) — WP-E7 Raw batch not runnable here")
      Self.rawSkip = skip
      throw skip
    }
    guard FileManager.default.fileExists(atPath: paths.transformerFile.path) else {
      let skip = XCTSkip("\(paths.transformerFile.path) absent — WP-E7 Raw batch not runnable here")
      Self.rawSkip = skip
      throw skip
    }
    XCTAssertEqual(paths.variant, .raw)
    XCTAssertEqual(paths.transformerFile.lastPathComponent, "raw.safetensors")

    // 2. Construction. The weights ARE on disk, so anything thrown here is a
    //    real fault in the pipeline — fail, never skip.
    do {
      // Production quantisation (ModelPool / prepare both pass 8).
      let pipeline = try Krea2Pipeline(paths: paths, quantizeTransformer: 8)
      Self.sharedRaw = pipeline
      return pipeline
    } catch {
      Self.rawInitFailure = error
      XCTFail("Krea2Pipeline(paths:quantizeTransformer: 8) failed with krea2-raw INSTALLED at "
        + "\(paths.transformerFile.path) — this is a regression (quantization, shape or memory), "
        + "not a missing model: \(error)")
      throw error
    }
  }

  /// `loadLoRAs` is async; the rest of the pipeline API is not. Same shape as
  /// `Krea2RecipeProvenanceTests`, with the error handed back to the caller
  /// so `XCTAssertThrowsError` can inspect it.
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

  // MARK: - Header facts (never hard-coded)

  private struct FileFacts {
    let pairs: Int
    let deltaKeys: Int
    let names: [String]
  }

  /// What the file itself offers, read straight from the safetensors header:
  /// one pair per `lora_A`/`lora_down` key, and every bare delta
  /// (`.diff` / `.diff_b` / `.set_weight`) the Krea-2 loader will hand to
  /// `LoRAPatchSession`.
  private func facts(of url: URL) throws -> FileFacts {
    let names = try SafeTensorsReader(fileURL: url).tensorNames
    func isDown(_ n: String) -> Bool {
      let bare = n.hasSuffix(".weight") ? String(n.dropLast(".weight".count)) : n
      return bare.hasSuffix(".lora_A") || bare.hasSuffix(".lora_down")
    }
    let deltaSuffixes = [".diff", ".diff_b", ".set_weight"]
    return FileFacts(
      pairs: names.filter(isDown).count,
      deltaKeys: names.filter { n in deltaSuffixes.contains(where: { n.hasSuffix($0) }) }.count,
      names: names)
  }

  /// A fingerprint of the resident transformer's parameters, used to prove a
  /// refusal touched NOTHING. EVERY parameter is summed, not a sample: the
  /// bare deltas a Krea-2 LoRA carries land on a handful of norm/modulation
  /// params out of thousands, and a sampled fingerprint would be free to miss
  /// all of them and read as "untouched" when it was not. Its sensitivity is
  /// itself asserted by `testFingerprintDetectsARealWeightMutation`.
  ///
  /// Note what this does and does not cover: dynamic LoRA adapters do not
  /// mutate base parameters at all (they hang off `LoRALinear` /
  /// `LoRAQuantizedLinear`), so the fingerprint catches DELTA application and
  /// `LoRAApplicator.hasDynamicLoRA` catches bound adapters. Both are
  /// asserted at every refusal below.
  private func fingerprint(_ dit: Krea2SingleStreamDiT) -> [String: Float] {
    var out: [String: Float] = [:]
    for (key, array) in dit.parameters().flattened() {
      out[key] = array.asType(.float32).sum().item(Float.self)
    }
    return out
  }

  // MARK: - AC-40: the Raw-relative kroma files bind completely on Raw

  /// AC-40 for `kroma-v0.2` — the file §3.7's numbers were verified against.
  func testKromaV02BindsCompletelyOnRaw() throws {
    let report = try assertBindsCompletelyOnRaw(Self.rawRelativeKroma[0])
    XCTAssertEqual(report.offered, 256, "§3.7: kroma-v0.2 offers 256 pairs")
    XCTAssertEqual(report.bound, 256, "§3.7: 256/256")
    XCTAssertEqual(report.deltasApplied, 0, "D15: kroma-v0.2 carries no bare deltas")
  }

  /// AC-40 for the second Raw-relative extraction. Skipped BY NAME until the
  /// offline extraction lands the file; it ships `.diff`/`.diff_b` bare
  /// deltas beside its pairs, so `deltasApplied` is > 0 here — and is still
  /// asserted against the file's own header count, never a literal.
  func testKromaV03BaseBindsCompletelyOnRaw() throws {
    let report = try assertBindsCompletelyOnRaw(Self.rawRelativeKroma[1])
    XCTAssertGreaterThan(
      report.deltasApplied, 0,
      "the v0.3-base extraction ships .diff/.diff_b beside its pairs")
  }

  /// Every pair the file offers binds a module on Raw: nothing unbound,
  /// nothing shape-rejected, and `deltasApplied` equal to the file's own
  /// bare-delta count. Skips by name when the file is not on disk.
  @discardableResult
  private func assertBindsCompletelyOnRaw(_ name: String) throws -> LoRAApplicationReport {
    guard let url = Self.locate(name) else {
      throw XCTSkip("\(name) not on disk (looked in \(Self.loraRoots.joined(separator: ", "))) — AC-40 skipped by name")
    }
    // The relativity is DECLARED by the seed table, not by the request —
    // that is the path a preset takes, so exercise it.
    XCTAssertEqual(
      Krea2LoRARelativity.seeded(forFilename: name), .raw,
      "\(name) must be seeded Raw-relative before it is ever applied")

    let k2 = try rawPipeline()
    XCTAssertEqual(k2.variant, .raw)
    XCTAssertEqual(k2.transformerQuantBits, 8)

    let f = try facts(of: url)
    XCTAssertGreaterThan(f.pairs, 0, "\(name): no lora_A/lora_down keys in the header")

    let reports = try applyStack(k2, [LoRAConfiguration(source: .local(url), scale: 0.6, role: "kroma")])
    XCTAssertEqual(reports.count, 1, "\(name)")
    let report = try XCTUnwrap(reports.first)
    print("[WP-E7] \(name): offered=\(report.offered) bound=\(report.bound) "
      + "quantizedBound=\(report.quantizedBound) shapeRejected=\(report.shapeRejected) "
      + "deltasApplied=\(report.deltasApplied) unbound=\(report.unbound.count) "
      + "(header: pairs=\(f.pairs) deltaKeys=\(f.deltaKeys))")

    XCTAssertEqual(report.offered, f.pairs, "\(name): every pair in the header must be offered")
    XCTAssertEqual(report.bound, report.offered, "\(name): §3.7 — kroma binds COMPLETELY on Raw")
    XCTAssertEqual(report.unbound, [], "\(name): unbound must be empty")
    XCTAssertEqual(report.shapeRejected, 0, "\(name): F32-onto-q8 is handled by LoRAQuantizedLinear")
    XCTAssertEqual(
      report.quantizedBound, report.bound,
      "\(name): every target on a q8 base is a quantized Linear")
    XCTAssertTrue(report.isComplete, "\(name)")
    XCTAssertEqual(
      report.deltasApplied, f.deltaKeys,
      "\(name): deltas_applied is the file's own bare-delta count (D15) — read, not assumed")

    // The E10 read-back pairs config with report without desyncing.
    let readBacks = try XCTUnwrap(
      RenderRecipe.loRAReadBacks(
        configs: k2.loadedLoRAConfigs, reports: k2.loadedLoRAReports,
        relativities: k2.loadedLoRARelativities),
      "\(name): config/report desync")
    XCTAssertEqual(readBacks.count, 1)
    XCTAssertEqual(readBacks[0].report.bound, report.bound)
    XCTAssertEqual(readBacks[0].configuration.role, "kroma")
    XCTAssertEqual(readBacks[0].configuration.scale, 0.6)
    // K-FIX-1 / Codex I6 — the E7 KNOWN GAP, now closed. This request
    // declares no `requiresBase`; the SEED table supplies `.raw` and the
    // guard enforces it, and the record now names that enforced value
    // instead of the request's silence.
    XCTAssertNil(readBacks[0].configuration.requiresBase,
                 "\(name): the REQUEST still declares nothing — that is the case under test")
    XCTAssertEqual(readBacks[0].resolvedRelativeTo, .raw,
                   "\(name): the pipeline kept the relativity it enforced")
    XCTAssertEqual(
      readBacks[0].relativeTo, .raw,
      "\(name): provenance records relative_to: raw, not null")

    try applyStack(k2, [])
    XCTAssertTrue(k2.loadedLoRAConfigs.isEmpty)
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: k2.transformer))
    return report
  }

  // MARK: - AC-40 second half: no alpha ⇒ the applied scale passes through

  /// §3.7: per-layer rank is dynamic (98–299) and safe because
  /// `effectiveScale(forLayer:)` reads each layer's own dims, and no `.alpha`
  /// keys exist so the requested scale passes through verbatim. Asserted at
  /// the file's real extreme ranks, found in the file rather than assumed.
  func testKromaHasNoAlphaSoEveryLayerScaleIsExactlyOne() throws {
    guard let url = Self.locate(Self.rawRelativeKroma[0]) else {
      throw XCTSkip("\(Self.rawRelativeKroma[0]) not on disk")
    }
    let weights = try LoRAWeightLoader.loadForKrea2(from: url)
    XCTAssertTrue(weights.layerAlphas.isEmpty, "kroma ships no per-layer .alpha tensors")

    var ranks: [String: Int] = [:]
    for (key, pair) in weights.weights where pair.down.ndim == 2 {
      ranks[key] = Swift.min(pair.down.dim(0), pair.down.dim(1))
    }
    let sorted = ranks.sorted { $0.value == $1.value ? $0.key < $1.key : $0.value < $1.value }
    let lowest = try XCTUnwrap(sorted.first)
    let highest = try XCTUnwrap(sorted.last)
    print("[WP-E7] rank spread: \(lowest.key)=\(lowest.value) … \(highest.key)=\(highest.value)")
    XCTAssertNotEqual(lowest.value, highest.value, "the ranks really are dynamic")

    XCTAssertEqual(weights.effectiveScale(forLayer: lowest.key), 1.0,
                   "no alpha ⇒ no rank normalisation at rank \(lowest.value)")
    XCTAssertEqual(weights.effectiveScale(forLayer: highest.key), 1.0,
                   "no alpha ⇒ no rank normalisation at rank \(highest.value)")
    // …and every other layer too, not just the two extremes.
    for key in ranks.keys {
      XCTAssertEqual(weights.effectiveScale(forLayer: key), 1.0, key)
    }
  }

  // MARK: - Positive control for the "nothing was mutated" evidence

  /// The refusal tests below claim the base is untouched by comparing
  /// `fingerprint(...)`. That claim is worth nothing unless the fingerprint
  /// would actually MOVE for a real mutation — so mutate the base for real
  /// and watch it move, then clear and watch it come back.
  ///
  /// The rank-64 turbo LoRA is the instrument: it is Raw-relative (seeded)
  /// and carries 7 bare deltas, which is the only thing in the LoRA path that
  /// writes a base parameter.
  func testFingerprintDetectsARealWeightMutation() throws {
    guard let turbo = Self.locate("krea2_turbo_lora_rank_64_bf16.safetensors") else {
      throw XCTSkip("krea2_turbo_lora_rank_64_bf16.safetensors not on disk — no delta-carrying LoRA to control with")
    }
    let k2 = try rawPipeline()
    try applyStack(k2, [])
    let before = fingerprint(k2.transformer)
    XCTAssertGreaterThan(before.count, 100, "the fingerprint must cover the whole parameter set")

    let reports = try applyStack(k2, [LoRAConfiguration(source: .local(turbo), scale: 1.0, role: "accel")])
    XCTAssertEqual(reports[0].bound, reports[0].offered)
    XCTAssertGreaterThan(reports[0].deltasApplied, 0, "this LoRA must carry deltas or it controls nothing")
    let during = fingerprint(k2.transformer)
    let moved = during.filter { before[$0.key] != $0.value }
    print("[WP-E7] fingerprint control: deltas=\(reports[0].deltasApplied) params moved=\(moved.count)")
    XCTAssertFalse(moved.isEmpty, "a real delta apply MUST move the fingerprint")

    try applyStack(k2, [])
    XCTAssertEqual(fingerprint(k2.transformer), before, "clearing restores every patched parameter")
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: k2.transformer))
  }

  // MARK: - AC-41 (negative): the Turbo-relative kroma is refused on Raw

  /// Todd's target state is Raw-only with the Raw-relative extraction as
  /// kroma; `kroma-lora-v0.3` is the Turbo delta (`L = v0.3base − v0.2base +
  /// v0.2turbo − krea2turbo`, 170 `.diff` keys authored against Turbo's
  /// norms) and applying it to Raw would silently produce a wrong model.
  /// It must be refused, and refused BEFORE any weight is touched.
  func testTurboRelativeKromaIsRefusedOnRawBeforeAnyMutation() throws {
    let k2 = try rawPipeline()
    try applyStack(k2, [])
    let before = fingerprint(k2.transformer)
    XCTAssertFalse(before.isEmpty)

    // The real file when it is on disk (relativity from the seed table);
    // otherwise the same guard driven by an explicit declaration on a file
    // that IS present — `resolveSource` would refuse a fictional path first.
    let config: LoRAConfiguration
    let expectedName: String
    if let url = Self.locate(Self.turboRelativeKroma) {
      XCTAssertEqual(Krea2LoRARelativity.seeded(forFilename: Self.turboRelativeKroma), .turbo)
      config = LoRAConfiguration(source: .local(url), scale: 0.6, role: "kroma")
      expectedName = url.deletingPathExtension().lastPathComponent
    } else {
      guard let stand = Self.locate(Self.rawRelativeKroma[0]) else {
        throw XCTSkip("no kroma file on disk — AC-41's negative case not runnable here")
      }
      print("[WP-E7] \(Self.turboRelativeKroma) not on disk — driving the guard from a declared requiresBase")
      config = LoRAConfiguration(source: .local(stand), scale: 0.6, requiresBase: .turbo, role: "kroma")
      expectedName = stand.deletingPathExtension().lastPathComponent
    }

    XCTAssertThrowsError(try applyStack(k2, [config])) { error in
      guard case LoRAError.incompatibleBase(let lora, let requires, let loaded) = error else {
        return XCTFail("expected incompatibleBase, got \(error)")
      }
      XCTAssertTrue(lora.contains(expectedName), "\(lora) should name \(expectedName)")
      XCTAssertEqual(requires, .turbo)
      XCTAssertEqual(loaded, .raw)
      print("[WP-E7] refusal: \(error.localizedDescription)")
    }

    XCTAssertTrue(k2.loadedLoRAConfigs.isEmpty, "AC-41: rollback held")
    XCTAssertTrue(k2.loadedLoRAReports.isEmpty, "no apply report exists for a refused LoRA")
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: k2.transformer))
    XCTAssertEqual(fingerprint(k2.transformer), before,
                   "the guard runs before the file is even read — the base must be untouched")
  }

  /// The mirror image: the same file with the CORRECT declaration is not
  /// refused, so the guard is discriminating and not merely off.
  func testRawRelativeKromaWithMatchingDeclarationIsAccepted() throws {
    guard let url = Self.locate(Self.rawRelativeKroma[0]) else {
      throw XCTSkip("\(Self.rawRelativeKroma[0]) not on disk")
    }
    let k2 = try rawPipeline()
    let reports = try applyStack(
      k2, [LoRAConfiguration(source: .local(url), scale: 0.3, requiresBase: .raw, role: "kroma")])
    XCTAssertEqual(reports.count, 1)
    XCTAssertEqual(reports[0].bound, reports[0].offered)
    try applyStack(k2, [])
  }

  // MARK: - AC-42 at the pipeline: a really-truncated kroma is refused

  /// Addendum A.2's MAJOR, at the real base. Three small REAL files are cut
  /// out of kroma-v0.2 (real tensors, real dims, real q8 targets):
  ///   * complete   — binds 4/4 under strict;
  ///   * missing one `lora_B` — the literal truncation: refused by the LOADER
  ///     as an orphan pair half, so the applicator is never reached;
  ///   * one pair aimed at a block the DiT does not have — refused by strict
  ///     apply with `partialApplication` naming the key.
  /// After each refusal the stack is empty and the base is byte-identical.
  func testTruncatedKromaSubsetIsRefusedByThePipelineWithRollback() throws {
    guard let source = Self.locate(Self.rawRelativeKroma[0]) else {
      throw XCTSkip("\(Self.rawRelativeKroma[0]) not on disk — AC-42 pipeline case not runnable here")
    }
    let k2 = try rawPipeline()
    try applyStack(k2, [])
    let before = fingerprint(k2.transformer)

    let reader = try SafeTensorsReader(fileURL: source)
    let projections = ["wq", "wk", "wv", "wo"]
    var subset: [String: MLXArray] = [:]
    for proj in projections {
      for half in ["lora_A", "lora_B"] {
        let key = "diffusion_model.blocks.0.attn.\(proj).\(half).weight"
        subset[key] = try reader.tensor(named: key)
      }
    }
    XCTAssertEqual(subset.count, 8)

    let scratch = FileManager.default.temporaryDirectory.appending(path: "wp-e7-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: scratch) }

    // 1. Complete subset — binds every pair under the pipeline's strict apply.
    let completeURL = scratch.appending(path: "kroma-subset-complete.safetensors")
    try MLX.save(arrays: subset, url: completeURL)
    let reports = try applyStack(k2, [LoRAConfiguration(source: .local(completeURL), scale: 0.5)])
    XCTAssertEqual(reports.count, 1)
    XCTAssertEqual(reports[0].offered, 4)
    XCTAssertEqual(reports[0].bound, 4, "the subset's four real pairs bind the real q8 modules")
    XCTAssertEqual(reports[0].deltasApplied, 0)
    try applyStack(k2, [])
    XCTAssertEqual(fingerprint(k2.transformer), before, "clearing restores the base")

    // 2. Truncated: one lora_B dropped. Refused at LOAD as an orphan half —
    //    earlier and harder than AC-42's `partialApplication`, and nothing
    //    downstream can see a half-pair.
    var truncated = subset
    truncated.removeValue(forKey: "diffusion_model.blocks.0.attn.wv.lora_B.weight")
    let truncatedURL = scratch.appending(path: "kroma-subset-truncated.safetensors")
    try MLX.save(arrays: truncated, url: truncatedURL)
    XCTAssertThrowsError(
      try applyStack(k2, [LoRAConfiguration(source: .local(truncatedURL), scale: 0.5)])
    ) { error in
      guard case LoRAError.invalidFormat(let message) = error else {
        return XCTFail("a truncated LoRA must be refused; got \(error)")
      }
      XCTAssertTrue(message.contains("orphan"), message)
      XCTAssertTrue(message.contains("blocks.0.attn.wv.lora_A"), message)
      print("[WP-E7] truncated-file refusal: \(message)")
    }
    XCTAssertTrue(k2.loadedLoRAConfigs.isEmpty)
    XCTAssertTrue(k2.loadedLoRAReports.isEmpty)
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: k2.transformer))
    XCTAssertEqual(fingerprint(k2.transformer), before, "a refused truncation leaves the base untouched")

    // 3. A pair aimed at a block the 28-block DiT does not have — this IS the
    //    `partialApplication` path, driven from a real file at the real base.
    var misTargeted = subset
    misTargeted["diffusion_model.blocks.999.attn.wq.lora_A.weight"] =
      try reader.tensor(named: "diffusion_model.blocks.0.attn.wq.lora_A.weight")
    misTargeted["diffusion_model.blocks.999.attn.wq.lora_B.weight"] =
      try reader.tensor(named: "diffusion_model.blocks.0.attn.wq.lora_B.weight")
    let misTargetedURL = scratch.appending(path: "kroma-subset-mistargeted.safetensors")
    try MLX.save(arrays: misTargeted, url: misTargetedURL)
    XCTAssertThrowsError(
      try applyStack(k2, [LoRAConfiguration(source: .local(misTargetedURL), scale: 0.5)])
    ) { error in
      guard case LoRAError.partialApplication(let name, let unbound) = error else {
        return XCTFail("expected partialApplication, got \(error)")
      }
      XCTAssertEqual(unbound, ["blocks.999.attn.wq.weight"])
      print("[WP-E7] partial-apply refusal: \(name ?? "<unnamed>") unbound=\(unbound)")
    }
    XCTAssertTrue(k2.loadedLoRAConfigs.isEmpty, "AC-42: base restored, nothing recorded")
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: k2.transformer))
    XCTAssertEqual(fingerprint(k2.transformer), before)

    // 4. And a stack that fails at position 3 rolls the WHOLE stack back.
    XCTAssertThrowsError(
      try applyStack(k2, [
        LoRAConfiguration(source: .local(completeURL), scale: 0.5),
        LoRAConfiguration(source: .local(completeURL), scale: 0.5),
        LoRAConfiguration(source: .local(misTargetedURL), scale: 0.5),
        LoRAConfiguration(source: .local(completeURL), scale: 0.5),
      ]))
    XCTAssertTrue(k2.loadedLoRAConfigs.isEmpty)
    XCTAssertFalse(LoRAApplicator.hasDynamicLoRA(in: k2.transformer))
    XCTAssertEqual(fingerprint(k2.transformer), before, "the whole stack rolled back, not just the bad one")
  }

  // MARK: - E7 item 5: does the library scanner classify the kroma files?

  /// Reported, not fixed (WP-E6 owns `detectCompatibilityFromKeys`). A kroma
  /// file the scanner calls `unknown` cannot be filtered to the Krea-2 family
  /// in the library UI, and would not be seeded with a relativity on scan.
  func testScannerClassifiesBothKromaFilesAsKrea2() throws {
    var checked = 0
    for name in Self.rawRelativeKroma + [Self.turboRelativeKroma] {
      guard let url = Self.locate(name) else { continue }
      checked += 1
      let result = try LoRAScanner.analyze(url)
      print("[WP-E7] scanner \(name): compatibility=\(result.compatibility) "
        + "format=\(result.format) rank=\(result.rank) keys=\(result.keyCount)")
      XCTAssertEqual(result.compatibility, ["krea2"], "\(name) misclassified — E7 item 5")
    }
    try XCTSkipIf(checked == 0, "no kroma file on disk to scan")
  }
}
