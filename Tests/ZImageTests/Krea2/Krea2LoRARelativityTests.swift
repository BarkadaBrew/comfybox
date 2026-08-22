import Foundation
import MLX
import XCTest

@testable import ZImage

/// WP-E6 (FDD §3.6, AC-41): LoRA relativity. A Krea-2 LoRA is extracted
/// against ONE base (`raw` or `turbo`) and is only meaningful on that base.
/// The declaration is never inferred from the file's contents: it comes from
/// `LoRAConfiguration.requiresBase` (declared on the request), else the
/// library entry's `krea2_relative`, else the seeded table below.
///
/// The pipeline-level half of AC-41 (kroma-lora-v0.3 on a loaded `.raw`
/// pipeline throws `incompatibleBase` and `loadedLoRAConfigs.isEmpty`
/// afterwards) needs a 26 GB checkpoint and lives in the integration phase;
/// the guard itself is a pure function, pinned here without weights.
final class Krea2LoRARelativityTests: XCTestCase {

  private var scratch: URL!

  override func setUpWithError() throws {
    scratch = FileManager.default.temporaryDirectory
      .appending(path: "krea2-relativity-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: scratch)
  }

  // MARK: - Seeded relativities (§3.6)

  func testSeededRelativities() {
    XCTAssertEqual(Krea2LoRARelativity.seeded(forFilename: "krea2_turbo_lora_rank_64_bf16.safetensors"), .raw)
    XCTAssertEqual(Krea2LoRARelativity.seeded(forFilename: "kroma-v0.2-base-lora-rank-384-fro-0985.safetensors"), .raw)
    XCTAssertEqual(Krea2LoRARelativity.seeded(forFilename: "kroma-lora-v0.3.safetensors"), .turbo)
    XCTAssertEqual(
      Krea2LoRARelativity.seeded(forFilename: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors"), .raw,
      "WP-E7: the second Raw-relative kroma extraction. Seeded BY NAME before the file lands so "
        + "its first load is guarded, and so it is never confused with the turbo-relative kroma-lora-v0.3.")
    XCTAssertEqual(Krea2LoRARelativity.seeded(forFilename: "kroma-v0.1.safetensors"), .turbo,
                   "v2 addition — the three live krea-film-* presets carry kroma-v0.1")
    // Stem match, not a substring match, and extension-insensitive.
    XCTAssertEqual(Krea2LoRARelativity.seeded(forFilename: "kroma-lora-v0.3"), .turbo)
    XCTAssertNil(Krea2LoRARelativity.seeded(forFilename: "kroma-lora-v0.3-extra.safetensors"))
    XCTAssertNil(Krea2LoRARelativity.seeded(forFilename: "purelens_krea2.safetensors"),
                 "an unseeded LoRA declares nothing — never inferred")
  }

  /// WP-E7: the two files whose names both say "v0.3" are relative to
  /// OPPOSITE bases — `kroma-lora-v0.3` is the Turbo delta (170 `.diff`
  /// keys authored against Turbo's norms), `kroma-v0.3-base-lora-…` is the
  /// Raw-relative extraction. A substring match would collapse them and
  /// silently let the Turbo one onto Raw, so pin the discrimination.
  func testTheTwoV03KromaFilesResolveToOppositeBases() {
    XCTAssertEqual(Krea2LoRARelativity.seeded(forFilename: "kroma-lora-v0.3.safetensors"), .turbo)
    XCTAssertEqual(
      Krea2LoRARelativity.seeded(forFilename: "kroma-v0.3-base-lora-rank-384-fro-0985.safetensors"), .raw)
    // Full paths resolve on the last component, and the match is case-insensitive.
    let url = URL(fileURLWithPath: "/x/y/KROMA-V0.3-BASE-LORA-RANK-384-FRO-0985.SAFETENSORS")
    XCTAssertEqual(
      Krea2LoRARelativity.required(for: LoRAConfiguration(source: .local(url)), resolvedURL: url), .raw)
    // …and the Raw seed does not leak onto a differently-ranked sibling.
    XCTAssertNil(Krea2LoRARelativity.seeded(forFilename: "kroma-v0.3-base-lora-rank-128-fro-0985.safetensors"))
  }

  // MARK: - The guard

  func testTurboRelativeLoRAOnRawThrowsIncompatibleBase() {
    XCTAssertThrowsError(
      try Krea2LoRARelativity.check(lora: "kroma-lora-v0.3", required: .turbo, loaded: .raw)
    ) { error in
      guard case LoRAError.incompatibleBase(let lora, let requires, let loaded) = error else {
        return XCTFail("expected incompatibleBase, got \(error)")
      }
      XCTAssertEqual(lora, "kroma-lora-v0.3")
      XCTAssertEqual(requires, .turbo)
      XCTAssertEqual(loaded, .raw)
      XCTAssertTrue(error.localizedDescription.contains("raw"), error.localizedDescription)
      XCTAssertTrue(error.localizedDescription.contains("turbo"), error.localizedDescription)
    }
  }

  func testRawRelativeLoRAOnTurboThrows() {
    XCTAssertThrowsError(
      try Krea2LoRARelativity.check(lora: "krea2_turbo_lora_rank_64_bf16", required: .raw, loaded: .turbo))
  }

  func testMatchingOrUndeclaredRelativityPasses() throws {
    try Krea2LoRARelativity.check(lora: "kroma-lora-v0.3", required: .turbo, loaded: .turbo)
    try Krea2LoRARelativity.check(lora: "krea2_turbo_lora_rank_64_bf16", required: .raw, loaded: .raw)
    try Krea2LoRARelativity.check(lora: "purelens_krea2", required: nil, loaded: .raw)
    try Krea2LoRARelativity.check(lora: "purelens_krea2", required: nil, loaded: .turbo)
  }

  /// `config.requiresBase ?? libraryEntry?.krea2Relative ?? seeded` — the
  /// declaration on the request wins over the seed; the seed fills in when
  /// nothing is declared.
  func testDeclaredRequirementWinsOverSeed() {
    let url = URL(fileURLWithPath: "/loras/kroma-lora-v0.3.safetensors")
    let declared = LoRAConfiguration(source: .local(url), scale: 0.6, requiresBase: .raw)
    XCTAssertEqual(Krea2LoRARelativity.required(for: declared, resolvedURL: url), .raw)

    let undeclared = LoRAConfiguration(source: .local(url), scale: 0.6)
    XCTAssertNil(undeclared.requiresBase)
    XCTAssertEqual(Krea2LoRARelativity.required(for: undeclared, resolvedURL: url), .turbo)

    let unknown = URL(fileURLWithPath: "/loras/purelens_krea2.safetensors")
    XCTAssertNil(Krea2LoRARelativity.required(
      for: LoRAConfiguration(source: .local(unknown)), resolvedURL: unknown))
  }

  // MARK: - Library entry: tolerant decode, seeded on scan, patchable

  private func entryJSON(relative: String?) -> Data {
    var fields: [String: Any] = [
      "id": "kroma-lora-v0.3", "filename": "kroma-lora-v0.3.safetensors",
      "relative_path": "kroma-lora-v0.3.safetensors", "size_bytes": 1, "model_compatibility": ["krea2"],
      "format": "lora", "rank": 256, "key_count": 690, "layer_targets": ["attention"],
      "triggerwords": [], "recommended_scale": 1.0, "scale_range": [0.0, 2.0], "tags": [],
      "category": "vault", "notes": "", "date_added": "2026-08-22", "quarantined": false,
    ]
    if let relative { fields["krea2_relative"] = relative }
    return try! JSONSerialization.data(withJSONObject: fields)
  }

  func testLibraryEntryDecodesKrea2RelativeTolerantly() throws {
    let dec = JSONDecoder()
    XCTAssertEqual(try dec.decode(LoRALibraryEntry.self, from: entryJSON(relative: "raw")).krea2Relative, .raw)
    XCTAssertEqual(try dec.decode(LoRALibraryEntry.self, from: entryJSON(relative: "turbo")).krea2Relative, .turbo)
    XCTAssertNil(try dec.decode(LoRALibraryEntry.self, from: entryJSON(relative: nil)).krea2Relative,
                 "absent → nil")
    XCTAssertNil(try dec.decode(LoRALibraryEntry.self, from: entryJSON(relative: "bogus")).krea2Relative,
                 "an unknown value must not poison the whole library.json — tolerant decode, nil")

    // Round-trips on the wire as `krea2_relative`.
    var entry = try dec.decode(LoRALibraryEntry.self, from: entryJSON(relative: nil))
    entry.krea2Relative = .turbo
    let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(entry)) as? [String: Any]
    XCTAssertEqual(encoded?["krea2_relative"] as? String, "turbo")
  }

  func testLibraryScanSeedsRelativityAndClassifiesKrea2() throws {
    // A tiny but real Krea-2-shaped LoRA under a seeded filename.
    let root = scratch.appending(path: "loras")
    try FileManager.default.createDirectory(at: root.appending(path: "vault"), withIntermediateDirectories: true)
    try MLX.save(
      arrays: [
        "diffusion_model.blocks.0.attn.wq.lora_A": MLXArray.zeros([4, 16]),
        "diffusion_model.blocks.0.attn.wq.lora_B": MLXArray.zeros([16, 4]),
      ],
      url: root.appending(path: "vault/kroma-lora-v0.3.safetensors"))
    try MLX.save(
      arrays: [
        "diffusion_model.blocks.0.attn.wq.lora_A": MLXArray.zeros([4, 16]),
        "diffusion_model.blocks.0.attn.wq.lora_B": MLXArray.zeros([16, 4]),
      ],
      url: root.appending(path: "vault/purelens_krea2.safetensors"))

    let library = try LoRALibrary(root: root, logger: .init(label: "test"))
    _ = try library.scan()

    let kroma = try XCTUnwrap(library.entry(for: "kroma-lora-v0.3.safetensors"))
    XCTAssertEqual(kroma.krea2Relative, .turbo, "seeded on first scan")
    XCTAssertEqual(kroma.modelCompatibility, ["krea2"], "scanner's Krea-2 branch (§3.6)")

    let unseeded = try XCTUnwrap(library.entry(for: "purelens_krea2.safetensors"))
    XCTAssertNil(unseeded.krea2Relative, "never inferred")
    XCTAssertEqual(unseeded.modelCompatibility, ["krea2"])

    // User declaration through the patch path persists and is read back.
    try library.update(unseeded.id, patch: LoRAEntryPatch(krea2Relative: .raw))
    XCTAssertEqual(library.entry(for: unseeded.id)?.krea2Relative, .raw)
    let reloaded = try LoRALibrary(root: root, logger: .init(label: "test"))
    XCTAssertEqual(reloaded.entry(for: unseeded.id)?.krea2Relative, .raw, "persisted in library.json")
  }
}
