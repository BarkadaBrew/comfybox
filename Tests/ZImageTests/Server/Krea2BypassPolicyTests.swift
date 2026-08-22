// Krea2BypassPolicyTests.swift — WP-E8, the strength policy (FDD §3.8, D10,
// ledger ruling 17:35 / 17:52).
//
// `bypass` is a DECLARED preset dial mirroring `kroma`: `{strength, file?}`,
// where `strength: 0` is a declaration and not an absence. What differs from
// kroma is the default when a preset declares nothing — it is DERIVED:
//
//   kroma.strength >  0  ⇒  bypass 0            (kroma already unlocks)
//   kroma.strength == 0  ⇒  bypass 1.0          (the WORKFLOW's strength)
//
// with an explicit preset `bypass`, and then a per-render override, winning
// in that order. Since 17:52 put kroma on in every mode, the derived default
// is OFF almost everywhere and the dial matters only for the rare kroma-free
// preset — so these tests exercise both branches, not just the live one.
//
// The 1.0 is QUOTED, never chosen: FDD §3.8 ("the preset default is 1.0 — the
// workflow author's figure") and §3.15's `krea2-reference` stack line
// `{ "filename": "krea2filterbypass_2vector.safetensors", "scale": 1.0 }`.

import Foundation
import XCTest

@testable import ZImage

final class Krea2BypassPolicyTests: XCTestCase {

  private func krea2Preset(
    kroma: KromaPolicy?, bypass: BypassPolicy? = nil, family: String? = "raw-stock"
  ) -> ImagePreset {
    ImagePreset(
      id: "p", name: "P", model: "krea2-raw",
      checkpointFamily: family, kroma: kroma, bypass: bypass)
  }

  // MARK: - The quoted constant

  func testTheWorkflowStrengthIsTheDocumentedOne() {
    XCTAssertEqual(Krea2BypassPolicy.workflowStrength, 1.0)
    XCTAssertEqual(Krea2BypassPolicy.workflowFile, "krea2_filter_bypass_2vector.safetensors")
    XCTAssertEqual(Krea2BypassPolicy.fedorFile, "krea2_filter_bypass_fedor.safetensors")
    // Recorded, NOT adopted (§9 Q4): the substitute's author recommends 3–5,
    // 3–5× the workflow's figure. If anyone ever "reconciles" these two
    // numbers into one, this is the test that says which is the recipe's.
    XCTAssertEqual(Krea2BypassPolicy.fedorRecommendedStrength, 3.0...5.0)
    XCTAssertFalse(
      Krea2BypassPolicy.fedorRecommendedStrength.contains(Krea2BypassPolicy.workflowStrength),
      "the two published recommendations are not near each other — that is the point")
  }

  // MARK: - The derived family default (17:35)

  func testKromaOnDerivesBypassOff() {
    for strength in [0.6, 1.0, 0.0001] {
      let resolved = Krea2BypassPolicy.resolve(for: krea2Preset(kroma: KromaPolicy(strength: strength)))
      XCTAssertEqual(resolved.strength, 0, "kroma \(strength) already unlocks — no bypass")
      XCTAssertFalse(resolved.isActive)
    }
  }

  func testKromaOffDerivesTheWorkflowStrength() {
    let resolved = Krea2BypassPolicy.resolve(for: krea2Preset(kroma: KromaPolicy(strength: 0)))
    XCTAssertEqual(resolved.strength, Krea2BypassPolicy.workflowStrength)
    XCTAssertTrue(resolved.isActive)
    XCTAssertEqual(resolved.file, Krea2BypassPolicy.workflowFile,
                   "the derived default names the WORKFLOW's artifact, not the substitute")
  }

  // MARK: - Explicit beats derived

  func testAnExplicitPresetBypassWinsOverTheDerivedDefault() {
    // kroma is ON (derived would be 0) but the preset says otherwise.
    let on = Krea2BypassPolicy.resolve(
      for: krea2Preset(kroma: KromaPolicy(strength: 0.6), bypass: BypassPolicy(strength: 2.0)))
    XCTAssertEqual(on.strength, 2.0)

    // kroma is OFF (derived would be 1.0) but the preset declares none.
    let off = Krea2BypassPolicy.resolve(
      for: krea2Preset(kroma: KromaPolicy(strength: 0), bypass: BypassPolicy(strength: 0)))
    XCTAssertEqual(off.strength, 0, "`strength: 0` is a DECLARATION, not an absence")
    XCTAssertFalse(off.isActive)
  }

  func testAnExplicitFileIsHonouredAndTheDefaultIsNamedOtherwise() {
    let fedor = Krea2BypassPolicy.resolve(
      for: krea2Preset(
        kroma: KromaPolicy(strength: 0),
        bypass: BypassPolicy(strength: 4.0, file: Krea2BypassPolicy.fedorFile)))
    XCTAssertEqual(fedor.strength, 4.0)
    XCTAssertEqual(fedor.file, Krea2BypassPolicy.fedorFile)

    let defaulted = Krea2BypassPolicy.resolve(
      for: krea2Preset(kroma: KromaPolicy(strength: 0), bypass: BypassPolicy(strength: 1.0)))
    XCTAssertEqual(defaulted.file, Krea2BypassPolicy.workflowFile,
                   "the resolved policy always names WHICH artifact — provenance is never ambiguous")
  }

  // MARK: - Per-render override beats everything

  func testAPerRenderOverrideWins() {
    let overDerived = Krea2BypassPolicy.resolve(
      for: krea2Preset(kroma: KromaPolicy(strength: 0.6)), requestStrength: 1.5)
    XCTAssertEqual(overDerived.strength, 1.5)

    let overExplicit = Krea2BypassPolicy.resolve(
      for: krea2Preset(kroma: KromaPolicy(strength: 0), bypass: BypassPolicy(strength: 1.0)),
      requestStrength: 0)
    XCTAssertEqual(overExplicit.strength, 0, "an override of 0 turns it off, and is not read as absent")

    // The override keeps the preset's chosen artifact.
    let keepsFile = Krea2BypassPolicy.resolve(
      for: krea2Preset(
        kroma: KromaPolicy(strength: 0),
        bypass: BypassPolicy(strength: 1.0, file: Krea2BypassPolicy.fedorFile)),
      requestStrength: 3.0)
    XCTAssertEqual(keepsFile.file, Krea2BypassPolicy.fedorFile)
    XCTAssertEqual(keepsFile.strength, 3.0)
  }

  // MARK: - Fail closed off the krea2 family

  /// The derived rule reads "no kroma ⇒ bypass on". A `zimage-*` preset has
  /// no kroma dial at all (D14 exempts it), so the preset-level entry point
  /// must NOT read that absence as "kroma off" and switch a Krea-2-only
  /// adapter on for a model that has no `txtfusion.projector`.
  func testANonKrea2PresetNeverDerivesABypass() {
    let zimage = ImagePreset(
      id: "z", name: "Z", model: "z-image-turbo", checkpointFamily: "zimage-turbo")
    XCTAssertEqual(Krea2BypassPolicy.resolve(for: zimage).strength, 0)

    let noModel = ImagePreset(id: "n", name: "N")
    XCTAssertEqual(Krea2BypassPolicy.resolve(for: noModel).strength, 0)

    let video = ImagePreset(id: "v", name: "V", mediaKind: "video", model: "krea2-raw",
                            kroma: KromaPolicy(strength: 0))
    XCTAssertEqual(Krea2BypassPolicy.resolve(for: video).strength, 0)

    // An EXPLICIT declaration is still honoured anywhere — the guard is about
    // the derived default only.
    let explicitZ = ImagePreset(
      id: "z2", name: "Z2", model: "z-image-turbo", checkpointFamily: "zimage-turbo",
      bypass: BypassPolicy(strength: 1.0))
    XCTAssertEqual(Krea2BypassPolicy.resolve(for: explicitZ).strength, 1.0)
  }

  /// The pure core, exercised directly — it is what the daemon-side family
  /// table (C8) mirrors.
  func testThePureResolverMatchesTheTable() {
    let cases: [(KromaPolicy?, BypassPolicy?, Double?, Double)] = [
      (KromaPolicy(strength: 0.6), nil, nil, 0),
      (KromaPolicy(strength: 0), nil, nil, 1.0),
      (nil, nil, nil, 1.0),                                  // "no kroma dial" == kroma off
      (KromaPolicy(strength: 0.6), BypassPolicy(strength: 5), nil, 5),
      (KromaPolicy(strength: 0.6), nil, 2.0, 2.0),
      (KromaPolicy(strength: 0), BypassPolicy(strength: 0), 3.0, 3.0),
    ]
    for (kroma, bypass, override, expected) in cases {
      XCTAssertEqual(
        Krea2BypassPolicy.resolve(bypass: bypass, kroma: kroma, requestStrength: override).strength,
        expected,
        "kroma=\(String(describing: kroma?.strength)) bypass=\(String(describing: bypass?.strength)) override=\(String(describing: override))")
    }
  }

  // MARK: - Validation (kroma's ranges)

  private func makeTempPath() throws -> URL {
    let dir = FileManager.default.temporaryDirectory
      .appending(path: "bypass-presets-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return dir.appending(path: "presets.json")
  }

  func testValidationMirrorsKroma() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)

    for bad in [-0.5, Double.nan, Double.infinity] {
      XCTAssertThrowsError(
        try store.upsert(krea2Preset(kroma: KromaPolicy(strength: 0), bypass: BypassPolicy(strength: bad)))
      ) { error in
        guard case PresetStoreError.validation(let message) = error else { return XCTFail("\(error)") }
        XCTAssertTrue(message.contains("bypass.strength"), message)
      }
    }

    XCTAssertThrowsError(
      try store.upsert(krea2Preset(kroma: KromaPolicy(strength: 0), bypass: BypassPolicy(strength: 1, file: "  ")))
    ) { error in
      guard case PresetStoreError.validation(let message) = error else { return XCTFail("\(error)") }
      XCTAssertTrue(message.contains("bypass.file"), message)
    }

    // A large strength is allowed — Fedor's own guidance goes to 5.
    XCTAssertNoThrow(
      try store.upsert(krea2Preset(kroma: KromaPolicy(strength: 0), bypass: BypassPolicy(strength: 5))))
  }

  /// Unlike `kroma` (O4a), an ABSENT `bypass` is not a configuration error —
  /// it is the derived default, which is the whole point of the 17:35 ruling.
  func testAnAbsentBypassIsNotAConfigurationError() throws {
    let store = PresetStore(path: try makeTempPath(), seedDefaults: false)
    XCTAssertNoThrow(try store.upsert(krea2Preset(kroma: KromaPolicy(strength: 0.6))))
    XCTAssertNil(store.get("p")?.bypass)
    XCTAssertEqual(Krea2BypassPolicy.resolve(for: try XCTUnwrap(store.get("p"))).strength, 0)
  }

  // MARK: - The five sites (the `videoTuning` regression class)

  func testBypassRoundTripsAtEverySite() throws {
    let bypass = BypassPolicy(strength: 1.0, file: Krea2BypassPolicy.workflowFile)
    let preset = ImagePreset(
      id: "ref", name: "Reference", model: "krea2-raw",
      checkpointFamily: "raw-stock", kroma: KromaPolicy(strength: 0), bypass: bypass)

    // 1. memberwise init
    XCTAssertEqual(preset.bypass, bypass)

    // 2. encoder (CodingKeys)
    let data = try JSONEncoder().encode(preset)
    let text = try XCTUnwrap(String(data: data, encoding: .utf8))
    XCTAssertTrue(text.contains("\"bypass\""), "encoder dropped bypass: \(text)")

    // 3. decoder (custom init(from:))
    XCTAssertEqual(try JSONDecoder().decode(ImagePreset.self, from: data), preset)

    // Wire shape, and absence staying absent.
    let wire = try JSONDecoder().decode(ImagePreset.self, from: Data(#"""
    {"id":"w","name":"W","model":"krea2-raw","kroma":{"strength":0},"bypass":{"strength":1}}
    """#.utf8))
    XCTAssertEqual(wire.bypass, BypassPolicy(strength: 1, file: nil))
    let bare = try JSONDecoder().decode(ImagePreset.self, from: Data(#"{"id":"r","name":"R"}"#.utf8))
    XCTAssertNil(bare.bypass)

    // 4. store round trip, 5. ResolvedPreset
    let path = try makeTempPath()
    let store = PresetStore(path: path, seedDefaults: false)
    try store.upsert(preset)
    let reopened = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(reopened.get("ref")?.bypass, bypass)
    XCTAssertEqual(try reopened.resolve("ref").bypass, bypass)
    XCTAssertNil(ResolvedPreset(preset: ImagePreset(id: "p", name: "P")).bypass)
  }
}
