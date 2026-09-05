// PresetShiftExpansionTests.swift — comfybox#154
//
// A preset has carried a declared `shift` since the Krea 2 recipe work, and
// `PresetStore.validate` has always refused a non-positive one — but nothing
// ever applied it on the image path: `expandingPreset` expanded `model`,
// `loras`, `steps`, `guidance` and `vae` and left `shift` on the shelf. So
// `POST /v1/generate {"preset": "zeta-chroma"}` rendered on the model's own
// schedule while the preset said 3.0, and reported success.
//
// These pin the expansion under the same rule the other declared fields
// follow: DECLARED only, and only when the request said nothing.

import XCTest

@testable import ZImage

final class PresetShiftExpansionTests: XCTestCase {

  /// A Z-Image preset shaped like the one Zeta Chroma wants: euler, shift 3.0.
  /// `checkpoint_family` is what makes the shift expandable — see
  /// `testKrea2PresetShiftIsNotExpanded` for why the gate exists.
  private func zetaChroma(shift: Double? = 3.0, checkpointFamily: String? = "zimage-base")
    -> PresetLoRAStack.Lookup
  {
    let preset = ImagePreset(
      id: "zeta-chroma", name: "Zeta Chroma", mediaKind: "image",
      model: "z-image-zeta-chroma", steps: 28, guidance: 5.0,
      loras: [],
      checkpointFamily: checkpointFamily,
      sampler: "euler", sigmaSchedule: "simple", shift: shift)
    return .resolved(ResolvedPreset(preset: preset), declared: preset)
  }

  /// `krea-kira-avocado` as it stands in the live `~/.comfybox/presets.json`
  /// TODAY: a krea2 raw-stock preset that declares `shift: 1.15`.
  ///
  /// On `main` that number is desktop-display-only — nothing on the image path
  /// reads it. Expanding it for every family would silently turn it into a real
  /// `ModelSamplingFlux` `mu` on Kira's production renders the moment this
  /// deploys. Four live presets are in this state (krea-kira, krea-kira-sfw,
  /// krea-kira-avocado, krea2-base).
  private func kreaKiraAvocado(shift: Double? = 1.15) -> PresetLoRAStack.Lookup {
    let preset = ImagePreset(
      id: "krea-kira-avocado", name: "Kira Avocado", mediaKind: "image",
      model: "krea2-raw", steps: 30, guidance: 3.5,
      loras: [LoraReference(filename: "Girly_Tiana.safetensors", scale: 0.6)],
      checkpointFamily: "raw-stock",
      sampler: "res_2s", sigmaSchedule: "krea2", shift: shift)
    return .resolved(ResolvedPreset(preset: preset), declared: preset)
  }

  private func expansion(_ decision: PresetLoRAStack) throws -> PresetExpansion {
    guard case .apply(let e) = decision else {
      XCTFail("expected .apply, got \(decision)")
      throw NSError(domain: "test", code: 1)
    }
    XCTAssertNil(e.unresolved, "expected an expanded preset, got \(e.unresolved!.code)")
    return e
  }

  func testDeclaredShiftIsAdoptedWhenTheRequestOmitsIt() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "zeta-chroma", lookup: zetaChroma(), requestLoras: nil))
    XCTAssertEqual(e.shift, 3.0)
  }

  func testRequestShiftWins() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "zeta-chroma", lookup: zetaChroma(), requestLoras: nil, requestShift: 1.73))
    XCTAssertNil(e.shift, "the preset's shift is not adopted — the request already named one")
  }

  /// CRITICAL, review r1 (1): a krea2 preset's declared `shift` must NOT be
  /// expanded. `shift` is `mu` on that family — a log-shift into
  /// `ModelSamplingFlux` — and applying the four live presets' 1.15 would
  /// change Kira's renders on deploy. Krea 2 keeps taking its shift from the
  /// REQUEST only, exactly as it did before #154.
  func testKrea2PresetShiftIsNotExpanded() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea-kira-avocado", lookup: kreaKiraAvocado(), requestLoras: nil))
    XCTAssertNil(e.shift, "a krea2 preset's shift must never become mu behind the caller")
    // Everything else the preset contributes is unaffected by the gate.
    XCTAssertEqual(e.model, "krea2-raw")
    XCTAssertEqual(e.steps, 30)
    XCTAssertEqual(e.guidance, 3.5)
    XCTAssertEqual(e.loras?.count, 1)
  }

  /// End-to-end at the `/v1/generate` seam: the payload's mu path is untouched
  /// and nothing can echo an `applied_shift` for that render.
  func testKrea2PresetLeavesThePayloadMuPathUntouched() throws {
    let payload = try decode(#"{"prompt":"a portrait","preset":"krea-kira-avocado"}"#)
    let out = try GeneratePayload.expandingPreset(payload) { _ in self.kreaKiraAvocado() }
    XCTAssertNil(out.shift, "no shift on the payload ⇒ Krea 2's resolution-dependent mu stands")
    XCTAssertNil(out.presetUnresolved, "the preset still expands — only its shift is gated")
    XCTAssertEqual(out.model, "krea2-raw")

    // And the replay body is not rewritten with a shift either.
    let original = #"{"prompt":"a portrait","preset":"krea-kira-avocado"}"#.data(using: .utf8)!
    let rewritten = WarmServer.rawBody(original, expandedWith: out)
    let object = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: rewritten) as? [String: Any])
    XCTAssertNil(object["shift"])

    // The response for such a render carries no `applied_shift` key at all.
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let response = GenerateResponse(
      success: true, outputPath: "/tmp/a.png", durationMs: 10, appliedShift: out.shift)
    let json = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: encoder.encode(response)) as? [String: Any])
    XCTAssertNil(json["applied_shift"])
  }

  /// A krea2 preset's shift is gated whichever way the family is declared —
  /// by `checkpoint_family` (above) or by the `model` spec alone.
  func testKrea2PresetShiftIsNotExpandedWithoutACheckpointFamily() throws {
    let preset = ImagePreset(
      id: "krea2-base", name: "Krea 2 base", mediaKind: "image",
      model: "krea2-raw", loras: [], shift: 1.15)
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "krea2-base",
      lookup: .resolved(ResolvedPreset(preset: preset), declared: preset),
      requestLoras: nil))
    XCTAssertNil(e.shift)
  }

  /// Either declaration route works: `checkpoint_family` (the fixture above)
  /// or a `model` spec the engine classifies as z-image.
  func testZImageModelSpecAloneIsEnoughToExpandTheShift() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "zeta-chroma",
      lookup: zetaChroma(checkpointFamily: nil),   // model "z-image-zeta-chroma"
      requestLoras: nil))
    XCTAssertEqual(e.shift, 3.0)
  }

  /// Fails CLOSED: a preset that declares neither `checkpoint_family` nor a
  /// model spec the engine classifies contributes no shift, because the engine
  /// cannot tell which of the two meanings the number carries.
  func testUnclassifiableFamilyPresetShiftIsNotExpanded() throws {
    let preset = ImagePreset(
      id: "mystery", name: "Mystery", mediaKind: "image",
      model: "some-community-checkpoint", loras: [], shift: 3.0)
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "mystery",
      lookup: .resolved(ResolvedPreset(preset: preset), declared: preset),
      requestLoras: nil))
    XCTAssertNil(e.shift, "an unclassifiable preset must not have its shift adopted")
  }

  func testUndeclaredShiftContributesNothing() throws {
    let e = try expansion(PresetLoRAStack.decide(
      presetId: "zeta-chroma", lookup: zetaChroma(shift: nil), requestLoras: nil))
    XCTAssertNil(e.shift)
  }

  /// `preset` is engine-set-from-the-wire only (the memberwise init pins it
  /// nil), so the seam tests below decode the body a client actually posts.
  private func decode(_ json: String) throws -> GeneratePayload {
    try JSONDecoder().decode(GeneratePayload.self, from: json.data(using: .utf8)!)
  }

  /// The `/v1/generate` seam: the expanded shift lands on the payload, and a
  /// request that carried its own keeps it.
  func testExpandingPresetPutsTheShiftOnThePayload() throws {
    let payload = try decode(#"{"prompt":"a portrait","preset":"zeta-chroma"}"#)
    let out = try GeneratePayload.expandingPreset(payload) { _ in self.zetaChroma() }
    XCTAssertEqual(out.shift, 3.0)
  }

  func testExpandingPresetLeavesAnExplicitRequestShiftAlone() throws {
    let payload = try decode(#"{"prompt":"a portrait","preset":"zeta-chroma","shift":1.73}"#)
    let out = try GeneratePayload.expandingPreset(payload) { _ in self.zetaChroma() }
    XCTAssertEqual(out.shift, 1.73)
  }

  func testNoPresetLeavesShiftUntouched() throws {
    let payload = GeneratePayload(prompt: "a portrait", shift: 3.0)
    let out = try GeneratePayload.expandingPreset(payload) { _ in
      XCTFail("no preset named — the store must not be consulted")
      return .notFound
    }
    XCTAssertEqual(out.shift, 3.0)
  }

  /// A preset the engine cannot expand stays a LABEL — its shift is not
  /// applied any more than its LoRA stack is.
  func testUnexpandablePresetContributesNoShift() throws {
    let payload = try decode(#"{"prompt":"a portrait","preset":"nope"}"#)
    let out = try GeneratePayload.expandingPreset(payload) { _ in .notFound }
    XCTAssertNil(out.shift)
    XCTAssertEqual(out.presetUnresolved, "nope")
  }

  /// A preset-owned shift must survive a crash-recovery replay: the rewritten
  /// body carries it, so the replayed job renders on the same schedule.
  func testRewrittenReplayBodyCarriesThePresetShift() throws {
    let original = #"{"prompt":"a portrait","preset":"zeta-chroma"}"#.data(using: .utf8)!
    var payload = try JSONDecoder().decode(GeneratePayload.self, from: original)
    payload = try GeneratePayload.expandingPreset(payload) { _ in self.zetaChroma() }
    let rewritten = WarmServer.rawBody(original, expandedWith: payload)
    let object = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: rewritten) as? [String: Any])
    XCTAssertEqual((object["shift"] as? NSNumber)?.floatValue, 3.0)
  }

  /// And a request that named its own shift is not rewritten — the body
  /// already says what it wants.
  func testRewrittenReplayBodyKeepsAnExplicitShift() throws {
    let original = #"{"prompt":"a portrait","preset":"zeta-chroma","shift":1.73}"#
      .data(using: .utf8)!
    var payload = try JSONDecoder().decode(GeneratePayload.self, from: original)
    payload = try GeneratePayload.expandingPreset(payload) { _ in self.zetaChroma() }
    let rewritten = WarmServer.rawBody(original, expandedWith: payload)
    let object = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: rewritten) as? [String: Any])
    XCTAssertEqual((object["shift"] as? NSNumber)?.floatValue, 1.73)
  }

  // MARK: - `applied_shift` on the response

  func testAppliedShiftIsAbsentByDefaultAndSnakeCasedWhenSet() throws {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase

    let quiet = GenerateResponse(success: true, outputPath: "/tmp/a.png", durationMs: 10)
    let quietJSON = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: encoder.encode(quiet)) as? [String: Any])
    XCTAssertNil(quietJSON["applied_shift"],
                 "a render on the model's own schedule must not grow a key")

    let shifted = GenerateResponse(
      success: true, outputPath: "/tmp/a.png", durationMs: 10, appliedShift: 3.0)
    let shiftedJSON = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: encoder.encode(shifted)) as? [String: Any])
    XCTAssertEqual((shiftedJSON["applied_shift"] as? NSNumber)?.floatValue, 3.0)
  }

  /// The async poll response carries the same field, and older persisted JSON
  /// that predates it still decodes.
  func testImageJobStatusCarriesAppliedShiftAndDecodesWithoutIt() throws {
    let status = ImageJobStatus(
      jobId: "j-1", status: .succeeded, source: "api", outputPath: "/tmp/a.png",
      durationMs: 10, error: nil, elapsedMs: 12, preemptRefused: nil, etaSec: nil,
      appliedShift: 3.0)
    XCTAssertEqual(status.appliedShift, 3.0)

    let legacy = #"""
    {"jobId":"j-1","status":"succeeded","source":"api","outputPath":"/tmp/a.png",
     "durationMs":10,"elapsedMs":12}
    """#.data(using: .utf8)!
    let decoded = try JSONDecoder().decode(ImageJobStatus.self, from: legacy)
    XCTAssertNil(decoded.appliedShift)
  }
}
