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
  private func zetaChroma(shift: Double? = 3.0) -> PresetLoRAStack.Lookup {
    let preset = ImagePreset(
      id: "zeta-chroma", name: "Zeta Chroma", mediaKind: "image",
      model: "zeta-chroma", steps: 28, guidance: 5.0,
      loras: [],
      checkpointFamily: "z-image",
      sampler: "euler", sigmaSchedule: "simple", shift: shift)
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
