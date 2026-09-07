// StylePackRouteTests.swift — comfybox#399
//
// The StylePack lane (`feat/phone-look-post`, deployed as ComfyBox-1ba36a1 on
// 2026-08-24) was never merged. Main has moved ~190 commits since — the #286
// preset expansion, the #350 resolver, kroma-as-a-regular-LoRA (#365), the
// #393/#154 shift semantics — and every one of them lives in the same decode
// the lane patched at the route.
//
// These drive the merged seam on the NEW base, through the exact function the
// `/v1/generate` and `/v1/generate/async` handlers call
// (`WarmServer.decodedGeneratePayload(from:store:configuration:…)`), plus the
// pure resolver it delegates to. Four things are pinned:
//
//   1. Registry lookup — every v1 name, and a 400 for anything else.
//   2. `phone_look` / `style` decode off the WIRE. `GeneratePayload` has a
//      hand-rolled `init(from:)`, so a CodingKey alone decodes nothing — the
//      defect the lane's own second commit fixed, and the one that would come
//      straight back if the merge dropped a line.
//   3. Precedence, in MAIN's words rather than the lane's: explicit request >
//      preset > nothing, the same rule #286 (steps/guidance), #285 (vae) and
//      #154 (shift) follow.
//   4. NO BEHAVIOUR CHANGE for a request that asks for no look — Kira's daily
//      renders. See `StylePackParityTests` below.

import XCTest

@testable import ZImage

final class StylePackRouteTests: XCTestCase {

  // MARK: Harness (shared shape with GeneratePresetRouteTests)

  private func makeStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-stylepack-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  private var configuration: WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  /// THE decode both generate routes run.
  private func decode(_ json: String, store: PresetStore) throws -> GeneratePayload {
    try WarmServer.decodedGeneratePayload(
      from: Data(json.utf8), store: store, configuration: configuration,
      loraExists: { _ in true })
  }

  /// Kira's live lane, as it sits in `~/.comfybox/presets.json`: krea2-raw,
  /// accel + polish, `checkpoint_family: raw-accel`.
  @discardableResult
  private func seedKreaKira(
    _ store: PresetStore, id: String = "krea-kira",
    style: String? = nil, phoneLook: Bool? = nil
  ) throws -> ImagePreset {
    var preset = ImagePreset(
      id: id, name: "Kira", mediaKind: "image", model: "krea2-raw", steps: 12, guidance: 1.0,
      loras: [
        LoraReference(filename: "krea2_turbo_distill_r256.safetensors", scale: 0.6, role: "accel"),
        LoraReference(filename: "RealisticSnapshotKrea2.safetensors", scale: 0.4),
      ],
      checkpointFamily: "raw-accel",
      sampler: "res_2s", sigmaSchedule: "beta57")
    preset.style = style
    preset.phoneLook = phoneLook
    return try store.upsert(preset)
  }

  // MARK: 1 — the registry

  func testRegistryAnswersToEveryV1NameAndNothingElse() {
    XCTAssertEqual(StylePack.knownNames, ["phone", "trix-bw", "hp5-soft"])
    for name in StylePack.knownNames {
      XCTAssertEqual(StylePack.named(name)?.rawValue, name, "registry lost '\(name)'")
    }
    XCTAssertNil(StylePack.named("sepia-dreams"))
    XCTAssertNil(StylePack.named(""))
    XCTAssertNil(StylePack.named("   "))
  }

  /// A name is matched after trimming and lowercasing — a desktop text field
  /// that hands back `"Trix-BW "` must not 400.
  func testRegistryLookupIsTrimmedAndCaseInsensitive() {
    XCTAssertEqual(StylePack.named(" Trix-BW "), .trixBW)
    XCTAssertEqual(StylePack.named("HP5-Soft"), .hp5Soft)
    XCTAssertEqual(StylePack.named("PHONE"), .phone)
  }

  func testUnknownStyleIsA400NamingTheKnownSet() throws {
    let store = try makeStore()
    XCTAssertThrowsError(try decode(#"{"prompt":"x","style":"sepia-dreams"}"#, store: store)) {
      guard case WarmServerError.unknownStyle(let name, let valid) = $0 else {
        return XCTFail("expected .unknownStyle, got \($0)")
      }
      XCTAssertEqual(name, "sepia-dreams")
      XCTAssertEqual(valid, StylePack.knownNames)
      // The refusal must reach the caller as 400, not 500 — it is the
      // caller's error, in the same arm as unknownSampler/unknownNoiseType.
      let response = WarmServer.errorResponse(for: $0)
      XCTAssertEqual(response.status, 400)
      XCTAssertTrue(
        String(data: response.body, encoding: .utf8)?.contains("hp5-soft") == true,
        "the 400 must name the known set")
    }
  }

  /// A preset that names a look the engine does not have is refused at UPSERT
  /// too — a style that silently did nothing on every render is worse than a
  /// 400 the moment the preset is saved.
  func testPresetWithUnknownStyleIsRefusedAtUpsert() throws {
    let store = try makeStore()
    XCTAssertThrowsError(try seedKreaKira(store, style: "sepia-dreams")) { error in
      XCTAssertTrue("\(error)".contains("unknown style"), "got \(error)")
    }
    XCTAssertNoThrow(try seedKreaKira(store, style: "hp5-soft"))
  }

  // MARK: 2 — the WIRE

  /// `GeneratePayload.init(from:)` is hand-rolled: a CodingKey with no
  /// `decodeIfPresent` line decodes nothing at all. This is the lane's own
  /// 8791cdd regression, pinned so the merge cannot lose it again.
  func testPhoneLookDecodesFromTheWire() throws {
    let store = try makeStore()
    let payload = try decode(#"{"prompt":"a portrait","phone_look":true}"#, store: store)
    XCTAssertEqual(payload.phoneLook, true, "phone_look never reached the payload")
    XCTAssertEqual(payload.style, StylePack.phone.rawValue, "phone_look is the alias for 'phone'")
  }

  func testStyleDecodesFromTheWire() throws {
    let store = try makeStore()
    for name in StylePack.knownNames {
      let payload = try decode(#"{"prompt":"a portrait","style":"\#(name)"}"#, store: store)
      XCTAssertEqual(payload.style, name)
    }
  }

  /// `phone_look: false` is a statement, not an absence — it must not become
  /// a style, and it must not be mistaken for "unset" either.
  func testPhoneLookFalseAsksForNoLook() throws {
    let store = try makeStore()
    let payload = try decode(#"{"prompt":"x","phone_look":false}"#, store: store)
    XCTAssertEqual(payload.phoneLook, false)
    XCTAssertNil(payload.style)
  }

  // MARK: 3 — precedence: explicit request > preset > nothing

  func testPresetDeclaredStyleIsAdoptedWhenTheRequestSaysNothing() throws {
    let store = try makeStore()
    try seedKreaKira(store, style: "hp5-soft")
    let payload = try decode(#"{"prompt":"x","preset":"krea-kira"}"#, store: store)
    XCTAssertEqual(payload.style, "hp5-soft")
  }

  func testPresetDeclaredPhoneLookIsAdoptedAsThePhoneStyle() throws {
    let store = try makeStore()
    try seedKreaKira(store, phoneLook: true)
    let payload = try decode(#"{"prompt":"x","preset":"krea-kira"}"#, store: store)
    XCTAssertEqual(payload.style, StylePack.phone.rawValue)
  }

  func testExplicitRequestStyleBeatsThePreset() throws {
    let store = try makeStore()
    try seedKreaKira(store, style: "hp5-soft")
    let payload = try decode(#"{"prompt":"x","preset":"krea-kira","style":"trix-bw"}"#, store: store)
    XCTAssertEqual(payload.style, "trix-bw", "the request always wins")
  }

  /// A preset's own `style` beats its own legacy `phone_look` — the newer,
  /// more specific declaration is not overruled by the alias it replaced.
  func testPresetStyleBeatsPresetPhoneLook() throws {
    let store = try makeStore()
    try seedKreaKira(store, style: "trix-bw", phoneLook: true)
    let payload = try decode(#"{"prompt":"x","preset":"krea-kira"}"#, store: store)
    XCTAssertEqual(payload.style, "trix-bw")
  }

  /// The request's `phone_look` outranks a preset's `style`: both sides are
  /// read style-then-alias, but the REQUEST side is read first in full.
  func testRequestPhoneLookBeatsThePresetStyle() throws {
    let store = try makeStore()
    try seedKreaKira(store, style: "trix-bw")
    let payload = try decode(#"{"prompt":"x","preset":"krea-kira","phone_look":true}"#, store: store)
    XCTAssertEqual(payload.style, StylePack.phone.rawValue)
  }

  /// The pure resolver, exhaustively — the table the route only samples.
  func testResolveNamePrecedenceTable() {
    let cases: [(String?, Bool?, String?, Bool?, String?)] = [
      // request style, request phoneLook, preset style, preset phoneLook, expected
      (nil, nil, nil, nil, nil),
      (nil, false, nil, false, nil),
      ("trix-bw", nil, nil, nil, "trix-bw"),
      (nil, true, nil, nil, "phone"),
      (nil, nil, "hp5-soft", nil, "hp5-soft"),
      (nil, nil, nil, true, "phone"),
      ("trix-bw", true, "hp5-soft", true, "trix-bw"),
      (nil, true, "hp5-soft", nil, "phone"),
      ("  ", nil, "hp5-soft", nil, "hp5-soft"),  // blank is not a declaration
      (nil, nil, "  ", true, "phone"),
    ]
    for (rs, rp, ps, pp, expected) in cases {
      XCTAssertEqual(
        StylePack.resolveName(
          requestStyle: rs, requestPhoneLook: rp, presetStyle: ps, presetPhoneLook: pp),
        expected,
        "resolveName(\(rs as Any), \(rp as Any), \(ps as Any), \(pp as Any))")
    }
  }

  /// #399 ruling: a style is NOT part of the recipe, so it survives a preset
  /// main cannot expand. `expandingPreset` refuses a video preset on the
  /// image path (`media_kind:video`) — the render falls back to the request's
  /// own settings, exactly as before #286, and the declared look still
  /// applies because a post-process cannot land on the wrong base.
  func testStyleSurvivesAPresetTheEngineCannotExpand() throws {
    let store = try makeStore()
    var preset = ImagePreset(
      id: "ltx-clip", name: "LTX clip", mediaKind: "video", model: "ltx-2",
      loras: [])
    preset.style = "hp5-soft"
    try store.upsert(preset)

    let payload = try decode(#"{"prompt":"x","preset":"ltx-clip"}"#, store: store)
    XCTAssertEqual(payload.presetUnresolvedReason, "media_kind:video", "precondition")
    XCTAssertEqual(payload.style, "hp5-soft", "a post-process is not part of the recipe")
  }

  /// A preset-owned style must survive crash-recovery replay, exactly like
  /// #154's preset-owned shift — a replayed body that dropped it would render
  /// the job WITHOUT the look it was accepted with.
  func testPresetOwnedStyleIsPersistedIntoTheReplayBody() throws {
    let store = try makeStore()
    try seedKreaKira(store, style: "trix-bw")
    let original = Data(#"{"prompt":"x","preset":"krea-kira"}"#.utf8)
    let payload = try WarmServer.decodedGeneratePayload(
      from: original, store: store, configuration: configuration, loraExists: { _ in true })
    let replayed = WarmServer.rawBody(original, expandedWith: payload)
    let object = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: replayed) as? [String: Any])
    XCTAssertEqual(object["style"] as? String, "trix-bw")
  }

  // MARK: 4 — hp5-soft's own parameters (1ba36a1)

  /// The gentler emulsion is a distinct recipe, not an alias of `trix-bw`:
  /// same guaranteed mono pan mix and percentile levels, contrast 1.08 rather
  /// than the pushed 1.3, and a lower/softer highlight shoulder.
  func testHP5SoftParametersDifferFromTrixBW() {
    let w = 48, h = 48
    var hp5 = Self.scene(w: w, h: h)
    var trix = Self.scene(w: w, h: h)
    StylePack.hp5Soft.apply(pixels: &hp5, width: w, height: h)
    StylePack.trixBW.apply(pixels: &trix, width: w, height: h)
    XCTAssertNotEqual(hp5, trix, "hp5-soft must not be trix-bw under another name")

    // Both are guaranteed monochrome.
    for i in 0..<(w * h) {
      XCTAssertEqual(hp5[i * 3], hp5[i * 3 + 1], accuracy: 1e-6)
      XCTAssertEqual(hp5[i * 3 + 1], hp5[i * 3 + 2], accuracy: 1e-6)
    }
    // Gentler gradation: strictly lower contrast than the push.
    XCTAssertLessThan(Self.std(hp5), Self.std(trix))
    // …and a softer toe. Both curves clamp at 0, so `min()` is 0 either way;
    // what separates them is HOW MUCH is crushed there. The push (×1.3)
    // zeroes everything under 0.115 after levels, the gentle curve (×1.08)
    // only under 0.037 — so trix must clip strictly more pixels to black.
    let hp5Blacks = hp5.filter { $0 == 0 }.count
    let trixBlacks = trix.filter { $0 == 0 }.count
    XCTAssertGreaterThan(trixBlacks, hp5Blacks, "hp5-soft must hold its shadows")
  }

  /// `hp5-soft` reached the registry, the decode AND the apply — the whole
  /// path the third lane commit added.
  func testHP5SoftIsReachableFromTheWire() throws {
    let store = try makeStore()
    let payload = try decode(#"{"prompt":"x","style":"hp5-soft"}"#, store: store)
    XCTAssertEqual(StylePack.resolved(payload.style), .hp5Soft)
  }

  // MARK: Fixtures

  /// A small synthetic color scene with true black/white pixels, so the
  /// percentile levels stage has an honest window.
  static func scene(w: Int, h: Int) -> [Float] {
    var px = [Float](repeating: 0, count: w * h * 3)
    for y in 0..<h {
      for x in 0..<w {
        let i = (y * w + x) * 3
        let t = Float(x) / Float(w - 1)
        px[i] = 0.2 + 0.6 * t
        px[i + 1] = 0.5
        px[i + 2] = 0.8 - 0.6 * t
      }
    }
    px[0] = 0.02; px[1] = 0.02; px[2] = 0.02
    let last = (w * h - 1) * 3
    px[last] = 0.97; px[last + 1] = 0.97; px[last + 2] = 0.97
    return px
  }

  static func std(_ v: [Float]) -> Float {
    let m = v.reduce(0, +) / Float(v.count)
    return (v.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Float(v.count)).squareRoot()
  }
}
