// StylePackParityTests.swift — comfybox#399, the no-behaviour-change pin
//
// The StylePack lane is being merged onto a main that has been serving Kira's
// daily renders for ~190 commits. Kira's presets declare no style and her
// daemon sends no `style` / `phone_look`, so the ONLY acceptable outcome for
// those renders is: nothing changes. Not "changes imperceptibly" — nothing.
//
// These are the tests that say so, at the two places a post-process could
// possibly leak into a render that did not ask for one:
//
//   * THE SCHEDULE. A style must never move a sigma. Pinned byte-for-byte on
//     the real `SigmaSchedule.krea2` grid, built from the fields the decode
//     actually produced for Kira's live preset shape.
//   * THE PIXELS. `StylePack.resolved(_:)` is the whole gate `runKrea2Generate`
//     consults before it touches the decoded image. nil ⇒ the buffer is never
//     read out of MLX, never rebuilt, never written. Pinned as a property of
//     that pure function, plus the buffer identity itself.
//
// …and at the wire, where an additive field could still have changed the
// bytes a crash-recovery replay stores.

import XCTest

@testable import ZImage

final class StylePackParityTests: XCTestCase {

  // MARK: Harness

  private func makeStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-stylepack-parity-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  private var configuration: WarmServerConfiguration {
    WarmServerConfiguration(allowedOutputDirectory: NSTemporaryDirectory())
  }

  private func decode(_ json: String, store: PresetStore) throws -> GeneratePayload {
    try WarmServer.decodedGeneratePayload(
      from: Data(json.utf8), store: store, configuration: configuration,
      loraExists: { _ in true })
  }

  /// `krea-kira-avocado` as it sits in `~/.comfybox/presets.json` today:
  /// krea2-raw, res_3s / bong_tangent, 20 steps, guidance 3.5, shift 1.15.
  @discardableResult
  private func seedAvocado(_ store: PresetStore, style: String? = nil) throws -> ImagePreset {
    var preset = ImagePreset(
      id: "krea-kira-avocado", name: "Kira avocado", mediaKind: "image",
      model: "krea2-raw", steps: 20, guidance: 3.5,
      loras: [LoraReference(filename: "Girly_Tiana.safetensors", scale: 0.6)],
      checkpointFamily: "raw-stock",
      sampler: "res_3s", sigmaSchedule: "bong_tangent", shift: 1.15)
    preset.style = style
    return try store.upsert(preset)
  }

  /// Exact bytes of a Float grid — `XCTAssertEqual` on `[Float]` already
  /// compares by value, but the byte view makes "byte-identical" literal and
  /// catches a -0.0/NaN difference `==` would hide.
  private func bytes(_ grid: [Float]) -> Data {
    grid.withUnsafeBufferPointer { Data(buffer: $0) }
  }

  // MARK: 1 — the sigma grid

  /// The grid Kira's live preset produces is identical whether or not the
  /// StylePack plumbing exists in the decode: same steps, same shift (=mu),
  /// same `SigmaSchedule.krea2` output, byte for byte.
  ///
  /// The two payloads differ ONLY in that one preset declares a look. If the
  /// merge had wired a style into anything the schedule reads — a step count,
  /// a shift, a sampler — this fails.
  func testStyleDoesNotMoveTheSigmaGrid() throws {
    let plain = try makeStore()
    try seedAvocado(plain)
    let styled = try makeStore()
    try seedAvocado(styled, style: "hp5-soft")

    let body = #"{"prompt":"a portrait","preset":"krea-kira-avocado"}"#
    let a = try decode(body, store: plain)
    let b = try decode(body, store: styled)

    XCTAssertNil(a.style, "precondition: no look asked for")
    XCTAssertEqual(b.style, "hp5-soft", "precondition: a look asked for")

    // Everything the sigma grid is built from.
    XCTAssertEqual(a.steps, b.steps)
    XCTAssertEqual(a.shift, b.shift)
    XCTAssertEqual(a.guidance, b.guidance)
    XCTAssertEqual(a.model, b.model)
    XCTAssertEqual(a.scheduler, b.scheduler)
    XCTAssertEqual(a.sigmaSchedule, b.sigmaSchedule)
    XCTAssertEqual(a.loras?.map(\.path), b.loras?.map(\.path))
    XCTAssertEqual(a.loras?.map(\.scale), b.loras?.map(\.scale))

    // …and the grid itself, for the whole live step range.
    for steps in [4, 8, 12, 20, 52] {
      let mu = Float(a.shift ?? 1.15)
      let gridA = SigmaSchedule.krea2(numSteps: steps, mu: mu)
      let gridB = SigmaSchedule.krea2(numSteps: steps, mu: Float(b.shift ?? 1.15))
      XCTAssertEqual(bytes(gridA), bytes(gridB), "sigma grid moved at \(steps) steps")
      XCTAssertEqual(gridA.count, steps + 1)
    }
  }

  /// The same statement for a request that names NO preset at all — the
  /// swap-first shape (`/v1/lora/swap` then generate) main calls "byte-identical
  /// to pre-#286 behaviour". Adding `style`/`phone_look` to the decoder must
  /// not have disturbed it.
  func testBarePromptRequestIsUntouched() throws {
    let store = try makeStore()
    let payload = try decode(#"{"prompt":"a portrait","steps":8,"seed":1234}"#, store: store)
    XCTAssertNil(payload.style)
    XCTAssertNil(payload.phoneLook)
    XCTAssertNil(payload.preset)
    XCTAssertEqual(payload.steps, 8)
    XCTAssertEqual(payload.seed, 1234)
    XCTAssertNil(payload.loras, "no preset ⇒ no engine-produced stack")
  }

  // MARK: 2 — the pixels

  /// `StylePack.resolved(_:)` IS the gate in `runKrea2Generate`. nil in ⇒ nil
  /// out ⇒ the `image.asArray` / `MLXArray(px, …)` round trip never runs, so
  /// the array that reaches `QwenImageIO.saveImage` is the array the VAE
  /// decoder produced — the same object, not an equal copy.
  func testNoStyleMeansThePixelPathIsNeverEntered() {
    XCTAssertNil(StylePack.resolved(nil))
    XCTAssertNil(StylePack.resolved(""))
    XCTAssertNil(StylePack.resolved("   "))
  }

  /// And the buffer identity, stated directly: running the production gate
  /// with no style leaves every float exactly where it was.
  func testNoStyleLeavesTheBufferByteIdentical() {
    let w = 40, h = 32
    let original = StylePackRouteTests.scene(w: w, h: h)
    var px = original
    if let style = StylePack.resolved(nil) {
      style.apply(pixels: &px, width: w, height: h)
    }
    XCTAssertEqual(bytes(px), bytes(original), "a no-style render must not touch one bit")
  }

  /// The counterweight: the gate is the ONLY thing standing between a render
  /// and a look. Every registered style changes the buffer — so the test
  /// above is pinning the gate, not an inert feature.
  func testEveryStyleActuallyChangesTheBuffer() {
    let w = 40, h = 32
    let original = StylePackRouteTests.scene(w: w, h: h)
    for name in StylePack.knownNames {
      var px = original
      let style = try? XCTUnwrap(StylePack.resolved(name))
      style?.apply(pixels: &px, width: w, height: h)
      XCTAssertNotEqual(bytes(px), bytes(original), "style '\(name)' did nothing")
      for v in px {
        XCTAssertFalse(v.isNaN, "style '\(name)' produced NaN")
        XCTAssertGreaterThanOrEqual(v, 0)
        XCTAssertLessThanOrEqual(v, 1)
      }
    }
  }

  /// Determinism: the same buffer through the same style twice is the same
  /// bytes. A post-process on the image path must not introduce a second
  /// source of run-to-run variation beside the seed.
  func testStylesAreDeterministic() {
    let w = 24, h = 24
    for name in StylePack.knownNames {
      var a = StylePackRouteTests.scene(w: w, h: h)
      var b = StylePackRouteTests.scene(w: w, h: h)
      StylePack.resolved(name)?.apply(pixels: &a, width: w, height: h)
      StylePack.resolved(name)?.apply(pixels: &b, width: w, height: h)
      XCTAssertEqual(bytes(a), bytes(b), "style '\(name)' is not deterministic")
    }
  }

  // MARK: 3 — the wire and the replay body

  /// The crash-recovery snapshot for a no-style render must be exactly the
  /// bytes main would have stored: the expanded recipe, and no `style` or
  /// `phone_look` key manufactured out of nowhere.
  func testReplayBodyGainsNoStyleKeyForANoStyleRender() throws {
    let store = try makeStore()
    try seedAvocado(store)
    let original = Data(#"{"prompt":"a portrait","preset":"krea-kira-avocado"}"#.utf8)
    let payload = try WarmServer.decodedGeneratePayload(
      from: original, store: store, configuration: configuration, loraExists: { _ in true })
    let replayed = WarmServer.rawBody(original, expandedWith: payload)
    let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: replayed) as? [String: Any])

    XCTAssertNil(object["style"], "no look was asked for; none may be persisted")
    XCTAssertNil(object["phone_look"])
    // What main DOES expand, unchanged.
    XCTAssertEqual(object["model"] as? String, "krea2-raw")
    XCTAssertEqual(object["steps"] as? Int, 20)
    XCTAssertEqual((object["guidance"] as? NSNumber)?.doubleValue, 3.5)
    XCTAssertNotNil(object["loras"])
  }

  /// A replayed no-style job decodes back to a no-style payload — the round
  /// trip cannot resurrect a look.
  func testReplayOfANoStyleJobStaysStyleless() throws {
    let store = try makeStore()
    try seedAvocado(store)
    let original = Data(#"{"prompt":"a portrait","preset":"krea-kira-avocado"}"#.utf8)
    let payload = try WarmServer.decodedGeneratePayload(
      from: original, store: store, configuration: configuration, loraExists: { _ in true })
    let replayed = WarmServer.rawBody(original, expandedWith: payload)
    // `gateSubmission: false` is the crash-recovery replay's own call.
    let again = try WarmServer.decodedGeneratePayload(
      from: replayed, store: store, configuration: configuration,
      loraExists: { _ in true }, gateSubmission: false)
    XCTAssertNil(again.style)
    XCTAssertEqual(again.steps, payload.steps)
    XCTAssertEqual(again.shift, payload.shift)
  }

  /// A preset saved before #399 has no `style` key at all — decoding it must
  /// leave both new fields nil rather than defaulting to a look.
  func testPresetJSONWithoutStyleDecodesToNoLook() throws {
    let json = #"""
    {"id":"legacy","name":"Legacy","media_kind":"image","model":"krea2-raw",
     "steps":12,"loras":[],"checkpoint_family":"raw-accel"}
    """#
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let preset = try decoder.decode(ImagePreset.self, from: Data(json.utf8))
    XCTAssertNil(preset.style)
    XCTAssertNil(preset.phoneLook)
  }

  /// …and the additive fields round-trip through the store's own JSON, both
  /// directions. `ImagePreset` has a hand-rolled `init(from:)` and a
  /// synthesized encoder driven by the SAME `CodingKeys`, so a field missing
  /// from that enum silently vanishes in both — the `videoTuning` regression
  /// the file's own comments warn about, and exactly what the lane shipped
  /// (it declared the stored properties and nothing else, so a
  /// preset-declared look never survived a write to `presets.json`).
  func testStyleAndPhoneLookRoundTripThroughPresetJSON() throws {
    var preset = ImagePreset(
      id: "look", name: "Look", mediaKind: "image", model: "krea2-raw", loras: [],
      checkpointFamily: "raw-accel")
    preset.style = "trix-bw"
    preset.phoneLook = true

    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    let data = try encoder.encode(preset)
    let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    XCTAssertEqual(object["style"] as? String, "trix-bw", "style dropped by the encoder")
    XCTAssertEqual(object["phone_look"] as? Bool, true, "phone_look dropped by the encoder")

    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let back = try decoder.decode(ImagePreset.self, from: data)
    XCTAssertEqual(back.style, "trix-bw")
    XCTAssertEqual(back.phoneLook, true)
  }

  /// The store itself, on disk: upsert → reload → the look is still there.
  func testStyleSurvivesAStoreReload() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-stylepack-reload-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
    let path = dir.appendingPathComponent("presets.json")

    var preset = ImagePreset(
      id: "look", name: "Look", mediaKind: "image", model: "krea2-raw", loras: [],
      checkpointFamily: "raw-accel")
    preset.style = "hp5-soft"
    _ = try PresetStore(path: path, seedDefaults: false).upsert(preset)

    let reloaded = PresetStore(path: path, seedDefaults: false)
    XCTAssertEqual(reloaded.get("look")?.style, "hp5-soft")
  }
}
