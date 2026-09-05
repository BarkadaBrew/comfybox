import XCTest

@testable import ZImage

/// WP-E20 (FDD §3.15, AC-44b, AC-44c) — the engine's own HTTP surface for
/// preset save/list/resolve, so the desktop app, the Krita bridge and MCP
/// callers (all of which read this preset store) see consistent behavior.
/// The route handlers are thin wrappers over the static functions exercised
/// here (`WarmServer.upsertPreset`, `presetsList`, `resolvePreset`), which
/// take the store explicitly.
///
/// Todd 2026-09-04: O4a (a krea2-family image preset must declare `kroma`)
/// is RETIRED — kroma is a regular LoRA, not an independent declaration —
/// so several tests below assert the OPPOSITE of what they used to: saving
/// and resolving now succeed where they used to 400/flag-invalid.
final class WarmServerPresetValidationTests: XCTestCase {

  private func makeStore() throws -> PresetStore {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-preset-route-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return PresetStore(path: dir.appendingPathComponent("presets.json"), seedDefaults: false)
  }

  private func http(_ routed: RoutedResponse) throws -> HTTPResponse {
    switch routed {
    case .json(let r), .error(let r), .shutdown(let r): return r
    case .websocketUpgrade: throw XCTSkip("unexpected websocket upgrade")
    }
  }

  private func body(_ r: HTTPResponse) -> String { String(decoding: r.body, as: UTF8.self) }

  private func jsonArray(_ r: HTTPResponse) throws -> [[String: Any]] {
    try XCTUnwrap(try JSONSerialization.jsonObject(with: r.body) as? [[String: Any]])
  }

  /// A krea2-family image preset with no structured `kroma` at all now
  /// saves (200) — O4a is retired. Its `loras[]` (a generic, undeclared-role
  /// "kroma-lora" row) is left completely alone: nothing migrates or
  /// reinterprets a plain LoRA just because its filename contains "kroma".
  func testPutKrea2PresetWithoutKromaFieldSaves() throws {
    let store = try makeStore()
    let payload = Data(#"""
    {"id":"krea-kira","name":"Kira","engine":"zimage","model":"krea2","steps":12,"guidance":1,
     "loras":[{"filename":"kroma-lora-v0.3.safetensors","scale":0.6}]}
    """#.utf8)
    let response = try http(WarmServer.upsertPreset(store: store, body: payload).0)
    XCTAssertEqual(response.status, 200, body(response))
    let stored = try XCTUnwrap(store.get("krea-kira"))
    XCTAssertNil(stored.kroma)
    XCTAssertEqual(stored.loras, [LoraReference(filename: "kroma-lora-v0.3.safetensors", scale: 0.6)])
  }

  /// A structured `kroma` with an explicit FILE migrates into `loras[]` as a
  /// regular, role-tagged entry and is echoed back as a derived view
  /// (`kromaDeprecated: true`). A `zimage-*` preset needs no kroma at all
  /// (unaffected by any of this, D14).
  func testDeclaredKromaMigratesAndZImagePresetsAreUnaffected() throws {
    let store = try makeStore()
    let krea = Data(#"""
    {"id":"krea-kira","name":"Kira","engine":"zimage","model":"krea2","checkpoint_family":"turbo",
     "kroma":{"strength":0.6,"file":"kroma-v0.3-base.safetensors"},"steps":12,"guidance":1}
    """#.utf8)
    let kreaResponse = try http(WarmServer.upsertPreset(store: store, body: krea).0)
    XCTAssertEqual(kreaResponse.status, 200, body(kreaResponse))
    let stored = try XCTUnwrap(store.get("krea-kira"))
    XCTAssertEqual(stored.checkpointFamily, "turbo")
    XCTAssertEqual(stored.kroma, KromaPolicy(strength: 0.6, file: "kroma-v0.3-base.safetensors"))
    XCTAssertEqual(stored.kromaDeprecated, true)
    XCTAssertEqual(stored.loras, [
      LoraReference(filename: "kroma-v0.3-base.safetensors", scale: 0.6, role: "kroma"),
    ])

    let zimage = Data(#"""
    {"id":"imported-cs-neutral","name":"Neutral","media_kind":"image","engine":"zimage",
     "model":"Tongyi-MAI/Z-Image-Turbo-BF16","steps":9,"guidance":3.5}
    """#.utf8)
    let zimageResponse = try http(WarmServer.upsertPreset(store: store, body: zimage).0)
    XCTAssertEqual(zimageResponse.status, 200, body(zimageResponse))
    XCTAssertNil(store.get("imported-cs-neutral")?.kroma)
  }

  /// A structured `kroma` with no file (the old "engine-default file" case)
  /// has nothing concrete to become a LoRA of — it migrates to nothing, the
  /// derived view is nil, and the loss is recorded in `migrationNotes`
  /// rather than silently swallowed (review r2, I3).
  func testDeclaredKromaWithNoFileMigratesToNothing() throws {
    let store = try makeStore()
    let payload = Data(#"""
    {"id":"krea-kira","name":"Kira","engine":"zimage","model":"krea2","kroma":{"strength":0.6}}
    """#.utf8)
    let response = try http(WarmServer.upsertPreset(store: store, body: payload).0)
    XCTAssertEqual(response.status, 200, body(response))
    let stored = try XCTUnwrap(store.get("krea-kira"))
    XCTAssertNil(stored.kroma)
    XCTAssertNil(stored.kromaDeprecated)
    XCTAssertEqual(stored.loras, [])
    XCTAssertEqual(stored.migrationNotes, ["kroma_dropped_no_file"])
  }

  /// The recipe-name resolver the generate path uses guards the preset too:
  /// a preset naming `uni_pc` is a 400 naming it (AC-15's rule, at the store).
  func testPutPresetWithUnknownSamplerIs400() throws {
    let store = try makeStore()
    let payload = Data(#"""
    {"id":"p","name":"P","engine":"zimage","model":"krea2-raw","kroma":{"strength":0},"sampler":"uni_pc"}
    """#.utf8)
    let response = try http(WarmServer.upsertPreset(store: store, body: payload).0)
    XCTAssertEqual(response.status, 400)
    XCTAssertTrue(body(response).contains("uni_pc"), body(response))
    XCTAssertNil(store.get("p"))
  }

  func testResolveRouteCarriesRES4LYFKnobsInSnakeCase() throws {
    let store = try makeStore()
    let payload = Data(#"""
    {"id":"res4lyf","name":"RES4LYF","noise_type":"pyramid","noise_alpha":-0.8,
     "implicit_steps":3,"c2":0.4}
    """#.utf8)
    let saved = try http(WarmServer.upsertPreset(store: store, body: payload).0)
    XCTAssertEqual(saved.status, 200, body(saved))

    let response = try http(
      WarmServer.resolvePreset(store: store, body: Data(#"{"id":"res4lyf"}"#.utf8)))
    XCTAssertEqual(response.status, 200, body(response))
    let json = try XCTUnwrap(
      JSONSerialization.jsonObject(with: response.body) as? [String: Any])
    XCTAssertEqual(json["noise_type"] as? String, "pyramid")
    XCTAssertEqual(json["noise_alpha"] as? Double, -0.8)
    XCTAssertEqual(json["implicit_steps"] as? Int, 3)
    XCTAssertEqual(json["c2"] as? Double, 0.4)
  }

  /// Todd 2026-09-04: `GET /v1/presets` no longer flags ANY of these entries
  /// invalid — O4a (the reason this fixture used to trip on "krea-film-apple")
  /// is retired. `invalid: false` across the board, and every entry resolves.
  /// (Was: "AC-44c — an invalid preset already on disk is served ... with
  /// invalid: true and the reason".)
  func testListFlagsNothingInvalidForKromaOnDisk() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-preset-route-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let path = dir.appendingPathComponent("presets.json")
    try Data("""
    {"presets":[
      {"id":"krea-film-apple","name":"Apple","engine":"zimage","model":"krea2","steps":8},
      {"id":"krea2-base","name":"Base","engine":"zimage","model":"kroma-v0.2-turbo","kroma":{"strength":0}},
      {"id":"imported-cs-control","name":"Control","mediaKind":"image","engine":"zimage","model":"z-image-turbo-bf16"}
    ]}
    """.utf8).write(to: path)
    let store = PresetStore(path: path, seedDefaults: false)

    let response = try http(WarmServer.presetsList(store: store))
    XCTAssertEqual(response.status, 200)
    let entries = try jsonArray(response)
    XCTAssertEqual(entries.map { $0["id"] as? String }, ["krea-film-apple", "krea2-base", "imported-cs-control"])
    XCTAssertEqual(entries.map { $0["invalid"] as? Bool }, [false, false, false])
    XCTAssertNil(entries[0]["invalid_reason"])
    // `kroma: {strength: 0}` migrates to no loras entry and no derived view —
    // the key is absent, not merely zeroed.
    XCTAssertNil(entries[1]["kroma"])
    // The preset's own fields are still there beside the flag (flat, not nested).
    XCTAssertEqual(entries[0]["model"] as? String, "krea2")

    // Every entry resolves now — none is refused.
    for id in ["krea-film-apple", "krea2-base", "imported-cs-control"] {
      let resolve = try http(WarmServer.resolvePreset(store: store, body: Data(#"{"id":"\#(id)"}"#.utf8)))
      XCTAssertEqual(resolve.status, 200, "\(id): \(body(resolve))")
    }
  }
}
