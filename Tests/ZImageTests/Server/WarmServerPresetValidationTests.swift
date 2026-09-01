import XCTest

@testable import ZImage

/// WP-E20 (FDD §3.15, AC-44b, AC-44c) — O4a on the engine's own HTTP surface,
/// so the desktop app, the Krita bridge and MCP callers (all of which read
/// this preset store) cannot save or select a krea2 image preset that does
/// not declare its kroma strength. The route handlers are thin wrappers over
/// the static functions exercised here (`WarmServer.upsertPreset`,
/// `presetsList`, `resolvePreset`), which take the store explicitly.
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

  /// AC-44b: `PUT /v1/presets` with a krea2-family image preset lacking
  /// `kroma` is a 400 naming the preset and the field; nothing is stored.
  func testPutKrea2PresetWithoutKromaIs400NamingPresetAndField() throws {
    let store = try makeStore()
    let payload = Data(#"""
    {"id":"krea-kira","name":"Kira","engine":"zimage","model":"krea2","steps":12,"guidance":1,
     "loras":[{"filename":"kroma-lora-v0.3.safetensors","scale":0.6}]}
    """#.utf8)
    let response = try http(WarmServer.upsertPreset(store: store, body: payload).0)
    XCTAssertEqual(response.status, 400)
    let text = body(response)
    XCTAssertTrue(text.contains("krea-kira"), text)
    XCTAssertTrue(text.contains("kroma"), text)
    XCTAssertNil(store.get("krea-kira"))
    XCTAssertTrue(store.list().isEmpty)
  }

  /// AC-44b: the same preset WITH a declared kroma saves (200), and a
  /// `zimage-*` preset without one is accepted (D14).
  func testPutDeclaredKromaAndZImagePresetsAreAccepted() throws {
    let store = try makeStore()
    let krea = Data(#"""
    {"id":"krea-kira","name":"Kira","engine":"zimage","model":"krea2","checkpoint_family":"turbo",
     "kroma":{"strength":0.6},"steps":12,"guidance":1}
    """#.utf8)
    let kreaResponse = try http(WarmServer.upsertPreset(store: store, body: krea).0)
    XCTAssertEqual(kreaResponse.status, 200, body(kreaResponse))
    XCTAssertEqual(store.get("krea-kira")?.kroma, KromaPolicy(strength: 0.6))
    XCTAssertEqual(store.get("krea-kira")?.checkpointFamily, "turbo")

    let zimage = Data(#"""
    {"id":"imported-cs-neutral","name":"Neutral","media_kind":"image","engine":"zimage",
     "model":"Tongyi-MAI/Z-Image-Turbo-BF16","steps":9,"guidance":3.5}
    """#.utf8)
    let zimageResponse = try http(WarmServer.upsertPreset(store: store, body: zimage).0)
    XCTAssertEqual(zimageResponse.status, 200, body(zimageResponse))
    XCTAssertNil(store.get("imported-cs-neutral")?.kroma)
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

  /// AC-44c: an invalid preset already on disk is served by `GET /v1/presets`
  /// with `invalid: true` and the reason, beside valid entries carrying
  /// `invalid: false` — so a client can show it and refuse to select it.
  func testListFlagsInvalidPresetOnDisk() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("comfybox-preset-route-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let path = dir.appendingPathComponent("presets.json")
    try Data("""
    {"presets":[
      {"id":"krea-film-apple","name":"Apple","engine":"zimage","model":"krea2","steps":8,
       "loras":[{"filename":"kroma-v0.1.safetensors","scale":1.0}]},
      {"id":"krea2-base","name":"Base","engine":"zimage","model":"kroma-v0.2-turbo","kroma":{"strength":0}},
      {"id":"imported-cs-control","name":"Control","mediaKind":"image","engine":"zimage","model":"z-image-turbo-bf16"}
    ]}
    """.utf8).write(to: path)
    let store = PresetStore(path: path, seedDefaults: false)

    let response = try http(WarmServer.presetsList(store: store))
    XCTAssertEqual(response.status, 200)
    let entries = try jsonArray(response)
    XCTAssertEqual(entries.map { $0["id"] as? String }, ["krea-film-apple", "krea2-base", "imported-cs-control"])
    XCTAssertEqual(entries[0]["invalid"] as? Bool, true)
    let reason = try XCTUnwrap(entries[0]["invalid_reason"] as? String)
    XCTAssertTrue(reason.contains("kroma"), reason)
    XCTAssertEqual(entries[1]["invalid"] as? Bool, false)
    XCTAssertNil(entries[1]["invalid_reason"])
    XCTAssertEqual(entries[2]["invalid"] as? Bool, false)
    // The preset's own fields are still there beside the flag (flat, not nested).
    XCTAssertEqual(entries[0]["model"] as? String, "krea2")
    XCTAssertEqual((entries[1]["kroma"] as? [String: Any])?["strength"] as? Double, 0)

    // And `POST /v1/presets/resolve` on the flagged one is a 400 naming it.
    let resolve = try http(WarmServer.resolvePreset(store: store, body: Data(#"{"id":"krea-film-apple"}"#.utf8)))
    XCTAssertEqual(resolve.status, 400)
    XCTAssertTrue(body(resolve).contains("krea-film-apple"), body(resolve))
    let resolveOK = try http(WarmServer.resolvePreset(store: store, body: Data(#"{"id":"krea2-base"}"#.utf8)))
    XCTAssertEqual(resolveOK.status, 200, body(resolveOK))
  }
}
