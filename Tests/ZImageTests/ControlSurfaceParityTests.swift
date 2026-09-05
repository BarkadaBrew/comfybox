// ControlSurfaceParityTests.swift — the anti-drift parity test
// (FDD-ui-api-parity §3.5, D5; §4.5 Phase 4; comfybox#300).
//
// Parses the dispatch switches from SOURCE as ground truth for routes and holds
// them, the compile-time ControlRegistry, MCPToolRegistry, ParityExemptions and
// the generated docs/api-reference.md to each other. §3.5's five assertions:
//   1. extract Set<RouteRef> from both dispatch files;
//   2. pin the parser — per-file tuple counts equal checked-in constants and
//      every non-comment `case (` occurrence is consumed by a recognizer
//      (an unrecognized arm fails as "unparsed dispatch arm at <file>:<line>");
//   3. every mutating `surface: .v1` route is claimed by ≥1 MCP tool via its
//      `routes:` field or listed in ParityExemptions with a non-empty reason;
//   4. every `.comfybox`-hosted descriptor's write route and mcpTool resolve to
//      real entries (kira-hosted descriptors are a coffeeshop-server contract
//      test's job);
//   5. walk a default (first-run-migrated) ComfyBoxServerConfig to its leaf
//      pointers — each has a descriptor or is in nonControlKeys.
// Plus the §4.5 test list: no dangling read.pointer; docs generate idempotent
// and byte-matching. Production families krea2 (images) + ltx2 (video) get the
// value-resolution depth coverage.

import XCTest
@testable import ZImage

final class ControlSurfaceParityTests: XCTestCase {

  // MARK: - Fixtures

  /// §3.5 assertion 2: the per-file tuple-count pins. A dispatch-arm add,
  /// remove, or multi-tuple change moves these — update the pin IN THE SAME
  /// REVIEW as the arm, alongside the tool/exemption/docs updates the other
  /// assertions demand. (Counting tuples, not lines: `:1641`-style multi-tuple
  /// arms count once per tuple — the v1 line-count pin would have silently
  /// dropped `/v1/queue/resume`.)
  private static let expectedWarmServerTuples = 93
  private static let expectedBridgeTuples = 17

  private static let repoRoot: URL = {
    // <root>/Tests/ZImageTests/ControlSurfaceParityTests.swift
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .deletingLastPathComponent()
      .deletingLastPathComponent()
  }()

  private static let warmParse: ControlSurfaceParser.Result = {
    let url = ControlSurfaceParser.warmServerSource(repoRoot: repoRoot)
    return (try? ControlSurfaceParser.parse(fileAt: url, surface: .v1))
      ?? ControlSurfaceParser.Result(routes: [], tupleCount: 0, problems: ["failed to read \(url.path)"])
  }()

  private static let bridgeParse: ControlSurfaceParser.Result = {
    let url = ControlSurfaceParser.comfyBridgeSource(repoRoot: repoRoot)
    return (try? ControlSurfaceParser.parse(fileAt: url, surface: .comfyUICompat))
      ?? ControlSurfaceParser.Result(routes: [], tupleCount: 0, problems: ["failed to read \(url.path)"])
  }()

  /// Segment-wise route-path match: a `{...}` template segment matches any one
  /// concrete segment; a trailing `*` (parser normalization for a no-slash
  /// `hasPrefix`) matches any remainder. Exact strings match themselves.
  private func path(_ concrete: String, matches template: String) -> Bool {
    if concrete == template { return true }
    let concreteSegments = concrete.split(separator: "/", omittingEmptySubsequences: false)
    let templateSegments = template.split(separator: "/", omittingEmptySubsequences: false)
    guard concreteSegments.count == templateSegments.count else { return false }
    for (c, t) in zip(concreteSegments, templateSegments) {
      if t.hasPrefix("{") && t.hasSuffix("}") { continue }
      if c != t { return false }
    }
    return true
  }

  private func parsedV1Route(method: String, path concrete: String) -> Bool {
    Self.warmParse.routes.contains { route in
      route.method == method && (route.path == concrete || path(concrete, matches: route.path))
    }
  }

  // MARK: - §3.5 assertions 1 + 2: parse both files, pin the parser

  func testWarmServerDispatchParsesCleanly() {
    XCTAssertEqual(
      Self.warmParse.problems, [],
      "Unparsed dispatch arms — teach the recognizer or fix the arm; do not skip")
    XCTAssertEqual(
      Self.warmParse.tupleCount, Self.expectedWarmServerTuples,
      "WarmServer.swift dispatch tuple count moved — update the pin in the same review as the arm change, plus tools/exemptions/docs as the other assertions demand")
    XCTAssertFalse(Self.warmParse.routes.isEmpty)
  }

  func testBridgeDispatchParsesCleanly() {
    XCTAssertEqual(Self.bridgeParse.problems, [])
    XCTAssertEqual(
      Self.bridgeParse.tupleCount, Self.expectedBridgeTuples,
      "ComfyBridge.swift dispatch tuple count moved — update the pin (and the enumeration below) in the same review")
  }

  /// §3.5 / R8: bridge routes are held to a DECLARED policy — no MCP tool
  /// required (ComfyUI/Krita clients speak their own protocol), but every one
  /// must be enumerated, so adding one is visible in review rather than
  /// invisible.
  func testBridgeRoutesAreEnumerated() {
    let expected: Set<RouteRef> = Set(
      [
        ("GET", "/system_stats"), ("GET", "/object_info"), ("GET", "/embeddings"),
        ("GET", "/settings"), ("GET", "/extensions"), ("GET", "/experiment/models"),
        ("GET", "/userdata*"), ("GET", "/users"), ("GET", "/queue"), ("GET", "/prompt"),
        ("GET", "/view"), ("GET", "/history"), ("GET", "/ws"),
        ("POST", "/queue"), ("POST", "/prompt"), ("POST", "/interrupt"), ("POST", "/upload/image"),
      ].map { RouteRef(method: $0.0, path: $0.1, surface: .comfyUICompat) })
    XCTAssertEqual(
      Self.bridgeParse.routes, expected,
      "Bridge dispatch changed — update this enumeration (and consider whether the new route belongs on the v1 surface instead)")
  }

  // MARK: - §3.5 assertion 3: every mutating v1 route claimed or exempted

  func testEveryMutatingV1RouteIsClaimedOrExempted() {
    let mutating = Self.warmParse.routes.filter { $0.method != "GET" }
    let claimed: Set<RouteRef> = Set(
      MCPToolRegistry.tools.flatMap(\.routes).map {
        RouteRef(method: $0.method, path: $0.path, surface: .v1)
      })
    let exempted = Set(ParityExemptions.all.map(\.route))

    for route in mutating {
      let isClaimed = claimed.contains(route)
      let isExempted = exempted.contains(route)
      XCTAssertTrue(
        isClaimed || isExempted,
        "Mutating route \(route.method) \(route.path) has no MCP tool claiming it (routes: field) and no ParityExemptions entry — add one or the other in this review")
    }
  }

  func testExemptionsAreReasonedAndNotStale() {
    for exemption in ParityExemptions.all {
      XCTAssertFalse(
        exemption.reason.trimmingCharacters(in: .whitespaces).isEmpty,
        "Exemption \(exemption.route.method) \(exemption.route.path) needs a non-empty reason")
      XCTAssertTrue(
        Self.warmParse.routes.contains(exemption.route),
        "Stale exemption: \(exemption.route.method) \(exemption.route.path) no longer parses from the dispatch switches — remove it")
    }
  }

  /// Claimed tool routes must be real dispatch routes too — a tool declaring a
  /// route the server does not serve is drift in the other direction.
  func testToolRouteClaimsResolveToParsedRoutes() {
    for tool in MCPToolRegistry.tools {
      for route in tool.routes {
        switch route.surface {
        case .v1:
          XCTAssertTrue(
            Self.warmParse.routes.contains(route),
            "Tool \(tool.name) claims \(route.method) \(route.path), which does not parse from WarmServer.swift")
        case .comfyUICompat:
          XCTAssertTrue(Self.bridgeParse.routes.contains(route))
        }
      }
    }
  }

  // MARK: - §3.5 assertion 4: descriptors resolve to real routes and tools

  func testDescriptorIdsAreUniqueAndSorted() {
    let ids = ControlRegistry.all.map(\.id)
    XCTAssertEqual(ids, ids.sorted(), "Registry must be sorted by id (deterministic wire/docs order)")
    XCTAssertEqual(ids.count, Set(ids).count, "Duplicate descriptor ids")
  }

  func testComfyboxDescriptorsResolveToRealRoutesAndTools() {
    for descriptor in ControlRegistry.all where descriptor.host == .comfybox {
      if let write = descriptor.write {
        XCTAssertEqual(write.host, .comfybox, "\(descriptor.id): comfybox descriptor with foreign write host")
        XCTAssertTrue(
          parsedV1Route(method: write.method, path: write.path),
          "\(descriptor.id): write route \(write.method) \(write.path) does not resolve to a parsed dispatch route")
      }
      if let read = descriptor.read {
        XCTAssertTrue(
          parsedV1Route(method: read.method, path: read.path),
          "\(descriptor.id): read route \(read.method) \(read.path) does not resolve to a parsed dispatch route")
      }
      if let tool = descriptor.mcpTool {
        XCTAssertNotNil(
          MCPToolRegistry.tool(named: tool),
          "\(descriptor.id): mcpTool '\(tool)' is not in MCPToolRegistry")
      }
    }
  }

  // MARK: - §3.5 assertion 5: every config key has a descriptor

  private func leafPointers(of json: Any, prefix: String = "") -> [String] {
    if let dict = json as? [String: Any] {
      if dict.isEmpty { return [] }
      return dict.flatMap { key, value -> [String] in
        let escaped = key
          .replacingOccurrences(of: "~", with: "~0")
          .replacingOccurrences(of: "/", with: "~1")
        return leafPointers(of: value, prefix: "\(prefix)/\(escaped)")
      }
    }
    // Arrays and scalars are leaves for coverage purposes.
    return [prefix]
  }

  func testEveryConfigKeyHasADescriptorOrIsDeclaredNonControl() throws {
    // "Someone added a config field and no descriptor" — the most likely
    // future drift. Walk the default document AFTER the first-run migration
    // (§3.3) so the renderDefaults/videoDefaults blocks are present too.
    var config = ComfyBoxServerConfig()
    _ = ServerConfigStore.migrateRenderAndVideoDefaults(into: &config, auditLog: nil)
    let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(config))

    let descriptorPointers: [String] = ControlRegistry.all.compactMap { descriptor in
      guard let write = descriptor.write, write.path == "/v1/config" else { return nil }
      return write.pointer
    }

    for pointer in leafPointers(of: object) {
      let covered = descriptorPointers.contains { descriptorPointer in
        pointer == descriptorPointer || pointer.hasPrefix(descriptorPointer + "/")
      }
      let declaredNonControl = ControlRegistry.nonControlKeys.contains { declared in
        pointer == declared || pointer.hasPrefix(declared + "/")
      }
      XCTAssertTrue(
        covered || declaredNonControl,
        "Config key \(pointer) has no ControlDescriptor and is not in ControlRegistry.nonControlKeys — add a descriptor (and regenerate docs) or declare it non-control")
    }
  }

  /// The registry's family lists come from ServerConfigStore; this pins them to
  /// the real dispatch enum so the two cannot silently drift (promised in
  /// ServerConfigStore.engineFamilies' doc comment).
  func testEngineFamiliesMatchWarmModelFamily() {
    XCTAssertEqual(
      Set(ServerConfigStore.engineFamilies),
      Set(WarmModelFamily.allCases.map(\.rawValue)))
    XCTAssertEqual(ServerConfigStore.videoEngineFamilies, ["ltx2"])
  }

  // MARK: - §4.5: no dangling read.pointer

  private func fullyPopulatedConfig() -> ComfyBoxServerConfig {
    let endpoint = AIProviderEndpoint(baseUrl: "http://localhost:1234/v1", model: "m", apiKey: "k")
    let render = RenderDefaultValues(width: 1024, height: 1024, steps: 9, guidance: 1.0)
    let video = VideoDefaultValues(width: 704, height: 448, frames: 97)
    return ComfyBoxServerConfig(
      port: 7870,
      host: "127.0.0.1",
      modelSpec: "krea2",
      allowedOutputDirectory: "/tmp/comfybox-out",
      seedvr2WeightsPath: "/tmp/seedvr2",
      providers: AIProviderRegistry(promptOptimization: endpoint, vision: endpoint, captioning: endpoint),
      replicate: ReplicateProviderConfig(
        apiKey: "k", baseUrl: "u", model: "m", imageModel: "im", videoModel: "vm"),
      contentModeDefaultPresets: ["banana": "preset-1"],
      krea2Models: ["krea2-raw": "/tmp/krea2-raw"],
      renderDefaults: RenderDefaultsConfig(
        default: render,
        byFamily: Dictionary(uniqueKeysWithValues: ServerConfigStore.engineFamilies.map { ($0, render) })),
      videoDefaults: VideoDefaultsConfig(
        default: video,
        byFamily: Dictionary(uniqueKeysWithValues: ServerConfigStore.videoEngineFamilies.map { ($0, video) })))
  }

  private func fullyPopulatedContentModes() -> ContentModeStore {
    // Built-ins leave optional fields (promptHint) empty on some modes; the
    // dangling-pointer check needs every field PRESENT so an absent key means
    // "wrong pointer", not "unset value".
    ContentModeStore(modes: ContentMode.allCases.map { mode in
      ContentModeDefinition(
        mode: mode, label: "L", summary: "S", guidanceBoost: 1.0,
        styleVariant: .neutral, promptHint: "hint", negativePromptAdditions: ["term"])
    })
  }

  func testNoDanglingReadPointer() {
    let values = ControlRegistry.resolveValues(
      config: fullyPopulatedConfig(),
      contentModes: fullyPopulatedContentModes(),
      queueDocument: ["is_paused": false])
    for descriptor in ControlRegistry.all {
      guard let read = descriptor.read, read.pointer != nil else { continue }
      XCTAssertNotNil(
        values[descriptor.id],
        "\(descriptor.id): read.pointer \(read.pointer ?? "") dangles — it dereferences to nothing even in a fully-populated document")
    }
  }

  // MARK: - Value resolution depth: production families (krea2 + ltx2)

  func testKrea2RenderDefaultsResolvePerRequest() {
    var config = ComfyBoxServerConfig()
    config.renderDefaults = RenderDefaultsConfig(
      byFamily: ["krea2": RenderDefaultValues(width: 832, height: 1216, steps: 32, guidance: 4.5)])
    let values = ControlRegistry.resolveValues(config: config, contentModes: ContentModeStore())
    XCTAssertEqual(values["render.defaults.krea2.width"], .int(832))
    XCTAssertEqual(values["render.defaults.krea2.height"], .int(1216))
    XCTAssertEqual(values["render.defaults.krea2.steps"], .int(32))
    XCTAssertEqual(values["render.defaults.krea2.guidance"], .double(4.5))
    // Unset knobs resolve to NO value (the engine constant applies) — never a
    // cached copy of some other document.
    XCTAssertNil(values["render.defaults.flux2.steps"])
    XCTAssertNil(values["render.defaults.steps"])
  }

  func testLTX2VideoDefaultsResolvePerRequest() {
    var config = ComfyBoxServerConfig()
    config.videoDefaults = VideoDefaultsConfig(
      byFamily: ["ltx2": VideoDefaultValues(width: 1024, height: 576, frames: 121)])
    let values = ControlRegistry.resolveValues(config: config, contentModes: ContentModeStore())
    XCTAssertEqual(values["video.defaults.ltx2.width"], .int(1024))
    XCTAssertEqual(values["video.defaults.ltx2.height"], .int(576))
    XCTAssertEqual(values["video.defaults.ltx2.frames"], .int(121))
  }

  func testEngineSeedDefaultsAreDeclaredOnDescriptors() throws {
    // krea2 seeds width/height only — steps/guidance track the loaded variant
    // (turbo 9 / raw 30) and MUST NOT be frozen as declared defaults (§3.3).
    func descriptor(_ id: String) throws -> ControlDescriptor {
      try XCTUnwrap(ControlRegistry.all.first { $0.id == id }, id)
    }
    XCTAssertEqual(try descriptor("render.defaults.krea2.width").defaultValue, .int(1024))
    XCTAssertEqual(try descriptor("render.defaults.krea2.height").defaultValue, .int(1024))
    XCTAssertNil(try descriptor("render.defaults.krea2.steps").defaultValue)
    XCTAssertNil(try descriptor("render.defaults.krea2.guidance").defaultValue)
    // ltx2 seeds are the engine's own prep-path fallbacks (704×448, 97 frames).
    XCTAssertEqual(try descriptor("video.defaults.ltx2.width").defaultValue, .int(704))
    XCTAssertEqual(try descriptor("video.defaults.ltx2.height").defaultValue, .int(448))
    XCTAssertEqual(try descriptor("video.defaults.ltx2.frames").defaultValue, .int(97))
  }

  func testControlsPayloadShape() throws {
    let data = try XCTUnwrap(ControlRegistry.controlsPayload(
      config: fullyPopulatedConfig(),
      contentModes: ContentModeStore(),
      queueDocument: ["is_paused": true]))
    let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    XCTAssertEqual(object["count"] as? Int, ControlRegistry.all.count)
    let controls = try XCTUnwrap(object["controls"] as? [[String: Any]])
    XCTAssertEqual(controls.count, ControlRegistry.all.count)

    let byId = Dictionary(uniqueKeysWithValues: controls.compactMap { entry in
      (entry["id"] as? String).map { ($0, entry) }
    })
    // One call answers "what can I change and how": a config knob carries its
    // machine-executable write action and its live value…
    let krea2Steps = try XCTUnwrap(byId["render.defaults.krea2.steps"])
    let write = try XCTUnwrap(krea2Steps["write"] as? [String: Any])
    XCTAssertEqual(write["method"] as? String, "PATCH")
    XCTAssertEqual(write["path"] as? String, "/v1/config")
    XCTAssertEqual(write["pointer"] as? String, "/renderDefaults/byFamily/krea2/steps")
    XCTAssertEqual(krea2Steps["value"] as? Int, 9)
    XCTAssertEqual(krea2Steps["mcpTool"] as? String, "patch_config")
    // …an action knob carries its POST and its live queue state…
    let pause = try XCTUnwrap(byId["queue.pause"])
    XCTAssertEqual((pause["write"] as? [String: Any])?["path"] as? String, "/v1/queue/pause")
    XCTAssertEqual(pause["value"] as? Bool, true)
    // …and a range encodes as {min, max}.
    let boost = try XCTUnwrap(byId["creative.contentMode.banana.guidanceBoost"])
    let range = try XCTUnwrap(boost["range"] as? [String: Any])
    XCTAssertEqual(range["min"] as? Double, ContentModeStore.guidanceBoostRange.lowerBound)
    XCTAssertEqual(range["max"] as? Double, ContentModeStore.guidanceBoostRange.upperBound)

    // Deterministic bytes: same inputs → identical payload (sync and async
    // arms share this function, so both emit these bytes).
    let again = try XCTUnwrap(ControlRegistry.controlsPayload(
      config: fullyPopulatedConfig(),
      contentModes: ContentModeStore(),
      queueDocument: ["is_paused": true]))
    XCTAssertEqual(data, again)
  }

  // MARK: - §4.5: docs generate idempotent and byte-matching

  func testAPIReferenceIsFresh() throws {
    let generated = try APIReferenceDoc.markdown(repoRoot: Self.repoRoot)
    let checkedInURL = Self.repoRoot.appendingPathComponent(APIReferenceDoc.relativeOutputPath())
    let checkedIn = try String(contentsOf: checkedInURL, encoding: .utf8)
    XCTAssertEqual(
      generated, checkedIn,
      "docs/api-reference.md is stale — run `comfybox docs generate` and commit the result")
  }

  func testDocsGenerationIsIdempotent() throws {
    let first = try APIReferenceDoc.markdown(repoRoot: Self.repoRoot)
    let second = try APIReferenceDoc.markdown(repoRoot: Self.repoRoot)
    XCTAssertEqual(first, second)
  }

  // MARK: - G2: tool route claims vs executor SOURCE (the hole G1 slipped through)

  /// Evidence extracted from MCPToolExecutor.swift: a path the executor really
  /// calls, with the HTTP method when the call form makes it derivable
  /// (`client.post("…")`, `call("PUT", "…")`, `executeGet("…")`, …) and nil
  /// when only the path literal is visible (variable/ternary paths).
  private struct ExecutorEvidence: Hashable {
    let method: String?
    let path: String
  }

  /// Claims that are intentionally indirect and reviewer-cleared. Today: the
  /// PUT-upsert aliases — the dispatch arms serve POST|PUT identically, the
  /// executor POSTs, and the tool claims both so the parity claim-map covers
  /// the PUT arm (same path, same surface).
  private static let executorClaimExemptions: Set<String> = [
    "create_preset PUT /v1/presets",
    "create_character PUT /v1/characters",
  ]

  private static let executorSource: String = {
    let url = repoRoot.appendingPathComponent("Sources/ZImage/MCP/MCPToolExecutor.swift")
    return (try? String(contentsOf: url, encoding: .utf8)) ?? ""
  }()

  private static func evidence(inLines lines: ArraySlice<String>) -> Set<ExecutorEvidence> {
    var result: Set<ExecutorEvidence> = []
    func matches(_ pattern: String, in line: String) -> [[String]] {
      guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
      let range = NSRange(line.startIndex..., in: line)
      return regex.matches(in: line, range: range).map { match in
        (1..<match.numberOfRanges).compactMap { index in
          Range(match.range(at: index), in: line).map { String(line[$0]) }
        }
      }
    }
    func normalize(_ path: String) -> String {
      // "\(expr)" interpolation segments are path parameters.
      (try? NSRegularExpression(pattern: #"\\\([^)]*\)"#)).map { regex in
        regex.stringByReplacingMatches(
          in: path, range: NSRange(path.startIndex..., in: path), withTemplate: "{id}")
      } ?? path
    }
    for line in lines {
      for m in matches(#"client\.(get|post|put|patch|delete)\(\s*"(/[^"]*)""#, in: line) {
        result.insert(ExecutorEvidence(method: m[0].uppercased(), path: normalize(m[1])))
      }
      for m in matches(#"call\(\s*"([A-Z]+)"\s*,\s*"(/[^"]*)""#, in: line) {
        result.insert(ExecutorEvidence(method: m[0], path: normalize(m[1])))
      }
      for m in matches(#"executeGet\(\s*"(/[^"]*)""#, in: line) {
        result.insert(ExecutorEvidence(method: "GET", path: normalize(m[0])))
      }
      for m in matches(#"executePostEmpty\(\s*"(/[^"]*)""#, in: line) {
        result.insert(ExecutorEvidence(method: "POST", path: normalize(m[0])))
      }
      // Any other path-shaped literal (variable paths, ternaries, helper
      // arguments like `route:` / executeNearlineAction) — method unknown.
      for m in matches(#""(/[^" ]*)""#, in: line) {
        result.insert(ExecutorEvidence(method: nil, path: normalize(m[0])))
      }
    }
    return result
  }

  /// tool name → executor evidence, from the dispatch switch in
  /// `MCPToolExecutor.execute(name:arguments:)`: each `case "tool":` body's
  /// inline evidence plus the evidence of every `execute*` method the body
  /// invokes (method bodies delimited by `func` starts).
  private static let executorEvidenceByTool: [String: Set<ExecutorEvidence>] = {
    let stripped = ControlSurfaceParser.stripComments(executorSource)
    let lines = stripped.components(separatedBy: "\n")

    // Method bodies: func name → line range.
    var funcStarts: [(name: String, line: Int)] = []
    for (index, line) in lines.enumerated() {
      if let regex = try? NSRegularExpression(pattern: #"func\s+(\w+)\s*\("#),
         let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
         let nameRange = Range(match.range(at: 1), in: line) {
        funcStarts.append((String(line[nameRange]), index))
      }
    }
    var evidenceByFunc: [String: Set<ExecutorEvidence>] = [:]
    var calleesByFunc: [String: Set<String>] = [:]
    func invokedHelpers(inLines body: ArraySlice<String>) -> Set<String> {
      // Helpers worth following: execute*/run* (runSetWarmPreset is the
      // composite behind set_warm_preset — one hop past executeSetWarmPreset).
      guard let regex = try? NSRegularExpression(pattern: #"((?:execute|run)\w+)\("#) else { return [] }
      var names: Set<String> = []
      for line in body {
        let range = NSRange(line.startIndex..., in: line)
        for match in regex.matches(in: line, range: range) {
          if let nameRange = Range(match.range(at: 1), in: line) {
            names.insert(String(line[nameRange]))
          }
        }
      }
      return names
    }
    for (offset, start) in funcStarts.enumerated() {
      let end = offset + 1 < funcStarts.count ? funcStarts[offset + 1].line : lines.count
      let body = lines[start.line..<end]
      evidenceByFunc[start.name] = evidence(inLines: body)
      calleesByFunc[start.name] = invokedHelpers(inLines: body)
    }
    // Transitive closure over the helper call graph, so evidence flows back
    // through composites (dispatch arm -> executeX -> runY -> call literals).
    func reachableEvidence(from roots: Set<String>) -> Set<ExecutorEvidence> {
      var visited: Set<String> = []
      var queue = Array(roots)
      var result: Set<ExecutorEvidence> = []
      while let name = queue.popLast() {
        guard !visited.contains(name) else { continue }
        visited.insert(name)
        if let found = evidenceByFunc[name] { result.formUnion(found) }
        if let callees = calleesByFunc[name] { queue.append(contentsOf: callees) }
      }
      return result
    }

    // Dispatch arms: case "tool": … (until the next case/default).
    var byTool: [String: Set<ExecutorEvidence>] = [:]
    var caseStarts: [(tool: String, line: Int)] = []
    for (index, line) in lines.enumerated() {
      if let regex = try? NSRegularExpression(pattern: #"case\s+"([a-z0-9_]+)"\s*:"#),
         let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)),
         let toolRange = Range(match.range(at: 1), in: line) {
        caseStarts.append((String(line[toolRange]), index))
      }
      if line.trimmingCharacters(in: .whitespaces).hasPrefix("default:"),
         let last = caseStarts.last, last.line < index {
        caseStarts.append(("__default__", index))
      }
    }
    for (offset, start) in caseStarts.enumerated() where start.tool != "__default__" {
      let end = offset + 1 < caseStarts.count ? caseStarts[offset + 1].line : lines.count
      let body = lines[start.line..<end]
      var toolEvidence = evidence(inLines: body)
      toolEvidence.formUnion(reachableEvidence(from: invokedHelpers(inLines: body)))
      byTool[start.tool, default: []].formUnion(toolEvidence)
    }
    return byTool
  }()

  func testToolRouteClaimsMatchExecutorSource() {
    XCTAssertFalse(Self.executorSource.isEmpty, "Could not read MCPToolExecutor.swift")
    for tool in MCPToolRegistry.tools where !tool.routes.isEmpty {
      guard let evidence = Self.executorEvidenceByTool[tool.name] else {
        XCTFail("Tool \(tool.name) declares routes but has no dispatch arm in MCPToolExecutor.execute — claim cannot be grounded")
        continue
      }
      for route in tool.routes {
        let key = "\(tool.name) \(route.method) \(route.path)"
        if Self.executorClaimExemptions.contains(key) { continue }
        let satisfied = evidence.contains { item in
          item.path == route.path && (item.method == nil || item.method == route.method)
        }
        XCTAssertTrue(
          satisfied,
          "Tool \(tool.name) claims \(route.method) \(route.path) (surface \(route.surface.rawValue)) but its executor never calls that path — executor evidence: \(evidence.map { "\($0.method ?? "?") \($0.path)" }.sorted()). Fix the routes: claim to declared reality or add a reasoned executorClaimExemptions entry")
      }
    }
  }

  /// The exemption table must stay grounded: every entry names a real tool and
  /// a route that tool actually claims.
  func testExecutorClaimExemptionsAreNotStale() {
    for key in Self.executorClaimExemptions {
      let parts = key.split(separator: " ").map(String.init)
      XCTAssertEqual(parts.count, 3, "Malformed exemption key: \(key)")
      guard parts.count == 3 else { continue }
      guard let tool = MCPToolRegistry.tool(named: parts[0]) else {
        XCTFail("Stale executor-claim exemption: no tool named \(parts[0])")
        continue
      }
      XCTAssertTrue(
        tool.routes.contains { $0.method == parts[1] && $0.path == parts[2] },
        "Stale executor-claim exemption: \(parts[0]) no longer claims \(parts[1]) \(parts[2])")
    }
  }

  // MARK: - Parser unit coverage (the recognizers §3.5 rule 3 requires)

  func testParserStripsCommentsAndPreservesStrings() {
    let source = """
      // case ("GET", "/fake"):
      /* case ("POST", "/fake"): */
      let url = "http://host/path" // trailing case ("PUT", "/fake"):
      case ("GET", "/real"):
      """
    let result = ControlSurfaceParser.parse(source: source, fileName: "x.swift", surface: .v1)
    XCTAssertEqual(result.routes, [RouteRef(method: "GET", path: "/real", surface: .v1)])
    XCTAssertEqual(result.tupleCount, 1)
    XCTAssertEqual(result.problems, [])
  }

  func testParserCountsMultiTupleArmsPerTuple() {
    let source = #"case ("POST", "/v1/queue/pause"), ("POST", "/v1/queue/resume"):"#
    let result = ControlSurfaceParser.parse(source: source, fileName: "x.swift", surface: .v1)
    XCTAssertEqual(result.tupleCount, 2)
    XCTAssertEqual(result.routes.count, 2)
  }

  func testParserAcceptsBothPrefixSuffixOrderings() {
    let a = #"case ("POST", _) where request.path.hasPrefix("/v1/queue/") && request.path.hasSuffix("/move"):"#
    let b = #"case ("POST", _) where request.path.hasSuffix("/move") && request.path.hasPrefix("/v1/queue/"):"#
    for source in [a, b] {
      let result = ControlSurfaceParser.parse(source: source, fileName: "x.swift", surface: .v1)
      XCTAssertEqual(
        result.routes, [RouteRef(method: "POST", path: "/v1/queue/{id}/move", surface: .v1)])
    }
  }

  func testParserAcceptsBridgeMethodWhereForm() {
    let source = #"case _ where request.method == "GET" && path.hasPrefix("/userdata"):"#
    let result = ControlSurfaceParser.parse(source: source, fileName: "x.swift", surface: .comfyUICompat)
    XCTAssertEqual(
      result.routes, [RouteRef(method: "GET", path: "/userdata*", surface: .comfyUICompat)])
  }

  func testParserFailsLoudOnUnrecognizedArm() {
    let source = #"case ("GET", _) where somethingElse(request):"#
    let result = ControlSurfaceParser.parse(source: source, fileName: "x.swift", surface: .v1)
    XCTAssertEqual(result.routes, [])
    XCTAssertEqual(result.problems.count, 1)
    XCTAssertTrue(result.problems[0].contains("unparsed dispatch arm at x.swift:1"))
  }
}
