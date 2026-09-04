// ControlRegistry.swift — Phase 4 discovery surface (FDD-ui-api-parity §3.4, D4;
// comfybox#300).
//
// The compile-time table behind `GET /v1/controls`. Anti-drift by construction
// (§3.4): the registry is load-bearing in three consumers — the route has no
// list of its own; `comfybox docs generate` emits `docs/api-reference.md` from
// the registry plus the parsed route table (with the parity test asserting the
// checked-in file byte-matches a fresh generation); and the anti-drift parity
// test (`ControlSurfaceParityTests`) holds descriptors, routes, MCP tools and
// exemptions to each other.
//
// The one rule (§3.4): the registry declares where a value lives and NEVER
// caches a copy. Values are resolved per-request by dereferencing
// `read.pointer` against `ServerConfigStore` / `ContentModeStore` / the live
// queue state — see ``ControlRegistry/controlsPayload(config:contentModes:queueDocument:)``.
//
// This file also hosts the two §3.4/§3.5 build-time consumers that read the
// registry alongside PARSED sources (never at serve time):
//   - ``ControlSurfaceParser`` — the §3.5 dispatch-switch parser (ground truth
//     for routes; comment-aware, tuple-counting, loud on unparsed arms);
//   - ``APIReferenceDoc`` — the `comfybox docs generate` markdown emitter.

import Foundation

// MARK: - Registry

public enum ControlRegistry {

  /// Convention shared by every config-document control: read via
  /// `GET /v1/config`, write via `PATCH /v1/config` (RFC 7386 merge patch)
  /// with the same JSON pointer, fronted by the `patch_config` MCP tool (§3.3:
  /// one route + one tool, no per-key drift engine).
  private static func configControl(
    id: String,
    pointer: String,
    title: String,
    summary: String,
    scope: ControlScope,
    type: ControlType,
    range: ClosedRange<Double>? = nil,
    unit: String? = nil,
    defaultValue: JSONValue? = nil,
    mutatesEngine: Bool = false,
    requiresRestart: Bool = false,
    since: String = "phase3"
  ) -> ControlDescriptor {
    ControlDescriptor(
      id: id, title: title, summary: summary, scope: scope, type: type,
      range: range, unit: unit, defaultValue: defaultValue,
      read: ActionRef(method: "GET", path: "/v1/config", pointer: pointer),
      write: ActionRef(method: "PATCH", path: "/v1/config", pointer: pointer),
      mcpTool: "patch_config",
      mutatesEngine: mutatesEngine, requiresRestart: requiresRestart, since: since)
  }

  private static func providerControls(capability: String, title: String) -> [ControlDescriptor] {
    [
      configControl(
        id: "provider.\(capability).baseUrl", pointer: "/providers/\(capability)/baseUrl",
        title: "\(title) base URL",
        summary: "OpenAI-compatible endpoint serving the \(capability) capability.",
        scope: .provider, type: .string),
      configControl(
        id: "provider.\(capability).model", pointer: "/providers/\(capability)/model",
        title: "\(title) model",
        summary: "Model identifier requested from the \(capability) endpoint.",
        scope: .provider, type: .string),
      configControl(
        id: "provider.\(capability).apiKey", pointer: "/providers/\(capability)/apiKey",
        title: "\(title) API key",
        summary: "Optional bearer credential for the \(capability) endpoint.",
        scope: .provider, type: .string),
    ]
  }

  private static func renderDefaultControls(family: String?) -> [ControlDescriptor] {
    // family == nil → the cross-family `default` block; else `byFamily.<family>`.
    let idBase = family.map { "render.defaults.\($0)" } ?? "render.defaults"
    let pointerBase = family.map { "/renderDefaults/byFamily/\($0)" } ?? "/renderDefaults/default"
    let who = family.map { "the \($0) family" } ?? "every family without a byFamily override"
    let seed = family.map { ServerConfigStore.engineSeed(family: $0) } ?? RenderDefaultValues()
    return [
      configControl(
        id: "\(idBase).width", pointer: "\(pointerBase)/width",
        title: "Default width",
        summary: "Config-layer default image width for \(who); resolution order request → preset → byFamily → default → engine constant (must be > 0).",
        scope: .engine, type: .int, unit: "px",
        defaultValue: seed.width.map { JSONValue.int($0) }, mutatesEngine: true),
      configControl(
        id: "\(idBase).height", pointer: "\(pointerBase)/height",
        title: "Default height",
        summary: "Config-layer default image height for \(who); resolution order request → preset → byFamily → default → engine constant (must be > 0).",
        scope: .engine, type: .int, unit: "px",
        defaultValue: seed.height.map { JSONValue.int($0) }, mutatesEngine: true),
      configControl(
        id: "\(idBase).steps", pointer: "\(pointerBase)/steps",
        title: "Default steps",
        summary: "Config-layer default inference step count for \(who) (must be > 0). flux2/krea2 have no seed here: their engine constant tracks the loaded checkpoint/variant.",
        scope: .engine, type: .int,
        defaultValue: seed.steps.map { JSONValue.int($0) }, mutatesEngine: true),
      configControl(
        id: "\(idBase).guidance", pointer: "\(pointerBase)/guidance",
        title: "Default guidance",
        summary: "Config-layer default guidance/CFG scale for \(who) (must be finite).",
        scope: .engine, type: .double,
        defaultValue: seed.guidance.map { JSONValue.double($0) }, mutatesEngine: true),
    ]
  }

  private static func videoDefaultControls(family: String?) -> [ControlDescriptor] {
    let idBase = family.map { "video.defaults.\($0)" } ?? "video.defaults"
    let pointerBase = family.map { "/videoDefaults/byFamily/\($0)" } ?? "/videoDefaults/default"
    let who = family.map { "the \($0) video engine" } ?? "every video family without a byFamily override"
    let seed = family.map { ServerConfigStore.videoEngineSeed(family: $0) } ?? VideoDefaultValues()
    return [
      configControl(
        id: "\(idBase).width", pointer: "\(pointerBase)/width",
        title: "Default video width",
        summary: "Config-layer default video width for \(who) (must be > 0).",
        scope: .engine, type: .int, unit: "px",
        defaultValue: seed.width.map { JSONValue.int($0) }, mutatesEngine: true),
      configControl(
        id: "\(idBase).height", pointer: "\(pointerBase)/height",
        title: "Default video height",
        summary: "Config-layer default video height for \(who) (must be > 0).",
        scope: .engine, type: .int, unit: "px",
        defaultValue: seed.height.map { JSONValue.int($0) }, mutatesEngine: true),
      configControl(
        id: "\(idBase).frames", pointer: "\(pointerBase)/frames",
        title: "Default video frames",
        summary: "Config-layer default frame count for \(who) (must be > 0; LTX-2 uses 1+8k).",
        scope: .engine, type: .int, unit: "frames",
        defaultValue: seed.frames.map { JSONValue.int($0) }, mutatesEngine: true),
    ]
  }

  private static func contentModeControls(mode: ContentMode) -> [ControlDescriptor] {
    // Defaults come from the SAME built-in table the engine uses, so the
    // registry cannot drift from ContentModeDefinition.builtin (R6 mitigation).
    let builtin = ContentModeDefinition.builtin(mode)
    let writePath = "/v1/content-modes/\(mode.rawValue)"
    // Read pointers address the keyed projection served in the controls
    // payload: { "<mode>": <definition> } (the list route serves an array;
    // pointers into array indices would be order-fragile).
    func descriptor(
      field: String, title: String, summary: String, type: ControlType,
      range: ClosedRange<Double>? = nil, allowed: [String]? = nil,
      defaultValue: JSONValue?
    ) -> ControlDescriptor {
      ControlDescriptor(
        id: "creative.contentMode.\(mode.rawValue).\(field)",
        title: "\(builtin.label): \(title)",
        summary: summary,
        scope: .creative, type: type, range: range, allowed: allowed,
        defaultValue: defaultValue,
        read: ActionRef(method: "GET", path: "/v1/content-modes", pointer: "/\(mode.rawValue)/\(field)"),
        write: ActionRef(method: "PUT", path: writePath, pointer: "/\(field)"),
        mcpTool: nil,  // No agent tool yet — PUT/DELETE /v1/content-modes/{mode} are reasoned ParityExemptions.
        mutatesEngine: true, since: "phase3")
    }
    return [
      descriptor(
        field: "guidanceBoost", title: "guidance boost",
        summary: "Added to the base guidance when the \(mode.rawValue) mode applies (suppressed on CFG-distilled models).",
        type: .double, range: ContentModeStore.guidanceBoostRange,
        defaultValue: .double(builtin.guidanceBoost)),
      descriptor(
        field: "promptHint", title: "prompt hint",
        summary: "Prompt fragment appended when the \(mode.rawValue) mode applies.",
        type: .string,
        defaultValue: builtin.promptHint.map { JSONValue.string($0) }),
      descriptor(
        field: "negativePromptAdditions", title: "negative prompt additions",
        summary: "Extra negative-prompt terms folded in for the \(mode.rawValue) mode.",
        type: .object,
        defaultValue: .array(builtin.negativePromptAdditions.map { JSONValue.string($0) })),
      descriptor(
        field: "styleVariant", title: "style variant",
        summary: "Style intent the renderer uses to pick preset/LoRA variants for the \(mode.rawValue) mode.",
        type: .enum, allowed: [
          ContentStyleVariant.neutral.rawValue,
          ContentStyleVariant.sensual.rawValue,
          ContentStyleVariant.nsfw.rawValue,
        ],
        defaultValue: .string(builtin.styleVariant.rawValue)),
    ]
  }

  private static func queueActionControls() -> [ControlDescriptor] {
    let read = ActionRef(method: "GET", path: "/v1/queue", pointer: "/is_paused")
    return [
      ControlDescriptor(
        id: "queue.pause", title: "Pause queue",
        summary: "Stop dequeuing new render jobs; the in-flight render finishes. Sync-servable during a render (0.B-2).",
        scope: .queue, type: .action, read: read,
        write: ActionRef(method: "POST", path: "/v1/queue/pause"),
        mcpTool: "pause_queue", since: "phase0"),
      ControlDescriptor(
        id: "queue.resume", title: "Resume queue",
        summary: "Resume dequeuing render jobs (served fire-and-forget — the one command that must wake the loop).",
        scope: .queue, type: .action, read: read,
        write: ActionRef(method: "POST", path: "/v1/queue/resume"),
        mcpTool: "resume_queue", since: "phase0"),
      ControlDescriptor(
        id: "queue.clear", title: "Clear queue",
        summary: "Drop all pending render jobs; the in-flight render is untouched."
          + " No MCP tool fronts this native route yet: the clear_queue tool posts the"
          + " ComfyUI-bridge queue-clear instead (declared reality, see ParityExemptions).",
        scope: .queue, type: .action,
        write: ActionRef(method: "POST", path: "/v1/queue/clear"),
        since: "phase0"),
      ControlDescriptor(
        id: "queue.interrupt", title: "Interrupt render",
        summary: "Abort the in-flight render at the next interruptible phase boundary.",
        scope: .queue, type: .action,
        write: ActionRef(method: "POST", path: "/v1/queue/interrupt"),
        mcpTool: "interrupt_render", mutatesEngine: true, since: "phase0"),
    ]
  }

  /// The compile-time table (§3.4), sorted by id for a deterministic wire and
  /// docs order. Every descriptor is `.comfybox`-hosted: Phase 2's federated
  /// `.kiraDaemon` descriptors (`docs/kira-control-api.md`) have not landed in
  /// this repo, and §3.5 step 4 covers them via a coffeeshop-server contract
  /// test, not here.
  public static let all: [ControlDescriptor] = {
    var controls: [ControlDescriptor] = []

    // Server identity — restart-scoped (FDD §3.3 hot-apply carve-out).
    controls.append(configControl(
      id: "server.port", pointer: "/port",
      title: "HTTP port",
      summary: "Warm-server listen port (canonical 7870). Applied on next server start.",
      scope: .engine, type: .int, range: 1...65535,
      defaultValue: .int(Int(ComfyBoxServerConfig.canonicalPort)),
      requiresRestart: true))
    controls.append(configControl(
      id: "server.host", pointer: "/host",
      title: "HTTP host",
      summary: "Warm-server bind address. Applied on next server start.",
      scope: .engine, type: .string,
      defaultValue: .string("127.0.0.1"),
      requiresRestart: true))

    // Model / engine paths.
    controls.append(configControl(
      id: "model.spec", pointer: "/modelSpec",
      title: "Warm model spec",
      summary: "The model spec the server warms at start; set_warm_preset writes it after activating (Phase 1). Applied on next server start.",
      scope: .model, type: .string, mutatesEngine: true, requiresRestart: true))
    controls.append(configControl(
      id: "model.krea2Models", pointer: "/krea2Models",
      title: "Krea-2 spec directories",
      summary: "Declared Krea-2 spec → model directory table, merged over the built-in defaults at server start (WP-E5).",
      scope: .model, type: .object, mutatesEngine: true, requiresRestart: true))
    controls.append(configControl(
      id: "engine.allowedOutputDirectory", pointer: "/allowedOutputDirectory",
      title: "Allowed output directory",
      summary: "Containment boundary for render output paths (distinct from the Desktop's local save location — §3.3).",
      scope: .engine, type: .string))
    controls.append(configControl(
      id: "engine.seedvr2WeightsPath", pointer: "/seedvr2WeightsPath",
      title: "SeedVR2 weights path",
      summary: "Weights directory for the SeedVR2 creative upscaler.",
      scope: .engine, type: .string, requiresRestart: true))

    // Image memory/resolution caps (#22, ImageMemoryPreflight) — read fresh
    // on every /v1/generate and ComfyUI-bridge /prompt call, same hot-apply
    // posture as renderDefaults.
    controls.append(configControl(
      id: "engine.imageMemoryCaps.maxLongEdge", pointer: "/imageMemoryCaps/maxLongEdge",
      title: "Max image long edge",
      summary: "Hard ceiling (px) on the longer of an image request's width/height — refused before any memory probing or model load.",
      scope: .engine, type: .int, unit: "px",
      defaultValue: .int(ImageMemoryCapsConfig.default.maxLongEdge), mutatesEngine: true))
    controls.append(configControl(
      id: "engine.imageMemoryCaps.maxPixels", pointer: "/imageMemoryCaps/maxPixels",
      title: "Max image pixels",
      summary: "Hard ceiling on an image request's width*height — refused before any memory probing or model load.",
      scope: .engine, type: .int,
      defaultValue: .int(ImageMemoryCapsConfig.default.maxPixels), mutatesEngine: true))
    controls.append(configControl(
      id: "engine.imageMemoryCaps.minAvailableHeadroomFraction", pointer: "/imageMemoryCaps/minAvailableHeadroomFraction",
      title: "Min memory headroom fraction",
      summary: "Fraction of live free system memory a render's estimated peak activation footprint must leave clear (0.10 = refuse above ~90% projected usage) — only enforced when enforceMemoryEstimate is true.",
      scope: .engine, type: .double, range: 0...1,
      defaultValue: .double(ImageMemoryCapsConfig.default.minAvailableHeadroomFraction), mutatesEngine: true))
    controls.append(configControl(
      id: "engine.imageMemoryCaps.enforceMemoryEstimate", pointer: "/imageMemoryCaps/enforceMemoryEstimate",
      title: "Enforce memory estimate",
      summary: "Whether the (uncalibrated) live-memory-budget estimate actually refuses a request. Default false — advisory only (logged + memory_estimate_bytes/memory_available_bytes on the response); the resolution cap above is always enforced regardless.",
      scope: .engine, type: .bool,
      defaultValue: .bool(ImageMemoryCapsConfig.default.enforceMemoryEstimate), mutatesEngine: true))

    // AI provider registry.
    controls += providerControls(capability: "promptOptimization", title: "Prompt optimization")
    controls += providerControls(capability: "vision", title: "Vision")
    controls += providerControls(capability: "captioning", title: "Captioning")
    for (field, note) in [
      ("apiKey", "API token"), ("baseUrl", "base URL"), ("model", "default model"),
      ("imageModel", "image model"), ("videoModel", "video model"),
    ] {
      controls.append(configControl(
        id: "provider.replicate.\(field)", pointer: "/replicate/\(field)",
        title: "Replicate \(note)",
        summary: "Replicate cloud proxy \(note) (remote video / fallback).",
        scope: .provider, type: .string))
    }

    // Creative layer.
    controls.append(configControl(
      id: "creative.contentModeDefaultPresets", pointer: "/contentModeDefaultPresets",
      title: "Content-mode default presets",
      summary: "Content mode (neutral/banana/avocado) → default preset id, applied when Generate's content mode changes.",
      scope: .creative, type: .object, mutatesEngine: true))
    for mode in ContentMode.allCases {
      controls += contentModeControls(mode: mode)
    }

    // Family-aware render/video defaults (FDD §3.3, D3). Families come from the
    // same lists the first-run migration seeds, which the parity test
    // cross-checks against WarmModelFamily so neither can silently drift.
    controls += renderDefaultControls(family: nil)
    for family in ServerConfigStore.engineFamilies {
      controls += renderDefaultControls(family: family)
    }
    controls += videoDefaultControls(family: nil)
    for family in ServerConfigStore.videoEngineFamilies {
      controls += videoDefaultControls(family: family)
    }

    // Queue control plane (Phase 0).
    controls += queueActionControls()

    return controls.sorted { $0.id < $1.id }
  }()

  /// Config-document JSON pointers that are deliberately NOT controls (§3.5
  /// step 5). Empty today: every key `ComfyBoxServerConfig` persists is
  /// discoverable. A future non-control key must be listed here (with the
  /// parity test failing until it is), so "someone added a config field and no
  /// descriptor" stays a loud event.
  public static let nonControlKeys: Set<String> = []

  // MARK: - Per-request value resolution (§3.4: never cached)

  /// Resolve current values for every descriptor with a readable pointer, by
  /// dereferencing `read.pointer` against the documents the descriptor's
  /// `read.path` names: `/v1/config` → the (lock-read) config snapshot,
  /// `/v1/content-modes` → the content-mode store keyed by mode,
  /// `/v1/queue` → the live queue payload. Pure over its inputs so tests
  /// never need `ServerConfigStore.shared` (K-FIX-1).
  public static func resolveValues(
    config: ComfyBoxServerConfig,
    contentModes: ContentModeStore,
    queueDocument: [String: Any]? = nil
  ) -> [String: JSONValue] {
    let encoder = JSONEncoder()
    let configObject = (try? encoder.encode(config))
      .flatMap { try? JSONSerialization.jsonObject(with: $0) } ?? [:]
    // Keyed projection for stable pointers (the list route serves an array).
    var modesObject: [String: Any] = [:]
    for definition in contentModes.listModes() {
      guard let data = try? encoder.encode(definition),
            let object = try? JSONSerialization.jsonObject(with: data) else { continue }
      modesObject[definition.mode.rawValue] = object
    }

    var values: [String: JSONValue] = [:]
    for descriptor in all {
      guard let read = descriptor.read, let pointer = read.pointer, read.host == .comfybox else { continue }
      let document: Any?
      switch read.path {
      case "/v1/config": document = configObject
      case "/v1/content-modes": document = modesObject
      case "/v1/queue": document = queueDocument
      default: document = nil
      }
      guard let document,
            let raw = JSONPointer.dereference(pointer, in: document),
            let value = JSONValue(any: raw) else { continue }
      values[descriptor.id] = value
    }
    return values
  }

  /// The `GET /v1/controls` payload: every descriptor plus its per-request
  /// resolved `value` (absent when unset or unresolvable — e.g. an action, or
  /// an optional config key with no value). Deterministic bytes (sorted keys,
  /// registry pre-sorted by id) so the async arm and the sync control plane
  /// emit identically.
  public static func controlsPayload(
    config: ComfyBoxServerConfig,
    contentModes: ContentModeStore,
    queueDocument: [String: Any]? = nil
  ) -> Data? {
    struct Entry: Encodable {
      let descriptor: ControlDescriptor
      let value: JSONValue?

      enum ValueKey: String, CodingKey { case value }

      func encode(to encoder: Encoder) throws {
        try descriptor.encode(to: encoder)
        var container = encoder.container(keyedBy: ValueKey.self)
        try container.encodeIfPresent(value, forKey: .value)
      }
    }
    struct Payload: Encodable {
      let controls: [Entry]
      let count: Int
    }

    let values = resolveValues(config: config, contentModes: contentModes, queueDocument: queueDocument)
    let entries = all.map { Entry(descriptor: $0, value: values[$0.id]) }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    return try? encoder.encode(Payload(controls: entries, count: entries.count))
  }
}

// MARK: - §3.5 dispatch-switch parser

/// Parses the two dispatch switches from SOURCE as ground truth for routes
/// (FDD §3.5, D5 — Swift has no runtime reflection over a `switch`, and
/// refactoring the arms into a data table would be a large risky edit to a file
/// Phases 0 and 3 already rewrote). The design's job is making parse failures
/// LOUD: every non-comment `case (` occurrence must be consumed by a
/// recognizer, and per-file tuple counts are pinned in the parity test.
public enum ControlSurfaceParser {

  public struct Result: Sendable {
    /// Routes extracted from recognized arms (normalized: `{id}` for a
    /// path-parameter segment, e.g. `/v1/queue/{id}/move`).
    public let routes: Set<RouteRef>
    /// Number of case-pattern tuples consumed — multi-tuple arms count each
    /// tuple (§3.5 rule 2: count tuples, not lines).
    public let tupleCount: Int
    /// Loud failures: "unparsed dispatch arm at <file>:<line>" (§3.5 rule /
    /// assertion 2). Empty on a clean parse.
    public let problems: [String]
  }

  /// Strip `//` line comments and (nested) `/* */` block comments, preserving
  /// string literals (route paths live in them) and line structure. §3.5 rule 1
  /// exists because `case (` occurs inside comments in `WarmServer.swift`.
  /// String-aware so a `"http://…"` literal never opens a bogus comment;
  /// interpolation-aware enough for `\(expr)` with balanced parens.
  static func stripComments(_ source: String) -> String {
    enum State {
      case code
      case lineComment
      case blockComment(depth: Int)
      case string(multiline: Bool)
      case rawString
      case interpolation(depth: Int, multiline: Bool, inInnerString: Bool)
    }
    var out = String()
    out.reserveCapacity(source.count)
    var state = State.code
    let chars = Array(source)
    var i = 0
    func peek(_ offset: Int) -> Character? {
      let index = i + offset
      return index < chars.count ? chars[index] : nil
    }
    while i < chars.count {
      let c = chars[i]
      switch state {
      case .code:
        if c == "/" && peek(1) == "/" {
          state = .lineComment
          out.append("  ")
          i += 2
          continue
        }
        if c == "/" && peek(1) == "*" {
          state = .blockComment(depth: 1)
          out.append("  ")
          i += 2
          continue
        }
        if c == "#" && peek(1) == "\"" {
          state = .rawString
          out.append("#\"")
          i += 2
          continue
        }
        if c == "\"" {
          if peek(1) == "\"" && peek(2) == "\"" {
            state = .string(multiline: true)
            out.append("\"\"\"")
            i += 3
            continue
          }
          state = .string(multiline: false)
        }
        out.append(c)
      case .lineComment:
        if c == "\n" {
          state = .code
          out.append(c)
        } else {
          out.append(" ")
        }
      case .blockComment(let depth):
        if c == "/" && peek(1) == "*" {
          state = .blockComment(depth: depth + 1)
          out.append("  ")
          i += 2
          continue
        }
        if c == "*" && peek(1) == "/" {
          state = depth > 1 ? .blockComment(depth: depth - 1) : .code
          out.append("  ")
          i += 2
          continue
        }
        out.append(c == "\n" ? "\n" : " ")
      case .rawString:
        if c == "\"" && peek(1) == "#" {
          state = .code
          out.append("\"#")
          i += 2
          continue
        }
        out.append(c)
      case .string(let multiline):
        if c == "\\" {
          if peek(1) == "(" {
            state = .interpolation(depth: 1, multiline: multiline, inInnerString: false)
            out.append("\\(")
            i += 2
            continue
          }
          out.append(c)
          if let next = peek(1) {
            out.append(next)
            i += 2
            continue
          }
        } else if multiline, c == "\"", peek(1) == "\"", peek(2) == "\"" {
          state = .code
          out.append("\"\"\"")
          i += 3
          continue
        } else if !multiline, c == "\"" {
          state = .code
          out.append(c)
        } else {
          out.append(c)
        }
      case .interpolation(let depth, let multiline, let inInnerString):
        if inInnerString {
          if c == "\\", peek(1) != nil {
            out.append(c)
            out.append(chars[i + 1])
            i += 2
            continue
          }
          if c == "\"" {
            state = .interpolation(depth: depth, multiline: multiline, inInnerString: false)
          }
          out.append(c)
        } else {
          if c == "\"" {
            state = .interpolation(depth: depth, multiline: multiline, inInnerString: true)
          } else if c == "(" {
            state = .interpolation(depth: depth + 1, multiline: multiline, inInnerString: false)
          } else if c == ")" {
            state = depth > 1
              ? .interpolation(depth: depth - 1, multiline: multiline, inInnerString: false)
              : .string(multiline: multiline)
          }
          out.append(c)
        }
      }
      i += 1
    }
    return out
  }

  /// Recognize one dispatch arm line. Handles (§3.5 rule 3):
  ///   - literal tuples, including multi-tuple arms:
  ///     `case ("POST", "/v1/queue/pause"), ("POST", "/v1/queue/resume"):`
  ///   - wildcard + `where` with `hasPrefix`/`hasSuffix` in EITHER order:
  ///     `case ("POST", _) where request.path.hasPrefix("…") && request.path.hasSuffix("…"):`
  ///   - the bridge's `case _ where request.method == "GET" && path.hasPrefix("…")` form.
  /// Returns the extracted routes, or nil when the line is not a recognizable arm.
  static func recognizeArm(line: String, surface: RouteSurface) -> [RouteRef]? {
    let trimmed = line.trimmingCharacters(in: .whitespaces)

    func stringLiterals(matching pattern: String) -> [String] {
      guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
      let range = NSRange(line.startIndex..., in: line)
      return regex.matches(in: line, range: range).compactMap { match in
        guard match.numberOfRanges > 1, let r = Range(match.range(at: 1), in: line) else { return nil }
        return String(line[r])
      }
    }

    func wildcardRoute(method: String) -> RouteRef? {
      let prefixes = stringLiterals(matching: #"hasPrefix\("([^"]*)"\)"#)
      let suffixes = stringLiterals(matching: #"hasSuffix\("([^"]*)"\)"#)
      guard prefixes.count == 1, suffixes.count <= 1 else { return nil }
      let prefix = prefixes[0]
      let suffix = suffixes.first ?? ""
      let path = prefix.hasSuffix("/") ? prefix + "{id}" + suffix : prefix + "*" + suffix
      return RouteRef(method: method, path: path, surface: surface)
    }

    if trimmed.hasPrefix("case _ where") {
      let methods = stringLiterals(matching: #"method\s*==\s*"([A-Z]+)""#)
      guard methods.count == 1, let route = wildcardRoute(method: methods[0]) else { return nil }
      return [route]
    }

    guard trimmed.hasPrefix("case (") else { return nil }

    if line.contains(" where ") || line.contains(") where") {
      let methods = stringLiterals(matching: #"\(\s*"([A-Z]+)"\s*,\s*_\s*\)"#)
      guard methods.count == 1, let route = wildcardRoute(method: methods[0]) else { return nil }
      return [route]
    }

    guard let tupleRegex = try? NSRegularExpression(pattern: #"\(\s*"([A-Z]+)"\s*,\s*"([^"]*)"\s*\)"#) else {
      return nil
    }
    let range = NSRange(line.startIndex..., in: line)
    let routes: [RouteRef] = tupleRegex.matches(in: line, range: range).compactMap { match in
      guard let methodRange = Range(match.range(at: 1), in: line),
            let pathRange = Range(match.range(at: 2), in: line) else { return nil }
      return RouteRef(method: String(line[methodRange]), path: String(line[pathRange]), surface: surface)
    }
    return routes.isEmpty ? nil : routes
  }

  /// Parse one dispatch file. Every non-comment `case (` occurrence (and the
  /// bridge's `case _ where request.method` form) must be consumed by
  /// ``recognizeArm(line:surface:)`` — anything else lands in `problems` as
  /// "unparsed dispatch arm at <file>:<line>" rather than being skipped.
  public static func parse(source: String, fileName: String, surface: RouteSurface) -> Result {
    let stripped = stripComments(source)
    var routes: Set<RouteRef> = []
    var tupleCount = 0
    var problems: [String] = []

    let lines = stripped.components(separatedBy: "\n")
    for (index, line) in lines.enumerated() {
      let trimmed = line.trimmingCharacters(in: .whitespaces)
      let isCandidate = trimmed.contains("case (")
        || trimmed.hasPrefix("case _ where request.method")
        || (trimmed.hasPrefix("case _ where") && trimmed.contains("method =="))
      guard isCandidate else { continue }
      if let armRoutes = recognizeArm(line: line, surface: surface) {
        tupleCount += armRoutes.count
        routes.formUnion(armRoutes)
      } else {
        problems.append("unparsed dispatch arm at \(fileName):\(index + 1): \(trimmed)")
      }
    }
    return Result(routes: routes, tupleCount: tupleCount, problems: problems)
  }

  public static func parse(fileAt url: URL, surface: RouteSurface) throws -> Result {
    let source = try String(contentsOf: url, encoding: .utf8)
    return parse(source: source, fileName: url.lastPathComponent, surface: surface)
  }

  /// The two dispatch files (§3.5: "parse both files" — bridge routes are real
  /// mutating routes reached BEFORE the main switch and must be enumerated).
  public static func warmServerSource(repoRoot: URL) -> URL {
    repoRoot.appendingPathComponent("Sources/ZImage/Server/WarmServer.swift")
  }

  public static func comfyBridgeSource(repoRoot: URL) -> URL {
    repoRoot.appendingPathComponent("Sources/ZImage/Server/ComfyBridge/ComfyBridge.swift")
  }
}

// MARK: - `comfybox docs generate` emitter

/// Emits `docs/api-reference.md` from the registry plus the parsed route table
/// (§3.4 consumer 3). Deterministic: sorted routes, registry pre-sorted by id,
/// no timestamps — the parity test asserts the checked-in file byte-matches a
/// fresh generation, so a route/control change without `comfybox docs generate`
/// fails CI.
public enum APIReferenceDoc {

  public static func relativeOutputPath() -> String { "docs/api-reference.md" }

  public static func markdown(repoRoot: URL) throws -> String {
    let warm = try ControlSurfaceParser.parse(
      fileAt: ControlSurfaceParser.warmServerSource(repoRoot: repoRoot), surface: .v1)
    let bridge = try ControlSurfaceParser.parse(
      fileAt: ControlSurfaceParser.comfyBridgeSource(repoRoot: repoRoot), surface: .comfyUICompat)
    let problems = warm.problems + bridge.problems
    guard problems.isEmpty else {
      throw NSError(
        domain: "APIReferenceDoc", code: 1,
        userInfo: [NSLocalizedDescriptionKey: "Dispatch parse failed:\n" + problems.joined(separator: "\n")])
    }
    return markdown(v1Routes: warm.routes, bridgeRoutes: bridge.routes)
  }

  static func markdown(v1Routes: Set<RouteRef>, bridgeRoutes: Set<RouteRef>) -> String {
    func sorted(_ routes: Set<RouteRef>) -> [RouteRef] {
      routes.sorted { ($0.path, $0.method) < ($1.path, $1.method) }
    }

    var toolsByRoute: [RouteRef: [String]] = [:]
    for tool in MCPToolRegistry.tools {
      for route in tool.routes {
        toolsByRoute[route, default: []].append(tool.name)
      }
    }
    let exemptionsByRoute = Dictionary(
      uniqueKeysWithValues: ParityExemptions.all.map { ($0.route, $0.reason) })

    var out = """
    # ComfyBox Server — API Reference

    > **GENERATED FILE — do not edit by hand.** Regenerate with `comfybox docs generate`
    > (run from the repo root). CI byte-compares this file against a fresh generation
    > (`ControlSurfaceParityTests`), so a stale copy fails the build.
    >
    > Sources of truth (FDD-ui-api-parity §3.4): the dispatch switches in
    > `Sources/ZImage/Server/WarmServer.swift` and
    > `Sources/ZImage/Server/ComfyBridge/ComfyBridge.swift` (parsed), the compile-time
    > `ControlRegistry`, `MCPToolRegistry`, and `ParityExemptions`.

    The warm server (`comfybox serve`, default port **7870**) exposes an HTTP/JSON API.
    `{id}` marks a path parameter. Mutating `v1` routes are each claimed by an MCP tool
    or carry a reasoned exemption (§3.5 assertion 3).

    > **See also: [`api-notes.md`](api-notes.md)** — the HAND-MAINTAINED companion with
    > body schemas and operational guidance (LTX-2 video bodies, `POST /v1/enhance`,
    > LoRA `role` semantics, Krea-2 preset kroma rules, startup imports). This file is
    > regenerated wholesale; durable prose belongs there, never here.

    ## Warm-server routes (`surface: v1`)

    | Method | Path | MCP tools | Exemption |
    |---|---|---|---|

    """
    for route in sorted(v1Routes) {
      let tools = toolsByRoute[route]?.sorted().joined(separator: ", ") ?? ""
      let exemption = exemptionsByRoute[route] ?? ""
      out += "| \(route.method) | `\(route.path)` | \(tools) | \(exemption) |\n"
    }

    out += """

    ## ComfyUI compatibility bridge routes (`surface: comfyUICompat`)

    Served by `ComfyBridge.route()` for ComfyUI/Krita clients, BEFORE the main switch.
    Declared policy (§3.5): these need no MCP tool, but every one must be enumerated here
    so adding one is visible in review. A tool listed here proxies the bridge path by
    declared contract (e.g. `clear_queue`).

    | Method | Path | MCP tools |
    |---|---|---|

    """
    for route in sorted(bridgeRoutes) {
      let tools = toolsByRoute[route]?.sorted().joined(separator: ", ") ?? ""
      out += "| \(route.method) | `\(route.path)` | \(tools) |\n"
    }

    out += """

    ## Controls (`GET /v1/controls`)

    One call answers "what can I change and how" (§3.4). Each row is a
    `ControlDescriptor`; `GET /v1/controls` returns these plus per-request resolved
    `value`s. Writes with a pointer are JSON-pointer targets for the write route's
    document (config writes: RFC 7386 merge patch via `PATCH /v1/config`).

    | Id | Scope | Type | Range | Default | Write | MCP tool | Restart |
    |---|---|---|---|---|---|---|---|

    """
    for descriptor in ControlRegistry.all {
      let range = descriptor.range.map { format(min: $0.lowerBound, max: $0.upperBound) } ?? ""
      let defaultValue = descriptor.defaultValue.map(render) ?? ""
      let write = descriptor.write.map { ref -> String in
        var text = "\(ref.method) `\(ref.path)`"
        if let pointer = ref.pointer { text += " @ `\(pointer)`" }
        return text
      } ?? ""
      let tool = descriptor.mcpTool ?? ""
      let restart = descriptor.requiresRestart ? "yes" : ""
      out += "| `\(descriptor.id)` | \(descriptor.scope.rawValue) | \(descriptor.type.rawValue) "
        + "| \(range) | \(defaultValue) | \(write) | \(tool) | \(restart) |\n"
    }

    return out
  }

  private static func format(min: Double, max: Double) -> String {
    "\(number(min))–\(number(max))"
  }

  private static func number(_ value: Double) -> String {
    value == value.rounded() && abs(value) < 1e15
      ? String(Int(value))
      : String(value)
  }

  private static func render(_ value: JSONValue) -> String {
    switch value {
    case .null: return "null"
    case .bool(let b): return b ? "true" : "false"
    case .int(let i): return String(i)
    case .double(let d): return number(d)
    case .string(let s): return s.isEmpty ? "\"\"" : "`\(s)`"
    case .array(let items): return items.isEmpty ? "[]" : "[" + items.map(render).joined(separator: ", ") + "]"
    case .object: return "(object)"
    }
  }
}
