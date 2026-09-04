import Foundation

// Task #9 Phase 1 (specs/ltx2-parameter-externalization.md §4): one resolution
// pass yielding every Tier A/B parameter with PROVENANCE. Exists because the
// 2026-07-30 guidance-rescale incident (env var silently absent → different
// render) was invisible: nothing showed which values a render actually used
// or where each came from.
//
// Precedence (spec §3): request > preset > configFile > env > builtin.
// Phase 1 resolves the configFile/env/builtin levels — the request/preset
// levels join in Phase 2 when the resolved object threads into the pipeline.

public enum LTX2ParamSource: String, Codable, Sendable {
  case request, preset, configFile, env, builtin
}

public struct LTX2ResolvedParam: Codable, Sendable, Equatable {
  public let name: String
  public let envKey: String?
  public let tier: String        // "A" render-shaping, "B" machine-shaped
  public let value: String
  public let source: LTX2ParamSource
  public let valid: Bool         // false: input was rejected or path missing
  public let note: String?
}

public enum LTX2ConfigResolver {

  // MARK: Registry

  private enum Kind {
    case float(ClosedRange<Double>)
    case int(ClosedRange<Int>)
    /// Renderer semantics: enabled ONLY when the RAW (untrimmed) value is
    /// exactly "1". Anything else is false — and flagged, so the readout
    /// never claims a value the renderer won't act on (Codex finding #17).
    case boolExactOne
    /// Renderer semantics: DISABLED only when raw value is exactly "0";
    /// any other value leaves it enabled.
    case boolNotZero
    case string
    case floatList
    case path
  }

  private struct Entry {
    let name: String
    let envKey: String?
    let tier: String
    let kind: Kind
    let builtin: String
    // Builtin defaults mirror the actual point-of-use reads, e.g.
    // LTX2Pipeline.swift:1029 (rescale 0), :1277 (plain vol 4500),
    // :1376 (refine vol 12_000), LTX2VideoGenerator.swift:606 (compression 35),
    // :686 (face anchor 0.5), LTX2PipelineConfig.swift:189 (stg 0).
  }

  private static let registry: [Entry] = [
    // Tier A — render-shaping
    Entry(name: "guidance_rescale", envKey: "LTX2_GUIDANCE_RESCALE", tier: "A", kind: .float(0...1), builtin: "0"),
    Entry(name: "cfg_schedule", envKey: "LTX2_CFG_SCHEDULE", tier: "A", kind: .floatList, builtin: ""),
    Entry(name: "stage1_sigmas", envKey: "LTX2_STAGE1_SIGMAS", tier: "A", kind: .floatList, builtin: ""),
    Entry(name: "refine_sigmas", envKey: "LTX2_REFINE_SIGMAS", tier: "A", kind: .floatList, builtin: ""),
    Entry(name: "two_stage", envKey: "LTX2_TWO_STAGE", tier: "A", kind: .boolExactOne, builtin: "false"),
    // Refine the AUDIO track on the second pass as well as the video. Was
    // ProcessInfo-only (LTX2_AUDIO_REFINE=1), so it could not be set per
    // render — only globally, with an engine restart. Tier A so it can ride
    // the request `tuning` block with the two-pass quality tier.
    Entry(name: "audio_refine", envKey: "LTX2_AUDIO_REFINE", tier: "A", kind: .boolExactOne, builtin: "false"),
    Entry(name: "cond_fps", envKey: "LTX2_COND_FPS", tier: "A", kind: .float(1...120), builtin: "model"),
    Entry(name: "delivery_short_edge", envKey: "LTX2_DELIVERY_SHORT_EDGE", tier: "A", kind: .int(0...4320), builtin: "0"),
    Entry(name: "refine_scale", envKey: "LTX2_REFINE_SCALE", tier: "A", kind: .float(1...2), builtin: "1.5"),
    Entry(name: "img_compression", envKey: "LTX2_I2V_COMPRESSION", tier: "A", kind: .int(0...100), builtin: "35"),
    Entry(name: "sampler", envKey: "LTX2_SAMPLER", tier: "A", kind: .string, builtin: ""),
    Entry(name: "stg_scale", envKey: "LTX2_STG_SCALE", tier: "A", kind: .float(0...20), builtin: "0"),
    Entry(name: "stg_blocks", envKey: "LTX2_STG_BLOCKS", tier: "A", kind: .string, builtin: ""),
    Entry(name: "face_anchor_strength", envKey: "LTX2_FACE_ANCHOR_STRENGTH", tier: "A", kind: .float(0...1), builtin: "0.5"),
    Entry(name: "ic_control", envKey: "LTX2_IC_CONTROL", tier: "A", kind: .boolNotZero, builtin: "true"),
    Entry(name: "ic_ref_strength", envKey: "LTX2_IC_REF_STRENGTH", tier: "A", kind: .float(0...1), builtin: "1.0"),
    Entry(name: "color_anchor", envKey: "LTX2_COLOR_ANCHOR", tier: "A", kind: .float(0...1), builtin: "0"),
    Entry(name: "nag_scale", envKey: "LTX2_NAG_SCALE", tier: "A", kind: .float(0...50), builtin: "0"),
    Entry(name: "nag_alpha", envKey: "LTX2_NAG_ALPHA", tier: "A", kind: .float(0...1), builtin: "0.25"),
    Entry(name: "nag_tau", envKey: "LTX2_NAG_TAU", tier: "A", kind: .float(1...10), builtin: "2.5"),
    Entry(name: "reanchor_interval", envKey: "LTX2_REANCHOR_INTERVAL", tier: "A", kind: .int(0...10_000), builtin: "0"),
    Entry(name: "reanchor_strength", envKey: "LTX2_REANCHOR_STRENGTH", tier: "A", kind: .float(0...1), builtin: "0"),
    // Temporal beat scheduling kill switch (comfybox#310): default ON —
    // disabled only when the RAW env value is exactly "0", mirroring the
    // renderer (LTX2Pipeline reads this via LTX2ResolvedVideoConfig, never
    // ProcessInfo directly). A request with no `beat_schedule` field costs
    // nothing regardless of this switch; it exists to kill the feature
    // globally without a request-shape change if it misbehaves live.
    Entry(name: "beat_schedule_enabled", envKey: "LTX2_BEAT_SCHEDULE", tier: "A", kind: .boolNotZero, builtin: "true"),
    // Tier B — machine-shaped
    Entry(name: "plain_decode_max_vol", envKey: "LTX2_PLAIN_DECODE_MAX_VOL", tier: "B", kind: .int(1...1_000_000), builtin: "4500"),
    Entry(name: "refine_max_vol", envKey: "LTX2_REFINE_MAX_VOL", tier: "B", kind: .int(1...1_000_000), builtin: "12000"),
    Entry(name: "decode_mode", envKey: "LTX2_DECODE_MODE", tier: "B", kind: .string, builtin: "auto"),
    Entry(name: "decode_tile", envKey: "LTX2_DECODE_TILE", tier: "B", kind: .string, builtin: ""),
    Entry(name: "upsampler_path", envKey: "LTX2_UPSAMPLER_PATH", tier: "B", kind: .path, builtin: ""),
    Entry(name: "video_bits_per_px", envKey: "LTX2_VIDEO_BITS_PER_PX", tier: "B", kind: .float(0.01...20), builtin: "0.5"),
  ]

  // MARK: Resolution

  /// Resolve every registry parameter from configFile > env > builtin.
  /// Pure given its inputs — production callers use the defaults.
  public static func resolveEffective(
    environment: [String: String] = ProcessInfo.processInfo.environment,
    configFile: [String: String]? = nil,
    fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
  ) -> [LTX2ResolvedParam] {
    let fileValues = configFile ?? loadConfigFileVideoSection()
    return registry.map { entry in
      resolve(entry, fileValues: fileValues, environment: environment, fileExists: fileExists)
    }
  }

  private static func resolve(
    _ entry: Entry,
    fileValues: [String: String],
    environment: [String: String],
    fileExists: (String) -> Bool
  ) -> LTX2ResolvedParam {
    let candidates: [(String?, LTX2ParamSource)] = [
      (fileValues[entry.name], .configFile),
      (entry.envKey.flatMap { environment[$0] }, .env),
    ]

    // Invalid values FALL THROUGH to the next lower-precedence source
    // (Codex finding #17) — flagged, never silently — instead of jumping
    // straight to builtin past a perfectly valid env value.
    var rejections: [String] = []

    for (raw, source) in candidates {
      guard let raw else { continue }
      let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !trimmed.isEmpty else { continue }

      switch validate(trimmed, raw: raw, kind: entry.kind, fileExists: fileExists) {
      case .ok(let canonical):
        return LTX2ResolvedParam(
          name: entry.name, envKey: entry.envKey, tier: entry.tier,
          value: canonical, source: source, valid: rejections.isEmpty,
          note: rejections.isEmpty ? nil : rejections.joined(separator: "; "))
      case .keepButFlag(let canonical, let note):
        // e.g. a path that doesn't exist: it IS what the render will try to
        // use, so report it as the effective value — flagged.
        return LTX2ResolvedParam(
          name: entry.name, envKey: entry.envKey, tier: entry.tier,
          value: canonical, source: source, valid: false,
          note: (rejections + [note]).joined(separator: "; "))
      case .reject(let note):
        rejections.append("\(source.rawValue) value '\(trimmed)' rejected: \(note)")
      }
    }

    return LTX2ResolvedParam(
      name: entry.name, envKey: entry.envKey, tier: entry.tier,
      value: entry.builtin, source: .builtin, valid: rejections.isEmpty,
      note: rejections.isEmpty ? nil : rejections.joined(separator: "; "))
  }

  private enum Validation {
    case ok(String)
    case keepButFlag(String, String)
    case reject(String)
  }

  private static func validate(
    _ trimmed: String, raw: String, kind: Kind, fileExists: (String) -> Bool
  ) -> Validation {
    switch kind {
    case .float(let range):
      guard let v = Double(trimmed), v.isFinite else { return .reject("not a finite number") }
      guard range.contains(v) else { return .reject("outside range \(range)") }
      return .ok(canonicalNumber(v))
    case .int(let range):
      guard let v = Int(trimmed) else { return .reject("not an integer") }
      guard range.contains(v) else { return .reject("outside range \(range)") }
      return .ok(String(v))
    case .boolExactOne:
      // Mirror the renderer EXACTLY: it compares the raw env string to "1".
      if raw == "1" { return .ok("true") }
      if raw == "0" { return .ok("false") }
      return .keepButFlag("false",
        "renderer enables this only for exactly '1' — got '\(raw)', treating as OFF")
    case .boolNotZero:
      if raw == "0" { return .ok("false") }
      if raw == "1" { return .ok("true") }
      return .keepButFlag("true",
        "renderer disables this only for exactly '0' — got '\(raw)', treating as ON")
    case .string:
      return .ok(trimmed)
    case .floatList:
      let parts = trimmed.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
      for p in parts where Double(p)?.isFinite != true {
        return .reject("'\(p)' is not a finite number")
      }
      return .ok(parts.joined(separator: ","))
    case .path:
      guard fileExists(trimmed) else { return .keepButFlag(trimmed, "path does not exist") }
      return .ok(trimmed)
    }
  }

  private static func canonicalNumber(_ v: Double) -> String {
    v == v.rounded() && abs(v) < 1e15
      ? String(Int(v))
      : String(v)
  }

  /// The `video` section of ~/.comfybox/config.json, values stringified.
  /// Missing file/section is an empty dict — env and builtins still resolve.
  public static func loadConfigFileVideoSection() -> [String: String] {
    let url = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".comfybox/config.json")
    guard let data = try? Data(contentsOf: url),
          let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let video = root["video"] as? [String: Any]
    else { return [:] }
    var out: [String: String] = [:]
    for (k, v) in video { out[k] = "\(v)" }
    return out
  }
}

// MARK: - Phase 2: typed resolution (request > preset > configFile > env > builtin)

/// Tier A tuning fields a REQUEST or PRESET may carry. All optional — nil
/// means "defer to the next precedence level". This is the wire shape for
/// both the video request extension and the preset `videoTuning` block.
public struct LTX2VideoTuning: Codable, Sendable, Equatable {
  public var guidanceRescale: Float?
  public var cfgSchedule: [Float]?
  public var stage1Sigmas: [Float]?
  public var refineSigmas: [Float]?
  public var twoStage: Bool?
  public var audioRefine: Bool?
  public var condFps: Float?
  public var imgCompression: Int?
  /// Delivery downscale: mux output short edge (0 = off). Render runs the
  /// full recipe; only the encoded file is scaled — supersampled delivery.
  public var deliveryShortEdge: Int?
  /// Refine target scale (1...2). The upsampler is fixed 2x; below 2 the
  /// upsampled latent is bilinearly resized down before the refine denoise.
  public var refineScale: Float?
  public var sampler: String?
  public var stgScale: Float?
  public var stgBlocks: String?
  public var faceAnchorStrength: Float?
  public var icControl: Bool?
  public var icRefStrength: Float?
  public var colorAnchor: Float?
  public var nagScale: Float?
  public var nagAlpha: Float?
  public var nagTau: Float?
  public var reanchorInterval: Int?
  public var reanchorStrength: Float?
  public init() {}
}

/// The authoritative, TYPED configuration for one render. Codex finding #14:
/// this object — not scattered env reads — is what render code must consume.
/// `params` carries the string readout (for the API/card/log); `provenance`
/// records which level supplied each final value.
public struct LTX2ResolvedVideoConfig: Sendable {
  // Tier A
  public let guidanceRescale: Float
  public let cfgSchedule: [Float]
  public let stage1Sigmas: [Float]
  public let refineSigmas: [Float]
  public let twoStage: Bool
  public let audioRefine: Bool
  public let condFps: Float?          // nil = model default fps
  public let imgCompression: Int
  public let deliveryShortEdge: Int
  public let refineScale: Float
  public let sampler: String
  public let stgScale: Float
  public let stgBlocks: String
  public let faceAnchorStrength: Float
  public let icControl: Bool
  public let icRefStrength: Float
  public let colorAnchor: Float
  public let nagScale: Float
  public let nagAlpha: Float
  public let nagTau: Float
  public let reanchorInterval: Int
  public let reanchorStrength: Float
  /// Temporal beat scheduling (comfybox#310) global kill switch. No
  /// request/preset override — `beat_schedule` presence/absence on the
  /// request is the per-render knob; this is only the emergency-off env.
  public let beatScheduleEnabled: Bool
  // Tier B
  public let plainDecodeMaxVol: Int
  public let refineMaxVol: Int
  public let decodeMode: String
  public let decodeTile: String
  public let upsamplerPath: String
  public let videoBitsPerPx: Double

  public let provenance: [String: LTX2ParamSource]
  public var params: [LTX2ResolvedParam]

  // Sampler-family helpers mirroring the legacy env conventions.
  public var samplerIsAncestral: Bool { sampler.lowercased().contains("ancestral") }
  public var samplerIsCfgPP: Bool { sampler.lowercased().contains("cfg_pp") }
  /// STG block subset; legacy default mid-stack triplet when unset.
  public var stgBlockSet: Set<Int> {
    let parsed = stgBlocks.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    return parsed.isEmpty ? [14, 15, 16] : Set(parsed)
  }
  /// Legacy semantics: sigma overrides need >= 2 entries to be meaningful.
  public var stage1SigmasOrNil: [Float]? { stage1Sigmas.count >= 2 ? stage1Sigmas : nil }
  public var refineSigmasEffective: [Float] {
    refineSigmas.count >= 2 ? refineSigmas : [0.85, 0.7250, 0.4219, 0.0]  // PinkCherry v1.5 pass-2
  }
  /// NAG enabled when scale > 0 (mirrors LTX2NAGConfig.fromEnvironment).
  public var nagConfig: LTX2NAGConfig? {
    nagScale > 0 ? LTX2NAGConfig(scale: nagScale, alpha: nagAlpha, tau: nagTau) : nil
  }

  /// Display string for one parameter's TYPED final value — used to overlay
  /// request/preset-level values onto the string readout rows.
  public func valueString(for name: String) -> String? {
    func fmt(_ v: Float) -> String { v == v.rounded() ? String(Int(v)) : String(v) }
    switch name {
    case "guidance_rescale": return fmt(guidanceRescale)
    case "cfg_schedule": return cfgSchedule.map(fmt).joined(separator: ",")
    case "stage1_sigmas": return stage1Sigmas.map(fmt).joined(separator: ",")
    case "refine_sigmas": return refineSigmas.map(fmt).joined(separator: ",")
    case "two_stage": return twoStage ? "true" : "false"
    case "audio_refine": return audioRefine ? "true" : "false"
    case "cond_fps": return condFps.map(fmt) ?? "model"
    case "img_compression": return String(imgCompression)
    case "delivery_short_edge": return String(deliveryShortEdge)
    case "refine_scale": return fmt(refineScale)
    case "sampler": return sampler
    case "stg_scale": return fmt(stgScale)
    case "stg_blocks": return stgBlocks
    case "face_anchor_strength": return fmt(faceAnchorStrength)
    case "ic_control": return icControl ? "true" : "false"
    case "ic_ref_strength": return fmt(icRefStrength)
    case "color_anchor": return fmt(colorAnchor)
    case "nag_scale": return fmt(nagScale)
    case "nag_alpha": return fmt(nagAlpha)
    case "nag_tau": return fmt(nagTau)
    case "reanchor_interval": return String(reanchorInterval)
    case "reanchor_strength": return fmt(reanchorStrength)
    case "beat_schedule_enabled": return beatScheduleEnabled ? "true" : "false"
    default: return nil
    }
  }
}

extension LTX2ConfigResolver {

  /// Build the authoritative config for one render. Base levels resolve via
  /// the string registry (configFile > env > builtin, with validation and
  /// fall-through); preset then request override per field, typed, no
  /// string round-trip.
  public static func resolveTyped(
    request: LTX2VideoTuning?,
    preset: LTX2VideoTuning?,
    environment: [String: String] = ProcessInfo.processInfo.environment,
    configFile: [String: String]? = nil,
    fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
  ) -> LTX2ResolvedVideoConfig {
    let base = resolveEffective(
      environment: environment, configFile: configFile, fileExists: fileExists)
    var byName: [String: LTX2ResolvedParam] = [:]
    for p in base { byName[p.name] = p }
    var provenance: [String: LTX2ParamSource] = [:]
    for p in base { provenance[p.name] = p.source }

    func str(_ name: String) -> String { byName[name]?.value ?? "" }
    func f(_ name: String) -> Float { Float(str(name)) ?? 0 }
    func i(_ name: String) -> Int { Int(str(name)) ?? 0 }
    func b(_ name: String) -> Bool { str(name) == "true" }
    func list(_ name: String) -> [Float] {
      str(name).split(separator: ",").compactMap { Float($0) }
    }

    /// request > preset for one field; records provenance on override.
    func pick<T>(_ name: String, _ base: T, _ presetV: T?, _ requestV: T?) -> T {
      if let requestV { provenance[name] = .request; return requestV }
      if let presetV { provenance[name] = .preset; return presetV }
      return base
    }

    let condFpsBase: Float? = str("cond_fps") == "model" ? nil : Float(str("cond_fps"))

    var config = LTX2ResolvedVideoConfig(
      guidanceRescale: pick("guidance_rescale", f("guidance_rescale"), preset?.guidanceRescale, request?.guidanceRescale),
      cfgSchedule: pick("cfg_schedule", list("cfg_schedule"), preset?.cfgSchedule, request?.cfgSchedule),
      stage1Sigmas: pick("stage1_sigmas", list("stage1_sigmas"), preset?.stage1Sigmas, request?.stage1Sigmas),
      refineSigmas: pick("refine_sigmas", list("refine_sigmas"), preset?.refineSigmas, request?.refineSigmas),
      twoStage: pick("two_stage", b("two_stage"), preset?.twoStage, request?.twoStage),
      audioRefine: pick("audio_refine", b("audio_refine"), preset?.audioRefine, request?.audioRefine),
      condFps: pick("cond_fps", condFpsBase, preset?.condFps, request?.condFps),
      imgCompression: pick("img_compression", i("img_compression"), preset?.imgCompression, request?.imgCompression),
      deliveryShortEdge: pick("delivery_short_edge", i("delivery_short_edge"), preset?.deliveryShortEdge, request?.deliveryShortEdge),
      refineScale: pick("refine_scale", f("refine_scale"), preset?.refineScale, request?.refineScale),
      sampler: pick("sampler", str("sampler"), preset?.sampler, request?.sampler),
      stgScale: pick("stg_scale", f("stg_scale"), preset?.stgScale, request?.stgScale),
      stgBlocks: pick("stg_blocks", str("stg_blocks"), preset?.stgBlocks, request?.stgBlocks),
      faceAnchorStrength: pick("face_anchor_strength", f("face_anchor_strength"), preset?.faceAnchorStrength, request?.faceAnchorStrength),
      icControl: pick("ic_control", b("ic_control"), preset?.icControl, request?.icControl),
      icRefStrength: pick("ic_ref_strength", f("ic_ref_strength"), preset?.icRefStrength, request?.icRefStrength),
      colorAnchor: pick("color_anchor", f("color_anchor"), preset?.colorAnchor, request?.colorAnchor),
      nagScale: pick("nag_scale", f("nag_scale"), preset?.nagScale, request?.nagScale),
      nagAlpha: pick("nag_alpha", f("nag_alpha"), preset?.nagAlpha, request?.nagAlpha),
      nagTau: pick("nag_tau", f("nag_tau"), preset?.nagTau, request?.nagTau),
      reanchorInterval: pick("reanchor_interval", i("reanchor_interval"), preset?.reanchorInterval, request?.reanchorInterval),
      reanchorStrength: pick("reanchor_strength", f("reanchor_strength"), preset?.reanchorStrength, request?.reanchorStrength),
      beatScheduleEnabled: b("beat_schedule_enabled"),
      plainDecodeMaxVol: i("plain_decode_max_vol"),
      refineMaxVol: i("refine_max_vol"),
      decodeMode: str("decode_mode"),
      decodeTile: str("decode_tile"),
      upsamplerPath: str("upsampler_path"),
      videoBitsPerPx: Double(str("video_bits_per_px")) ?? 0.5,
      provenance: provenance,
      params: base
    )
    // Overlay typed request/preset picks onto the string readout (2026-08-11):
    // `base` alone showed env values/provenance even when a request override
    // WON — the effective-config log lied, silently defeating the
    // "verify your override landed" discipline (found via the tarn1 sigma
    // A/B: renders were correct, the log said env).
    config.params = base.map { row in
      guard let src = provenance[row.name] else { return row }
      return LTX2ResolvedParam(
        name: row.name, envKey: row.envKey, tier: row.tier,
        value: config.valueString(for: row.name) ?? row.value,
        source: src, valid: row.valid, note: row.note)
    }
    return config
  }
}
