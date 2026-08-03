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
    case bool               // env convention: "1" = true
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
    Entry(name: "two_stage", envKey: "LTX2_TWO_STAGE", tier: "A", kind: .bool, builtin: "false"),
    Entry(name: "cond_fps", envKey: "LTX2_COND_FPS", tier: "A", kind: .float(1...120), builtin: "model"),
    Entry(name: "img_compression", envKey: "LTX2_I2V_COMPRESSION", tier: "A", kind: .int(0...100), builtin: "35"),
    Entry(name: "sampler", envKey: "LTX2_SAMPLER", tier: "A", kind: .string, builtin: ""),
    Entry(name: "stg_scale", envKey: "LTX2_STG_SCALE", tier: "A", kind: .float(0...20), builtin: "0"),
    Entry(name: "stg_blocks", envKey: "LTX2_STG_BLOCKS", tier: "A", kind: .string, builtin: ""),
    Entry(name: "face_anchor_strength", envKey: "LTX2_FACE_ANCHOR_STRENGTH", tier: "A", kind: .float(0...1), builtin: "0.5"),
    Entry(name: "ic_control", envKey: "LTX2_IC_CONTROL", tier: "A", kind: .bool, builtin: "true"),
    Entry(name: "color_anchor", envKey: "LTX2_COLOR_ANCHOR", tier: "A", kind: .float(0...1), builtin: "0"),
    Entry(name: "nag_scale", envKey: "LTX2_NAG_SCALE", tier: "A", kind: .float(0...50), builtin: "0"),
    Entry(name: "nag_alpha", envKey: "LTX2_NAG_ALPHA", tier: "A", kind: .float(0...1), builtin: "0.25"),
    Entry(name: "nag_tau", envKey: "LTX2_NAG_TAU", tier: "A", kind: .float(1...10), builtin: "2.5"),
    Entry(name: "reanchor_interval", envKey: "LTX2_REANCHOR_INTERVAL", tier: "A", kind: .int(0...10_000), builtin: "0"),
    Entry(name: "reanchor_strength", envKey: "LTX2_REANCHOR_STRENGTH", tier: "A", kind: .float(0...1), builtin: "0"),
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

    for (raw, source) in candidates {
      guard let raw else { continue }
      let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
      guard !trimmed.isEmpty else { continue }

      switch validate(trimmed, kind: entry.kind, fileExists: fileExists) {
      case .ok(let canonical):
        return LTX2ResolvedParam(
          name: entry.name, envKey: entry.envKey, tier: entry.tier,
          value: canonical, source: source, valid: true, note: nil)
      case .keepButFlag(let canonical, let note):
        // e.g. a path that doesn't exist: it IS what the render will try to
        // use, so report it as the effective value — flagged.
        return LTX2ResolvedParam(
          name: entry.name, envKey: entry.envKey, tier: entry.tier,
          value: canonical, source: source, valid: false, note: note)
      case .reject(let note):
        // Loud fallback (spec §4 validation): never silently use garbage.
        return LTX2ResolvedParam(
          name: entry.name, envKey: entry.envKey, tier: entry.tier,
          value: entry.builtin, source: .builtin, valid: false,
          note: "\(source.rawValue) value '\(trimmed)' rejected: \(note)")
      }
    }

    return LTX2ResolvedParam(
      name: entry.name, envKey: entry.envKey, tier: entry.tier,
      value: entry.builtin, source: .builtin, valid: true, note: nil)
  }

  private enum Validation {
    case ok(String)
    case keepButFlag(String, String)
    case reject(String)
  }

  private static func validate(_ raw: String, kind: Kind, fileExists: (String) -> Bool) -> Validation {
    switch kind {
    case .float(let range):
      guard let v = Double(raw), v.isFinite else { return .reject("not a finite number") }
      guard range.contains(v) else { return .reject("outside range \(range)") }
      return .ok(canonicalNumber(v))
    case .int(let range):
      guard let v = Int(raw) else { return .reject("not an integer") }
      guard range.contains(v) else { return .reject("outside range \(range)") }
      return .ok(String(v))
    case .bool:
      return .ok(raw == "1" || raw.lowercased() == "true" ? "true" : "false")
    case .string:
      return .ok(raw)
    case .floatList:
      let parts = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
      for p in parts where Double(p)?.isFinite != true {
        return .reject("'\(p)' is not a finite number")
      }
      return .ok(parts.joined(separator: ","))
    case .path:
      guard fileExists(raw) else { return .keepButFlag(raw, "path does not exist") }
      return .ok(raw)
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
