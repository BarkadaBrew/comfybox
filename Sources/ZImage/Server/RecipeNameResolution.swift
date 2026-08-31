// RecipeNameResolution.swift — fail-loud sampler / sigma-schedule name resolution
//
// WP-E4 (docs/FDD-krea2-raw-recipe.md §3.4, D22, D25). Replaces the
// `parseSchedulerKind` / `parseSigmaScheduleKind` coercions that turned every
// unrecognised name into euler / flow with no error and no log — the exact
// silent substitution the PRD traces a lost week to. A name either resolves
// (as an enum raw value or a DECLARED alias) or throws a `WarmServerError`
// that names the offending value and the valid set; the HTTP layer maps that
// to 400.
//
// The alias table is the single source of truth for three surfaces that used
// to drift independently: the /v1/generate decoder, the ComfyUI bridge's
// advertised `/object_info` option lists, and the MCP `generate_image` schema.

import Foundation

/// Resolved sampler / schedule names for one request: what will run AND what
/// was asked for, so the provenance record (WP-E10) can carry
/// `sigma_schedule: "flow", sigma_schedule_requested: "normal"` (D22).
struct ResolvedRecipeNames: Sendable, Equatable {
  /// nil == the request carried no sampler; the request builder applies its
  /// own default (euler). The resolver never smuggles a default in.
  let scheduler: SchedulerKind?
  /// The raw string the caller sent (after alias/prefix handling it may differ
  /// from `scheduler.rawValue`).
  let schedulerRequested: String?
  let sigmaSchedule: SigmaScheduleKind?
  let sigmaScheduleRequested: String?
}

enum RecipeNameResolver {

  // MARK: - Declared aliases (D22 — preserved in full)

  /// ComfyUI / RES4LYF sampler spellings → engine kinds. Raw enum values
  /// (`dpmpp-2m`, …) resolve as well; this table is the additional set.
  static let samplerAliases: [String: SchedulerKind] = [
    "res_2s": .res2s,
    "dpmpp_2m": .dpmplusplus2m,
    "dpmpp_2s_ancestral": .dpmplusplus2sa,
  ]

  /// ComfyUI schedule spellings → engine kinds. `normal` /
  /// `sgm_uniform` / `ddim_uniform` are KEPT as aliases of `.flow` (D22,
  /// reversed from FDD v1): Krita AI Diffusion's built-in styles send exactly
  /// these names by default, and our own `/object_info` advertises them.
  /// The alias is made visible rather than silent — `ResolvedRecipeNames`
  /// carries the requested string alongside the resolved kind.
  static let sigmaScheduleAliases: [String: SigmaScheduleKind] = [
    "beta57": .beta57,
    "normal": .flow,
    "sgm_uniform": .flow,
    "ddim_uniform": .flow,
  ]

  /// RES4LYF's sampler dropdown groups names under UI prefixes; stripping them
  /// lets a workflow value paste verbatim (`exponential/res_2s` → `res_2s`,
  /// `linear/ralston_3s` → `ralston_3s`).
  static let res4lyfPrefixes: [String] = ["exponential/", "multistep/", "linear/"]

  // MARK: - Valid sets (what the error message lists)

  /// Every sampler name the resolver accepts: enum raw values ∪ aliases.
  static var validSamplerNames: [String] {
    SchedulerKind.allCases.map(\.rawValue) + samplerAliases.keys.sorted()
  }

  /// Every sigma-schedule name the resolver accepts: enum raw values ∪ aliases.
  static var validSigmaScheduleNames: [String] {
    SigmaScheduleKind.allCases.map(\.rawValue) + sigmaScheduleAliases.keys.sorted()
  }

  // MARK: - Advertised lists (ComfyUI bridge `/object_info`, AC-17)

  /// The `sampler_name` option list: one ComfyUI-spelled entry per
  /// `SchedulerKind`, in enum order — the alias spelling when one is declared
  /// (`dpmpp_2m`, not `dpmpp-2m`), else the raw value. Generated, so a new
  /// kind is advertised the commit it lands and nothing phantom survives.
  static var advertisedSamplerNames: [String] {
    SchedulerKind.allCases.map { kind in
      samplerAliases.sorted { $0.key < $1.key }.first { $0.value == kind }?.key ?? kind.rawValue
    }
  }

  /// The `scheduler` option list: the ComfyUI names Krita's styles send first
  /// (all declared aliases), then every enum raw value not already covered —
  /// so the union equals `SigmaScheduleKind.allCases` ∪ the aliases exactly.
  static var advertisedSigmaScheduleNames: [String] {
    let comfyFirst = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]
    var seen = Set<String>()
    var out: [String] = []
    for name in comfyFirst + SigmaScheduleKind.allCases.map(\.rawValue) + sigmaScheduleAliases.keys.sorted() {
      if seen.insert(name).inserted { out.append(name) }
    }
    return out
  }

  // MARK: - Resolution

  /// Resolve a sampler name. `nil` in → `nil` out (absent is absent). An
  /// unknown name throws `WarmServerError.unknownSampler(name:valid:)`.
  static func resolveSchedulerKind(_ raw: String?) throws -> SchedulerKind? {
    guard let raw else { return nil }
    let name = stripPrefixes(raw)
    if let kind = samplerAliases[name] { return kind }
    if let kind = SchedulerKind(rawValue: name) { return kind }
    throw WarmServerError.unknownSampler(name: raw, valid: validSamplerNames)
  }

  /// Resolve a sigma-schedule name. `nil` in → `nil` out. An unknown name
  /// throws `WarmServerError.unknownSigmaSchedule(name:valid:)`.
  static func resolveSigmaScheduleKind(_ raw: String?) throws -> SigmaScheduleKind? {
    guard let raw else { return nil }
    if let kind = sigmaScheduleAliases[raw] { return kind }
    if let kind = SigmaScheduleKind(rawValue: raw) { return kind }
    throw WarmServerError.unknownSigmaSchedule(name: raw, valid: validSigmaScheduleNames)
  }

  /// Resolve both names of a request at once.
  static func resolve(scheduler: String?, sigmaSchedule: String?) throws -> ResolvedRecipeNames {
    ResolvedRecipeNames(
      scheduler: try resolveSchedulerKind(scheduler),
      schedulerRequested: scheduler,
      sigmaSchedule: try resolveSigmaScheduleKind(sigmaSchedule),
      sigmaScheduleRequested: sigmaSchedule)
  }

  private static func stripPrefixes(_ raw: String) -> String {
    for prefix in res4lyfPrefixes where raw.hasPrefix(prefix) {
      return String(raw.dropFirst(prefix.count))
    }
    return raw
  }
}
