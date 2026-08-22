// Krea2LoRARelativity.swift — LoRA ↔ Krea-2 base relativity (WP-E6, FDD §3.6, AC-41).
//
// A Krea-2 LoRA is extracted against ONE physical base (`raw` or `turbo`)
// and is only meaningful on that base: kroma-lora-v0.3 carries 170 deltas
// authored against Turbo's norms; the rank-64 turbo LoRA IS (W_turbo − W_raw)
// and doubles Turbo's acceleration if applied on Turbo. The relativity is
// DECLARED, never inferred from tensor contents — the resolution order is
//
//   LoRAConfiguration.requiresBase   (the request)
//   ?? LoRALibraryEntry.krea2Relative (the library, user-editable)
//   ?? Krea2LoRARelativity.seeded     (the four files we know)
//
// and `nil` at the end declares nothing (the guard passes).

import Foundation

public enum Krea2LoRARelativity {

  /// Seeded relativities, keyed by filename stem (§3.6 — `kroma-v0.1` added
  /// in v2: it is what the three live `krea-film-*` presets carry).
  public static let seeded: [String: Krea2Variant] = [
    "krea2_turbo_lora_rank_64_bf16": .raw,
    "kroma-v0.2-base-lora-rank-384-fro-0985": .raw,
    "kroma-lora-v0.3": .turbo,
    "kroma-v0.1": .turbo,
  ]

  /// Exact stem match (case-insensitive, extension ignored). A substring is
  /// NOT a match — `kroma-lora-v0.3-extra` declares nothing.
  public static func seeded(forFilename filename: String) -> Krea2Variant? {
    let stem = (filename as NSString).lastPathComponent
    let bare = stem.lowercased().hasSuffix(".safetensors")
      ? String(stem.dropLast(".safetensors".count)) : stem
    let lower = bare.lowercased()
    return seeded.first { $0.key.lowercased() == lower }?.value
  }

  /// `config.requiresBase ?? seeded(forFilename:)`. The library layer
  /// (`LoRALibraryEntry.krea2Relative`) is folded into `requiresBase` by the
  /// server before the config reaches the pipeline, so this is the last
  /// word the pipeline needs.
  public static func required(for config: LoRAConfiguration, resolvedURL: URL) -> Krea2Variant? {
    config.requiresBase ?? seeded(forFilename: resolvedURL.lastPathComponent)
  }

  /// The guard. Pure, so AC-41 is testable without a checkpoint. Throws
  /// ``LoRAError/incompatibleBase(lora:requires:loaded:)`` when a declared
  /// relativity contradicts the loaded variant; `nil` passes.
  public static func check(lora: String, required: Krea2Variant?, loaded: Krea2Variant) throws {
    guard let required, required != loaded else { return }
    throw LoRAError.incompatibleBase(lora: lora, requires: required, loaded: loaded)
  }
}
