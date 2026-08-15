// ComfyBoxOutputNaming.swift — human-readable gallery filenames.
//
// Todd 2026-08-11: "the image saved in the gallery [should] have the
// comfybox-model-tier_name instead of zimage-krea2-number". Default output
// names now say WHAT rendered them and for WHICH tier, with a sortable
// timestamp instead of an opaque UUID:
//
//   comfybox-kroma-v0.2-avocado-20260811-121530-a3f2.png
//
// The model segment comes from the ACTIVE pool spec (a directory spec uses
// its last path component), the tier from the request's content mode
// ("manual" when absent — desktop/API renders without a mode).

import Foundation

public enum ComfyBoxOutputNaming {

  /// Sources that carry no information worth a filename segment.
  private static let uninformativeSources: Set<String> = ["", "api", "manual"]

  /// Build a default gallery filename:
  /// comfybox-<model>[-<preset>]-<tier>[-<source>]-<stamp>-<salt>.<ext>
  /// The preset segment appears when a preset produced the render (the recipe
  /// name — tier→preset mappings drift, the filename shouldn't); the source
  /// segment only when it says something a segment hasn't (bree, desktop,
  /// winner-rerender — never bare "api"). `date` is injectable for tests;
  /// the 4-hex salt guards same-second collisions.
  public static func defaultFilename(
    modelSpec: String?,
    presetId: String? = nil,
    contentMode: String?,
    source: String? = nil,
    date: Date = Date(),
    ext: String = "png"
  ) -> String {
    let model = shortModelName(modelSpec)
    let tier = sanitize(contentMode ?? "").isEmpty ? "manual" : sanitize(contentMode!)
    var segments = ["comfybox", model]
    let preset = sanitize(presetId ?? "")
    if !preset.isEmpty { segments.append(preset) }
    segments.append(tier)
    let src = sanitize(source ?? "")
    if !uninformativeSources.contains(src) && src != tier { segments.append(src) }
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd-HHmmss"
    formatter.locale = Locale(identifier: "en_US_POSIX")
    formatter.timeZone = TimeZone.current
    segments.append(formatter.string(from: date))
    segments.append(String(format: "%04x", UInt16.random(in: 0...UInt16.max)))
    return segments.joined(separator: "-") + ".\(ext)"
  }

  /// "…/LocalModels/kroma-v0.2" → "kroma-v0.2"; "krea2" → "krea2";
  /// nil/empty → "model".
  static func shortModelName(_ spec: String?) -> String {
    guard let spec, !spec.isEmpty else { return "model" }
    let last = (spec as NSString).lastPathComponent
    let cleaned = sanitize(last)
    return cleaned.isEmpty ? "model" : cleaned
  }

  /// Lowercase; keep letters/digits/dot/dash; everything else → dash;
  /// collapse runs and trim edge dashes.
  static func sanitize(_ value: String) -> String {
    let mapped = value.lowercased().map { c -> Character in
      (c.isLetter || c.isNumber || c == "." || c == "-") ? c : "-"
    }
    var out = ""
    var lastWasDash = false
    for c in mapped {
      if c == "-" {
        if !lastWasDash && !out.isEmpty { out.append(c) }
        lastWasDash = true
      } else {
        out.append(c)
        lastWasDash = false
      }
    }
    while out.hasSuffix("-") { out.removeLast() }
    return out
  }
}
