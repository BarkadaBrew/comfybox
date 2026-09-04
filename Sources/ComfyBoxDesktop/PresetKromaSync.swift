// PresetKromaSync.swift — the single test for "is this the Kroma row" (#276)
//
// `ServerPreset.kroma` is the structured, editable declaration of the Kroma
// adapter (WP-E20, D14); `loras[]` is the flat, executable stack the engine
// actually applies. Before this file, five call sites across PresetView.swift
// and GenerationView.swift each hand-rolled their own test for "does this
// generic LoRA row duplicate the structured Kroma entry" — and they had
// already drifted: GenerationView's tests matched on `role == "kroma"` OR a
// filename match, but PresetView's editor (`ServerPresetEditor.init` and
// `buildPreset()`) matched on filename only. Whenever a preset's structured
// `kroma.file` is nil — legitimate for `kroma: {strength: 0}` (off) or for
// "use the engine's family-default file" — PresetView's filter reduces to
// "filename != nil", which is vacuously true, so a legacy/duplicated
// `role: "kroma"` row in `loras[]` was never stripped: it survived being
// opened and re-saved in the editor, reintroducing the exact dual
// representation #276 exists to remove.
//
// This is the fix: ONE pure predicate, used everywhere a generic LoRA list
// must exclude the structured Kroma entry, so the editor and the executor can
// never again disagree about what counts as "the Kroma row".
import Foundation

public enum PresetKromaSync {

  /// True when a generic LoRA entry is a mirror/duplicate of the structured
  /// `kroma` declaration — by declared role, OR by naming the pinned Kroma
  /// file. Role wins first because it is unambiguous even when the preset
  /// pins no file (the engine-default case): the entry still says what it is.
  public static func isKromaMirror(role: String?, filename: String, kromaFile: String?) -> Bool {
    if let role, role.lowercased() == "kroma" { return true }
    if let kromaFile, !kromaFile.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
       filename == kromaFile {
      return true
    }
    return false
  }

  /// `source` with every entry that mirrors the structured `kroma`
  /// declaration removed. The structured field is the only thing the UI ever
  /// WRITES for Kroma; this is the one place a flat, generic LoRA list gets
  /// DERIVED from it — never the other way around.
  public static func strippingKromaMirror<T>(
    from source: [T], kromaFile: String?, role: (T) -> String?, filename: (T) -> String
  ) -> [T] {
    source.filter { !isKromaMirror(role: role($0), filename: filename($0), kromaFile: kromaFile) }
  }
}
