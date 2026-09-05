// CheckpointFamilyResolver.swift — comfybox#359
//
// The server's `checkpoint_family` field is not a broad family name: it is
// one of exactly five client policy labels (`PresetStore.checkpointFamilies`)
// — `turbo` | `raw-accel` | `raw-stock` for Krea-2, `zimage-turbo` |
// `zimage-base` for Z-Image — and `PresetStore.validate` REJECTS anything
// else. `GET /v1/model/family` only answers the broad, engine-detectable
// half (family + physical/declared variant); the accel-vs-stock split within
// Krea-2 "raw" depends on whether THIS preset's own `loras[]` declares an
// accelerator (`role: "accel"`), which only the preset document has. This is
// the one place the two are combined — pure, so the mapping is testable
// without a live engine.

import Foundation

public enum CheckpointFamilyResolver {

    /// - Parameters:
    ///   - family: `ModelFamilyInfo.family` — `"krea2"` | `"z-image"` | nil.
    ///   - variant: `ModelFamilyInfo.variant` — krea2 `"turbo"` | `"raw"`;
    ///     z-image `"turbo"` | `"base"`; nil.
    ///   - hasAccelLora: does this PRESET'S OWN `loras[]` declare a LoRA with
    ///     `role == "accel"`? Only meaningful for krea2 `"raw"` — Turbo and
    ///     Z-Image checkpoints have no accel/stock split.
    /// - Returns: one of `PresetStore.checkpointFamilies`, or nil when the
    ///   engine could not classify the spec at all (nothing to write).
    public static func resolve(family: String?, variant: String?, hasAccelLora: Bool) -> String? {
        switch (family, variant) {
        case ("krea2", "turbo"):
            return "turbo"
        case ("krea2", "raw"):
            return hasAccelLora ? "raw-accel" : "raw-stock"
        case ("z-image", "turbo"):
            return "zimage-turbo"
        case ("z-image", "base"):
            return "zimage-base"
        default:
            // Includes a recognized family with an undetermined variant
            // (e.g. a declared krea2 alias whose directory could not be
            // read) — never guess between turbo/raw or turbo/base.
            return nil
        }
    }

    /// Convenience over a preset's own `loras[]` — the one caller-supplied
    /// fact this resolver needs beyond the engine's answer.
    public static func resolve(family: String?, variant: String?, loras: [ServerPresetLora]) -> String? {
        resolve(family: family, variant: variant, hasAccelLora: loras.contains { $0.role == "accel" })
    }
}
