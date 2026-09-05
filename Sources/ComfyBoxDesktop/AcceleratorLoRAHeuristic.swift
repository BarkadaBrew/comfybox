// AcceleratorLoRAHeuristic.swift — comfybox#359 (fix round 1)
//
// `checkpoint_family` splits Krea-2 "raw" into `raw-accel` vs `raw-stock`
// purely on whether the preset's own `loras[]` declares an accelerator
// (`role: "accel"`). For the 26 desktop-saved presets the roles are all
// `null`, so a naive backfill would silently label an accelerated stack
// `raw-stock` — the wrong recipe under the right name, which is exactly the
// class of bug #286/#350 exist to prevent.
//
// So this NEVER decides a role. It only recognizes that a filename LOOKS
// like an accelerator, which makes the missing role a question for a human
// rather than something to guess at. A preset that trips it is reported as
// `needsReview` and nothing is written.
//
// Pure, no I/O — the whole point is that the policy lives in one testable
// place instead of inline in a view model.

import Foundation

public enum AcceleratorLoRAHeuristic {

    /// Substrings that, in a LoRA filename, mean "this is very likely a
    /// step-reduction / acceleration adapter". Case-insensitive.
    ///
    /// Ordered as the controller specified them; kept as a stored list so a
    /// future addition is one edit with one test, not a scattered `contains`
    /// chain.
    public static let markers: [String] = [
        "turbo", "distill", "lightning", "accel", "dmd", "hyper", "lcm",
    ]

    /// Does this filename look like an accelerator? Matches on the file's
    /// last path component, so a preset that names a LoRA by absolute path
    /// reads the same as one that names it bare.
    public static func looksLikeAccelerator(filename: String) -> Bool {
        let name = (filename as NSString).lastPathComponent.lowercased()
        guard !name.isEmpty else { return false }
        return markers.contains { name.contains($0) }
    }

    /// Is a declared role missing? Whitespace-only counts as missing — the
    /// server treats `""` as no role either.
    public static func hasDeclaredRole(_ role: String?) -> Bool {
        guard let role = role?.trimmingCharacters(in: .whitespacesAndNewlines) else { return false }
        return !role.isEmpty
    }

    /// The names of every LoRA that looks like an accelerator but declares no
    /// role. Non-empty ⇒ the accel/stock split cannot be decided without a
    /// human, so a backfill must stop.
    public static func unroledAcceleratorCandidates(_ loras: [ServerPresetLora]) -> [String] {
        loras
            .filter { !hasDeclaredRole($0.role) && looksLikeAccelerator(filename: $0.filename) }
            .map { ($0.filename as NSString).lastPathComponent }
    }
}
