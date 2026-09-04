// PresetKromaSyncTests.swift — the single "is this the Kroma row" test (#276)

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("PresetKromaSync")
struct PresetKromaSyncTests {

    @Test("matches by declared role, regardless of filename")
    func matchesByRole() {
        #expect(PresetKromaSync.isKromaMirror(role: "kroma", filename: "anything.safetensors", kromaFile: nil))
        #expect(PresetKromaSync.isKromaMirror(role: "Kroma", filename: "anything.safetensors", kromaFile: "other.safetensors"))
    }

    @Test("matches by pinned filename when no role is declared")
    func matchesByFilename() {
        #expect(PresetKromaSync.isKromaMirror(role: nil, filename: "kroma-v0.1.safetensors", kromaFile: "kroma-v0.1.safetensors"))
    }

    @Test("an unrelated LoRA with no role and a different filename is not a mirror")
    func doesNotMatchUnrelatedLora() {
        #expect(!PresetKromaSync.isKromaMirror(role: nil, filename: "detail.safetensors", kromaFile: "kroma-v0.1.safetensors"))
        #expect(!PresetKromaSync.isKromaMirror(role: "accel", filename: "detail.safetensors", kromaFile: "kroma-v0.1.safetensors"))
    }

    /// #276 root cause: `kroma.file == nil` is the legitimate "off" declaration
    /// (`kroma: {strength: 0}`) or "use the engine-default file" state. A
    /// filename-only test — what `PresetView.swift`'s editor used before this
    /// fix — degrades to "filename != nil", which is vacuously true, so a
    /// legacy `role: "kroma"` duplicate in `loras[]` was never recognized as
    /// a mirror and survived edit-and-save. The role test alone must still
    /// catch it.
    @Test("catches a role-tagged duplicate even when kroma.file is nil (the bug #276 fixes)")
    func catchesDuplicateWhenKromaFileIsNil() {
        #expect(PresetKromaSync.isKromaMirror(role: "kroma", filename: "kroma-v0.1.safetensors", kromaFile: nil))
    }

    @Test("an empty kromaFile does not match every LoRA by empty-string coincidence")
    func emptyKromaFileDoesNotMatchEverything() {
        #expect(!PresetKromaSync.isKromaMirror(role: nil, filename: "detail.safetensors", kromaFile: ""))
    }

    @Test("strippingKromaMirror removes only the mirrored entry")
    func strippingRemovesOnlyTheMirror() {
        struct Row: Equatable { var filename: String; var scale: Double; var role: String? }
        let rows = [
            Row(filename: "kroma-v0.1.safetensors", scale: 0.6, role: "kroma"),
            Row(filename: "detail.safetensors", scale: 0.5, role: nil),
            Row(filename: "krea2_turbo.safetensors", scale: 1.0, role: "accel"),
        ]
        let stripped = PresetKromaSync.strippingKromaMirror(
            from: rows, kromaFile: nil, role: { $0.role }, filename: { $0.filename })
        #expect(stripped.map(\.filename) == ["detail.safetensors", "krea2_turbo.safetensors"])
    }

    @Test("strippingKromaMirror also strips a bare filename match with no role")
    func strippingByFilenameAlone() {
        struct Row: Equatable { var filename: String; var role: String? }
        let rows = [
            Row(filename: "kroma-v0.1.safetensors", role: nil),
            Row(filename: "detail.safetensors", role: nil),
        ]
        let stripped = PresetKromaSync.strippingKromaMirror(
            from: rows, kromaFile: "kroma-v0.1.safetensors", role: { $0.role }, filename: { $0.filename })
        #expect(stripped.map(\.filename) == ["detail.safetensors"])
    }
}
