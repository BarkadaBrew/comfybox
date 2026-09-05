// AcceleratorLoRAHeuristicTests.swift — comfybox#359 (fix round 1)

import Testing
@testable import ComfyBoxDesktop

@Suite("AcceleratorLoRAHeuristic")
struct AcceleratorLoRAHeuristicTests {

    @Test("every marker the controller named is recognized, case-insensitively")
    func markersMatch() {
        let names = [
            "krea2_turbo_distill_r256.safetensors",
            "Krea2-TURBO.safetensors",
            "some_lightning_8step.safetensors",
            "accel-lora.safetensors",
            "dmd2_4step.safetensors",
            "Hyper-SD-8steps.safetensors",
            "lcm_lora_v1.safetensors",
            // Round 2 additions.
            "krea2_8step.safetensors",
            "someLora_4Step_v2.safetensors",
            "sdxl-flash-lora.safetensors",
            "TCD_adapter.safetensors",
            "pcm_deterministic_4s.safetensors",
            "generic_step_reducer.safetensors",
        ]
        for name in names {
            #expect(AcceleratorLoRAHeuristic.looksLikeAccelerator(filename: name), "\(name)")
        }
    }

    @Test("an ordinary style LoRA is not an accelerator")
    func styleLorasDoNotMatch() {
        for name in ["cutifier_krea2.safetensors", "autocord_75.safetensors", "kroma-v0.2.safetensors"] {
            #expect(!AcceleratorLoRAHeuristic.looksLikeAccelerator(filename: name), "\(name)")
        }
    }

    @Test("matching reads the last path component, so a full path behaves like a bare name")
    func matchesOnLastPathComponent() {
        #expect(AcceleratorLoRAHeuristic.looksLikeAccelerator(
            filename: "/Users/todd/LoRAs/krea2_turbo_distill_r256.safetensors"))
        // The DIRECTORY naming an accelerator must not make a style LoRA one.
        #expect(!AcceleratorLoRAHeuristic.looksLikeAccelerator(
            filename: "/Users/todd/turbo-loras/cutifier_krea2.safetensors"))
    }

    @Test("an empty filename is never an accelerator")
    func emptyIsNotAnAccelerator() {
        #expect(!AcceleratorLoRAHeuristic.looksLikeAccelerator(filename: ""))
        #expect(!AcceleratorLoRAHeuristic.looksLikeAccelerator(filename: "   "))
    }

    @Test("a declared role is only declared when it is non-empty")
    func roleDeclaration() {
        #expect(!AcceleratorLoRAHeuristic.hasDeclaredRole(nil))
        #expect(!AcceleratorLoRAHeuristic.hasDeclaredRole(""))
        #expect(!AcceleratorLoRAHeuristic.hasDeclaredRole("   "))
        #expect(AcceleratorLoRAHeuristic.hasDeclaredRole("accel"))
        #expect(AcceleratorLoRAHeuristic.hasDeclaredRole("style"))
    }

    @Test("the real shape of the 26 presets: an accelerator with no role is flagged")
    func unroledAcceleratorIsFlagged() {
        let loras = [
            ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: nil),
            ServerPresetLora(filename: "cutifier_krea2.safetensors", scale: 0.8, role: nil),
        ]
        #expect(AcceleratorLoRAHeuristic.unroledAcceleratorCandidates(loras)
            == ["krea2_turbo_distill_r256.safetensors"])
    }

    @Test("once the role is declared — accel OR anything else — nothing is flagged")
    func declaredRoleClearsTheFlag() {
        for role in ["accel", "style"] {
            let loras = [ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: role)]
            #expect(AcceleratorLoRAHeuristic.unroledAcceleratorCandidates(loras).isEmpty, "role \(role)")
        }
    }

    @Test("a preset with no accelerator-looking LoRA at all is not flagged")
    func noCandidates() {
        let loras = [ServerPresetLora(filename: "cutifier_krea2.safetensors", scale: 0.8, role: nil)]
        #expect(AcceleratorLoRAHeuristic.unroledAcceleratorCandidates(loras).isEmpty)
    }
}
