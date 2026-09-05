// CheckpointFamilyResolverTests.swift — comfybox#359

import Testing
@testable import ComfyBoxDesktop

@Suite("CheckpointFamilyResolver")
struct CheckpointFamilyResolverTests {

    @Test("krea2 turbo variant maps to the turbo policy label regardless of accel")
    func krea2Turbo() {
        #expect(CheckpointFamilyResolver.resolve(family: "krea2", variant: "turbo", hasAccelLora: false) == "turbo")
        #expect(CheckpointFamilyResolver.resolve(family: "krea2", variant: "turbo", hasAccelLora: true) == "turbo")
    }

    @Test("krea2 raw variant splits on whether the preset declares an accel LoRA")
    func krea2RawSplitsOnAccel() {
        #expect(CheckpointFamilyResolver.resolve(family: "krea2", variant: "raw", hasAccelLora: true) == "raw-accel")
        #expect(CheckpointFamilyResolver.resolve(family: "krea2", variant: "raw", hasAccelLora: false) == "raw-stock")
    }

    @Test("z-image variants map to their zimage- labels")
    func zImageVariants() {
        #expect(CheckpointFamilyResolver.resolve(family: "z-image", variant: "turbo", hasAccelLora: false) == "zimage-turbo")
        #expect(CheckpointFamilyResolver.resolve(family: "z-image", variant: "base", hasAccelLora: false) == "zimage-base")
    }

    @Test("an unresolved family, an unresolved variant, or an unknown family is nil — never a guess")
    func unresolvedIsNil() {
        #expect(CheckpointFamilyResolver.resolve(family: nil, variant: nil, hasAccelLora: false) == nil)
        #expect(CheckpointFamilyResolver.resolve(family: "krea2", variant: nil, hasAccelLora: true) == nil)
        #expect(CheckpointFamilyResolver.resolve(family: "z-image", variant: nil, hasAccelLora: false) == nil)
        #expect(CheckpointFamilyResolver.resolve(family: "chroma", variant: "turbo", hasAccelLora: false) == nil)
    }

    @Test("the loras[] convenience reads role == accel, nothing else")
    func lorasConvenienceReadsAccelRole() {
        let withAccel = [
            ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: "accel"),
            ServerPresetLora(filename: "style.safetensors", scale: 0.6, role: nil),
        ]
        #expect(CheckpointFamilyResolver.resolve(family: "krea2", variant: "raw", loras: withAccel) == "raw-accel")

        let withoutAccel = [
            ServerPresetLora(filename: "krea2_turbo_distill_r256.safetensors", scale: 1.0, role: nil),
            ServerPresetLora(filename: "style.safetensors", scale: 0.6, role: "kroma"),
        ]
        #expect(CheckpointFamilyResolver.resolve(family: "krea2", variant: "raw", loras: withoutAccel) == "raw-stock")
    }
}
