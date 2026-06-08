// PresetManagerTests.swift — Tests for PresetManager and GenerationPreset

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("GenerationPreset")
struct GenerationPresetTests {
    @Test("default values")
    func defaults() {
        let preset = GenerationPreset(name: "Test")
        #expect(preset.name == "Test")
        #expect(preset.promptTemplate == "")
        #expect(preset.modelId == nil)
        #expect(preset.loras.isEmpty)
        #expect(preset.steps == 9)
        #expect(preset.guidance == 3.5)
        #expect(preset.width == 1024)
        #expect(preset.height == 1024)
        #expect(preset.sampler == nil)
    }

    @Test("codable round-trip")
    func codable() throws {
        let preset = TestData.makePreset(name: "My Preset", promptTemplate: "a test prompt", steps: 20, guidance: 7.5)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(preset)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(GenerationPreset.self, from: data)
        #expect(decoded.name == "My Preset")
        #expect(decoded.promptTemplate == "a test prompt")
        #expect(decoded.steps == 20)
        #expect(decoded.guidance == 7.5)
        #expect(decoded.id == preset.id)
    }

    @Test("identifiable via id")
    func identifiable() {
        let preset = GenerationPreset(id: "unique", name: "Test")
        #expect(preset.id == "unique")
    }

    @Test("sendable conformance")
    func sendable() {
        let preset = TestData.makePreset()
        let _: any Sendable = preset
        _ = preset
    }
}

@Suite("PresetLoRA")
struct PresetLoRATests {
    @Test("default scale is 1.0")
    func defaultScale() {
        let lora = PresetLoRA(id: "test", filename: "test.safetensors")
        #expect(lora.scale == 1.0)
    }

    @Test("codable round-trip")
    func codable() throws {
        let lora = PresetLoRA(id: "lora-1", filename: "style.safetensors", scale: 0.75)
        let data = try JSONEncoder().encode(lora)
        let decoded = try JSONDecoder().decode(PresetLoRA.self, from: data)
        #expect(decoded.id == "lora-1")
        #expect(decoded.filename == "style.safetensors")
        #expect(decoded.scale == 0.75)
    }

    @Test("identifiable via id")
    func identifiable() {
        let lora = PresetLoRA(id: "my-lora", filename: "test.safetensors")
        #expect(lora.id == "my-lora")
    }
}
