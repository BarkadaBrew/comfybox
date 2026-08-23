// PresetManager.swift — Preset CRUD with JSON persistence
//
// Manages saved generation presets — named combinations of prompt
// template, model, LoRAs, steps, guidance, resolution, and sampler.
// Persists to ~/.comfybox/presets.json. Observable for SwiftUI binding.

import Foundation

/// Structured Kroma recipe value carried into the Generate tab. Kroma is not
/// a generic `loras[]` row in persisted presets, but a render still needs its
/// file + adjustable strength when the preset is applied interactively.
public struct PresetKroma: Codable, Sendable, Equatable {
    public var strength: Double
    public var file: String?

    public init(strength: Double, file: String? = nil) {
        self.strength = strength
        self.file = file
    }
}

/// A saved generation preset.
public struct GenerationPreset: Identifiable, Codable, Sendable {
    public var id: String
    public var name: String
    public var promptTemplate: String
    public var negativePrompt: String?
    public var modelId: String?
    public var loras: [PresetLoRA]
    public var kroma: PresetKroma?
    public var steps: Int
    public var guidance: Float
    public var width: Int
    public var height: Int
    public var sampler: String?
    /// Fixed seed to reproduce the preset (nil/0 = random).
    public var seed: UInt64?
    public var createdAt: Date
    public var modifiedAt: Date

    public init(
        id: String = UUID().uuidString,
        name: String,
        promptTemplate: String = "",
        negativePrompt: String? = nil,
        modelId: String? = nil,
        loras: [PresetLoRA] = [],
        kroma: PresetKroma? = nil,
        steps: Int = 9,
        guidance: Float = 3.5,
        width: Int = 1024,
        height: Int = 1024,
        sampler: String? = nil,
        seed: UInt64? = nil,
        createdAt: Date = Date(),
        modifiedAt: Date = Date()
    ) {
        self.seed = seed
        self.id = id
        self.name = name
        self.promptTemplate = promptTemplate
        self.negativePrompt = negativePrompt
        self.modelId = modelId
        self.loras = loras
        self.kroma = kroma
        self.steps = steps
        self.guidance = guidance
        self.width = width
        self.height = height
        self.sampler = sampler
        self.createdAt = createdAt
        self.modifiedAt = modifiedAt
    }
}

/// A LoRA reference stored in a preset.
public struct PresetLoRA: Codable, Sendable, Identifiable {
    public var id: String
    public var filename: String
    public var scale: Float
    public var role: String?

    public init(id: String, filename: String, scale: Float = 1.0, role: String? = nil) {
        self.id = id
        self.filename = filename
        self.scale = scale
        self.role = role
    }
}

@Observable
public final class PresetManager {
    public var presets: [GenerationPreset] = []

    private static var storagePath: String {
        let dir = NSString(string: "~/.comfybox").expandingTildeInPath
        return (dir as NSString).appendingPathComponent("presets.json")
    }

    public init() {
        load()
    }

    // MARK: - CRUD

    /// Create a new preset from the current generation parameters.
    public func create(
        name: String,
        promptTemplate: String,
        modelId: String?,
        loras: [LoRASelection],
        steps: Int,
        guidance: Float,
        width: Int,
        height: Int,
        sampler: String? = nil
    ) -> GenerationPreset {
        let preset = GenerationPreset(
            name: name,
            promptTemplate: promptTemplate,
            modelId: modelId,
            loras: loras.map {
                PresetLoRA(id: $0.id, filename: $0.filename, scale: $0.scale, role: $0.role)
            },
            steps: steps,
            guidance: guidance,
            width: width,
            height: height,
            sampler: sampler
        )
        presets.append(preset)
        save()
        return preset
    }

    /// Update an existing preset.
    public func update(_ preset: GenerationPreset) {
        guard let index = presets.firstIndex(where: { $0.id == preset.id }) else { return }
        var updated = preset
        updated.modifiedAt = Date()
        presets[index] = updated
        save()
    }

    /// Delete a preset by ID.
    public func delete(id: String) {
        presets.removeAll { $0.id == id }
        save()
    }

    /// Duplicate a preset with a new name.
    public func duplicate(_ preset: GenerationPreset) -> GenerationPreset {
        var copy = preset
        copy.id = UUID().uuidString
        copy.name = "\(preset.name) (Copy)"
        copy.createdAt = Date()
        copy.modifiedAt = Date()
        presets.append(copy)
        save()
        return copy
    }

    // MARK: - Persistence

    private func load() {
        let path = Self.storagePath
        guard FileManager.default.fileExists(atPath: path),
              let data = FileManager.default.contents(atPath: path) else {
            return
        }
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            presets = try decoder.decode([GenerationPreset].self, from: data)
        } catch {
            // Corrupt file — start fresh.
            presets = []
        }
    }

    private func save() {
        let path = Self.storagePath
        let dir = (path as NSString).deletingLastPathComponent
        try? FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true
        )
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            encoder.dateEncodingStrategy = .iso8601
            let data = try encoder.encode(presets)
            try data.write(to: URL(fileURLWithPath: path))
        } catch {
            // Non-fatal — presets will reload from last good save.
        }
    }
}
