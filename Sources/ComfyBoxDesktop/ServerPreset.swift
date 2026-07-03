// ServerPreset.swift — Desktop mirror of the server's ImagePreset
//
// The canonical preset store lives on the server (~/.comfybox/presets.json,
// /v1/presets) and is shared with Bree/Telegram. This mirrors EVERY field the
// server persists: upsert REPLACES the whole document, so a client that
// round-trips only the fields it edits would silently erase the rest
// (engine/mode/provider on the legacy image-service presets, prompt affixes…).

import Foundation

public struct ServerPresetLora: Codable, Sendable, Equatable, Identifiable {
    public var filename: String
    public var scale: Double

    public var id: String { filename }

    public init(filename: String, scale: Double = 1.0) {
        self.filename = filename
        self.scale = scale
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        filename = try c.decodeIfPresent(String.self, forKey: .filename) ?? ""
        scale = try c.decodeIfPresent(Double.self, forKey: .scale) ?? 1.0
    }
}

/// Upscale step attached to a preset (passthrough; not edited in the UI yet).
public struct ServerPresetUpscale: Codable, Sendable, Equatable {
    public var enabled: Bool
    public var mode: String?
    public var scale: Double?
}

public struct ServerPreset: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var description: String

    // Routing (legacy image-service fields — passthrough, not edited)
    public var mediaKind: String?
    public var provider: String?
    public var engine: String?
    public var mode: String?

    // Model
    public var model: String?
    public var customModelPath: String?
    public var baseModel: String?

    // Prompt shaping
    public var prompt: String?
    public var negativePrompt: String?
    public var promptPrefix: String?
    public var promptSuffix: String?
    public var injectedKeywords: [String]?

    // Parameters
    public var steps: Int?
    public var guidance: Double?
    public var seed: Int?
    public var width: Int?
    public var height: Int?

    public var loras: [ServerPresetLora]
    public var scheduler: String?
    public var upscale: ServerPresetUpscale?

    public init(
        id: String = UUID().uuidString,
        name: String,
        description: String = "",
        mediaKind: String? = nil,
        provider: String? = nil,
        engine: String? = nil,
        mode: String? = nil,
        model: String? = nil,
        customModelPath: String? = nil,
        baseModel: String? = nil,
        prompt: String? = nil,
        negativePrompt: String? = nil,
        promptPrefix: String? = nil,
        promptSuffix: String? = nil,
        injectedKeywords: [String]? = nil,
        steps: Int? = nil,
        guidance: Double? = nil,
        seed: Int? = nil,
        width: Int? = nil,
        height: Int? = nil,
        loras: [ServerPresetLora] = [],
        scheduler: String? = nil,
        upscale: ServerPresetUpscale? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.mediaKind = mediaKind
        self.provider = provider
        self.engine = engine
        self.mode = mode
        self.model = model
        self.customModelPath = customModelPath
        self.baseModel = baseModel
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.promptPrefix = promptPrefix
        self.promptSuffix = promptSuffix
        self.injectedKeywords = injectedKeywords
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.width = width
        self.height = height
        self.loras = loras
        self.scheduler = scheduler
        self.upscale = upscale
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        name = try c.decodeIfPresent(String.self, forKey: .name) ?? ""
        description = try c.decodeIfPresent(String.self, forKey: .description) ?? ""
        mediaKind = try c.decodeIfPresent(String.self, forKey: .mediaKind)
        provider = try c.decodeIfPresent(String.self, forKey: .provider)
        engine = try c.decodeIfPresent(String.self, forKey: .engine)
        mode = try c.decodeIfPresent(String.self, forKey: .mode)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        customModelPath = try c.decodeIfPresent(String.self, forKey: .customModelPath)
        baseModel = try c.decodeIfPresent(String.self, forKey: .baseModel)
        prompt = try c.decodeIfPresent(String.self, forKey: .prompt)
        negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
        promptPrefix = try c.decodeIfPresent(String.self, forKey: .promptPrefix)
        promptSuffix = try c.decodeIfPresent(String.self, forKey: .promptSuffix)
        injectedKeywords = try c.decodeIfPresent([String].self, forKey: .injectedKeywords)
        steps = try c.decodeIfPresent(Int.self, forKey: .steps)
        guidance = try c.decodeIfPresent(Double.self, forKey: .guidance)
        seed = try c.decodeIfPresent(Int.self, forKey: .seed)
        width = try c.decodeIfPresent(Int.self, forKey: .width)
        height = try c.decodeIfPresent(Int.self, forKey: .height)
        loras = try c.decodeIfPresent([ServerPresetLora].self, forKey: .loras) ?? []
        scheduler = try c.decodeIfPresent(String.self, forKey: .scheduler)
        upscale = try c.decodeIfPresent(ServerPresetUpscale.self, forKey: .upscale)
    }

    /// Map to the local apply-to-Generate shape.
    public func toGenerationPreset() -> GenerationPreset {
        GenerationPreset(
            id: id,
            name: name,
            promptTemplate: prompt ?? "",
            modelId: customModelPath ?? model,
            loras: loras.map { PresetLoRA(id: $0.filename, filename: $0.filename, scale: Float($0.scale)) },
            steps: steps ?? 9,
            guidance: Float(guidance ?? 3.5),
            width: width ?? 1024,
            height: height ?? 1024,
            sampler: scheduler
        )
    }
}
