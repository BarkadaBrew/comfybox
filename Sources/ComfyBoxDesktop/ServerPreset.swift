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

/// Kroma policy (engine `KromaPolicy`, WP-E20 / D14): a first-class field,
/// never a `loras[]` entry. Passthrough — the engine refuses a krea2 image
/// preset that lacks it, so dropping it here would turn every desktop edit
/// of such a preset into a 400.
public struct ServerPresetKroma: Codable, Sendable, Equatable {
    public var strength: Double
    public var file: String?

    public init(strength: Double, file: String? = nil) {
        self.strength = strength
        self.file = file
    }
}

/// Second-stage recipe (engine `PresetStage`, WP-E20 / D4). Passthrough.
public struct ServerPresetStage: Codable, Sendable, Equatable {
    public var sampler: String?
    public var sigmaSchedule: String?
    public var steps: Int?
    public var denoise: Double?
    public var eta: Double?
    public var bongmath: Bool?
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

    // WP-E9/WP-E20 recipe + policy fields (passthrough, not edited in the UI
    // yet). Upsert REPLACES the stored document, so every engine field must
    // round-trip here or a desktop save silently erases it (FDD §7.3).
    public var vae: String?
    public var checkpointFamily: String?
    public var kroma: ServerPresetKroma?
    public var sampler: String?
    public var sigmaSchedule: String?
    public var shift: Double?
    public var eta: Double?
    public var bongmath: Bool?
    public var stage2: ServerPresetStage?

    // Read-only validity flag the engine attaches on GET (WP-E20, AC-44c).
    // Never sent back: the engine recomputes it on every save.
    public var invalid: Bool?
    public var invalidReason: String?

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
        upscale: ServerPresetUpscale? = nil,
        vae: String? = nil,
        checkpointFamily: String? = nil,
        kroma: ServerPresetKroma? = nil,
        sampler: String? = nil,
        sigmaSchedule: String? = nil,
        shift: Double? = nil,
        eta: Double? = nil,
        bongmath: Bool? = nil,
        stage2: ServerPresetStage? = nil
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
        self.vae = vae
        self.checkpointFamily = checkpointFamily
        self.kroma = kroma
        self.sampler = sampler
        self.sigmaSchedule = sigmaSchedule
        self.shift = shift
        self.eta = eta
        self.bongmath = bongmath
        self.stage2 = stage2
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, description
        case mediaKind, provider, engine, mode
        case model, customModelPath, baseModel
        case prompt, negativePrompt, promptPrefix, promptSuffix, injectedKeywords
        case steps, guidance, seed, width, height
        case loras, scheduler, upscale
        case vae, checkpointFamily, kroma, sampler, sigmaSchedule, shift, eta, bongmath, stage2
        case invalid, invalidReason
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(description, forKey: .description)
        try c.encodeIfPresent(mediaKind, forKey: .mediaKind)
        try c.encodeIfPresent(provider, forKey: .provider)
        try c.encodeIfPresent(engine, forKey: .engine)
        try c.encodeIfPresent(mode, forKey: .mode)
        try c.encodeIfPresent(model, forKey: .model)
        try c.encodeIfPresent(customModelPath, forKey: .customModelPath)
        try c.encodeIfPresent(baseModel, forKey: .baseModel)
        try c.encodeIfPresent(prompt, forKey: .prompt)
        try c.encodeIfPresent(negativePrompt, forKey: .negativePrompt)
        try c.encodeIfPresent(promptPrefix, forKey: .promptPrefix)
        try c.encodeIfPresent(promptSuffix, forKey: .promptSuffix)
        try c.encodeIfPresent(injectedKeywords, forKey: .injectedKeywords)
        try c.encodeIfPresent(steps, forKey: .steps)
        try c.encodeIfPresent(guidance, forKey: .guidance)
        try c.encodeIfPresent(seed, forKey: .seed)
        try c.encodeIfPresent(width, forKey: .width)
        try c.encodeIfPresent(height, forKey: .height)
        try c.encode(loras, forKey: .loras)
        try c.encodeIfPresent(scheduler, forKey: .scheduler)
        try c.encodeIfPresent(upscale, forKey: .upscale)
        try c.encodeIfPresent(vae, forKey: .vae)
        try c.encodeIfPresent(checkpointFamily, forKey: .checkpointFamily)
        try c.encodeIfPresent(kroma, forKey: .kroma)
        try c.encodeIfPresent(sampler, forKey: .sampler)
        try c.encodeIfPresent(sigmaSchedule, forKey: .sigmaSchedule)
        try c.encodeIfPresent(shift, forKey: .shift)
        try c.encodeIfPresent(eta, forKey: .eta)
        try c.encodeIfPresent(bongmath, forKey: .bongmath)
        try c.encodeIfPresent(stage2, forKey: .stage2)
        // `invalid` / `invalidReason` are deliberately NOT encoded.
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
        vae = try c.decodeIfPresent(String.self, forKey: .vae)
        checkpointFamily = try c.decodeIfPresent(String.self, forKey: .checkpointFamily)
        kroma = try c.decodeIfPresent(ServerPresetKroma.self, forKey: .kroma)
        sampler = try c.decodeIfPresent(String.self, forKey: .sampler)
        sigmaSchedule = try c.decodeIfPresent(String.self, forKey: .sigmaSchedule)
        shift = try c.decodeIfPresent(Double.self, forKey: .shift)
        eta = try c.decodeIfPresent(Double.self, forKey: .eta)
        bongmath = try c.decodeIfPresent(Bool.self, forKey: .bongmath)
        stage2 = try c.decodeIfPresent(ServerPresetStage.self, forKey: .stage2)
        invalid = try c.decodeIfPresent(Bool.self, forKey: .invalid)
        invalidReason = try c.decodeIfPresent(String.self, forKey: .invalidReason)
    }

    /// Map to the local apply-to-Generate shape.
    public func toGenerationPreset() -> GenerationPreset {
        GenerationPreset(
            id: id,
            name: name,
            promptTemplate: prompt ?? "",
            negativePrompt: negativePrompt,
            modelId: customModelPath ?? model,
            loras: loras.map { PresetLoRA(id: $0.filename, filename: $0.filename, scale: Float($0.scale)) },
            steps: steps ?? 9,
            guidance: Float(guidance ?? 3.5),
            width: width ?? 1024,
            height: height ?? 1024,
            sampler: scheduler,
            seed: seed.map { UInt64(truncatingIfNeeded: $0) }
        )
    }
}
