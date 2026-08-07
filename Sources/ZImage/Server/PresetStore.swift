// PresetStore.swift — Generation presets, persisted to ~/.comfybox/presets.json.
//
// Straight port of the Coffee Shop image service's ImagePreset concept and preset
// resolution (see coffeeshop-image-service: src/types.ts `ImagePreset`,
// src/service.ts `resolveJobRequest`/`validatePreset`, src/config.ts `DEFAULT_PRESETS`).
//
// A preset is a reusable bundle of generation parameters (prompt bits, engine/model/mode,
// steps, guidance, dimensions, LoRAs, …) keyed by id/name/description. `resolve(id)` merges
// a preset onto system defaults to yield a fully-populated parameter set — the behavior the
// Node service exposed at `/v1/presets/resolve`.
//
// Persistence mirrors ``ComfyBoxServerConfig``: JSON under ~/.comfybox, tolerant decode
// (older/partial files load with defaults), atomic writes.

import Foundation

// MARK: - LoRA reference

/// One LoRA a preset applies, by filename + scale. Port of `LoraReference` (types.ts).
public struct LoraReference: Codable, Equatable, Sendable {
  public var filename: String
  public var scale: Double

  public init(filename: String, scale: Double) {
    self.filename = filename
    self.scale = scale
  }
}

// MARK: - Post-render upscale

/// Optional post-render upscale config. When enabled, callers auto-chain an upscale job
/// after the base render. Port of `ImagePreset.upscale` (types.ts).
public struct PresetUpscale: Codable, Equatable, Sendable {
  public var enabled: Bool
  public var mode: String?   // "seedvr2" | "controlnet"
  public var scale: Double?  // default 2

  public init(enabled: Bool, mode: String? = nil, scale: Double? = nil) {
    self.enabled = enabled
    self.mode = mode
    self.scale = scale
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
    mode = try c.decodeIfPresent(String.self, forKey: .mode)
    scale = try c.decodeIfPresent(Double.self, forKey: .scale)
  }
}

// MARK: - ImagePreset

/// A named, reusable set of generation parameters. Port of `ImagePreset` (types.ts).
///
/// Decoding is tolerant: only `id`/`name` are truly needed to round-trip; every other field
/// falls back to nil / empty so partial or older files load cleanly. Enum-like fields
/// (`mediaKind`, `provider`, `engine`, `mode`, `model`) are kept as free-form strings for the
/// same forward-compatibility reason the Node config used string unions.
public struct ImagePreset: Codable, Equatable, Sendable, Identifiable {
  public var id: String
  public var name: String
  public var description: String

  // Routing / engine selection.
  public var mediaKind: String?   // "image" | "video"
  public var provider: String?    // "local" | "replicate" | "auto"
  public var engine: String?      // "mflux" | "zimage"
  public var mode: String?        // MfluxExecutable (e.g. "z-image-turbo", "generate")
  public var model: String?       // SupportedModel or custom
  public var customModelPath: String?
  public var baseModel: String?

  // Prompt shaping.
  public var prompt: String?
  public var negativePrompt: String?
  public var promptPrefix: String?
  public var promptSuffix: String?
  public var injectedKeywords: [String]?

  // Numeric generation params.
  public var steps: Int?
  /// Tier A video tuning block (task #9 Phase 2) — preset-level overrides
  /// resolved between request fields and config.json/env.
  public var videoTuning: LTX2VideoTuning?
  public var guidance: Double?
  public var seed: Int?
  public var width: Int?
  public var height: Int?

  // Adapters + scheduler + post-processing.
  public var loras: [LoraReference]
  public var scheduler: String?
  public var upscale: PresetUpscale?

  public init(
    id: String,
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
    loras: [LoraReference] = [],
    scheduler: String? = nil,
    upscale: PresetUpscale? = nil
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

  private enum CodingKeys: String, CodingKey {
    case id, name, description
    case mediaKind, provider, engine, mode, model, customModelPath, baseModel
    case prompt, negativePrompt, promptPrefix, promptSuffix, injectedKeywords
    case steps, guidance, seed, width, height
    case loras, scheduler, upscale
    // Missing until 2026-08-07: with it absent, BOTH the custom decoder and
    // the synthesized encoder dropped videoTuning — every preset-level Tier-A
    // tuning write since task #9 Phase 2 silently vanished on the JSON/API
    // path. The desktop tuning UI was writing values nothing ever read.
    case videoTuning
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decodeIfPresent(String.self, forKey: .id) ?? ""
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
    // Tolerate a malformed `loras` value by treating it as empty rather than failing the decode.
    loras = ((try? c.decodeIfPresent([LoraReference].self, forKey: .loras)) ?? nil) ?? []
    scheduler = try c.decodeIfPresent(String.self, forKey: .scheduler)
    upscale = try c.decodeIfPresent(PresetUpscale.self, forKey: .upscale)
    videoTuning = try c.decodeIfPresent(LTX2VideoTuning.self, forKey: .videoTuning)
  }
}

// MARK: - Resolution defaults + result

/// System defaults a preset is merged onto in ``PresetStore/resolve(_:)``.
///
/// Mirrors the fallback ladder in the Node `resolveJobRequest`: steps→4, width/height→512,
/// provider→"local", engine→"mflux", mediaKind→"image".
public struct PresetDefaults: Equatable, Sendable {
  public var mediaKind: String
  public var provider: String
  public var engine: String
  public var steps: Int
  public var width: Int
  public var height: Int
  public var guidance: Double?

  public init(
    mediaKind: String = "image",
    provider: String = "local",
    engine: String = "mflux",
    steps: Int = 4,
    width: Int = 512,
    height: Int = 512,
    guidance: Double? = nil
  ) {
    self.mediaKind = mediaKind
    self.provider = provider
    self.engine = engine
    self.steps = steps
    self.width = width
    self.height = height
    self.guidance = guidance
  }

  public static let standard = PresetDefaults()
}

/// A fully-resolved parameter set: a preset merged onto ``PresetDefaults``. Every routing and
/// numeric field callers need to launch a render is populated (nil only where genuinely
/// optional, e.g. `seed`, `model`, `negativePrompt`).
public struct ResolvedPreset: Codable, Equatable, Sendable {
  public var id: String
  public var name: String
  public var description: String

  public var mediaKind: String
  public var provider: String
  public var engine: String
  public var mode: String?
  public var model: String?
  public var customModelPath: String?
  public var baseModel: String?

  public var prompt: String?
  public var negativePrompt: String?
  public var promptPrefix: String?
  public var promptSuffix: String?
  public var injectedKeywords: [String]

  public var steps: Int
  public var guidance: Double?
  public var seed: Int?
  public var width: Int
  public var height: Int

  public var loras: [LoraReference]
  public var scheduler: String?
  public var upscale: PresetUpscale?

  public init(preset: ImagePreset, defaults: PresetDefaults = .standard) {
    id = preset.id
    name = preset.name
    description = preset.description
    mediaKind = preset.mediaKind ?? defaults.mediaKind
    provider = preset.provider ?? defaults.provider
    engine = preset.engine ?? defaults.engine
    mode = preset.mode
    model = preset.model
    customModelPath = preset.customModelPath
    baseModel = preset.baseModel
    prompt = preset.prompt
    negativePrompt = preset.negativePrompt
    promptPrefix = preset.promptPrefix
    promptSuffix = preset.promptSuffix
    injectedKeywords = preset.injectedKeywords ?? []
    steps = preset.steps ?? defaults.steps
    guidance = preset.guidance ?? defaults.guidance
    seed = preset.seed
    width = preset.width ?? defaults.width
    height = preset.height ?? defaults.height
    loras = preset.loras
    scheduler = preset.scheduler
    upscale = preset.upscale
  }
}

// MARK: - Errors

public enum PresetStoreError: Error, Equatable, CustomStringConvertible {
  case validation(String)
  case notFound(String)

  public var description: String {
    switch self {
    case .validation(let m): return "Preset validation failed: \(m)"
    case .notFound(let id): return "Preset not found: \(id)"
    }
  }
}

// MARK: - PresetStore

/// Persists generation presets to `~/.comfybox/presets.json` and resolves them against
/// system defaults. Thread-safe: all access is guarded by an internal lock.
public final class PresetStore: @unchecked Sendable {

  private let path: URL
  private let fileManager: FileManager
  private let defaults: PresetDefaults
  private let lock = NSLock()
  private var presets: [ImagePreset]

  /// On-disk envelope. Wrapping the array in an object leaves room for schema growth
  /// (e.g. a future `version` or `presetMap`) without breaking older readers.
  private struct PresetFile: Codable {
    var presets: [ImagePreset]
  }

  /// `~/.comfybox/presets.json`.
  public static func defaultPath() -> URL {
    ComfyBoxServerConfig.homeDirectory().appendingPathComponent(".comfybox/presets.json")
  }

  /// Seed presets written on first run (file absent). Port of `DEFAULT_PRESETS` (config.ts).
  public static let defaultPresets: [ImagePreset] = [
    ImagePreset(
      id: "zimage-chat",
      name: "Z-Image Chat",
      description: "Fast chat lane preset",
      mediaKind: "image",
      provider: "local",
      engine: "zimage",
      mode: "z-image-turbo",
      model: "z-image-turbo",
      steps: 8,
      guidance: 1,
      width: 512,
      height: 512,
      loras: []
    ),
    ImagePreset(
      id: "schnell-hq",
      name: "Schnell HQ",
      description: "Higher-quality mflux preset",
      mediaKind: "image",
      provider: "local",
      engine: "mflux",
      mode: "generate",
      model: "schnell",
      steps: 4,
      guidance: 3.5,
      width: 1024,
      height: 1024,
      loras: []
    ),
  ]

  /// Load presets from `path`. If the file is absent, seed ``defaultPresets`` and persist them.
  /// A malformed/partial file loads tolerantly (recoverable entries survive; the rest default).
  public init(
    path: URL = PresetStore.defaultPath(),
    defaults: PresetDefaults = .standard,
    seedDefaults: Bool = true,
    fileManager: FileManager = .default
  ) {
    self.path = path
    self.defaults = defaults
    self.fileManager = fileManager

    if fileManager.fileExists(atPath: path.path), let data = try? Data(contentsOf: path) {
      self.presets = PresetStore.decode(data)
    } else if seedDefaults {
      self.presets = PresetStore.defaultPresets
      try? PresetStore.persist(self.presets, to: path, fileManager: fileManager)
    } else {
      self.presets = []
    }
  }

  // MARK: Reads

  /// All presets, in insertion order.
  public func list() -> [ImagePreset] {
    lock.lock(); defer { lock.unlock() }
    return presets
  }

  /// The preset with `id`, or nil.
  public func get(_ id: String) -> ImagePreset? {
    lock.lock(); defer { lock.unlock() }
    return presets.first { $0.id == id }
  }

  // MARK: Writes

  /// Insert `preset`, or replace the existing one with the same id. Validates first; on success
  /// the store is persisted atomically. Returns the (validated) stored preset.
  @discardableResult
  public func upsert(_ preset: ImagePreset) throws -> ImagePreset {
    let validated = try PresetStore.validate(preset)
    lock.lock(); defer { lock.unlock() }
    if let idx = presets.firstIndex(where: { $0.id == validated.id }) {
      presets[idx] = validated
    } else {
      presets.append(validated)
    }
    try PresetStore.persist(presets, to: path, fileManager: fileManager)
    return validated
  }

  /// Delete the preset with `id`. Returns true if one was removed. Persists on change.
  @discardableResult
  public func delete(_ id: String) throws -> Bool {
    lock.lock(); defer { lock.unlock() }
    let before = presets.count
    presets.removeAll { $0.id == id }
    let changed = presets.count != before
    if changed {
      try PresetStore.persist(presets, to: path, fileManager: fileManager)
    }
    return changed
  }

  // MARK: Legacy import

  /// Default location of the old Coffee Shop image-service presets (one
  /// JSON file per preset).
  public static func legacyImageServiceDirectory() -> URL {
    URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
      .appendingPathComponent(".coffeeshop/image-service/presets", isDirectory: true)
  }

  /// One legacy image-service preset file (per-file JSON shape).
  private struct LegacyPreset: Decodable {
    struct Lora: Decodable { let path: String?; let scale: Double? }
    let id: String?
    let name: String?
    let description: String?
    let model: String?
    let steps: Int?
    let guidance: Double?
    let width: Int?
    let height: Int?
    let loras: [Lora]?
    let injectedKeywords: String?   // legacy stored a single comma string
    let negativePrompt: String?
  }

  /// Import presets from the old image-service (one JSON per file), merging
  /// idempotently. Legacy ids are prefixed `imported-` so a built-in preset
  /// of the same name is never clobbered and a re-run is a no-op. Returns the
  /// number newly added.
  @discardableResult
  public func importLegacyImageService(
    from directory: URL = PresetStore.legacyImageServiceDirectory()
  ) -> Int {
    guard let files = try? fileManager.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: nil)
    else { return 0 }

    var added = 0
    for file in files where file.pathExtension == "json" {
      guard let data = try? Data(contentsOf: file),
            let legacy = try? JSONDecoder().decode(LegacyPreset.self, from: data),
            let legacyId = legacy.id ?? file.deletingPathExtension().lastPathComponent as String?
      else { continue }

      let importedId = "imported-\(legacyId)"
      if get(importedId) != nil { continue }  // already imported

      let keywords = (legacy.injectedKeywords ?? "")
        .split(separator: ",")
        .map { $0.trimmingCharacters(in: .whitespaces) }
        .filter { !$0.isEmpty }
      let negative = (legacy.negativePrompt?.isEmpty == false) ? legacy.negativePrompt : nil
      let loras = (legacy.loras ?? []).compactMap { lora -> LoraReference? in
        guard let path = lora.path, !path.isEmpty else { return nil }
        return LoraReference(filename: path, scale: lora.scale ?? 1.0)
      }

      let preset = ImagePreset(
        id: importedId,
        name: legacy.name ?? legacyId,
        description: legacy.description ?? "",
        mediaKind: "image",
        provider: "local",
        engine: "zimage",
        model: legacy.model,
        negativePrompt: negative,
        injectedKeywords: keywords.isEmpty ? nil : keywords,
        steps: legacy.steps,
        guidance: legacy.guidance,
        width: legacy.width,
        height: legacy.height,
        loras: loras
      )
      if (try? upsert(preset)) != nil { added += 1 }
    }
    return added
  }

  // MARK: Resolve

  /// Merge the preset `id` onto the store's ``PresetDefaults`` and return the fully-populated
  /// parameter set. Port of the `/v1/presets/resolve` behavior. Throws ``PresetStoreError/notFound(_:)``.
  public func resolve(_ id: String) throws -> ResolvedPreset {
    guard let preset = get(id) else { throw PresetStoreError.notFound(id) }
    return ResolvedPreset(preset: preset, defaults: defaults)
  }

  /// Resolve an in-hand preset against the store's defaults without a lookup.
  public func resolve(preset: ImagePreset) -> ResolvedPreset {
    ResolvedPreset(preset: preset, defaults: defaults)
  }

  // MARK: - Validation

  /// Port of `validatePreset` (service.ts): required non-empty `id`/`name`; when present,
  /// `steps`/`width`/`height` must be positive integers, and every LoRA scale must be finite.
  /// Kept lenient on the string enum fields (like the tolerant decode) — the client owns them.
  static func validate(_ preset: ImagePreset) throws -> ImagePreset {
    if preset.id.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      throw PresetStoreError.validation(#"required field "id" is missing or empty"#)
    }
    if preset.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      throw PresetStoreError.validation(#"required field "name" is missing or empty"#)
    }
    for (label, value) in [("steps", preset.steps), ("width", preset.width), ("height", preset.height)] {
      if let v = value, v <= 0 {
        throw PresetStoreError.validation("required field \"\(label)\" must be positive (got \(v))")
      }
    }
    for (i, lora) in preset.loras.enumerated() {
      if lora.filename.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        throw PresetStoreError.validation("loras[\(i)].filename is missing or empty")
      }
      if !lora.scale.isFinite {
        throw PresetStoreError.validation("loras[\(i)].scale must be a finite number")
      }
    }
    return preset
  }

  // MARK: - Persistence helpers

  /// Tolerant decode: accepts the `{ "presets": [...] }` envelope or a bare `[...]` array,
  /// and falls back to empty on unrecoverable input.
  static func decode(_ data: Data) -> [ImagePreset] {
    let decoder = JSONDecoder()
    if let file = try? decoder.decode(PresetFile.self, from: data) {
      return file.presets
    }
    if let array = try? decoder.decode([ImagePreset].self, from: data) {
      return array
    }
    return []
  }

  static func persist(_ presets: [ImagePreset], to path: URL, fileManager: FileManager) throws {
    let dir = path.deletingLastPathComponent()
    try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(PresetFile(presets: presets))
    try data.write(to: path, options: .atomic)
  }
}
