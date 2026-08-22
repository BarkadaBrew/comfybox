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
import Logging

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

// MARK: - Kroma policy (WP-E20, D14)

/// Kroma as a FIRST-CLASS preset field, never a `loras[]` entry (D14): absence
/// from a list is indistinguishable from "off", which is exactly what O4a
/// forbids. `strength: 0` is a declaration ("no kroma") and validates;
/// an absent `kroma` on a krea2-family image preset is a configuration error.
/// `file` nil = the family-correct default (resolved by the client policy
/// layer, §3.17); set it to pin a specific artifact (e.g. `kroma-v0.1`).
public struct KromaPolicy: Codable, Equatable, Sendable {
  public var strength: Double
  public var file: String?

  public init(strength: Double, file: String? = nil) {
    self.strength = strength
    self.file = file
  }
}

// MARK: - Second-stage recipe (WP-E20, D4)

/// The optional second stage of a two-stage recipe (O5), as a preset declares
/// it: every field optional so a preset can state only what it pins. The
/// request-side shape (`stage2` on `/v1/generate`) is WP-E17's; this is the
/// stored declaration that feeds it.
public struct PresetStage: Codable, Equatable, Sendable {
  public var sampler: String?
  public var sigmaSchedule: String?
  public var steps: Int?
  public var denoise: Double?
  public var eta: Double?
  public var bongmath: Bool?

  public init(
    sampler: String? = nil,
    sigmaSchedule: String? = nil,
    steps: Int? = nil,
    denoise: Double? = nil,
    eta: Double? = nil,
    bongmath: Bool? = nil
  ) {
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
    self.steps = steps
    self.denoise = denoise
    self.eta = eta
    self.bongmath = bongmath
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
  /// WP-E9 (FDD §3.9, D16): path of the VAE file this preset decodes through.
  /// nil = the model directory's VAE (the no-regression default). Wan is
  /// never ambient — it is a named field that appears in every record.
  public var vae: String?

  // WP-E20 (FDD §3.15): the recipe as a preset declares it. Every one of these
  // must appear at ALL FIVE sites (stored property, CodingKeys, init(from:),
  // memberwise init, ResolvedPreset) — the `videoTuning` lesson.
  /// Client policy label (D7): "turbo" | "raw-accel" | "raw-stock" |
  /// "zimage-turbo" | "zimage-base". Never a physical fact — that is
  /// `Krea2Variant`, which is reported, not requested.
  public var checkpointFamily: String?
  /// Required of krea2-family image presets (O4a, D14).
  public var kroma: KromaPolicy?
  /// Sampler name, as `/v1/generate` accepts it (`res_2s`, `dpmpp_2m`, …).
  public var sampler: String?
  /// Sigma-schedule name (`beta`, `karras`, `flow`, …).
  public var sigmaSchedule: String?
  /// Explicit flow shift (D3: the reference preset states 1.15).
  public var shift: Double?
  /// SDE eta (T2). 0 = deterministic.
  public var eta: Double?
  /// RES4LYF bongmath fixed point (T3).
  public var bongmath: Bool?
  /// Optional second-stage detail pass (O5, D4).
  public var stage2: PresetStage?

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
    upscale: PresetUpscale? = nil,
    vae: String? = nil,
    checkpointFamily: String? = nil,
    kroma: KromaPolicy? = nil,
    sampler: String? = nil,
    sigmaSchedule: String? = nil,
    shift: Double? = nil,
    eta: Double? = nil,
    bongmath: Bool? = nil,
    stage2: PresetStage? = nil
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
    case mediaKind, provider, engine, mode, model, customModelPath, baseModel
    case prompt, negativePrompt, promptPrefix, promptSuffix, injectedKeywords
    case steps, guidance, seed, width, height
    case loras, scheduler, upscale
    // Missing until 2026-08-07: with it absent, BOTH the custom decoder and
    // the synthesized encoder dropped videoTuning — every preset-level Tier-A
    // tuning write since task #9 Phase 2 silently vanished on the JSON/API
    // path. The desktop tuning UI was writing values nothing ever read.
    case videoTuning
    // WP-E9: same regression class — a field must be listed here AND in the
    // custom decoder, or both directions silently drop it.
    case vae
    // WP-E20: the nine recipe/policy fields (AC-58 round-trips every one).
    case checkpointFamily, kroma, sampler, sigmaSchedule, shift, eta, bongmath, stage2
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
    vae = try c.decodeIfPresent(String.self, forKey: .vae)
    checkpointFamily = try c.decodeIfPresent(String.self, forKey: .checkpointFamily)
    kroma = try c.decodeIfPresent(KromaPolicy.self, forKey: .kroma)
    sampler = try c.decodeIfPresent(String.self, forKey: .sampler)
    sigmaSchedule = try c.decodeIfPresent(String.self, forKey: .sigmaSchedule)
    shift = try c.decodeIfPresent(Double.self, forKey: .shift)
    eta = try c.decodeIfPresent(Double.self, forKey: .eta)
    bongmath = try c.decodeIfPresent(Bool.self, forKey: .bongmath)
    stage2 = try c.decodeIfPresent(PresetStage.self, forKey: .stage2)
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
  /// WP-E9: nil = the model directory's VAE.
  public var vae: String?
  // WP-E20: carried through verbatim — a preset's recipe fields have no
  // system default; absent means "the engine's default for the variant",
  // which the record (RenderRecipe, WP-E10) then names.
  public var checkpointFamily: String?
  public var kroma: KromaPolicy?
  public var sampler: String?
  public var sigmaSchedule: String?
  public var shift: Double?
  public var eta: Double?
  public var bongmath: Bool?
  public var stage2: PresetStage?

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
    vae = preset.vae
    checkpointFamily = preset.checkpointFamily
    kroma = preset.kroma
    sampler = preset.sampler
    sigmaSchedule = preset.sigmaSchedule
    shift = preset.shift
    eta = preset.eta
    bongmath = preset.bongmath
    stage2 = preset.stage2
  }
}

// MARK: - Errors

public enum PresetStoreError: Error, Equatable, CustomStringConvertible {
  case validation(String)
  case notFound(String)
  /// WP-E20 (AC-44c): a preset that is on disk but failed validation at load.
  /// It stays listed (flagged) so it can be fixed, and it can never resolve.
  case invalid(id: String, reason: String)

  public var description: String {
    switch self {
    case .validation(let m): return "Preset validation failed: \(m)"
    case .notFound(let id): return "Preset not found: \(id)"
    case .invalid(let id, let reason): return "Preset \"\(id)\" is invalid and cannot be selected: \(reason)"
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
  /// WP-E20 (AC-44c): id → reason for every preset that is on disk but fails
  /// validation. Populated at load, cleared by a successful `upsert`/`delete`.
  private var invalidReasons: [String: String] = [:]
  private let logger: Logger

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
    fileManager: FileManager = .default,
    logger: Logger = Logger(label: "comfybox.presets")
  ) {
    self.path = path
    self.defaults = defaults
    self.fileManager = fileManager
    self.logger = logger

    if fileManager.fileExists(atPath: path.path), let data = try? Data(contentsOf: path) {
      let loaded = PresetStore.decodeEntries(data)
      self.presets = loaded.presets
      self.invalidReasons = loaded.undecodable
    } else if seedDefaults {
      self.presets = PresetStore.defaultPresets
      try? PresetStore.persist(self.presets, to: path, fileManager: fileManager)
    } else {
      self.presets = []
    }
    revalidate()
  }

  /// WP-E20 (AC-44c): run ``validate(_:)`` over every loaded preset. An entry
  /// that fails is logged at error and flagged — it stays in ``list()`` so the
  /// desktop app can show and fix it, but ``resolve(_:)`` refuses it and
  /// ``listing()`` serves it with `invalid: true`. Entries that failed to
  /// decode keep their decode reason.
  public func revalidate() {
    lock.lock(); defer { lock.unlock() }
    var reasons: [String: String] = [:]
    for preset in presets {
      if let decodeReason = invalidReasons[preset.id], decodeReason.hasPrefix(PresetStore.undecodablePrefix) {
        reasons[preset.id] = decodeReason
        continue
      }
      do {
        _ = try PresetStore.validate(preset)
      } catch let error as PresetStoreError {
        guard case .validation(let message) = error else { continue }
        reasons[preset.id] = message
      } catch {
        reasons[preset.id] = error.localizedDescription
      }
    }
    invalidReasons = reasons
    for (id, reason) in reasons.sorted(by: { $0.key < $1.key }) {
      logger.error("Preset \"\(id)\" is invalid and cannot be selected: \(reason)")
    }
  }

  // MARK: Reads

  /// All presets, in insertion order.
  public func list() -> [ImagePreset] {
    lock.lock(); defer { lock.unlock() }
    return presets
  }

  /// The preset with `id`, or nil. Returns flagged presets too — editing
  /// (`upsert`) is how they get fixed; selection goes through ``resolve(_:)``.
  public func get(_ id: String) -> ImagePreset? {
    lock.lock(); defer { lock.unlock() }
    return presets.first { $0.id == id }
  }

  /// WP-E20 (AC-44c): why `id` is invalid, or nil when it is valid/unknown.
  public func validationError(for id: String) -> String? {
    lock.lock(); defer { lock.unlock() }
    return invalidReasons[id]
  }

  /// Ids of every flagged preset, in store order.
  public var invalidPresetIds: [String] {
    lock.lock(); defer { lock.unlock() }
    return presets.map(\.id).filter { invalidReasons[$0] != nil }
  }

  /// One entry of `GET /v1/presets`: the preset's own fields, flat, plus the
  /// validity flag (`invalid`, `invalid_reason`) so nothing downstream can
  /// select a flagged preset without seeing why.
  public struct PresetListing: Encodable, Equatable, Sendable {
    public let preset: ImagePreset
    public let invalid: Bool
    public let invalidReason: String?

    private enum FlagKeys: String, CodingKey { case invalid, invalidReason }

    public func encode(to encoder: Encoder) throws {
      try preset.encode(to: encoder)
      var c = encoder.container(keyedBy: FlagKeys.self)
      try c.encode(invalid, forKey: .invalid)
      try c.encodeIfPresent(invalidReason, forKey: .invalidReason)
    }
  }

  /// All presets with their validity, in insertion order (what the API serves).
  public func listing() -> [PresetListing] {
    lock.lock(); defer { lock.unlock() }
    return presets.map {
      PresetListing(preset: $0, invalid: invalidReasons[$0.id] != nil, invalidReason: invalidReasons[$0.id])
    }
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
    invalidReasons[validated.id] = nil
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
      invalidReasons[id] = nil
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
    // WP-E20 (AC-44c): a flagged preset can never be selected.
    if let reason = validationError(for: id) {
      throw PresetStoreError.invalid(id: id, reason: reason)
    }
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
    try validateRecipeFields(preset)
    try validateKromaPolicy(preset)
    return preset
  }

  // MARK: Checkpoint family + kroma (WP-E20, D7, D14, O4a)

  /// The five client policy labels (D7). `turbo`/`raw-accel`/`raw-stock` are
  /// the krea2 families; `zimage-*` keep today's path and need no kroma.
  public static let krea2CheckpointFamilies: Set<String> = ["turbo", "raw-accel", "raw-stock"]
  public static let zimageCheckpointFamilies: Set<String> = ["zimage-turbo", "zimage-base"]
  public static var checkpointFamilies: [String] {
    (krea2CheckpointFamilies.sorted() + zimageCheckpointFamilies.sorted())
  }

  /// Does this preset resolve to a krea2 family? A declared `checkpointFamily`
  /// answers outright; otherwise the `model` spec decides — the four Turbo
  /// aliases, the declared spec→directory table (`krea2-raw`,
  /// `kroma-v0.2-turbo`, config `krea2Models`), or an existing directory that
  /// `Krea2ModelDetection.detect` recognises. Never a filename guess (F3).
  public static func resolvesToKrea2Family(_ preset: ImagePreset) -> Bool {
    if let family = preset.checkpointFamily {
      return krea2CheckpointFamilies.contains(family)
    }
    guard let model = preset.model, !model.isEmpty else { return false }
    if Krea2ModelDetection.isKnownKrea2Model(model) { return true }
    let expanded = (model as NSString).expandingTildeInPath
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue {
      return Krea2ModelDetection.isKrea2ModelDirectory(URL(fileURLWithPath: expanded, isDirectory: true))
    }
    return false
  }

  /// Image/video discriminator for the kroma rule. The live store's eight Krea
  /// entries carry `mediaKind: null` (FDD §3.16 — they are image presets by
  /// `engine: "zimage"`), and a krea2 checkpoint is only ever an image
  /// checkpoint — so anything not declared `"video"` is an image preset here.
  static func isImagePreset(_ preset: ImagePreset) -> Bool {
    preset.mediaKind?.lowercased() != "video"
  }

  /// O4a on the engine (FDD §3.15, AC-44b/44c): a krea2-family image preset
  /// must declare `kroma` — `{strength: 0}` is a declaration; absence is a
  /// configuration error naming the preset and the field. Scoped by family:
  /// the `zimage-*` presets are exempt (D14).
  static func validateKromaPolicy(_ preset: ImagePreset) throws {
    if let family = preset.checkpointFamily,
       !krea2CheckpointFamilies.contains(family), !zimageCheckpointFamilies.contains(family) {
      throw PresetStoreError.validation(
        "preset \"\(preset.id)\": unknown checkpointFamily \"\(family)\" — expected one of "
          + checkpointFamilies.joined(separator: ", "))
    }
    if let kroma = preset.kroma {
      if !kroma.strength.isFinite || kroma.strength < 0 {
        throw PresetStoreError.validation(
          "preset \"\(preset.id)\": kroma.strength must be a finite number >= 0 (got \(kroma.strength))")
      }
      if let file = kroma.file, file.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        throw PresetStoreError.validation("preset \"\(preset.id)\": kroma.file is empty — omit it for the family default")
      }
    }
    if isImagePreset(preset), resolvesToKrea2Family(preset), preset.kroma == nil {
      throw PresetStoreError.validation(
        "preset \"\(preset.id)\": krea2-family image preset (model \"\(preset.model ?? "")\""
          + (preset.checkpointFamily.map { ", checkpointFamily \"\($0)\"" } ?? "")
          + ") must declare \"kroma\" ({\"strength\": <number>, \"file\": <optional>}) — "
          + "an absent kroma is a configuration error (O4a); declare {\"strength\": 0} for none")
    }
  }

  /// The recipe fields go through the SAME resolver `/v1/generate` uses
  /// (WP-E4), so a preset can never name a sampler or schedule the engine
  /// does not have, and the numeric knobs are range-checked here rather than
  /// at render time.
  static func validateRecipeFields(_ preset: ImagePreset) throws {
    func named(_ error: Error) -> PresetStoreError {
      .validation("preset \"\(preset.id)\": " + ((error as? LocalizedError)?.errorDescription ?? "\(error)"))
    }
    do {
      _ = try RecipeNameResolver.resolveSchedulerKind(preset.sampler)
      _ = try RecipeNameResolver.resolveSigmaScheduleKind(preset.sigmaSchedule)
      _ = try RecipeNameResolver.resolveSchedulerKind(preset.stage2?.sampler)
      _ = try RecipeNameResolver.resolveSigmaScheduleKind(preset.stage2?.sigmaSchedule)
    } catch {
      throw named(error)
    }
    if let shift = preset.shift, !(shift.isFinite && shift > 0) {
      throw PresetStoreError.validation("preset \"\(preset.id)\": shift must be a finite number > 0 (got \(shift))")
    }
    if let eta = preset.eta, !(eta.isFinite && eta >= 0) {
      throw PresetStoreError.validation("preset \"\(preset.id)\": eta must be a finite number >= 0 (got \(eta))")
    }
    if let vae = preset.vae, vae.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      throw PresetStoreError.validation("preset \"\(preset.id)\": vae is empty — omit it for the model directory's VAE")
    }
    if let stage2 = preset.stage2 {
      if let steps = stage2.steps, steps <= 0 {
        throw PresetStoreError.validation("preset \"\(preset.id)\": stage2.steps must be positive (got \(steps))")
      }
      if let denoise = stage2.denoise, !(denoise.isFinite && denoise > 0 && denoise <= 1) {
        throw PresetStoreError.validation("preset \"\(preset.id)\": stage2.denoise must be in (0, 1] (got \(denoise))")
      }
      if let eta = stage2.eta, !(eta.isFinite && eta >= 0) {
        throw PresetStoreError.validation("preset \"\(preset.id)\": stage2.eta must be a finite number >= 0 (got \(eta))")
      }
    }
  }

  // MARK: - Persistence helpers

  /// Tolerant decode: accepts the `{ "presets": [...] }` envelope or a bare `[...]` array,
  /// and falls back to empty on unrecoverable input. Undecodable entries are
  /// kept as flagged placeholders — see ``decodeEntries(_:)``.
  static func decode(_ data: Data) -> [ImagePreset] {
    decodeEntries(data).presets
  }

  static let undecodablePrefix = "could not decode preset"

  /// Per-entry decode (WP-E20). Before this, ONE malformed entry failed the
  /// whole-file decode and the store silently loaded EMPTY — every preset
  /// gone until the file was hand-fixed. Now each entry decodes on its own;
  /// one that fails is kept as an `id`/`name` placeholder with its reason in
  /// `undecodable`, so it is listed, flagged and never silently dropped.
  static func decodeEntries(_ data: Data) -> (presets: [ImagePreset], undecodable: [String: String]) {
    guard let root = try? JSONSerialization.jsonObject(with: data) else { return ([], [:]) }
    let rawEntries: [Any]
    if let envelope = root as? [String: Any], let array = envelope["presets"] as? [Any] {
      rawEntries = array
    } else if let array = root as? [Any] {
      rawEntries = array
    } else {
      return ([], [:])
    }
    let decoder = JSONDecoder()
    var presets: [ImagePreset] = []
    var undecodable: [String: String] = [:]
    for (index, raw) in rawEntries.enumerated() {
      guard let object = raw as? [String: Any],
            let entryData = try? JSONSerialization.data(withJSONObject: object)
      else { continue }
      do {
        presets.append(try decoder.decode(ImagePreset.self, from: entryData))
      } catch {
        let id = (object["id"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "#\(index)"
        let name = (object["name"] as? String) ?? id
        presets.append(ImagePreset(id: id, name: name))
        undecodable[id] = "\(undecodablePrefix) \"\(id)\": \(Self.describeDecodingError(error))"
      }
    }
    return (presets, undecodable)
  }

  private static func describeDecodingError(_ error: Error) -> String {
    guard let decodingError = error as? DecodingError else { return "\(error)" }
    func path(_ context: DecodingError.Context) -> String {
      context.codingPath.map(\.stringValue).joined(separator: ".")
    }
    switch decodingError {
    case .keyNotFound(let key, let context):
      let prefix = path(context)
      return "missing key \"\(prefix.isEmpty ? "" : prefix + ".")\(key.stringValue)\""
    case .typeMismatch(_, let context), .valueNotFound(_, let context), .dataCorrupted(let context):
      return "\(context.debugDescription) at \"\(path(context))\""
    @unknown default:
      return "\(decodingError)"
    }
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
