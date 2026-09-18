// SequenceDocument.swift — a Sequence is the saved, replayable product of
// Director (FDD-ltx-director-tab §4.9.2, WP13).
//
// Todd 2026-09-17: "the concept of sequences exist in the system but is not
// utilized. The product from director would be a sequence. Sequences have
// their own sidecar in the way comfyui has JSON workflows."
//
// So every Director render writes `<output>.sequence.json` beside the mp4 (and
// beside the existing generation sidecar, which stays exactly as it is — one
// describes the render, this describes the WORK). Dropping that mp4 back on
// the Director tab reopens the timeline that made it, the way a ComfyUI PNG
// restores its graph.
//
// What it must carry to be replayable:
//   * the timeline as submitted (not as authored — snapped, resolved),
//   * the plan with each chunk's resolved seed and recipe hash,
//   * the assets by path and content hash, so a missing one is a clear error
//     rather than a silently different render,
//   * the outputs, and
//   * the engine build, because "same timeline, different engine" is the
//     first thing to suspect when a replay does not match.

import CryptoKit
import Foundation

public struct SequenceAsset: Codable, Sendable, Equatable {
  /// keyframe | audio | reference
  public var role: String
  public var path: String
  public var sha256: String?

  public init(role: String, path: String, sha256: String? = nil) {
    self.role = role
    self.path = path
    self.sha256 = sha256
  }
}

public struct SequenceOutput: Codable, Sendable, Equatable {
  /// mp4 | gif | lastframe | thumb
  public var kind: String
  public var path: String
  public var frames: Int?
  public var durationSeconds: Double?

  public init(kind: String, path: String, frames: Int? = nil, durationSeconds: Double? = nil) {
    self.kind = kind
    self.path = path
    self.frames = frames
    self.durationSeconds = durationSeconds
  }
}

public struct SequenceChunkRecord: Codable, Sendable, Equatable {
  public var index: Int
  public var startFrame: Int
  public var frames: Int
  /// The seed the engine actually used (UInt64, as the generation record
  /// stores it) — a replay reproduces the render only if this is exact.
  public var seed: UInt64?
  public var recipeHash: String?

  public init(
    index: Int, startFrame: Int, frames: Int, seed: UInt64? = nil, recipeHash: String? = nil
  ) {
    self.index = index
    self.startFrame = startFrame
    self.frames = frames
    self.seed = seed
    self.recipeHash = recipeHash
  }

  enum CodingKeys: String, CodingKey {
    case index, frames, seed
    case startFrame = "start_frame"
    case recipeHash = "recipe_hash"
  }
}

public struct SequenceEngineRecord: Codable, Sendable, Equatable {
  public var buildSha: String?
  public var stitchPath: String?
  public var toneMatch: Bool?
  public var audioSource: String?

  public init(
    buildSha: String? = nil, stitchPath: String? = nil, toneMatch: Bool? = nil,
    audioSource: String? = nil
  ) {
    self.buildSha = buildSha
    self.stitchPath = stitchPath
    self.toneMatch = toneMatch
    self.audioSource = audioSource
  }

  enum CodingKeys: String, CodingKey {
    case buildSha = "build_sha"
    case stitchPath = "stitch_path"
    case toneMatch = "tone_match"
    case audioSource = "audio_source"
  }
}

/// The document itself.
public struct SequenceDocument: Codable, Sendable, Equatable {
  public static let schemaName = "comfybox.sequence"
  public static let schemaVersion = 1
  /// Written beside the mp4: `clip.mp4` -> `clip.sequence.json`.
  public static let sidecarSuffix = ".sequence.json"

  public var schema: String
  public var version: Int
  public var id: String
  /// director | frames — `frames` is reserved for the daemon's image sequences.
  public var kind: String
  public var name: String
  public var createdAt: Date
  public var source: String?
  /// The library/preset ids this was assembled from, when it was.
  public var presetId: String?
  public var libraryItemIds: [String]

  public var timeline: DirectorTimeline
  public var chunks: [SequenceChunkRecord]
  public var assets: [SequenceAsset]
  public var outputs: [SequenceOutput]
  public var engine: SequenceEngineRecord

  public init(
    id: String,
    kind: String = "director",
    name: String,
    createdAt: Date = Date(),
    source: String? = nil,
    presetId: String? = nil,
    libraryItemIds: [String] = [],
    timeline: DirectorTimeline,
    chunks: [SequenceChunkRecord] = [],
    assets: [SequenceAsset] = [],
    outputs: [SequenceOutput] = [],
    engine: SequenceEngineRecord = SequenceEngineRecord()
  ) {
    schema = Self.schemaName
    version = Self.schemaVersion
    self.id = id
    self.kind = kind
    self.name = name
    self.createdAt = createdAt
    self.source = source
    self.presetId = presetId
    self.libraryItemIds = libraryItemIds
    self.timeline = timeline
    self.chunks = chunks
    self.assets = assets
    self.outputs = outputs
    self.engine = engine
  }

  enum CodingKeys: String, CodingKey {
    case schema, version, id, kind, name, source, timeline, chunks, assets, outputs, engine
    case createdAt = "created_at"
    case presetId = "preset_id"
    case libraryItemIds = "library_item_ids"
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    schema = try c.decodeIfPresent(String.self, forKey: .schema) ?? Self.schemaName
    version = try c.decodeIfPresent(Int.self, forKey: .version) ?? 1
    guard version <= Self.schemaVersion else {
      throw DecodingError.dataCorruptedError(
        forKey: .version, in: c,
        debugDescription:
          "sequence schema version \(version) is newer than this engine understands (\(Self.schemaVersion))")
    }
    id = try c.decode(String.self, forKey: .id)
    kind = try c.decodeIfPresent(String.self, forKey: .kind) ?? "director"
    name = try c.decodeIfPresent(String.self, forKey: .name) ?? id
    if let text = try? c.decode(String.self, forKey: .createdAt) {
      createdAt = SequenceDocument.iso.date(from: text) ?? Date()
    } else {
      createdAt = Date()
    }
    source = try c.decodeIfPresent(String.self, forKey: .source)
    presetId = try c.decodeIfPresent(String.self, forKey: .presetId)
    libraryItemIds = try c.decodeIfPresent([String].self, forKey: .libraryItemIds) ?? []
    timeline = try c.decode(DirectorTimeline.self, forKey: .timeline)
    chunks = try c.decodeIfPresent([SequenceChunkRecord].self, forKey: .chunks) ?? []
    assets = try c.decodeIfPresent([SequenceAsset].self, forKey: .assets) ?? []
    outputs = try c.decodeIfPresent([SequenceOutput].self, forKey: .outputs) ?? []
    engine = try c.decodeIfPresent(SequenceEngineRecord.self, forKey: .engine) ?? SequenceEngineRecord()
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(schema, forKey: .schema)
    try c.encode(version, forKey: .version)
    try c.encode(id, forKey: .id)
    try c.encode(kind, forKey: .kind)
    try c.encode(name, forKey: .name)
    try c.encode(Self.iso.string(from: createdAt), forKey: .createdAt)
    try c.encodeIfPresent(source, forKey: .source)
    try c.encodeIfPresent(presetId, forKey: .presetId)
    try c.encode(libraryItemIds, forKey: .libraryItemIds)
    try c.encode(timeline, forKey: .timeline)
    try c.encode(chunks, forKey: .chunks)
    try c.encode(assets, forKey: .assets)
    try c.encode(outputs, forKey: .outputs)
    try c.encode(engine, forKey: .engine)
  }

  static let iso: ISO8601DateFormatter = {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime]
    return f
  }()
}

// MARK: - Sidecar IO

public enum SequenceSidecar {

  /// `~/Pictures/ComfyBox/clip.mp4` -> `~/Pictures/ComfyBox/clip.sequence.json`
  public static func path(forMediaAt mediaPath: String) -> String {
    (mediaPath as NSString).deletingPathExtension + SequenceDocument.sidecarSuffix
  }

  /// Write beside the media. Non-fatal by contract: a render that succeeded
  /// must not be reported as failed because its sidecar could not be written
  /// (the same rule the generation sidecar follows).
  @discardableResult
  public static func write(_ document: SequenceDocument, forMediaAt mediaPath: String) -> Bool {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    guard let data = try? encoder.encode(document) else { return false }
    let url = URL(fileURLWithPath: path(forMediaAt: mediaPath))
    return (try? data.write(to: url, options: .atomic)) != nil
  }

  /// Read the sequence for a rendered file, if it has one.
  public static func read(forMediaAt mediaPath: String) -> SequenceDocument? {
    read(at: path(forMediaAt: mediaPath))
  }

  public static func read(at sidecarPath: String) -> SequenceDocument? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: sidecarPath)) else { return nil }
    return try? JSONDecoder().decode(SequenceDocument.self, from: data)
  }

  /// Content hash of an asset, so a replay can say "that keyframe changed"
  /// instead of quietly rendering something else.
  public static func sha256(ofFileAt path: String) -> String? {
    guard let data = try? Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
    else { return nil }
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  /// Every asset a timeline references, hashed.
  public static func assets(of timeline: DirectorTimeline) -> [SequenceAsset] {
    var out: [SequenceAsset] = []
    for keyframe in timeline.keyframes {
      guard let path = keyframe.imagePath, !path.isEmpty else { continue }
      out.append(SequenceAsset(role: "keyframe", path: path, sha256: sha256(ofFileAt: path)))
    }
    for clip in timeline.audioClips {
      guard !clip.audioPath.isEmpty else { continue }
      out.append(SequenceAsset(role: "audio", path: clip.audioPath, sha256: sha256(ofFileAt: clip.audioPath)))
    }
    return out
  }

  /// What a replay should warn about before it spends the GPU.
  public struct ReplayCheck: Codable, Sendable, Equatable {
    public var missingAssets: [String]
    public var changedAssets: [String]
    public var recipeDrift: Bool
    public var engineChanged: Bool
    /// The chunks did not all render under the same recipe. A long sequence
    /// can span hours, and a preset edited halfway through changes later
    /// chunks only — so the clip on disk was never reproducible as one thing,
    /// whatever the recipe is now. Only detectable since every chunk records
    /// its OWN hash; before that they all reported chunk 0's.
    public var chunksDisagree: Bool
    /// Chunks with no seed recorded. A replay of those is a different render,
    /// so it is named rather than left to be discovered afterwards.
    public var chunksMissingSeed: [Int]

    public var ok: Bool {
      missingAssets.isEmpty && changedAssets.isEmpty && !recipeDrift && !engineChanged
        && !chunksDisagree && chunksMissingSeed.isEmpty
    }

    enum CodingKeys: String, CodingKey {
      case missingAssets = "missing_assets"
      case changedAssets = "changed_assets"
      case recipeDrift = "recipe_drift"
      case engineChanged = "engine_changed"
      case chunksDisagree = "chunks_disagree"
      case chunksMissingSeed = "chunks_missing_seed"
    }
  }

  /// Compare a stored sequence against the world as it is now.
  public static func check(
    _ document: SequenceDocument, currentRecipeHash: String?, currentEngineBuild: String?
  ) -> ReplayCheck {
    var missing: [String] = []
    var changed: [String] = []
    for asset in document.assets {
      guard FileManager.default.fileExists(atPath: asset.path) else {
        missing.append(asset.path)
        continue
      }
      if let stored = asset.sha256, let now = sha256(ofFileAt: asset.path), stored != now {
        changed.append(asset.path)
      }
    }
    let storedRecipe = document.chunks.first?.recipeHash
    let drift = storedRecipe != nil && currentRecipeHash != nil && storedRecipe != currentRecipeHash
    let engineChanged = document.engine.buildSha != nil && currentEngineBuild != nil
      && document.engine.buildSha != currentEngineBuild
    let hashes = Set(document.chunks.compactMap(\.recipeHash))
    return ReplayCheck(
      missingAssets: missing, changedAssets: changed,
      recipeDrift: drift, engineChanged: engineChanged,
      chunksDisagree: hashes.count > 1,
      chunksMissingSeed: document.chunks.filter { $0.seed == nil }.map(\.index))
  }
}
