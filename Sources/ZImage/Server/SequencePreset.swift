// SequencePreset.swift — reusable Director parameters
// (FDD-ltx-director-tab §4.9.3, WP14; PRD-creative-library release one).
//
// Todd 2026-09-17: "include a preset system so a User can CRUD presets for
// Video length and any other parameters and the AI assisted system will
// generate a video based on those parameters."
//
// An image/video preset answers "how is this rendered". A SEQUENCE preset
// answers "what shape is this piece of work": how long, at what rate and
// aspect, with keyframes where, prompts changing how often, audio from where,
// and which render recipe underneath. It is the contract the drafter fills.
//
// It lives in its own store rather than as a third `mediaKind` on
// `PresetStore`: every existing switch there reads "not video" as "image"
// (PresetStore.isImagePreset, resolvedLoRAFamily, validateLTX2ImagePreset), so
// a third kind would have to be threaded through paths that have nothing to do
// with timelines. A sequence preset REFERENCES a video preset by id for the
// render recipe, which keeps one source of truth for sampler settings.

import Foundation

public struct SequencePreset: Codable, Sendable, Equatable, Identifiable {
  public var id: String
  public var name: String
  public var description: String?

  // Shape
  /// Authoring length. The engine snaps to `1 + 8k` at compile time.
  public var lengthSeconds: Double
  public var fps: Int
  public var width: Int
  public var height: Int

  // Recipe
  /// A video preset id — the sampler/LoRA recipe this renders through.
  public var videoPresetId: String?
  /// An image preset id — used when the drafter has to render keyframe stills.
  public var keyframeImagePresetId: String?
  public var steps: Int?
  public var negativePrompt: String?
  public var character: String?

  // Authoring policy — what the drafter must honour.
  /// `fflf` (first and last) | `every_n_seconds:<n>` | `first_only` | `none`
  public var keyframePolicy: String
  /// Target seconds per prompt segment.
  public var segmentCadenceSeconds: Double?
  /// `generated` | `imported` | `driven` (§4.7 audio-driven chunks)
  public var audio: String
  /// `fixed:<n>` | `random`
  public var seedPolicy: String
  /// Free guidance handed to the drafter ("static camera, slow push-in").
  public var authorNotes: String?

  // Export defaults (§4.8)
  public var gifFps: Int?
  public var gifWidth: Int?

  public var createdAt: Date
  public var updatedAt: Date

  public init(
    id: String,
    name: String,
    description: String? = nil,
    lengthSeconds: Double = 10,
    fps: Int = 24,
    width: Int = 576,
    height: Int = 896,
    videoPresetId: String? = nil,
    keyframeImagePresetId: String? = nil,
    steps: Int? = nil,
    negativePrompt: String? = nil,
    character: String? = nil,
    keyframePolicy: String = "first_only",
    segmentCadenceSeconds: Double? = nil,
    audio: String = "generated",
    seedPolicy: String = "random",
    authorNotes: String? = nil,
    gifFps: Int? = nil,
    gifWidth: Int? = nil,
    createdAt: Date = Date(),
    updatedAt: Date = Date()
  ) {
    self.id = id
    self.name = name
    self.description = description
    self.lengthSeconds = lengthSeconds
    self.fps = fps
    self.width = width
    self.height = height
    self.videoPresetId = videoPresetId
    self.keyframeImagePresetId = keyframeImagePresetId
    self.steps = steps
    self.negativePrompt = negativePrompt
    self.character = character
    self.keyframePolicy = keyframePolicy
    self.segmentCadenceSeconds = segmentCadenceSeconds
    self.audio = audio
    self.seedPolicy = seedPolicy
    self.authorNotes = authorNotes
    self.gifFps = gifFps
    self.gifWidth = gifWidth
    self.createdAt = createdAt
    self.updatedAt = updatedAt
  }

  enum CodingKeys: String, CodingKey {
    case id, name, description, fps, width, height, steps, character, audio
    case lengthSeconds = "length_seconds"
    case videoPresetId = "video_preset_id"
    case keyframeImagePresetId = "keyframe_image_preset_id"
    case negativePrompt = "negative_prompt"
    case keyframePolicy = "keyframe_policy"
    case segmentCadenceSeconds = "segment_cadence_seconds"
    case seedPolicy = "seed_policy"
    case authorNotes = "author_notes"
    case gifFps = "gif_fps"
    case gifWidth = "gif_width"
    case createdAt = "created_at"
    case updatedAt = "updated_at"
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decode(String.self, forKey: .id)
    name = try c.decodeIfPresent(String.self, forKey: .name) ?? id
    description = try c.decodeIfPresent(String.self, forKey: .description)
    lengthSeconds = try c.decodeIfPresent(Double.self, forKey: .lengthSeconds) ?? 10
    fps = try c.decodeIfPresent(Int.self, forKey: .fps) ?? 24
    width = try c.decodeIfPresent(Int.self, forKey: .width) ?? 576
    height = try c.decodeIfPresent(Int.self, forKey: .height) ?? 896
    videoPresetId = try c.decodeIfPresent(String.self, forKey: .videoPresetId)
    keyframeImagePresetId = try c.decodeIfPresent(String.self, forKey: .keyframeImagePresetId)
    steps = try c.decodeIfPresent(Int.self, forKey: .steps)
    negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
    character = try c.decodeIfPresent(String.self, forKey: .character)
    keyframePolicy = try c.decodeIfPresent(String.self, forKey: .keyframePolicy) ?? "first_only"
    segmentCadenceSeconds = try c.decodeIfPresent(Double.self, forKey: .segmentCadenceSeconds)
    audio = try c.decodeIfPresent(String.self, forKey: .audio) ?? "generated"
    seedPolicy = try c.decodeIfPresent(String.self, forKey: .seedPolicy) ?? "random"
    authorNotes = try c.decodeIfPresent(String.self, forKey: .authorNotes)
    gifFps = try c.decodeIfPresent(Int.self, forKey: .gifFps)
    gifWidth = try c.decodeIfPresent(Int.self, forKey: .gifWidth)
    createdAt = Self.date(c, .createdAt) ?? Date()
    updatedAt = Self.date(c, .updatedAt) ?? Date()
  }

  public func encode(to encoder: Encoder) throws {
    var c = encoder.container(keyedBy: CodingKeys.self)
    try c.encode(id, forKey: .id)
    try c.encode(name, forKey: .name)
    try c.encodeIfPresent(description, forKey: .description)
    try c.encode(lengthSeconds, forKey: .lengthSeconds)
    try c.encode(fps, forKey: .fps)
    try c.encode(width, forKey: .width)
    try c.encode(height, forKey: .height)
    try c.encodeIfPresent(videoPresetId, forKey: .videoPresetId)
    try c.encodeIfPresent(keyframeImagePresetId, forKey: .keyframeImagePresetId)
    try c.encodeIfPresent(steps, forKey: .steps)
    try c.encodeIfPresent(negativePrompt, forKey: .negativePrompt)
    try c.encodeIfPresent(character, forKey: .character)
    try c.encode(keyframePolicy, forKey: .keyframePolicy)
    try c.encodeIfPresent(segmentCadenceSeconds, forKey: .segmentCadenceSeconds)
    try c.encode(audio, forKey: .audio)
    try c.encode(seedPolicy, forKey: .seedPolicy)
    try c.encodeIfPresent(authorNotes, forKey: .authorNotes)
    try c.encodeIfPresent(gifFps, forKey: .gifFps)
    try c.encodeIfPresent(gifWidth, forKey: .gifWidth)
    try c.encode(SequenceDocument.iso.string(from: createdAt), forKey: .createdAt)
    try c.encode(SequenceDocument.iso.string(from: updatedAt), forKey: .updatedAt)
  }

  private static func date(
    _ c: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys
  ) -> Date? {
    guard let text = try? c.decode(String.self, forKey: key) else { return nil }
    return SequenceDocument.iso.date(from: text)
  }

  // MARK: - Derived shape

  /// Frames before snapping. The compiler snaps UP to `1 + 8k`.
  public var requestedFrames: Int {
    max(1, Int((lengthSeconds * Double(fps)).rounded()))
  }

  /// The length the engine will actually render: `1 + 8k`, at least 97, at
  /// most 4609 (16 chunks) — the Director limits, applied here so a preset can
  /// never promise a length Director would refuse.
  public var snappedFrames: Int {
    let requested = max(requestedFrames, 97)
    // The smallest 1 + 8k that covers the request: 240 -> 241, not 249.
    let steps = Int(ceil(Double(requested - 1) / 8.0))
    return min(1 + steps * 8, 4609)
  }

  public var snappedSeconds: Double {
    Double(snappedFrames) / Double(max(fps, 1))
  }

  /// Chunks Director will compile this into, at ≤ 289 frames each.
  public var chunkCount: Int {
    max(1, Int(ceil(Double(snappedFrames) / 289.0)))
  }

  /// Keyframe positions the drafter must fill, in frames.
  public func keyframeFrames() -> [Int] {
    let last = snappedFrames - 1
    switch keyframePolicy {
    case "none":
      return []
    case "fflf":
      return [0, last]
    case "first_only":
      return [0]
    default:
      guard keyframePolicy.hasPrefix("every_n_seconds:"),
            let seconds = Double(keyframePolicy.dropFirst("every_n_seconds:".count)),
            seconds > 0
      else { return [0] }
      let step = max(8, Int((seconds * Double(fps)).rounded()) / 8 * 8)
      return stride(from: 0, through: last, by: step).map { $0 }
    }
  }

  /// Prompt-segment boundaries the drafter must write to, in frames.
  public func segmentSpans() -> [(start: Int, length: Int)] {
    guard let cadence = segmentCadenceSeconds, cadence > 0 else {
      return [(0, snappedFrames)]
    }
    let step = max(8, Int((cadence * Double(fps)).rounded()))
    var spans: [(Int, Int)] = []
    var start = 0
    while start < snappedFrames {
      let length = min(step, snappedFrames - start)
      // A final sliver is folded into the previous segment rather than
      // shipped as a one-frame prompt nobody can see.
      if length < step / 2, var last = spans.popLast() {
        last.1 += length
        spans.append(last)
      } else {
        spans.append((start, length))
      }
      start += step
    }
    return spans.map { (start: $0.0, length: $0.1) }
  }

  public var seed: Int? {
    guard seedPolicy.hasPrefix("fixed:") else { return nil }
    return Int(seedPolicy.dropFirst("fixed:".count))
  }
}

// MARK: - Store

public enum SequencePresetError: Error, LocalizedError, Equatable {
  case invalid(String)
  case notFound(String)

  public var errorDescription: String? {
    switch self {
    case .invalid(let m): return m
    case .notFound(let id): return "sequence preset '\(id)' not found"
    }
  }
}

public final class SequencePresetStore: @unchecked Sendable {
  private let path: URL
  private let lock = NSLock()
  private var byId: [String: SequencePreset] = [:]

  public init(path: URL) {
    self.path = path
    if let data = try? Data(contentsOf: path),
       let file = try? JSONDecoder().decode([SequencePreset].self, from: data) {
      byId = Dictionary(uniqueKeysWithValues: file.map { ($0.id, $0) })
    }
  }

  public convenience init(home: String = NSHomeDirectory()) {
    self.init(
      path: URL(fileURLWithPath: home).appendingPathComponent(".comfybox/sequence-presets.json"))
  }

  public func all() -> [SequencePreset] {
    lock.lock(); defer { lock.unlock() }
    return byId.values.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
  }

  public func preset(id: String) -> SequencePreset? {
    lock.lock(); defer { lock.unlock() }
    return byId[id]
  }

  @discardableResult
  public func upsert(_ incoming: SequencePreset, now: Date = Date()) throws -> SequencePreset {
    var preset = incoming
    try Self.validate(&preset)
    lock.lock(); defer { lock.unlock() }
    if let existing = byId[preset.id] { preset.createdAt = existing.createdAt }
    preset.updatedAt = now
    byId[preset.id] = preset
    try persist()
    return preset
  }

  @discardableResult
  public func delete(id: String) throws -> Bool {
    lock.lock(); defer { lock.unlock() }
    guard byId.removeValue(forKey: id) != nil else { return false }
    try persist()
    return true
  }

  static func validate(_ preset: inout SequencePreset) throws {
    preset.id = preset.id.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !preset.id.isEmpty else { throw SequencePresetError.invalid("'id' is required") }
    guard !preset.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw SequencePresetError.invalid("'name' is required")
    }
    guard preset.lengthSeconds > 0, preset.lengthSeconds <= 600 else {
      throw SequencePresetError.invalid("'length_seconds' must be 0…600")
    }
    guard (1...60).contains(preset.fps) else {
      throw SequencePresetError.invalid("'fps' must be 1…60")
    }
    guard preset.width > 0, preset.height > 0 else {
      throw SequencePresetError.invalid("'width'/'height' must be positive")
    }
    guard ["generated", "imported", "driven"].contains(preset.audio) else {
      throw SequencePresetError.invalid("'audio' must be generated, imported or driven")
    }
    let policy = preset.keyframePolicy
    let policyOK = ["none", "fflf", "first_only"].contains(policy)
      || (policy.hasPrefix("every_n_seconds:")
          && (Double(policy.dropFirst("every_n_seconds:".count)) ?? 0) > 0)
    guard policyOK else {
      throw SequencePresetError.invalid(
        "'keyframe_policy' must be none, first_only, fflf or every_n_seconds:<n>")
    }
    guard preset.seedPolicy == "random"
      || (preset.seedPolicy.hasPrefix("fixed:") && Int(preset.seedPolicy.dropFirst(6)) != nil)
    else {
      throw SequencePresetError.invalid("'seed_policy' must be random or fixed:<n>")
    }
    // A driving voice track has to land somewhere the model can follow: a
    // clip with no keyframe at all has no face to move (§4.7).
    if preset.audio == "driven", preset.keyframePolicy == "none" {
      throw SequencePresetError.invalid(
        "'audio: driven' needs at least a first keyframe — the voice drives a face")
    }
    if let cadence = preset.segmentCadenceSeconds, cadence <= 0 {
      throw SequencePresetError.invalid("'segment_cadence_seconds' must be positive")
    }
  }

  private func persist() throws {
    try FileManager.default.createDirectory(
      at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let sorted = byId.values.sorted { $0.id < $1.id }
    try encoder.encode(sorted).write(to: path, options: .atomic)
  }
}
