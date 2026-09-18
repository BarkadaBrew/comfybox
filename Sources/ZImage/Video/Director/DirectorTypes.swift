// DirectorTypes.swift — the Director timeline document (LTX-2 Director tab, WP1).
//
// One authored document, engine-agnostic and versioned, that the compiler
// (WP2b) turns into per-chunk /v1/video/generate bodies. snake_case on the
// wire and in `.cbdirector` files (identical bytes); camelCase in Swift. The
// snake_case mapping is done ONLY by the encoder/decoder key strategy so the
// WarmServer route decoder (`JSONDecoder` + `.convertFromSnakeCase`) and
// `DirectorJSON.decoder()` produce identical values.
//
// Every field marked "default" in docs/FDD-ltx-director-tab.md §4.1 (as
// amended by the Phase 1 deltas) decodes through an explicit `init(from:)`
// with `decodeIfPresent`, so a minimal hand-written or MCP-shaped timeline is
// enough. Unknown keys are ignored (Phase 2 additive safety). Frames are the
// source of truth; seconds are derived in the UI only.

import Foundation

// MARK: - Timeline

public struct DirectorTimeline: Codable, Sendable, Equatable {

  /// Current schema version. A missing `version` decodes as 1; anything
  /// newer is rejected with `DirectorError.unsupportedVersion`.
  public static let currentVersion = 1

  public struct LoRARef: Codable, Sendable, Equatable {
    public var path: String
    public var scale: Float
    public var role: String?

    public init(path: String, scale: Float = 1.0, role: String? = nil) {
      self.path = path
      self.scale = scale
      self.role = role
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      path = try c.decode(String.self, forKey: .path)
      scale = try c.decodeIfPresent(Float.self, forKey: .scale) ?? 1.0
      role = try c.decodeIfPresent(String.self, forKey: .role)
    }
  }

  public struct Settings: Codable, Sendable, Equatable {
    /// The builtin frame rate a timeline without `fps` is displayed and
    /// planned at until the server resolves the real one (request > preset >
    /// config > builtin, via `prepareLocalVideo`).
    public static let defaultFps = 24

    /// Timeline frame rate, 1...120. nil (omitted on the wire) => the preset /
    /// config / builtin fps applies; the submit route resolves it from chunk
    /// 0's prepared request before compiling the bodies it renders.
    public var fps: Int?
    /// Output width, multiple of 32.
    public var width: Int
    /// Output height, multiple of 32.
    public var height: Int
    /// Timeline length in frames; the validator snaps UP to 1+8k and requires
    /// >= 97 after the snap (the production floor) and <= 16 chunks.
    public var lengthFrames: Int
    /// Base seed; chunk k renders with `seed + k`. nil (omitted on the wire)
    /// => the preset / builtin seed applies (resolved at submit from chunk 0).
    public var seed: UInt64?
    /// Server video preset id, resolved per chunk by prepareLocalVideo.
    public var preset: String?
    /// Empty => the preset's LoRAs apply.
    public var loras: [LoRARef]
    /// nil => the preset negative applies.
    public var negativePrompt: String?
    /// nil => preset/config default.
    public var steps: Int?
    /// nil => every chunk body sends skip_character_injection so a
    /// keyframe-less chunk 0 never silently becomes the default character.
    public var character: String?

    public init(
      fps: Int? = 24, width: Int, height: Int, lengthFrames: Int, seed: UInt64? = 42,
      preset: String? = nil, loras: [LoRARef] = [], negativePrompt: String? = nil,
      steps: Int? = nil, character: String? = nil
    ) {
      self.fps = fps
      self.width = width
      self.height = height
      self.lengthFrames = lengthFrames
      self.seed = seed
      self.preset = preset
      self.loras = loras
      self.negativePrompt = negativePrompt
      self.steps = steps
      self.character = character
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      // No decode defaults for fps/seed: an absent value must stay absent so
      // the preset/config value wins (request > preset > config > builtin).
      fps = try c.decodeIfPresent(Int.self, forKey: .fps)
      width = try c.decode(Int.self, forKey: .width)
      height = try c.decode(Int.self, forKey: .height)
      lengthFrames = try c.decode(Int.self, forKey: .lengthFrames)
      seed = try c.decodeIfPresent(UInt64.self, forKey: .seed)
      preset = try c.decodeIfPresent(String.self, forKey: .preset)
      loras = try c.decodeIfPresent([LoRARef].self, forKey: .loras) ?? []
      negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
      steps = try c.decodeIfPresent(Int.self, forKey: .steps)
      character = try c.decodeIfPresent(String.self, forKey: .character)
    }
  }

  public struct Keyframe: Codable, Sendable, Equatable {
    public var id: String
    /// Required unless `imageBase64` is present.
    public var imagePath: String?
    /// Embedded PNG bytes. Materialized by the server before compile; the
    /// desktop writes it only when "Embed assets" is chosen on save.
    public var imageBase64: String?
    /// Timeline frame, 0...length-1, MUST be a multiple of 8 (the latent grid
    /// — the engine floors frame/8, the desktop ruler snaps to the same grid).
    public var frame: Int
    /// (0, 1]; 1.0 = the frame is frozen to the image.
    public var strength: Float
    /// true pins the image to frame length-1 (the compiler ignores `frame`;
    /// the validator rewrites it). At most one per timeline.
    public var isEndFrame: Bool

    public init(
      id: String, imagePath: String? = nil, imageBase64: String? = nil, frame: Int,
      strength: Float = 1.0, isEndFrame: Bool = false
    ) {
      self.id = id
      self.imagePath = imagePath
      self.imageBase64 = imageBase64
      self.frame = frame
      self.strength = strength
      self.isEndFrame = isEndFrame
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      id = try c.decode(String.self, forKey: .id)
      imagePath = try c.decodeIfPresent(String.self, forKey: .imagePath)
      imageBase64 = try c.decodeIfPresent(String.self, forKey: .imageBase64)
      frame = try c.decodeIfPresent(Int.self, forKey: .frame) ?? 0
      strength = try c.decodeIfPresent(Float.self, forKey: .strength) ?? 1.0
      isEndFrame = try c.decodeIfPresent(Bool.self, forKey: .isEndFrame) ?? false
    }
  }

  public struct PromptSegment: Codable, Sendable, Equatable {
    public var id: String
    public var startFrame: Int
    public var lengthFrames: Int
    public var prompt: String

    public init(id: String, startFrame: Int, lengthFrames: Int, prompt: String) {
      self.id = id
      self.startFrame = startFrame
      self.lengthFrames = lengthFrames
      self.prompt = prompt
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      id = try c.decode(String.self, forKey: .id)
      startFrame = try c.decodeIfPresent(Int.self, forKey: .startFrame) ?? 0
      lengthFrames = try c.decodeIfPresent(Int.self, forKey: .lengthFrames) ?? 0
      prompt = try c.decodeIfPresent(String.self, forKey: .prompt) ?? ""
    }
  }

  public struct AudioClip: Codable, Sendable, Equatable {
    public var id: String
    /// Anything AVAssetReader decodes (WAV/AIFF/MP3/M4A/AAC).
    public var audioPath: String
    /// Timeline frame where the (trimmed) clip start lands.
    public var startFrame: Int
    /// Frames of the clip to use (padded with silence if the file is shorter).
    public var lengthFrames: Int
    /// Frames skipped at the head of the file.
    public var trimStartFrames: Int
    /// Linear gain, >= 0.
    public var gain: Float
    /// AUDIO-DRIVEN (FDD §4.7, WP11): this clip is the VOICE, and every chunk
    /// it covers renders against its own slice of it, so lips follow real
    /// words. The final assembly still muxes this track untouched, so splices
    /// cannot click or drift. At most one clip on a timeline may set it.
    public var drivesVideo: Bool

    public init(
      id: String, audioPath: String, startFrame: Int, lengthFrames: Int,
      trimStartFrames: Int = 0, gain: Float = 1.0, drivesVideo: Bool = false
    ) {
      self.id = id
      self.audioPath = audioPath
      self.startFrame = startFrame
      self.lengthFrames = lengthFrames
      self.trimStartFrames = trimStartFrames
      self.gain = gain
      self.drivesVideo = drivesVideo
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      id = try c.decode(String.self, forKey: .id)
      audioPath = try c.decode(String.self, forKey: .audioPath)
      startFrame = try c.decodeIfPresent(Int.self, forKey: .startFrame) ?? 0
      lengthFrames = try c.decodeIfPresent(Int.self, forKey: .lengthFrames) ?? 0
      trimStartFrames = try c.decodeIfPresent(Int.self, forKey: .trimStartFrames) ?? 0
      gain = try c.decodeIfPresent(Float.self, forKey: .gain) ?? 1.0
      drivesVideo = try c.decodeIfPresent(Bool.self, forKey: .drivesVideo) ?? false
    }
  }

  public enum AudioMode: String, Codable, Sendable, Equatable {
    /// Each chunk generates its own audio from noise (seams across chunks).
    case generated
    /// `audio_clips` are mixed onto the timeline; chunks render with audio off.
    case imported
    /// Phase 2: generate only where nothing was imported. Rejected in Phase 1.
    case inpaint
  }

  public struct AudioSettings: Codable, Sendable, Equatable {
    public var mode: AudioMode

    public init(mode: AudioMode = .generated) {
      self.mode = mode
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      mode = try c.decodeIfPresent(AudioMode.self, forKey: .mode) ?? .generated
    }
  }

  /// Phase 2 placeholder (IC-LoRA reference track). Non-empty is rejected.
  public struct ReferenceClip: Codable, Sendable, Equatable {
    public var id: String
    public var videoPath: String
    public var startFrame: Int
    public var lengthFrames: Int
    public var trimStartFrames: Int
    public var strength: Float
    public var attention: Float

    public init(
      id: String, videoPath: String, startFrame: Int, lengthFrames: Int,
      trimStartFrames: Int = 0, strength: Float = 1.0, attention: Float = 0.65
    ) {
      self.id = id
      self.videoPath = videoPath
      self.startFrame = startFrame
      self.lengthFrames = lengthFrames
      self.trimStartFrames = trimStartFrames
      self.strength = strength
      self.attention = attention
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      id = try c.decode(String.self, forKey: .id)
      videoPath = try c.decode(String.self, forKey: .videoPath)
      startFrame = try c.decodeIfPresent(Int.self, forKey: .startFrame) ?? 0
      lengthFrames = try c.decodeIfPresent(Int.self, forKey: .lengthFrames) ?? 0
      trimStartFrames = try c.decodeIfPresent(Int.self, forKey: .trimStartFrames) ?? 0
      strength = try c.decodeIfPresent(Float.self, forKey: .strength) ?? 1.0
      attention = try c.decodeIfPresent(Float.self, forKey: .attention) ?? 0.65
    }
  }

  /// Phase 2 placeholder (temporal re-render of a range). Enabled is rejected.
  public struct Retake: Codable, Sendable, Equatable {
    public var enabled: Bool
    public var videoPath: String?
    public var startFrame: Int
    public var lengthFrames: Int
    public var strength: Float
    public var prompt: String?

    public init(
      enabled: Bool = false, videoPath: String? = nil, startFrame: Int = 0,
      lengthFrames: Int = 0, strength: Float = 1.0, prompt: String? = nil
    ) {
      self.enabled = enabled
      self.videoPath = videoPath
      self.startFrame = startFrame
      self.lengthFrames = lengthFrames
      self.strength = strength
      self.prompt = prompt
    }

    public init(from decoder: Decoder) throws {
      let c = try decoder.container(keyedBy: CodingKeys.self)
      enabled = try c.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
      videoPath = try c.decodeIfPresent(String.self, forKey: .videoPath)
      startFrame = try c.decodeIfPresent(Int.self, forKey: .startFrame) ?? 0
      lengthFrames = try c.decodeIfPresent(Int.self, forKey: .lengthFrames) ?? 0
      strength = try c.decodeIfPresent(Float.self, forKey: .strength) ?? 1.0
      prompt = try c.decodeIfPresent(String.self, forKey: .prompt)
    }
  }

  public var version: Int
  public var settings: Settings
  /// Required, non-empty after trim. Conditions every chunk.
  public var globalPrompt: String
  public var keyframes: [Keyframe]
  public var promptSegments: [PromptSegment]
  /// Only honoured when `audio.mode == .imported`.
  public var audioClips: [AudioClip]
  public var audio: AudioSettings
  public var referenceClips: [ReferenceClip]
  public var retake: Retake

  public init(
    version: Int = DirectorTimeline.currentVersion,
    settings: Settings,
    globalPrompt: String,
    keyframes: [Keyframe] = [],
    promptSegments: [PromptSegment] = [],
    audioClips: [AudioClip] = [],
    audio: AudioSettings = AudioSettings(),
    referenceClips: [ReferenceClip] = [],
    retake: Retake = Retake()
  ) {
    self.version = version
    self.settings = settings
    self.globalPrompt = globalPrompt
    self.keyframes = keyframes
    self.promptSegments = promptSegments
    self.audioClips = audioClips
    self.audio = audio
    self.referenceClips = referenceClips
    self.retake = retake
  }

  public init(from decoder: Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    let v = try c.decodeIfPresent(Int.self, forKey: .version) ?? DirectorTimeline.currentVersion
    guard v >= 1, v <= DirectorTimeline.currentVersion else {
      throw DirectorError.unsupportedVersion(v)
    }
    version = v
    settings = try c.decode(Settings.self, forKey: .settings)
    globalPrompt = try c.decodeIfPresent(String.self, forKey: .globalPrompt) ?? ""
    keyframes = try c.decodeIfPresent([Keyframe].self, forKey: .keyframes) ?? []
    promptSegments = try c.decodeIfPresent([PromptSegment].self, forKey: .promptSegments) ?? []
    audioClips = try c.decodeIfPresent([AudioClip].self, forKey: .audioClips) ?? []
    audio = try c.decodeIfPresent(AudioSettings.self, forKey: .audio) ?? AudioSettings()
    referenceClips = try c.decodeIfPresent([ReferenceClip].self, forKey: .referenceClips) ?? []
    retake = try c.decodeIfPresent(Retake.self, forKey: .retake) ?? Retake()
  }
}

// MARK: - Issues

/// One validation finding. `ids` name the timeline elements involved so the
/// desktop can select them on the canvas.
public struct DirectorIssue: Codable, Sendable, Equatable {
  public enum Severity: String, Codable, Sendable, Equatable {
    case error
    case warning
  }

  public var severity: Severity
  /// snake_case code from the route contract (e.g. `keyframe_off_grid`).
  public var code: String
  public var message: String
  public var ids: [String]

  public init(severity: Severity, code: String, message: String, ids: [String] = []) {
    self.severity = severity
    self.code = code
    self.message = message
    self.ids = ids
  }

  public static func error(_ code: String, _ message: String, ids: [String] = []) -> DirectorIssue {
    DirectorIssue(severity: .error, code: code, message: message, ids: ids)
  }

  public static func warning(_ code: String, _ message: String, ids: [String] = []) -> DirectorIssue {
    DirectorIssue(severity: .warning, code: code, message: message, ids: ids)
  }
}

// MARK: - Errors

public enum DirectorError: Error, LocalizedError, CustomStringConvertible, Equatable {
  /// The timeline failed validation; the issues carry every error and warning.
  case invalid([DirectorIssue])
  /// `version` newer than this build understands.
  case unsupportedVersion(Int)
  /// A `.cbdirector` file could not be read or parsed.
  case fileUnreadable(String)
  /// Upstream `timeline_data` could not be mapped.
  case importFailed(String)
  /// Chunk `chunk` (0-based) failed at `stage` (render|lastframe|audio|stitch|fps).
  case chunkFailed(chunk: Int, stage: String, message: String)
  /// The final concat/mux failed.
  case stitchFailed(String)

  public var description: String {
    switch self {
    case .invalid(let issues):
      let errors = issues.filter { $0.severity == .error }
      let codes = errors.map(\.code).joined(separator: ", ")
      return "timeline invalid: \(codes.isEmpty ? "no errors reported" : codes)"
    case .unsupportedVersion(let v):
      return "unsupported timeline version \(v) (this build reads version \(DirectorTimeline.currentVersion))"
    case .fileUnreadable(let m):
      return "timeline file unreadable: \(m)"
    case .importFailed(let m):
      return "timeline import failed: \(m)"
    case .chunkFailed(let chunk, let stage, let message):
      return "chunk \(chunk + 1) (\(stage)): \(message)"
    case .stitchFailed(let m):
      return "stitch failed: \(m)"
    }
  }

  public var errorDescription: String? { description }
}

// MARK: - Plan

/// The compiled chunk plan: returned by /validate, on the 202, on every
/// status poll and stored in the trace payload as `director_plan`. A pure
/// record — WP1 fills the chunk skeleton (index/start/end/frames/seed/
/// carry_over/audio), keyframe_ticks and boundary_frames; the compiler (WP2b)
/// fills per-chunk keyframes, prompt_segments and beat_schedule.
public struct DirectorPlan: Codable, Sendable, Equatable {

  public struct BeatEntry: Codable, Sendable, Equatable {
    public var text: String
    public var startFrac: Float
    public var endFrac: Float

    public init(text: String, startFrac: Float, endFrac: Float) {
      self.text = text
      self.startFrac = startFrac
      self.endFrac = endFrac
    }
  }

  public struct ChunkKeyframe: Codable, Sendable, Equatable {
    public var id: String
    public var localFrame: Int
    public var strength: Float

    public init(id: String, localFrame: Int, strength: Float) {
      self.id = id
      self.localFrame = localFrame
      self.strength = strength
    }
  }

  public struct Chunk: Codable, Sendable, Equatable {
    public var index: Int
    public var startFrame: Int
    /// Inclusive; shared with chunk index+1's start.
    public var endFrame: Int
    public var frames: Int
    /// `settings.seed + index`; nil (omitted) when the timeline names no
    /// seed and the plan was built before the server resolved one (/validate).
    public var seed: UInt64?
    /// true for every chunk after the first: frame 0 is the previous chunk's
    /// rendered last frame, not a user image.
    public var carryOver: Bool
    /// User keyframes conditioning THIS chunk (a boundary keyframe is listed
    /// on the chunk it ends only).
    public var keyframes: [ChunkKeyframe]
    /// Ids of prompt segments overlapping this chunk, in time order.
    public var promptSegments: [String]
    public var beatSchedule: [BeatEntry]
    /// "generated" | "none".
    public var audio: String

    public init(
      index: Int, startFrame: Int, endFrame: Int, frames: Int, seed: UInt64?, carryOver: Bool,
      keyframes: [ChunkKeyframe] = [], promptSegments: [String] = [],
      beatSchedule: [BeatEntry] = [], audio: String
    ) {
      self.index = index
      self.startFrame = startFrame
      self.endFrame = endFrame
      self.frames = frames
      self.seed = seed
      self.carryOver = carryOver
      self.keyframes = keyframes
      self.promptSegments = promptSegments
      self.beatSchedule = beatSchedule
      self.audio = audio
    }
  }

  public struct KeyframeTick: Codable, Sendable, Equatable {
    public var id: String
    public var frame: Int

    public init(id: String, frame: Int) {
      self.id = id
      self.frame = frame
    }
  }

  public var lengthFrames: Int
  /// `settings.fps`, or `Settings.defaultFps` when the timeline names none
  /// and the plan was built before the server resolved it.
  public var fps: Int
  public var width: Int
  public var height: Int
  /// "generated" | "imported".
  public var audioMode: String
  public var chunks: [Chunk]
  public var keyframeTicks: [KeyframeTick]
  /// start_k for k > 0 (desktop overlay ticks).
  public var boundaryFrames: [Int]
  public var warnings: [DirectorIssue]

  public init(
    lengthFrames: Int, fps: Int, width: Int, height: Int, audioMode: String,
    chunks: [Chunk], keyframeTicks: [KeyframeTick], boundaryFrames: [Int],
    warnings: [DirectorIssue]
  ) {
    self.lengthFrames = lengthFrames
    self.fps = fps
    self.width = width
    self.height = height
    self.audioMode = audioMode
    self.chunks = chunks
    self.keyframeTicks = keyframeTicks
    self.boundaryFrames = boundaryFrames
    self.warnings = warnings
  }
}

// MARK: - Audio probe (validator input; WP2d supplies the AVFoundation impl)

/// Cheap metadata about an audio file: duration and whether an audio track
/// exists. Never a full decode.
public struct AudioProbe: Sendable, Equatable {
  public var durationSeconds: Double
  public var hasAudioTrack: Bool
  public var sampleRate: Int?
  public var channels: Int?

  public init(durationSeconds: Double, hasAudioTrack: Bool, sampleRate: Int? = nil, channels: Int? = nil) {
    self.durationSeconds = durationSeconds
    self.hasAudioTrack = hasAudioTrack
    self.sampleRate = sampleRate
    self.channels = channels
  }
}

// MARK: - JSON

/// The one encoder/decoder pair for timelines, issues and plans. The file
/// format and the wire format are the same bytes: snake_case keys, sorted,
/// pretty-printed on disk.
public enum DirectorJSON {
  public static func encoder(pretty: Bool = true) -> JSONEncoder {
    let e = JSONEncoder()
    e.keyEncodingStrategy = .convertToSnakeCase
    e.outputFormatting = pretty ? [.sortedKeys, .prettyPrinted] : [.sortedKeys]
    return e
  }

  public static func decoder() -> JSONDecoder {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return d
  }
}

// MARK: - Compilation (WP2b)

/// One rendered chunk as the compiler emits it: a JSON-ready
/// /v1/video/generate body (single pass, no carry-over) plus the names the
/// orchestrator uses for its intermediates.
///
/// `body` holds JSON scalars, strings, arrays and dictionaries only (it is
/// built for `JSONSerialization` and never mutated after construction), which
/// is why the struct is `@unchecked Sendable`: `[String: Any]` cannot state
/// that itself.
public struct CompiledChunk: @unchecked Sendable {
  public let index: Int
  public let span: DirectorMath.ChunkSpan
  /// The compiled request body WITHOUT the carry-over keyframe — chunk k > 0
  /// gets it from `bodyWithCarryOver(imagePath:)` once the orchestrator has
  /// extracted chunk k-1's last frame.
  public let body: [String: Any]
  /// k-1 for every chunk after the first (the orchestrator supplies the
  /// rendered last frame of that chunk as this chunk's frame-0 condition).
  public let carryOverFromChunk: Int?
  /// "director-<session>-chunk<k>.mp4"
  public let outputName: String
  /// "director-<session>-chunk<k>-lastframe.png"
  public let lastFrameName: String
  /// Whether the body asks the engine for generated audio.
  public let wantsAudio: Bool

  public init(
    index: Int, span: DirectorMath.ChunkSpan, body: [String: Any], carryOverFromChunk: Int?,
    outputName: String, lastFrameName: String, wantsAudio: Bool
  ) {
    self.index = index
    self.span = span
    self.body = body
    self.carryOverFromChunk = carryOverFromChunk
    self.outputName = outputName
    self.lastFrameName = lastFrameName
    self.wantsAudio = wantsAudio
  }

  /// The body with `{image_path, frame: 0, strength: 1.0}` prepended to
  /// `keyframes` (creating the array when absent). Pure: the receiver is
  /// unchanged. WP2a's mutual exclusion means the carry-over never goes
  /// through `image_path`/`strength`.
  public func bodyWithCarryOver(imagePath: String) -> [String: Any] {
    var out = body
    var keyframes = out["keyframes"] as? [[String: Any]] ?? []
    keyframes.insert(["image_path": imagePath, "frame": 0, "strength": 1.0], at: 0)
    out["keyframes"] = keyframes
    return out
  }
}

/// The compiler's output: the plan every client sees plus one body per chunk.
public struct DirectorCompilation: Sendable {
  public let plan: DirectorPlan
  public let chunks: [CompiledChunk]

  public init(plan: DirectorPlan, chunks: [CompiledChunk]) {
    self.plan = plan
    self.chunks = chunks
  }
}
