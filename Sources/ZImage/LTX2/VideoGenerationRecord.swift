// VideoGenerationRecord.swift — the video twin of the PNG generation record
// (comfybox#401).
//
// PNGs embed a full provenance record in EXIF `UserComment` JSON
// (`QwenImageIO.ImageMetadata.generation`, `Sources/ZImage/Util/ImageIO.swift`).
// An .mp4 has no equivalent: Kira's i2v enrichment and the daemon's provenance
// recovery can read a PNG back but get nothing from a clip. This is the same
// record shape, reused rather than reinvented (ticket ruling 1): `prompt`,
// `seed`, `loras`, `steps`, `model` — plus the video-only facts a clip carries
// that a still doesn't (`frames`, `fps`, resolved dims, `refine`/`two_pass`,
// `audio`).
//
// WIRE SHAPE: snake_case via `.convertToSnakeCase`/`.convertFromSnakeCase` —
// the same convention `RenderRecipe` documents (properties stay camelCase,
// Codable is synthesized, the encoder strategy does the renaming). One
// consequence is load-bearing: `Sources/ComfyBoxDesktop/DAM/AssetIngestor.swift`
// already reads a `.json` sidecar next to ANY media file (image or video) at
// `readSidecar(for:)` and expects `prompt` / `seed` / `steps` / `model` /
// `loras[].name` / `loras[].scale` — exactly these keys. Writing this sidecar
// makes the gallery/DAM ingest path (ruling 3) work with no changes there.
//
// `dimensionReason` mirrors the `dimension_reason` field #405/#408
// (`VideoDimensionResolver`) adds to the async render trace and the
// `/v1/video/generate` response. It was left `nil` here when this ticket
// landed because that work was still in flight; #408 wired it up:
// `WarmServer.buildLocalVideoRequest` now sets `LTX2VideoRequest
// .dimensionReason` from the resolver's `reason.rawValue`, so the sidecar
// records WHY a clip has the shape it has — `"source_aspect"` when the i2v
// source image decided it, `"explicit"` when the caller did, `"default"` for
// the fall-through. Still optional: a caller that supplies no reason records
// none.

import Foundation
import Logging

/// One provenance record for one written .mp4 — the sidecar's exact content,
/// and (optionally) the mp4's own metadata atom's content.
public struct VideoGenerationRecord: Codable, Sendable, Equatable {
  public let prompt: String
  public let negativePrompt: String?
  /// nil only for an aggregate record (the storyboard assembly, which has no
  /// single seed) — every per-shot/per-render record carries one.
  public let seed: UInt64?
  public let steps: Int?
  /// Caller-visible step request, retained separately when a resolved sigma
  /// schedule determines a different number of executed denoise steps.
  public let requestedSteps: Int?
  /// CFG guidance — LTX's primary motion lever (>1 amplifies the action
  /// direction). Same field/key `ImageMetadata.generation`'s `guidance`
  /// carries on the PNG side.
  public let guidance: Float?
  /// Physical model file basename (no directory, no extension) — mirrors
  /// `ImageMetadata.generation`'s `model` field.
  public let model: String

  /// Requested budget, not necessarily what was encoded — see `resolvedWidth`/
  /// `resolvedHeight`.
  public let width: Int
  public let height: Int

  public let frames: Int
  public let fps: Int
  /// Actual encoded pixel dimensions. With two-stage refine on, this is the
  /// 2x-refined size, not `width`/`height` (`LTX2VideoGenerator.render` uses
  /// the decoded frame dims for exactly this reason — see its comment above
  /// `writeMP4`).
  public let resolvedWidth: Int
  public let resolvedHeight: Int
  /// `"source_aspect" | "explicit" | "default"` — why the render has the
  /// shape it has (`VideoDimensionResolver`, comfybox#405). `nil` only when
  /// the caller supplied no reason.
  public let dimensionReason: String?

  /// Whether `two_stage` (the 1.5x/2x HQ refine pass) was requested for this
  /// render.
  public let twoPass: Bool
  /// Whether the refine pass actually ran. `false` when `twoPass` is `false`
  /// (never requested) OR when it was requested but skipped — see
  /// `refineSkippedReason` for which.
  public let refine: Bool
  /// Non-nil only when `twoPass` was true and the refine could not run
  /// (`LTX2RefineGate`) — mirrors `VideoJobStatus.refineSkipped`.
  public let refineSkippedReason: String?
  /// Synchronized audio track present in this render.
  public let audio: Bool
  /// Resolved execution recipe. Optional so sidecars written before these
  /// fields existed remain decodable.
  public let sampler: String?
  public let stage1Sigmas: [Float]?
  public let refineSigmas: [Float]?
  public let nagScale: Float?
  public let nagAlpha: Float?
  public let nagTau: Float?
  public let nagApplied: Bool?
  public let audioRefine: Bool?
  /// `"t2v" | "i2v" | "extend" | "storyboard"` — how this clip was produced.
  /// `"extend"` is an i2v render whose request asked for more than one chunk
  /// (`extendToSeconds > 0`); a plain i2v single chunk stays `"i2v"`.
  public let kind: String
  /// Which app/persona generated it — same field `ImageMetadata.generation`
  /// writes under the `source` key (parameter name `generatedBy` there;
  /// named `source` here to land on the identical wire key).
  public let source: String?
  /// Content mode (neutral/apple/banana/avocado) — same key
  /// `ImageMetadata.generation`'s `content_mode` carries.
  public let contentMode: String?
  /// `true` only on the mp4 atom's copy of a record that exceeded
  /// `VideoGenerationRecord.atomSizeCap` (ruling 5) — the sidecar this
  /// field's record sits next to is ALWAYS the full, untruncated record;
  /// this flag only ever appears embedded in the container header.
  public let truncated: Bool

  public let loras: [LoRAEntry]

  public struct LoRAEntry: Codable, Sendable, Equatable {
    public let name: String
    public let scale: Float
    public init(name: String, scale: Float) {
      self.name = name
      self.scale = scale
    }
  }

  public init(
    prompt: String, negativePrompt: String? = nil, seed: UInt64? = nil, steps: Int? = nil,
    requestedSteps: Int? = nil,
    guidance: Float? = nil,
    model: String, width: Int, height: Int, frames: Int, fps: Int,
    resolvedWidth: Int, resolvedHeight: Int, dimensionReason: String? = nil,
    twoPass: Bool, refine: Bool, refineSkippedReason: String? = nil, audio: Bool,
    sampler: String? = nil, stage1Sigmas: [Float]? = nil, refineSigmas: [Float]? = nil,
    nagScale: Float? = nil, nagAlpha: Float? = nil, nagTau: Float? = nil,
    nagApplied: Bool? = nil, audioRefine: Bool? = nil,
    kind: String, source: String? = nil, contentMode: String? = nil, truncated: Bool = false,
    loras: [LoRAEntry] = []
  ) {
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.seed = seed
    self.steps = steps
    self.requestedSteps = requestedSteps
    self.guidance = guidance
    self.model = model
    self.width = width
    self.height = height
    self.frames = frames
    self.fps = fps
    self.resolvedWidth = resolvedWidth
    self.resolvedHeight = resolvedHeight
    self.dimensionReason = dimensionReason
    self.twoPass = twoPass
    self.refine = refine
    self.refineSkippedReason = refineSkippedReason
    self.audio = audio
    self.sampler = sampler
    self.stage1Sigmas = stage1Sigmas
    self.refineSigmas = refineSigmas
    self.nagScale = nagScale
    self.nagAlpha = nagAlpha
    self.nagTau = nagTau
    self.nagApplied = nagApplied
    self.audioRefine = audioRefine
    self.kind = kind
    self.source = source
    self.contentMode = contentMode
    self.truncated = truncated
    self.loras = loras
  }
}

// MARK: - Building (pure — testable without weights)

extension VideoGenerationRecord {
  /// A file's basename with its extension stripped — the same normalisation
  /// `ImageMetadata.generation`'s `model`/`loras[].name` use.
  public static func basename(_ path: String) -> String {
    (((path as NSString).lastPathComponent) as NSString).deletingPathExtension
  }

  /// Classify what kind of render produced this clip, from the request alone.
  public static func kind(initImagePath: String?, extendToSeconds: Float) -> String {
    guard initImagePath != nil else { return "t2v" }
    return extendToSeconds > 0 ? "extend" : "i2v"
  }

  /// Build the record for one `LTX2VideoGenerator.render()` output. Pure —
  /// takes only value types, no pipeline/model access — so it's testable
  /// without weights.
  ///
  /// `dimensionReason` comes from `request.dimensionReason`, which
  /// `WarmServer.buildLocalVideoRequest` sets from the resolver's
  /// `reason.rawValue` (comfybox#405). The resolver runs a layer up, in
  /// `prepareLocalVideo`, which is why the value travels ON the request
  /// rather than being recomputed at this generator-level `render()`.
  ///
  /// `configGuidance` (review round 3, ruling 1): `request.guidance` is only
  /// an OVERRIDE — `pipeline.generateT2V`/`generateI2V` resolve the CFG scale
  /// they actually run as `guidance ?? config.guidance`
  /// (`LTX2Pipeline.swift`), and `LTX2PipelineConfig.guidance` is never nil
  /// (3.5 dev / 1.0 distilled by default). Recording `request.guidance`
  /// alone wrote `guidance: null` for the common case — every render that
  /// took the config default. The caller (`LTX2VideoGenerator.render`)
  /// passes `pipeline.config.guidance`; the preset fallback
  /// (`request override, else preset guidance, else pipeline.config.guidance`)
  /// happens one layer up, in `WarmServer.buildLocalVideoRequest`, which now
  /// bakes `videoPreset?.guidance` into `request.guidance` itself (the same
  /// pattern `steps`/`seed` already used) — so by the time it reaches here,
  /// `request.guidance` already reflects "request override, else preset".
  public static func build(
    request: LTX2VideoRequest,
    transformerFile: String,
    frameCount: Int,
    resolvedWidth: Int,
    resolvedHeight: Int,
    twoStageRequested: Bool,
    refineSkippedReason: String?,
    audioWritten: Bool,
    configGuidance: Float,
    actualSteps: Int? = nil,
    sampler: String? = nil,
    stage1Sigmas: [Float]? = nil,
    refineSigmas: [Float]? = nil,
    nagConfig: LTX2NAGConfig? = nil,
    nagApplied: Bool? = nil,
    audioRefine: Bool? = nil
  ) -> VideoGenerationRecord {
    VideoGenerationRecord(
      prompt: request.prompt,
      negativePrompt: request.negativePrompt,
      seed: request.seed,
      steps: actualSteps ?? request.steps,
      requestedSteps: request.steps,
      guidance: request.guidance ?? configGuidance,
      model: basename(transformerFile),
      width: request.width,
      height: request.height,
      frames: frameCount,
      fps: request.fps,
      resolvedWidth: resolvedWidth,
      resolvedHeight: resolvedHeight,
      dimensionReason: request.dimensionReason,
      twoPass: twoStageRequested,
      refine: twoStageRequested && refineSkippedReason == nil,
      refineSkippedReason: refineSkippedReason,
      audio: audioWritten,
      sampler: sampler,
      stage1Sigmas: stage1Sigmas,
      refineSigmas: refineSigmas,
      nagScale: nagConfig?.scale,
      nagAlpha: nagConfig?.alpha,
      nagTau: nagConfig?.tau,
      nagApplied: nagApplied,
      audioRefine: audioRefine,
      kind: kind(initImagePath: request.initImagePath, extendToSeconds: request.extendToSeconds),
      source: request.source,
      contentMode: request.contentMode,
      loras: request.effectiveLoRAs.map { LoRAEntry(name: basename($0.path), scale: $0.scale) }
    )
  }
}

// MARK: - JSON (encode/decode round trip)

extension VideoGenerationRecord {
  /// `.sortedKeys` for the same reason `ImageMetadata.generation` uses it on
  /// the PNG side (WP-E10): deterministic bytes for the same record, so the
  /// sidecar and the mp4 atom carry byte-identical JSON and a whole-file hash
  /// is comparable across runs.
  public func encodeJSON() throws -> Data {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    encoder.outputFormatting = [.sortedKeys]
    return try encoder.encode(self)
  }

  public static func decodeJSON(_ data: Data) throws -> VideoGenerationRecord {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(VideoGenerationRecord.self, from: data)
  }

  /// Review round 2, ruling 5: the mp4 metadata atom lives in the container
  /// header — an unbounded record there is a header an unrelated tool has to
  /// buffer whole just to seek the file. The sidecar has no such constraint
  /// and always carries the full record; only the atom is capped.
  public static let atomSizeCap = 64 * 1024

  /// JSON for the mp4 atom: the full record when it fits under
  /// `atomSizeCap`, otherwise a minimal record — prompt clipped to 200
  /// characters, every other optional field dropped — with `truncated: true`
  /// so a reader of the atom alone can tell it isn't the whole story (the
  /// sidecar next to the file is). Never throws; `nil` only if even the
  /// truncated record can't encode.
  public func atomJSON() -> Data? {
    guard let full = try? encodeJSON() else { return nil }
    guard full.count > Self.atomSizeCap else { return full }
    let clipped = VideoGenerationRecord(
      prompt: String(prompt.prefix(200)), seed: seed, steps: steps, model: model,
      width: width, height: height, frames: frames, fps: fps,
      resolvedWidth: resolvedWidth, resolvedHeight: resolvedHeight,
      twoPass: twoPass, refine: refine, audio: audio, kind: kind, truncated: true)
    return try? clipped.encodeJSON()
  }

  /// Convenience for a `writeMP4(generationRecordJSON:)` call site — the
  /// atom's bytes as UTF-8, or `nil` if neither the full nor the clipped
  /// record could encode.
  public var atomJSONString: String? {
    atomJSON().flatMap { String(data: $0, encoding: .utf8) }
  }
}

// MARK: - Sidecar (mandatory sink — ruling 2)

/// Writes/locates the `.json` sidecar next to a media file, using the exact
/// convention the desktop editor's image sidecars already use
/// (`EditSidecar.sidecarPath(forImageAt:)`,
/// `Sources/ComfyBoxDesktop/Edit/EditSidecar.swift`) and the DAM ingestor
/// already reads for video (`AssetIngestor.readSidecar`,
/// `Sources/ComfyBoxDesktop/DAM/AssetIngestor.swift`): `<basename>.json`,
/// same directory, extension stripped and replaced.
public enum VideoSidecar {
  private static let logger = Logger(label: "z-image.video-metadata")

  public static func path(forMediaAt mediaPath: String) -> String {
    ((mediaPath as NSString).deletingPathExtension) + ".json"
  }

  /// Best-effort, atomic write. Never throws: a render that produced a real
  /// clip must not fail because the sidecar couldn't be written (same
  /// contract as the PNG side's `appliedRecordNotEncodable` — the media file
  /// is the primary artifact). Returns whether the write succeeded so the
  /// caller can log with its own render-scoped context.
  @discardableResult
  public static func write(_ record: VideoGenerationRecord, forMediaAt mediaPath: String) -> Bool {
    do {
      let data = try record.encodeJSON()
      try data.write(to: URL(fileURLWithPath: path(forMediaAt: mediaPath)), options: .atomic)
      return true
    } catch {
      logger.error(
        "comfybox#401: could not write the generation-record sidecar for \(mediaPath) (\(error)) — the clip itself is still written.")
      return false
    }
  }

  public static func read(forMediaAt mediaPath: String) -> VideoGenerationRecord? {
    guard let data = FileManager.default.contents(atPath: path(forMediaAt: mediaPath)) else { return nil }
    return try? VideoGenerationRecord.decodeJSON(data)
  }
}
