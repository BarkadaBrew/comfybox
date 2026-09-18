// DirectorCompiler.swift — validated timeline -> per-chunk render bodies + plan.
//
// Pure and deterministic: the same snapped timeline, session and source
// always produce byte-identical bodies (after sorted-keys serialization) and
// an equal DirectorPlan. No I/O, no engine dependencies — the bodies are the
// generic /v1/video/generate wire shape (WP2a `keyframes[]`, `beat_schedule`)
// and WP2c feeds each one through `prepareLocalVideo` exactly like the
// storyboard does per shot.
//
// Chunking follows DirectorMath.chunkLayout: every chunk is a single pass of
// at most 289 frames, so `extend_to_seconds: 0` is pinned explicitly (a preset
// carrying extend/duration must not turn a chunk into a continuation render,
// which would strip its beat schedule) and `identity_anchor_strength: 0` keeps
// the engine's extended-render anchor from colliding with an end keyframe.
//
// Boundary ownership (Phase 1 delta from FDD §4.2 rule 1): a keyframe whose
// frame equals start_k (k > 0) conditions chunk k-1's LAST frame only. Chunk k
// conditions its frame 0 on the rendered carry-over frame the orchestrator
// extracts from chunk k-1's mp4, never on the user image — the engine's
// continuation loop has the same shape, and the validator warns
// `keyframe_on_chunk_boundary` so the desktop can show it.

import Foundation

public enum DirectorCompiler {
  /// Tuning sent on every chunk whose prompt changes inside the chunk.
  /// Colour anchor only: margin 0 sharpened a lighting switch (R5) but made an
  /// action switch later on R3 (wave +4.0 s vs +2.8 s at margin 2), so the
  /// beat window margin stays at the engine default.
  static let promptSwitchTuning: [String: Any] = ["color_anchor": 0.0]


  /// Compile a validation result. Throws `DirectorError.invalid` when the
  /// validation failed, or when a keyframe still has no `image_path` (the
  /// route materializes `image_base64` to a file BEFORE compiling; the
  /// compiler never guesses a path).
  public static func compile(
    _ validated: DirectorValidation, session: String, source: String
  ) throws -> DirectorCompilation {
    guard validated.ok else { throw DirectorError.invalid(validated.issues) }
    let snapped = validated.snapped
    let plan = validated.plan ?? self.plan(for: snapped, warnings: validated.issues.filter { $0.severity == .warning })
    let chunks = try bodies(for: snapped, plan: plan, session: session, source: source)
    return DirectorCompilation(plan: plan, chunks: chunks)
  }

  // MARK: - Plan

  /// The full plan for a snapped timeline: chunk spans, seeds, carry-over,
  /// per-chunk keyframes / prompt segment ids / beat schedule / audio flag,
  /// keyframe ticks, boundary frames and the validator's warnings.
  /// `layout` lets the VALIDATOR hand in boundaries it already moved into
  /// pauses (WP11) — it is the only caller that can read the voice. Omitted,
  /// the layout is the plain arithmetic one, which is what a caller with no
  /// audio access should get.
  public static func plan(
    for snapped: DirectorTimeline, warnings: [DirectorIssue] = [],
    layout providedLayout: [DirectorMath.ChunkSpan]? = nil,
    joinsInPauses: Bool = false
  ) -> DirectorPlan {
    let layout = providedLayout ?? DirectorMath.chunkLayout(
      lengthFrames: snapped.settings.lengthFrames, maxFrames: snapped.chunkCeilingFrames)
    let generated = snapped.audio.mode == .generated
    let chunks = layout.map { span -> DirectorPlan.Chunk in
      let cond = conditioning(for: snapped, span: span)
      return DirectorPlan.Chunk(
        index: span.index, startFrame: span.startFrame, endFrame: span.endFrame,
        frames: span.frames, seed: seed(for: snapped, chunk: span.index),
        carryOver: span.index > 0,
        keyframes: cond.keyframes.map { .init(id: $0.keyframe.id, localFrame: $0.localFrame, strength: $0.keyframe.strength) },
        promptSegments: cond.segments.map(\.segment.id),
        beatSchedule: cond.segments.map { .init(text: $0.segment.prompt, startFrac: $0.startFrac, endFrac: $0.endFrac) },
        audio: generated ? "generated" : "none")
    }
    return DirectorPlan(
      lengthFrames: snapped.settings.lengthFrames,
      fps: snapped.settings.fps ?? DirectorTimeline.Settings.defaultFps,
      width: snapped.settings.width, height: snapped.settings.height,
      audioMode: snapped.audio.mode.rawValue,
      chunks: chunks,
      keyframeTicks: placedKeyframes(snapped).map { .init(id: $0.id, frame: $0.frame) },
      boundaryFrames: layout.dropFirst().map(\.startFrame),
      joinsInPauses: joinsInPauses,
      warnings: warnings.filter { $0.severity == .warning })
  }

  // MARK: - Bodies

  /// One single-pass /v1/video/generate body per chunk of `plan`.
  public static func bodies(
    for snapped: DirectorTimeline, plan: DirectorPlan, session: String, source: String
  ) throws -> [CompiledChunk] {
    let unmaterialized = snapped.keyframes.filter { ($0.imagePath ?? "").isEmpty }
    guard unmaterialized.isEmpty else {
      throw DirectorError.invalid([
        .error("keyframe_image_missing",
               "keyframes \(unmaterialized.map(\.id).joined(separator: ", ")) have no image_path (image_base64 must be materialized before compiling)",
               ids: unmaterialized.map(\.id))
      ])
    }
    let settings = snapped.settings
    let generated = snapped.audio.mode == .generated
    // The PLAN owns the spans. Recomputing them would silently discard
    // boundaries moved into pauses (WP11): the plan would describe one set of
    // joins and the rendered bodies another, a disagreement that shows up
    // only as a bad clip.
    let layout = plan.chunks.map {
      DirectorMath.ChunkSpan(index: $0.index, startFrame: $0.startFrame, frames: $0.frames)
    }
    precondition(!layout.isEmpty, "a plan must carry at least one chunk")

    return layout.map { span in
      let cond = conditioning(for: snapped, span: span)
      let outputName = "director-\(session)-chunk\(span.index).mp4"
      let lastFrameName = "director-\(session)-chunk\(span.index)-lastframe.png"

      var prompt = snapped.globalPrompt
      if !cond.segments.isEmpty {
        prompt += "\n" + cond.segments.map(\.segment.prompt).joined(separator: "\n")
      }

      var body: [String: Any] = [
        "prompt": prompt,
        "width": settings.width,
        "height": settings.height,
        "frames": span.frames,
        // Explicit single pass: wins over any preset extend/duration so the
        // chunk never becomes a continuation render (which would strip
        // beat_schedule and re-anchor at 0.5).
        "extend_to_seconds": 0.0,
        "identity_anchor_strength": 0.0,
        "audio": generated,
        // Enhancement rewrites the prompt and the beats would fail to locate.
        "enhance": false,
        // No character => never let a keyframe-less chunk 0 (T2V) silently
        // become the server's default character.
        "skip_character_injection": settings.character == nil,
        "output_path": outputName,
        "source": source,
      ]
      // fps/seed: nil => omitted, so the preset/config value applies
      // (request > preset > config > builtin), like steps/negative below.
      if let fps = settings.fps { body["fps"] = fps }
      if let seed = seed(for: snapped, chunk: span.index) { body["seed"] = seed }
      if let negative = settings.negativePrompt { body["negative_prompt"] = negative }
      if let steps = settings.steps { body["steps"] = steps }
      if let preset = settings.preset { body["preset"] = preset }
      if !settings.loras.isEmpty {
        body["loras"] = settings.loras.map { lora -> [String: Any] in
          var entry: [String: Any] = ["path": lora.path, "scale": jsonNumber(lora.scale)]
          if let role = lora.role { entry["role"] = role }
          return entry
        }
      }
      if let character = settings.character { body["character"] = character }
      if !cond.segments.isEmpty {
        body["beat_schedule"] = cond.segments.map { seg -> [String: Any] in
          ["text": seg.segment.prompt, "start_frac": jsonNumber(seg.startFrac), "end_frac": jsonNumber(seg.endFrac)]
        }
      }
      // A chunk that switches prompts mid-render turns the colour anchor off
      // (ladder R5, 2026-09-17): the anchor renormalises every frame to frame
      // 0 and erased a red-to-blue lighting change entirely. Exposure at
      // chunk seams is DirectorToneMatch's job. Request tuning outranks preset
      // and config, so this applies only to these chunks.
      if cond.segments.count >= 2 {
        body["tuning"] = Self.promptSwitchTuning
      }
      // AUDIO-DRIVEN (FDD §4.7, WP11): a clip marked `drives_video` conditions
      // every chunk it covers on ITS OWN SLICE of the voice, so lips follow
      // real words. The chunk renders its audio stream (that is what the video
      // attends to); the stitcher still muxes the untouched master track, so
      // splices cannot click or drift.
      if let driving = snapped.audioClips.first(where: \.drivesVideo),
         span.startFrame < driving.startFrame + driving.lengthFrames,
         span.startFrame + span.frames > driving.startFrame {
        body["audio"] = true
        body["audio_condition_path"] = driving.audioPath
        // Where this chunk starts inside the FILE: the timeline offset plus
        // whatever the clip trimmed off its head.
        body["audio_condition_start_frame"] =
          max(0, span.startFrame - driving.startFrame) + driving.trimStartFrames
      }
      if !cond.keyframes.isEmpty {
        body["keyframes"] = cond.keyframes.map { kf -> [String: Any] in
          ["image_path": kf.keyframe.imagePath ?? "", "frame": kf.localFrame, "strength": jsonNumber(kf.keyframe.strength)]
        }
      }

      return CompiledChunk(
        index: span.index, span: span, body: body,
        carryOverFromChunk: span.index > 0 ? span.index - 1 : nil,
        outputName: outputName, lastFrameName: lastFrameName,
        wantsAudio: generated)
    }
  }

  // MARK: - Shared per-chunk conditioning

  struct PlacedKeyframe {
    let keyframe: DirectorTimeline.Keyframe
    let localFrame: Int
  }

  struct PlacedSegment {
    let segment: DirectorTimeline.PromptSegment
    let startFrac: Float
    let endFrac: Float
  }

  struct ChunkConditioning {
    /// User keyframes this chunk conditions on, sorted by local frame
    /// (frame-0 entry first when present).
    let keyframes: [PlacedKeyframe]
    /// Prompt segments overlapping [start_k, end_k + 1) in start order, each
    /// with its fractions of THIS chunk's duration.
    let segments: [PlacedSegment]
  }

  /// Keyframes with their effective timeline frame (end-frame keyframes pin
  /// to length-1 regardless of `frame`), in timeline order (stable on id).
  static func placedKeyframes(_ snapped: DirectorTimeline) -> [(id: String, frame: Int, keyframe: DirectorTimeline.Keyframe)] {
    let last = snapped.settings.lengthFrames - 1
    return snapped.keyframes
      .map { (id: $0.id, frame: $0.isEndFrame ? last : $0.frame, keyframe: $0) }
      .sorted { ($0.frame, $0.id) < ($1.frame, $1.id) }
  }

  static func conditioning(for snapped: DirectorTimeline, span: DirectorMath.ChunkSpan) -> ChunkConditioning {
    let keyframes = placedKeyframes(snapped)
      .filter { DirectorMath.chunkOwnsKeyframe(span, frame: $0.frame) }
      .map { PlacedKeyframe(keyframe: $0.keyframe, localFrame: DirectorMath.localFrame(global: $0.frame, chunk: span)) }
      .sorted { $0.localFrame < $1.localFrame }

    let segments = snapped.promptSegments
      .sorted { ($0.startFrame, $0.id) < ($1.startFrame, $1.id) }
      .compactMap { seg -> PlacedSegment? in
        guard let fracs = DirectorMath.beatFractions(
          segmentStart: seg.startFrame, segmentLength: seg.lengthFrames, chunk: span,
          sharesEndFrame: span.endFrame < snapped.settings.lengthFrames - 1)
        else { return nil }
        return PlacedSegment(segment: seg, startFrac: fracs.startFrac, endFrac: fracs.endFrac)
      }
    return ChunkConditioning(keyframes: keyframes, segments: segments)
  }

  static func seed(for snapped: DirectorTimeline, chunk: Int) -> UInt64? {
    snapped.settings.seed.map { $0 &+ UInt64(chunk) }
  }

  /// The timeline with `fps`/`seed` pinned to the values the server resolved
  /// for chunk 0 (preset/config/builtin when the timeline named none).
  /// Recompiling this makes every chunk body, the plan and the sidecar agree
  /// with what actually renders; a timeline that already named both is
  /// returned unchanged.
  public static func resolvingDefaults(
    _ snapped: DirectorTimeline, fps: Int, seed: UInt64
  ) -> DirectorTimeline {
    var out = snapped
    if out.settings.fps == nil { out.settings.fps = fps }
    if out.settings.seed == nil { out.settings.seed = seed }
    return out
  }

  /// Float -> the shortest Double that round-trips the same decimal text, so
  /// 0.8 serializes as `0.8` (not `0.800000011920929`) and decodes back to
  /// the same Float.
  static func jsonNumber(_ value: Float) -> Double {
    Double(String(value)) ?? Double(value)
  }
}
