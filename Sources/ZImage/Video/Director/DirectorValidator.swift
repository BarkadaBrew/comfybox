// DirectorValidator.swift — pure validation of a DirectorTimeline.
//
// Every error/warning code in the route contract is emitted from the timeline
// plus two injected closures (file existence, audio probe), so the validator
// is deterministic and unit-testable with no I/O. The default closures use
// FileManager and, until WP2d supplies the AVFoundation probe, no audio probe
// (the probe-derived codes are then simply not emitted).

import Foundation

/// The validator's result. `snapped` is the timeline the compiler consumes:
/// length snapped up to 1+8k, end-frame keyframes rewritten to length-1,
/// audio clips clipped to the timeline.
public struct DirectorValidation: Sendable, Equatable {
  public let snapped: DirectorTimeline
  public let issues: [DirectorIssue]
  /// false iff any issue has severity error.
  public let ok: Bool
  /// nil when !ok; otherwise the compiler's full plan (`DirectorCompiler.plan`):
  /// chunk spans + seeds + per-chunk keyframes/prompt segments/beats, ticks,
  /// boundary frames and the warnings.
  public let plan: DirectorPlan?

  public init(snapped: DirectorTimeline, issues: [DirectorIssue], ok: Bool, plan: DirectorPlan?) {
    self.snapped = snapped
    self.issues = issues
    self.ok = ok
    self.plan = plan
  }
}

public enum DirectorValidator {

  /// The production audio probe: AVURLAsset metadata only (duration + audio
  /// track presence + rate/channels), never a sample decode — so /validate
  /// stays cheap. Injectable per call for tests; nil (no probe) on platforms
  /// without AVFoundation, where the probe-driven codes are simply not
  /// emitted.
  public static func defaultAudioProbe(_ path: String) -> AudioProbe? {
    #if canImport(AVFoundation) && canImport(CoreGraphics)
    return DirectorAudioIngest.probe(path: path)
    #else
    return nil
    #endif
  }

  public static func validate(
    _ timeline: DirectorTimeline,
    fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) },
    audioProbe: (String) -> AudioProbe? = DirectorValidator.defaultAudioProbe,
    // WP11: pauses in a driving voice, so joins can be placed inside one.
    // Injected for the same reason `fileExists` and `audioProbe` are — a
    // validator test must not need a real wav on disk.
    pauseRuns: (String, Int, Int) -> [Range<Int>] = DirectorAudioIngest.pauseRuns(atPath:fps:frameCount:)
  ) -> DirectorValidation {
    var issues: [DirectorIssue] = []
    var snapped = timeline
    // nil fps => preset/config/builtin, resolved by the server; the builtin
    // stands in for the frame<->seconds arithmetic below.
    let fps = timeline.settings.fps ?? DirectorTimeline.Settings.defaultFps

    // MARK: Structure

    if timeline.globalPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      issues.append(.error("missing_global_prompt", "global_prompt is required and must be non-empty"))
    }
    if fps < 1 || fps > 120 {
      issues.append(.error("invalid_fps", "settings.fps must be 1...120 (got \(fps))"))
    }
    let w = timeline.settings.width, h = timeline.settings.height
    if w <= 0 || h <= 0 || w % 32 != 0 || h % 32 != 0 {
      issues.append(.error("invalid_dimensions", "settings.width/height must be positive multiples of 32 (got \(w)x\(h))"))
    }

    // MARK: Length

    let requested = timeline.settings.lengthFrames
    let length = DirectorMath.snapLengthUp(requested)
    snapped.settings.lengthFrames = length
    if length != requested {
      issues.append(.warning("length_snapped", "length_frames \(requested) snapped up to \(length) (1 + 8k)"))
    }
    let driven = timeline.isAudioDriven
    // Same ceiling for both since the 145 revert — a driving flag must not
    // cost timeline length.
    let ceiling = DirectorMath.maxChunkFrames
    let maxLength = 1 + DirectorMath.latentStride * ((ceiling - 1) / DirectorMath.latentStride)
      * DirectorMath.maxChunks
    var lengthValid = true
    if length < DirectorMath.minTimelineFrames {
      lengthValid = false
      issues.append(.error("timeline_too_short", "timeline must be at least \(DirectorMath.minTimelineFrames) frames after snapping (got \(length))"))
    } else if length > maxLength {
      lengthValid = false
      issues.append(.error("timeline_too_long", "timeline must be at most \(maxLength) frames (\(DirectorMath.maxChunks) chunks; got \(length))"))
    }
    var layout = lengthValid
      ? DirectorMath.chunkLayout(lengthFrames: length, maxFrames: ceiling) : []
    // WP11, Todd: "use thoughtful pauses to span the joins" (2026-09-18) and
    // "lip sync works well on a single clip — joining them intelligently
    // should avoid issues" (2026-09-19). Sync inside a chunk is not the
    // problem; a seam is the only discontinuity a sequence has, and a mouth
    // caught mid-word jumping across one is what a viewer notices. So move
    // each join into the deepest silence within reach.
    var joinsInPauses = false
    if driven, layout.count > 1,
       let voice = timeline.audioClips.first(where: { $0.drivesVideo }) {
      let fps = snapped.settings.fps ?? DirectorTimeline.Settings.defaultFps
      let runs = pauseRuns(voice.audioPath, fps, length)
      if !runs.isEmpty {
        let before = layout.map(\.startFrame)
        layout = DirectorMath.spanningPauses(
          layout, lengthFrames: length, maxFrames: ceiling, pauses: runs)
        joinsInPauses = before != layout.map(\.startFrame)
      }
    }

    // MARK: Ids

    var seenIds = Set<String>()
    var duplicateIds: [String] = []
    let allIds = timeline.keyframes.map(\.id) + timeline.promptSegments.map(\.id)
      + timeline.audioClips.map(\.id) + timeline.referenceClips.map(\.id)
    for id in allIds where !seenIds.insert(id).inserted {
      if !duplicateIds.contains(id) { duplicateIds.append(id) }
    }
    if !duplicateIds.isEmpty {
      issues.append(.error("duplicate_id", "ids must be unique across keyframes, prompt segments, audio clips and reference clips", ids: duplicateIds))
    }

    // MARK: Keyframes

    let endFrames = timeline.keyframes.filter(\.isEndFrame)
    if endFrames.count > 1 {
      issues.append(.error("multiple_end_frames", "at most one keyframe may be the end frame", ids: endFrames.map(\.id)))
    }
    for i in snapped.keyframes.indices where snapped.keyframes[i].isEndFrame {
      snapped.keyframes[i].frame = length - 1
    }

    var placed: [(id: String, frame: Int, strength: Float)] = []
    for kf in snapped.keyframes {
      var inRange = true
      if kf.frame < 0 || kf.frame > length - 1 {
        inRange = false
        issues.append(.error("keyframe_out_of_range", "keyframe \(kf.id) frame \(kf.frame) is outside 0...\(length - 1)", ids: [kf.id]))
      } else if kf.frame % DirectorMath.latentStride != 0 {
        issues.append(.error("keyframe_off_grid", "keyframe \(kf.id) frame \(kf.frame) is not a multiple of \(DirectorMath.latentStride) (the latent grid)", ids: [kf.id]))
      }
      if !(kf.strength > 0 && kf.strength <= 1) {
        issues.append(.error("invalid_strength", "keyframe \(kf.id) strength must be in (0, 1] (got \(kf.strength))", ids: [kf.id]))
      }
      let hasBase64 = !(kf.imageBase64 ?? "").isEmpty
      if let path = kf.imagePath, !path.isEmpty {
        if !hasBase64 && !fileExists(path) {
          issues.append(.error("keyframe_image_unreadable", "keyframe \(kf.id) image not found: \(path)", ids: [kf.id]))
        }
      } else if !hasBase64 {
        issues.append(.error("keyframe_image_missing", "keyframe \(kf.id) needs image_path or image_base64", ids: [kf.id]))
      }
      if inRange { placed.append((kf.id, kf.frame, kf.strength)) }
    }

    // Collisions: two keyframes in the same latent bucket (last writer wins in
    // applyConditioning). Reported once per bucket.
    let buckets = Dictionary(grouping: placed, by: { $0.frame / DirectorMath.latentStride })
    for bucket in buckets.keys.sorted() {
      let members = buckets[bucket]!
      if members.count > 1 {
        issues.append(.error("keyframes_collide", "keyframes share latent frame \(bucket) (frames \(members.map { String($0.frame) }.joined(separator: ", ")))", ids: members.map(\.id)))
      }
    }
    // Close keyframes: adjacent distinct buckets closer than 24 frames (the
    // multi-keyframe spike's jump-cut finding).
    let ordered = placed.sorted { $0.frame < $1.frame }
    for i in 1..<max(1, ordered.count) {
      let a = ordered[i - 1], b = ordered[i]
      let gap = b.frame - a.frame
      let sameBucket = a.frame / DirectorMath.latentStride == b.frame / DirectorMath.latentStride
      if !sameBucket && gap < DirectorMath.closeKeyframeFrames {
        issues.append(.warning("keyframes_close", "keyframes \(a.id) and \(b.id) are \(gap) frames apart (< \(DirectorMath.closeKeyframeFrames)); expect a jump cut", ids: [a.id, b.id]))
      }
    }
    // Boundary keyframes condition chunk k-1's last frame only; chunk k sees
    // the rendered carry-over at strength 1.0.
    if layout.count > 1 {
      let boundaries = Set(layout.dropFirst().map(\.startFrame))
      for kf in ordered where boundaries.contains(kf.frame) {
        issues.append(.warning("keyframe_on_chunk_boundary", "keyframe \(kf.id) sits on a chunk boundary (frame \(kf.frame)): it conditions the chunk it ends; the next chunk starts from the rendered frame", ids: [kf.id]))
        if kf.strength < 1 {
          issues.append(.warning("keyframe_partial_strength_on_boundary", "keyframe \(kf.id) has strength \(kf.strength) on a boundary: the next chunk sees the rendered carry-over at 1.0", ids: [kf.id]))
        }
      }
    }
    if !placed.contains(where: { $0.frame == 0 }) {
      issues.append(.warning("no_keyframes", "no keyframe at frame 0: chunk 0 renders text-to-video"))
    }

    // MARK: Prompt segments

    var validSegments: [DirectorTimeline.PromptSegment] = []
    for seg in timeline.promptSegments {
      if seg.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        issues.append(.error("segment_empty_prompt", "prompt segment \(seg.id) has an empty prompt", ids: [seg.id]))
      }
      if seg.startFrame < 0 || seg.lengthFrames < 1 || seg.startFrame + seg.lengthFrames > length {
        issues.append(.error("segment_out_of_range", "prompt segment \(seg.id) [\(seg.startFrame), \(seg.startFrame + seg.lengthFrames)) must lie within 0...\(length) with length >= 1", ids: [seg.id]))
        continue
      }
      if seg.lengthFrames < DirectorMath.latentStride {
        issues.append(.warning("segment_shorter_than_latent_frame", "prompt segment \(seg.id) is \(seg.lengthFrames) frames (< \(DirectorMath.latentStride), one latent frame)", ids: [seg.id]))
      } else if seg.lengthFrames < DirectorMath.biasWindowFrames {
        issues.append(.warning("segment_shorter_than_bias_window", "prompt segment \(seg.id) is \(seg.lengthFrames) frames (< \(DirectorMath.biasWindowFrames)): the beat bias has no flat window", ids: [seg.id]))
      }
      validSegments.append(seg)
    }
    let sortedSegments = validSegments.sorted { ($0.startFrame, $0.id) < ($1.startFrame, $1.id) }
    for i in 1..<max(1, sortedSegments.count) {
      let a = sortedSegments[i - 1], b = sortedSegments[i]
      if b.startFrame < a.startFrame + a.lengthFrames {
        issues.append(.warning("segments_overlap", "prompt segments \(a.id) and \(b.id) overlap", ids: [a.id, b.id]))
      }
    }

    // MARK: Audio

    switch timeline.audio.mode {
    case .inpaint:
      issues.append(.error("audio_mode_unsupported", "audio.mode \"inpaint\" is Phase 2; use \"generated\" or \"imported\""))
    case .generated:
      if !timeline.audioClips.isEmpty {
        issues.append(.warning("audio_clips_ignored", "audio_clips are ignored while audio.mode is \"generated\"", ids: timeline.audioClips.map(\.id)))
      }
      if layout.count > 1 {
        issues.append(.warning("generated_audio_seams", "audio.mode \"generated\" over \(layout.count) chunks: each chunk generates its own audio from noise (seams at chunk boundaries)"))
      }
    case .imported:
      var kept: [DirectorTimeline.AudioClip] = []
      for var clip in timeline.audioClips {
        // The mixer places a clip at its start frame and takes length_frames
        // after trim_start_frames: a negative start/trim or an empty clip
        // would be silently misplaced or dropped (DirectorImport defaults a
        // missing upstream length to 0), so they are errors, not guesses.
        if clip.startFrame < 0 || clip.lengthFrames < 1 || clip.trimStartFrames < 0 {
          issues.append(.error(
            "audio_clip_out_of_range",
            "audio clip \(clip.id) needs start_frame >= 0, length_frames >= 1 and trim_start_frames >= 0 (got start \(clip.startFrame), length \(clip.lengthFrames), trim \(clip.trimStartFrames))",
            ids: [clip.id]))
          continue
        }
        if clip.gain < 0 {
          issues.append(.error("invalid_gain", "audio clip \(clip.id) gain must be >= 0 (got \(clip.gain))", ids: [clip.id]))
        }
        if !fileExists(clip.audioPath) {
          issues.append(.error("audio_clip_missing", "audio clip \(clip.id) file not found: \(clip.audioPath)", ids: [clip.id]))
        } else if let probe = audioProbe(clip.audioPath) {
          if !probe.hasAudioTrack || probe.durationSeconds <= 0 {
            issues.append(.error("audio_clip_undecodable", "audio clip \(clip.id) has no decodable audio track: \(clip.audioPath)", ids: [clip.id]))
          } else if fps > 0, DirectorMath.seconds(frames: clip.trimStartFrames, fps: fps) >= probe.durationSeconds {
            issues.append(.warning("audio_trim_past_end", "audio clip \(clip.id) trim_start_frames \(clip.trimStartFrames) is past the end of the file (\(probe.durationSeconds) s): the clip is silence", ids: [clip.id]))
          }
        }
        if clip.startFrame + clip.lengthFrames > length {
          issues.append(.warning("audio_clip_past_end", "audio clip \(clip.id) runs past the timeline end and is trimmed to frame \(length)", ids: [clip.id]))
          if clip.startFrame >= length { continue }
          clip.lengthFrames = length - clip.startFrame
        }
        kept.append(clip)
      }
      snapped.audioClips = kept
    }

    // MARK: Phase 2 placeholders

    if !timeline.referenceClips.isEmpty {
      issues.append(.error("reference_clips_unsupported", "reference_clips are Phase 2", ids: timeline.referenceClips.map(\.id)))
    }
    if timeline.retake.enabled {
      issues.append(.error("retake_unsupported", "retake is Phase 2"))
    }

    let ok = !issues.contains { $0.severity == .error }
    let plan = ok
      ? DirectorCompiler.plan(
          for: snapped, warnings: issues.filter { $0.severity == .warning },
          layout: layout, joinsInPauses: joinsInPauses)
      : nil
    return DirectorValidation(snapped: snapped, issues: issues, ok: ok, plan: plan)
  }
}
