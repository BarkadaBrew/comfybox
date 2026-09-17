// DirectorImport.swift — import shim for the upstream LTX Director
// `timeline_data` JSON.
//
// A clean-room BEHAVIOUR mapping of the field names documented in
// docs/FDD-ltx-director-tab.md §2 (global_prompt; segments[] with
// type/start/length/imageFile|imageData/prompt/trimStart/isEndFrame/strength;
// audioSegments[] with audioFile/start/length/trimStart/gain; motionSegments[];
// the retake block; frame_rate; start_frame/end_frame). No upstream code, no
// upstream assets. Parsing is lenient (JSONSerialization): unknown keys are
// ignored and numbers may be Int or Double.

import Foundation

public enum DirectorImport {

  /// Map an upstream timeline onto a DirectorTimeline. `settings` supplies
  /// width/height/seed/preset etc.; `fps` is overridden by `frame_rate` when
  /// present and `length_frames` is derived from the content (max of
  /// `end_frame` and the last segment end, snapped up). Unsupported upstream
  /// content (video/motion segments, the retake block) becomes
  /// `unsupported_upstream_segment` warnings rather than failing the import.
  public static func fromUpstream(
    timelineData: Data, settings: DirectorTimeline.Settings
  ) throws -> (timeline: DirectorTimeline, warnings: [DirectorIssue]) {
    let object: Any
    do {
      object = try JSONSerialization.jsonObject(with: timelineData, options: [.fragmentsAllowed])
    } catch {
      throw DirectorError.importFailed("timeline_data is not JSON: \(error.localizedDescription)")
    }
    guard let root = object as? [String: Any] else {
      throw DirectorError.importFailed("timeline_data must be a JSON object")
    }
    let globalPrompt = string(root["global_prompt"]) ?? ""
    let segments = array(root["segments"])
    let audioSegments = array(root["audioSegments"])
    let motionSegments = array(root["motionSegments"])
    guard !globalPrompt.isEmpty || !segments.isEmpty || !audioSegments.isEmpty else {
      throw DirectorError.importFailed("timeline_data has no global_prompt and no segments")
    }

    var out = settings
    if let rate = number(root["frame_rate"]), rate > 0 {
      out.fps = Int(rate.rounded())
    }

    var warnings: [DirectorIssue] = []
    var keyframes: [DirectorTimeline.Keyframe] = []
    var prompts: [DirectorTimeline.PromptSegment] = []
    var clips: [DirectorTimeline.AudioClip] = []
    var maxEnd = Int((number(root["end_frame"]) ?? 0).rounded())

    for (i, seg) in segments.enumerated() {
      let type = (string(seg["type"]) ?? "image").lowercased()
      let start = int(seg["start"]) ?? 0
      let length = max(0, int(seg["length"]) ?? 0)
      maxEnd = max(maxEnd, start + length)
      switch type {
      case "image":
        let n = keyframes.count + 1
        let imagePath = string(seg["imageFile"]) ?? string(seg["image_path"])
        var base64 = string(seg["imageData"]) ?? string(seg["image_base64"])
        if let b = base64, let comma = b.firstIndex(of: ","), b.hasPrefix("data:") {
          base64 = String(b[b.index(after: comma)...])
        }
        keyframes.append(.init(
          id: "k\(n)",
          imagePath: imagePath?.isEmpty == false ? imagePath : nil,
          imageBase64: base64?.isEmpty == false ? base64 : nil,
          frame: start,
          strength: Float(number(seg["strength"]) ?? 1.0),
          isEndFrame: bool(seg["isEndFrame"]) ?? false))
        if let prompt = string(seg["prompt"]), !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, length > 0 {
          prompts.append(.init(id: "p\(n)", startFrame: start, lengthFrames: length, prompt: prompt))
        }
      case "audio":
        if let clip = audioClip(seg, index: clips.count + 1) { clips.append(clip) }
      default:
        warnings.append(.warning(
          "unsupported_upstream_segment",
          "segments[\(i)] of type \"\(type)\" (reference/motion guide) is Phase 2 and was not imported"))
      }
    }
    for seg in audioSegments {
      let start = int(seg["start"]) ?? 0
      let length = max(0, int(seg["length"]) ?? 0)
      maxEnd = max(maxEnd, start + length)
      if let clip = audioClip(seg, index: clips.count + 1) { clips.append(clip) }
    }
    for i in motionSegments.indices {
      warnings.append(.warning(
        "unsupported_upstream_segment",
        "motionSegments[\(i)] (motion guide) is Phase 2 and was not imported"))
    }
    if (bool(root["retakeMode"]) ?? false) || (string(root["retakeVideo"])?.isEmpty == false) {
      warnings.append(.warning(
        "unsupported_upstream_segment",
        "the retake block is Phase 2 and was not imported"))
    }

    out.lengthFrames = DirectorMath.snapLengthUp(maxEnd)
    let timeline = DirectorTimeline(
      settings: out,
      globalPrompt: globalPrompt,
      keyframes: keyframes,
      promptSegments: prompts,
      audioClips: clips,
      audio: .init(mode: clips.isEmpty ? .generated : .imported))
    return (timeline, warnings)
  }

  // MARK: - Lenient accessors

  private static func audioClip(_ seg: [String: Any], index: Int) -> DirectorTimeline.AudioClip? {
    guard let path = string(seg["audioFile"]) ?? string(seg["audio_path"]), !path.isEmpty else { return nil }
    return .init(
      id: "a\(index)", audioPath: path,
      startFrame: int(seg["start"]) ?? 0,
      lengthFrames: max(0, int(seg["length"]) ?? 0),
      trimStartFrames: max(0, int(seg["trimStart"]) ?? 0),
      gain: Float(number(seg["gain"]) ?? 1.0))
  }

  private static func array(_ v: Any?) -> [[String: Any]] {
    (v as? [Any])?.compactMap { $0 as? [String: Any] } ?? []
  }

  private static func string(_ v: Any?) -> String? {
    v as? String
  }

  private static func number(_ v: Any?) -> Double? {
    if let n = v as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() { return n.doubleValue }
    if let s = v as? String { return Double(s) }
    return nil
  }

  private static func int(_ v: Any?) -> Int? {
    number(v).map { Int($0.rounded()) }
  }

  private static func bool(_ v: Any?) -> Bool? {
    if let b = v as? Bool { return b }
    if let n = v as? NSNumber { return n.boolValue }
    if let s = v as? String { return ["true", "1", "yes"].contains(s.lowercased()) }
    return nil
  }
}
