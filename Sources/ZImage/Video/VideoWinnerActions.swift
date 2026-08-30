import Foundation

/// Pure JSON transforms behind the gallery "winner actions" (2026-08-10):
/// the scheduler standardizes on cheap 480p/4s exploration, and clips worth
/// keeping get improved post-hoc — `POST /v1/video/rerender` replays the
/// exact request at a 720p budget (same seed, same already-enhanced prompt →
/// the same clip, larger), and `POST /v1/video/extend` chains a fresh 4s
/// continuation from the clip's extracted last frame.
///
/// Both actions rebuild a `/v1/video/generate` body from the sanitized
/// request JSON stored in the render trace's `submitted` payload, then feed
/// it back through the normal local-video path — so preset resolution, dims
/// budgeting, folding and validation are never duplicated here.
public enum VideoWinnerActions {

  public enum ActionError: Error, LocalizedError, CustomStringConvertible {
    case malformedRequestJSON
    case missingPrompt

    public var description: String {
      switch self {
      case .malformedRequestJSON:
        return "Stored request_json is not a JSON object — trace predates replay support?"
      case .missingPrompt:
        return "No prompt available: the trace has none stored, so 'prompt' is required"
      }
    }

    public var errorDescription: String? { description }
  }

  /// Key variants to strip regardless of the caller's casing convention:
  /// the daemon posts snake_case, but the decoder accepts camelCase too.
  private static func remove(_ keys: [String], from body: inout [String: Any]) {
    for key in keys {
      body.removeValue(forKey: key)
    }
  }

  /// Compact a raw `/v1/video/generate` body for storage in the render trace:
  /// the one field that doesn't belong in an append-only log is inline image
  /// bytes (`image_base64` can be megabytes; the resolved temp path is stored
  /// separately). Returns nil when the body isn't a JSON object.
  public static func sanitizedRequestJSON(fromBody body: Data) -> String? {
    guard var obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any] else {
      return nil
    }
    remove(["image_base64", "imageBase64"], from: &obj)
    guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
      let json = String(data: data, encoding: .utf8)
    else { return nil }
    return json
  }

  /// Read an Int-ish value under any of the given key variants.
  private static func intValue(_ keys: [String], in body: [String: Any]) -> Int? {
    for key in keys {
      if let n = body[key] as? Int { return n }
      if let n = body[key] as? Double { return Int(n) }
    }
    return nil
  }

  /// Rebuild the winner's request at a named resolution budget. The trace's
  /// resolved seed and effective (post-enhancement, post-injection) prompt are
  /// pinned so the replay is deterministic; explicit dims are dropped so the
  /// new budget actually applies, with their orientation preserved as an
  /// aspect_ratio (dims without one would silently default to 16:9 landscape).
  public static func rerenderBody(
    requestJSON: String,
    resolvedSeed: String?,
    effectivePrompt: String?,
    resolution: String,
    initImagePath: String? = nil,
    tuningOverride: LTX2VideoTuning? = nil
  ) throws -> Data {
    guard var body = (try? JSONSerialization.jsonObject(with: Data(requestJSON.utf8))) as? [String: Any]
    else { throw ActionError.malformedRequestJSON }

    body["resolution"] = resolution
    if body["aspect_ratio"] == nil, body["aspectRatio"] == nil,
      let width = intValue(["width"], in: body), let height = intValue(["height"], in: body)
    {
      body["aspect_ratio"] = height > width ? "9:16" : "16:9"
    }
    remove(["width", "height", "output_path", "outputPath", "image_base64", "imageBase64"], from: &body)
    // An i2v winner submitted via image_base64 has no image_path left after
    // sanitizing — restore the trace's resolved init image or the replay
    // silently flips to t2v.
    if body["image_path"] == nil, body["imagePath"] == nil, let initImagePath {
      body["image_path"] = initImagePath
    }
    if let seed = resolvedSeed.flatMap({ Int($0) }) {
      body["seed"] = seed
    }
    if let prompt = effectivePrompt, !prompt.isEmpty {
      body["prompt"] = prompt
    }
    // The stored prompt is already enhanced, identity-composed AND preset-
    // wrapped — running any of those passes again would double-compose and
    // push the scene text past the tokenizer cap.
    body["enhance"] = false
    body["skip_character_injection"] = true
    body["skip_preset_prompt"] = true
    body["source"] = "winner-rerender"
    // Tier-A overrides applied ON TOP of the replayed request, so a cheap
    // single-pass explore can be promoted to a two-pass keeper on the SAME
    // seed. Merged key-by-key over any tuning the original request carried:
    // the override is the more specific intent for this replay.
    if let tuningOverride,
      let encoded = try? { () -> Data in
        let enc = JSONEncoder()
        // Mirrors WarmServer's .convertFromSnakeCase decode convention, so the
        // merged keys match what the resolver reads (audio_refine, two_stage).
        enc.keyEncodingStrategy = .convertToSnakeCase
        return try enc.encode(tuningOverride)
      }(),
      let overrides = (try? JSONSerialization.jsonObject(with: encoded)) as? [String: Any]
    {
      var merged = (body["tuning"] as? [String: Any]) ?? [:]
      for (k, v) in overrides { merged[k] = v }
      if !merged.isEmpty { body["tuning"] = merged }
    }
    return try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
  }

  /// Snap a duration to the trained 1+8k frame grid (97f = 4s @ 24fps),
  /// capped at the validated 289f single-pass window.
  static func snappedFrames(seconds: Int, fps: Int) -> Int {
    let target = max(seconds * fps, 9)
    return min(289, ((target - 2) / 8) * 8 + 9)
  }

  /// Build a continuation request: i2v from the clip's extracted last frame at
  /// the 4s/480p standard. A continuation is NEW content, so the winner's seed
  /// is dropped (fresh sampling) and the caller may supply a new motion prompt;
  /// otherwise the trace's effective prompt carries the scene forward.
  public static func extendBody(
    requestJSON: String?,
    framePath: String,
    seconds: Int,
    prompt: String?,
    effectivePrompt: String?,
    freshSeed: UInt64? = nil
  ) throws -> Data {
    var body: [String: Any] = [:]
    if let json = requestJSON {
      guard let obj = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any]
      else { throw ActionError.malformedRequestJSON }
      body = obj
    }

    let callerPrompt = prompt.flatMap { $0.isEmpty ? nil : $0 }
    guard let effective = [callerPrompt, effectivePrompt, body["prompt"] as? String]
      .compactMap({ $0 }).first(where: { !$0.isEmpty })
    else { throw ActionError.missingPrompt }

    let fps = (body["fps"] as? Int) ?? 24
    remove(
      ["seed", "duration", "extend_to_seconds", "extendToSeconds",
       "width", "height", "output_path", "outputPath", "image_base64", "imageBase64",
       "skip_preset_prompt", "skipPresetPrompt"],
      from: &body)
    body["prompt"] = effective
    body["image_path"] = framePath
    body["frames"] = snappedFrames(seconds: seconds, fps: fps)
    body["resolution"] = "480p"
    // The local video path defaults a missing seed to a CONSTANT (42 or the
    // preset seed) — mint one or every extend of a clip renders identically.
    body["seed"] = Int(truncatingIfNeeded: freshSeed ?? UInt64.random(in: 0...0xFFFF_FFFF))
    body["enhance"] = false
    body["skip_character_injection"] = true
    // A stored effective prompt is already preset-wrapped; a caller's fresh
    // motion prompt is raw text and should pick up the preset trigger words
    // exactly as the original render did.
    if callerPrompt == nil {
      body["skip_preset_prompt"] = true
    }
    body["source"] = "winner-extend"
    return try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
  }
}
