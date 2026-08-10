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

  /// Rebuild the winner's request at a named resolution budget. The trace's
  /// resolved seed and effective (post-enhancement, post-injection) prompt are
  /// pinned so the replay is deterministic; explicit dims are dropped so the
  /// new budget actually applies.
  public static func rerenderBody(
    requestJSON: String,
    resolvedSeed: String?,
    effectivePrompt: String?,
    resolution: String
  ) throws -> Data {
    guard var body = (try? JSONSerialization.jsonObject(with: Data(requestJSON.utf8))) as? [String: Any]
    else { throw ActionError.malformedRequestJSON }

    body["resolution"] = resolution
    remove(["width", "height", "output_path", "outputPath", "image_base64", "imageBase64"], from: &body)
    if let seed = resolvedSeed.flatMap({ Int($0) }) {
      body["seed"] = seed
    }
    if let prompt = effectivePrompt, !prompt.isEmpty {
      body["prompt"] = prompt
    }
    // The stored prompt is already enhanced and identity-composed — running
    // either pass again would double-inject and truncate.
    body["enhance"] = false
    body["skip_character_injection"] = true
    body["source"] = "winner-rerender"
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
    effectivePrompt: String?
  ) throws -> Data {
    var body: [String: Any] = [:]
    if let json = requestJSON {
      guard let obj = (try? JSONSerialization.jsonObject(with: Data(json.utf8))) as? [String: Any]
      else { throw ActionError.malformedRequestJSON }
      body = obj
    }

    guard let effective = [prompt, effectivePrompt, body["prompt"] as? String]
      .compactMap({ $0 }).first(where: { !$0.isEmpty })
    else { throw ActionError.missingPrompt }

    let fps = (body["fps"] as? Int) ?? 24
    remove(
      ["seed", "duration", "extend_to_seconds", "extendToSeconds",
       "width", "height", "output_path", "outputPath", "image_base64", "imageBase64"],
      from: &body)
    body["prompt"] = effective
    body["image_path"] = framePath
    body["frames"] = snappedFrames(seconds: seconds, fps: fps)
    body["resolution"] = "480p"
    body["enhance"] = false
    body["skip_character_injection"] = true
    body["source"] = "winner-extend"
    return try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
  }
}
