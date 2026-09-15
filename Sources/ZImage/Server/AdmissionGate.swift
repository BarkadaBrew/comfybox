import Foundation

/// Runtime admission policy for the warm server — "local mode".
///
/// Todd 2026-09-15: "Is there a way to start the engine in listen mode only
/// so no automation grabs it?" and "there should be logic that other services
/// via mcp or api know it is local mode." Binding to 127.0.0.1 would work but
/// is all-or-nothing, needs a launchd edit + restart, and refuses connections
/// (an error the daemons retry noisily). This gate is a runtime switch:
///
/// - `open`  — every caller is admitted (the default; a restart always comes
///   back open — the policy is in-memory on purpose).
/// - `local` — SUBMIT routes (anything that enqueues work) are admitted only
///   from loopback connections, or from a remote caller whose request
///   `source` starts with an allow-listed prefix. Everything else on a submit
///   route gets **503 + `{"deferred": true, "admission_mode": "local"}`**,
///   which the Coffee Shop daemons already treat as "engine busy, wait" rather
///   than a failure. Read routes (health, queue, status, presets…) are never
///   gated, so callers can SEE the mode: `/health` and `/v1/queue` carry
///   `admission_mode`, and the MCP `server_health` / `queue_status` tools
///   proxy those.
///
/// Toggle: `GET/POST /v1/queue/admission` (`{"mode": "local" | "open",
/// "allow_sources": ["ladder-"]}`) — the menu-bar "Local Mode" item posts it.
public enum AdmissionMode: String, Codable, Sendable {
  case open
  case local
}

public struct AdmissionSnapshot: Codable, Sendable, Equatable {
  public let mode: AdmissionMode
  public let allowSources: [String]
  /// ISO-8601 of the last change; nil while the boot default (`open`) holds.
  public let since: String?
}

public final class AdmissionGate: @unchecked Sendable {
  private let lock = NSLock()
  private var mode: AdmissionMode = .open
  private var allowSources: [String] = []
  private var since: Date? = nil

  public init() {}

  public func snapshot() -> AdmissionSnapshot {
    lock.lock(); defer { lock.unlock() }
    return AdmissionSnapshot(
      mode: mode, allowSources: allowSources,
      since: since.map { ISO8601DateFormatter().string(from: $0) })
  }

  /// Set the policy. Prefixes are trimmed; empty ones dropped.
  public func set(mode newMode: AdmissionMode, allowSources prefixes: [String] = []) {
    lock.lock(); defer { lock.unlock() }
    mode = newMode
    allowSources = prefixes.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    since = Date()
  }

  /// Routes that enqueue work. Read/status/config routes are never gated —
  /// a caller must always be able to learn the mode.
  public static func isSubmit(method: String, path: String) -> Bool {
    guard method.uppercased() == "POST" else { return false }
    if path.hasPrefix("/v1/workflows/") { return path.hasSuffix("/run") }
    let prefixes = [
      "/v1/generate",          // image sync + async
      "/v1/video/generate",    // video sync + async
      "/v1/video/rerender", "/v1/video/extend",
      "/v1/storyboard/render", "/v1/montage/",
      "/v1/upscale", "/v1/img2img", "/v1/inpaint", "/v1/edit", "/v1/photoshoot",
      "/prompt",               // ComfyUI-bridge submit
    ]
    return prefixes.contains { path == $0 || path.hasPrefix($0) }
  }

  /// `nil` = admit. Otherwise the human-readable refusal reason.
  public func refusal(method: String, path: String, isLoopbackPeer: Bool, source: String?) -> String? {
    guard Self.isSubmit(method: method, path: path) else { return nil }
    lock.lock()
    let m = mode, allow = allowSources
    lock.unlock()
    guard m == .local else { return nil }
    if isLoopbackPeer { return nil }
    if let s = source?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty,
       allow.contains(where: { s.hasPrefix($0) }) {
      return nil
    }
    let allowed = allow.isEmpty ? "loopback callers only" : "loopback callers + sources \(allow.joined(separator: ", "))"
    return "engine is in local mode — remote submissions are deferred (\(allowed))"
  }

  /// Lenient `source` extraction from a submit body: `{"source": "…"}` at the
  /// top level, else nil. Never throws — a malformed body is the route's problem.
  public static func submitSource(from body: Data) -> String? {
    guard !body.isEmpty,
          let obj = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
          let s = obj["source"] as? String
    else { return nil }
    return s
  }
}

/// The 503 body remote callers see in local mode. `deferred: true` is the
/// contract the daemons key on (engine busy → wait), `admission_mode` is why.
struct AdmissionRefusal: Encodable {
  let success: Bool = false
  let deferred: Bool = true
  let admissionMode: String
  let error: String
}
