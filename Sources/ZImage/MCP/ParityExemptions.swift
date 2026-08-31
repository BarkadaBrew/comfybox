// ParityExemptions.swift — reasoned exemptions for the anti-drift parity test
// (FDD-ui-api-parity §3.5 assertion 3, §4.2; comfybox#300).
//
// Every mutating `surface: .v1` route must be claimed by ≥1 MCP tool via its
// `routes:` field, OR listed here with a non-empty reason. The parity test
// (`ControlSurfaceParityTests`) also fails on a STALE exemption (one whose
// route no longer parses from the dispatch switches), so this list can only
// shrink or be consciously grown in review — it cannot rot silently.
//
// Paths use the parser's normalization: `{id}` for a path-parameter segment.

import Foundation

/// One exempted route and the reason it needs no MCP tool.
public struct ParityExemption: Sendable {
  public let route: RouteRef
  public let reason: String

  public init(method: String, path: String, reason: String) {
    self.route = RouteRef(method: method, path: path, surface: .v1)
    self.reason = reason
  }
}

public enum ParityExemptions {
  public static let all: [ParityExemption] = [
    ParityExemption(
      method: "POST", path: "/v1/generate/async",
      reason: "Async job variant of /v1/generate; MCP agents call generate_image (a synchronous MCP call"
        + " wrapping the same render). A dedicated async tool was deferred from the Phase 1 worklist."),
    ParityExemption(
      method: "POST", path: "/v1/video/generate",
      reason: "Synchronous variant; the generate_video tool proxies POST /v1/video/generate/async"
        + " (job-based) so an agent is never blocked for a whole video render."),
    ParityExemption(
      method: "POST", path: "/v1/loras/import",
      reason: "Desktop drag-and-drop import (local file paths on the server host); agent flows"
        + " discover LoRAs via lora_scan / nearline_stage instead."),
    ParityExemption(
      method: "DELETE", path: "/v1/workflows/{id}",
      reason: "Desktop workflow management; delete_workflow tool deferred from the Phase 1 worklist"
        + " (no agent caller deletes workflows today)."),
    ParityExemption(
      method: "POST", path: "/v1/video/traces/{id}/promote",
      reason: "Video-trace curation is Desktop/gallery-driven today; promote_video_trace tool"
        + " deferred from the Phase 1 worklist."),
    ParityExemption(
      method: "POST", path: "/v1/video/traces/{id}/rating",
      reason: "Video-trace curation is Desktop/gallery-driven today; rate_video_trace tool"
        + " deferred from the Phase 1 worklist."),
    ParityExemption(
      method: "PUT", path: "/v1/content-modes/{id}",
      reason: "Phase 3 creative-layer write (Class E); no agent caller edits content modes yet."
        + " Discoverable via GET /v1/controls (creative.contentMode.* descriptors)."),
    ParityExemption(
      method: "DELETE", path: "/v1/content-modes/{id}",
      reason: "Reverts a mode to its built-in definition; same posture as the PUT — no agent"
        + " caller yet, discoverable via GET /v1/controls."),
    ParityExemption(
      method: "POST", path: "/v1/presets/resolve",
      reason: "POST-for-body READ: resolves a preset against the loaded model without changing"
        + " state. Not a mutation; list_presets covers agent reads."),
    ParityExemption(
      method: "POST", path: "/v1/video/config/effective",
      reason: "POST-for-body READ: echoes the effective video config for a hypothetical request"
        + " without changing state (GET variant also exists)."),
  ]

  /// Lookup by parsed route.
  public static func reason(for route: RouteRef) -> String? {
    all.first { $0.route == route }?.reason
  }
}
