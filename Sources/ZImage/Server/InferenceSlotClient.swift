// InferenceSlotClient.swift — how an engine-local Glimmer caller holds a
// top-priority inference slot on the warm server.
//
// FDD-glimmer-gpu-slot §3.3 (E6) + Codex review of #465. The desktop captioner,
// the desktop image assistant and MCP `repair_image` call Glimmer's
// /chat/completions directly; without a slot the call crawls under a render
// and never parks an LTX-2 video. The protocol is the engine's
// `/v1/queue/inference-slot` (acquire → poll → renew → release).
//
// Rules this client keeps (the daemon's first cut broke two of them):
//  - Renew from the moment a slot id exists, not from the grant — a wait
//    behind a long image render outlives the lease otherwise.
//  - One deadline per request, set by the caller. The wait budget is the
//    deadline minus a reserve for the generation itself; when the engine's
//    eta already exceeds it, fail at once with a human reason.
//  - Cancellation ends the wait and releases the slot.
//  - Release on every path, from a detached task so a cancelled caller
//    still sends the DELETE.
//  - Engine without the routes (404/405) → run unslotted; engine unreachable
//    → fail open (the call proceeds, as before slots existed).

import Foundation

public enum InferenceSlotWaitError: Error, LocalizedError, Equatable {
  /// The GPU stayed busy past the caller's wait budget.
  case gpuBusy(etaSec: Double?)
  /// The caller was cancelled while waiting.
  case cancelled

  public var errorDescription: String? {
    switch self {
    case .gpuBusy(let eta?):
      return "The GPU is busy rendering (about \(Int(eta.rounded())) s left)."
    case .gpuBusy(nil):
      return "The GPU is busy rendering."
    case .cancelled:
      return "Stopped waiting for the GPU."
    }
  }
}

public struct InferenceSlotClient: Sendable {

  /// One non-granted poll, for a status line.
  public struct Wait: Sendable, Equatable {
    public let state: String
    public let etaSec: Double?
    public let waitedSec: Double

    public init(state: String, etaSec: Double?, waitedSec: Double) {
      self.state = state
      self.etaSec = etaSec
      self.waitedSec = waitedSec
    }

    public var statusText: String {
      if state == "preempting" { return "Waiting for the GPU (pausing the video render…)" }
      if let etaSec { return "Waiting for the GPU (~\(Int(etaSec.rounded()))s)" }
      return "Waiting for the GPU…"
    }
  }

  public static let defaultGlimmerEndpoints = ["10.0.100.134:11234", "127.0.0.1:11234", "localhost:11234"]

  /// `COMFYBOX_GLIMMER_ENDPOINTS` (comma-separated host:port) overrides the default list.
  public static func glimmerEndpoints(environment: [String: String]) -> [String] {
    guard let raw = environment["COMFYBOX_GLIMMER_ENDPOINTS"] else { return defaultGlimmerEndpoints }
    let list = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
    return list.isEmpty ? defaultGlimmerEndpoints : list
  }

  public static func isGlimmerEndpoint(_ baseURL: String, endpoints: [String]) -> Bool {
    guard let url = URL(string: baseURL), let host = url.host?.lowercased() else { return false }
    let port = url.port ?? (url.scheme == "https" ? 443 : 80)
    return endpoints.contains { $0.lowercased() == "\(host):\(port)" }
  }

  let transport: WarmServerTransport
  let endpoints: [String]
  let ttlSec: TimeInterval
  let pollSec: TimeInterval
  /// Wait budget when the caller gives no deadline.
  let maxWaitSec: TimeInterval
  let now: @Sendable () -> Date
  let sleep: @Sendable (TimeInterval) async throws -> Void

  public init(
    transport: WarmServerTransport,
    endpoints: [String] = InferenceSlotClient.glimmerEndpoints(environment: ProcessInfo.processInfo.environment),
    ttlSec: TimeInterval = 180,
    pollSec: TimeInterval = 1,
    maxWaitSec: TimeInterval = 900,
    now: @escaping @Sendable () -> Date = { Date() },
    sleep: @escaping @Sendable (TimeInterval) async throws -> Void = { try await Task.sleep(nanoseconds: UInt64($0 * 1_000_000_000)) }
  ) {
    self.transport = transport
    self.endpoints = endpoints
    self.ttlSec = ttlSec
    self.pollSec = pollSec
    self.maxWaitSec = maxWaitSec
    self.now = now
    self.sleep = sleep
  }

  private static let base = "/v1/queue/inference-slot"

  /// Run `body` holding a slot when `baseURL` is Glimmer; otherwise run it as is.
  /// - Parameters:
  ///   - deadline: the caller's overall time limit (nil → `maxWaitSec` of waiting).
  ///   - reserveSec: time kept back from the deadline for the generation itself.
  ///   - onWait: called for every non-granted state, including the acquire answer.
  public func withSlot<T: Sendable>(
    holder: String, baseURL: String, deadline: Date?, reserveSec: TimeInterval,
    onWait: (@Sendable (Wait) -> Void)?,
    _ body: () async throws -> T
  ) async throws -> T {
    guard Self.isGlimmerEndpoint(baseURL, endpoints: endpoints) else { return try await body() }
    let started = now()
    let waitUntil = deadline.map { $0.addingTimeInterval(-reserveSec) } ?? started.addingTimeInterval(maxWaitSec)

    guard let slotId = try await acquireGranted(
      holder: holder, started: started, waitUntil: waitUntil, onWait: onWait)
    else {
      return try await body()  // engine without slots, or unreachable: fail open
    }

    let transport = self.transport
    let ttl = ttlSec
    let renewer = Task.detached {
      while !Task.isCancelled {
        try? await Task.sleep(nanoseconds: UInt64(max(0.5, ttl / 2) * 1_000_000_000))
        if Task.isCancelled { break }
        _ = try? await transport.post("\(Self.base)/\(slotId)/renew", body: Self.json(["ttl_sec": ttl]))
      }
    }
    do {
      let result = try await body()
      renewer.cancel()
      await release(slotId)
      return result
    } catch {
      renewer.cancel()
      await release(slotId)
      throw error
    }
  }

  /// Acquire and wait for `granted`. Returns the slot id, or nil to run unslotted.
  private func acquireGranted(
    holder: String, started: Date, waitUntil: Date, onWait: (@Sendable (Wait) -> Void)?
  ) async throws -> String? {
    var acquired: (id: String, state: String, eta: Double?)
    do {
      let (status, data) = try await transport.post(
        Self.base, body: Self.json(["holder": holder, "ttl_sec": ttlSec]))
      if status == 404 || status == 405 { return nil }
      guard (200..<300).contains(status), let obj = Self.object(data), let id = obj["slot_id"] as? String else {
        return nil
      }
      acquired = (id, obj["state"] as? String ?? "waiting", Self.double(obj["eta_sec"]))
    } catch {
      return nil
    }

    let slotId = acquired.id
    var state = acquired.state
    var eta = acquired.eta
    var lastRenew = now()

    while state != "granted" {
      if Task.isCancelled {
        await release(slotId)
        throw InferenceSlotWaitError.cancelled
      }
      let remaining = waitUntil.timeIntervalSince(now())
      if remaining <= 0 || (state == "waiting" && (eta ?? 0) > remaining) {
        await release(slotId)
        throw InferenceSlotWaitError.gpuBusy(etaSec: eta)
      }
      onWait?(Wait(state: state, etaSec: eta, waitedSec: now().timeIntervalSince(started)))

      do {
        try await sleep(pollSec)
      } catch {
        await release(slotId)
        throw InferenceSlotWaitError.cancelled
      }
      if Task.isCancelled {
        await release(slotId)
        throw InferenceSlotWaitError.cancelled
      }

      // Renew while waiting: the lease must outlive a long wait.
      if now().timeIntervalSince(lastRenew) >= ttlSec / 2 {
        _ = try? await transport.post("\(Self.base)/\(slotId)/renew", body: Self.json(["ttl_sec": ttlSec]))
        lastRenew = now()
      }

      guard let (status, data) = try? await transport.get("\(Self.base)/\(slotId)") else { continue }
      if status == 404 {
        // Expired or engine restarted: take a fresh slot rather than run unslotted.
        return try await acquireGranted(holder: holder, started: started, waitUntil: waitUntil, onWait: onWait)
      }
      if let obj = Self.object(data) {
        state = obj["state"] as? String ?? state
        eta = Self.double(obj["eta_sec"])
      }
    }
    return slotId
  }

  /// DELETE from a detached task: a cancelled caller still releases.
  private func release(_ slotId: String) async {
    let transport = self.transport
    await Task.detached {
      _ = try? await transport.delete("\(Self.base)/\(slotId)")
    }.value
  }

  private static func json(_ obj: [String: Any]) -> Data {
    (try? JSONSerialization.data(withJSONObject: obj)) ?? Data()
  }

  private static func object(_ data: Data) -> [String: Any]? {
    (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
  }

  private static func double(_ value: Any?) -> Double? {
    if let d = value as? Double { return d }
    if let i = value as? Int { return Double(i) }
    return nil
  }
}
