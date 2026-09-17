// InferenceSlots.swift — leased, top-priority GPU slots for local LLM/VLM
// inference (Glimmer on mlx-serve shares this Mac's GPU with every render).
//
// FDD-glimmer-gpu-slot §3.1. Todd 2026-09-17: "any glimmer action requires a
// scheduled GPU lease slot with top priority."
//
//  - `InferenceSlotTable`: who holds a slot and until when. While any slot is
//    held the job loop starts no render (independent of the operator's manual
//    pause). Every slot is a TTL lease, so a crashed client cannot pin the GPU.
//  - `InferenceHold`: the idle → requested → parked handshake between the slot
//    route (which asks an in-flight LTX-2 video to checkpoint) and the video's
//    preemption episode (which parks until the table is empty, then resumes).
//  - `InferenceSlotGrant`: the pure rule a client's `state` is computed from.
//
// Everything here is lock-based and never touches the coordinator actor, so
// the routes answer during a render (sync control plane).

import Foundation

public struct InferenceSlot: Sendable, Equatable {
  public let id: String
  public let holder: String
  public let acquiredAt: Date
  public var expiresAt: Date
}

public final class InferenceSlotTable: @unchecked Sendable {
  public static let defaultTTL: TimeInterval = 180
  public static let maxTTL: TimeInterval = 900
  public static let minTTL: TimeInterval = 1

  private let lock = NSLock()
  private let now: @Sendable () -> Date
  private var slots: [String: InferenceSlot] = [:]
  private var waiters: [UUID: CheckedContinuation<Void, Never>] = [:]
  private var onEmptyHandler: (@Sendable () -> Void)?

  public init(now: @escaping @Sendable () -> Date = { Date() }) {
    self.now = now
  }

  /// Fired once each time the table goes from held to empty (last release or
  /// last expiry). Called outside the lock.
  public var onEmpty: (@Sendable () -> Void)? {
    get { lock.lock(); defer { lock.unlock() }; return onEmptyHandler }
    set { lock.lock(); onEmptyHandler = newValue; lock.unlock() }
  }

  static func clampTTL(_ ttl: TimeInterval?) -> TimeInterval {
    guard let ttl else { return defaultTTL }
    return min(maxTTL, max(minTTL, ttl))
  }

  public func acquire(holder: String, ttl: TimeInterval?) -> InferenceSlot {
    let seconds = Self.clampTTL(ttl)
    let t = now()
    let slot = InferenceSlot(id: UUID().uuidString, holder: holder, acquiredAt: t, expiresAt: t.addingTimeInterval(seconds))
    lock.lock()
    slots[slot.id] = slot
    lock.unlock()
    scheduleSweep(after: seconds)
    return slot
  }

  public func renew(id: String, ttl: TimeInterval?) -> InferenceSlot? {
    let seconds = Self.clampTTL(ttl)
    let t = now()
    lock.lock()
    guard var slot = slots[id], slot.expiresAt > t else {
      lock.unlock()
      return nil
    }
    slot.expiresAt = t.addingTimeInterval(seconds)
    slots[id] = slot
    lock.unlock()
    scheduleSweep(after: seconds)
    return slot
  }

  @discardableResult
  public func release(id: String) -> Bool {
    lock.lock()
    guard slots.removeValue(forKey: id) != nil else {
      lock.unlock()
      return false
    }
    let emptied = pruneLocked(at: now()) || slots.isEmpty
    let fire = emptied ? drainLocked() : nil
    lock.unlock()
    fire?()
    return true
  }

  public func get(id: String) -> InferenceSlot? {
    let t = now()
    lock.lock(); defer { lock.unlock() }
    guard let slot = slots[id], slot.expiresAt > t else { return nil }
    return slot
  }

  public func activeSlots() -> [InferenceSlot] {
    sweep()
    let t = now()
    lock.lock(); defer { lock.unlock() }
    return slots.values.filter { $0.expiresAt > t }.sorted { $0.acquiredAt < $1.acquiredAt }
  }

  public func isHeld() -> Bool {
    let t = now()
    lock.lock(); defer { lock.unlock() }
    return slots.values.contains { $0.expiresAt > t }
  }

  /// Drop expired slots; if that empties a previously held table, notify.
  public func sweep() {
    lock.lock()
    let hadAny = !slots.isEmpty
    _ = pruneLocked(at: now())
    let fire = (hadAny && slots.isEmpty) ? drainLocked() : nil
    lock.unlock()
    fire?()
  }

  /// Returns once no slot is held. Returns immediately when already empty;
  /// returns early (without releasing anything) when the calling task is
  /// cancelled.
  public func waitUntilFree() async {
    sweep()
    let id = UUID()
    await withTaskCancellationHandler {
      await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
        let t = now()
        lock.lock()
        if Task.isCancelled || !slots.values.contains(where: { $0.expiresAt > t }) {
          lock.unlock()
          cont.resume()
          return
        }
        waiters[id] = cont
        lock.unlock()
      }
    } onCancel: {
      lock.lock()
      let cont = waiters.removeValue(forKey: id)
      lock.unlock()
      cont?.resume()
    }
  }

  // MARK: - Private

  /// Removes expired slots. Returns true when something was removed and the
  /// table is now empty. Caller holds the lock.
  private func pruneLocked(at t: Date) -> Bool {
    let before = slots.count
    slots = slots.filter { $0.value.expiresAt > t }
    return before > 0 && slots.isEmpty
  }

  /// Takes the waiters and the handler; returns a closure that resumes/fires
  /// them outside the lock. Caller holds the lock.
  private func drainLocked() -> (() -> Void) {
    let pending = Array(waiters.values)
    waiters.removeAll()
    let handler = onEmptyHandler
    return {
      pending.forEach { $0.resume() }
      handler?()
    }
  }

  private func scheduleSweep(after seconds: TimeInterval) {
    DispatchQueue.global().asyncAfter(deadline: .now() + seconds + 0.05) { [weak self] in
      self?.sweep()
    }
  }
}

/// idle → requested (a slot route asked the in-flight video to checkpoint)
/// → parked (the video's preemption episode claimed the request and is holding
/// the GPU free) → idle (slots released, video resumes).
public final class InferenceHold: @unchecked Sendable {
  private enum Phase { case idle, requested, parked }
  private let lock = NSLock()
  private var phase: Phase = .idle

  public init() {}

  public func request() -> Bool {
    lock.lock(); defer { lock.unlock() }
    guard phase == .idle else { return false }
    phase = .requested
    return true
  }

  public func takeRequest() -> Bool {
    lock.lock(); defer { lock.unlock() }
    guard phase == .requested else { return false }
    phase = .parked
    return true
  }

  public func cancelIfRequested() -> Bool {
    lock.lock(); defer { lock.unlock() }
    guard phase == .requested else { return false }
    phase = .idle
    return true
  }

  public func unpark() {
    lock.lock(); defer { lock.unlock() }
    if phase == .parked { phase = .idle }
  }

  public var isRequested: Bool { lock.lock(); defer { lock.unlock() }; return phase == .requested }
  public var isParked: Bool { lock.lock(); defer { lock.unlock() }; return phase == .parked }
}

public enum InferenceSlotState: String, Sendable {
  case granted, preempting, waiting
}

public enum InferenceSlotGrant {
  /// A parked video or an idle GPU grants; a pending hold request is
  /// preempting; anything else on the GPU (an image render, a video that could
  /// not be asked to yield) is waiting.
  public static func state(isParked: Bool, isRequested: Bool, gpuBusy: Bool) -> InferenceSlotState {
    if isParked || !gpuBusy { return .granted }
    return isRequested ? .preempting : .waiting
  }

  /// Remaining seconds of the render on the GPU, from its progress: elapsed ·
  /// (100 − pct) / pct. Nil when progress is unknown or at either end.
  public static func etaSec(progressPct: Int?, renderStartedAt: Date?, now: Date) -> Double? {
    guard let pct = progressPct, pct >= 1, pct <= 99, let started = renderStartedAt else { return nil }
    let elapsed = now.timeIntervalSince(started)
    guard elapsed > 0 else { return nil }
    return elapsed * Double(100 - pct) / Double(pct)
  }
}

// MARK: - Route

/// Everything the `/v1/queue/inference-slot` routes read or do, injected so the
/// protocol is unit-testable without a live render. The server supplies
/// lock-based readers (never the coordinator actor) and the hold-raising
/// closure that talks to the #1479 preemption machinery.
public struct InferenceSlotRouteContext: Sendable {
  public let table: InferenceSlotTable
  public let hold: InferenceHold
  /// Anything on the GPU right now (image render, video render, model op).
  public let gpuBusy: @Sendable () -> Bool
  /// An LTX-2 video render specifically — the only preemptible kind.
  public let videoRendering: @Sendable () -> Bool
  /// Progress of whatever is on the GPU, for `eta_sec`.
  public let progress: @Sendable () -> (Int?, Date?)
  /// Ask the in-flight video to checkpoint for the slots. Called at most once
  /// per pending request (the route checks `hold` first).
  public let raiseHold: @Sendable () -> Void
  public let now: @Sendable () -> Date

  public init(
    table: InferenceSlotTable, hold: InferenceHold,
    gpuBusy: @escaping @Sendable () -> Bool, videoRendering: @escaping @Sendable () -> Bool,
    progress: @escaping @Sendable () -> (Int?, Date?), raiseHold: @escaping @Sendable () -> Void,
    now: @escaping @Sendable () -> Date = { Date() }
  ) {
    self.table = table
    self.hold = hold
    self.gpuBusy = gpuBusy
    self.videoRendering = videoRendering
    self.progress = progress
    self.raiseHold = raiseHold
    self.now = now
  }
}

public enum InferenceSlotRoute {
  static let base = "/v1/queue/inference-slot"

  public static func matches(method: String, path: String) -> Bool {
    if path == base { return method == "POST" || method == "GET" }
    guard path.hasPrefix(base + "/") else { return false }
    let rest = path.dropFirst(base.count + 1)
    guard !rest.isEmpty else { return false }
    if rest.hasSuffix("/renew") { return method == "POST" && rest.split(separator: "/").count == 2 }
    return (method == "GET" || method == "DELETE") && !rest.contains("/")
  }

  public static func handle(method: String, path: String, body: Data, ctx: InferenceSlotRouteContext) -> (status: Int, payload: [String: Any]) {
    if path == base {
      return method == "POST" ? acquire(body: body, ctx: ctx) : list(ctx: ctx)
    }
    let rest = String(path.dropFirst(base.count + 1))
    if method == "POST", rest.hasSuffix("/renew") {
      let id = String(rest.dropLast("/renew".count))
      guard let slot = ctx.table.renew(id: id, ttl: ttl(from: body)) else { return notFound(id) }
      return (200, status(of: slot, ctx: ctx))
    }
    switch method {
    case "GET":
      guard let slot = ctx.table.get(id: rest) else { return notFound(rest) }
      return (200, status(of: slot, ctx: ctx))
    case "DELETE":
      let released = ctx.table.release(id: rest)
      return (200, ["released": released, "active_slots": ctx.table.activeSlots().count])
    default:
      return (405, ["error": "method not allowed"])
    }
  }

  // MARK: Private

  private static func acquire(body: Data, ctx: InferenceSlotRouteContext) -> (status: Int, payload: [String: Any]) {
    let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
    guard let holder = (obj?["holder"] as? String)?.trimmingCharacters(in: .whitespaces), !holder.isEmpty else {
      return (400, ["error": "'holder' is required (who is taking the slot, e.g. \"kira:chat\")"])
    }
    let slot = ctx.table.acquire(holder: holder, ttl: ttl(from: body))
    // Top priority over video: ask the in-flight render to checkpoint, once.
    if ctx.videoRendering(), !ctx.hold.isParked, !ctx.hold.isRequested {
      ctx.raiseHold()
    }
    return (200, status(of: slot, ctx: ctx))
  }

  private static func list(ctx: InferenceSlotRouteContext) -> (status: Int, payload: [String: Any]) {
    let slots = ctx.table.activeSlots()
    return (200, ["active_slots": slots.count, "slots": slots.map { status(of: $0, ctx: ctx) }])
  }

  private static func status(of slot: InferenceSlot, ctx: InferenceSlotRouteContext) -> [String: Any] {
    let state = InferenceSlotGrant.state(
      isParked: ctx.hold.isParked, isRequested: ctx.hold.isRequested, gpuBusy: ctx.gpuBusy())
    var out: [String: Any] = [
      "slot_id": slot.id,
      "holder": slot.holder,
      "state": state.rawValue,
      "expires_at": ISO8601DateFormatter().string(from: slot.expiresAt),
      "active_slots": ctx.table.activeSlots().count,
    ]
    if state == .waiting {
      let (pct, started) = ctx.progress()
      if let eta = InferenceSlotGrant.etaSec(progressPct: pct, renderStartedAt: started, now: ctx.now()) {
        out["eta_sec"] = eta
      }
    }
    return out
  }

  private static func ttl(from body: Data) -> TimeInterval? {
    let obj = (try? JSONSerialization.jsonObject(with: body)) as? [String: Any]
    if let n = obj?["ttl_sec"] as? NSNumber { return n.doubleValue }
    return nil
  }

  private static func notFound(_ id: String) -> (status: Int, payload: [String: Any]) {
    (404, ["error": "unknown or expired inference slot '\(id)'"])
  }
}
