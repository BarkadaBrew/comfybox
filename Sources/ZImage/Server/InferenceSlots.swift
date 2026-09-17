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
