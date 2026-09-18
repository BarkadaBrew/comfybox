// QueueLanes.swift — interactive work cuts the line; batch work fills the
// gaps and can be held for tonight (FDD-ltx-director-tab §4.10.9, WP23).
//
// Todd 2026-09-17, on Sequences: "Pure FIFO blocks the GPU for hours. Daytime
// interactive work needs to cut the line. Overnight batch work should stay
// overnight."
//
// Three tiers, strict priority, highest first:
//   1. INFERENCE — a held Glimmer slot parks renders entirely. That already
//      exists (InferenceSlots) and is not re-implemented here.
//   2. INTERACTIVE — images, single clips, anything a person is waiting for.
//   3. BATCH — sequences, multi-chunk Director timelines, bulk re-renders.
//
// A batch job also carries a DISPOSITION: run now (eligible immediately), run
// tonight (eligible when the overnight window opens), or run at a given time.
// A held job never touches the GPU before its time, however idle the GPU is —
// that is the whole point: daytime GPU stays free.
//
// Everything here is pure. Selection takes the queue and a clock; the window
// maths takes a calendar. The live queue calls in.

import Foundation

public enum QueueLane: String, Codable, Sendable, CaseIterable {
  case interactive
  case batch
}

/// When a batch job becomes eligible.
public enum QueueDisposition: Equatable, Sendable {
  /// Eligible immediately (still below interactive work).
  case now
  /// Eligible when the overnight window next opens.
  case tonight
  /// Eligible at an explicit time.
  case at(Date)

  public init?(raw: String, holdUntil: Date?) {
    switch raw.lowercased() {
    case "now": self = .now
    case "tonight": self = .tonight
    case "at":
      guard let holdUntil else { return nil }
      self = .at(holdUntil)
    default: return nil
    }
  }
}

/// What a batch job does when the window closes under it.
public enum QueueSpill: String, Codable, Sendable {
  /// Stop at the next segment boundary and resume tomorrow night (default).
  case wait
  /// Keep going as a `now` job — still below interactive work.
  case idle
  /// Stop at the window's end, full stop.
  case strict
}

/// The overnight window, as configured. Times are local to `timeZone`.
public struct BatchWindow: Codable, Sendable, Equatable {
  /// "23:00"
  public var start: String
  /// "07:00"
  public var end: String
  public var timeZone: String

  public init(start: String = "23:00", end: String = "07:00", timeZone: String = "America/New_York") {
    self.start = start
    self.end = end
    self.timeZone = timeZone
  }

  enum CodingKeys: String, CodingKey {
    case start, end
    case timeZone = "time_zone"
  }

  var zone: TimeZone { TimeZone(identifier: timeZone) ?? .current }

  static func minutes(_ text: String) -> Int? {
    let parts = text.split(separator: ":")
    guard parts.count == 2, let hour = Int(parts[0]), let minute = Int(parts[1]),
          (0...23).contains(hour), (0...59).contains(minute)
    else { return nil }
    return hour * 60 + minute
  }

  public var isValid: Bool {
    guard let start = Self.minutes(start), let end = Self.minutes(end) else { return false }
    return start != end
  }

  private func calendar() -> Calendar {
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = zone
    return calendar
  }

  /// Is the window open at `now`? A window that crosses midnight (the default
  /// 23:00–07:00) is open on BOTH sides of it.
  public func isOpen(at now: Date) -> Bool {
    guard let startMinutes = Self.minutes(start), let endMinutes = Self.minutes(end) else {
      return false
    }
    let calendar = calendar()
    let components = calendar.dateComponents([.hour, .minute], from: now)
    let current = (components.hour ?? 0) * 60 + (components.minute ?? 0)
    if startMinutes < endMinutes {
      return current >= startMinutes && current < endMinutes
    }
    // Crosses midnight.
    return current >= startMinutes || current < endMinutes
  }

  /// The next moment the window opens. If it is open now, that is now.
  public func nextOpening(after now: Date) -> Date {
    guard let startMinutes = Self.minutes(start) else { return now }
    if isOpen(at: now) { return now }
    let calendar = calendar()
    var components = calendar.dateComponents([.year, .month, .day], from: now)
    components.hour = startMinutes / 60
    components.minute = startMinutes % 60
    components.second = 0
    guard let todayAtStart = calendar.date(from: components) else { return now }
    if todayAtStart > now { return todayAtStart }
    return calendar.date(byAdding: .day, value: 1, to: todayAtStart) ?? now
  }

  /// When the window next closes after `now` (used by `spill: wait`).
  public func nextClosing(after now: Date) -> Date? {
    guard let endMinutes = Self.minutes(end), isOpen(at: now) else { return nil }
    let calendar = calendar()
    var components = calendar.dateComponents([.year, .month, .day], from: now)
    components.hour = endMinutes / 60
    components.minute = endMinutes % 60
    components.second = 0
    guard let todayAtEnd = calendar.date(from: components) else { return nil }
    return todayAtEnd > now
      ? todayAtEnd
      : calendar.date(byAdding: .day, value: 1, to: todayAtEnd)
  }
}

/// One job's scheduling facts, as the queue sees them.
public struct QueueSchedule: Codable, Sendable, Equatable {
  public var lane: QueueLane
  public var holdUntil: Date?
  public var spill: QueueSpill

  public init(lane: QueueLane = .interactive, holdUntil: Date? = nil, spill: QueueSpill = .wait) {
    self.lane = lane
    self.holdUntil = holdUntil
    self.spill = spill
  }

  enum CodingKeys: String, CodingKey {
    case lane, spill
    case holdUntil = "hold_until"
  }

  public func isEligible(at now: Date) -> Bool {
    guard let holdUntil else { return true }
    return holdUntil <= now
  }

  /// Resolve a submitted disposition into a concrete hold time.
  public static func resolve(
    lane: QueueLane, disposition: QueueDisposition, window: BatchWindow, now: Date
  ) -> Date? {
    guard lane == .batch else { return nil }  // interactive is never held
    switch disposition {
    case .now: return nil
    case .tonight: return window.nextOpening(after: now)
    case .at(let date): return date > now ? date : nil
    }
  }
}

public enum QueueLaneScheduler {

  /// Pick the next job to run. Interactive first, then batch, each in
  /// submission order, and a held job is invisible until its time.
  ///
  /// `runsWhilePaused` keeps the existing carve-out: while the queue is paused
  /// (or an inference slot is held) only those operations run, and lanes do
  /// not override that.
  public static func nextIndex<Job>(
    in jobs: [Job],
    now: Date,
    paused: Bool,
    schedule: (Job) -> QueueSchedule,
    runsWhilePaused: (Job) -> Bool
  ) -> Int? {
    if paused {
      return jobs.firstIndex(where: runsWhilePaused)
    }
    let eligible = jobs.enumerated().filter { schedule($0.element).isEligible(at: now) }
    if let interactive = eligible.first(where: { schedule($0.element).lane == .interactive }) {
      return interactive.offset
    }
    return eligible.first(where: { schedule($0.element).lane == .batch })?.offset
  }

  /// A batch job that is running when the window closes: `wait` and `strict`
  /// stop, `idle` keeps going. Returns the new hold time for a job that must
  /// stand down (nil when it may continue).
  public static func holdAfterWindow(
    spill: QueueSpill, window: BatchWindow, now: Date
  ) -> Date? {
    switch spill {
    case .idle: return nil
    case .wait: return window.nextOpening(after: now)
    case .strict:
      // Held indefinitely: only an explicit disposition change releases it.
      return Date.distantFuture
    }
  }

  /// How many chunks a night holds, for an honest estimate.
  /// (~25 GPU minutes per chunk, measured 2026-09-17.)
  public static func chunksPerNight(window: BatchWindow, minutesPerChunk: Int = 25) -> Int {
    guard let start = BatchWindow.minutes(window.start),
          let end = BatchWindow.minutes(window.end), minutesPerChunk > 0
    else { return 0 }
    let span = start < end ? end - start : (24 * 60 - start) + end
    return span / minutesPerChunk
  }

  /// Nights a job of `chunks` needs, given the window.
  public static func nightsNeeded(chunks: Int, window: BatchWindow, minutesPerChunk: Int = 25) -> Int {
    let perNight = chunksPerNight(window: window, minutesPerChunk: minutesPerChunk)
    guard perNight > 0 else { return 0 }
    return Int(ceil(Double(chunks) / Double(perNight)))
  }
}
