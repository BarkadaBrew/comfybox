import Foundation

/// #1479: per-phase render timings. Feeds the preemption refusal guard and
/// /v1/queue observability. Lock-protected (NOT an actor) so the render loop
/// can record with no actor hop — same pattern as LiveHealthState (#217).
public enum LTX2Phase: String, CaseIterable, Sendable {
  case modelLoad, textEncode, baseDenoise, refineDenoise, vaeDecode, vocoder, postProcess
}

public struct LTX2PhaseTelemetryView: Sendable {
  public let phases: [String: (meanSec: Double, samples: Int)]
  public let meanStepSec: Double?
  public let maxUninterruptibleSec: Double?
  public let currentPhase: String?
}

public final class LTX2PhaseTelemetry: @unchecked Sendable {
  private let lock = NSLock()
  private var totals: [LTX2Phase: (sumSec: Double, samples: Int)] = [:]
  private var open: [LTX2Phase: Double] = [:]
  private var current: LTX2Phase?
  private var stepSumSec = 0.0
  private var stepSamples = 0

  public init() {}

  public func begin(_ phase: LTX2Phase, nowMs: Double = Date().timeIntervalSince1970 * 1000) {
    lock.lock(); defer { lock.unlock() }
    open[phase] = nowMs; current = phase
  }

  public func end(_ phase: LTX2Phase, nowMs: Double = Date().timeIntervalSince1970 * 1000) {
    lock.lock(); defer { lock.unlock() }
    guard let started = open.removeValue(forKey: phase) else { return }
    let prev = totals[phase] ?? (0, 0)
    totals[phase] = (prev.sumSec + (nowMs - started) / 1000.0, prev.samples + 1)
    if current == phase { current = nil }
  }

  public func recordStep(seconds: Double) {
    lock.lock(); defer { lock.unlock() }
    stepSumSec += seconds; stepSamples += 1
  }

  public func view() -> LTX2PhaseTelemetryView {
    lock.lock(); defer { lock.unlock() }
    var phases: [String: (meanSec: Double, samples: Int)] = [:]
    for (p, t) in totals where t.samples > 0 {
      phases[p.rawValue] = (t.sumSec / Double(t.samples), t.samples)
    }
    let denoise: Set<LTX2Phase> = [.baseDenoise, .refineDenoise]
    let maxUninterruptible = totals
      .filter { !denoise.contains($0.key) && $0.value.samples > 0 }
      .map { $0.value.sumSec / Double($0.value.samples) }
      .max()
    return LTX2PhaseTelemetryView(
      phases: phases,
      meanStepSec: stepSamples > 0 ? stepSumSec / Double(stepSamples) : nil,
      maxUninterruptibleSec: maxUninterruptible,
      currentPhase: current?.rawValue)
  }
}
