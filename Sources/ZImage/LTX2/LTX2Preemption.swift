import Foundation
import MLX
import MLXRandom

/// #1479 preemption vocabulary. See docs/superpowers/specs/2026-08-15-1479-…
///
/// PreemptionSignal is lock-protected, NOT an actor: the coordinator actor is
/// blocked for the whole synchronous GPU render (WarmServer.swift:4216), so
/// the render loop must read this with no actor hop — the #217 pattern.
public final class PreemptionSignal: @unchecked Sendable {
  private let lock = NSLock()
  private var raised = false
  public init() {}
  public func raise() { lock.lock(); raised = true; lock.unlock() }
  public func clear() { lock.lock(); raised = false; lock.unlock() }
  public var isRaised: Bool { lock.lock(); defer { lock.unlock() }; return raised }
}

/// Base constant for per-step video noise keys. Distinct from the audio
/// path's 0xA0D10/11/12 (LTX2Pipeline.swift:288 area) so streams never collide.
public let ltx2VideoNoiseKeyBase: UInt64 = 0xD0D10

/// Ancestral/SDE noise for the video stream.
/// Seeded: per-step derived key -> independent of global stream position,
/// which is exactly what makes bit-identical resume possible.
/// Unseeded: plain global stream, byte-for-byte the pre-#1479 behaviour.
public func ancestralVideoNoise(shape: [Int], seed: UInt64?, step: Int) -> MLXArray {
  if let seed {
    let key = MLXRandom.key(seed &+ ltx2VideoNoiseKeyBase &+ UInt64(step))
    return MLXRandom.normal(shape, key: key).asType(.float32)
  }
  return MLXRandom.normal(shape, dtype: .float32)
}

public enum LTX2DenoiseResult {
  case completed(MLXArray)
  case yielded(LTX2ResumeState)
}

/// Checkpoint = ALL non-weight tensors (spec rule). In-memory only —
/// deliberately dies with the process, unlike the isPaused sentinel.
public struct LTX2ResumeState {
  public var videoLatents: MLXArray
  public var stepIndex: Int
  public var sigmas: [Float]
  public var phase: LTX2Phase        // .baseDenoise or .refineDenoise
  public var chunkIndex: Int
  public var seed: UInt64?
  public var audioLatents: MLXArray?
  public var audioNoiseKey: MLXArray?
  public var configFingerprint: String
}
