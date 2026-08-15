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
///
/// The step term is multiplied by a large odd constant (golden-ratio-derived,
/// same one MLX/xxhash-style mixers use) before folding into the key. Do NOT
/// simplify this back to a bare `&+ UInt64(step)` — the production chunk
/// scheduler derives each chunk's seed as `request.seed + UInt64(chunk)`
/// (`LTX2VideoGenerator.swift`), so with a bare step offset, chunk `c` step
/// `i` and chunk `c+1` step `i-1` fold to the identical key
/// (`seed + c + i == seed + (c+1) + (i-1)`) — every multi-chunk render was
/// silently reusing bit-identical noise tensors across chunk/step pairs
/// (codex review, 2026-08-15). Multiplying the step by a constant that isn't
/// 1 makes `chunk_delta == step_delta` impossible to satisfy for any nonzero
/// chunk/step deltas actually produced by the scheduler.
public func ancestralVideoNoise(shape: [Int], seed: UInt64?, step: Int) -> MLXArray {
  if let seed {
    let key = MLXRandom.key(seed &+ ltx2VideoNoiseKeyBase &+ (UInt64(step) &* 0x9E37_79B9_7F4A_7C15))
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
///
/// CRITICAL: `MLXArray` is a reference type wrapping a lazily-evaluated compute graph.
/// This initializer **materializes on capture** by calling `eval()` on `videoLatents`,
/// `audioLatents`, and `audioNoiseKey` before assignment. This enforces the invariant
/// that a resume state is always a concrete snapshot, never a live graph handle, so
/// downstream code cannot accidentally alias the checkpoint through in-place mutation.
/// Note: `eval()` on an already-evaluated array is a cheap no-op (the yield point sits
/// between steps, where latents are already evaluated).
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

  /// Initialize a resume state, materializing all array fields to ensure
  /// the checkpoint is a concrete snapshot independent of future mutations.
  public init(
    videoLatents: MLXArray,
    stepIndex: Int,
    sigmas: [Float],
    phase: LTX2Phase,
    chunkIndex: Int,
    seed: UInt64?,
    audioLatents: MLXArray?,
    audioNoiseKey: MLXArray?,
    configFingerprint: String
  ) {
    eval(videoLatents)
    self.videoLatents = videoLatents
    self.stepIndex = stepIndex
    self.sigmas = sigmas
    self.phase = phase
    self.chunkIndex = chunkIndex
    self.seed = seed
    if let audioLatents {
      eval(audioLatents)
      self.audioLatents = audioLatents
    } else {
      self.audioLatents = nil
    }
    if let audioNoiseKey {
      eval(audioNoiseKey)
      self.audioNoiseKey = audioNoiseKey
    } else {
      self.audioNoiseKey = nil
    }
    self.configFingerprint = configFingerprint
  }
}
