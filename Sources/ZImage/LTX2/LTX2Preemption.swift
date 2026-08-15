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

/// Opaque generator-level continuation attached to a checkpoint.
///
/// #1479 Task 4 location decision: `VideoGeneratorHolder.release()`
/// (`Sources/ZImage/Server/VideoGeneratorHolder.swift`) does
/// `generator?.unload(); generator = nil` — it drops the holder's only strong
/// reference, so the `LTX2VideoGenerator` INSTANCE is deallocated by the very
/// eviction preemption performs. Anything parked in a `private var
/// pendingResumeContext` on the generator would die with it. The continuation
/// therefore travels WITH the checkpoint, which the coordinator holds, and a
/// freshly constructed generator can resume from it.
///
/// Deliberately opaque: this file is the preemption vocabulary and stays free
/// of CoreGraphics and request types. The coordinator only has to carry it
/// back; the generator that created it downcasts.
public protocol LTX2ResumeContext: AnyObject {}

/// Fingerprint stamped on a checkpoint taken BEFORE any denoise loop began —
/// a chunk boundary, or a raised signal observed before the model even loads.
/// There is nothing to validate against: the phase restarts from its own
/// deterministic beginning under whatever config is current, exactly as a
/// fresh render would. Loop-internal and phase-boundary checkpoints always
/// carry a real fingerprint and ARE validated.
public let ltx2NotStartedFingerprint = "#1479-not-started"

/// Why a resume was refused. Never silently restart from step 0 (spec, Error
/// handling): a config that drifted between checkpoint and resume is a real
/// bug, and hiding it costs a 15-minute render's worth of wrong output.
public enum LTX2ResumeError: Error, LocalizedError, Equatable {
  case configFingerprintMismatch(checkpoint: String, current: String)
  case sigmaScheduleMismatch(checkpoint: [Float], current: [Float], firstDifferingIndex: Int?)
  case stepOutOfRange(step: Int, steps: Int)

  public var errorDescription: String? {
    switch self {
    case .configFingerprintMismatch(let checkpoint, let current):
      return "LTX-2 resume refused: render config changed between checkpoint and resume — checkpoint '\(checkpoint)' vs current '\(current)'."
    case .sigmaScheduleMismatch(let checkpoint, let current, let idx):
      let at = idx.map { " (first difference at index \($0): \(checkpoint[$0]) vs \(current[$0]))" } ?? ""
      return "LTX-2 resume refused: sigma schedule changed between checkpoint and resume — \(checkpoint.count) vs \(current.count) sigmas\(at)."
    case .stepOutOfRange(let step, let steps):
      return "LTX-2 resume refused: checkpoint step \(step) is outside the current schedule's 0..<\(steps) range."
    }
  }
}

/// Pure resume-admissibility check. Separated from the pipeline so it is unit
/// testable without model weights (Task 4 step 3); the render-level behaviour
/// is Task 6's integration test.
///
/// The fingerprint alone is NOT sufficient: `denoiseConfigFingerprint` covers
/// dims/steps/sampler/cfg but omits `cfgSchedule`, `forceDeterministic` and the
/// sigma VALUES — and a sigma-schedule change (e.g. the tarn1 schedule adopted
/// in 46217ef) silently re-times every remaining step. So the sigmas are
/// compared array-wise against the freshly resolved schedule as well.
/// Exact equality is intended: both sides come from the same deterministic
/// schedule function, so any difference at all IS a config change.
public enum LTX2ResumeValidator {
  public static func validate(
    checkpointFingerprint: String,
    currentFingerprint: String,
    checkpointSigmas: [Float],
    currentSigmas: [Float],
    stepIndex: Int
  ) throws {
    guard checkpointFingerprint == currentFingerprint else {
      throw LTX2ResumeError.configFingerprintMismatch(
        checkpoint: checkpointFingerprint, current: currentFingerprint)
    }
    if checkpointSigmas != currentSigmas {
      let idx = (0..<min(checkpointSigmas.count, currentSigmas.count))
        .first { checkpointSigmas[$0] != currentSigmas[$0] }
      throw LTX2ResumeError.sigmaScheduleMismatch(
        checkpoint: checkpointSigmas, current: currentSigmas, firstDifferingIndex: idx)
    }
    let numSteps = max(0, currentSigmas.count - 1)
    guard stepIndex >= 0, stepIndex < numSteps || (stepIndex == 0 && numSteps == 0) else {
      throw LTX2ResumeError.stepOutOfRange(step: stepIndex, steps: numSteps)
    }
  }
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

  /// The two-stage refine's CLEAN base latent (the upsampled, pre-re-noise
  /// `upLatent`). Non-nil ONLY for a checkpoint taken INSIDE the refine loop:
  /// that loop's denoise mask re-injects this every step, and it cannot be
  /// recovered from the partially-refined latents. nil means "the refine has
  /// not started" — for `phase == .refineDenoise` that is the free unwind
  /// point between base and refine, where `videoLatents` are the finished BASE
  /// latents and the refine runs from its own beginning.
  public var refineCleanLatents: MLXArray?

  /// Generator-level continuation (request, chunk position, frames already
  /// rendered, conditioning images). Held here rather than on the generator
  /// because eviction deallocates the generator — see `LTX2ResumeContext`.
  public var context: LTX2ResumeContext?

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
    configFingerprint: String,
    refineCleanLatents: MLXArray? = nil,
    context: LTX2ResumeContext? = nil
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
    if let refineCleanLatents {
      eval(refineCleanLatents)
      self.refineCleanLatents = refineCleanLatents
    } else {
      self.refineCleanLatents = nil
    }
    self.context = context
  }
}
