// HeavyModelAdmission.swift — single-heavy-model residency accounting.
//
// The engine can only afford ONE heavy generative model resident in unified
// memory at a time. A resident image model (krea2 alone footprints ~75GB in
// practice, far above the pool's 22GB estimate) plus LTX-2 video (~60-70GB)
// exceeds a 128GB machine's physical RAM and trips OS_REASON_JETSAM.
//
// The ModelPool's VRAM budget is BLIND to LTX-2 (it loads entirely outside the
// pool), so image + video could co-reside and OOM. This type owns the pure
// accounting decision — "evict the other class, and can this load proceed?" —
// so it is unit-testable with injected byte figures, independent of live
// machine state. WarmServer wires it to MemoryProbe for the live numbers.
//
// Issue: #218

import Foundation

/// Which class of heavy model is competing for unified memory.
public enum HeavyModelClass: String, Sendable, Equatable {
  case image
  case video
}

/// Pure admission-control logic for loading a heavy model into unified memory.
public struct HeavyModelAdmission: Sendable, Equatable {
  /// Physical-RAM headroom to keep free after a heavy load (default 15GB).
  /// Leaving headroom avoids jetsam from transient allocation spikes during
  /// the first denoise/decode pass.
  public let headroomBytes: UInt64

  public init(headroomBytes: UInt64 = 15 * 1024 * 1024 * 1024) {
    self.headroomBytes = headroomBytes
  }

  /// Conservative resident-footprint estimate for the LTX-2 video stack
  /// (transformer + connector + VAE + Gemma text encoder), used when deciding
  /// whether an incoming image load must first evict a resident video model.
  public static let ltx2EstimateBytes: UInt64 = 65 * 1024 * 1024 * 1024

  public struct Decision: Sendable, Equatable {
    /// Release the currently-resident models of the *other* class first.
    public let evictOther: Bool
    /// Whether the load may proceed at all.
    public let admit: Bool
    /// Whether headroom will be satisfied (false ⇒ admitted but tight).
    public let hasHeadroom: Bool
    public let reason: String
  }

  /// Projected free memory once the other heavy class is released.
  /// `availableBytes` already excludes the other class's wired arrays, so
  /// releasing it adds `otherResidentBytes` back to the free pool.
  public func projectedFreeAfterEvict(availableBytes: UInt64, otherResidentBytes: UInt64) -> UInt64 {
    availableBytes &+ otherResidentBytes
  }

  /// Decide whether a heavy load of `needBytes` may proceed.
  ///
  /// - Parameters:
  ///   - needBytes: estimated footprint of the model we are about to load.
  ///   - availableBytes: reclaimable/free system memory now
  ///     (`MemoryProbe.systemAvailableMemoryBytes()`).
  ///   - otherResidentBytes: estimated footprint of the OTHER heavy class
  ///     currently resident (0 if none). Always evicted first.
  public func decide(needBytes: UInt64, availableBytes: UInt64, otherResidentBytes: UInt64) -> Decision {
    let evict = otherResidentBytes > 0
    let projectedFree = projectedFreeAfterEvict(availableBytes: availableBytes, otherResidentBytes: otherResidentBytes)

    if projectedFree >= needBytes &+ headroomBytes {
      return Decision(
        evictOther: evict, admit: true, hasHeadroom: true,
        reason: "ok: projected \(projectedFree >> 20)MB free ≥ need \(needBytes >> 20)MB + headroom \(headroomBytes >> 20)MB")
    }
    if projectedFree >= needBytes {
      return Decision(
        evictOther: evict, admit: true, hasHeadroom: false,
        reason: "tight: projected \(projectedFree >> 20)MB free ≥ need \(needBytes >> 20)MB but below headroom target")
    }
    return Decision(
      evictOther: evict, admit: false, hasHeadroom: false,
      reason: "refuse: projected \(projectedFree >> 20)MB free < need \(needBytes >> 20)MB — would OOM")
  }

  /// Post-eviction gate: after the other class has actually been released and
  /// memory re-probed, is there room for `needBytes`? Admits when free memory
  /// covers the need (headroom preferred but not mandatory once we are already
  /// single-resident — refusing here would strand the only heavy model).
  public func admitsAfterEvict(needBytes: UInt64, freeBytes: UInt64) -> Bool {
    freeBytes >= needBytes
  }
}
