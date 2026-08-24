import Foundation

/// Swap-time residency-restore decision (#218 companion).
///
/// `runGenerate` restores an evicted image model before rendering, but a
/// swap-first client (kira-daemon renders images as swap → generate) reaches
/// `runSwap` first — and a swap against a non-resident pipeline used to throw.
/// The client then fails the render closed and never calls generate, so the
/// pipeline is never restored: one video render (or a fresh boot) permanently
/// deadlocks image creation. `runSwap` consults this decision before applying
/// LoRAs so the swap restores residency the same way generate does.
public enum SwapResidencyRestore: Sendable, Equatable {
  /// Pipeline resident (or nothing sensible to load) — apply the swap as-is.
  case none
  /// #218 eviction flag is set — run `reloadImageModelIfEvicted`, the one
  /// place that owns clearing the flag.
  case reloadEvicted
  /// Fresh boot: no eviction recorded but the family pipeline is nil — load
  /// this model spec before applying the swap.
  case load(modelSpec: String)

  public static func decide(
    imageModelsEvicted: Bool,
    familyPipelineMissing: Bool,
    restoreSpec: String?
  ) -> SwapResidencyRestore {
    if imageModelsEvicted { return .reloadEvicted }
    guard familyPipelineMissing, let spec = restoreSpec else { return .none }
    return .load(modelSpec: spec)
  }
}
