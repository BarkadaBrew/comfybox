import Foundation

/// #286 (review round 1, I2) — "is the stack we are about to load already the
/// one that is loaded?"
///
/// `ZImagePipeline.loadLoRAs` has always short-circuited an identical stack
/// (`ZImagePipeline.swift:914`). The Krea 2 and Flux 2 pipelines do not: they
/// unload and reload every adapter on every call. That was tolerable while a
/// per-request stack only arrived on the explicit `loras` path; once a named
/// preset applies its stack on EVERY render, a 5-10 adapter preset would clear
/// and re-bind the whole stack for each one — pure latency and unified-memory
/// churn on a daemon that renders around the clock, on a Mac that shares its
/// memory with LM Studio and the embeddings service.
///
/// Pure, so the comparison can be tested without loading weights.
public enum LoRAStackIdentity {

  /// Same adapters, same order, same scales, same declared roles.
  ///
  /// Compared on the SOURCE as the pipeline holds it — a full local path or a
  /// `repo/file` HuggingFace id — because that is what would be loaded. (The
  /// looser name-only comparison belongs to
  /// ``PresetLoRAStack/isSameStack(_:_:)``, which diffs a client's flat list
  /// against a preset's bare filenames; here both sides are already resolved,
  /// so an exact match is the honest test and a mismatch only costs a reload.)
  ///
  /// `requiresBase` is deliberately NOT compared: it is metadata about the
  /// adapter's extraction base, folded in identically on both sides before this
  /// is called, and never changes what binds.
  public static func isSameStack(_ resident: [LoRAConfiguration], _ requested: [LoRAConfiguration]) -> Bool {
    guard resident.count == requested.count else { return false }
    for (a, b) in zip(resident, requested) {
      guard a.source == b.source else { return false }
      guard a.scale == b.scale else { return false }
      guard a.role == b.role else { return false }
    }
    return true
  }
}
