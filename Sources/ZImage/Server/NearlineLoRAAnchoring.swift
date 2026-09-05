// NearlineLoRAAnchoring.swift — #273 fix round 1 (C1): connects
// NearlineLibrary staging to LoRALibrary's own path bookkeeping so
// anchoring fixes what the issue actually named — a LoRALibraryEntry whose
// `relative_path` points at attached/detachable storage (212 of 213
// entries, measured 2026-08-21).
//
// Free functions (no WarmServer dependency) so they are directly
// unit-testable against real LoRALibrary/NearlineLibrary instances rooted
// at temp directories.

import Foundation

public enum NearlineLoRAAnchoring {
  /// Anchor (or un-anchor) a LoRA by nearline filename.
  ///
  /// On anchor (`anchored == true`): stages the nearline item in via
  /// `NearlineLibrary.setAnchored` (which throws
  /// `NearlineError.insufficientCapacity` if the staging budget can't fit
  /// it, or `.sourceMissing` if the attached volume is unmounted), then —
  /// if a `LoRALibraryEntry` matches this filename — rewrites that entry's
  /// `relativePath` to the freshly staged internal path and sets
  /// `anchored: true` on it, through `LoRALibrary.update(_:patch:)` only
  /// (never by touching library.json directly).
  ///
  /// On un-anchor (`anchored == false`): clears both flags. The file is
  /// left exactly where it is — `relativePath` is not touched.
  ///
  /// Not every nearline item has a matching library entry (a checkpoint, or
  /// a LoRA never imported into the library) — in that case only the
  /// nearline-level anchor takes effect, and the library is left alone.
  @discardableResult
  public static func setAnchored(
    name: String, anchored: Bool, loraLibrary: LoRALibrary?, nearlineLibrary: NearlineLibrary
  ) throws -> NearlineItem {
    let nearlineItem = try nearlineLibrary.setAnchored(name: name, anchored: anchored)

    guard let loraLibrary, let libraryEntry = loraLibrary.entry(for: name) else {
      return nearlineItem
    }

    var patch = LoRAEntryPatch(anchored: anchored)
    if anchored, let stagedPath = nearlineItem.stagedPath {
      patch.relativePath = stagedPath
    }
    try loraLibrary.update(libraryEntry.id, patch: patch)
    return nearlineItem
  }
}
