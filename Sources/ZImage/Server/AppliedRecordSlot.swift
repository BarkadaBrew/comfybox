// AppliedRecordSlot.swift — the tri-state `applied` slot on the JSON sinks
// (K-FIX-1 round 2, C4 client-lane finding).
//
// `RenderRecipe?` cannot carry the distinction the client needs, because a
// synthesized `Codable` OMITS a nil Optional. So on the wire these two were
// the same thing — an absent `applied` key:
//
//   * a NON-KREA-2 render, which has no provenance record by design (D12); and
//   * a KREA 2 render whose record was REFUSED, because the pipeline's loaded
//     LoRA configs and bind reports disagreed in length and §3.10's fail-closed
//     rule says a partial record is worse than none (`loRAReadBacks` → nil).
//
// The second is "engine-incomplete": the render happened, it is Krea 2, and the
// engine is telling you it could not vouch for what applied. The client has to
// be able to see that, and it could not.
//
// WIRE CONTRACT (documented here because three sinks share it):
//
//   key ABSENT  — no Krea 2 provenance applies: another family, or no Krea 2
//                 render has completed yet, or the record was invalidated
//                 because the checkpoint it described is no longer resident.
//   `null`      — a Krea 2 render completed and its record was REFUSED.
//                 Provenance is engine-incomplete for this render.
//   object      — the record.
//
// Mechanism: a single-value `Codable` wrapper. `nil` slot ⇒ the synthesized
// encoder omits the key (absent); a slot holding `nil` ⇒ `encodeNil` writes a
// literal `null`; a slot holding a record ⇒ the object. `RenderRecipe` itself
// is untouched.

import Foundation

/// One sink's `applied` value. See the wire contract above.
public struct AppliedRecordSlot: Codable, Sendable, Equatable {

  /// The record, or `nil` when it was refused (the slot itself being present
  /// is what says a Krea 2 render happened).
  public let record: RenderRecipe?

  public init(record: RenderRecipe?) {
    self.record = record
  }

  /// A Krea 2 render whose provenance read-back was refused — encodes as
  /// `null`, not as an absent key.
  public static let refused = AppliedRecordSlot(record: nil)

  public init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    record = container.decodeNil() ? nil : try container.decode(RenderRecipe.self)
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.singleValueContainer()
    if let record {
      try container.encode(record)
    } else {
      // The whole point: an explicit `null`, never an omitted key.
      try container.encodeNil()
    }
  }
}

/// Make the DECODE side keep the distinction too.
///
/// A synthesized `Codable` decodes a property of type `AppliedRecordSlot?`
/// with `decodeIfPresent`, and the standard implementation folds an explicit
/// `null` into `.none` — so a persisted `"applied": null` would come back
/// indistinguishable from an absent key, and `ImageJobStatus` (the one sink
/// that is re-read from disk after a restart) would forget that the record was
/// refused. This more specific overload is what the synthesized decoder picks,
/// and it separates the two:
///
///   key missing        → `nil`        (absent)
///   key present, null  → `.refused`   (present slot, no record)
///   key present, object→ the record
extension KeyedDecodingContainer {
  func decodeIfPresent(
    _ type: AppliedRecordSlot.Type, forKey key: Key
  ) throws -> AppliedRecordSlot? {
    guard contains(key) else { return nil }
    if try decodeNil(forKey: key) { return .refused }
    return try decode(AppliedRecordSlot.self, forKey: key)
  }
}
