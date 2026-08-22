import XCTest
@testable import ZImage

/// WP-E10 "E9b" (FDD D17, AC-59a; Addendum A.2, E5 review MAJOR): the
/// mandatory `krea2 handoff:` line names the OUTGOING and INCOMING spec/variant
/// so a slow A/B is attributable — and it fires only on an actual base swap.
/// The E5 implementation also fired on a no-op re-activation of the resident
/// model, which is noise that looks like a 67 s eviction. The line is a pure
/// function of (outgoing, incoming, loadTimeMs) so it is asserted here with no
/// pool and no weights.
extension ModelPoolHandoffTests {

  func testHandoffLogLine() throws {
    let line = try XCTUnwrap(Krea2Handoff.logLine(
      outgoing: .init(spec: "kroma-v0.2-turbo", variant: "turbo"),
      incoming: .init(spec: "krea2-raw", variant: "raw"),
      loadTimeMs: 67_123))
    XCTAssertEqual(line, "krea2 handoff: kroma-v0.2-turbo/turbo → krea2-raw/raw (loadTimeMs=67123)")
  }

  /// A non-krea2 → krea2 switch (and the reverse) is a handoff that touches
  /// the family; both sides are named with non-empty variants.
  func testHandoffLogLineAcrossFamilies() throws {
    let into = try XCTUnwrap(Krea2Handoff.logLine(
      outgoing: .init(spec: "z-image-turbo", variant: "flux1"),
      incoming: .init(spec: "krea2-raw", variant: "raw"),
      loadTimeMs: 10))
    XCTAssertEqual(into, "krea2 handoff: z-image-turbo/flux1 → krea2-raw/raw (loadTimeMs=10)")
    let outOf = try XCTUnwrap(Krea2Handoff.logLine(
      outgoing: .init(spec: "krea2-raw", variant: "raw"),
      incoming: .init(spec: "z-image-turbo", variant: "flux1"),
      loadTimeMs: 10))
    XCTAssertEqual(outOf, "krea2 handoff: krea2-raw/raw → z-image-turbo/flux1 (loadTimeMs=10)")
  }

  /// No-op re-activation of the resident base is NOT a handoff — no line.
  func testNoLineOnNoOpReactivation() {
    XCTAssertNil(Krea2Handoff.logLine(
      outgoing: .init(spec: "krea2-raw", variant: "raw"),
      incoming: .init(spec: "krea2-raw", variant: "raw"),
      loadTimeMs: 3))
  }

  /// A cold start has nothing to hand off from — no line.
  func testNoLineWhenNothingWasResident() {
    XCTAssertNil(Krea2Handoff.logLine(
      outgoing: nil,
      incoming: .init(spec: "krea2-raw", variant: "raw"),
      loadTimeMs: 3))
  }

  /// A swap that touches no krea2 base is not this family's line.
  func testNoLineWhenNeitherSideIsKrea2() {
    XCTAssertNil(Krea2Handoff.logLine(
      outgoing: .init(spec: "z-image-turbo", variant: "flux1"),
      incoming: .init(spec: "klein-4b", variant: "flux2"),
      loadTimeMs: 3))
  }

  /// The variant descriptor is read off the loaded pipeline, never guessed:
  /// a krea2 side whose variant is unknown is reported as such, not as turbo.
  func testUnknownVariantIsNamedUnknown() throws {
    let line = try XCTUnwrap(Krea2Handoff.logLine(
      outgoing: .init(spec: "z-image-turbo", variant: "flux1"),
      incoming: Krea2Handoff.Side(spec: "krea2-raw", family: .krea2, krea2Variant: nil),
      loadTimeMs: 1))
    XCTAssertTrue(line.hasSuffix("→ krea2-raw/unknown (loadTimeMs=1)"), line)
    let side = Krea2Handoff.Side(spec: "kroma-v0.2-turbo", family: .krea2, krea2Variant: .turbo)
    XCTAssertEqual(side.variant, "turbo")
    XCTAssertTrue(side.isKrea2)
    let other = Krea2Handoff.Side(spec: "klein-4b", family: .flux2, krea2Variant: nil)
    XCTAssertEqual(other.variant, "flux2")
    XCTAssertFalse(other.isKrea2)
  }
}
