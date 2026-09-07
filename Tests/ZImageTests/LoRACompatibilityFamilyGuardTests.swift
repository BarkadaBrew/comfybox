import Foundation
import XCTest

@testable import ZImage

/// #402 (coffeeshop-server #1681, the Luxe_Sensual incident): unit coverage
/// for the pure cross-family LoRA guard —
/// `LoRACompatibility.familyGroup(forCompatibilityTag:)` and
/// `LoRACompatibility.checkFamily(modelCompatibility:targetFamily:)`. No
/// file I/O, no server, no weights — exactly the validator the ruling asked
/// for: "one pure validator using the library's existing model_compatibility
/// / role fields", unknown compatibility allowed with a warning, never a
/// refusal.
final class LoRACompatibilityFamilyGuardTests: XCTestCase {

  // MARK: - familyGroup(forCompatibilityTag:)

  func testKnownTagsGroupToStableFamilies() {
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "z-image"), "z-image")
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "klein-9b"), "flux2-klein")
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "klein-4b"), "flux2-klein")
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "chroma"), "chroma")
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "krea2"), "krea2")
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "krea-2-turbo"), "krea2")
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "ltx"), "ltx")
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "ltx2"), "ltx")
  }

  /// comfybox#393: the LoRA library's own "flux1" tag (real Flux.1-dev/
  /// schnell) is a DIFFERENT thing from this engine's `WarmModelFamily.flux1`
  /// case (Z-Image/Lumina2 under an internal name) — the two must never
  /// collapse to the same group.
  func testRealFlux1TagIsItsOwnGroupDistinctFromZImage() {
    XCTAssertEqual(LoRACompatibility.familyGroup(forCompatibilityTag: "flux1"), "flux1")
    XCTAssertNotEqual(
      LoRACompatibility.familyGroup(forCompatibilityTag: "flux1"),
      LoRACompatibility.familyGroup(forCompatibilityTag: "z-image"))
  }

  func testUnknownTagAndUnrecognizedStringBothReturnNil() {
    XCTAssertNil(LoRACompatibility.familyGroup(forCompatibilityTag: "unknown"))
    XCTAssertNil(LoRACompatibility.familyGroup(forCompatibilityTag: "not-a-real-tag"))
  }

  // MARK: - checkFamily: the four required scenarios

  /// An image (z-image) LoRA used on a video (ltx) target: confidently
  /// mismatched, rejected, naming the LoRA's family.
  func testImageLoRAOnVideoTargetIsRejected() {
    let decision = LoRACompatibility.checkFamily(modelCompatibility: ["z-image"], targetFamily: "ltx")
    XCTAssertFalse(decision.allowed)
    XCTAssertEqual(decision.loraFamilies, ["z-image"])
    XCTAssertNil(decision.warning, "a confident mismatch is a refusal, not a warning")
  }

  /// A video (ltx) LoRA used on an image (z-image) target — the Luxe_Sensual
  /// incident this ticket exists for.
  func testVideoLoRAOnImageTargetIsRejected() {
    let decision = LoRACompatibility.checkFamily(modelCompatibility: ["ltx"], targetFamily: "z-image")
    XCTAssertFalse(decision.allowed)
    XCTAssertEqual(decision.loraFamilies, ["ltx"])
  }

  /// comfybox#393: a krea2 LoRA on the flux1 (Z-Image) family — the target
  /// string here is what `WarmModelFamily.flux1.loraCompatibilityFamily`
  /// resolves to ("z-image"), not the raw "flux1".
  func testKrea2LoRAOnFlux1FamilyIsRejected() {
    let decision = LoRACompatibility.checkFamily(modelCompatibility: ["krea2"], targetFamily: "z-image")
    XCTAssertFalse(decision.allowed)
    XCTAssertEqual(decision.loraFamilies, ["krea2"])
  }

  // MARK: - Fix round 1/2, Critical 2: the flux1/Flux-derived scanner ambiguity

  /// A LoRA whose ONLY declared family is "flux1" is ambiguous (ANY
  /// pre-fix-scan entry could be a mistagged Flux 2 Klein, Chroma, or
  /// Krea 2 LoRA — the scanner fix in this PR only protects FUTURE scans)
  /// against every Flux-derived target and must warn, never throw. Fix
  /// round 2 broadened this from krea2-only (round 1) after review named
  /// two more real library entries the narrower carve-out still 400'd:
  /// fk-adrianoanal (actually Klein) and chroma-unlocked-v47.
  func testFlux1OnlyTagIsAmbiguousAgainstEveryFluxDerivedTarget() {
    for target in ["krea2", "flux2-klein", "chroma"] {
      let decision = LoRACompatibility.checkFamily(modelCompatibility: ["flux1"], targetFamily: target)
      XCTAssertTrue(decision.allowed, "flux1 must warn, not reject, against \(target)")
      XCTAssertNotNil(decision.warning, "flux1 must carry a warning against \(target)")
    }
  }

  /// The exception is scoped to a BARE "flux1" tag — a LoRA that ALSO
  /// declares a confidently different, unrelated family (e.g. ltx) is a
  /// real mismatch, not covered by the ambiguity carve-out.
  func testFlux1PlusUnrelatedTagOnKrea2TargetStillRejects() {
    let decision = LoRACompatibility.checkFamily(modelCompatibility: ["flux1", "ltx"], targetFamily: "krea2")
    XCTAssertFalse(decision.allowed)
  }

  /// The ambiguity carve-out is fatal ONLY against the two architectures
  /// Flux shares nothing with — z-image (Lumina2) and ltx (LTX-2's own
  /// transformer) — a "flux1" tag against either is still a confident,
  /// real mismatch.
  func testFlux1TagStillRejectsAgainstZImageAndLtx() {
    for target in ["z-image", "ltx"] {
      let decision = LoRACompatibility.checkFamily(modelCompatibility: ["flux1"], targetFamily: target)
      XCTAssertFalse(decision.allowed, "flux1 must still reject against \(target)")
    }
  }

  /// Unknown compatibility (absent, or the explicit "unknown" value, or a
  /// string this build doesn't recognize) is ALWAYS allowed, with a warning
  /// — never a refusal (ruling 2).
  func testUnknownCompatibilityIsAllowedWithWarning() {
    for tags: [String] in [[], ["unknown"], ["some-future-family"]] {
      let decision = LoRACompatibility.checkFamily(modelCompatibility: tags, targetFamily: "z-image")
      XCTAssertTrue(decision.allowed, "unknown compatibility must never be refused: \(tags)")
      XCTAssertNotNil(decision.warning, "unknown compatibility must carry a warning: \(tags)")
      XCTAssertTrue(decision.loraFamilies.isEmpty)
    }
  }

  // MARK: - Matching families

  func testMatchingFamilyIsAllowedWithNoWarning() {
    let decision = LoRACompatibility.checkFamily(modelCompatibility: ["krea2"], targetFamily: "krea2")
    XCTAssertTrue(decision.allowed)
    XCTAssertNil(decision.warning)
  }

  /// An entry may legitimately declare more than one compatible family — one
  /// matching the target is enough.
  func testMultipleDeclaredTagsAllowedWhenOneMatches() {
    let decision = LoRACompatibility.checkFamily(
      modelCompatibility: ["klein-9b", "z-image"], targetFamily: "z-image")
    XCTAssertTrue(decision.allowed)
  }

  /// Aliases are normalized before comparison — "krea-2-turbo" must not
  /// spuriously mismatch a "krea2" target.
  func testAliasedTagsNormalizeToTheSameGroup() {
    let decision = LoRACompatibility.checkFamily(modelCompatibility: ["krea-2-turbo"], targetFamily: "krea2")
    XCTAssertTrue(decision.allowed)
  }
}
