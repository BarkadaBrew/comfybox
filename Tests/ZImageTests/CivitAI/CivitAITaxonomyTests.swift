import XCTest
@testable import ZImage

/// `CivitAITaxonomy.inferAct` is a NEW, best-effort keyword classifier
/// invented for #234 (no prior taxonomy existed in this codebase — see the
/// file header). These tests pin its documented behavior so a future edit
/// doesn't silently reshuffle category boundaries.
final class CivitAITaxonomyTests: XCTestCase {

  func testInfersPoseFromNameKeyword() {
    XCTAssertEqual(
      CivitAITaxonomy.inferAct(modelType: "LORA", name: "Dynamic Pose Pack", tags: []),
      CivitAITaxonomy.Act.pose.rawValue)
  }

  func testInfersClothingFromTags() {
    XCTAssertEqual(
      CivitAITaxonomy.inferAct(modelType: "LORA", name: "Whatever", tags: ["lingerie", "outfit"]),
      CivitAITaxonomy.Act.clothing.rawValue)
  }

  func testInfersStyleFromCheckpointType() {
    XCTAssertEqual(
      CivitAITaxonomy.inferAct(modelType: "Checkpoint", name: "My Realistic Model", tags: []),
      CivitAITaxonomy.Act.style.rawValue)
  }

  func testInfersConceptFromTextualInversionType() {
    XCTAssertEqual(
      CivitAITaxonomy.inferAct(modelType: "TextualInversion", name: "Bad Hands Fix", tags: []),
      CivitAITaxonomy.Act.concept.rawValue)
  }

  func testInfersConceptAsLoraFallbackWhenNoKeywordsMatch() {
    XCTAssertEqual(
      CivitAITaxonomy.inferAct(modelType: "LORA", name: "xyz123", tags: []),
      CivitAITaxonomy.Act.concept.rawValue)
  }

  func testInfersCharacterFromNameKeyword() {
    XCTAssertEqual(
      CivitAITaxonomy.inferAct(modelType: "LORA", name: "Original Character OC design", tags: []),
      CivitAITaxonomy.Act.character.rawValue)
  }

  func testReturnsNilWhenThereIsNoSignalAtAll() {
    XCTAssertNil(CivitAITaxonomy.inferAct(modelType: "", name: "", tags: []))
  }

  func testKeywordMatchIsCaseInsensitive() {
    XCTAssertEqual(
      CivitAITaxonomy.inferAct(modelType: "LORA", name: "DYNAMIC POSE PACK", tags: []),
      CivitAITaxonomy.Act.pose.rawValue)
  }

  func testMoreSpecificKeywordBucketsAreCheckedBeforeTypeFallback() {
    // A LORA (which would otherwise fall back to .concept) with a pose
    // keyword in its tags must classify as .pose, not .concept.
    XCTAssertEqual(
      CivitAITaxonomy.inferAct(modelType: "LORA", name: "Anything", tags: ["pose"]),
      CivitAITaxonomy.Act.pose.rawValue)
  }
}
