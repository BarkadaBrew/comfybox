import XCTest
@testable import ZImage

/// Extraction logic for the prompt-repository harvester (#234), exercised
/// against a canned/fixture CivitAI `/api/v1/models` JSON response — no live
/// network calls. `CivitAIHarvestExtractor.entries(from:)` is a pure function
/// (model -> [PromptRepositoryEntry]), so the fixture is decoded straight
/// into `CivitAIModel` and handed to it directly.
final class CivitAIHarvestExtractionTests: XCTestCase {

  /// A representative single-model fixture: two versions, one with an HTML
  /// description over the 500-char excerpt cap, one with a short plain
  /// description. Neither version's `images` carry `meta.prompt` — modeling
  /// the 2026-07-15 ground-truth finding that the field is gated even with a
  /// valid API key, which is why the harvester does not rely on it.
  private static let fixtureJSON = """
  {
    "id": 12345,
    "name": "Cinematic Portrait Style",
    "type": "LORA",
    "nsfw": false,
    "tags": ["portrait", "cinematic", "photography"],
    "creator": {"username": "someartist"},
    "stats": {"downloadCount": 9000, "thumbsUpCount": 450},
    "modelVersions": [
      {
        "id": 111,
        "name": "v1.0",
        "baseModel": "Z-Image",
        "trainedWords": ["cnmtc style", "portrait lighting"],
        "description": "<p>A <strong>cinematic</strong> portrait LoRA.</p><p>\(String(repeating: "Extra detail. ", count: 60))</p>",
        "files": [{"name": "model.safetensors", "sizeKB": 200000, "downloadUrl": "https://civitai.com/api/download/models/111", "primary": true}],
        "images": [{"url": "https://image.civitai.com/abc.jpeg"}]
      },
      {
        "id": 222,
        "name": "v2.0 (pruned)",
        "baseModel": "Z-Image",
        "trainedWords": ["cnmtc style v2"],
        "description": "Short plain-text description.",
        "files": [{"name": "model-pruned.safetensors", "sizeKB": 100000, "downloadUrl": "https://civitai.com/api/download/models/222", "primary": false}],
        "images": []
      }
    ]
  }
  """

  private func decodeFixtureModel() throws -> CivitAIModel {
    try JSONDecoder().decode(CivitAIModel.self, from: Data(Self.fixtureJSON.utf8))
  }

  func testExtractionProducesOneEntryPerModelVersion() throws {
    let model = try decodeFixtureModel()
    let entries = CivitAIHarvestExtractor.entries(from: model)
    XCTAssertEqual(entries.count, 2)
    XCTAssertEqual(Set(entries.map(\.sourceVersionId)), Set([111, 222]))
    XCTAssertTrue(entries.allSatisfy { $0.sourceModelId == 12345 })
    XCTAssertTrue(entries.allSatisfy { $0.modelName == "Cinematic Portrait Style" })
  }

  func testExtractionCapturesTrainedWordsPerVersion() throws {
    let model = try decodeFixtureModel()
    let entries = CivitAIHarvestExtractor.entries(from: model)
    let v1 = try XCTUnwrap(entries.first { $0.sourceVersionId == 111 })
    let v2 = try XCTUnwrap(entries.first { $0.sourceVersionId == 222 })
    XCTAssertEqual(v1.trainedWords, ["cnmtc style", "portrait lighting"])
    XCTAssertEqual(v2.trainedWords, ["cnmtc style v2"])
    XCTAssertEqual(v1.baseModel, "Z-Image")
  }

  func testExtractionCapturesModelLevelTagsOnEveryEntry() throws {
    let model = try decodeFixtureModel()
    let entries = CivitAIHarvestExtractor.entries(from: model)
    XCTAssertTrue(entries.allSatisfy { $0.tags == ["portrait", "cinematic", "photography"] })
  }

  func testExtractionStripsHTMLFromDescription() throws {
    let model = try decodeFixtureModel()
    let entries = CivitAIHarvestExtractor.entries(from: model)
    let v1 = try XCTUnwrap(entries.first { $0.sourceVersionId == 111 })
    let excerpt = try XCTUnwrap(v1.descriptionExcerpt)
    XCTAssertFalse(excerpt.contains("<"))
    XCTAssertFalse(excerpt.contains(">"))
    XCTAssertTrue(excerpt.contains("cinematic portrait LoRA"))
  }

  func testExtractionTruncatesDescriptionToTheStoreLimit() throws {
    let model = try decodeFixtureModel()
    let entries = CivitAIHarvestExtractor.entries(from: model)
    let v1 = try XCTUnwrap(entries.first { $0.sourceVersionId == 111 })
    let excerpt = try XCTUnwrap(v1.descriptionExcerpt)
    XCTAssertLessThanOrEqual(excerpt.count, PromptRepositoryEntry.descriptionExcerptLimit)
    // The un-truncated stripped text is long enough that the cap actually bit.
    XCTAssertEqual(excerpt.count, PromptRepositoryEntry.descriptionExcerptLimit)
  }

  func testExtractionKeepsShortPlainDescriptionsIntact() throws {
    let model = try decodeFixtureModel()
    let entries = CivitAIHarvestExtractor.entries(from: model)
    let v2 = try XCTUnwrap(entries.first { $0.sourceVersionId == 222 })
    XCTAssertEqual(v2.descriptionExcerpt, "Short plain-text description.")
  }

  /// Documents the deliberate #234 decision: the harvester does not call
  /// `images(modelVersionId:)` to chase `meta.prompt` (gated, unreliable,
  /// costs an extra request per version) — rawPrompt stays nil from harvest.
  func testExtractionLeavesRawPromptNilByDesign() throws {
    let model = try decodeFixtureModel()
    let entries = CivitAIHarvestExtractor.entries(from: model)
    XCTAssertTrue(entries.allSatisfy { $0.rawPrompt == nil })
  }

  func testExtractionInfersActTaxonomyFromModelTypeAndTags() throws {
    let model = try decodeFixtureModel()
    let entries = CivitAIHarvestExtractor.entries(from: model)
    // "portrait"/"cinematic" tags -> style, per CivitAITaxonomy's keyword match.
    XCTAssertTrue(entries.allSatisfy { $0.actTaxonomy == "style" })
  }

  func testExtractionOnAModelWithNoVersionsProducesNoEntries() throws {
    // CivitAIModel has no synthesized memberwise init (it defines a custom
    // Decodable init), so build the "no versions" case through the decoder,
    // same as production always does.
    let model = try JSONDecoder().decode(CivitAIModel.self, from: Data(#"{"id":1,"name":"Empty"}"#.utf8))
    XCTAssertEqual(CivitAIHarvestExtractor.entries(from: model), [])
  }

  // MARK: - stripHTML helper

  func testStripHTMLHandlesEntitiesAndNilAndEmpty() {
    XCTAssertNil(CivitAIHarvestExtractor.stripHTML(nil))
    XCTAssertNil(CivitAIHarvestExtractor.stripHTML(""))
    XCTAssertNil(CivitAIHarvestExtractor.stripHTML("<p></p>"))
    XCTAssertEqual(
      CivitAIHarvestExtractor.stripHTML("<p>Tom &amp; Jerry&#39;s &quot;show&quot;</p>"),
      "Tom & Jerry's \"show\"")
  }
}
