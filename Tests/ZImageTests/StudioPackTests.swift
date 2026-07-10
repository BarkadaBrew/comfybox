import XCTest
@testable import ZImage

final class StudioPackTests: XCTestCase {

  // MARK: - Decoding

  func testDecodesFullPackFromJSON() throws {
    let json = """
    {
      "id": "test-pack", "name": "Test Pack", "description": "d", "domain": "test", "version": 2,
      "prompt_prefix": "pre", "prompt_suffix": "suf", "negative_prompt": "neg",
      "model": "z-image-turbo", "steps": 12, "guidance": 1.5, "scheduler": "euler",
      "width": 768, "height": 1152,
      "lora_stack": [{"loraId": "some-lora", "scale": 0.6, "optional": false}],
      "svg_defaults": {"enabled": true, "preset": "logo"},
      "camera_angle": "lowAngle", "camera_orientation": "front", "lighting_style": "soft",
      "template_categories": ["a", "b"],
      "templates": [{"id": "t1", "name": "Template 1", "category": "a", "template": "{x}", "slots": []}],
      "qa_rules": [{"id": "r1", "description": "desc", "required": true}],
      "mcp_tags": ["tag1"]
    }
    """
    let pack = try JSONDecoder().decode(StudioPack.self, from: Data(json.utf8))
    XCTAssertEqual(pack.id, "test-pack")
    XCTAssertEqual(pack.version, 2)
    XCTAssertEqual(pack.promptPrefix, "pre")
    XCTAssertEqual(pack.promptSuffix, "suf")
    XCTAssertEqual(pack.loraStack.first?.loraId, "some-lora")
    XCTAssertEqual(pack.loraStack.first?.optional, false)
    XCTAssertEqual(pack.svgDefaults?.preset, "logo")
    XCTAssertEqual(pack.templateCategories, ["a", "b"])
    XCTAssertEqual(pack.templates.first?.id, "t1")
    XCTAssertEqual(pack.qaRules.first?.required, true)
  }

  func testRoundTripsThroughEncodeDecode() throws {
    let pack = BuiltInStudioPacks.lifeDesignHealthcare
    let data = try JSONEncoder().encode(pack)
    let decoded = try JSONDecoder().decode(StudioPack.self, from: data)
    XCTAssertEqual(pack, decoded)
  }

  // MARK: - Built-in pack defaults

  func testLifeDesignPackHasHealthcareVectorDefaults() {
    let pack = BuiltInStudioPacks.lifeDesignHealthcare
    XCTAssertEqual(pack.id, "life-design-healthcare")
    XCTAssertEqual(pack.domain, "healthcare-training")
    XCTAssertTrue(pack.promptSuffix?.contains("faceless") ?? false)
    XCTAssertTrue(pack.negativePrompt?.contains("photorealistic") ?? false)
    XCTAssertEqual(pack.svgDefaults?.enabled, true)
    XCTAssertEqual(pack.svgDefaults?.preset, "simplified")
    XCTAssertFalse(pack.templateCategories.isEmpty)
    XCTAssertFalse(pack.qaRules.isEmpty)
    // Ships without a dedicated LoRA — prompt/SVG defaults only (per PRD open question).
    XCTAssertTrue(pack.loraStack.isEmpty)
  }

  // MARK: - Prompt composition

  func testComposePromptJoinsPrefixSubjectSuffix() {
    let pack = StudioPack(
      id: "p", name: "P", description: "", domain: "d",
      promptPrefix: "prefix", promptSuffix: "suffix"
    )
    XCTAssertEqual(pack.composePrompt(subject: "a cat"), "prefix, a cat, suffix")
  }

  func testComposePromptHandlesEmptySubjectAndMissingParts() {
    let pack = StudioPack(id: "p", name: "P", description: "", domain: "d", promptSuffix: "suffix")
    XCTAssertEqual(pack.composePrompt(subject: ""), "suffix")

    let bare = StudioPack(id: "p", name: "P", description: "", domain: "d")
    XCTAssertEqual(bare.composePrompt(subject: "x"), "x")
  }

  // MARK: - Resolving against the local LoRA library (missing-optional-LoRA behavior)

  func testResolveDropsMissingOptionalLoraWithWarning() {
    let pack = StudioPack(
      id: "p", name: "P", description: "", domain: "d",
      loraStack: [StudioPackLoRARef(loraId: "not-installed", scale: 1.0, optional: true)]
    )
    let recipe = StudioPackResolver.resolve(pack: pack, availableLoraIds: [])
    XCTAssertTrue(recipe.loras.isEmpty)
    XCTAssertEqual(recipe.warnings.count, 1)
    XCTAssertTrue(recipe.warnings[0].contains("not-installed"))
  }

  func testResolveFlagsMissingRequiredLoraButDoesNotThrow() {
    let pack = StudioPack(
      id: "p", name: "P", description: "", domain: "d",
      loraStack: [StudioPackLoRARef(loraId: "must-have", scale: 1.0, optional: false)]
    )
    let recipe = StudioPackResolver.resolve(pack: pack, availableLoraIds: [])
    XCTAssertTrue(recipe.loras.isEmpty)
    XCTAssertTrue(recipe.warnings[0].contains("required"))
  }

  func testResolveKeepsLoraWhenInstalled() {
    let pack = StudioPack(
      id: "p", name: "P", description: "", domain: "d",
      loraStack: [StudioPackLoRARef(loraId: "installed-lora", scale: 0.8)]
    )
    let recipe = StudioPackResolver.resolve(pack: pack, availableLoraIds: ["installed-lora"])
    XCTAssertEqual(recipe.loras.count, 1)
    XCTAssertTrue(recipe.warnings.isEmpty)
  }

  func testResolveLifeDesignPackWithNoLorasProducesNoWarnings() {
    // The built-in pack ships with an empty LoRA stack — resolving it must
    // never produce a "missing LoRA" warning.
    let recipe = StudioPackResolver.resolve(
      pack: BuiltInStudioPacks.lifeDesignHealthcare, subject: "CPR training", availableLoraIds: []
    )
    XCTAssertTrue(recipe.warnings.isEmpty)
    XCTAssertTrue(recipe.prompt.contains("CPR training"))
    XCTAssertTrue(recipe.prompt.contains("faceless"))
  }

  // MARK: - Template rendering (FR-2 / #198)

  func testTemplateRendersProvidedSlotValues() {
    let template = StudioPackTemplate(
      id: "t", name: "T", category: "c",
      template: "{clinician_role} performing CPR on {patient_role} in {setting}",
      slots: [
        StudioPackTemplateSlot(id: "clinician_role", label: "Clinician", defaultValue: "a nurse"),
        StudioPackTemplateSlot(id: "patient_role", label: "Patient", defaultValue: "a patient"),
        StudioPackTemplateSlot(id: "setting", label: "Setting", defaultValue: "a hospital room"),
      ]
    )
    let rendered = template.render(slotValues: [
      "clinician_role": "a paramedic", "patient_role": "an elderly man", "setting": "an ambulance",
    ])
    XCTAssertEqual(rendered, "a paramedic performing CPR on an elderly man in an ambulance")
  }

  func testTemplateFallsBackToDefaultsForMissingOrBlankSlots() {
    let template = StudioPackTemplate(
      id: "t", name: "T", category: "c",
      template: "{role} in {place}",
      slots: [
        StudioPackTemplateSlot(id: "role", label: "Role", defaultValue: "a nurse"),
        StudioPackTemplateSlot(id: "place", label: "Place", defaultValue: "a clinic"),
      ]
    )
    // Missing key entirely, and blank/whitespace-only value both fall back to default.
    XCTAssertEqual(template.render(slotValues: ["place": "   "]), "a nurse in a clinic")
    XCTAssertEqual(template.render(slotValues: [:]), "a nurse in a clinic")
  }

  func testTemplateLeavesUnrecognizedPlaceholdersAsIs() {
    let template = StudioPackTemplate(
      id: "t", name: "T", category: "c",
      template: "{known} and {unknown}",
      slots: [StudioPackTemplateSlot(id: "known", label: "Known", defaultValue: "X")]
    )
    XCTAssertEqual(template.render(slotValues: [:]), "X and {unknown}")
  }

  func testLifeDesignPackShipsFiveTemplatesMatchingCategories() {
    let pack = BuiltInStudioPacks.lifeDesignHealthcare
    let templateIds = Set(pack.templates.map { $0.id })
    XCTAssertEqual(templateIds, Set(pack.templateCategories))
  }

  // MARK: - Resolving through a template (composer)

  func testResolveWithTemplateRendersAndComposesWithPackStyle() throws {
    let pack = BuiltInStudioPacks.lifeDesignHealthcare
    let template = try XCTUnwrap(pack.templates.first { $0.id == "clinical-handoff" })
    let recipe = StudioPackResolver.resolve(
      pack: pack, template: template,
      slotValues: ["procedure": "medication reconciliation", "clinician_role": "a pharmacist", "patient_role": "an outpatient"],
      availableLoraIds: []
    )
    XCTAssertEqual(recipe.templateId, "clinical-handoff")
    XCTAssertTrue(recipe.prompt.contains("medication reconciliation training scene"))
    XCTAssertTrue(recipe.prompt.contains("a pharmacist assisting an outpatient"))
    // Pack style layer still applied on top of the rendered template.
    XCTAssertTrue(recipe.prompt.contains("faceless"))
    XCTAssertTrue(recipe.warnings.isEmpty)
  }

  func testResolveWithSubjectLeavesTemplateIdNil() {
    let recipe = StudioPackResolver.resolve(
      pack: BuiltInStudioPacks.lifeDesignHealthcare, subject: "x", availableLoraIds: []
    )
    XCTAssertNil(recipe.templateId)
  }

  // MARK: - Library loading (built-in vs. user override)

  func testLoadAllReturnsBuiltInsWhenNoUserDirectoryExists() {
    let missingDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("studio-packs-test-\(UUID().uuidString)")
    let (packs, errors) = StudioPackLibrary.loadAll(from: missingDir)
    XCTAssertTrue(errors.isEmpty)
    XCTAssertTrue(packs.contains { $0.id == "life-design-healthcare" })
  }

  func testUserPackOverridesBuiltInWithMatchingId() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("studio-packs-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let override = StudioPack(
      id: "life-design-healthcare", name: "Custom Life Design", description: "override", domain: "healthcare-training"
    )
    let data = try JSONEncoder().encode(override)
    try data.write(to: dir.appendingPathComponent("life-design-healthcare.json"))

    let (packs, errors) = StudioPackLibrary.loadAll(from: dir)
    XCTAssertTrue(errors.isEmpty)
    let loaded = try XCTUnwrap(packs.first { $0.id == "life-design-healthcare" })
    XCTAssertEqual(loaded.name, "Custom Life Design")
  }

  func testUserDirectoryCanAddANewPackAlongsideBuiltIns() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("studio-packs-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    let extra = StudioPack(id: "my-custom-pack", name: "Mine", description: "", domain: "custom")
    try JSONEncoder().encode(extra).write(to: dir.appendingPathComponent("mine.json"))

    let (packs, _) = StudioPackLibrary.loadAll(from: dir)
    XCTAssertTrue(packs.contains { $0.id == "life-design-healthcare" })
    XCTAssertTrue(packs.contains { $0.id == "my-custom-pack" })
  }

  func testInvalidUserPackFileIsReportedNotThrown() throws {
    let dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("studio-packs-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }

    try Data("not valid json".utf8).write(to: dir.appendingPathComponent("broken.json"))

    let (packs, errors) = StudioPackLibrary.loadAll(from: dir)
    XCTAssertEqual(errors.count, 1)
    // Built-ins still load fine despite the broken user file.
    XCTAssertTrue(packs.contains { $0.id == "life-design-healthcare" })
  }
}
