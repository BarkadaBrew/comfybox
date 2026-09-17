import Foundation
import XCTest

@testable import ZImage

/// Studio Packs migrating into the Library (PRD L8; Todd §11.2 "migrate and
/// retire in v1").
final class LibraryStudioPackImportTests: XCTestCase {

  private var dir: URL!
  private var store: LibraryStore!

  override func setUpWithError() throws {
    dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("studio-lib-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    store = LibraryStore(directory: dir)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: dir)
  }

  private func pack() -> StudioPack {
    StudioPack(
      id: "life-design", name: "Life Design", description: "Healthcare training imagery",
      domain: "healthcare", version: 2,
      promptPrefix: "clinical photography,", promptSuffix: ", neutral background",
      negativePrompt: "cartoon, blurry",
      model: "krea2-raw", steps: 12, guidance: 3.5, scheduler: "euler",
      width: 1024, height: 1536,
      loraStack: [StudioPackLoRARef(loraId: "realism.safetensors", scale: 0.6)],
      cameraAngle: "eye_level", lightingStyle: "softbox",
      templateCategories: ["Clinical"],
      templates: [
        StudioPackTemplate(
          id: "bedside", name: "Bedside", category: "Clinical",
          template: "a {clinician_role} at a patient bedside in {setting}",
          slots: [
            StudioPackTemplateSlot(
              id: "clinician_role", label: "Role", placeholder: "nurse",
              defaultValue: "nurse", options: ["nurse", "physician"]),
          ]),
      ],
      mcpTags: ["training"])
  }

  private func importPack(_ pack: StudioPack, dryRun: Bool = false) throws -> LibraryPackReport {
    var report = LibraryPackReport(
      packId: "studio-packs", packName: "Studio Packs", creator: nil,
      format: "comfybox-studio-pack", dryRun: dryRun)
    try LibraryStudioPackImporter.importPack(pack, into: store, report: &report, dryRun: dryRun)
    return report
  }

  func testAPackBecomesATemplateALookARecipeAndACollection() throws {
    let report = try importPack(pack())
    XCTAssertEqual(report.imported["template"], 1)
    XCTAssertEqual(report.imported["look"], 1)
    XCTAssertEqual(report.imported["recipe"], 1)
    XCTAssertEqual(report.collections, 1)

    let collection = try XCTUnwrap(store.allCollections().first)
    XCTAssertEqual(collection.id, "studio-pack:life-design")
    XCTAssertEqual(collection.membership, "pack")
    for item in store.allItems() {
      XCTAssertEqual(item.packCollections, ["studio-pack:life-design"])
      XCTAssertEqual(item.collections, [], "a migration never files anything as the user's")
    }
  }

  func testTemplateKeepsItsSlotsDefaultsAndOptions() throws {
    _ = try importPack(pack())
    let template = try XCTUnwrap(store.item(id: "studio-pack:life-design#template:bedside"))
    XCTAssertEqual(template.kind, .template)
    let slot = try XCTUnwrap(template.slots?.first { $0.id == "clinician_role" })
    XCTAssertEqual(slot.label, "Role")
    XCTAssertEqual(slot.defaultValue, "nurse")
    XCTAssertEqual(slot.options, ["nurse", "physician"])
    // The body used a second marker the pack never declared; migration declares it.
    XCTAssertTrue(template.slots?.contains { $0.id == "setting" } == true)
    XCTAssertEqual(
      LibraryStore.fill(template: template, values: ["setting": "an ICU"]),
      "a nurse at a patient bedside in an ICU",
      "slot defaults still apply after migration")
  }

  func testLookCarriesThePromptShapeAndRecipeCarriesTheSettings() throws {
    _ = try importPack(pack())
    let look = try XCTUnwrap(store.item(id: "studio-pack:life-design#look"))
    XCTAssertEqual(look.kind, .look)
    XCTAssertEqual(look.promptPrefix, "clinical photography,")
    XCTAssertEqual(look.promptSuffix, ", neutral background")
    XCTAssertEqual(look.negativePrompt, "cartoon, blurry")
    XCTAssertEqual(look.value, "eye_level, softbox", "camera and lighting become its text")
    XCTAssertEqual(look.notes, "Healthcare training imagery")

    let recipe = try XCTUnwrap(store.item(id: "studio-pack:life-design#recipe"))
    XCTAssertEqual(recipe.kind, .recipe)
    XCTAssertEqual(recipe.model, "krea2-raw")
    XCTAssertEqual(recipe.steps, 12)
    XCTAssertEqual(recipe.guidance, 3.5)
    XCTAssertEqual(recipe.width, 1024)
    XCTAssertEqual(recipe.loras?.first?.filename, "realism.safetensors")
    XCTAssertEqual(
      recipe.loras?.first?.scale ?? 0, 0.6, accuracy: 1e-6,
      "the pack stores scale as Float; the library keeps Double")
  }

  func testFacetsDescribeThePack() throws {
    _ = try importPack(pack())
    let template = try XCTUnwrap(store.item(id: "studio-pack:life-design#template:bedside"))
    XCTAssertEqual(template.facets["pack"], ["Life Design"])
    XCTAssertEqual(template.facets["domain"], ["healthcare"])
    XCTAssertEqual(template.facets["category"], ["Clinical"])
    XCTAssertEqual(template.facets["tag"], ["training"])
  }

  func testAPackWithNoRecipeOrLookProducesOnlyTemplates() throws {
    let bare = StudioPack(
      id: "bare", name: "Bare", description: "", domain: "",
      templates: [StudioPackTemplate(id: "t", name: "T", category: "", template: "a photo")])
    let report = try importPack(bare)
    XCTAssertEqual(report.imported["template"], 1)
    XCTAssertNil(report.imported["look"])
    XCTAssertNil(report.imported["recipe"])
  }

  func testMigrationIsIdempotentAndDryRunWritesNothing() throws {
    _ = try importPack(pack())
    let count = store.allItems().count
    _ = try importPack(pack())
    XCTAssertEqual(store.allItems().count, count, "re-running updates in place")

    let fresh = LibraryStore(directory: dir.appendingPathComponent("dry"))
    var report = LibraryPackReport(
      packId: "studio-packs", packName: "Studio Packs", creator: nil,
      format: "comfybox-studio-pack", dryRun: true)
    try LibraryStudioPackImporter.importPack(pack(), into: fresh, report: &report, dryRun: true)
    XCTAssertEqual(report.imported["template"], 1)
    XCTAssertTrue(fresh.allItems().isEmpty)
  }

  func testImportAllReadsTheInstalledPacks() throws {
    // No user pack directory: the built-ins alone must still migrate.
    let report = try LibraryStudioPackImporter.importAll(
      into: store, from: dir.appendingPathComponent("no-such-dir"))
    XCTAssertGreaterThan(report.total, 0, "the built-in pack migrates")
    XCTAssertGreaterThan(report.collections, 0)
    XCTAssertTrue(
      store.allItems().allSatisfy { $0.source.hasPrefix("pack:studio-pack:") },
      "everything records the pack it came from")
  }
}
