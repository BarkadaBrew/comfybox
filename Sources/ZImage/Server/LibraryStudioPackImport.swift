// LibraryStudioPackImport.swift — Studio Packs become Library material
// (PRD docs/PRD-creative-library.md, L8; Todd's decision §11.2: migrate and
// retire in v1).
//
// A Studio Pack (docs/prd-comfybox-studio-packs.md) is a read-only bundle of
// slot templates plus a production recipe: prompt prefix/suffix, negative,
// model and sampler defaults, a LoRA stack and QA rules. Every one of those
// has a home in the Library:
//
//   pack.templates  -> `template` items, slots and all (the shapes already match)
//   prefix/suffix/negative/camera/lighting -> ONE `look` item per pack
//   model/steps/guidance/scheduler/dims/LoRAs -> ONE `recipe` item per pack
//   pack itself -> a `collection`, so a pack's material stays browsable together
//
// Migration is idempotent: ids are derived from the pack id, so re-running
// updates in place. Pack-owned collection membership lands in
// `packCollections`, which is what lets Todd file these anywhere he likes
// without the next migration undoing it.

import Foundation

public enum LibraryStudioPackImporter {

  /// Import every installed Studio Pack (built-ins plus `~/.comfybox/studio-packs`).
  @discardableResult
  public static func importAll(
    into store: LibraryStore,
    from directory: URL = StudioPackLibrary.defaultPackDirectory,
    dryRun: Bool = false
  ) throws -> LibraryPackReport {
    let loaded = StudioPackLibrary.loadAll(from: directory)
    var report = LibraryPackReport(
      packId: "studio-packs", packName: "Studio Packs", creator: nil,
      format: "comfybox-studio-pack", dryRun: dryRun)
    for error in loaded.errors {
      report.skipped.append(.init(what: "studio pack", reason: error.localizedDescription))
    }
    for pack in loaded.packs {
      try importPack(pack, into: store, report: &report, dryRun: dryRun)
    }
    return report
  }

  /// Import one pack. Exposed for tests and for a future "install this pack" action.
  public static func importPack(
    _ pack: StudioPack, into store: LibraryStore, report: inout LibraryPackReport,
    dryRun: Bool = false
  ) throws {
    let collectionId = "studio-pack:\(pack.id)"
    if !dryRun {
      try store.upsertCollection(LibraryCollection(
        id: collectionId, name: pack.name, parentId: nil, displayOrder: 0, membership: "pack"))
    }
    report.collections += 1

    func add(_ item: LibraryEntry, what: String) {
      guard !dryRun else {
        report.imported[item.kind.rawValue, default: 0] += 1
        return
      }
      do {
        try store.upsert(item)
        report.imported[item.kind.rawValue, default: 0] += 1
      } catch {
        report.skipped.append(.init(what: what, reason: error.localizedDescription))
      }
    }

    // Templates: `{slotId}` markers and slot defaults already match ours.
    for template in pack.templates {
      var item = LibraryEntry(
        id: "studio-pack:\(pack.id)#template:\(template.id)",
        kind: .template,
        name: template.name,
        value: template.template,
        facets: facets(for: pack, category: template.category),
        packCollections: [collectionId],
        source: "pack:studio-pack:\(pack.id)",
        slots: template.slots.map {
          LibrarySlot(
            id: $0.id,
            label: $0.label.isEmpty ? $0.id : $0.label,
            placeholder: $0.placeholder.isEmpty ? nil : $0.placeholder,
            defaultValue: $0.defaultValue.isEmpty ? nil : $0.defaultValue,
            options: $0.options.isEmpty ? nil : $0.options)
        })
      // A pack template may use a marker it never declared. The Library
      // refuses that (an undeclared slot is a silent blank waiting to happen),
      // so declare what the body actually uses.
      let used = LibraryStore.slotMarkers(in: template.template)
      let declared = Set(item.slots?.map(\.id) ?? [])
      for missing in used.subtracting(declared).sorted() {
        item.slots?.append(LibrarySlot(id: missing, label: missing))
      }
      add(item, what: "template \(template.id)")
    }

    // The look: everything about how a pack's prompts read.
    let hasLook = [pack.promptPrefix, pack.promptSuffix, pack.negativePrompt,
                   pack.cameraAngle, pack.cameraOrientation, pack.lightingStyle]
      .contains { ($0?.isEmpty == false) }
    if hasLook {
      var value = [pack.cameraAngle, pack.cameraOrientation, pack.lightingStyle]
        .compactMap { $0 }
        .filter { !$0.isEmpty }
        .joined(separator: ", ")
      if value.isEmpty { value = pack.promptSuffix ?? pack.promptPrefix ?? "" }
      let look = LibraryEntry(
        id: "studio-pack:\(pack.id)#look",
        kind: .look,
        name: "\(pack.name) look",
        value: value,
        facets: facets(for: pack, category: nil),
        packCollections: [collectionId],
        notes: pack.description.isEmpty ? nil : pack.description,
        source: "pack:studio-pack:\(pack.id)",
        promptPrefix: pack.promptPrefix,
        promptSuffix: pack.promptSuffix,
        negativePrompt: pack.negativePrompt)
      add(look, what: "look for \(pack.id)")
    }

    // The recipe: the render settings the pack recommends.
    let hasRecipe = pack.model != nil || pack.steps != nil || pack.guidance != nil
      || pack.width != nil || !pack.loraStack.isEmpty
    if hasRecipe {
      let recipe = LibraryEntry(
        id: "studio-pack:\(pack.id)#recipe",
        kind: .recipe,
        name: "\(pack.name) recipe",
        value: "",
        packCollections: [collectionId],
        source: "pack:studio-pack:\(pack.id)",
        loras: pack.loraStack.map { LoraReference(filename: $0.loraId, scale: Double($0.scale)) },
        model: pack.model,
        steps: pack.steps,
        guidance: pack.guidance.map { Double($0) },
        width: pack.width,
        height: pack.height)
      add(recipe, what: "recipe for \(pack.id)")
    }
  }

  static func facets(for pack: StudioPack, category: String?) -> [String: [String]] {
    var out: [String: [String]] = ["pack": [pack.name]]
    if !pack.domain.isEmpty { out["domain"] = [pack.domain] }
    if let category, !category.isEmpty { out["category"] = [category] }
    if !pack.mcpTags.isEmpty { out["tag"] = pack.mcpTags }
    return out
  }
}
