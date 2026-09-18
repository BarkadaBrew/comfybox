import Foundation
import Testing

@testable import ComfyBoxDesktop

/// Assembling a prompt from library material (PRD-creative-library L4). The
/// pure parts: query URLs, look application, insertion, slot routing.
@Suite("Creative Library in Generate")
struct LibraryPromptTests {

    // MARK: query building

    @Test("a query is stable and escaped")
    func queryPath() {
        var spec = LibraryQuerySpec(kinds: ["component", "wardrobe_item"], text: "neon alley")
        spec.facets = ["mood": ["Calm", "Bold"]]
        spec.category = "Tops & Layers"
        spec.minRating = 4
        spec.favoritesOnly = true
        let path = spec.path
        #expect(path.hasPrefix("/v1/library/items?"))
        #expect(path.contains("kind=component,wardrobe_item"))
        #expect(path.contains("facet.mood=Calm,Bold"))
        #expect(path.contains("q=neon%20alley"))
        #expect(path.contains("min_rating=4"))
        #expect(path.contains("favorites=true"))
        // Same spec, same URL — order never wobbles.
        #expect(spec.path == path)
    }

    @Test("ampersands in a search term cannot start a new parameter")
    func queryEscaping() {
        let spec = LibraryQuerySpec(kinds: [], text: "a&b=c")
        #expect(!spec.path.contains("q=a&b=c"))
        #expect(spec.path.contains("q=a%26b%3Dc"))
    }

    @Test("an empty search term is omitted rather than sent blank")
    func blankSearchOmitted() {
        var spec = LibraryQuerySpec()
        spec.text = "   "
        #expect(!spec.path.contains("q="))
    }

    // MARK: looks

    private func look(prefix: String? = nil, suffix: String? = nil, value: String = "") -> LibraryItemDTO {
        var item = LibraryItemDTO(id: "l1", kind: "look", name: "Analog", value: value)
        item.promptPrefix = prefix
        item.promptSuffix = suffix
        return item
    }

    @Test("a look wraps the prompt body")
    func applyLook() {
        let result = LibraryPromptModel.applyLook(
            look(prefix: "shot on film", suffix: "grainy, warm", value: "soft window light"),
            to: "a woman at a counter")
        #expect(result == "shot on film, a woman at a counter, soft window light, grainy, warm")
    }

    @Test("a look with only a suffix does not leave a leading comma")
    func applyLookSuffixOnly() {
        #expect(LibraryPromptModel.applyLook(look(suffix: "warm"), to: "a cat") == "a cat, warm")
    }

    @Test("applying a look to an empty prompt yields just the look")
    func applyLookToEmpty() {
        #expect(LibraryPromptModel.applyLook(look(prefix: "film"), to: "   ") == "film")
    }

    // MARK: insertion

    @Test("inserting a component appends with one separator")
    func insertComponent() {
        #expect(LibraryPromptModel.insert("a neon alley", into: "a woman") == "a woman, a neon alley")
        #expect(
            LibraryPromptModel.insert("a neon alley", into: "a woman,") == "a woman, a neon alley",
            "an existing comma is not doubled")
        #expect(LibraryPromptModel.insert("a neon alley", into: "") == "a neon alley")
        #expect(LibraryPromptModel.insert("  ", into: "a woman") == "a woman")
    }

    // MARK: templates

    @Test("choosing a template seeds its slot defaults")
    @MainActor
    func templateDefaults() {
        let model = LibraryPromptModel()
        var template = LibraryItemDTO(
            id: "t1", kind: "template", name: "Portrait", value: "wearing {OUTFIT} in {SCENE}")
        template.slots = [
            LibrarySlotDTO(id: "OUTFIT"),
            LibrarySlotDTO(id: "SCENE", defaultValue: "a quiet cafe"),
        ]
        model.chooseTemplate(template)
        #expect(model.slotValues["SCENE"] == "a quiet cafe")
        #expect(model.slotValues["OUTFIT"] == "")
        #expect(model.usedItemIds.contains("t1"))
        model.clearTemplate()
        #expect(model.template == nil)
        #expect(model.slotValues.isEmpty)
    }

    @Test("a slot opens the picker that fits it")
    @MainActor
    func slotPickerRouting() {
        let section = LibraryPromptSection(
            model: LibraryPromptModel(),
            client: LibraryClient(engine: EngineService()),
            prompt: .constant(""), negativePrompt: .constant(""), isExpanded: .constant(true))
        #expect(section.pickerMode(for: "OUTFIT") == .outfits)
        #expect(section.pickerMode(for: "scene") == .components(kind: "scene"))
        #expect(section.pickerMode(for: "LIGHTING") == .components(kind: "lighting"))
        #expect(section.pickerMode(for: "BRAND") == .components(kind: nil))
    }

    @Test("saving a template declares every marker its body uses")
    func slotScanner() {
        let markers = LibrarySlotScanner.markers(in: "wearing {OUTFIT} in {SCENE}, {OUTFIT} again")
        #expect(markers == ["OUTFIT", "SCENE"])
        #expect(LibrarySlotScanner.markers(in: "no slots here").isEmpty)
    }

    // MARK: rows

    @Test("a picker row says what an item is")
    func subtitles() {
        var wardrobe = LibraryItemDTO(id: "w", kind: "wardrobe_item", name: "latex bustier", value: "x")
        wardrobe.category = "Bodywear"
        wardrobe.subtype = "Corsets"
        #expect(wardrobe.subtitle == "Bodywear › Corsets")

        var template = LibraryItemDTO(id: "t", kind: "template", name: "P", value: "x")
        template.slots = [LibrarySlotDTO(id: "OUTFIT"), LibrarySlotDTO(id: "NAME")]
        #expect(template.subtitle == "OUTFIT, NAME")

        var recipe = LibraryItemDTO(id: "r", kind: "recipe", name: "R", value: "")
        recipe.model = "kroma-v0.3-base"
        recipe.steps = 8
        #expect(recipe.subtitle == "kroma-v0.3-base · 8 steps")
    }

    @Test("the wire shape decodes, including snake_case and missing fields")
    func decoding() throws {
        let json = """
            [{"id":"w1","kind":"wardrobe_item","name":"tee","value":"a tee",
              "item_type":"piece","pack_collections":["c1"],"use_count":3,
              "component_kind":null,"facets":{"mood":["Calm"]}}]
            """
        let items = try JSONDecoder().decode([LibraryItemDTO].self, from: Data(json.utf8))
        let item = try #require(items.first)
        #expect(item.itemType == "piece")
        #expect(item.packCollections == ["c1"])
        #expect(item.useCount == 3)
        #expect(item.facets["mood"] == ["Calm"])
        #expect(item.rating == 0, "absent fields fall back, they do not fail the decode")
    }
}
