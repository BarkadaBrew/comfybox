// LibraryClient.swift — the desktop's view of the Creative Library
// (PRD docs/PRD-creative-library.md, L4).
//
// The engine owns the library (LibraryStore + /v1/library/*), so the app holds
// no second copy: it queries, fills templates, and writes back. Query building
// and decoding are pure and tested; only `fetch`/`send` touch the network.

import Foundation

// MARK: - Wire types

public struct LibrarySlotDTO: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var label: String?
    public var placeholder: String?
    public var defaultValue: String?
    public var options: [String]?

    enum CodingKeys: String, CodingKey {
        case id, label, placeholder, options
        case defaultValue = "default"
    }

    public init(
        id: String, label: String? = nil, placeholder: String? = nil,
        defaultValue: String? = nil, options: [String]? = nil
    ) {
        self.id = id
        self.label = label
        self.placeholder = placeholder
        self.defaultValue = defaultValue
        self.options = options
    }
}

public struct LibraryItemDTO: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var kind: String
    public var name: String
    public var value: String
    public var facets: [String: [String]]
    public var collections: [String]
    public var packCollections: [String]
    public var rating: Int
    public var favorite: Bool
    public var archived: Bool
    public var notes: String?
    public var previewRef: String?
    public var source: String
    public var useCount: Int
    public var slots: [LibrarySlotDTO]?
    public var componentKind: String?
    public var category: String?
    public var subtype: String?
    public var itemType: String?
    public var items: [String]?
    public var promptPrefix: String?
    public var promptSuffix: String?
    public var negativePrompt: String?
    public var model: String?
    public var steps: Int?
    public var guidance: Double?
    public var width: Int?
    public var height: Int?

    enum CodingKeys: String, CodingKey {
        case id, kind, name, value, facets, collections, rating, favorite, archived, notes
        case source, slots, category, subtype, items, model, steps, guidance, width, height
        case packCollections = "pack_collections"
        case previewRef = "preview_ref"
        case useCount = "use_count"
        case componentKind = "component_kind"
        case itemType = "item_type"
        case promptPrefix = "prompt_prefix"
        case promptSuffix = "prompt_suffix"
        case negativePrompt = "negative_prompt"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        kind = try c.decode(String.self, forKey: .kind)
        name = try c.decode(String.self, forKey: .name)
        value = try c.decodeIfPresent(String.self, forKey: .value) ?? ""
        facets = try c.decodeIfPresent([String: [String]].self, forKey: .facets) ?? [:]
        collections = try c.decodeIfPresent([String].self, forKey: .collections) ?? []
        packCollections = try c.decodeIfPresent([String].self, forKey: .packCollections) ?? []
        rating = try c.decodeIfPresent(Int.self, forKey: .rating) ?? 0
        favorite = try c.decodeIfPresent(Bool.self, forKey: .favorite) ?? false
        archived = try c.decodeIfPresent(Bool.self, forKey: .archived) ?? false
        notes = try c.decodeIfPresent(String.self, forKey: .notes)
        previewRef = try c.decodeIfPresent(String.self, forKey: .previewRef)
        source = try c.decodeIfPresent(String.self, forKey: .source) ?? "local"
        useCount = try c.decodeIfPresent(Int.self, forKey: .useCount) ?? 0
        slots = try c.decodeIfPresent([LibrarySlotDTO].self, forKey: .slots)
        componentKind = try c.decodeIfPresent(String.self, forKey: .componentKind)
        category = try c.decodeIfPresent(String.self, forKey: .category)
        subtype = try c.decodeIfPresent(String.self, forKey: .subtype)
        itemType = try c.decodeIfPresent(String.self, forKey: .itemType)
        items = try c.decodeIfPresent([String].self, forKey: .items)
        promptPrefix = try c.decodeIfPresent(String.self, forKey: .promptPrefix)
        promptSuffix = try c.decodeIfPresent(String.self, forKey: .promptSuffix)
        negativePrompt = try c.decodeIfPresent(String.self, forKey: .negativePrompt)
        model = try c.decodeIfPresent(String.self, forKey: .model)
        steps = try c.decodeIfPresent(Int.self, forKey: .steps)
        guidance = try c.decodeIfPresent(Double.self, forKey: .guidance)
        width = try c.decodeIfPresent(Int.self, forKey: .width)
        height = try c.decodeIfPresent(Int.self, forKey: .height)
    }

    public init(id: String, kind: String, name: String, value: String) {
        self.id = id
        self.kind = kind
        self.name = name
        self.value = value
        facets = [:]
        collections = []
        packCollections = []
        rating = 0
        favorite = false
        archived = false
        source = "local"
        useCount = 0
    }

    /// What the user sees in a picker row's subtitle.
    public var subtitle: String {
        switch kind {
        case "wardrobe_item":
            return [category, subtype].compactMap { $0 }.joined(separator: " › ")
        case "template":
            let ids = (slots ?? []).map(\.id)
            return ids.isEmpty ? "no slots" : ids.joined(separator: ", ")
        case "recipe":
            return [model, steps.map { "\($0) steps" }].compactMap { $0 }.joined(separator: " · ")
        default:
            return componentKind ?? ""
        }
    }
}

public struct LibraryCollectionDTO: Codable, Sendable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var color: String?
    public var parentId: String?
    public var displayOrder: Int
    public var membership: String

    enum CodingKeys: String, CodingKey {
        case id, name, color, membership
        case parentId = "parent_id"
        case displayOrder = "display_order"
    }
}

public struct LibraryFillResult: Codable, Sendable, Equatable {
    public let prompt: String
    public let unfilled: [String]
}

// MARK: - Query

public struct LibraryQuerySpec: Sendable, Equatable {
    public var kinds: [String] = []
    public var facets: [String: [String]] = [:]
    public var collection: String?
    public var componentKind: String?
    public var category: String?
    public var text: String?
    public var minRating: Int?
    public var favoritesOnly = false
    public var order = "recent"
    public var limit = 200

    public init(kinds: [String] = [], text: String? = nil) {
        self.kinds = kinds
        self.text = text
    }

    /// `/v1/library/items?kind=a,b&facet.mood=Calm&q=…`. Sorted so the same
    /// query always produces the same URL (cheap to cache, easy to test).
    public var path: String {
        var pairs: [(String, String)] = []
        if !kinds.isEmpty { pairs.append(("kind", kinds.joined(separator: ","))) }
        for (axis, values) in facets where !values.isEmpty {
            pairs.append(("facet.\(axis)", values.joined(separator: ",")))
        }
        if let collection { pairs.append(("collection", collection)) }
        if let componentKind { pairs.append(("component_kind", componentKind)) }
        if let category { pairs.append(("category", category)) }
        if let text, !text.trimmingCharacters(in: .whitespaces).isEmpty {
            pairs.append(("q", text.trimmingCharacters(in: .whitespaces)))
        }
        if let minRating { pairs.append(("min_rating", String(minRating))) }
        if favoritesOnly { pairs.append(("favorites", "true")) }
        pairs.append(("order", order))
        pairs.append(("limit", String(limit)))

        let encoded = pairs
            .sorted { $0.0 == $1.0 ? $0.1 < $1.1 : $0.0 < $1.0 }
            .map { key, value in
                let k = key.addingPercentEncoding(withAllowedCharacters: .urlQueryValueAllowed) ?? key
                let v = value.addingPercentEncoding(withAllowedCharacters: .urlQueryValueAllowed) ?? value
                return "\(k)=\(v)"
            }
            .joined(separator: "&")
        return "/v1/library/items?\(encoded)"
    }
}

extension CharacterSet {
    /// `&`, `=` and `+` must not survive into a query value.
    static let urlQueryValueAllowed: CharacterSet = {
        var set = CharacterSet.urlQueryAllowed
        set.remove(charactersIn: "&=+?#")
        return set
    }()
}

// MARK: - Client

@MainActor
@Observable
public final class LibraryClient {
    public private(set) var items: [LibraryItemDTO] = []
    public private(set) var collections: [LibraryCollectionDTO] = []
    public private(set) var facets: [String: [String: Int]] = [:]
    public private(set) var isLoading = false
    public private(set) var lastError: String?

    private let engine: EngineService

    public init(engine: EngineService) {
        self.engine = engine
    }

    private static func decoder() -> JSONDecoder { JSONDecoder() }

    public func search(_ spec: LibraryQuerySpec) async {
        isLoading = true
        defer { isLoading = false }
        do {
            let data = try await engine.libraryGet(spec.path)
            items = try Self.decoder().decode([LibraryItemDTO].self, from: data)
            lastError = nil
        } catch {
            lastError = error.localizedDescription
        }
    }

    public func loadCollections() async {
        guard let data = try? await engine.libraryGet("/v1/library/collections") else { return }
        collections = (try? Self.decoder().decode([LibraryCollectionDTO].self, from: data)) ?? []
    }

    public func loadFacets() async {
        guard let data = try? await engine.libraryGet("/v1/library/facets") else { return }
        facets = (try? Self.decoder().decode([String: [String: Int]].self, from: data)) ?? [:]
    }

    /// Fill a template server-side, so the desktop and every agent get the
    /// same prompt from the same slots.
    public func fill(
        templateId: String, values: [String: String], outfitId: String? = nil
    ) async -> LibraryFillResult? {
        var body: [String: Any] = ["template_id": templateId, "values": values]
        if let outfitId { body["outfit_id"] = outfitId }
        guard let payload = try? JSONSerialization.data(withJSONObject: body),
              let data = try? await engine.libraryPost("/v1/library/fill", body: payload)
        else { return nil }
        return try? Self.decoder().decode(LibraryFillResult.self, from: data)
    }

    /// Record that an item was used. Fire-and-forget: a failed count must
    /// never interrupt a render.
    public func markUsed(_ id: String) {
        let encoded = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
        Task { [engine] in
            _ = try? await engine.libraryPost("/v1/library/used/\(encoded)", body: Data("{}".utf8))
        }
    }

    @discardableResult
    public func save(_ item: LibraryItemDTO) async -> Bool {
        guard let payload = try? JSONEncoder().encode(LibraryUpsertBody(item)) else { return false }
        do {
            _ = try await engine.libraryPost("/v1/library/items", body: payload)
            return true
        } catch {
            lastError = error.localizedDescription
            return false
        }
    }
}

/// The write shape: only the fields a client owns (the engine keeps
/// `created_at`, `use_count` and provenance).
struct LibraryUpsertBody: Encodable {
    let id: String
    let kind: String
    let name: String
    let value: String
    let facets: [String: [String]]
    let collections: [String]
    let rating: Int
    let favorite: Bool
    let notes: String?
    let category: String?
    let subtype: String?
    let itemType: String?
    let componentKind: String?
    let items: [String]?
    let slots: [LibrarySlotDTO]?
    let promptPrefix: String?
    let promptSuffix: String?
    let negativePrompt: String?
    let source: String

    enum CodingKeys: String, CodingKey {
        case id, kind, name, value, facets, collections, rating, favorite, notes
        case category, subtype, items, slots, source
        case itemType = "item_type"
        case componentKind = "component_kind"
        case promptPrefix = "prompt_prefix"
        case promptSuffix = "prompt_suffix"
        case negativePrompt = "negative_prompt"
    }

    init(_ item: LibraryItemDTO) {
        id = item.id
        kind = item.kind
        name = item.name
        value = item.value
        facets = item.facets
        collections = item.collections
        rating = item.rating
        favorite = item.favorite
        notes = item.notes
        category = item.category
        subtype = item.subtype
        itemType = item.itemType
        componentKind = item.componentKind
        items = item.items
        slots = item.slots
        promptPrefix = item.promptPrefix
        promptSuffix = item.promptSuffix
        negativePrompt = item.negativePrompt
        source = item.source
    }
}
