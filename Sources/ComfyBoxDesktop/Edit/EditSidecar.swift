// EditSidecar.swift — the `edit` block in a derived asset's adjacent sidecar
//
// Desktop convention: `<image basename>.json` next to the image (see
// AssetIngestor.readSidecar). The exporter writes generation fields the
// ingestor already understands plus this block; this file reads it back.

import Foundation

public struct EditSidecar: Codable, Equatable, Sendable {
    public var version: Int
    public var sourcePath: String
    public var sourceAssetId: String?
    public var recipe: EditRecipe
    public var editor: String
    public var createdAt: Date

    public init(version: Int, sourcePath: String, sourceAssetId: String?, recipe: EditRecipe, editor: String, createdAt: Date) {
        self.version = version; self.sourcePath = sourcePath; self.sourceAssetId = sourceAssetId
        self.recipe = recipe; self.editor = editor; self.createdAt = createdAt
    }

    enum CodingKeys: String, CodingKey {
        case version, recipe, editor
        case sourcePath = "source_path"
        case sourceAssetId = "source_asset_id"
        case createdAt = "created_at"
    }

    // Hand-written so `source_asset_id` always appears in the JSON — as an
    // explicit `null` when absent — rather than being dropped by the
    // synthesized `encodeIfPresent` an Optional stored property would get by
    // default. The `edit` block is a fixed six-key contract; a caller
    // scanning for `source_asset_id` should never have to distinguish
    // "absent" from "null".
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(version, forKey: .version)
        try c.encode(sourcePath, forKey: .sourcePath)
        if let id = sourceAssetId {
            try c.encode(id, forKey: .sourceAssetId)
        } else {
            try c.encodeNil(forKey: .sourceAssetId)
        }
        try c.encode(recipe, forKey: .recipe)
        try c.encode(editor, forKey: .editor)
        try c.encode(createdAt, forKey: .createdAt)
    }

    static let decoder: JSONDecoder = { let d = JSONDecoder(); d.dateDecodingStrategy = .iso8601; return d }()
    static let encoder: JSONEncoder = { let e = JSONEncoder(); e.dateEncodingStrategy = .iso8601; return e }()

    public static func sidecarPath(forImageAt imagePath: String) -> String {
        ((imagePath as NSString).deletingPathExtension) + ".json"
    }

    public static func read(forImageAt imagePath: String) -> EditSidecar? {
        guard let data = FileManager.default.contents(atPath: sidecarPath(forImageAt: imagePath)),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let edit = obj["edit"],
              let editData = try? JSONSerialization.data(withJSONObject: edit)
        else { return nil }
        return try? decoder.decode(EditSidecar.self, from: editData)
    }

    /// Parses only the envelope fields (`version`, `source_path`, `source_asset_id`) out of
    /// the `edit` block, without decoding `recipe`. A sidecar written by a newer ComfyBox can
    /// carry a `recipe` shape this build's `EditRecipe` cannot decode — `read(forImageAt:)`
    /// would then fail outright and its caller could mistake "sidecar exists but is newer than
    /// we understand" for "no sidecar at all". Callers should check this envelope's `version`
    /// first and only fall through to `read(forImageAt:)` for a full decode once it is known to
    /// be compatible.
    public static func readEnvelope(forImageAt imagePath: String) -> (version: Int, sourcePath: String, sourceAssetId: String?)? {
        guard let data = FileManager.default.contents(atPath: sidecarPath(forImageAt: imagePath)),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let edit = obj["edit"] as? [String: Any],
              let version = edit[CodingKeys.version.rawValue] as? Int,
              let sourcePath = edit[CodingKeys.sourcePath.rawValue] as? String
        else { return nil }
        let sourceAssetId = edit[CodingKeys.sourceAssetId.rawValue] as? String
        return (version, sourcePath, sourceAssetId)
    }

    /// Follows the `source_path` chain to the root original. Unbounded in the
    /// number of hops it will follow (an edit chain has no fixed depth
    /// limit), but a chain that revisits a path it has already seen is
    /// malformed — a fixed-count hop cap would silently stop at an arbitrary
    /// point in a cycle, so instead this tracks every path it has visited
    /// and, on a repeat, reports the failure explicitly: the starting image
    /// itself with a `nil` asset id, which callers can treat as "sidecar
    /// exists but the chain is broken" rather than mistaking it for a real
    /// root.
    public static func rootSource(forImageAt imagePath: String) -> (path: String, assetId: String?) {
        var path = imagePath
        var assetId: String? = nil
        var visited: Set<String> = [imagePath]
        while let sc = read(forImageAt: path) {
            let nextPath = sc.sourcePath
            if visited.contains(nextPath) {
                return (imagePath, nil)
            }
            visited.insert(nextPath)
            path = nextPath
            assetId = sc.sourceAssetId
        }
        return (path, assetId)
    }

    /// Failure building the `edit` block's JSON representation. In practice
    /// only the "encoded to something other than a JSON object" branch below
    /// is unreachable for this struct's shape; it exists so the guard has a
    /// concrete error to throw rather than force-unwrapping.
    public enum JSONObjectError: Error {
        case notAnObject
    }

    /// JSON dictionary for the `edit` key. Throws (rather than swallowing
    /// into `[:]`) so a genuinely unrepresentable recipe — e.g. a NaN
    /// adjustment value, which `JSONEncoder` refuses to encode — surfaces to
    /// the exporter instead of silently writing an empty/partial block.
    public func jsonObject() throws -> [String: Any] {
        let data = try Self.encoder.encode(self)
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw JSONObjectError.notAnObject
        }
        return obj
    }
}
