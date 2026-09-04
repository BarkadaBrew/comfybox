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

    public static func rootSource(forImageAt imagePath: String) -> (path: String, assetId: String?) {
        var path = imagePath
        var assetId: String? = nil
        var hops = 0
        while hops < 32, let sc = read(forImageAt: path) {
            path = sc.sourcePath; assetId = sc.sourceAssetId; hops += 1
        }
        return (path, assetId)
    }

    public var jsonObject: [String: Any] {
        guard let data = try? Self.encoder.encode(self),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return obj
    }
}
