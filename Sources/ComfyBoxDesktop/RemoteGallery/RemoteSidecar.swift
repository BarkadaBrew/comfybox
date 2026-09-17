// RemoteSidecar.swift — what travels with an asset to a remote gallery.
//
// Todd 2026-09-17 chose "everything: the file plus its recipe and prompt", so
// the drive is useful on its own: the sidecar beside each file says what the
// asset is, how it was made, and that it came out of the vault
// (FDD-remote-galleries §3.2).

import Foundation

public struct RemoteSidecar: Codable, Sendable {
    public var schemaVersion: Int
    public var assetID: String
    public var filename: String
    public var kind: String
    public var fileSize: Int64
    public var sha256: String?
    public var width: Int?
    public var height: Int?
    public var createdAt: Date
    public var sentAt: Date
    /// True when this came out of the Mac's vault: the remote is now the
    /// privacy boundary.
    public var sensitive: Bool

    // Recipe
    public var prompt: String?
    public var negativePrompt: String?
    public var seed: Int?
    public var steps: Int?
    public var guidance: Double?
    public var modelFamily: String?
    public var preset: String?
    public var loras: String?
    public var contentMode: String?
    public var characterName: String?
    public var caption: String?
    public var rating: Int
    public var favorite: Bool
    /// The path it had on the Mac, for a human reading the drive later.
    public var originalPath: String

    public static let currentSchemaVersion = 1

    public init(asset: DAMAsset, sha256: String?, sentAt: Date, sensitive: Bool) {
        self.schemaVersion = Self.currentSchemaVersion
        self.assetID = asset.id
        self.filename = asset.filename
        self.kind = asset.kind
        self.fileSize = asset.fileSize
        self.sha256 = sha256 ?? asset.sha256
        self.width = asset.width
        self.height = asset.height
        self.createdAt = asset.createdAt
        self.sentAt = sentAt
        self.sensitive = sensitive
        self.prompt = asset.prompt
        self.negativePrompt = asset.negativePrompt
        self.seed = asset.seed
        self.steps = asset.steps
        self.guidance = asset.guidance.map { Double($0) }
        self.modelFamily = asset.modelFamily
        self.contentMode = asset.contentMode
        self.characterName = asset.characterName
        self.rating = asset.rating
        self.favorite = asset.favorite
        self.originalPath = asset.absolutePath
    }

    public func encoded() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(self)
    }
}
