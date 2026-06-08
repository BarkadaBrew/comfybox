// DAMAsset.swift — Digital Asset Management model
//
// Core data model for tracked image/video assets. Each generated image
// is recorded in the DAM database with its generation parameters,
// file metadata, and user annotations (rating, favorite, content mode).

import Foundation

public struct DAMAsset: Identifiable, Sendable {
    public let id: String
    public let kind: String  // "image" or "video"
    public let filename: String
    public let absolutePath: String
    public let fileSize: Int64
    public let sha256: String?
    public let width: Int?
    public let height: Int?
    public let createdAt: Date
    public let modifiedAt: Date
    public let ingestedAt: Date
    public let orphaned: Bool
    public let prompt: String?
    public let negativePrompt: String?
    public let seed: Int?
    public let steps: Int?
    public let guidance: Double?
    public let modelFamily: String?
    public let rating: Int
    public let favorite: Bool
    public let contentMode: String?
    public let characterName: String?

    public init(
        id: String = UUID().uuidString,
        kind: String = "image",
        filename: String,
        absolutePath: String,
        fileSize: Int64 = 0,
        sha256: String? = nil,
        width: Int? = nil,
        height: Int? = nil,
        createdAt: Date = Date(),
        modifiedAt: Date = Date(),
        ingestedAt: Date = Date(),
        orphaned: Bool = false,
        prompt: String? = nil,
        negativePrompt: String? = nil,
        seed: Int? = nil,
        steps: Int? = nil,
        guidance: Double? = nil,
        modelFamily: String? = nil,
        rating: Int = 0,
        favorite: Bool = false,
        contentMode: String? = nil,
        characterName: String? = nil
    ) {
        self.id = id
        self.kind = kind
        self.filename = filename
        self.absolutePath = absolutePath
        self.fileSize = fileSize
        self.sha256 = sha256
        self.width = width
        self.height = height
        self.createdAt = createdAt
        self.modifiedAt = modifiedAt
        self.ingestedAt = ingestedAt
        self.orphaned = orphaned
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.seed = seed
        self.steps = steps
        self.guidance = guidance
        self.modelFamily = modelFamily
        self.rating = rating
        self.favorite = favorite
        self.contentMode = contentMode
        self.characterName = characterName
    }
}
