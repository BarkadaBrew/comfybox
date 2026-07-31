// CatalogModels.swift — pure value types for the asset catalog.
//
// No I/O, no SQLite, no Foundation-beyond-Date. Everything here is safe to
// construct in a test without touching a database or the filesystem.

import Foundation

/// The isolation domain. Kira is the ONLY exception; everything else — Todd's
/// work, Bree's, the Studio bot's — is `shared`. Never add a third value.
public enum CatalogRealm: String, CaseIterable, Sendable, Codable {
    case kira
    case shared
}

/// A relation between two assets.
public enum AssetRelation: String, Sendable, Codable {
    /// `from` (a clip) was animated from `to` (a still).
    case i2vSource = "i2v_source"
    /// `from` (a clip) is a component of `to` (a scene / montage / storyboard).
    case memberOf = "member_of"
}

public struct AssetEdge: Sendable, Equatable, Codable {
    public let fromAssetID: String
    public let toAssetID: String
    public let relation: AssetRelation

    public init(fromAssetID: String, toAssetID: String, relation: AssetRelation) {
        self.fromAssetID = fromAssetID
        self.toAssetID = toAssetID
        self.relation = relation
    }
}

/// A body of work. `parentID == nil` is a root; a child's parent must itself be
/// a root (two levels, enforced in CatalogStore).
public struct CatalogCollection: Sendable, Equatable, Codable {
    public let id: String
    public let slug: String
    public let name: String
    public let parentID: String?
    /// nil = shared vocabulary anything may contribute to.
    /// .kira = exists only in her realm and is visible only to her.
    public let realm: CatalogRealm?
    public let description: String?

    public init(id: String = UUID().uuidString, slug: String, name: String,
                parentID: String? = nil, realm: CatalogRealm? = nil,
                description: String? = nil) {
        self.id = id
        self.slug = slug
        self.name = name
        self.parentID = parentID
        self.realm = realm
        self.description = description
    }
}

/// Where an asset's bytes live. One asset, many locations — the gallery home
/// plus any downstream server copies.
public struct AssetLocation: Sendable, Equatable, Codable {
    public let host: String
    public let path: String
    public let mtime: Date

    public init(host: String, path: String, mtime: Date) {
        self.host = host
        self.path = path
        self.mtime = mtime
    }
}

public struct CatalogAsset: Sendable, Equatable {
    public let id: String
    public let kind: String            // "image" | "video"
    public let filename: String
    public let absolutePath: String
    public let sha256: String?
    public let fileSize: Int64
    public let width: Int?
    public let height: Int?
    public let createdAt: Date

    // Isolation and provenance
    public let realm: CatalogRealm
    public let source: String?         // which application requested it
    public let sealed: Bool

    // Text (always nil when sealed)
    public let prompt: String?
    public let negativePrompt: String?
    public let promptRaw: String?
    /// The prompt after character / trigger injection — a third real spelling
    /// in the sidecars, indexed for search alongside the other two.
    public let promptInjected: String?
    public let caption: String?
    public let captionSource: String?

    // Generation facets
    public let seed: Int?
    public let steps: Int?
    public let guidance: Double?
    public let modelFamily: String?
    public let preset: String?
    public let loras: String?          // JSON array as stored
    public let renderID: String?
    public let contentMode: String?    // the fruit tier
    public let characterName: String?

    // Seed metadata — names mirror RenderSeedMeta 1:1
    public let lane: String?
    public let arc: String?
    public let theme: String?
    public let stock: String?
    public let genre: String?
    public let family: String?
    public let style: String?

    // Video
    public let mode: String?           // "i2v" | "t2v"
    public let durationMs: Int?
    public let fps: Double?
    public let frames: Int?
    public let resolution: String?
    public let aspectRatio: String?

    // User annotation
    public let rating: Int
    public let favorite: Bool

    public init(
        id: String = UUID().uuidString, kind: String = "image",
        filename: String, absolutePath: String,
        sha256: String? = nil, fileSize: Int64 = 0,
        width: Int? = nil, height: Int? = nil, createdAt: Date = Date(),
        realm: CatalogRealm = .shared, source: String? = nil, sealed: Bool = false,
        prompt: String? = nil, negativePrompt: String? = nil,
        promptRaw: String? = nil, promptInjected: String? = nil,
        caption: String? = nil, captionSource: String? = nil,
        seed: Int? = nil, steps: Int? = nil, guidance: Double? = nil,
        modelFamily: String? = nil, preset: String? = nil, loras: String? = nil,
        renderID: String? = nil, contentMode: String? = nil, characterName: String? = nil,
        lane: String? = nil, arc: String? = nil, theme: String? = nil,
        stock: String? = nil, genre: String? = nil, family: String? = nil, style: String? = nil,
        mode: String? = nil, durationMs: Int? = nil, fps: Double? = nil, frames: Int? = nil,
        resolution: String? = nil, aspectRatio: String? = nil,
        rating: Int = 0, favorite: Bool = false
    ) {
        self.id = id; self.kind = kind
        self.filename = filename; self.absolutePath = absolutePath
        self.sha256 = sha256; self.fileSize = fileSize
        self.width = width; self.height = height; self.createdAt = createdAt
        self.realm = realm; self.source = source; self.sealed = sealed
        // Sealed rows carry facets only — the text is dropped at construction so
        // no caller can smuggle it past the rule by forgetting to check.
        self.prompt = sealed ? nil : prompt
        self.negativePrompt = sealed ? nil : negativePrompt
        self.promptRaw = sealed ? nil : promptRaw
        self.promptInjected = sealed ? nil : promptInjected
        self.caption = sealed ? nil : caption
        self.captionSource = sealed ? nil : captionSource
        // `theme` belongs with the text above, not with the facets below. It
        // LOOKS like a facet — a short seed-metadata field sitting between
        // `arc` and `stock` — but MetadataReader fills it from the render
        // journal's free-text intent line, and in the live catalog those run to
        // full explicit sentences. It escaped this nulling for as long as it was
        // filed by position rather than by what it holds.
        self.seed = seed; self.steps = steps; self.guidance = guidance
        self.modelFamily = modelFamily; self.preset = preset; self.loras = loras
        self.renderID = renderID; self.contentMode = contentMode
        self.characterName = characterName
        self.lane = lane; self.arc = arc; self.theme = sealed ? nil : theme
        self.stock = stock; self.genre = genre; self.family = family; self.style = style
        self.mode = mode; self.durationMs = durationMs; self.fps = fps; self.frames = frames
        self.resolution = resolution; self.aspectRatio = aspectRatio
        self.rating = rating; self.favorite = favorite
    }
}
