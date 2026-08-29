// CivitAIModels.swift — Wire models for the CivitAI public API
//
// Tolerant Codable mirrors of /api/v1/models and /api/v1/images responses
// (civitai.com; civitai.red serves the same API shape). Only the fields the
// browser uses are decoded; everything decodes defensively because the API
// omits/nulls fields freely.

import Foundation

public struct CivitAIModelsPage: Decodable, Sendable {
    public var items: [CivitAIModel] = []
    public var nextCursor: String?

    private enum CodingKeys: String, CodingKey { case items, metadata }
    private enum MetadataKeys: String, CodingKey { case nextCursor }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        items = (try? c.decodeIfPresent([CivitAIModel].self, forKey: .items)) ?? []
        if let meta = try? c.nestedContainer(keyedBy: MetadataKeys.self, forKey: .metadata) {
            nextCursor = try? meta.decodeIfPresent(String.self, forKey: .nextCursor)
        }
    }
}

public struct CivitAIModel: Decodable, Sendable, Identifiable {
    public var id: Int = 0
    public var name: String = ""
    public var type: String = ""          // "LORA", "Checkpoint", …
    public var nsfw: Bool = false
    public var tags: [String] = []
    public var creatorName: String?
    public var downloadCount: Int = 0
    public var thumbsUpCount: Int = 0
    public var modelVersions: [CivitAIModelVersion] = []

    private enum CodingKeys: String, CodingKey {
        case id, name, type, nsfw, tags, creator, stats, modelVersions
    }
    private enum CreatorKeys: String, CodingKey { case username }
    private enum StatsKeys: String, CodingKey { case downloadCount, thumbsUpCount }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decodeIfPresent(Int.self, forKey: .id)) ?? 0
        name = (try? c.decodeIfPresent(String.self, forKey: .name)) ?? ""
        type = (try? c.decodeIfPresent(String.self, forKey: .type)) ?? ""
        nsfw = (try? c.decodeIfPresent(Bool.self, forKey: .nsfw)) ?? false
        tags = (try? c.decodeIfPresent([String].self, forKey: .tags)) ?? []
        if let creator = try? c.nestedContainer(keyedBy: CreatorKeys.self, forKey: .creator) {
            creatorName = try? creator.decodeIfPresent(String.self, forKey: .username)
        }
        if let stats = try? c.nestedContainer(keyedBy: StatsKeys.self, forKey: .stats) {
            downloadCount = (try? stats.decodeIfPresent(Int.self, forKey: .downloadCount)) ?? 0
            thumbsUpCount = (try? stats.decodeIfPresent(Int.self, forKey: .thumbsUpCount)) ?? 0
        }
        modelVersions = (try? c.decodeIfPresent([CivitAIModelVersion].self, forKey: .modelVersions)) ?? []
    }

    /// First image URL across versions, for the card thumbnail.
    public var thumbnailURL: URL? {
        for version in modelVersions {
            if let image = version.images.first, let url = URL(string: image.url) {
                return url
            }
        }
        return nil
    }
}

public struct CivitAIModelVersion: Decodable, Sendable, Identifiable {
    public var id: Int = 0
    public var name: String = ""
    public var baseModel: String = ""
    public var trainedWords: [String] = []
    public var files: [CivitAIFile] = []
    public var images: [CivitAIImage] = []
    /// Free-text version description (CivitAI serves this as HTML). Used by
    /// the prompt-repository harvester (issue #234) as a fallback prompt
    /// source alongside `trainedWords`, since `meta.prompt` on images is
    /// gated even with a valid API key.
    public var description: String?

    private enum CodingKeys: String, CodingKey {
        case id, name, baseModel, trainedWords, files, images, description
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = (try? c.decodeIfPresent(Int.self, forKey: .id)) ?? 0
        name = (try? c.decodeIfPresent(String.self, forKey: .name)) ?? ""
        baseModel = (try? c.decodeIfPresent(String.self, forKey: .baseModel)) ?? ""
        trainedWords = (try? c.decodeIfPresent([String].self, forKey: .trainedWords)) ?? []
        files = (try? c.decodeIfPresent([CivitAIFile].self, forKey: .files)) ?? []
        images = (try? c.decodeIfPresent([CivitAIImage].self, forKey: .images)) ?? []
        description = try? c.decodeIfPresent(String.self, forKey: .description)
    }

    /// The file to download: the primary model file, else the first.
    public var primaryFile: CivitAIFile? {
        files.first(where: { $0.primary }) ?? files.first
    }
}

public struct CivitAIFile: Decodable, Sendable, Identifiable {
    public var name: String = ""
    public var sizeKB: Double = 0
    public var downloadUrl: String = ""
    public var primary: Bool = false

    public var id: String { downloadUrl.isEmpty ? name : downloadUrl }

    private enum CodingKeys: String, CodingKey { case name, sizeKB, downloadUrl, primary }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = (try? c.decodeIfPresent(String.self, forKey: .name)) ?? ""
        sizeKB = (try? c.decodeIfPresent(Double.self, forKey: .sizeKB)) ?? 0
        downloadUrl = (try? c.decodeIfPresent(String.self, forKey: .downloadUrl)) ?? ""
        primary = (try? c.decodeIfPresent(Bool.self, forKey: .primary)) ?? false
    }

    public var sizeLabel: String {
        let mb = sizeKB / 1024
        return mb >= 1024 ? String(format: "%.1f GB", mb / 1024) : String(format: "%.0f MB", mb)
    }
}

public struct CivitAIImage: Decodable, Sendable {
    public var url: String = ""
    public var prompt: String?

    private enum CodingKeys: String, CodingKey { case url, meta }
    private enum MetaKeys: String, CodingKey { case prompt }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        url = (try? c.decodeIfPresent(String.self, forKey: .url)) ?? ""
        // meta can be null, absent, or an arbitrary object.
        if let meta = try? c.nestedContainer(keyedBy: MetaKeys.self, forKey: .meta) {
            prompt = try? meta.decodeIfPresent(String.self, forKey: .prompt)
        }
    }
}

/// /api/v1/images response — used for prompt scraping on a model version.
public struct CivitAIImagesPage: Decodable, Sendable {
    public var items: [CivitAIImage] = []

    private enum CodingKeys: String, CodingKey { case items }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        items = (try? c.decodeIfPresent([CivitAIImage].self, forKey: .items)) ?? []
    }
}
