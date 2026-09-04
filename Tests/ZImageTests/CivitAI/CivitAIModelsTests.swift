// CivitAIModelsTests.swift — CivitAI wire-format decoding

import Testing
import Foundation
@testable import ZImage

@Suite("CivitAIModels")
struct CivitAIModelsTests {
    /// Condensed live shape from api/v1/models (2026-07-03).
    private static let sample = #"""
    {
      "items": [
        {
          "id": 999001,
          "name": "Seeping — Bioluminescent World Morph",
          "type": "LORA",
          "nsfw": false,
          "tags": ["style", "morph"],
          "creator": {"username": "someone", "image": null},
          "stats": {"downloadCount": 328, "thumbsUpCount": 41, "commentCount": 2},
          "modelVersions": [
            {
              "id": 2880341,
              "name": "v1.0",
              "baseModel": "Z-Image",
              "trainedWords": ["s33ping"],
              "files": [
                {"name": "seeping_zimage_v1.safetensors", "sizeKB": 20820.4,
                 "downloadUrl": "https://civitai.com/api/download/models/2880341",
                 "primary": true, "type": "Model"},
                {"name": "extras.zip", "sizeKB": 100, "downloadUrl": "https://x/y", "primary": false}
              ],
              "images": [
                {"url": "https://image.civitai.com/abc/width=450/1.jpeg",
                 "meta": {"prompt": "a glowing forest, s33ping style", "steps": 30}},
                {"url": "https://image.civitai.com/abc/width=450/2.jpeg", "meta": null}
              ]
            }
          ]
        },
        {"id": 999002, "name": "Sparse Model", "type": "Checkpoint", "modelVersions": []}
      ],
      "metadata": {"nextCursor": "abc123"}
    }
    """#

    @Test("decodes the live models shape")
    func decodesModels() throws {
        let page = try JSONDecoder().decode(CivitAIModelsPage.self, from: Data(Self.sample.utf8))
        #expect(page.items.count == 2)
        #expect(page.nextCursor == "abc123")

        let model = page.items[0]
        #expect(model.name.hasPrefix("Seeping"))
        #expect(model.type == "LORA")
        #expect(model.creatorName == "someone")
        #expect(model.downloadCount == 328)

        let version = model.modelVersions[0]
        #expect(version.baseModel == "Z-Image")
        #expect(version.trainedWords == ["s33ping"])
        #expect(version.primaryFile?.name == "seeping_zimage_v1.safetensors")
        #expect(version.primaryFile?.sizeLabel == "20 MB")

        // Prompt scraped from image meta; null meta tolerated.
        #expect(version.images.count == 2)
        #expect(version.images[0].prompt == "a glowing forest, s33ping style")
        #expect(version.images[1].prompt == nil)

        // Thumbnail comes from the first version image.
        #expect(model.thumbnailURL?.absoluteString.contains("1.jpeg") == true)
    }

    @Test("sparse model decodes with defaults")
    func sparseModel() throws {
        let page = try JSONDecoder().decode(CivitAIModelsPage.self, from: Data(Self.sample.utf8))
        let sparse = page.items[1]
        #expect(sparse.name == "Sparse Model")
        #expect(sparse.downloadCount == 0)
        #expect(sparse.modelVersions.isEmpty)
        #expect(sparse.thumbnailURL == nil)
    }

    @Test("primary file falls back to first when none marked primary")
    func primaryFallback() throws {
        let json = #"{"id": 1, "name": "v", "files": [{"name": "only.safetensors", "downloadUrl": "https://x"}]}"#
        let version = try JSONDecoder().decode(CivitAIModelVersion.self, from: Data(json.utf8))
        #expect(version.primaryFile?.name == "only.safetensors")
    }

    @Test("file size labels")
    func sizeLabels() throws {
        let small = try JSONDecoder().decode(CivitAIFile.self, from: Data(#"{"name":"a","sizeKB":512000}"#.utf8))
        #expect(small.sizeLabel == "500 MB")
        let large = try JSONDecoder().decode(CivitAIFile.self, from: Data(#"{"name":"b","sizeKB":6815744}"#.utf8))
        #expect(large.sizeLabel == "6.5 GB")
    }
}
