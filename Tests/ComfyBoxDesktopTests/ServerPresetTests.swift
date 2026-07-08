// ServerPresetTests.swift — Wire-format tests for the server preset mirror

import Testing
import XCTest
import Foundation
@testable import ComfyBoxDesktop

@Suite("ServerPreset")
struct ServerPresetTests {
    /// Verbatim shape of a legacy image-service preset as served by /v1/presets.
    private static let liveJSON = #"""
    {
      "loras": [{"filename": "detail.safetensors", "scale": 0.8}],
      "model": "z-image-turbo",
      "name": "Z-Image Chat",
      "media_kind": "image",
      "steps": 8,
      "guidance": 1,
      "height": 512,
      "engine": "zimage",
      "mode": "z-image-turbo",
      "width": 512,
      "description": "Fast chat lane preset",
      "provider": "local",
      "id": "zimage-chat"
    }
    """#

    private func snakeDecoder() -> JSONDecoder {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        return d
    }

    @Test("decodes the live snake_case wire format")
    func decodesLiveShape() throws {
        let p = try snakeDecoder().decode(ServerPreset.self, from: Data(Self.liveJSON.utf8))
        #expect(p.id == "zimage-chat")
        #expect(p.name == "Z-Image Chat")
        #expect(p.engine == "zimage")
        #expect(p.mediaKind == "image")
        #expect(p.steps == 8)
        #expect(p.width == 512)
        #expect(p.loras.first?.filename == "detail.safetensors")
        #expect(p.loras.first?.scale == 0.8)
    }

    @Test("round-trip preserves legacy routing fields the UI does not edit")
    func roundTripPreservesLegacyFields() throws {
        var p = try snakeDecoder().decode(ServerPreset.self, from: Data(Self.liveJSON.utf8))
        p.name = "Renamed"  // the only edit
        let encoded = try JSONEncoder().encode(p)
        let dict = try JSONSerialization.jsonObject(with: encoded) as? [String: Any]
        // engine/mode/provider/mediaKind survive an edit-and-save cycle.
        #expect(dict?["engine"] as? String == "zimage")
        #expect(dict?["provider"] as? String == "local")
        #expect(dict?["mediaKind"] as? String == "image")
        #expect(dict?["name"] as? String == "Renamed")
    }

    @Test("maps to a GenerationPreset for apply")
    func toGenerationPreset() throws {
        let p = ServerPreset(
            id: "x", name: "X", model: "z-image-turbo",
            prompt: "a test", steps: 12, guidance: 4.0,
            width: 1280, height: 1280,
            loras: [ServerPresetLora(filename: "a.safetensors", scale: 0.6)],
            scheduler: "euler"
        )
        let g = p.toGenerationPreset()
        #expect(g.promptTemplate == "a test")
        #expect(g.modelId == "z-image-turbo")
        #expect(g.steps == 12)
        #expect(g.width == 1280)
        #expect(g.loras.first?.scale == 0.6)
        #expect(g.sampler == "euler")
    }

    @Test("customModelPath wins over model for apply")
    func customModelPathWins() {
        let p = ServerPreset(id: "y", name: "Y", model: "z-image-turbo",
                             customModelPath: "/models/custom.safetensors")
        #expect(p.toGenerationPreset().modelId == "/models/custom.safetensors")
    }

    @Test("tolerant decode of a minimal preset")
    func minimalDecode() throws {
        let p = try snakeDecoder().decode(ServerPreset.self, from: Data(#"{"name": "Bare"}"#.utf8))
        #expect(p.name == "Bare")
        #expect(!p.id.isEmpty)
        #expect(p.loras.isEmpty)
        #expect(p.steps == nil)
    }
}

final class ServerPresetNegativeTests: XCTestCase {
    func testToGenerationPresetCarriesNegative() {
        let sp = ServerPreset(id: "k", name: "Kira", model: "m",
                              negativePrompt: "blurry, watermark")
        XCTAssertEqual(sp.toGenerationPreset().negativePrompt, "blurry, watermark")
    }
}
