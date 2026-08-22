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

    /// WP-E8: the tenth dial. `ServerPreset` has a HAND-WRITTEN encoder, so
    /// any engine field missing from it is silently erased by the next
    /// desktop save (upsert replaces the whole document — the `videoTuning`
    /// regression class, one process over).
    @Test("a desktop save does not erase the engine's bypass dial")
    func roundTripPreservesBypass() throws {
        let wire = #"""
        {"id":"raw-stock","name":"Raw","model":"krea2-raw","kroma":{"strength":0},
         "bypass":{"strength":1,"file":"krea2_filter_bypass_2vector.safetensors"}}
        """#
        var p = try snakeDecoder().decode(ServerPreset.self, from: Data(wire.utf8))
        #expect(p.bypass == ServerPresetBypass(strength: 1, file: "krea2_filter_bypass_2vector.safetensors"))

        p.name = "Renamed"
        let dict = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(p)) as? [String: Any])
        let bypass = try #require(dict["bypass"] as? [String: Any],
                                  "a desktop save dropped `bypass` — the engine's dial is erased")
        #expect(bypass["strength"] as? Double == 1)
        #expect(bypass["file"] as? String == "krea2_filter_bypass_2vector.safetensors")

        // Absent stays absent — never encoded as a default.
        let bare = try snakeDecoder().decode(
            ServerPreset.self, from: Data(#"{"id":"a","name":"A"}"#.utf8))
        #expect(bare.bypass == nil)
        let bareDict = try #require(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(bare)) as? [String: Any])
        #expect(bareDict["bypass"] == nil)
    }

    /// WP-E20: the engine's nine recipe/policy fields (+ WP-E9 `vae`) and the
    /// read-only validity flag. Upsert replaces the whole document, so a
    /// desktop edit-and-save must carry every one of them back — and must NOT
    /// send the flag, which the engine recomputes.
    @Test("round-trip preserves the krea2 recipe fields; the validity flag is read-only")
    func roundTripPreservesRecipeFieldsAndReadsValidity() throws {
        let wire = #"""
        {"id":"krea2-reference","name":"Reference","model":"krea2-raw","engine":"zimage",
         "checkpoint_family":"raw-accel","kroma":{"strength":0.6,"file":"kroma-v0.1.safetensors"},
         "vae":"/vae/Wan2_1_VAE_fp32.safetensors","sampler":"res_2s","sigma_schedule":"beta",
         "shift":1.15,"eta":0.5,"bongmath":true,
         "stage2":{"sampler":"dpmpp_2m","sigma_schedule":"karras","steps":2,"denoise":0.2,"eta":0.5,"bongmath":true},
         "invalid":true,"invalid_reason":"preset \"krea2-reference\": must declare kroma"}
        """#
        var p = try snakeDecoder().decode(ServerPreset.self, from: Data(wire.utf8))
        #expect(p.checkpointFamily == "raw-accel")
        #expect(p.kroma == ServerPresetKroma(strength: 0.6, file: "kroma-v0.1.safetensors"))
        #expect(p.vae == "/vae/Wan2_1_VAE_fp32.safetensors")
        #expect(p.sampler == "res_2s")
        #expect(p.sigmaSchedule == "beta")
        #expect(p.shift == 1.15)
        #expect(p.eta == 0.5)
        #expect(p.bongmath == true)
        #expect(p.stage2?.sampler == "dpmpp_2m")
        #expect(p.stage2?.sigmaSchedule == "karras")
        #expect(p.stage2?.steps == 2)
        #expect(p.stage2?.denoise == 0.2)
        #expect(p.invalid == true)
        #expect(p.invalidReason?.contains("kroma") == true)

        p.name = "Renamed"
        let dict = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(p)) as? [String: Any])
        #expect(dict["checkpointFamily"] as? String == "raw-accel")
        #expect((dict["kroma"] as? [String: Any])?["strength"] as? Double == 0.6)
        #expect((dict["kroma"] as? [String: Any])?["file"] as? String == "kroma-v0.1.safetensors")
        #expect(dict["vae"] as? String == "/vae/Wan2_1_VAE_fp32.safetensors")
        #expect(dict["sampler"] as? String == "res_2s")
        #expect(dict["sigmaSchedule"] as? String == "beta")
        #expect(dict["shift"] as? Double == 1.15)
        #expect(dict["eta"] as? Double == 0.5)
        #expect(dict["bongmath"] as? Bool == true)
        #expect((dict["stage2"] as? [String: Any])?["denoise"] as? Double == 0.2)
        #expect(dict["invalid"] == nil)
        #expect(dict["invalidReason"] == nil)
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
