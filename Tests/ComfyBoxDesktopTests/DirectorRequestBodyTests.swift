// DirectorRequestBodyTests.swift — WP3: the desktop's POST /v1/video/director envelope.

import Foundation
import Testing
import ZImage
@testable import ComfyBoxDesktop

@MainActor
@Suite("Director request body")
struct DirectorRequestBodyTests {

    private func timeline() -> DirectorTimeline {
        var t = directorTestTimeline()
        t.keyframes = [
            .init(id: "k1", imagePath: "/tmp/a.png", frame: 0),
            .init(id: "k2", imagePath: "/tmp/b.png", imageBase64: "AAAA", frame: 288, isEndFrame: true),
        ]
        t.promptSegments = [.init(id: "p1", startFrame: 0, lengthFrames: 96, prompt: "she turns")]
        return t
    }

    private func json(_ data: Data) throws -> [String: Any] {
        try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test("envelope carries timeline, output_path and source desktop")
    func envelopeHasTimelineOutputPathSourceDesktop() throws {
        let data = try EngineService.directorRequestBody(timeline: timeline(), outputPath: "/out/director-1.mp4")
        let body = try json(data)
        #expect(body["output_path"] as? String == "/out/director-1.mp4")
        #expect(body["source"] as? String == "desktop")
        #expect(body["timeline"] is [String: Any])
        // The engine decodes the same envelope.
        struct Envelope: Decodable { let timeline: DirectorTimeline; let outputPath: String?; let source: String? }
        let decoded = try DirectorJSON.decoder().decode(Envelope.self, from: data)
        #expect(decoded.timeline.keyframes.count == 2)
        #expect(decoded.source == "desktop")
    }

    @Test("timeline keys are snake_case")
    func timelineKeysAreSnakeCase() throws {
        let text = String(decoding: try EngineService.directorRequestBody(timeline: timeline(), outputPath: "/o.mp4"), as: UTF8.self)
        for key in ["length_frames", "is_end_frame", "prompt_segments", "global_prompt", "image_path"] {
            #expect(text.contains("\"\(key)\""))
        }
        for key in ["lengthFrames", "isEndFrame", "promptSegments", "globalPrompt", "imagePath"] {
            #expect(!text.contains("\"\(key)\""))
        }
    }

    @Test("image_base64 is stripped where the image path exists locally")
    func noImageBase64InBody() throws {
        let text = String(decoding: try EngineService.directorRequestBody(
            timeline: timeline(), outputPath: "/o.mp4", fileExists: { _ in true }), as: UTF8.self)
        #expect(!text.contains("image_base64"))
        let validate = String(decoding: try EngineService.directorValidateBody(
            timeline: timeline(), fileExists: { _ in true }), as: UTF8.self)
        #expect(!validate.contains("image_base64"))
    }

    @Test("a portable keyframe (missing path + embedded image) keeps image_base64 on submit and validate")
    func portableKeyframeKeepsBase64() throws {
        let exists: (String) -> Bool = { $0 == "/tmp/a.png" }  // b.png only exists as base64
        struct Envelope: Decodable { let timeline: DirectorTimeline }
        for data in [
            try EngineService.directorRequestBody(timeline: timeline(), outputPath: "/o.mp4", fileExists: exists),
            try EngineService.directorValidateBody(timeline: timeline(), fileExists: exists),
        ] {
            let decoded = try DirectorJSON.decoder().decode(Envelope.self, from: data)
            #expect(decoded.timeline.keyframes.first { $0.id == "k1" }?.imageBase64 == nil)
            #expect(decoded.timeline.keyframes.first { $0.id == "k2" }?.imageBase64 == "AAAA")
            #expect(decoded.timeline.keyframes.first { $0.id == "k2" }?.imagePath == "/tmp/b.png")
        }
    }
}
