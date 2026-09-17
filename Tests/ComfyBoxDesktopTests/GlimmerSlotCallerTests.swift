// GlimmerSlotCallerTests.swift — desktop Glimmer callers hold an inference slot
// (FDD-glimmer-gpu-slot E6, Codex review of #465) and show the wait.

import AppKit
import Foundation
import Testing
import ZImage
@testable import ComfyBoxDesktop

@MainActor
@Suite("Glimmer slot callers")
struct GlimmerSlotCallerTests {

    /// Port 9 refuses fast, so the model call itself fails without a real model;
    /// the endpoint list marks it as Glimmer for the test.
    static let glimmer = "http://127.0.0.1:9/v1"
    static let endpoints = ["127.0.0.1:9"]

    private func engine(_ transport: FakeEngineTransport, provider: AIProviderEndpoint) throws -> EngineService {
        var config = ComfyBoxServerConfig()
        config.providers = AIProviderRegistry(
            promptOptimization: provider, vision: provider, captioning: provider, assistant: provider)
        let json = String(decoding: try JSONEncoder().encode(config), as: UTF8.self)
        transport.always("/v1/config", .init(200, json))
        transport.always("/v1/loras", .init(200, #"{"loras":[]}"#))
        transport.always("/v1/presets", .init(200, "[]"))
        return makeTestEngine(transport: transport, outputDirectory: NSTemporaryDirectory())
    }

    private func pngFile() throws -> String {
        let rep = try #require(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 8, pixelsHigh: 8, bitsPerSample: 8, samplesPerPixel: 4,
            hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0))
        let data = try #require(rep.representation(using: .png, properties: [:]))
        let path = NSTemporaryDirectory() + "slot-caller-\(UUID().uuidString).png"
        try data.write(to: URL(fileURLWithPath: path))
        return path
    }

    // MARK: - Captioning

    @Test("captioning on Glimmer waits for a slot, reports the wait, and releases")
    func captioningReportsWaitAndReleases() async throws {
        let transport = FakeEngineTransport()
        let engine = try engine(transport, provider: .init(baseUrl: Self.glimmer, model: "glimmer"))
        transport.script("/v1/queue/inference-slot", [.init(200, #"{"slot_id":"c1","state":"waiting","eta_sec":5}"#)])
        transport.script("/v1/queue/inference-slot/c1", [.init(200, #"{"state":"granted"}"#)])
        transport.always("/v1/queue/inference-slot/c1", .init(200, #"{"released":true}"#))

        let vision = VisionService(engine: engine, glimmerEndpoints: Self.endpoints)
        final class Box: @unchecked Sendable { var statuses: [String] = [] }
        let box = Box()
        let path = try pngFile()
        await #expect(throws: (any Error).self) {
            _ = try await vision.describe(imagePath: path, onWaitStatus: { box.statuses.append($0) })
        }
        #expect(box.statuses.first == "Waiting for the GPU (~5s)")
        let acquire = try #require(transport.body(of: "POST", "/v1/queue/inference-slot"))
        #expect(acquire["holder"] as? String == "comfybox-desktop:caption")
        #expect(transport.requestCount("DELETE", "/v1/queue/inference-slot/c1") == 1)
    }

    @Test("captioning with the GPU busy past the budget fails with a human reason")
    func captioningBusyFailsWithReason() async throws {
        let transport = FakeEngineTransport()
        let engine = try engine(transport, provider: .init(baseUrl: Self.glimmer, model: "glimmer"))
        transport.script("/v1/queue/inference-slot", [.init(200, #"{"slot_id":"c2","state":"waiting","eta_sec":7200}"#)])
        transport.always("/v1/queue/inference-slot/c2", .init(200, #"{"released":true}"#))

        let vision = VisionService(engine: engine, glimmerEndpoints: Self.endpoints)
        let path = try pngFile()
        do {
            _ = try await vision.describe(imagePath: path)
            Issue.record("expected the GPU-busy error")
        } catch {
            #expect(error.localizedDescription == "The GPU is busy rendering (about 7200 s left).")
        }
        #expect(transport.requestCount("DELETE", "/v1/queue/inference-slot/c2") == 1)
    }

    @Test("captioning on a non-Glimmer provider takes no slot")
    func captioningNonGlimmerTakesNoSlot() async throws {
        let transport = FakeEngineTransport()
        let engine = try engine(transport, provider: .init(baseUrl: Self.glimmer, model: "lmstudio"))
        let vision = VisionService(engine: engine, glimmerEndpoints: ["10.0.100.134:11234"])
        let path = try pngFile()
        await #expect(throws: (any Error).self) { _ = try await vision.describe(imagePath: path) }
        #expect(!transport.requests.contains { $0.path.contains("inference-slot") })
    }

    // MARK: - Assistant

    @Test("assistant shows 'Waiting for the GPU' while waiting and clears it after")
    func assistantShowsWaitStatus() async throws {
        let transport = FakeEngineTransport()
        let engine = try engine(transport, provider: .init(baseUrl: Self.glimmer, model: "glimmer"))
        transport.script("/v1/queue/inference-slot", [.init(200, #"{"slot_id":"a1","state":"preempting"}"#)])
        transport.script("/v1/queue/inference-slot/a1", [
            .init(200, #"{"state":"preempting"}"#), .init(200, #"{"state":"granted"}"#),
        ])
        transport.always("/v1/queue/inference-slot/a1", .init(200, #"{"released":true}"#))

        let agent = AgentService(engine: engine, glimmerEndpoints: Self.endpoints)
        let send = Task { await agent.send("hello") }
        var seen: String?
        for _ in 0..<300 where seen == nil {
            seen = agent.waitStatus
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        await send.value
        #expect(seen == "Waiting for the GPU (pausing the video render…)")
        #expect(agent.waitStatus == nil, "cleared once the wait is over")
        #expect(AgentService.progressText(waitStatus: nil) == "Thinking…")
        #expect(AgentService.progressText(waitStatus: "Waiting for the GPU (~9s)") == "Waiting for the GPU (~9s)")
        let acquire = try #require(transport.body(of: "POST", "/v1/queue/inference-slot"))
        #expect(acquire["holder"] as? String == "comfybox-desktop:assistant")
        #expect(transport.requestCount("DELETE", "/v1/queue/inference-slot/a1") == 1)
    }

    @Test("assistant with the GPU busy past its budget says so")
    func assistantBusySaysSo() async throws {
        let transport = FakeEngineTransport()
        let engine = try engine(transport, provider: .init(baseUrl: Self.glimmer, model: "glimmer"))
        transport.script("/v1/queue/inference-slot", [.init(200, #"{"slot_id":"a2","state":"waiting","eta_sec":3600}"#)])
        transport.always("/v1/queue/inference-slot/a2", .init(200, #"{"released":true}"#))

        let agent = AgentService(engine: engine, glimmerEndpoints: Self.endpoints)
        await agent.send("hello")
        #expect(agent.lastError == "The GPU is busy rendering (about 3600 s left).")
        #expect(agent.waitStatus == nil)
        #expect(transport.requestCount("DELETE", "/v1/queue/inference-slot/a2") == 1)
    }
}
