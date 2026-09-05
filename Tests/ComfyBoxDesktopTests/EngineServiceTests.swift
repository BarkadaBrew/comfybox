// EngineServiceTests.swift — Tests for EngineService state, types, and models

import Testing
import Foundation
@testable import ComfyBoxDesktop

@MainActor
@Suite("EngineService")
struct EngineServiceTests {
    @Test("initial state is disconnected")
    func initialState() {
        let engine = EngineService()
        #expect(engine.connectionState.label == "Disconnected")
        #expect(!engine.connectionState.isConnected)
        #expect(engine.currentModel == nil)
        #expect(engine.queueCount == 0)
        #expect(!engine.isGenerating)
        #expect(engine.lastGeneratedImagePath == nil)
        #expect(engine.lastError == nil)
        #expect(engine.availableModels.isEmpty)
        #expect(engine.poolModels.isEmpty)
        #expect(engine.availableLoras.isEmpty)
    }

    @Test("disconnect resets all state")
    func disconnectResetsState() {
        let engine = EngineService()
        engine.currentModel = "some-model"
        engine.queueCount = 5
        engine.availableModels = [TestData.makeModelInfo()]
        engine.poolModels = [TestData.makePoolModel()]
        engine.availableLoras = [TestData.makeLoRAInfo()]

        engine.disconnect()

        #expect(engine.connectionState.label == "Disconnected")
        #expect(engine.currentModel == nil)
        #expect(engine.queueCount == 0)
        #expect(engine.availableModels.isEmpty)
        #expect(engine.poolModels.isEmpty)
        #expect(engine.availableLoras.isEmpty)
        #expect(engine.queueInfo == nil)
    }

    @Test("server host and port configurable")
    func configurable() {
        let engine = EngineService()
        engine.serverHost = "10.0.0.5"
        engine.serverPort = 9999
        #expect(engine.serverHost == "10.0.0.5")
        #expect(engine.serverPort == 9999)
    }

    @Test("output directory is configurable")
    func outputDir() {
        let engine = EngineService()
        engine.outputDirectory = "/tmp/test-output"
        #expect(engine.outputDirectory == "/tmp/test-output")
    }

    @Test("isGenerating starts false")
    func notGenerating() {
        let engine = EngineService()
        #expect(!engine.isGenerating)
    }

    @Test("isLoadingModel starts false")
    func notLoadingModel() {
        let engine = EngineService()
        #expect(!engine.isLoadingModel)
    }

    @Test("isSwappingLoras starts false")
    func notSwappingLoras() {
        let engine = EngineService()
        #expect(!engine.isSwappingLoras)
    }

    @Test("lastError is settable")
    func lastError() {
        let engine = EngineService()
        engine.lastError = "test error"
        #expect(engine.lastError == "test error")
        engine.lastError = nil
        #expect(engine.lastError == nil)
    }

    @Test("queueInfo starts nil")
    func queueInfoNil() {
        let engine = EngineService()
        #expect(engine.queueInfo == nil)
    }
}

@Suite("Server response decoding")
struct ServerResponseDecodingTests {
    // The server encodes all /v1/* and /health responses with
    // JSONEncoder.keyEncodingStrategy = .convertToSnakeCase, so client
    // structs must decode snake_case keys.

    @Test("generate response decodes snake_case keys")
    func generateResponseSnakeCase() throws {
        let json = Data("""
            {"success": true, "output_path": "/tmp/out/comfybox-123.png", "duration_ms": 4211}
            """.utf8)
        let response = try JSONDecoder().decode(ServerGenerateResponse.self, from: json)
        #expect(response.success)
        #expect(response.outputPath == "/tmp/out/comfybox-123.png")
        #expect(response.durationMs == 4211)
    }

    @Test("generate response fails without snake_case keys decoded")
    func generateResponseRejectsMissingKeys() {
        let json = Data("""
            {"success": true}
            """.utf8)
        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder().decode(ServerGenerateResponse.self, from: json)
        }
    }

    @Test("health response decodes snake_case keys including progress telemetry")
    func healthResponseSnakeCase() throws {
        let json = Data("""
            {
                "status": "ok",
                "model": "tongyi/z-image-turbo",
                "model_family": "zimage",
                "loaded": true,
                "is_rendering": true,
                "pending_count": 2,
                "render_count": 17,
                "uptime_seconds": 360,
                "last_render_duration_ms": 5300,
                "last_error": null,
                "memory_usage_mb": 8192,
                "current_job_id": "job-42",
                "progress_percent": 55.5,
                "loras": [{"source": "/tmp/style.safetensors", "scale": 0.8}]
            }
            """.utf8)
        let health = try JSONDecoder().decode(ServerHealthResponse.self, from: json)
        #expect(health.status == "ok")
        #expect(health.model == "tongyi/z-image-turbo")
        #expect(health.modelFamily == "zimage")
        #expect(health.isRendering == true)
        #expect(health.pendingCount == 2)
        #expect(health.renderCount == 17)
        #expect(health.uptimeSeconds == 360)
        #expect(health.lastRenderDurationMs == 5300)
        #expect(health.lastError == nil)
        #expect(health.memoryUsageMB == 8192)
        #expect(health.currentJobId == "job-42")
        #expect(health.progressPercent == 55.5)
        #expect(health.loras?.count == 1)
        #expect(health.loras?[0].scale == 0.8)
    }

    @Test("health response tolerates missing progress fields")
    func healthResponseWithoutProgress() throws {
        let json = Data("""
            {"status": "ok", "model": "some-model"}
            """.utf8)
        let health = try JSONDecoder().decode(ServerHealthResponse.self, from: json)
        #expect(health.status == "ok")
        #expect(health.currentJobId == nil)
        #expect(health.progressPercent == nil)
    }

    @Test("progress_percent decodes integer values")
    func integerProgress() throws {
        let json = Data("""
            {"status": "ok", "progress_percent": 43, "current_job_id": null}
            """.utf8)
        let health = try JSONDecoder().decode(ServerHealthResponse.self, from: json)
        #expect(health.progressPercent == 43.0)
        #expect(health.currentJobId == nil)
    }
}

// MARK: - #273 Nearline anchoring

@MainActor
@Suite("EngineService nearline anchoring")
struct NearlineAnchorParsingTests {
    @Test("anchored true parses from GET /v1/nearline items")
    func anchoredTrueParses() {
        let json = Data("""
            {"roots": [], "cache_limit_gb": 10, "staged_mb": 2,
             "items": [{"name": "pinned.safetensors", "path": "/x/pinned.safetensors",
                        "size_mb": 2, "kind": "lora", "staged": true, "anchored": true}]}
            """.utf8)
        let engine = EngineService()
        let catalog = engine.parseNearline(json)
        #expect(catalog?.items.first?.anchored == true)
    }

    @Test("anchored key absent (older server) defaults to false")
    func anchoredAbsentDefaultsFalse() {
        let json = Data("""
            {"roots": [], "cache_limit_gb": 10, "staged_mb": 0,
             "items": [{"name": "legacy.safetensors", "path": "/x/legacy.safetensors",
                        "size_mb": 2, "kind": "lora", "staged": false}]}
            """.utf8)
        let engine = EngineService()
        let catalog = engine.parseNearline(json)
        #expect(catalog?.items.first?.anchored == false)
    }

    @Test("setNearlineAnchor throws notConnected when disconnected")
    func setAnchorRequiresConnection() async {
        let engine = EngineService()
        await #expect(throws: EngineServiceError.self) {
            _ = try await engine.setNearlineAnchor(kind: "lora", id: "x.safetensors", anchored: true)
        }
    }
}

@Suite("ServerConnectionState")
struct ServerConnectionStateTests {
    @Test("disconnected state")
    func disconnected() {
        let state = ServerConnectionState.disconnected
        #expect(state.label == "Disconnected")
        #expect(!state.isConnected)
    }

    @Test("connecting state")
    func connecting() {
        let state = ServerConnectionState.connecting
        #expect(state.label == "Connecting...")
        #expect(!state.isConnected)
    }

    @Test("connected state")
    func connected() {
        let state = ServerConnectionState.connected
        #expect(state.label == "Connected")
        #expect(state.isConnected)
    }

    @Test("error state includes message")
    func errorState() {
        let state = ServerConnectionState.error("timeout")
        #expect(state.label == "Error: timeout")
        #expect(!state.isConnected)
    }

    @Test("error state with empty message")
    func errorEmpty() {
        let state = ServerConnectionState.error("")
        #expect(state.label == "Error: ")
        #expect(!state.isConnected)
    }
}

@Suite("GenerationRequest")
struct GenerationRequestTests {
    @Test("default values")
    func defaults() {
        let req = GenerationRequest()
        #expect(req.prompt == "")
        #expect(req.width == 1024)
        #expect(req.height == 1024)
        #expect(req.steps == 9)
        #expect(req.guidance == 3.5)
        #expect(req.seed == 0)
        #expect(req.modelId == nil)
        #expect(req.loras.isEmpty)
    }

    @Test("custom values preserved")
    func customValues() {
        let req = GenerationRequest(
            prompt: "test prompt",
            width: 768,
            height: 512,
            steps: 25,
            guidance: 7.0,
            seed: 12345,
            modelId: "my-model",
            loras: [LoRASelection(id: "l1", filename: "l1.safetensors", scale: 0.8)]
        )
        #expect(req.prompt == "test prompt")
        #expect(req.width == 768)
        #expect(req.height == 512)
        #expect(req.steps == 25)
        #expect(req.guidance == 7.0)
        #expect(req.seed == 12345)
        #expect(req.modelId == "my-model")
        #expect(req.loras.count == 1)
        #expect(req.loras[0].scale == 0.8)
    }
}

@Suite("LoRASelection")
struct LoRASelectionTests {
    @Test("default scale is 1.0")
    func defaultScale() {
        let sel = LoRASelection(id: "test", filename: "test.safetensors")
        #expect(sel.scale == 1.0)
    }

    @Test("equatable works")
    func equatable() {
        let a = LoRASelection(id: "test", filename: "test.safetensors", scale: 0.5)
        let b = LoRASelection(id: "test", filename: "test.safetensors", scale: 0.5)
        let c = LoRASelection(id: "other", filename: "other.safetensors", scale: 0.5)
        #expect(a == b)
        #expect(a != c)
    }

    @Test("identifiable via id")
    func identifiable() {
        let sel = LoRASelection(id: "unique-id", filename: "test.safetensors")
        #expect(sel.id == "unique-id")
    }
}

@Suite("EngineServiceError")
struct EngineServiceErrorTests {
    @Test("notConnected description")
    func notConnected() {
        let error = EngineServiceError.notConnected
        #expect(error.errorDescription?.contains("Not connected") == true)
    }

    @Test("emptyPrompt description")
    func emptyPrompt() {
        let error = EngineServiceError.emptyPrompt
        #expect(error.errorDescription?.contains("empty") == true)
    }

    @Test("serverError includes status and message")
    func serverError() {
        let error = EngineServiceError.serverError(500, "internal error")
        #expect(error.errorDescription?.contains("500") == true)
        #expect(error.errorDescription?.contains("internal error") == true)
    }

    @Test("generationFailed includes message")
    func generationFailed() {
        let error = EngineServiceError.generationFailed("out of memory")
        #expect(error.errorDescription?.contains("out of memory") == true)
    }
}

@Suite("ModelInfo")
struct ModelInfoTests {
    @Test("stores all properties")
    func properties() {
        let model = TestData.makeModelInfo(id: "test-id", family: "flux", displayName: "Test Model")
        #expect(model.id == "test-id")
        #expect(model.family == "flux")
        #expect(model.displayName == "Test Model")
        #expect(model.parametersBillions == 12.0)
        #expect(model.defaultSteps == 9)
        #expect(model.defaultGuidance == 3.5)
        #expect(model.supportsGuidance)
        #expect(model.supportsLoRA)
        #expect(model.defaultResolution == "1024x1024")
        #expect(model.estimatedVRAM_GB == 24.0)
        #expect(model.huggingFaceId == "test-org/test-model")
    }
}

@Suite("PoolModelInfo")
struct PoolModelInfoTests {
    @Test("stores all properties")
    func properties() {
        let pool = TestData.makePoolModel(id: "pool-1", model: "test-model", active: true)
        #expect(pool.id == "pool-1")
        #expect(pool.model == "test-model")
        #expect(pool.active)
        #expect(pool.vramMB == 24576)
        #expect(pool.family == "flux")
    }

    @Test("inactive model")
    func inactive() {
        let pool = TestData.makePoolModel(active: false)
        #expect(!pool.active)
    }
}

@Suite("LoRAInfo")
struct LoRAInfoTests {
    @Test("stores all properties")
    func properties() {
        let lora = TestData.makeLoRAInfo(id: "lora-test", filename: "style.safetensors", isActive: true)
        #expect(lora.id == "lora-test")
        #expect(lora.filename == "style.safetensors")
        #expect(lora.isActive)
        #expect(lora.rank == 16)
        #expect(lora.recommendedScale == 0.8)
        #expect(!lora.quarantined)
        #expect(lora.modelCompatibility == "flux")
        #expect(lora.format == "safetensors")
        #expect(lora.sizeBytes == 50_000_000)
    }

    @Test("quarantined lora")
    func quarantined() {
        let lora = TestData.makeLoRAInfo(quarantined: true)
        #expect(lora.quarantined)
    }

    @Test("tags and triggerwords preserved")
    func tagsAndTriggers() {
        let lora = TestData.makeLoRAInfo()
        #expect(lora.tags.contains("test"))
        #expect(lora.triggerwords.contains("testtrigger"))
        #expect(lora.category == "style")
    }
}

@Suite("QueueInfo")
struct QueueInfoTests {
    @Test("stores all properties")
    func properties() {
        let info = TestData.makeQueueInfo(isRendering: true, pendingCount: 3, renderCount: 100)
        #expect(info.isRendering)
        #expect(info.pendingCount == 3)
        #expect(info.renderCount == 100)
        #expect(info.uptimeSeconds == 3600)
        #expect(info.lastRenderDurationMs == 15000)
        #expect(info.lastError == nil)
        #expect(info.memoryUsageMB == 8192)
    }

    @Test("idle queue info")
    func idle() {
        let info = TestData.makeQueueInfo(isRendering: false, pendingCount: 0)
        #expect(!info.isRendering)
        #expect(info.pendingCount == 0)
    }

    @Test("progress telemetry defaults to nil")
    func progressDefaultsNil() {
        let info = TestData.makeQueueInfo()
        #expect(info.currentJobId == nil)
        #expect(info.progressPercent == nil)
    }

    @Test("progress telemetry fields stored")
    func progressFields() {
        let info = TestData.makeQueueInfo(
            isRendering: true, currentJobId: "job-1", progressPercent: 42.0
        )
        #expect(info.currentJobId == "job-1")
        #expect(info.progressPercent == 42.0)
    }
}
