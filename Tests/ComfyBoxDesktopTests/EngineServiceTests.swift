// EngineServiceTests.swift — Tests for EngineService state, types, and models

import Testing
import Foundation
@testable import ComfyBoxDesktop

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
}
