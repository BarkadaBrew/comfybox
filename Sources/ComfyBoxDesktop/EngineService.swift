// EngineService.swift — SwiftUI-reactive wrapper for WarmServer
//
// Observable class that communicates with the WarmServer via its HTTP API
// using WarmServerClient from the ZImage library. The server itself runs
// as a separate process (the ComfyBox CLI); this service connects to it
// as an HTTP client.

import Foundation
import SwiftUI
import ZImage

/// Generation parameters submitted to the server.
public struct GenerationRequest: Sendable {
    public var prompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    public var seed: UInt64  // 0 = random
    public var modelId: String?
    public var loras: [LoRASelection]
    /// img2img reference image path (nil = pure text-to-image).
    public var initImagePath: String?
    /// How much the reference constrains the result: 0 = ignore, 1 = copy.
    public var imageStrength: Float?
    /// DyPE high-resolution scaling method: "ntk", "yarn", or nil/"none".
    public var dype: String?

    public init(
        prompt: String = "",
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 9,
        guidance: Float = 3.5,
        seed: UInt64 = 0,
        modelId: String? = nil,
        loras: [LoRASelection] = [],
        initImagePath: String? = nil,
        imageStrength: Float? = nil,
        dype: String? = nil
    ) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
        self.dype = dype
        self.modelId = modelId
        self.loras = loras
        self.initImagePath = initImagePath
        self.imageStrength = imageStrength
    }
}

/// A LoRA selected for generation with its scale.
public struct LoRASelection: Sendable, Identifiable, Equatable {
    public var id: String
    public var filename: String
    public var scale: Float

    public init(id: String, filename: String, scale: Float = 1.0) {
        self.id = id
        self.filename = filename
        self.scale = scale
    }
}

/// Decoded generation response from the server.
/// The server encodes all /v1/* responses with `.convertToSnakeCase`.
struct ServerGenerateResponse: Decodable {
    let success: Bool
    let outputPath: String
    let durationMs: Int

    enum CodingKeys: String, CodingKey {
        case success
        case outputPath = "output_path"
        case durationMs = "duration_ms"
    }
}

/// Decoded health response from the server.
struct ServerHealthResponse: Decodable {
    let status: String
    let model: String?
    let modelFamily: String?
    let loaded: Bool?
    let isRendering: Bool?
    let pendingCount: Int?
    let renderCount: Int?
    let uptimeSeconds: Int?
    let lastRenderDurationMs: Int?
    let lastError: String?
    let loras: [ServerLoRAState]?
    let memoryUsageMB: UInt64?
    let currentJobId: String?
    let progressPercent: Double?

    enum CodingKeys: String, CodingKey {
        case status, model, loaded, loras
        case modelFamily = "model_family"
        case isRendering = "is_rendering"
        case pendingCount = "pending_count"
        case renderCount = "render_count"
        case uptimeSeconds = "uptime_seconds"
        case lastRenderDurationMs = "last_render_duration_ms"
        case lastError = "last_error"
        case memoryUsageMB = "memory_usage_mb"
        case currentJobId = "current_job_id"
        case progressPercent = "progress_percent"
    }
}

/// LoRA state from health endpoint.
struct ServerLoRAState: Decodable {
    let source: String
    let scale: Float
}

/// Connection status to the WarmServer.
public enum ServerConnectionState: Sendable {
    case disconnected
    case connecting
    case connected
    case error(String)

    public var label: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .error(let msg): return "Error: \(msg)"
        }
    }

    public var isConnected: Bool {
        if case .connected = self { return true }
        return false
    }
}

// MARK: - Model Info

/// Information about a model available on the server.
public struct ModelInfo: Sendable, Identifiable {
    public let id: String
    public let family: String
    public let variant: String
    public let quantization: String
    public let displayName: String
    public let description: String
    public let parametersBillions: Float
    public let defaultSteps: Int
    public let defaultGuidance: Float
    public let supportsGuidance: Bool
    public let supportsLoRA: Bool
    public let defaultResolution: String
    public let estimatedVRAM_GB: Float
    public let huggingFaceId: String
}

/// Information about a model loaded in the server's model pool.
public struct PoolModelInfo: Sendable, Identifiable {
    public let id: String
    public let model: String
    public let family: String
    public let vramMB: Int
    public let active: Bool
    public let lastUsed: String
}

// MARK: - LoRA Info

/// Information about a LoRA in the server's library.
public struct LoRAInfo: Sendable, Identifiable {
    public let id: String
    public let filename: String
    public let modelCompatibility: String
    public let format: String
    public let rank: Int
    public let sizeBytes: Int
    public let quarantined: Bool
    public let tags: [String]
    public let category: String
    public let triggerwords: [String]
    public let recommendedScale: Float
    public let isActive: Bool
}

// MARK: - Queue Info

/// Information about the server's render queue.
public struct QueueInfo: Sendable {
    public let isRendering: Bool
    public let pendingCount: Int
    public let renderCount: Int
    public let uptimeSeconds: Int
    public let lastRenderDurationMs: Int?
    public let lastError: String?
    public let memoryUsageMB: UInt64
    public let currentJobId: String?
    public let progressPercent: Double?
}

@Observable
@MainActor
public final class EngineService {
    // MARK: - Published State

    public var connectionState: ServerConnectionState = .disconnected
    public var currentModel: String?
    public var currentModelFamily: String?
    public var queueCount: Int = 0
    public var isGenerating: Bool = false
    public var lastGeneratedImagePath: String?
    public var lastError: String?
    public var lastDurationMs: Int?

    // Model pool state
    public var availableModels: [ModelInfo] = []
    public var poolModels: [PoolModelInfo] = []
    public var isLoadingModel: Bool = false

    // LoRA state
    public var availableLoras: [LoRAInfo] = []
    public var activeLoraIds: [String] = []
    public var isSwappingLoras: Bool = false

    // Queue state
    public var queueInfo: QueueInfo?

    // MARK: - Configuration

    public var serverHost: String = "127.0.0.1"
    public var serverPort: UInt16 = 7870
    public var outputDirectory: String = NSString(string: "~/Pictures/ComfyBox").expandingTildeInPath

    // MARK: - Private

    private var client: WarmServerClient?
    // nonisolated(unsafe) so the nonisolated deinit can cancel it; Task.cancel()
    // is thread-safe, and all other accesses happen on the main actor.
    private nonisolated(unsafe) var healthPollTask: Task<Void, Never>?

    public init() {
        let config = AppConfig.load()
        self.serverHost = config.serverHost
        self.serverPort = config.serverPort
        self.outputDirectory = NSString(string: config.outputDirectory).expandingTildeInPath
    }

    deinit {
        healthPollTask?.cancel()
    }

    // MARK: - Connection Management

    /// Connect to an already-running WarmServer instance.
    public func connect() {
        connectionState = .connecting
        lastError = nil

        let newClient = WarmServerClient(host: serverHost, port: serverPort)
        self.client = newClient

        // Start polling health to verify connection and track state.
        healthPollTask?.cancel()
        healthPollTask = Task { [weak self] in
            // Initial connection check.
            await self?.pollHealth()

            // Fetch models and LoRAs on first connect.
            if let self = self, self.connectionState.isConnected {
                await self.refreshModels()
                await self.refreshLoras()
            }

            // Poll every 3 seconds.
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(3))
                if Task.isCancelled { break }
                await self?.pollHealth()
            }
        }
    }

    /// Disconnect from the server (stops polling, does NOT shut down the server).
    public func disconnect() {
        healthPollTask?.cancel()
        healthPollTask = nil
        client = nil
        connectionState = .disconnected
        currentModel = nil
        currentModelFamily = nil
        queueCount = 0
        availableModels = []
        poolModels = []
        availableLoras = []
        activeLoraIds = []
        queueInfo = nil
    }

    // MARK: - Generation

    /// Attach the fruit mode to a request body as `content_mode`. Always sent
    /// (including neutral) so the server's behavior is explicit, never inferred.
    nonisolated static func attachingContentMode(_ base: [String: Any], mode: ContentMode) -> [String: Any] {
        var out = base
        out["content_mode"] = mode.rawValue
        return out
    }

    /// Submit a generation request to the server. Returns the output file path on success.
    @discardableResult
    public func generate(_ request: GenerationRequest) async throws -> String {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EngineServiceError.emptyPrompt
        }

        isGenerating = true
        lastError = nil

        // Poll health quickly while the render runs so progress_percent updates
        // smoothly (the idle poll is every 3s).
        let progressPoll = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(700))
                await self?.pollHealth()
            }
        }
        defer { isGenerating = false; progressPoll.cancel() }

        // Build the output path.
        let timestamp = Int(Date().timeIntervalSince1970)
        let outputFilename = "comfybox-\(timestamp).png"

        // Ensure output directory exists.
        try FileManager.default.createDirectory(
            atPath: outputDirectory,
            withIntermediateDirectories: true
        )
        let outputPath = (outputDirectory as NSString).appendingPathComponent(outputFilename)

        // Build JSON payload matching the server API GeneratePayload.
        var payloadDict: [String: Any] = [
            "source": "desktop",
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance": request.guidance,
            "outputPath": outputPath
        ]

        if request.seed > 0 {
            payloadDict["seed"] = request.seed
        }

        // img2img: reference image + strength (server maps strength->denoise).
        if let initImagePath = request.initImagePath, !initImagePath.isEmpty {
            payloadDict["imagePath"] = initImagePath
            payloadDict["imageStrength"] = request.imageStrength ?? 0.6
        }
        // DyPE high-resolution scaling (ntk / yarn).
        if let dype = request.dype, dype != "none", !dype.isEmpty {
            payloadDict["dype"] = dype
        }

        let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)
        let (status, responseData) = try await client.post("/v1/generate", body: bodyData)

        guard status == 200 else {
            let errorMessage: String
            if let json = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
               let msg = json["error"] as? String {
                errorMessage = msg
            } else {
                errorMessage = "Server returned status \(status)"
            }
            lastError = errorMessage
            throw EngineServiceError.serverError(status, errorMessage)
        }

        let response: ServerGenerateResponse
        do {
            response = try JSONDecoder().decode(ServerGenerateResponse.self, from: responseData)
        } catch {
            let msg = "Failed to decode server response: \(error.localizedDescription)"
            lastError = msg
            throw EngineServiceError.generationFailed(msg)
        }

        guard response.success else {
            let msg = "Generation reported failure"
            lastError = msg
            throw EngineServiceError.generationFailed(msg)
        }

        lastGeneratedImagePath = response.outputPath
        lastDurationMs = response.durationMs
        return response.outputPath
    }

    // MARK: - Model Management

    /// Fetch the list of all available models from the server registry.
    public func refreshModels() async {
        guard let client = client, connectionState.isConnected else { return }

        do {
            let (status, data) = try await client.get("/v1/models")
            guard status == 200 else { return }

            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let models = json["models"] as? [[String: Any]] else { return }

            availableModels = models.compactMap { dict -> ModelInfo? in
                guard let id = dict["id"] as? String,
                      let family = dict["family"] as? String,
                      let variant = dict["variant"] as? String,
                      let quantization = dict["quantization"] as? String,
                      let displayName = dict["display_name"] as? String,
                      let description = dict["description"] as? String else { return nil }

                return ModelInfo(
                    id: id,
                    family: family,
                    variant: variant,
                    quantization: quantization,
                    displayName: displayName,
                    description: description,
                    parametersBillions: (dict["parameters_b"] as? Float) ?? 0,
                    defaultSteps: (dict["default_steps"] as? Int) ?? 9,
                    defaultGuidance: (dict["default_guidance"] as? Float) ?? 3.5,
                    supportsGuidance: (dict["supports_guidance"] as? Bool) ?? false,
                    supportsLoRA: (dict["supports_lora"] as? Bool) ?? false,
                    defaultResolution: (dict["default_resolution"] as? String) ?? "1024x1024",
                    estimatedVRAM_GB: (dict["estimated_vram_gb"] as? Float) ?? 0,
                    huggingFaceId: (dict["huggingface_id"] as? String) ?? ""
                )
            }

            // Also refresh pool status.
            await refreshPool()
        } catch {
            // Non-fatal — models list is informational.
        }
    }

    /// Fetch the current model pool status.
    public func refreshPool() async {
        guard let client = client, connectionState.isConnected else { return }

        do {
            let (status, data) = try await client.get("/v1/model/pool")
            guard status == 200 else { return }

            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let pool = json["pool"] as? [[String: Any]] else { return }

            poolModels = pool.compactMap { dict -> PoolModelInfo? in
                guard let model = dict["model"] as? String,
                      let family = dict["family"] as? String else { return nil }

                let poolKey = model.replacingOccurrences(of: "/", with: "-").lowercased()
                return PoolModelInfo(
                    id: poolKey,
                    model: model,
                    family: family,
                    vramMB: (dict["vram_mb"] as? Int) ?? 0,
                    active: (dict["active"] as? Bool) ?? false,
                    lastUsed: (dict["last_used"] as? String) ?? ""
                )
            }
        } catch {
            // Non-fatal.
        }
    }

    /// Load a model into the server's model pool.
    public func loadModel(id: String, quantization: String? = nil, activate: Bool = true) async throws {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }

        isLoadingModel = true
        defer { isLoadingModel = false }

        var payloadDict: [String: Any] = [
            "model": id,
            "activate": activate,
            "wait": true
        ]
        if let q = quantization {
            payloadDict["quantization"] = q
        }

        let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)
        let (status, responseData) = try await client.post("/v1/model/load", body: bodyData)

        guard status == 200 else {
            let errorMessage = parseErrorMessage(from: responseData) ?? "Server returned status \(status)"
            throw EngineServiceError.serverError(status, errorMessage)
        }

        await refreshPool()
    }

    /// Activate an already-loaded model in the pool.
    public func activateModel(id: String) async throws {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }

        let payloadDict: [String: Any] = ["model": id]
        let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)
        let (status, responseData) = try await client.post("/v1/model/activate", body: bodyData)

        guard status == 200 else {
            let errorMessage = parseErrorMessage(from: responseData) ?? "Server returned status \(status)"
            throw EngineServiceError.serverError(status, errorMessage)
        }

        await refreshPool()
    }

    /// Unload a model from the pool.
    public func unloadModel(id: String) async throws {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }

        let payloadDict: [String: Any] = ["model": id]
        let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)
        let (status, responseData) = try await client.post("/v1/model/unload", body: bodyData)

        guard status == 200 else {
            let errorMessage = parseErrorMessage(from: responseData) ?? "Server returned status \(status)"
            throw EngineServiceError.serverError(status, errorMessage)
        }

        await refreshPool()
    }

    // MARK: - LoRA Management

    /// Fetch the list of available LoRAs from the server's library.
    public func refreshLoras() async {
        guard let client = client, connectionState.isConnected else { return }

        do {
            let (status, data) = try await client.get("/v1/loras")
            guard status == 200 else { return }

            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let loras = json["loras"] as? [[String: Any]],
                  let activeLoras = json["active_loras"] as? [String] else { return }

            activeLoraIds = activeLoras

            availableLoras = loras.compactMap { dict -> LoRAInfo? in
                guard let id = dict["id"] as? String,
                      let filename = dict["filename"] as? String else { return nil }

                return LoRAInfo(
                    id: id,
                    filename: filename,
                    modelCompatibility: (dict["model_compatibility"] as? String) ?? "unknown",
                    format: (dict["format"] as? String) ?? "unknown",
                    rank: (dict["rank"] as? Int) ?? 0,
                    sizeBytes: (dict["size_bytes"] as? Int) ?? 0,
                    quarantined: (dict["quarantined"] as? Bool) ?? false,
                    tags: (dict["tags"] as? [String]) ?? [],
                    category: (dict["category"] as? String) ?? "",
                    triggerwords: (dict["triggerwords"] as? [String]) ?? [],
                    recommendedScale: (dict["recommended_scale"] as? Float) ?? 1.0,
                    isActive: activeLoras.contains(id)
                )
            }
        } catch {
            // Non-fatal.
        }
    }

    /// Swap the active LoRAs on the server.
    public func swapLoras(_ selections: [LoRASelection]) async throws {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }

        isSwappingLoras = true
        defer { isSwappingLoras = false }

        let loraEntries = selections.map { lora -> [String: Any] in
            ["path": lora.id, "scale": lora.scale]
        }
        let payloadDict: [String: Any] = ["loras": loraEntries]
        let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)
        let (status, responseData) = try await client.post("/v1/lora/swap", body: bodyData)

        guard status == 200 else {
            let errorMessage = parseErrorMessage(from: responseData) ?? "Server returned status \(status)"
            throw EngineServiceError.serverError(status, errorMessage)
        }

        await refreshLoras()
    }

    // MARK: - Health Polling

    private func pollHealth() async {
        guard let client = client else {
            connectionState = .disconnected
            return
        }

        do {
            let (status, data) = try await client.get("/health")
            guard status == 200 else {
                connectionState = .error("Server returned \(status)")
                return
            }

            let decoder = JSONDecoder()
            let health = try decoder.decode(ServerHealthResponse.self, from: data)

            connectionState = .connected
            currentModel = health.model
            currentModelFamily = health.modelFamily
            let pending = health.pendingCount ?? 0
            let rendering = (health.isRendering ?? false) ? 1 : 0
            queueCount = pending + rendering

            // Update queue info from health data.
            queueInfo = QueueInfo(
                isRendering: health.isRendering ?? false,
                pendingCount: health.pendingCount ?? 0,
                renderCount: health.renderCount ?? 0,
                uptimeSeconds: health.uptimeSeconds ?? 0,
                lastRenderDurationMs: health.lastRenderDurationMs,
                lastError: health.lastError,
                memoryUsageMB: health.memoryUsageMB ?? 0,
                currentJobId: health.currentJobId,
                progressPercent: health.progressPercent
            )
        } catch is WarmServerClientError {
            connectionState = .disconnected
            queueInfo = nil
        } catch {
            connectionState = .error(error.localizedDescription)
            queueInfo = nil
        }
    }

    // MARK: - Prompt Enhancement

    /// Send a prompt to the server's LLM enhancement endpoint.
    /// Returns the enhanced prompt string on success.
    public func enhancePrompt(_ prompt: String, contentMode: ContentMode = .neutral) async throws -> String {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }

        let payloadDict = Self.attachingContentMode(["prompt": prompt], mode: contentMode)
        let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)
        let (status, responseData) = try await client.post("/v1/enhance", body: bodyData)

        guard status == 200 else {
            let errorMessage = parseErrorMessage(from: responseData) ?? "Server returned status \(status)"
            throw EngineServiceError.serverError(status, errorMessage)
        }

        guard let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let enhanced = json["prompt"] as? String else {
            throw EngineServiceError.generationFailed("Invalid enhance response")
        }

        return enhanced
    }

    // MARK: - Character Registry

    /// Fetch registered characters from the server. `/v1/characters` returns a bare
    /// JSON array (snake_case); `default_loras` are `{filename, scale}` objects.
    public func fetchCharacters() async -> [CharacterEntry] {
        guard let client = client, connectionState.isConnected else { return [] }

        do {
            let (status, data) = try await client.get("/v1/characters")
            guard status == 200 else { return [] }

            guard let arr = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { return [] }

            return arr.compactMap { dict -> CharacterEntry? in
                guard let id = dict["id"] as? String,
                      let name = dict["name"] as? String else { return nil }

                let loras: [String]
                if let objs = dict["default_loras"] as? [[String: Any]] {
                    loras = objs.compactMap { $0["filename"] as? String }
                } else {
                    loras = (dict["default_loras"] as? [String]) ?? []
                }

                return CharacterEntry(
                    id: id,
                    name: name,
                    kind: (dict["kind"] as? String) ?? "character",
                    description: (dict["description"] as? String) ?? "",
                    base: dict["base"] as? String,
                    banana: dict["banana"] as? String,
                    avocado: dict["avocado"] as? String,
                    defaultLoras: loras,
                    promptSnippet: (dict["prompt_snippet"] as? String) ?? "",
                    negativePrompt: dict["negative_prompt"] as? String,
                    triggerWords: dict["trigger_words"] as? String,
                    tags: (dict["tags"] as? [String]) ?? []
                )
            }
        } catch {
            return []
        }
    }

    /// Create or update a character. The server accepts camelCase input (tolerant decode).
    /// Sends every field — the server upsert replaces the whole entry, so omitting
    /// a field (e.g. the description tiers) would silently erase it.
    public func saveCharacter(_ c: CharacterEntry) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        var dict: [String: Any] = [
            "id": c.id,
            "name": c.name,
            "kind": c.kind,
            "description": c.description,
            "promptSnippet": c.promptSnippet,
            "tags": c.tags,
            "defaultLoras": c.defaultLoras.map { ["filename": $0, "scale": 1.0] as [String: Any] }
        ]
        if let base = c.base { dict["base"] = base }
        if let banana = c.banana { dict["banana"] = banana }
        if let avocado = c.avocado { dict["avocado"] = avocado }
        if let negativePrompt = c.negativePrompt { dict["negativePrompt"] = negativePrompt }
        if let triggerWords = c.triggerWords { dict["triggerWords"] = triggerWords }
        let body = try JSONSerialization.data(withJSONObject: dict)
        let (status, data) = try await client.post("/v1/characters", body: body)
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Failed to save character")
        }
    }

    // MARK: - Nearline storage (/v1/nearline)

    /// One catalog entry on attached storage, possibly staged locally.
    public struct NearlineEntry: Identifiable, Sendable, Equatable {
        public let name: String
        public let path: String
        public let sizeMB: Double
        public let kind: String
        public let staged: Bool
        public var id: String { name }

        public var sizeLabel: String {
            sizeMB >= 1024 ? String(format: "%.1f GB", sizeMB / 1024) : String(format: "%.0f MB", sizeMB)
        }
    }

    public struct NearlineCatalog: Sendable, Equatable {
        public var roots: [String] = []
        public var cacheLimitGB: Double = 0
        public var stagedMB: Double = 0
        public var items: [NearlineEntry] = []
    }

    private func parseNearline(_ data: Data) -> NearlineCatalog? {
        guard let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        var catalog = NearlineCatalog()
        catalog.roots = (dict["roots"] as? [String]) ?? []
        catalog.cacheLimitGB = (dict["cache_limit_gb"] as? Double) ?? 0
        catalog.stagedMB = (dict["staged_mb"] as? Double) ?? 0
        catalog.items = ((dict["items"] as? [[String: Any]]) ?? []).compactMap { item in
            guard let name = item["name"] as? String else { return nil }
            return NearlineEntry(
                name: name,
                path: (item["path"] as? String) ?? "",
                sizeMB: (item["size_mb"] as? Double) ?? 0,
                kind: (item["kind"] as? String) ?? "lora",
                staged: (item["staged"] as? Bool) ?? false
            )
        }
        return catalog
    }

    /// Fetch the nearline catalog (attached-storage models/LoRAs).
    public func fetchNearline() async -> NearlineCatalog? {
        guard let client = client, connectionState.isConnected else { return nil }
        guard let (status, data) = try? await client.get("/v1/nearline"), status == 200 else { return nil }
        return parseNearline(data)
    }

    /// Rescan the attached-storage roots. Returns the fresh catalog.
    public func scanNearline() async throws -> NearlineCatalog? {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let (status, data) = try await client.post("/v1/nearline/scan", body: Data("{}".utf8))
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Nearline scan failed")
        }
        return parseNearline(data)
    }

    /// Stage an item to local storage (or evict its staged copy).
    public func nearlineAction(_ action: String, name: String) async throws -> NearlineCatalog? {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let body = try JSONSerialization.data(withJSONObject: ["name": name])
        let (status, data) = try await client.post("/v1/nearline/\(action)", body: body)
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Nearline \(action) failed")
        }
        return parseNearline(data)
    }

    /// Evict every staged nearline item to free the primary drive. Returns the
    /// refreshed catalog and how many were evicted.
    @discardableResult
    public func evictAllStaged() async throws -> (catalog: NearlineCatalog?, evicted: Int) {
        guard client != nil, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let catalog = await fetchNearline()
        let staged = catalog?.items.filter { $0.staged } ?? []
        var last: NearlineCatalog?
        var count = 0
        for item in staged {
            if let cat = try? await nearlineAction("evict", name: item.name) { last = cat; count += 1 }
        }
        if last == nil { last = await fetchNearline() }
        return (last, count)
    }

    /// Quarantine (or release) a LoRA in the server's library.
    public func quarantineLora(id: String, quarantine: Bool) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let path = "/v1/loras/\(id)/quarantine"
        let (status, data): (Int, Data)
        if quarantine {
            (status, data) = try await client.post(path, body: Data("{}".utf8))
        } else {
            (status, data) = try await client.delete(path)
        }
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Quarantine failed")
        }
        await refreshLoras()
    }

    /// Free/total bytes of the volume backing the home directory (the primary
    /// drive staged copies land on).
    public nonisolated static func primaryDiskInfo() -> (freeGB: Double, totalGB: Double)? {
        let url = URL(fileURLWithPath: NSHomeDirectory())
        guard let values = try? url.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey, .volumeTotalCapacityKey]) else { return nil }
        let gb = 1024.0 * 1024.0 * 1024.0
        let free = Double(values.volumeAvailableCapacityForImportantUsage ?? 0) / gb
        let total = Double(values.volumeTotalCapacity ?? 0) / gb
        return total > 0 ? (free, total) : nil
    }

    // MARK: - Inpaint (/v1/generate with base image + mask)

    /// Inpaint the masked (white) region of `baseImagePNG` using `prompt`.
    /// `maskPNG` is white where content should be regenerated, black elsewhere.
    public func inpaint(
        baseImagePNG: Data, maskPNG: Data, prompt: String, negativePrompt: String? = nil,
        width: Int, height: Int, steps: Int, guidance: Float, denoise: Float,
        maskGrow: Int = 8, maskFeather: Int = 8, seed: UInt64 = 0, outputPath: String
    ) async throws -> String {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        var body: [String: Any] = [
            "prompt": prompt,
            "width": width, "height": height,
            "steps": steps, "guidance": guidance,
            "denoise": denoise,
            "maskGrow": maskGrow, "maskFeather": maskFeather,
            "inpaint_image_base64": baseImagePNG.base64EncodedString(),
            "mask_base64": maskPNG.base64EncodedString(),
            "outputPath": outputPath,
        ]
        if let n = negativePrompt, !n.isEmpty { body["negativePrompt"] = n }
        if seed > 0 { body["seed"] = seed }
        let data = try JSONSerialization.data(withJSONObject: body)
        let (status, responseData) = try await client.post("/v1/generate", body: data)
        guard status == 200,
              let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let out = (json["output_path"] ?? json["outputPath"]) as? String
        else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: responseData) ?? "Inpaint failed")
        }
        return out
    }

    // MARK: - Video (LTX-2 local, /v1/video/generate)

    public struct VideoRequest: Sendable {
        public var prompt: String
        public var initImagePath: String?   // nil = text-to-video
        public var width: Int
        public var height: Int
        public var frames: Int               // 1 + 8k
        public var steps: Int
        public var seed: UInt64
        public var strength: Float
        public var extendToSeconds: Float
        public var loraPath: String?
        public var loraStrength: Float
        public var outputPath: String

        public init(
            prompt: String, initImagePath: String? = nil,
            width: Int = 704, height: Int = 448, frames: Int = 97,
            steps: Int = 8, seed: UInt64 = 42, strength: Float = 1.0,
            extendToSeconds: Float = 0, loraPath: String? = nil,
            loraStrength: Float = 1.0, outputPath: String
        ) {
            self.prompt = prompt; self.initImagePath = initImagePath
            self.width = width; self.height = height; self.frames = frames
            self.steps = steps; self.seed = seed; self.strength = strength
            self.extendToSeconds = extendToSeconds
            self.loraPath = loraPath; self.loraStrength = loraStrength
            self.outputPath = outputPath
        }
    }

    public struct VideoResult: Sendable {
        public let outputPath: String
        public let frameCount: Int
        public let durationSeconds: Double
        public let elapsedSeconds: Double
    }

    /// Generate a video locally via LTX-2. Long-running (minutes); the local
    /// backend responds synchronously with the MP4 path.
    public func generateVideo(_ request: VideoRequest) async throws -> VideoResult {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        var body: [String: Any] = [
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "frames": request.frames,
            "steps": request.steps,
            "seed": request.seed,
            "strength": request.strength,
            "extend_to_seconds": request.extendToSeconds,
            "output_path": request.outputPath,
        ]
        if let initImagePath = request.initImagePath, !initImagePath.isEmpty {
            body["image_path"] = initImagePath
        }
        if let loraPath = request.loraPath, !loraPath.isEmpty {
            body["lora_path"] = loraPath
            body["lora_strength"] = request.loraStrength
        }
        let bodyData = try JSONSerialization.data(withJSONObject: body)
        let (status, data) = try await client.post("/v1/video/generate", body: bodyData)
        guard status == 200,
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let outputPath = json["output_path"] as? String
        else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Video generation failed (is the server started with --ltx2-weights?)")
        }
        return VideoResult(
            outputPath: outputPath,
            frameCount: (json["frame_count"] as? Int) ?? 0,
            durationSeconds: (json["duration_seconds"] as? Double) ?? 0,
            elapsedSeconds: (json["elapsed_seconds"] as? Double) ?? 0
        )
    }

    // MARK: - Upscale (/v1/upscale — SeedVR2 creative upscale)

    /// Upscale an image to a target long-side resolution. Returns the output
    /// file path written by the server. Runs through the render queue, so it
    /// can take a while on large targets.
    @discardableResult
    public func upscale(imagePath: String, targetResolution: Int) async throws -> String {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let body = try JSONSerialization.data(withJSONObject: [
            "image_path": imagePath,
            "target_resolution": targetResolution,
        ])
        let (status, data) = try await client.post("/v1/upscale", body: body)
        guard status == 200,
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let outputPath = json["output_path"] as? String
        else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Upscale failed")
        }
        return outputPath
    }

    /// Rescan the server's LoRA library (after downloading a new file into it).
    public func scanLoras() async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let (status, data) = try await client.post("/v1/loras/scan", body: Data("{}".utf8))
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "LoRA scan failed")
        }
    }

    // MARK: - Queue management (/v1/queue)

    /// One pending job in the server render queue.
    public struct QueueJob: Identifiable, Sendable, Equatable {
        public let id: String
        public let kind: String
        public let summary: String
        public let source: String
        public let enqueuedAt: Date?
    }

    /// Snapshot of the server queue: active operation + pending jobs.
    public struct QueueJobList: Sendable, Equatable {
        public var isRendering: Bool = false
        public var isPaused: Bool = false
        public var activeJobId: String?
        public var activeSummary: String?
        public var activeSource: String?
        public var progressPercent: Int?
        public var pending: [QueueJob] = []
        public var renderCount: Int = 0
        public var failedCount: Int = 0
    }

    /// Fetch the detailed queue listing.
    public func fetchQueueJobs() async -> QueueJobList? {
        guard let client = client, connectionState.isConnected else { return nil }
        do {
            let (status, data) = try await client.get("/v1/queue")
            guard status == 200,
                  let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            let iso = ISO8601DateFormatter()
            var list = QueueJobList()
            list.isRendering = (dict["is_rendering"] as? Bool) ?? false
            list.isPaused = (dict["is_paused"] as? Bool) ?? false
            list.activeJobId = dict["active_job_id"] as? String
            list.activeSummary = dict["active_summary"] as? String
            list.activeSource = dict["active_source"] as? String
            list.progressPercent = dict["progress_percent"] as? Int
            list.renderCount = (dict["render_count"] as? Int) ?? 0
            list.failedCount = (dict["failed_count"] as? Int) ?? 0
            list.pending = ((dict["pending"] as? [[String: Any]]) ?? []).compactMap { job in
                guard let id = job["id"] as? String else { return nil }
                return QueueJob(
                    id: id,
                    kind: (job["kind"] as? String) ?? "job",
                    summary: (job["summary"] as? String) ?? "",
                    source: (job["source"] as? String) ?? "api",
                    enqueuedAt: (job["enqueued_at"] as? String).flatMap { iso.date(from: $0) }
                )
            }
            return list
        } catch {
            return nil
        }
    }

    /// Cancel the in-flight render (pending jobs continue).
    public func interruptRender() async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let (status, data) = try await client.post("/v1/queue/interrupt", body: Data("{}".utf8))
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Interrupt failed")
        }
    }

    /// Cancel one pending job by id.
    public func cancelQueueJob(id: String) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let enc = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
        let (status, data) = try await client.delete("/v1/queue/\(enc)")
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Cancel failed")
        }
    }

    /// Clear every pending job (the active render continues).
    public func clearQueue() async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let (status, data) = try await client.post("/v1/queue/clear", body: Data("{}".utf8))
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Clear failed")
        }
    }

    /// Pause or resume the queue (a paused queue finishes the current render but
    /// starts no new pending jobs).
    public func setQueuePaused(_ paused: Bool) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let (status, data) = try await client.post("/v1/queue/\(paused ? "pause" : "resume")", body: Data("{}".utf8))
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Pause failed")
        }
    }

    /// Reorder a pending job. direction: up | down | top | bottom.
    public func moveQueueJob(id: String, direction: String) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let enc = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
        let body = Data("{\"direction\":\"\(direction)\"}".utf8)
        let (status, data) = try await client.post("/v1/queue/\(enc)/move", body: body)
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Move failed")
        }
    }

    // MARK: - Presets (/v1/presets — the server-side canonical store)

    /// Fetch the server preset list (bare snake_case array).
    public func fetchPresets() async -> [ServerPreset] {
        guard let client = client, connectionState.isConnected else { return [] }
        do {
            let (status, data) = try await client.get("/v1/presets")
            guard status == 200 else { return [] }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode([ServerPreset].self, from: data)
        } catch {
            return []
        }
    }

    /// Create or update a server preset. Sends the full document (camelCase,
    /// tolerated by the server) — upsert replaces the stored preset.
    public func savePreset(_ preset: ServerPreset) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let body = try JSONEncoder().encode(preset)
        let (status, data) = try await client.post("/v1/presets", body: body)
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Failed to save preset")
        }
    }

    /// Import presets from the old image-service (idempotent). Returns the
    /// number newly imported.
    @discardableResult
    public func importLegacyPresets() async throws -> Int {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let (status, data) = try await client.post("/v1/presets/import-legacy", body: Data("{}".utf8))
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Import failed")
        }
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return (json?["imported"] as? Int) ?? 0
    }

    /// Delete a server preset by id.
    public func deletePreset(id: String) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let enc = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
        let (status, data) = try await client.delete("/v1/presets/\(enc)")
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Failed to delete preset")
        }
    }

    /// Delete a character by id.
    public func deleteCharacter(id: String) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let enc = id.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? id
        let (status, data) = try await client.delete("/v1/characters/\(enc)")
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Failed to delete character")
        }
    }

    // MARK: - Server Config (/v1/config)

    /// Fetch the server config document (~/.comfybox/config.json). The config document
    /// is canonical camelCase, decoded with a plain decoder (not the snake_case API DTOs).
    public func fetchServerConfig() async throws -> ComfyBoxServerConfig {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }
        let (status, data) = try await client.get("/v1/config")
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Failed to load config")
        }
        return try JSONDecoder().decode(ComfyBoxServerConfig.self, from: data)
    }

    /// Persist the full server config document. PUT replaces the document, so callers
    /// should fetch, mutate, and save to preserve fields they don't manage.
    public func saveServerConfig(_ config: ComfyBoxServerConfig) async throws {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }
        let body = try JSONEncoder().encode(config)
        let (status, data) = try await client.put("/v1/config", body: body)
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Failed to save config")
        }
    }

    // MARK: - Helpers

    private func parseErrorMessage(from data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json["error"] as? String
    }
}

// MARK: - Errors

public enum EngineServiceError: Error, LocalizedError {
    case notConnected
    case emptyPrompt
    case serverError(Int, String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notConnected:
            return "Not connected to WarmServer. Start ComfyBox or click Connect."
        case .emptyPrompt:
            return "Prompt cannot be empty"
        case .serverError(let status, let msg):
            return "Server error (\(status)): \(msg)"
        case .generationFailed(let msg):
            return "Generation failed: \(msg)"
        }
    }
}
