// EngineService.swift — SwiftUI-reactive wrapper for WarmServer
//
// Observable class that communicates with the WarmServer via its HTTP API
// using WarmServerClient from the ZImage library. The server itself runs
// as a separate process (the ComfyBox CLI); this service connects to it
// as an HTTP client.

import Foundation
import SwiftUI
import AppKit
import ZImage

/// Generation parameters submitted to the server.
public struct GenerationRequest: Sendable {
    public var prompt: String
    public var negativePrompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var guidance: Float
    /// Solver name (`res_2s`, `deis_3m`, `euler`, …). Nil = model default.
    public var sampler: String?
    /// Sigma/noise schedule (`krea2`, `beta`, `bong_tangent`, …).
    /// Nil = model default.
    public var sigmaSchedule: String?
    public var seed: UInt64  // 0 = random
    public var modelId: String?
    public var loras: [LoRASelection]
    /// img2img reference image path (nil = pure text-to-image).
    public var initImagePath: String?
    /// How much the reference constrains the result: 0 = ignore, 1 = copy.
    public var imageStrength: Float?
    /// DyPE high-resolution scaling method: "ntk", "yarn", or nil/"none".
    public var dype: String?
    /// Text-conditioning gain on the Krea2 fusion projector (projector-scale
    /// trick). 1.0 = neutral; >1 strengthens prompt adherence with no CFG cost.
    public var projectorScale: Float

    public init(
        prompt: String = "",
        negativePrompt: String = "",
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 9,
        guidance: Float = 3.5,
        projectorScale: Float = 1.0,
        sampler: String? = nil,
        sigmaSchedule: String? = nil,
        seed: UInt64 = 0,
        modelId: String? = nil,
        loras: [LoRASelection] = [],
        initImagePath: String? = nil,
        imageStrength: Float? = nil,
        dype: String? = nil
    ) {
        self.prompt = prompt
        self.negativePrompt = negativePrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.sampler = sampler
        self.sigmaSchedule = sigmaSchedule
        self.seed = seed
        self.dype = dype
        self.modelId = modelId
        self.loras = loras
        self.initImagePath = initImagePath
        self.imageStrength = imageStrength
        self.projectorScale = projectorScale
    }
}

/// A LoRA selected for generation with its scale.
public struct LoRASelection: Sendable, Identifiable, Equatable, Codable {
    public var id: String
    public var filename: String
    public var scale: Float
    public var role: String?

    public init(id: String, filename: String, scale: Float = 1.0, role: String? = nil) {
        self.id = id
        self.filename = filename
        self.scale = scale
        self.role = role
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
    let isPaused: Bool?
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
        case isPaused = "is_paused"
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
    /// Engine-level creation gate, mirrored from /health `is_paused` every
    /// poll — so a pause toggled from ANY surface (toolbar, HTTP API, MCP)
    /// shows truthfully here within one poll cycle.
    public var queuePaused: Bool = false
    /// Counter rather than a bool so a queued/background generate() running
    /// alongside the foreground one doesn't clear this out from under it —
    /// isGenerating stays true until the LAST concurrent call finishes.
    private var activeGenerateCount: Int = 0
    public var isGenerating: Bool { activeGenerateCount > 0 }
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
    /// Why the LoRA catalog is empty, WHEN it is empty because the fetch failed
    /// rather than because there are genuinely no LoRAs. Every failure path in
    /// refreshLoras() used to return silently, so a server-side fault (e.g.
    /// /v1/loras hanging) presented as a permanently empty list with no
    /// explanation and no retry (observed 2026-08-10).
    public var loraLoadError: String?
    public var isSwappingLoras: Bool = false

    // Queue state
    public var queueInfo: QueueInfo?

    /// Latest live-denoising preview frame, polled alongside health while a
    /// render is active (GH #216). nil when idle or before the first frame.
    public var livePreviewImage: NSImage?

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
            var sinceLoraRetry = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(3))
                if Task.isCancelled { break }
                await self?.pollHealth()
                // Self-heal the catalog. refreshLoras() ran ONCE on connect, so a
                // failure there left the list empty until the app was restarted.
                // Retry ~every 30s while it is empty or errored; a healthy,
                // populated catalog costs nothing.
                sinceLoraRetry += 1
                if sinceLoraRetry >= 10, let self = self, self.connectionState.isConnected,
                   self.availableLoras.isEmpty || self.loraLoadError != nil {
                    sinceLoraRetry = 0
                    await self.refreshLoras()
                }
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
    public func generate(_ request: GenerationRequest, contentMode: ContentMode = .neutral) async throws -> String {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }
        guard !request.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw EngineServiceError.emptyPrompt
        }

        activeGenerateCount += 1
        lastError = nil
        // Clear any stale frame from a previous render so the preview pane
        // doesn't briefly show the last render's final denoising step.
        livePreviewImage = nil

        // Poll health quickly while the render runs so progress_percent updates
        // smoothly (the idle poll is every 3s).
        let progressPoll = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(700))
                await self?.pollHealth()
            }
        }
        defer { activeGenerateCount -= 1; progressPoll.cancel() }

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

        if let sampler = request.sampler?.trimmingCharacters(in: .whitespacesAndNewlines),
           !sampler.isEmpty {
            // `sampler` is the user-facing alias accepted by GeneratePayload;
            // it maps to the engine's historical `scheduler` field.
            payloadDict["sampler"] = sampler
        }
        if let schedule = request.sigmaSchedule?.trimmingCharacters(in: .whitespacesAndNewlines),
           !schedule.isEmpty {
            payloadDict["sigma_schedule"] = schedule
        }

        if !request.negativePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            payloadDict["negative_prompt"] = request.negativePrompt
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
        // Projector-scale trick: CFG-free prompt-adherence gain. 1.0 = omit
        // (byte-identical); the engine also treats absent as neutral.
        if request.projectorScale != 1.0 {
            payloadDict["projector_scale"] = request.projectorScale
        }

        payloadDict = Self.attachingContentMode(payloadDict, mode: contentMode)

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

        // Submit with wait:false + poll (2026-08-11): a big-model switch takes
        // ~70s and the old wait:true request died to SwiftUI task cancellation
        // ("Network error: cancelled") whenever the initiating view churned —
        // while the engine finished the switch anyway and the UI showed a
        // false failure. The 202 submit returns in milliseconds; polling is
        // cheap and each iteration is individually cancellation-safe.
        var payloadDict: [String: Any] = [
            "model": id,
            "activate": activate,
            "wait": false
        ]
        if let q = quantization {
            payloadDict["quantization"] = q
        }

        let bodyData = try JSONSerialization.data(withJSONObject: payloadDict)
        let (status, responseData) = try await client.post("/v1/model/load", body: bodyData)

        guard status == 200 || status == 202 else {
            let errorMessage = parseErrorMessage(from: responseData) ?? "Server returned status \(status)"
            throw EngineServiceError.serverError(status, errorMessage)
        }

        // Poll the pool until the load lands (activate also flips /health's
        // model). Budget generously: 25GB load + 8-bit quantization ≈ 70-80s.
        let deadline = Date().addingTimeInterval(300)
        // The pool may key the entry by a resolved local DIRECTORY rather than
        // the catalog id (e.g. id "kroma-v0.2-turbo" → pool model
        // "/Users/…/LocalModels/kroma-v0.2"), so match leniently in both
        // directions, including against the path's last component.
        let wantedLower = id.lowercased()
        func poolHasModel() -> Bool {
            poolModels.contains { entry in
                let modelLower = entry.model.lowercased()
                let lastComponent = (entry.model as NSString).lastPathComponent.lowercased()
                return modelLower == wantedLower
                    || modelLower.contains(wantedLower)
                    || wantedLower.contains(lastComponent)
            }
        }
        while Date() < deadline {
            try await Task.sleep(nanoseconds: 2_000_000_000)
            await refreshPool()
            if poolHasModel() { break }
        }
        // currentModel/currentModelFamily otherwise only update on the next
        // 3-second health poll — refreshing immediately closes the window
        // where a caller (preset apply, model selector) reads a stale active
        // model right after switching (a mismatched result could otherwise
        // reach the gallery tagged with the wrong model/LoRAs).
        await pollHealth()
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
        await pollHealth()
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

    // MARK: - Creation controls (pause / purge)

    /// Pause or resume ALL creation at the engine. The gate is engine-side and
    /// persistent, so it stops every initiator — schedulers on the server,
    /// chat tools, gallery buttons, direct API/MCP callers — and survives an
    /// engine restart. The current job finishes; nothing new starts.
    public func setCreationPaused(_ paused: Bool) async throws {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }
        let (status, responseData) = try await client.post(paused ? "/v1/queue/pause" : "/v1/queue/resume", body: Data())
        guard status == 200 else {
            let errorMessage = parseErrorMessage(from: responseData) ?? "Server returned status \(status)"
            throw EngineServiceError.serverError(status, errorMessage)
        }
        queuePaused = paused   // optimistic; the next health poll confirms
    }

    /// Purge the queue: drop every pending job AND interrupt the in-flight
    /// render. Interrupt-after-clear order matters — clearing first means the
    /// interrupted job cannot be followed by the next pending one.
    public func purgeQueue() async throws {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }
        let (clearStatus, clearData) = try await client.post("/v1/queue/clear", body: Data())
        guard clearStatus == 200 else {
            let errorMessage = parseErrorMessage(from: clearData) ?? "Server returned status \(clearStatus)"
            throw EngineServiceError.serverError(clearStatus, errorMessage)
        }
        // Best-effort: no job may be running, and an interrupt on an idle
        // engine is not an error worth surfacing.
        _ = try? await client.post("/v1/queue/interrupt", body: Data())
        queueCount = 0
    }

    // MARK: - LoRA Management

    /// Fetch the list of available LoRAs from the server's library.
    public func refreshLoras() async {
        guard let client = client, connectionState.isConnected else { return }

        do {
            let (status, data) = try await client.get("/v1/loras")
            guard status == 200 else {
                loraLoadError = "LoRA list failed: server returned \(status)"
                return
            }

            guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let loras = json["loras"] as? [[String: Any]],
                  let activeLoras = json["active_loras"] as? [String] else {
                loraLoadError = "LoRA list failed: unreadable response"
                return
            }
            loraLoadError = nil

            activeLoraIds = activeLoras

            availableLoras = loras.compactMap { dict -> LoRAInfo? in
                guard let id = dict["id"] as? String,
                      let filename = dict["filename"] as? String else { return nil }

                return LoRAInfo(
                    id: id,
                    filename: filename,
                    // The server sends an ARRAY (e.g. ["ltx"]); the String cast alone
                    // failed for every entry, so all 195 LoRAs fell into "Uncategorized".
                    modelCompatibility: (dict["model_compatibility"] as? String)
                        ?? (dict["model_compatibility"] as? [String])?.joined(separator: ", ")
                        ?? "unknown",
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
            // Non-fatal for rendering, but the user must not be shown a silently
            // empty catalog — record it so the UI can say so and the poll loop
            // can retry.
            loraLoadError = "LoRA list failed: \(error.localizedDescription)"
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
            // Server resolves LoRAs by filename (e.g. "Anneliese_Zbase3.safetensors"),
            // NOT by the slugified library id ("anneliese-zbase3") — sending the id
            // silently fails to resolve and renders with no LoRAs.
            var entry: [String: Any] = ["path": lora.filename, "scale": lora.scale]
            if let role = lora.role { entry["role"] = role }
            return entry
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
            queuePaused = health.isPaused ?? false
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

            if health.isRendering ?? false {
                await fetchLivePreview()
            } else if livePreviewImage != nil {
                livePreviewImage = nil
            }
        } catch is WarmServerClientError {
            connectionState = .disconnected
            queueInfo = nil
            livePreviewImage = nil
        } catch {
            connectionState = .error(error.localizedDescription)
            queueInfo = nil
            livePreviewImage = nil
        }
    }

    /// Fetch the current live-denoising preview frame (GH #216). Best-effort:
    /// a miss (204, no frame yet) or transient error just leaves the last
    /// frame in place rather than flickering the preview pane.
    private func fetchLivePreview() async {
        guard let client else { return }
        do {
            let (status, data) = try await client.get("/v1/generate/preview")
            guard status == 200, !data.isEmpty, let image = NSImage(data: data) else { return }
            livePreviewImage = image
        } catch {
            // Best-effort — ignore.
        }
    }

    // MARK: - Prompt Enhancement

    /// Send a prompt to the server's LLM enhancement endpoint.
    /// Returns the enhanced prompt string on success.
    /// Full enhance result with lineage (task #19): the optimized prompt plus
    /// the server-minted attempt id, outcome, and template provenance.
    public struct EnhanceOutcome: Sendable {
        public let prompt: String
        public let enhanced: Bool
        public let attemptId: String?
        public let outcome: String?
        public let templateId: String?
        public let templateHash: String?
    }

    public func enhancePromptDetailed(
        _ prompt: String, contentMode: ContentMode = .neutral, mediaKind: String = "video"
    ) async throws -> EnhanceOutcome {
        guard let client = client, connectionState.isConnected else {
            throw EngineServiceError.notConnected
        }
        var dict = Self.attachingContentMode(["prompt": prompt], mode: contentMode)
        dict["media_kind"] = mediaKind
        let bodyData = try JSONSerialization.data(withJSONObject: dict)
        let (status, responseData) = try await client.post("/v1/enhance", body: bodyData)
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: responseData) ?? "enhance failed")
        }
        guard let json = try JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let enhanced = json["prompt"] as? String else {
            throw EngineServiceError.generationFailed("Invalid enhance response")
        }
        return EnhanceOutcome(
            prompt: enhanced,
            enhanced: json["enhanced"] as? Bool ?? true,
            attemptId: json["optimization_attempt_id"] as? String,
            outcome: json["optimizer_outcome"] as? String,
            templateId: json["template_id"] as? String,
            templateHash: json["template_hash"] as? String)
    }

    public struct RenderTraceSummary: Codable, Sendable, Identifiable {
        public let renderId: String
        public let taskKind: String
        public let events: [String]
        public let submittedAt: Date?
        public let status: String
        public let prompt: String?
        public let outputPath: String?
        public let optimizationAttemptId: String?
        public let config: String?
        public let error: String?
        public let rating: String?
        public var id: String { renderId }
    }

    /// Newest-first render traces (task #19 Prompt Lab feed).
    public func fetchRenderTraces(limit: Int = 50) async -> [RenderTraceSummary] {
        guard let client = client, connectionState.isConnected else { return [] }
        do {
            let (status, data) = try await client.get("/v1/video/traces?limit=\(limit)")
            guard status == 200 else { return [] }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            decoder.dateDecodingStrategy = .iso8601
            struct Wrapper: Decodable { let traces: [RenderTraceSummary] }
            return try decoder.decode(Wrapper.self, from: data).traces
        } catch { return [] }
    }

    /// Promote a trace's prompt pair into the optimizer exemplar set.
    public func promoteTraceExemplar(renderId: String) async {
        guard let client = client, connectionState.isConnected else { return }
        _ = try? await client.post("/v1/video/traces/\(renderId)/promote", body: Data("{}".utf8))
    }

    /// Append a human verdict to a trace.
    public func rateRenderTrace(renderId: String, vote: String, axis: String = "overall") async {
        guard let client = client, connectionState.isConnected else { return }
        let body = try? JSONSerialization.data(withJSONObject: ["vote": vote, "axis": axis])
        _ = try? await client.post("/v1/video/traces/\(renderId)/rating", body: body ?? Data())
    }

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

    /// Persist trigger words for a LoRA library entry to the server.
    public func updateLoRATriggerwords(id: String, triggerwords: [String]) async throws {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let body = try JSONSerialization.data(withJSONObject: ["triggerwords": triggerwords])
        let (status, data) = try await client.post("/v1/loras/\(id)/update", body: body)
        guard status == 200 else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Update failed")
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
        /// LoRAs managed the same way as image generation's — via the LoRA
        /// library picker, not a raw path field. `loraPath`/`loraStrength`
        /// still work for a single ad-hoc LoRA.
        public var loras: [LoRASelection]
        public var outputPath: String
        /// Tier A tuning overrides (task #9 Phase 3) — snake_case keys as the
        /// server's LTX2VideoTuning expects; nil entries omitted.
        public var tuning: [String: Any]?
        /// Lineage reference from /v1/enhance (task #19).
        public var optimizationAttemptId: String?

        public init(
            prompt: String, initImagePath: String? = nil,
            width: Int = 704, height: Int = 448, frames: Int = 97,
            steps: Int = 8, seed: UInt64 = 42, strength: Float = 1.0,
            extendToSeconds: Float = 0, loraPath: String? = nil,
            loraStrength: Float = 1.0, loras: [LoRASelection] = [], outputPath: String,
            tuning: [String: Any]? = nil, optimizationAttemptId: String? = nil
        ) {
            self.prompt = prompt; self.initImagePath = initImagePath
            self.width = width; self.height = height; self.frames = frames
            self.steps = steps; self.seed = seed; self.strength = strength
            self.extendToSeconds = extendToSeconds
            self.loraPath = loraPath; self.loraStrength = loraStrength
            self.loras = loras
            self.outputPath = outputPath
            self.tuning = tuning
            self.optimizationAttemptId = optimizationAttemptId
        }
    }

    public struct VideoResult: Sendable {
        public let outputPath: String
        public let frameCount: Int
        public let durationSeconds: Double
        public let elapsedSeconds: Double
    }

    /// Live status of an async LOCAL video job — decoded from
    /// GET /v1/video/status/{id}. Mirrors the server's `VideoJobStatus`.
    public struct VideoJobStatus: Sendable {
        public let jobId: String
        public let status: String            // queued | processing | succeeded | failed
        public let outputPath: String?
        public let error: String?
        public let progressPercent: Int?
        public let frameCount: Int?
        public let videoDurationSeconds: Int?
        public let elapsedMs: Int?

        public var isTerminal: Bool { status == "succeeded" || status == "failed" }
    }

    /// Build the JSON body shared by the sync and async video paths.
    /// `forceLocal` pins the render to on-device LTX-2 (never paid cloud).
    private func videoRequestBody(_ request: VideoRequest, forceLocal: Bool) -> [String: Any] {
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
            "source": "desktop",
        ]
        if forceLocal { body["backend"] = "local" }
        if let initImagePath = request.initImagePath, !initImagePath.isEmpty {
            body["image_path"] = initImagePath
        }
        if let loraPath = request.loraPath, !loraPath.isEmpty {
            body["lora_path"] = loraPath
            body["lora_strength"] = request.loraStrength
        }
        if !request.loras.isEmpty {
            // Same convention as image LoRA swap: the server resolves by
            // filename, not the slugified library id.
            body["loras"] = request.loras.map {
                var entry: [String: Any] = ["path": $0.filename, "scale": $0.scale]
                if let role = $0.role { entry["role"] = role }
                return entry
            }
        }
        if let tuning = request.tuning, !tuning.isEmpty {
            body["tuning"] = tuning
        }
        if let attemptId = request.optimizationAttemptId {
            body["optimization_attempt_id"] = attemptId
        }
        return body
    }

    private func parseVideoJobStatus(_ data: Data) -> VideoJobStatus? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let jobId = json["job_id"] as? String,
              let status = json["status"] as? String
        else { return nil }
        return VideoJobStatus(
            jobId: jobId,
            status: status,
            outputPath: json["output_path"] as? String,
            error: json["error"] as? String,
            progressPercent: json["progress_percent"] as? Int,
            frameCount: json["frame_count"] as? Int,
            videoDurationSeconds: json["video_duration_seconds"] as? Int,
            elapsedMs: json["elapsed_ms"] as? Int
        )
    }

    /// Submit an async LOCAL video render. Returns immediately with the job id —
    /// the render runs on the server's GPU queue. Poll `pollVideoStatus` for
    /// completion. This is the path a long (multi-minute) render must take: the
    /// HTTP request returns in milliseconds, so nothing times out mid-render.
    public func submitVideoJob(_ request: VideoRequest) async throws -> String {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let bodyData = try JSONSerialization.data(withJSONObject: videoRequestBody(request, forceLocal: true))
        let (status, data) = try await client.post("/v1/video/generate/async", body: bodyData)
        guard status == 202, let job = parseVideoJobStatus(data) else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Video job submit failed (is the server started with --ltx2-weights?)")
        }
        return job.jobId
    }

    /// Poll GET /v1/video/status/{id} until the job reaches a terminal state.
    /// `onProgress` fires with the latest 0-100 percent on every poll (mirrors
    /// the `pollHealth()` loop). Throws on failure; returns the MP4 result on
    /// success. `pollInterval` seconds between polls.
    public func pollVideoStatus(
        jobId: String,
        pollInterval: Double = 2.0,
        onProgress: @MainActor (VideoJobStatus) -> Void = { _ in }
    ) async throws -> VideoResult {
        guard let client = client else { throw EngineServiceError.notConnected }
        while true {
            let (status, data) = try await client.get("/v1/video/status/\(jobId)")
            guard status == 200, let job = parseVideoJobStatus(data) else {
                throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "Video status poll failed")
            }
            onProgress(job)
            switch job.status {
            case "succeeded":
                guard let outputPath = job.outputPath else {
                    throw EngineServiceError.serverError(200, "Video job succeeded without an output path")
                }
                return VideoResult(
                    outputPath: outputPath,
                    frameCount: job.frameCount ?? 0,
                    durationSeconds: Double(job.videoDurationSeconds ?? 0),
                    elapsedSeconds: Double(job.elapsedMs ?? 0) / 1000.0
                )
            case "failed":
                throw EngineServiceError.serverError(200, job.error ?? "Video generation failed")
            default:
                try await Task.sleep(for: .seconds(pollInterval))
            }
        }
    }

    /// Generate a video locally via LTX-2 (SYNCHRONOUS — blocks for the whole
    /// render). Kept for backward compatibility; new callers should use
    /// `submitVideoJob` + `pollVideoStatus` so a long render doesn't freeze the
    /// UI or hit a request timeout.
    public func generateVideo(_ request: VideoRequest) async throws -> VideoResult {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let bodyData = try JSONSerialization.data(withJSONObject: videoRequestBody(request, forceLocal: false))
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

    /// One imported entry as returned by POST /v1/loras/import.
    public struct ImportedLoRA: Sendable {
        public let id: String
        public let filename: String
        public let modelCompatibility: [String]
        public let triggerwords: [String]
    }

    /// Import a LoRA file from a server-local path into the library under the
    /// given category folder (spec 2026-08-10). The route copies, rescans and
    /// returns the indexed entry; an already-known filename returns the
    /// existing entry rather than erroring.
    public func importLora(path: String, category: String) async throws -> ImportedLoRA {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let body = try JSONSerialization.data(withJSONObject: ["path": path, "category": category])
        let (status, data) = try await client.post("/v1/loras/import", body: body)
        guard status == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let id = json["id"] as? String,
              let filename = json["filename"] as? String
        else {
            throw EngineServiceError.serverError(status, parseErrorMessage(from: data) ?? "LoRA import failed")
        }
        return ImportedLoRA(
            id: id,
            filename: filename,
            modelCompatibility: (json["model_compatibility"] as? [String]) ?? [],
            triggerwords: (json["triggerwords"] as? [String]) ?? [])
    }

    // MARK: - Video winner actions (spec 2026-08-10)

    /// Replay a rendered clip's exact request at a higher resolution budget.
    /// Returns the async job id (poll /v1/video/status or watch the Queue tab).
    public func rerenderVideo(path: String, resolution: String = "720p") async throws -> String {
        try await submitWinnerAction(
            route: "/v1/video/rerender", body: ["path": path, "resolution": resolution])
    }

    /// Chain a fresh continuation from a clip's last frame at the 4s standard.
    public func extendVideo(path: String, seconds: Int = 4, prompt: String? = nil) async throws -> String {
        var body: [String: Any] = ["path": path, "seconds": seconds]
        if let prompt, !prompt.isEmpty { body["prompt"] = prompt }
        return try await submitWinnerAction(route: "/v1/video/extend", body: body)
    }

    private func submitWinnerAction(route: String, body: [String: Any]) async throws -> String {
        guard let client = client, connectionState.isConnected else { throw EngineServiceError.notConnected }
        let data = try JSONSerialization.data(withJSONObject: body)
        let (status, response) = try await client.post(route, body: data)
        guard status == 202 || status == 200,
              let json = try? JSONSerialization.jsonObject(with: response) as? [String: Any],
              let jobId = json["job_id"] as? String
        else {
            throw EngineServiceError.serverError(
                status, parseErrorMessage(from: response) ?? "Video action failed")
        }
        return jobId
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
        /// LTX-2 render phase from /v1/queue (baseDenoise, refineDenoise, ...).
        public var phase: String?
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
            list.phase = dict["phase"] as? String
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

    /// Effective LTX-2 video config with per-parameter provenance
    /// (GET /v1/video/config/effective, task #9 Phase 1).
    public func fetchEffectiveVideoConfig() async -> [EffectiveVideoParam] {
        guard let client = client, connectionState.isConnected else { return [] }
        do {
            let (status, data) = try await client.get("/v1/video/config/effective")
            guard status == 200 else { return [] }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            struct Wrapper: Decodable { let params: [EffectiveVideoParam] }
            return try decoder.decode(Wrapper.self, from: data).params
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
