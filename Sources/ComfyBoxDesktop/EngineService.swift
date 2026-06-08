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

    public init(
        prompt: String = "",
        width: Int = 1024,
        height: Int = 1024,
        steps: Int = 9,
        guidance: Float = 3.5,
        seed: UInt64 = 0
    ) {
        self.prompt = prompt
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.seed = seed
    }
}

/// Decoded generation response from the server.
private struct ServerGenerateResponse: Decodable {
    let success: Bool
    let outputPath: String
    let durationMs: Int
}

/// Decoded health response from the server.
private struct ServerHealthResponse: Decodable {
    let status: String
    let model: String?
    let queuePending: Int?
    let queueActive: Int?

    enum CodingKeys: String, CodingKey {
        case status
        case model
        case queuePending = "queue_pending"
        case queueActive = "queue_active"
    }
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

@Observable
public final class EngineService {
    // MARK: - Published State

    public var connectionState: ServerConnectionState = .disconnected
    public var currentModel: String?
    public var queueCount: Int = 0
    public var isGenerating: Bool = false
    public var lastGeneratedImagePath: String?
    public var lastError: String?
    public var lastDurationMs: Int?

    // MARK: - Configuration

    public var serverHost: String = "127.0.0.1"
    public var serverPort: UInt16 = 7862
    public var outputDirectory: String = NSString(string: "~/Pictures/ComfyBox").expandingTildeInPath

    // MARK: - Private

    private var client: WarmServerClient?
    private var healthPollTask: Task<Void, Never>?

    public init() {}

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
        queueCount = 0
    }

    // MARK: - Generation

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

        defer { isGenerating = false }

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

        let decoder = JSONDecoder()
        let response = try decoder.decode(ServerGenerateResponse.self, from: responseData)

        guard response.success else {
            let msg = "Generation reported failure"
            lastError = msg
            throw EngineServiceError.generationFailed(msg)
        }

        lastGeneratedImagePath = response.outputPath
        lastDurationMs = response.durationMs
        return response.outputPath
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
            queueCount = (health.queuePending ?? 0) + (health.queueActive ?? 0)
        } catch is WarmServerClientError {
            connectionState = .disconnected
        } catch {
            connectionState = .error(error.localizedDescription)
        }
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
