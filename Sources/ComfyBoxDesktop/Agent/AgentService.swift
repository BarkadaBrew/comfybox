// AgentService.swift — Chat assistant for image creation (Dan's v1.3 via LM Studio)
//
// A conversational helper backed by the configured prompt-optimization
// provider (Dan's dans-pe-v1.3.0 heresy model on LM Studio). It holds the
// conversation, calls the OpenAI-compatible /chat/completions endpoint, and
// lets the user push a suggested prompt into Generate. Request assembly and
// response parsing are pure so they're testable without a network.

import Foundation

public struct AgentMessage: Identifiable, Equatable, Sendable {
    public enum Role: String, Sendable { case system, user, assistant }
    public let id: String
    public let role: Role
    public var text: String

    public init(id: String = UUID().uuidString, role: Role, text: String) {
        self.id = id
        self.role = role
        self.text = text
    }
}

@Observable
@MainActor
public final class AgentService {
    public var messages: [AgentMessage] = []
    public var isThinking = false
    public var lastError: String?
    /// Resolved from the server config; nil until configured/available.
    public var modelName: String?

    private let engine: EngineService
    private let session: URLSession

    /// Steers the model toward being a ComfyBox image-creation assistant.
    public nonisolated static let systemPrompt = """
    You are the ComfyBox image assistant, helping the user craft images with a \
    local Z-Image / Flux generation stack on macOS. Help refine prompts, suggest \
    camera framing, lighting, lenses, composition, negative prompts, and sensible \
    step/guidance settings. When you propose a final image prompt, put it on its \
    own line prefixed exactly with "PROMPT:" so the app can offer to use it. Be \
    concise and concrete.
    """

    public init(engine: EngineService, session: URLSession = .shared) {
        self.engine = engine
        self.session = session
    }

    // MARK: - Conversation

    public func reset() {
        messages.removeAll()
        lastError = nil
    }

    /// Send a user turn and append the assistant's reply.
    public func send(_ text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isThinking else { return }
        messages.append(AgentMessage(role: .user, text: trimmed))
        isThinking = true
        lastError = nil
        defer { isThinking = false }

        do {
            let endpoint = try await resolveEndpoint()
            modelName = endpoint.model
            let reply = try await complete(endpoint: endpoint)
            messages.append(AgentMessage(role: .assistant, text: reply))
        } catch {
            lastError = error.localizedDescription
        }
    }

    /// Extract the suggested prompt from an assistant message, if it marked one
    /// with a "PROMPT:" line; else the whole message trimmed.
    public nonisolated static func suggestedPrompt(from text: String) -> String? {
        for line in text.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.lowercased().hasPrefix("prompt:") {
                let value = trimmed.dropFirst("prompt:".count).trimmingCharacters(in: .whitespaces)
                if !value.isEmpty { return value }
            }
        }
        return nil
    }

    // MARK: - Provider

    private struct ResolvedEndpoint {
        let baseURL: URL
        let model: String
        let apiKey: String?
    }

    private func resolveEndpoint() async throws -> ResolvedEndpoint {
        let config = try await engine.fetchServerConfig()
        guard let provider = config.providers.promptOptimization else {
            throw AgentError.noProvider
        }
        // The stored baseUrl is an OpenAI-style root that usually ends in /v1.
        var base = provider.baseUrl
        while base.hasSuffix("/") { base.removeLast() }
        if !base.hasSuffix("/v1") { base += "/v1" }
        guard let url = URL(string: base + "/chat/completions") else {
            throw AgentError.badURL
        }
        return ResolvedEndpoint(baseURL: url, model: provider.model, apiKey: provider.apiKey)
    }

    /// The chat payload sent to the provider (system + full history).
    nonisolated static func requestBody(model: String, messages: [AgentMessage]) -> [String: Any] {
        var wire: [[String: String]] = [["role": "system", "content": systemPrompt]]
        for message in messages where message.role != .system {
            wire.append(["role": message.role.rawValue, "content": message.text])
        }
        return [
            "model": model,
            "messages": wire,
            "temperature": 0.8,
            "max_tokens": 800,
            "stream": false,
        ]
    }

    /// Pull the assistant text out of an OpenAI-compatible chat response.
    nonisolated static func parseReply(_ data: Data) -> String? {
        guard let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String
        else { return nil }
        return content.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func complete(endpoint: ResolvedEndpoint) async throws -> String {
        var request = URLRequest(url: endpoint.baseURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key = endpoint.apiKey, !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        request.httpBody = try JSONSerialization.data(
            withJSONObject: Self.requestBody(model: endpoint.model, messages: messages))

        let (data, response) = try await session.data(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw AgentError.requestFailed((response as? HTTPURLResponse)?.statusCode ?? -1)
        }
        guard let reply = Self.parseReply(data) else { throw AgentError.emptyReply }
        return reply
    }
}

public enum AgentError: LocalizedError {
    case noProvider
    case badURL
    case requestFailed(Int)
    case emptyReply

    public var errorDescription: String? {
        switch self {
        case .noProvider:
            return "No assistant model configured (Settings → AI Providers → Prompt Optimization)."
        case .badURL:
            return "The assistant endpoint URL is invalid."
        case .requestFailed(let code):
            return "Assistant request failed (HTTP \(code)). Is LM Studio running with the model loaded?"
        case .emptyReply:
            return "The assistant returned an empty response."
        }
    }
}
