// AgentService.swift — Chat assistant for image creation (Dan's v1.3 via LM Studio)
//
// A conversational helper backed by the configured prompt-optimization
// provider (Dan's dans-pe-v1.3.0 heresy model on LM Studio). It holds the
// conversation, calls the OpenAI-compatible /chat/completions endpoint, and
// lets the user push a suggested prompt into Generate. Request assembly and
// response parsing are pure so they're testable without a network.

import Foundation

/// A structured generation-parameter change the assistant can emit (as a
/// fenced ```json block) so it can drive the Generate view's fields directly.
/// Every field is optional — only the ones present are applied.
public struct AgentAction: Equatable, Sendable {
    public var prompt: String?
    public var negativePrompt: String?
    public var steps: Int?
    public var guidance: Double?
    public var width: Int?
    public var height: Int?
    public var seed: Int?
    public var loras: [String]?
    /// When true, the Generate view kicks off a render after applying.
    public var generate: Bool?

    /// True when at least one field is set.
    public var hasChanges: Bool {
        prompt != nil || negativePrompt != nil || steps != nil || guidance != nil
            || width != nil || height != nil || seed != nil || loras != nil || generate == true
    }

    /// A short human summary of what will change, for the UI.
    public var summary: String {
        var parts: [String] = []
        if prompt != nil { parts.append("prompt") }
        if negativePrompt != nil { parts.append("negative") }
        if let s = steps { parts.append("steps \(s)") }
        if let g = guidance { parts.append(String(format: "guidance %.1f", g)) }
        if let w = width, let h = height { parts.append("\(w)×\(h)") }
        if let seed { parts.append("seed \(seed)") }
        if let loras, !loras.isEmpty { parts.append("\(loras.count) LoRA\(loras.count == 1 ? "" : "s")") }
        if generate == true { parts.append("generate") }
        return parts.joined(separator: " · ")
    }
}

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
    /// The parameter action from the most recent assistant reply, if any.
    public var lastAction: AgentAction?

    private let engine: EngineService
    private let session: URLSession

    /// Steers the model toward being a ComfyBox image-creation assistant that
    /// can also drive the Generate view's controls via a JSON action block.
    public nonisolated static let systemPrompt = """
    You are the ComfyBox image assistant, helping the user craft images with a \
    local Z-Image / Flux generation stack on macOS. Help refine prompts, suggest \
    camera framing, lighting, lenses, composition, negative prompts, and sensible \
    step/guidance settings.

    You can also SET the generation controls. When the user asks you to configure, \
    change, or apply settings (or asks you to generate), include a fenced json code \
    block containing only the keys you want to change, from: prompt, negative_prompt, \
    steps (int), guidance (number), width (int), height (int), seed (int), loras \
    (array of "filename" or "filename=scale"), generate (bool, true to start a \
    render). Example:
    ```json
    {"prompt": "kira at golden hour, 85mm", "steps": 9, "guidance": 3.5, "width": 1024, "height": 1536}
    ```
    Keep prose brief and put the json block last. If you only suggest a prompt without \
    other settings, you may instead put it on a line prefixed exactly with "PROMPT:".
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
            lastAction = Self.parseAction(from: reply)
        } catch {
            lastError = error.localizedDescription
        }
    }

    /// Extract a generation-parameter action from a reply's ```json block.
    /// Returns nil when there's no block or it carries no recognized keys.
    public nonisolated static func parseAction(from text: String) -> AgentAction? {
        guard let jsonString = extractJSONBlock(from: text),
              let data = jsonString.data(using: .utf8),
              let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }

        func int(_ key: String) -> Int? {
            if let i = object[key] as? Int { return i }
            if let d = object[key] as? Double { return Int(d) }
            return nil
        }
        func double(_ key: String) -> Double? {
            if let d = object[key] as? Double { return d }
            if let i = object[key] as? Int { return Double(i) }
            return nil
        }

        var action = AgentAction()
        action.prompt = (object["prompt"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        action.negativePrompt = (object["negative_prompt"] as? String)
            ?? (object["negativePrompt"] as? String)
        action.steps = int("steps")
        action.guidance = double("guidance")
        action.width = int("width")
        action.height = int("height")
        action.seed = int("seed")
        action.loras = object["loras"] as? [String]
        action.generate = object["generate"] as? Bool
        // Drop empty-string prompt fields.
        if action.prompt?.isEmpty == true { action.prompt = nil }
        if action.negativePrompt?.isEmpty == true { action.negativePrompt = nil }

        return action.hasChanges ? action : nil
    }

    /// The contents of the first ```json fenced block (or the first bare {...}).
    nonisolated static func extractJSONBlock(from text: String) -> String? {
        // Prefer a ```json ... ``` fence.
        if let fenceStart = text.range(of: "```json", options: .caseInsensitive) {
            let afterFence = text[fenceStart.upperBound...]
            if let fenceEnd = afterFence.range(of: "```") {
                return String(afterFence[..<fenceEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        // Fall back to a balanced { … } span.
        guard let open = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var index = open
        while index < text.endIndex {
            let ch = text[index]
            if ch == "{" { depth += 1 }
            else if ch == "}" {
                depth -= 1
                if depth == 0 {
                    return String(text[open...index])
                }
            }
            index = text.index(after: index)
        }
        return nil
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
