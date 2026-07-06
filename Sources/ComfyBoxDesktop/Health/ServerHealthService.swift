// ServerHealthService.swift — Fetch get_server_health from the coffeeshop daemon
//
// Calls the MCP bridge's tools/execute endpoint. The result envelope varies by
// server, so we tolerantly dig out the health payload. Until the server ships
// the tool, fetch() surfaces a clear unavailable state rather than an error
// wall.

import Foundation

@Observable
@MainActor
public final class ServerHealthService {
    /// MCP bridge tools/execute endpoint (loopback on the server; reachable from
    /// the Mac via the documented SSH tunnel / front-desk — see issue #936).
    public var endpoint: String

    public private(set) var health: ServerHealth?
    public private(set) var lastError: String?
    public private(set) var lastFetched: Date?
    public private(set) var isLoading = false

    private var polling = false

    public init(endpoint: String = ServerHealthService.defaultEndpoint) {
        self.endpoint = endpoint
    }

    public nonisolated static let defaultEndpoint = "http://10.0.100.232:3777/v1/tools/execute"

    /// Pull periodically (the desktop pulls from the server; no server push).
    /// Idempotent — a second call is a no-op.
    public func startPolling(every seconds: UInt64 = 60) {
        guard !polling else { return }
        polling = true
        Task { [weak self] in
            while let self, self.polling {
                await self.fetch()
                try? await Task.sleep(nanoseconds: seconds * 1_000_000_000)
            }
        }
    }

    public func stopPolling() { polling = false }

    /// Dig the `get_server_health` payload out of a tools/execute envelope.
    /// Handles a bare object, `{result: …}`, or MCP `{content:[{text: "json"}]}`.
    public nonisolated static func decodeHealth(from data: Data) -> ServerHealth? {
        let decoder = JSONDecoder()
        if let direct = try? decoder.decode(ServerHealth.self, from: data),
           direct != ServerHealth() { return direct }
        guard let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        // {result: {...}}
        if let result = obj["result"] as? [String: Any],
           let sub = try? JSONSerialization.data(withJSONObject: result),
           let h = try? decoder.decode(ServerHealth.self, from: sub) { return h }
        // MCP content: [{type:text, text:"<json>"}]
        if let content = obj["content"] as? [[String: Any]] {
            for block in content {
                if let text = block["text"] as? String,
                   let td = text.data(using: .utf8),
                   let h = try? decoder.decode(ServerHealth.self, from: td) { return h }
            }
        }
        return nil
    }

    public func fetch() async {
        guard let url = URL(string: endpoint) else { lastError = "Bad endpoint"; return }
        isLoading = true
        defer { isLoading = false }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 8
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONSerialization.data(withJSONObject: [
            "tool": "get_server_health", "arguments": [:],
        ])
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard code == 200 else {
                lastError = code == 404
                    ? "get_server_health not available yet (pending coffeeshop-server tool)."
                    : "Server returned HTTP \(code)."
                health = nil
                return
            }
            if let h = Self.decodeHealth(from: data) {
                health = h; lastError = nil; lastFetched = Date()
            } else {
                lastError = "Couldn't parse server health response."
            }
        } catch {
            lastError = "Server health unreachable: \(error.localizedDescription)"
            health = nil
        }
    }
}

