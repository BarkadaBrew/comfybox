// KiraClient.swift — thin client for the kira-daemon control API (comfybox#240 D1)
//
// The Kira tab is a THIN CLIENT of a headless local service: every lever is an
// HTTP/WS call to the kira-daemon; no orchestration lives in the app. The tab
// targets a (host, port, token) binding — the Linux daemon during the interim,
// 127.0.0.1 after the Kira Muse Mac migration — same UI, same contract, only
// the binding value differs (dashboard FDD §0/§9).
//
// D1 scope: binding + health strip. The dashboard/state/scheduler/compute
// cards (D2+) bind to the Workstream A service contract (/v1/kira/*) once it
// lands in the daemon.

import Foundation

/// Host binding for the Kira tab — host-agnostic by construction.
public struct KiraHostBinding: Codable, Equatable, Sendable {
    public var host: String
    public var port: Int

    /// Default binding is loopback: the kira-daemon's api listener is
    /// 127.0.0.1-only on the server (reach it via `ssh -N -L 3787:127.0.0.1:3787`
    /// during the interim), and after the Mac migration it IS loopback — so the
    /// default needs no change when Kira moves home.
    public init(host: String = "127.0.0.1", port: Int = 3787) {
        self.host = host
        self.port = port
    }

    public var baseURL: URL? {
        URL(string: "http://\(host):\(port)")
    }

    /// Whether this binding points at the local Mac (post-migration) — service
    /// management (F2) only applies then; remote daemons are systemd-managed.
    public var isLocal: Bool {
        host == "127.0.0.1" || host == "localhost"
    }
}

/// Parsed `GET /health` snapshot for the health strip (F1). Field names match
/// the daemon's subsystem-health payload; parsing is tolerant — an absent
/// field renders as unknown rather than failing the whole strip.
public struct KiraHealthSnapshot: Equatable, Sendable {
    public var status: String
    public var name: String
    public var isRunning: Bool
    public var isPaused: Bool
    public var energy: Double?
    public var autonomousRenderEnabled: Bool?
    public var toolCount: Int?
    public var fetchedAt: Date

    public static func parse(_ data: Data, fetchedAt: Date = Date()) -> KiraHealthSnapshot? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        let renderControls = json["renderControls"] as? [String: Any]
        return KiraHealthSnapshot(
            status: json["status"] as? String ?? "unknown",
            name: json["name"] as? String ?? "kira",
            isRunning: json["isRunning"] as? Bool ?? false,
            isPaused: json["isPaused"] as? Bool ?? false,
            energy: (json["energy"] as? NSNumber)?.doubleValue,
            autonomousRenderEnabled: renderControls?["autonomousRenderEnabled"] as? Bool,
            toolCount: (json["tools"] as? [Any])?.count,
            fetchedAt: fetchedAt)
    }
}

/// Consolidated `GET /v1/kira/state` read model (A1). Tolerant parsing —
/// absent slices render as absent, never as fabricated defaults.
public struct KiraStateSnapshot: Equatable, Sendable {
    public var mood: String?
    public var energy: String?
    public var arcPhase: String?
    /// closeness / warmth / desire / playfulness, 0–100.
    public var scores: [(name: String, value: Double)]
    public var dynamicLine: String?
    public var campaignBeat: String?
    public var agenda: [String]
    public var recentLines: [String]
    /// FDD §3.1 world slice — not served yet (A3 gap); nil until it lands.
    public var worldPresent: Bool
    public var fetchedAt: Date

    public static func == (lhs: KiraStateSnapshot, rhs: KiraStateSnapshot) -> Bool {
        lhs.fetchedAt == rhs.fetchedAt
    }

    public static func parse(_ data: Data, fetchedAt: Date = Date()) -> KiraStateSnapshot? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let state = json["state"] as? [String: Any] else { return nil }
        let now = state["now"] as? [String: Any]
        let relationship = state["relationship"] as? [String: Any]
        let active = state["active"] as? [String: Any]
        let recent = state["recent"] as? [String: Any]
        let scoreDict = relationship?["scores"] as? [String: Any] ?? [:]
        let scoreOrder = ["closeness", "warmth", "desire", "playfulness"]
        let scores: [(String, Double)] = scoreOrder.compactMap { key in
            (scoreDict[key] as? NSNumber).map { (key, $0.doubleValue) }
        }
        return KiraStateSnapshot(
            mood: now?["mood"] as? String,
            energy: now?["energy"] as? String,
            arcPhase: now?["arcPhase"] as? String,
            scores: scores,
            dynamicLine: relationship?["dynamic"] as? String,
            campaignBeat: active?["campaignBeat"] as? String,
            agenda: (active?["agenda"] as? [String]) ?? [],
            recentLines: (recent?["lines"] as? [String]) ?? [],
            worldPresent: state["world"] != nil,
            fetchedAt: fetchedAt)
    }
}

/// `GET /v1/kira/content-scheduler/status` (A2).
public struct KiraSchedulerStatus: Equatable, Sendable {
    public var paused: Bool
    public var enabled: Bool
    public var intervalMinutes: Int?
    public var imageCount: Int?
    public var videoCount: Int?
    public var videoMode: String?

    public static func parse(_ data: Data) -> KiraSchedulerStatus? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let paused = json["paused"] as? Bool else { return nil }
        let config = json["config"] as? [String: Any]
        return KiraSchedulerStatus(
            paused: paused,
            enabled: config?["enabled"] as? Bool ?? false,
            intervalMinutes: (config?["intervalMinutes"] as? NSNumber)?.intValue,
            imageCount: (config?["imageCount"] as? NSNumber)?.intValue,
            videoCount: (config?["videoCount"] as? NSNumber)?.intValue,
            videoMode: config?["videoMode"] as? String)
    }
}

/// `GET /v1/kira/media/recent` item (A5).
public struct KiraMediaItem: Identifiable, Equatable, Sendable {
    public var id: String { path }
    public var kind: String
    public var path: String
    public var mtimeMs: Double
}

/// Suggestion-box entry (`/v1/kira/suggestions`). Kinds: image/video seed one
/// render (consumed FIFO), session themes one cycle (consumed), arc is sticky
/// context until removed.
public struct KiraSuggestion: Identifiable, Equatable, Sendable {
    public var id: String
    public var kind: String
    public var text: String
    public var status: String
    public var createdAt: String

    public static func parse(_ item: [String: Any]) -> KiraSuggestion? {
        guard let id = item["id"] as? String,
              let kind = item["kind"] as? String,
              let text = item["text"] as? String else { return nil }
        return KiraSuggestion(
            id: id, kind: kind, text: text,
            status: item["status"] as? String ?? "pending",
            createdAt: item["createdAt"] as? String ?? "")
    }
}

@Observable
@MainActor
public final class KiraClient {
    private static let bindingDefaultsKey = "kira.hostBinding"
    /// Fast poll cadence while the tab is visible (FDD §2: the health strip is
    /// the one surface that always polls fast — a stale strip is worse than a
    /// stale mood chip).
    public static let healthPollSeconds: TimeInterval = 5

    public var binding: KiraHostBinding {
        didSet {
            guard binding != oldValue else { return }
            persistBinding()
            health = nil
            lastError = nil
            if pollTask != nil { startPolling() }  // re-point the live poll
        }
    }

    public private(set) var health: KiraHealthSnapshot?
    public private(set) var lastError: String?
    public private(set) var isPolling = false

    // Dashboard read models (D2–D4), slow-polled while the tab is visible.
    public private(set) var state: KiraStateSnapshot?
    public private(set) var stateError: String?
    public private(set) var scheduler: KiraSchedulerStatus?
    public private(set) var contentMode: String?
    public private(set) var allowedModes: [String] = []
    public private(set) var recentMedia: [KiraMediaItem] = []
    public private(set) var suggestions: [KiraSuggestion] = []
    public private(set) var suggestionKinds: [String] = ["image", "video", "arc", "session"]
    /// Raw compute payload — the SSH-bridge proxy is slow/flaky (FDD R-6), so
    /// this is fetched on demand and rendered tolerantly.
    public private(set) var computeSummary: String?
    public private(set) var computeError: String?
    public private(set) var computeLoading = false
    /// In-flight control action (pause/resume/mode) — disables the controls.
    public private(set) var actionInFlight = false
    public private(set) var actionError: String?

    private var pollTask: Task<Void, Never>?
    private var dashboardTask: Task<Void, Never>?

    public init() {
        if let data = UserDefaults.standard.data(forKey: Self.bindingDefaultsKey),
           let stored = try? JSONDecoder().decode(KiraHostBinding.self, from: data) {
            self.binding = stored
        } else {
            self.binding = KiraHostBinding()
        }
    }

    /// Daemon API token, from the Keychain (never plaintext on disk).
    public var token: String {
        get { AppSecrets.value(.kiraDaemon) ?? "" }
        set { AppSecrets.set(.kiraDaemon, newValue.isEmpty ? nil : newValue) }
    }

    public var isReachable: Bool { health != nil && lastError == nil }

    private func persistBinding() {
        if let data = try? JSONEncoder().encode(binding) {
            UserDefaults.standard.set(data, forKey: Self.bindingDefaultsKey)
        }
    }

    // MARK: - Health polling (F1)

    public func startPolling() {
        stopPolling()
        isPolling = true
        pollTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                await self.refreshHealth()
                try? await Task.sleep(nanoseconds: UInt64(Self.healthPollSeconds * 1_000_000_000))
            }
        }
        dashboardTask = Task { [weak self] in
            while let self, !Task.isCancelled {
                await self.refreshDashboard()
                try? await Task.sleep(nanoseconds: 20_000_000_000)
            }
        }
    }

    public func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
        dashboardTask?.cancel()
        dashboardTask = nil
        isPolling = false
    }

    public func refreshHealth() async {
        guard let base = binding.baseURL else {
            lastError = "Invalid host binding"
            return
        }
        var request = URLRequest(url: base.appendingPathComponent("health"))
        request.timeoutInterval = 4
        let bearer = token
        if !bearer.isEmpty {
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                lastError = "No HTTP response"
                return
            }
            guard http.statusCode == 200 else {
                lastError = http.statusCode == 401
                    ? "Unauthorized — set the daemon API token"
                    : "HTTP \(http.statusCode)"
                return
            }
            guard let snapshot = KiraHealthSnapshot.parse(data) else {
                lastError = "Unparseable /health payload"
                return
            }
            health = snapshot
            lastError = nil
        } catch {
            // The strip must degrade to an explicit "core unreachable", never a
            // confident stale value (FDD §2 truthfulness guard).
            lastError = error.localizedDescription
        }
    }

    // MARK: - HTTP plumbing

    private func request(_ path: String, method: String = "GET",
                         body: [String: Any]? = nil, timeout: TimeInterval = 8) -> URLRequest? {
        guard let base = binding.baseURL else { return nil }
        var request = URLRequest(url: base.appendingPathComponent(path))
        request.httpMethod = method
        request.timeoutInterval = timeout
        let bearer = token
        if !bearer.isEmpty {
            request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try? JSONSerialization.data(withJSONObject: body)
        }
        return request
    }

    private func fetch(_ path: String, timeout: TimeInterval = 8) async -> Data? {
        guard let request = request(path, timeout: timeout) else { return nil }
        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return data
    }

    // MARK: - Dashboard (D2–D4, Workstream A bindings)

    public func refreshDashboard() async {
        async let stateData = fetch("v1/kira/state")
        async let schedulerData = fetch("v1/kira/content-scheduler/status")
        async let modeData = fetch("v1/kira/content-mode")
        async let mediaData = fetch("v1/kira/media/recent")
        async let suggestionsData = fetch("v1/kira/suggestions")

        if let data = await stateData, let parsed = KiraStateSnapshot.parse(data) {
            state = parsed
            stateError = nil
        } else if state == nil {
            stateError = "GET /v1/kira/state unavailable"
        }
        if let data = await schedulerData {
            scheduler = KiraSchedulerStatus.parse(data)
        }
        if let data = await modeData,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            contentMode = json["mode"] as? String
            allowedModes = json["allowed"] as? [String] ?? allowedModes
        }
        if let data = await mediaData,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let items = json["items"] as? [[String: Any]] {
            recentMedia = items.prefix(12).compactMap { item in
                guard let kind = item["kind"] as? String,
                      let path = item["path"] as? String else { return nil }
                return KiraMediaItem(
                    kind: kind, path: path,
                    mtimeMs: (item["mtimeMs"] as? NSNumber)?.doubleValue ?? 0)
            }
        }
        if let data = await suggestionsData,
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let items = json["suggestions"] as? [[String: Any]] {
            suggestions = items.compactMap(KiraSuggestion.parse).reversed()
            if let kinds = json["kinds"] as? [String], !kinds.isEmpty {
                suggestionKinds = kinds
            }
        }
    }

    /// Drop an idea in the box — Kira picks it up on her next content cycle.
    public func addSuggestion(kind: String, text: String) async {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        await perform("v1/kira/suggestions", method: "POST", body: ["kind": kind, "text": trimmed])
    }

    public func deleteSuggestion(_ id: String) async {
        await perform("v1/kira/suggestions/\(id)", method: "DELETE")
    }

    /// URL for a media thumbnail via the daemon's traversal-guarded file serve.
    public func mediaFileURL(for item: KiraMediaItem) -> URL? {
        guard let base = binding.baseURL,
              var components = URLComponents(url: base.appendingPathComponent("v1/kira/media/file"),
                                             resolvingAgainstBaseURL: false) else { return nil }
        components.queryItems = [URLQueryItem(name: "path", value: item.path)]
        return components.url
    }

    /// Fetch media bytes with the auth header (AsyncImage can't send Bearer).
    public func loadMediaData(for item: KiraMediaItem) async -> Data? {
        guard let url = mediaFileURL(for: item) else { return nil }
        var urlRequest = URLRequest(url: url)
        urlRequest.timeoutInterval = 10
        let bearer = token
        if !bearer.isEmpty {
            urlRequest.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        }
        guard let (data, response) = try? await URLSession.shared.data(for: urlRequest),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return data
    }

    /// GET /v1/kira/compute — proxied over the SSH/MCP bridge, so it can take
    /// tens of seconds or 502 (FDD R-6). On-demand only.
    public func refreshCompute() async {
        computeLoading = true
        defer { computeLoading = false }
        computeError = nil
        guard let request = request("v1/kira/compute", timeout: 45) else { return }
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard status == 200,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let compute = json["compute"] else {
                let detail = String(data: data.prefix(200), encoding: .utf8) ?? ""
                computeError = "HTTP \(status) \(detail)"
                return
            }
            // Opaque-tolerant render: pretty-print whatever the proxy returned.
            if let pretty = try? JSONSerialization.data(withJSONObject: compute, options: [.prettyPrinted, .sortedKeys]),
               let text = String(data: pretty, encoding: .utf8) {
                computeSummary = String(text.prefix(1500))
            } else {
                computeSummary = "\(compute)"
            }
        } catch {
            computeError = error.localizedDescription
        }
    }

    // MARK: - Controls (writes)

    private func perform(_ path: String, method: String, body: [String: Any]? = nil) async {
        actionInFlight = true
        actionError = nil
        defer { actionInFlight = false }
        guard let request = request(path, method: method, body: body, timeout: 10) else { return }
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard (200..<300).contains(status) else {
                let detail = String(data: data.prefix(200), encoding: .utf8) ?? ""
                actionError = "\(method) \(path): HTTP \(status) \(detail)"
                return
            }
        } catch {
            actionError = error.localizedDescription
        }
        await refreshDashboard()
    }

    /// B1: toggles the Kira CONTENT scheduler sentinel — deliberately not the
    /// generic /v1/scheduler (the FDD §3.2 conflation trap).
    public func setSchedulerPaused(_ paused: Bool) async {
        await perform("v1/kira/content-scheduler/\(paused ? "pause" : "resume")", method: "POST")
    }

    /// B5: set the default content mode (allowlisted server-side).
    public func setContentMode(_ mode: String) async {
        await perform("v1/kira/content-mode", method: "PUT", body: ["mode": mode])
    }
}
