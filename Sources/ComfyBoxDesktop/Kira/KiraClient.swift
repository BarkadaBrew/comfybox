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

    private var pollTask: Task<Void, Never>?

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
    }

    public func stopPolling() {
        pollTask?.cancel()
        pollTask = nil
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
}
