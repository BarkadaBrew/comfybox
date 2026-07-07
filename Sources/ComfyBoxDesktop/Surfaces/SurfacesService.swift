// SurfacesService.swift — Track the surfaces the Coffeeshop suite is exposed on
//
// The desktop app is the hub; a "surface" is an external place that consumes
// ComfyBox — Krita (AI Diffusion → ComfyUI bridge), the Krita MCP plugin, MCP
// clients (Bree), CLI, etc. This service detects each surface's real state so
// the Applications tab can show what's wired up and what's broken.
//
// Detection is pure/synchronous filesystem + reachability checks; kept off the
// server so it works even when the daemon is down.

import Foundation

public struct Surface: Identifiable, Sendable, Equatable {
    public enum Health: String, Sendable { case ok, degraded, off, unknown }
    public var id: String
    public var name: String
    public var detail: String       // one-line status
    public var health: Health
    public var endpoint: String?    // what it connects to, if any
    public var hint: String?        // actionable next step when not ok
}

@Observable
@MainActor
public final class SurfacesService {
    public private(set) var surfaces: [Surface] = []
    public var serverPort: Int = 7870

    public init() {}

    public func refresh(bridgeReachable: Bool) {
        var out: [Surface] = []
        out.append(Self.comfyBridgeSurface(port: serverPort, reachable: bridgeReachable))
        out.append(Self.kritaSurface(port: serverPort))
        out.append(Self.kritaMCPSurface())
        out.append(Self.mcpSurface())
        surfaces = out
    }

    // MARK: - ComfyUI bridge (what external clients connect to)

    nonisolated static func comfyBridgeSurface(port: Int, reachable: Bool) -> Surface {
        Surface(
            id: "comfyui-bridge",
            name: "ComfyUI Bridge",
            detail: reachable ? "Serving ComfyUI-compatible API (Krita, external clients)"
                              : "Server not reachable — start the ComfyBox daemon",
            health: reachable ? .ok : .off,
            endpoint: "http://127.0.0.1:\(port)",
            hint: reachable ? nil : "Launch the ComfyBox server (launchctl / serve)."
        )
    }

    // MARK: - Krita AI Diffusion

    nonisolated static var kritaResourceDir: String {
        NSString(string: "~/Library/Application Support/krita").expandingTildeInPath
    }
    nonisolated static var kritaConfigPath: String {
        NSString(string: "~/Library/Preferences/kritarc").expandingTildeInPath
    }

    /// Inspect the Krita AI Diffusion install: present? enabled? pointed at us?
    /// duplicate .action (the known "docker disappears" bug)?
    nonisolated static func kritaSurface(port: Int) -> Surface {
        let fm = FileManager.default
        let pluginDir = (kritaResourceDir as NSString).appendingPathComponent("pykrita/ai_diffusion")
        guard fm.fileExists(atPath: pluginDir) else {
            return Surface(id: "krita", name: "Krita AI Diffusion", detail: "Plugin not installed",
                           health: .off, endpoint: nil,
                           hint: "Install the Krita AI Diffusion plugin, then point it at this Mac.")
        }
        let version = pluginVersion(at: (pluginDir as NSString).appendingPathComponent("__init__.py"))
        let enabled = configFlag("enable_ai_diffusion", in: kritaConfigPath)
        let settings = pluginSettings()
        let serverURL = settings?["server_url"] as? String
        let pointsAtUs = serverURL?.contains(":\(port)") ?? false

        var health: Surface.Health = .ok
        var bits: [String] = []
        var hint: String?
        if let version { bits.append("v\(version)") }
        if !enabled {
            health = .off; bits.append("disabled")
            hint = "Enable it in Krita → Settings → Python Plugin Manager → AI Image Diffusion."
        } else {
            bits.append("enabled")
            // Docker not showing? It's toggled per-window, not by the plugin state.
            hint = "If the panel is missing, toggle Krita → Settings → Dockers → AI Image Generation."
        }
        if let serverURL {
            bits.append(pointsAtUs ? "→ ComfyBox" : "→ \(serverURL)")
            if !pointsAtUs && health == .ok {
                health = .degraded
                hint = "Set the plugin's ComfyUI server to http://127.0.0.1:\(port)."
            }
        }
        let endpoint = serverURL.map { $0.contains("://") ? $0 : "http://\($0)" }
        return Surface(id: "krita", name: "Krita AI Diffusion",
                       detail: bits.joined(separator: " · "),
                       health: health, endpoint: endpoint, hint: hint)
    }

    /// The known Krita bug: ai_diffusion.action present in BOTH the plugin dir
    /// and the shared actions/ dir → actions register twice.
    nonisolated static func hasDuplicateActionFile() -> Bool {
        let fm = FileManager.default
        let stray = (kritaResourceDir as NSString).appendingPathComponent("actions/ai_diffusion.action")
        let own = (kritaResourceDir as NSString).appendingPathComponent("pykrita/ai_diffusion/ai_diffusion.action")
        return fm.fileExists(atPath: stray) && fm.fileExists(atPath: own)
    }

    nonisolated static func pluginSettings() -> [String: Any]? {
        let p = (kritaResourceDir as NSString).appendingPathComponent("ai_diffusion/settings.json")
        guard let data = FileManager.default.contents(atPath: p),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        return obj
    }

    nonisolated static func pluginVersion(at initPath: String) -> String? {
        guard let text = try? String(contentsOfFile: initPath, encoding: .utf8) else { return nil }
        // __version__ = "1.51.0"
        guard let r = text.range(of: #"__version__\s*=\s*"([^"]+)""#, options: .regularExpression) else { return nil }
        return text[r].split(separator: "\"").dropFirst().first.map(String.init)
    }

    // MARK: - Krita MCP

    nonisolated static func kritaMCPSurface() -> Surface {
        let desktop = (kritaResourceDir as NSString).appendingPathComponent("pykrita/kritamcp.desktop")
        let present = FileManager.default.fileExists(atPath: desktop)
        let enabled = configFlag("enable_kritamcp", in: kritaConfigPath)
        return Surface(
            id: "krita-mcp", name: "Krita MCP",
            detail: present ? (enabled ? "Installed · enabled" : "Installed · disabled") : "Not installed",
            health: present ? (enabled ? .ok : .off) : .off, endpoint: nil,
            hint: (present && !enabled) ? "Enable in Krita → Python Plugin Manager → Krita MCP." : nil)
    }

    // MARK: - MCP (Bree / assistants)

    nonisolated static func mcpSurface() -> Surface {
        // Bree's daemon exposes mcp_comfybox tools; presence of the config is a
        // reasonable proxy that the MCP surface is wired.
        let breeConfig = NSString(string: "~/.bree/config.json").expandingTildeInPath
        let present = FileManager.default.fileExists(atPath: breeConfig)
        return Surface(
            id: "mcp", name: "MCP (Bree / assistants)",
            detail: present ? "ComfyBox tools exposed to Bree (LoRA, generate)" : "No Bree config found",
            health: present ? .ok : .unknown, endpoint: nil, hint: nil)
    }

    // MARK: - Helpers

    /// Read a `key=true` flag from a KConfig-style ini file.
    nonisolated static func configFlag(_ key: String, in path: String) -> Bool {
        guard let text = try? String(contentsOfFile: path, encoding: .utf8) else { return false }
        for line in text.split(separator: "\n") {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.hasPrefix("\(key)=") { return t.hasSuffix("true") }
        }
        return false
    }
}
