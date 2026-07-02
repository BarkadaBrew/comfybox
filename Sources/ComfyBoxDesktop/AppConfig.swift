// AppConfig.swift — Lightweight JSON config loader
//
// Reads ~/.comfybox/config.json at launch for server connection
// and output directory overrides. Falls back to compiled defaults
// when the file is missing or malformed. This is the "bootstrap"
// config — DesktopSettings (desktop-config.json) layer on top.

import Foundation

struct AppConfig: Codable {
    var serverHost: String = "127.0.0.1"
    var serverPort: UInt16 = 7870
    var outputDirectory: String = "~/Pictures/ComfyBox"

    static let configPath = NSString(string: "~/.comfybox/config.json").expandingTildeInPath

    // Shared with the server's ComfyBoxServerConfig: canonical keys plus legacy
    // desktop aliases for reading older files. The server owns the full document;
    // the desktop only manages these three connection fields.
    private enum CodingKeys: String, CodingKey {
        case host, port, allowedOutputDirectory
        case serverHost, serverPort, outputDirectory // legacy
    }

    init() {}

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        serverHost = try c.decodeIfPresent(String.self, forKey: .host)
            ?? c.decodeIfPresent(String.self, forKey: .serverHost) ?? "127.0.0.1"
        serverPort = try c.decodeIfPresent(UInt16.self, forKey: .port)
            ?? c.decodeIfPresent(UInt16.self, forKey: .serverPort) ?? 7870
        outputDirectory = try c.decodeIfPresent(String.self, forKey: .allowedOutputDirectory)
            ?? c.decodeIfPresent(String.self, forKey: .outputDirectory) ?? "~/Pictures/ComfyBox"
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(serverHost, forKey: .host)
        try c.encode(serverPort, forKey: .port)
        try c.encode(outputDirectory, forKey: .allowedOutputDirectory)
    }

    /// Load config from disk. Returns defaults on any failure.
    static func load() -> AppConfig {
        let path = configPath
        guard FileManager.default.fileExists(atPath: path),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let config = try? JSONDecoder().decode(AppConfig.self, from: data) else {
            return AppConfig()
        }
        return config
    }

    /// Save the connection fields into the shared ~/.comfybox/config.json, preserving
    /// any server-owned keys (providers, replicate, modelSpec, …) already in the file.
    func save() throws {
        let dir = NSString(string: "~/.comfybox").expandingTildeInPath
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let url = URL(fileURLWithPath: AppConfig.configPath)

        var root: [String: Any] = [:]
        if let data = try? Data(contentsOf: url),
           let existing = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            root = existing
        }
        // Write canonical keys; drop legacy aliases so the file converges.
        root["host"] = serverHost
        root["port"] = Int(serverPort)
        root["allowedOutputDirectory"] = outputDirectory
        root.removeValue(forKey: "serverHost")
        root.removeValue(forKey: "serverPort")
        root.removeValue(forKey: "outputDirectory")

        let data = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
    }
}
