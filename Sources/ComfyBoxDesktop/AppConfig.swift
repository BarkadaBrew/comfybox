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

    /// Save config to disk, creating ~/.comfybox/ if needed.
    func save() throws {
        let dir = NSString(string: "~/.comfybox").expandingTildeInPath
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        try data.write(to: URL(fileURLWithPath: AppConfig.configPath))
    }
}
