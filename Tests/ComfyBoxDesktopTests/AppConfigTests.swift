// AppConfigTests.swift — Tests for AppConfig

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("AppConfig")
struct AppConfigTests {
    @Test("default config has expected values")
    func defaults() {
        let config = AppConfig()
        #expect(config.serverHost == "127.0.0.1")
        #expect(config.serverPort == 7870)
        #expect(config.outputDirectory == "~/Pictures/ComfyBox")
    }

    @Test("config round-trips through JSON")
    func jsonRoundTrip() throws {
        var config = AppConfig()
        config.serverHost = "192.168.1.100"
        config.serverPort = 8080
        config.outputDirectory = "/tmp/output"

        let encoder = JSONEncoder()
        let data = try encoder.encode(config)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(AppConfig.self, from: data)

        #expect(decoded.serverHost == "192.168.1.100")
        #expect(decoded.serverPort == 8080)
        #expect(decoded.outputDirectory == "/tmp/output")
    }

    @Test("load returns defaults for missing file")
    func loadMissing() {
        let config = AppConfig.load()
        #expect(!config.serverHost.isEmpty)
        #expect(config.serverPort > 0)
    }

    @Test("configPath points to comfybox directory")
    func configPath() {
        let path = AppConfig.configPath
        #expect(path.contains(".comfybox"))
        #expect(path.contains("config.json"))
    }
}
