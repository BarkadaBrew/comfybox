// DesktopSettingsTests.swift — Tests for DesktopSettings

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("DesktopSettings")
struct DesktopSettingsTests {
    @Test("default settings have expected values")
    func defaults() {
        let settings = DesktopSettings.defaultSettings
        #expect(settings.serverHost == "127.0.0.1")
        #expect(settings.serverPort == 7870)
        #expect(settings.autoConnect == true)
        #expect(settings.defaultSteps == 9)
        #expect(settings.defaultGuidance == 3.5)
        #expect(settings.defaultWidth == 1024)
        #expect(settings.defaultHeight == 1024)
        #expect(settings.thumbnailSize == 180)
        #expect(settings.gallerySortDefault == "date")
    }

    @Test("settings round-trip through JSON encoding")
    func roundTrip() throws {
        var settings = DesktopSettings.defaultSettings
        settings.serverHost = "10.0.0.5"
        settings.serverPort = 9999
        settings.autoConnect = false
        settings.defaultSteps = 25
        let encoder = JSONEncoder()
        let data = try encoder.encode(settings)
        let decoder = JSONDecoder()
        let decoded = try decoder.decode(DesktopSettings.self, from: data)
        #expect(decoded.serverHost == "10.0.0.5")
        #expect(decoded.serverPort == 9999)
        #expect(decoded.autoConnect == false)
        #expect(decoded.defaultSteps == 25)
    }

    @Test("load returns defaults when config file missing")
    func loadMissing() {
        let settings = DesktopSettings.load()
        #expect(!settings.serverHost.isEmpty)
    }

    @Test("thumbnail sizes are positive")
    func thumbnailSizes() {
        let settings = DesktopSettings.defaultSettings
        #expect(settings.thumbnailSize > 0)
    }

    @Test("configPath is in comfybox directory")
    func configPath() {
        let path = DesktopSettings.configPath
        #expect(path.contains(".comfybox"))
        #expect(path.contains("desktop-config.json"))
    }
}
