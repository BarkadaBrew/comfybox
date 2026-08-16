// DesktopSettingsTests.swift — Tests for DesktopSettings

import Testing
import Foundation
import SwiftUI
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

    @Test("uiScale maps to the dynamic type ladder, default when unset")
    func uiScaleMapping() {
        #expect(DesktopSettings.dynamicTypeSize(for: nil) == .large)
        #expect(DesktopSettings.dynamicTypeSize(for: "default") == .large)
        #expect(DesktopSettings.dynamicTypeSize(for: "large") == .xLarge)
        #expect(DesktopSettings.dynamicTypeSize(for: "xlarge") == .xxLarge)
        #expect(DesktopSettings.dynamicTypeSize(for: "xxlarge") == .xxxLarge)
        #expect(DesktopSettings.dynamicTypeSize(for: "garbage") == .large)
    }

    @Test("archiveRoots defaults to nil, defaultArchiveRoot expands the tilde")
    func archiveRootsDefaults() {
        #expect(DesktopSettings.defaultSettings.archiveRoots == nil)
        let expected = NSString(string: "~/.comfybox/archives").expandingTildeInPath
        #expect(DesktopSettings.defaultArchiveRoot == expected)
        #expect(!DesktopSettings.defaultArchiveRoot.hasPrefix("~"))
    }

    @Test("an old config JSON written before archiveRoots existed still decodes")
    func decodesOldConfigWithoutArchiveRoots() throws {
        // Deliberately omits "archiveRoots" (and other optionals added over
        // time) to simulate a desktop-config.json written by an older build.
        let json = """
        {
            "serverHost": "127.0.0.1",
            "serverPort": 7870,
            "autoConnect": true,
            "outputDirectory": "/tmp/output",
            "defaultSteps": 9,
            "defaultGuidance": 3.5,
            "defaultWidth": 1024,
            "defaultHeight": 1024,
            "thumbnailSize": 180,
            "gallerySortDefault": "date"
        }
        """
        let decoded = try JSONDecoder().decode(DesktopSettings.self, from: Data(json.utf8))
        #expect(decoded.archiveRoots == nil)
        #expect(decoded.serverHost == "127.0.0.1")
        #expect(decoded.watchedServices == nil)
    }
}
