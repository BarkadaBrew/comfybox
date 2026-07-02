// ViewTypeTests.swift — Tests for types defined in view files
//
// Tests types like ResolutionPreset, GallerySortOrder, FlowLayout,
// and DraggableAsset without rendering any SwiftUI views. These run
// headlessly over SSH without issues.

import SwiftUI
import Testing
@testable import ComfyBoxDesktop

@Suite("ResolutionPreset")
struct ResolutionPresetTests {
    @Test("presets contain standard resolutions")
    func standardResolutions() {
        let presets = ResolutionPreset.presets
        #expect(presets.count >= 4)
        let labels = presets.map { $0.label }
        #expect(labels.contains("1024 x 1024"))
        #expect(labels.contains("512 x 512"))
    }

    @Test("presets have unique IDs")
    func uniqueIds() {
        let ids = ResolutionPreset.presets.map { $0.id }
        #expect(Set(ids).count == ids.count)
    }

    @Test("label format is width x height")
    func labelFormat() {
        let preset = ResolutionPreset(id: "test", width: 800, height: 600)
        #expect(preset.label == "800 x 600")
    }

    @Test("hashable conformance")
    func hashable() {
        let a = ResolutionPreset(id: "a", width: 1024, height: 1024)
        let b = ResolutionPreset(id: "a", width: 1024, height: 1024)
        #expect(a == b)
    }
}

@Suite("GallerySortOrder")
struct GallerySortOrderTests {
    @Test("all cases have display names")
    func displayNames() {
        for order in GallerySortOrder.allCases {
            #expect(!order.rawValue.isEmpty)
        }
    }

    @Test("all expected cases exist")
    func allCases() {
        let cases = GallerySortOrder.allCases
        #expect(cases.count == 3)
        #expect(cases.contains(.date))
        #expect(cases.contains(.rating))
        #expect(cases.contains(.favorite))
    }

    @Test("raw values are human-readable")
    func rawValues() {
        #expect(GallerySortOrder.date.rawValue == "Date")
        #expect(GallerySortOrder.rating.rawValue == "Rating")
        #expect(GallerySortOrder.favorite.rawValue == "Favorites First")
    }
}

@Suite("FlowLayout")
struct FlowLayoutTests {
    @Test("default spacing")
    func defaultSpacing() {
        let layout = FlowLayout()
        #expect(layout.spacing == 4)
    }

    @Test("custom spacing")
    func customSpacing() {
        let layout = FlowLayout(spacing: 12)
        #expect(layout.spacing == 12)
    }

    @Test("conforms to Layout protocol")
    func conformsToLayout() {
        let _: any Layout = FlowLayout()
    }
}

@Suite("DraggableAsset")
struct DraggableAssetTests {
    @Test("stores path")
    func storesPath() {
        let d = DraggableAsset(path: "/tmp/image.png")
        #expect(d.path == "/tmp/image.png")
    }
}
