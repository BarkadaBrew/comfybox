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
        #expect(presets.count >= 8)
        let sizes = presets.map { "\($0.width)x\($0.height)" }
        #expect(sizes.contains("1024x1024"))
        #expect(sizes.contains("512x512"))
        // Todd's staple aspect ratios (headshot, full body, landscape).
        #expect(sizes.contains("1280x1280"))
        #expect(sizes.contains("1024x1536"))
        #expect(sizes.contains("1536x1024"))
    }

    @Test("presets have unique IDs")
    func uniqueIds() {
        let ids = ResolutionPreset.presets.map { $0.id }
        #expect(Set(ids).count == ids.count)
    }

    @Test("label format is width × height, with an optional aspect hint")
    func labelFormat() {
        let plain = ResolutionPreset(id: "test", width: 800, height: 600)
        #expect(plain.label == "800 × 600")
        let hinted = ResolutionPreset(id: "test2", width: 800, height: 600, hint: "4:3")
        #expect(hinted.label == "800 × 600  (4:3)")
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
