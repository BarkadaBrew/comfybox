// ModelsViewTests.swift — #273: pure presentation logic for the nearline
// row's Anchor/Un-anchor control, extracted from the SwiftUI body so it is
// unit-testable without a live server or view hosting.

import Testing
@testable import ComfyBoxDesktop

@Suite("NearlineAnchorRowViewModel")
struct NearlineAnchorRowViewModelTests {
    private func entry(anchored: Bool, staged: Bool = true) -> EngineService.NearlineEntry {
        EngineService.NearlineEntry(
            name: "x.safetensors", path: "/vol/x.safetensors", sizeMB: 4, kind: "lora",
            staged: staged, anchored: anchored)
    }

    @Test("unanchored item shows Anchor with no pin, next tap requests true")
    func unanchoredState() {
        let vm = NearlineAnchorRowViewModel(item: entry(anchored: false))
        #expect(vm.pinGlyphVisible == false)
        #expect(vm.buttonTitle == "Anchor")
        #expect(vm.nextAnchoredValue == true)
    }

    @Test("anchored item shows Un-anchor with a pin, next tap requests false")
    func anchoredState() {
        let vm = NearlineAnchorRowViewModel(item: entry(anchored: true))
        #expect(vm.pinGlyphVisible == true)
        #expect(vm.buttonTitle == "Un-anchor")
        #expect(vm.nextAnchoredValue == false)
    }

    @Test("anchor state is independent of staged state")
    func anchoredButNotStaged() {
        // An item can be anchored=true in flight (flag set, stage() not yet
        // observed by this client) — the row must still offer "Un-anchor".
        let vm = NearlineAnchorRowViewModel(item: entry(anchored: true, staged: false))
        #expect(vm.buttonTitle == "Un-anchor")
    }
}
