// CameraDirectiveTests.swift — Camera phrase composition + shot templates

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("CameraDirective")
struct CameraDirectiveTests {
    @Test("default composes shot size + front view, omits eye-level and no-lens")
    func defaultPhrase() {
        let d = CameraDirective()
        #expect(d.phrase == "medium shot, front view")
    }

    @Test("full directive includes angle and lens")
    func fullPhrase() {
        // Lens phrases are prose descriptions of the visual effect, not lens
        // jargon — the Qwen text encoders parse prose ("85mm" reads as noise,
        // "background melting into soft creamy blur" reads as intent).
        let d = CameraDirective(orientation: .threeQuarter, angle: .lowAngle, shotSize: .closeUp, lens: .mm85)
        #expect(d.phrase == "close-up shot, three-quarter view, low-angle shot looking up, flattering portrait compression, tight subject framing, background melting into soft creamy blur with only the subject in sharp focus")
    }

    @Test("eye-level angle is omitted; none lens is omitted")
    func omissions() {
        let d = CameraDirective(orientation: .profile, angle: .eyeLevel, shotSize: .wide, lens: .none)
        #expect(d.phrase == "wide establishing shot, profile side view")
    }

    @Test("appended joins onto an existing prompt")
    func appended() {
        let d = CameraDirective(orientation: .front, angle: .highAngle, shotSize: .medium)
        #expect(d.appended(to: "a woman in a cafe") == "a woman in a cafe, medium shot, front view, high-angle shot looking down")
        #expect(d.appended(to: "  ") == d.phrase)   // empty prompt -> just the phrase
    }

    @Test("all enum cases produce a non-empty label")
    func labels() {
        #expect(CameraDirective.Angle.allCases.allSatisfy { !$0.label.isEmpty })
        #expect(CameraDirective.Orientation.allCases.allSatisfy { !$0.label.isEmpty })
        #expect(CameraDirective.ShotSize.allCases.allSatisfy { !$0.label.isEmpty })
        #expect(CameraDirective.Lens.allCases.allSatisfy { !$0.label.isEmpty })
    }
}

@Suite("ShotTemplateStore")
struct ShotTemplateStoreTests {
    @Test("parses the legacy shot-templates object")
    func parse() {
        let object: [String: Any] = [
            "cinematic-wide-establishing": [
                "directive": "Cinematic wide establishing shot — 24mm lens.",
                "tags": ["wide", "establishing"],
                "contentMode": "neutral",
            ],
            "cinematic-insert-macro": [
                "directive": "100mm macro insert shot.",
                "tags": ["insert", "macro"],
                "contentMode": "avocado",
            ],
            "bad-entry": ["tags": ["x"]],   // no directive -> skipped
        ]
        let templates = ShotTemplateStore.parse(object: object)
        #expect(templates.count == 2)
        // Sorted by id.
        #expect(templates.first?.id == "cinematic-insert-macro")
        #expect(templates.first?.name == "Cinematic Insert Macro")
        #expect(templates.last?.tags == ["wide", "establishing"])
        #expect(templates.last?.contentMode == "neutral")
    }

    @Test("missing file yields no templates")
    func missing() {
        let templates = ShotTemplateStore.parse(contentsOf: URL(fileURLWithPath: "/nope/shots.json"))
        #expect(templates.isEmpty)
    }
}
