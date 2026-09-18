// NSFWPresentationTests.swift — one rule for "may this be shown", used by
// every surface that puts an image on screen.
//
// Todd 2026-09-18: "dashboard recents should honor NSFW on/off. Current
// behavior is recents are shown even if image display is off in galleries."
// The dashboard had no gate at all: it drew the twelve newest thumbnails
// whatever the gallery was set to.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("NSFW presentation")
struct NSFWPresentationTests {

    @Test("an SFW asset is always shown, whatever the gate says")
    func sfwAlwaysShows() {
        for mode in NSFWFilterMode.allCases {
            for revealed in [true, false] {
                #expect(ContentRating.presentation(isNSFW: false, mode: mode, unlocked: false, gateRevealed: revealed) == .show)
            }
        }
    }

    @Test("with the app gate closed, mature content is omitted — not blurred")
    func gateClosedOmits() {
        for mode in NSFWFilterMode.allCases {
            #expect(ContentRating.presentation(isNSFW: true, mode: mode, unlocked: false, gateRevealed: false) == .omit,
                    "mode \(mode.rawValue)")
        }
    }

    @Test("an unlocked session still respects a closed app gate")
    func unlockedDoesNotBeatTheAppGate() {
        #expect(ContentRating.presentation(isNSFW: true, mode: .show, unlocked: true, gateRevealed: false) == .omit)
    }

    @Test("with the gate open, the filter mode decides")
    func modeDecidesWhenRevealed() {
        #expect(ContentRating.presentation(isNSFW: true, mode: .show, unlocked: false, gateRevealed: true) == .show)
        #expect(ContentRating.presentation(isNSFW: true, mode: .blur, unlocked: false, gateRevealed: true) == .blur)
        #expect(ContentRating.presentation(isNSFW: true, mode: .blur, unlocked: true, gateRevealed: true) == .show)
        #expect(ContentRating.presentation(isNSFW: true, mode: .hide, unlocked: false, gateRevealed: true) == .omit)
        #expect(ContentRating.presentation(isNSFW: true, mode: .hide, unlocked: true, gateRevealed: true) == .show)
    }

    @Test("the filter mode is app-wide and survives a restart")
    func modeIsPersisted() {
        let original = DesktopSettings.load().nsfwFilterMode
        defer {
            var settings = DesktopSettings.load()
            settings.nsfwFilterMode = original
            settings.save()
        }
        var settings = DesktopSettings.load()
        settings.nsfwFilterMode = NSFWFilterMode.hide.rawValue
        settings.save()
        #expect(DesktopSettings.load().resolvedNSFWFilterMode == .hide,
                "the gallery's choice is what the dashboard reads")
        settings.nsfwFilterMode = nil
        settings.save()
        #expect(DesktopSettings.load().resolvedNSFWFilterMode == .blur, "default stays Blur NSFW")
    }
}
