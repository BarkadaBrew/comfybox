// EditRecipeTests.swift
import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("EditRecipe")
struct EditRecipeTests {
    @Test("default recipe is identity and version 1")
    func defaults() {
        let r = EditRecipe()
        #expect(r.isIdentity)
        #expect(r.version == 1)
        #expect(r.local == nil)
    }

    @Test("any change breaks identity")
    func nonIdentity() {
        var r = EditRecipe(); r.adjustments.exposure = 0.5; #expect(!r.isIdentity)
        var g = EditRecipe(); g.geometry.flipH = true; #expect(!g.isIdentity)
        var c = EditRecipe(); c.adjustments.curves.rgb = [CurvePoint(x: 0.5, y: 0.6)]; #expect(!c.isIdentity)
        var s = EditRecipe(); s.subject.removeBackground = true; #expect(!s.isIdentity)
        var l = EditRecipe(); l.local = EditLocalLayer(); #expect(!l.isIdentity)
        var v = EditRecipe(); v.version = 2; #expect(!v.isIdentity)
    }

    @Test("JSON round trip preserves every field")
    func roundTrip() throws {
        var r = EditRecipe()
        r.version = 7   // a non-default value, so the round trip can't coincidentally
                        // pass by comparing two recipes that both silently defaulted
        r.geometry.crop = CGRect(x: 0.1, y: 0.2, width: 0.5, height: 0.6)
        r.geometry.straightenDegrees = -3.5; r.geometry.quarterTurns = 3
        r.geometry.flipH = true; r.geometry.flipV = true
        r.adjustments.exposure = 1.25
        r.adjustments.contrast = 0.11
        r.adjustments.highlights = -0.22
        r.adjustments.shadows = 0.33
        r.adjustments.whites = -0.44
        r.adjustments.blacks = 0.55
        r.adjustments.temperature = -0.4
        r.adjustments.tint = 0.66
        r.adjustments.vibrance = -0.77
        r.adjustments.saturation = 0.88
        r.adjustments.sharpen = 0.15
        r.adjustments.noiseReduction = 0.25
        r.adjustments.vignette = 0.35
        r.adjustments.curves.rgb = [CurvePoint(x: 0.4, y: 0.45)]
        r.adjustments.curves.r = [CurvePoint(x: 0.25, y: 0.3)]
        r.adjustments.curves.g = [CurvePoint(x: 0.6, y: 0.55)]
        r.adjustments.curves.b = [CurvePoint(x: 0.7, y: 0.2)]
        var layer = EditLocalLayer()
        layer.feather = 0.3
        layer.adjustments.exposure = 0.9
        layer.adjustments.contrast = 0.12
        layer.adjustments.highlights = -0.13
        layer.adjustments.shadows = 0.5
        layer.adjustments.temperature = 0.14
        layer.adjustments.tint = -0.15
        layer.adjustments.saturation = 0.16
        layer.adjustments.sharpen = 0.17
        layer.mask.append(MaskStroke(points: [CGPoint(x: 0.1, y: 0.1)], size: 0.05, erase: false))
        r.local = layer
        r.subject = EditSubject(removeBackground: true, invert: true)
        let data = try JSONEncoder().encode(r)
        let back = try JSONDecoder().decode(EditRecipe.self, from: data)
        #expect(back == r)
    }

    @Test("curve normalization sorts, clamps, and inserts endpoints")
    func normalization() {
        let pts = [CurvePoint(x: 0.8, y: 1.4), CurvePoint(x: 0.2, y: -0.1)]
        let n = ToneCurves.normalized(pts)
        #expect(n.first == CurvePoint(x: 0, y: 0))
        #expect(n.last == CurvePoint(x: 1, y: 1))
        #expect(n[1] == CurvePoint(x: 0.2, y: 0))
        #expect(n[2] == CurvePoint(x: 0.8, y: 1))
        #expect(ToneCurves.normalized([]) == [CurvePoint(x: 0, y: 0), CurvePoint(x: 1, y: 1)])
    }

    @Test("curve normalization coalesces duplicate x-values, keeping the last")
    func normalizationCoalescesDuplicates() {
        let dup = ToneCurves.normalized([CurvePoint(x: 0.5, y: 0.2), CurvePoint(x: 0.5, y: 0.8)])
        #expect(dup == [CurvePoint(x: 0, y: 0), CurvePoint(x: 0.5, y: 0.8), CurvePoint(x: 1, y: 1)])

        let clampedToZero = ToneCurves.normalized([CurvePoint(x: -0.5, y: 0.3)])
        #expect(clampedToZero == [CurvePoint(x: 0, y: 0.3), CurvePoint(x: 1, y: 1)])

        let a = ToneCurves.sample([CurvePoint(x: 0.5, y: 0.1), CurvePoint(x: 0.5, y: 0.9)], at: 0.499)
        let b = ToneCurves.sample([CurvePoint(x: 0.5, y: 0.1), CurvePoint(x: 0.5, y: 0.9)], at: 0.501)
        #expect(abs(a - b) < 0.05)
    }

    @Test("curve sampling is identity when empty and interpolates through points")
    func sampling() {
        #expect(abs(ToneCurves.sample([], at: 0.37) - 0.37) < 1e-9)
        let pts = [CurvePoint(x: 0.5, y: 0.8)]
        #expect(abs(ToneCurves.sample(pts, at: 0.5) - 0.8) < 1e-9)
        #expect(ToneCurves.sample(pts, at: 0.25) > 0.25)     // lifted between 0 and the point
        #expect(ToneCurves.sample(pts, at: 0.0) == 0 && ToneCurves.sample(pts, at: 1.0) == 1)
    }

    @Test("restrictedToLocal zeroes unsupported fields")
    func restricted() {
        var a = EditAdjustments()
        a.exposure = 1; a.vibrance = 1; a.vignette = 1; a.blacks = 1; a.curves.rgb = [CurvePoint(x: 0.5, y: 0.7)]
        let r = a.restrictedToLocal
        #expect(r.exposure == 1)
        #expect(r.vibrance == 0 && r.vignette == 0 && r.blacks == 0 && r.curves.isIdentity)
    }
}
