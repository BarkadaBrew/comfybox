// EditRendererTests.swift
import Testing
import Foundation
import CoreImage
import CoreGraphics
@testable import ComfyBoxDesktop

@Suite("EditRenderer")
struct EditRendererTests {
    static let context = CIContext(options: [.useSoftwareRenderer: true,
                                             .workingColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                             .outputColorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])

    func rendered(_ source: CGImage, _ recipe: EditRecipe, mask: CIImage? = nil) -> CGImage {
        let out = EditRenderer.render(source: CIImage(cgImage: source), recipe: recipe, subjectMask: mask)
        return Self.context.createCGImage(out, from: out.extent)!
    }

    @Test("identity recipe is pixel-identical")
    func identity() {
        let src = EditTestSupport.horizontalGradient(width: 64, height: 32)
        let out = rendered(src, EditRecipe())
        #expect(out.width == 64 && out.height == 32)
        for x in [0, 17, 40, 63] {
            #expect(abs(Int(EditTestSupport.gray(out, x: x, y: 10)) - Int(EditTestSupport.gray(src, x: x, y: 10))) <= 1)
        }
    }

    @Test("exposure +1 brightens mid grey")
    func exposure() {
        let src = EditTestSupport.solid(r: 100, g: 100, b: 100, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.exposure = 1
        #expect(EditTestSupport.gray(rendered(src, r), x: 8, y: 8) > 150)
    }

    @Test("positive temperature warms: red exceeds blue on grey")
    func warm() {
        let src = EditTestSupport.solid(r: 128, g: 128, b: 128, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.temperature = 1
        let p = EditTestSupport.pixel(rendered(src, r), x: 8, y: 8)
        #expect(p.r > p.b + 10)
    }

    @Test("saturation -1 makes a red image grey")
    func desaturate() {
        let src = EditTestSupport.solid(r: 200, g: 40, b: 40, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.saturation = -1
        let p = EditTestSupport.pixel(rendered(src, r), x: 8, y: 8)
        #expect(abs(Int(p.r) - Int(p.g)) < 8 && abs(Int(p.g) - Int(p.b)) < 8)
    }

    @Test("rgb curve lifting the midpoint brightens mid grey and leaves black/white")
    func curve() {
        let src = EditTestSupport.horizontalGradient(width: 256, height: 4)
        var r = EditRecipe(); r.adjustments.curves.rgb = [CurvePoint(x: 0.5, y: 0.75)]
        let out = rendered(src, r)
        #expect(EditTestSupport.gray(out, x: 128, y: 1) > 170)
        #expect(EditTestSupport.gray(out, x: 0, y: 1) < 4)
        #expect(EditTestSupport.gray(out, x: 255, y: 1) > 251)
    }

    @Test("flipH mirrors the gradient")
    func flip() {
        let src = EditTestSupport.horizontalGradient(width: 64, height: 8)
        var r = EditRecipe(); r.geometry.flipH = true
        let out = rendered(src, r)
        #expect(EditTestSupport.gray(out, x: 0, y: 4) > 240 && EditTestSupport.gray(out, x: 63, y: 4) < 15)
    }

    @Test("one quarter turn swaps dimensions and moves the bright edge to the bottom")
    func rotate() {
        let src = EditTestSupport.horizontalGradient(width: 64, height: 32)
        var r = EditRecipe(); r.geometry.quarterTurns = 1
        let out = rendered(src, r)
        #expect(out.width == 32 && out.height == 64)
        // Clockwise: the right (bright) edge becomes the bottom edge.
        #expect(EditTestSupport.gray(out, x: 16, y: 63) > 240 && EditTestSupport.gray(out, x: 16, y: 0) < 15)
    }

    @Test("crop yields the expected size and region")
    func crop() {
        let src = EditTestSupport.horizontalGradient(width: 100, height: 50)
        var r = EditRecipe(); r.geometry.crop = CGRect(x: 0.5, y: 0.0, width: 0.5, height: 1.0)
        let out = rendered(src, r)
        #expect(out.width == 50 && out.height == 50)
        #expect(EditTestSupport.gray(out, x: 0, y: 25) > 120)   // right half of the gradient
    }

    @Test("straighten leaves no transparent corners and shrinks the frame")
    func straighten() {
        let src = EditTestSupport.solid(r: 90, g: 90, b: 90, width: 120, height: 80)
        var r = EditRecipe(); r.geometry.straightenDegrees = 10
        let out = rendered(src, r)
        #expect(out.width < 120 && out.height < 80)
        for (x, y) in [(0, 0), (out.width - 1, 0), (0, out.height - 1), (out.width - 1, out.height - 1)] {
            #expect(EditTestSupport.pixel(out, x: x, y: y).a == 255)
        }
    }

    @Test("parameter mappings pin their endpoints")
    func mappings() {
        #expect(EditRenderer.contrastParameter(-1) == 0.5 && EditRenderer.contrastParameter(1) == 1.5)
        #expect(EditRenderer.saturationParameter(-1) == 0 && EditRenderer.saturationParameter(0) == 1)
        #expect(EditRenderer.tintTarget(1) == 150)
        #expect(EditRenderer.highlightAmount(-1) == 0.3 && EditRenderer.highlightAmount(1) == 1)
        #expect(EditRenderer.shadowAmount(0.5) == 0.5)
        let s = EditRenderer.largestInscribedSize(width: 100, height: 100, angleRadians: 0)
        #expect(s.width == 100 && s.height == 100)
        let t = EditRenderer.largestInscribedSize(width: 100, height: 50, angleRadians: .pi / 18)
        #expect(t.width < 100 && t.height < 50 && t.width > 60)
    }
}
