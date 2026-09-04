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
        #expect(EditRenderer.contrastParameter(0) == 1)
        #expect(EditRenderer.saturationParameter(-1) == 0 && EditRenderer.saturationParameter(0) == 1)
        #expect(EditRenderer.saturationParameter(1) == 2)
        #expect(EditRenderer.tintTarget(1) == 150)
        #expect(EditRenderer.tintTarget(-1) == -150 && EditRenderer.tintTarget(0) == 0)
        #expect(EditRenderer.highlightAmount(-1) == 0.3 && EditRenderer.highlightAmount(1) == 1)
        #expect(EditRenderer.highlightAmount(0) == 1)
        #expect(EditRenderer.shadowAmount(0.5) == 0.5)
        #expect(EditRenderer.shadowAmount(-1) == -1 && EditRenderer.shadowAmount(1) == 1)
        #expect(EditRenderer.temperatureTarget(0) == 6500)
        // Pin both endpoints against the actual numbers the code's sign produces:
        // 6500 - v*3000, so +1 -> 3500 (warmer / lower Kelvin target) and -1 -> 9500.
        #expect(EditRenderer.temperatureTarget(1) == 3500)
        #expect(EditRenderer.temperatureTarget(-1) == 9500)
        let hi = EditRenderer.whitesBlacksCurve(whites: 1, blacks: -1, highlights: 0)
        #expect(abs(hi[1].y - 0.10) < 1e-9)
        #expect(abs(hi[3].y - 0.90) < 1e-9)
        let hiHighlights = EditRenderer.whitesBlacksCurve(whites: 0, blacks: 0, highlights: 1)
        #expect(abs(hiHighlights[3].y - 0.85) < 1e-9)
        let s = EditRenderer.largestInscribedSize(width: 100, height: 100, angleRadians: 0)
        #expect(s.width == 100 && s.height == 100)
        let t = EditRenderer.largestInscribedSize(width: 100, height: 50, angleRadians: .pi / 18)
        #expect(t.width < 100 && t.height < 50 && t.width > 60)
    }

    @Test("per-channel r curve raises only red")
    func curveChannelR() {
        let src = EditTestSupport.solid(r: 128, g: 128, b: 128, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.curves.r = [CurvePoint(x: 0.5, y: 0.9)]
        let p = EditTestSupport.pixel(rendered(src, r), x: 8, y: 8)
        #expect(p.r > 190)
        #expect(abs(Int(p.g) - 128) <= 2)
        #expect(abs(Int(p.b) - 128) <= 2)
    }

    @Test("per-channel g curve raises only green")
    func curveChannelG() {
        let src = EditTestSupport.solid(r: 128, g: 128, b: 128, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.curves.g = [CurvePoint(x: 0.5, y: 0.9)]
        let p = EditTestSupport.pixel(rendered(src, r), x: 8, y: 8)
        #expect(p.g > 190)
        #expect(abs(Int(p.r) - 128) <= 2)
        #expect(abs(Int(p.b) - 128) <= 2)
    }

    @Test("per-channel b curve raises only blue")
    func curveChannelB() {
        let src = EditTestSupport.solid(r: 128, g: 128, b: 128, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.curves.b = [CurvePoint(x: 0.5, y: 0.9)]
        let p = EditTestSupport.pixel(rendered(src, r), x: 8, y: 8)
        #expect(p.b > 190)
        #expect(abs(Int(p.r) - 128) <= 2)
        #expect(abs(Int(p.g) - 128) <= 2)
    }

    @Test("per-channel red curve preserves alpha on a half-transparent image")
    func curveChannelRPreservesAlpha() {
        let src = EditTestSupport.solidRGBA(r: 128, g: 128, b: 128, a: 128, width: 16, height: 16)
        var r = EditRecipe(); r.adjustments.curves.r = [CurvePoint(x: 0.5, y: 0.9)]
        let p = EditTestSupport.pixel(rendered(src, r), x: 8, y: 8)
        #expect(abs(Int(p.a) - 128) <= 1)
        #expect(p.r > p.g)
    }

    @Test("crop is correct against a non-origin extent")
    func cropTranslatedExtent() {
        let src = EditTestSupport.horizontalGradient(width: 100, height: 50)
        let translated = CIImage(cgImage: src).transformed(by: CGAffineTransform(translationX: 37, y: 19))
        var g = EditGeometry(); g.crop = CGRect(x: 0.5, y: 0.0, width: 0.5, height: 1.0)
        let out = EditRenderer.applyGeometry(translated, g)
        #expect(out.extent.width == 50)
        let cg = Self.context.createCGImage(out, from: out.extent)!
        #expect(EditTestSupport.gray(cg, x: 0, y: 25) > 120)
    }

    @Test("local layer with a left-half mask brightens only the masked half")
    func localLayer() {
        let src = EditTestSupport.solid(r: 100, g: 100, b: 100, width: 100, height: 40)
        var r = EditRecipe()
        var layer = EditLocalLayer()
        // A wide vertical stroke covering x in 0…0.5 (size is a fraction of width).
        layer.mask.append(MaskStroke(points: [CGPoint(x: 0.25, y: 0.0), CGPoint(x: 0.25, y: 1.0)], size: 0.5, erase: false))
        layer.adjustments.exposure = 1.5
        r.local = layer
        let out = rendered(src, r)
        #expect(EditTestSupport.gray(out, x: 10, y: 20) > 150)
        #expect(abs(Int(EditTestSupport.gray(out, x: 90, y: 20)) - 100) <= 2)
    }

    @Test("local layer ignores fields outside the local subset")
    func localRestricted() {
        let src = EditTestSupport.solid(r: 100, g: 100, b: 100, width: 40, height: 40)
        var r = EditRecipe()
        var layer = EditLocalLayer()
        layer.mask.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 2, erase: false))
        layer.adjustments.vignette = 1   // not in the local subset
        r.local = layer
        let out = rendered(src, r)
        #expect(abs(Int(EditTestSupport.gray(out, x: 2, y: 2)) - 100) <= 2)
    }

    @Test("subject removal makes the unmasked region transparent; invert flips it")
    func subject() {
        let src = EditTestSupport.solid(r: 50, g: 120, b: 200, width: 40, height: 40)
        // Mask: left half white (subject), right half black.
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.25, y: 0), CGPoint(x: 0.25, y: 1)], size: 0.5, erase: false))
        let mask = CIImage(cgImage: MaskRasterizer.render(strokes, size: CGSize(width: 40, height: 40))!)
        var r = EditRecipe(); r.subject.removeBackground = true
        let out = rendered(src, r, mask: mask)
        #expect(EditTestSupport.pixel(out, x: 5, y: 20).a == 255)
        #expect(EditTestSupport.pixel(out, x: 35, y: 20).a == 0)
        r.subject.invert = true
        let inv = rendered(src, r, mask: mask)
        #expect(EditTestSupport.pixel(inv, x: 5, y: 20).a == 0)
        #expect(EditTestSupport.pixel(inv, x: 35, y: 20).a == 255)
    }

    @Test("subject flag without a mask is a no-op")
    func subjectNoMask() {
        let src = EditTestSupport.solid(r: 50, g: 120, b: 200, width: 8, height: 8)
        var r = EditRecipe(); r.subject.removeBackground = true
        #expect(EditTestSupport.pixel(rendered(src, r), x: 4, y: 4).a == 255)
    }

    @Test("subject mask follows geometry: flipH moves the transparent half")
    func subjectFollowsGeometry() {
        let src = EditTestSupport.solid(r: 50, g: 120, b: 200, width: 40, height: 40)
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.25, y: 0), CGPoint(x: 0.25, y: 1)], size: 0.5, erase: false))
        let mask = CIImage(cgImage: MaskRasterizer.render(strokes, size: CGSize(width: 40, height: 40))!)
        var r = EditRecipe(); r.subject.removeBackground = true; r.geometry.flipH = true
        let out = rendered(src, r, mask: mask)
        #expect(EditTestSupport.pixel(out, x: 35, y: 20).a == 255)
        #expect(EditTestSupport.pixel(out, x: 5, y: 20).a == 0)
    }

    @Test("local layer aligns the mask when the image extent is not at the origin")
    func localLayerTranslatedExtent() {
        let src = EditTestSupport.solid(r: 100, g: 100, b: 100, width: 100, height: 40)
        let translated = CIImage(cgImage: src).transformed(by: CGAffineTransform(translationX: 37, y: 19))
        var layer = EditLocalLayer()
        layer.mask.append(MaskStroke(points: [CGPoint(x: 0.25, y: 0.0), CGPoint(x: 0.25, y: 1.0)], size: 0.5, erase: false))
        layer.adjustments.exposure = 1.5
        let out = EditRenderer.applyLocalLayer(translated, layer)
        let cg = Self.context.createCGImage(out, from: out.extent)!
        #expect(EditTestSupport.gray(cg, x: 10, y: 20) > 150)
        #expect(abs(Int(EditTestSupport.gray(cg, x: 90, y: 20)) - 100) <= 2)
    }

    @Test("subject mask aligns with a translated source extent")
    func subjectTranslatedExtent() {
        let src = EditTestSupport.solid(r: 50, g: 120, b: 200, width: 40, height: 40)
        let translatedSource = CIImage(cgImage: src).transformed(by: CGAffineTransform(translationX: 37, y: 19))
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.25, y: 0), CGPoint(x: 0.25, y: 1)], size: 0.5, erase: false))
        let mask = CIImage(cgImage: MaskRasterizer.render(strokes, size: CGSize(width: 40, height: 40))!)
        var r = EditRecipe(); r.subject.removeBackground = true
        let out = EditRenderer.render(source: translatedSource, recipe: r, subjectMask: mask)
        let cg = Self.context.createCGImage(out, from: out.extent)!
        #expect(EditTestSupport.pixel(cg, x: 5, y: 20).a == 255)
        #expect(EditTestSupport.pixel(cg, x: 35, y: 20).a == 0)
    }
}
