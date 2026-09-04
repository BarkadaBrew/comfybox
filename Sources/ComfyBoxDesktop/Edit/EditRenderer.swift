// EditRenderer.swift — pure Core Image pipeline for EditRecipe
//
// Order: geometry → global adjustments → local layer → subject alpha.
// No I/O, no caching, no main-actor requirement. Every recipe→filter
// mapping is a small static func so tests can pin it.

import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics

public enum EditRenderer {

    public static func render(source: CIImage, recipe: EditRecipe, subjectMask: CIImage?) -> CIImage {
        var image = applyGeometry(source, recipe.geometry)
        image = applyAdjustments(image, recipe.adjustments)
        image = applyLocalLayer(image, recipe.local)
        image = applySubject(image, recipe.subject, mask: subjectMask, geometry: recipe.geometry, sourceExtent: source.extent)
        return image
    }

    // MARK: - Geometry

    static func applyGeometry(_ input: CIImage, _ g: EditGeometry) -> CIImage {
        var image = input
        let turns = ((g.quarterTurns % 4) + 4) % 4
        if turns > 0 {
            // CI rotates counter-clockwise for positive angles; quarterTurns is clockwise.
            image = image.transformed(by: CGAffineTransform(rotationAngle: -CGFloat(turns) * .pi / 2))
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        }
        if g.flipH {
            image = image.transformed(by: CGAffineTransform(scaleX: -1, y: 1))
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: 0))
        }
        if g.flipV {
            image = image.transformed(by: CGAffineTransform(scaleX: 1, y: -1))
            image = image.transformed(by: CGAffineTransform(translationX: 0, y: -image.extent.minY))
        }
        if abs(g.straightenDegrees) > 0.001 {
            let angle = CGFloat(g.straightenDegrees) * .pi / 180
            let w = image.extent.width, h = image.extent.height
            let center = CGPoint(x: image.extent.midX, y: image.extent.midY)
            var t = CGAffineTransform(translationX: center.x, y: center.y)
            t = t.rotated(by: -angle)          // clockwise for positive degrees
            t = t.translatedBy(x: -center.x, y: -center.y)
            let rotated = image.transformed(by: t)
            let fit = largestInscribedSize(width: w, height: h, angleRadians: angle)
            // `largestInscribedSize` is tangent to the rotated quad's edges, so the crop rect
            // must round INWARD (never outward, as `.integral` does) or the corner pixels
            // sample past the rotated content into transparent extrapolation. Round each edge
            // toward the interior independently (no blanket margin, so we keep the full extent
            // the fit actually earns) rather than rounding a width/height pair, which can drift.
            let minX = (center.x - fit.width / 2).rounded(.up)
            let minY = (center.y - fit.height / 2).rounded(.up)
            let maxX = (center.x + fit.width / 2).rounded(.down)
            let maxY = (center.y + fit.height / 2).rounded(.down)
            let cropRect = CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
            image = rotated.cropped(to: cropRect)
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        }
        if let c = g.crop {
            let extent = image.extent
            let w = extent.width, h = extent.height
            // Normalized crop has a top-left origin; CI extents are bottom-up. Edges are computed
            // relative to the extent's own origin (never assumed to be (0,0) — a prior transform,
            // e.g. a flip or rotation, can leave it elsewhere) and independently rounded per edge
            // rather than rounding an origin/size pair, which can let the far edge drift a pixel.
            let left = extent.minX + (c.minX * w).rounded()
            let right = extent.minX + (c.maxX * w).rounded()
            let bottom = extent.minY + ((1 - c.maxY) * h).rounded()
            let top = extent.minY + ((1 - c.minY) * h).rounded()
            let rect = CGRect(x: left, y: bottom, width: right - left, height: top - bottom)
            image = image.cropped(to: rect.intersection(extent))
            image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        }
        return image
    }

    /// Largest axis-aligned rectangle with the source's aspect that fits inside the source rotated by `angle`.
    static func largestInscribedSize(width w: CGFloat, height h: CGFloat, angleRadians: CGFloat) -> CGSize {
        let s = abs(sin(angleRadians)), c = abs(cos(angleRadians))
        if s < 1e-9 { return CGSize(width: w, height: h) }
        let widthIsLonger = w >= h
        let side = widthIsLonger ? h : w
        let long = widthIsLonger ? w : h
        if side <= 2 * s * c * long || abs(s - c) < 1e-9 {
            let x = 0.5 * side
            return widthIsLonger ? CGSize(width: x / s, height: x / c) : CGSize(width: x / c, height: x / s)
        }
        let cos2 = c * c - s * s
        return CGSize(width: (w * c - h * s) / cos2, height: (h * c - w * s) / cos2)
    }

    // MARK: - Global adjustments

    static func applyAdjustments(_ input: CIImage, _ a: EditAdjustments) -> CIImage {
        var image = input
        if a.exposure != 0 {
            let f = CIFilter.exposureAdjust(); f.inputImage = image; f.ev = Float(a.exposure); image = f.outputImage ?? image
        }
        if a.temperature != 0 || a.tint != 0 {
            let f = CIFilter.temperatureAndTint(); f.inputImage = image
            f.neutral = CIVector(x: 6500, y: 0)
            f.targetNeutral = CIVector(x: temperatureTarget(a.temperature), y: tintTarget(a.tint))
            image = f.outputImage ?? image
        }
        if a.highlights < 0 || a.shadows != 0 {
            let f = CIFilter.highlightShadowAdjust(); f.inputImage = image
            f.highlightAmount = Float(highlightAmount(a.highlights)); f.shadowAmount = Float(shadowAmount(a.shadows))
            f.radius = 3
            image = f.outputImage ?? image
        }
        if a.whites != 0 || a.blacks != 0 || a.highlights > 0 {
            image = toneCurve(image, whitesBlacksCurve(whites: a.whites, blacks: a.blacks, highlights: a.highlights))
        }
        if a.contrast != 0 || a.saturation != 0 {
            let f = CIFilter.colorControls(); f.inputImage = image
            f.contrast = Float(contrastParameter(a.contrast)); f.saturation = Float(saturationParameter(a.saturation)); f.brightness = 0
            image = f.outputImage ?? image
        }
        if a.vibrance != 0 {
            let f = CIFilter.vibrance(); f.inputImage = image; f.amount = Float(a.vibrance); image = f.outputImage ?? image
        }
        if !a.curves.isIdentity { image = applyCurves(image, a.curves) }
        if a.sharpen > 0 {
            let f = CIFilter.sharpenLuminance(); f.inputImage = image; f.sharpness = Float(a.sharpen * 2); f.radius = 1.69
            image = f.outputImage?.cropped(to: input.extent) ?? image
        }
        if a.noiseReduction > 0 {
            let f = CIFilter.noiseReduction(); f.inputImage = image; f.noiseLevel = Float(a.noiseReduction * 0.1); f.sharpness = 0.4
            image = f.outputImage?.cropped(to: input.extent) ?? image
        }
        if a.vignette > 0 {
            let f = CIFilter.vignette(); f.inputImage = image; f.intensity = Float(a.vignette); f.radius = 1.5
            image = f.outputImage ?? image
        }
        return image.cropped(to: input.extent)
    }

    static func contrastParameter(_ v: Double) -> Double { 1 + 0.5 * v }
    static func saturationParameter(_ v: Double) -> Double { 1 + v }
    /// Sign chosen so positive = warmer; the `warm` test pins the direction. If it fails, flip the sign here.
    static func temperatureTarget(_ v: Double) -> CGFloat { CGFloat(6500 - v * 3000) }
    static func tintTarget(_ v: Double) -> CGFloat { CGFloat(v * 150) }
    // Written as (10 + 7·min(v,0)) / 10 rather than 1 + 0.7·min(v,0) so that the
    // v == -1 endpoint lands on the exact Double value 0.3 (the naive form is one
    // ULP off due to 0.7's binary rounding).
    static func highlightAmount(_ v: Double) -> Double { (10 + min(v, 0) * 7) / 10 }
    static func shadowAmount(_ v: Double) -> Double { v }

    /// Blacks move the 0.25 point, whites the 0.75 point, positive highlights lift 0.75 further.
    static func whitesBlacksCurve(whites: Double, blacks: Double, highlights: Double) -> [CurvePoint] {
        let hi = max(highlights, 0)
        return [CurvePoint(x: 0, y: 0),
                CurvePoint(x: 0.25, y: min(max(0.25 + blacks * 0.15, 0), 1)),
                CurvePoint(x: 0.5, y: 0.5),
                CurvePoint(x: 0.75, y: min(max(0.75 + whites * 0.15 + hi * 0.1, 0), 1)),
                CurvePoint(x: 1, y: 1)]
    }

    /// Five-point CIToneCurve sampled from an arbitrary control-point curve.
    static func toneCurve(_ image: CIImage, _ pts: [CurvePoint]) -> CIImage {
        let f = CIFilter.toneCurve(); f.inputImage = image
        let xs: [Double] = [0, 0.25, 0.5, 0.75, 1]
        let ys = xs.map { ToneCurves.sample(pts, at: $0) }
        f.point0 = CGPoint(x: xs[0], y: ys[0]); f.point1 = CGPoint(x: xs[1], y: ys[1]); f.point2 = CGPoint(x: xs[2], y: ys[2])
        f.point3 = CGPoint(x: xs[3], y: ys[3]); f.point4 = CGPoint(x: xs[4], y: ys[4])
        return f.outputImage ?? image
    }

    static func applyCurves(_ input: CIImage, _ c: ToneCurves) -> CIImage {
        var image = input
        if !c.rgb.isEmpty { image = toneCurve(image, c.rgb) }
        for (channel, pts) in [(0, c.r), (1, c.g), (2, c.b)] where !pts.isEmpty {
            let curved = toneCurve(image, pts)
            image = replaceChannel(of: image, with: curved, channel: channel)
        }
        return image
    }

    /// Take `channel` (0=r,1=g,2=b) from `donor`, the other two from `base`. Alpha from base.
    ///
    /// `CIColorMatrix` always unpremultiplies its input by the input's own alpha, applies the
    /// matrix, then RE-premultiplies the result by whatever alpha the matrix just computed. That
    /// means an output alpha of 0 forces the output color to 0 too, no matter what the matrix's
    /// r/g/b rows compute — so the original "zero the donor side's alpha, then add" approach
    /// silently discarded the very channel we meant to carry over (confirmed empirically: isolating
    /// donor's channel with `take.aVector = (0,0,0,0)` always read back as (0,0,0,0)). Feeding
    /// `unpremultiplyingAlpha()`-converted images into the same matrices doesn't help either, since
    /// the *output* of `CIColorMatrix` is what gets re-premultiplied, regardless of the input's tag.
    ///
    /// Fix: never let a `CIColorMatrix` output alpha 0. Both `keep` and `take` keep their real,
    /// identical alpha (base's), so `CIAdditionCompositing` sums two valid, undistorted images —
    /// correct per-channel color, but alpha doubled (base.alpha + base.alpha). A final matrix then
    /// corrects the doubled alpha back down (×0.5) while pre-compensating the RGB by ×2 to exactly
    /// cancel that same internal re-premultiply step, so the already-correct color survives intact.
    static func replaceChannel(of base: CIImage, with donor: CIImage, channel: Int) -> CIImage {
        func vec(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ a: CGFloat) -> CIVector { CIVector(x: r, y: g, z: b, w: a) }
        let keep = CIFilter.colorMatrix(); keep.inputImage = base
        let take = CIFilter.colorMatrix(); take.inputImage = donor
        keep.rVector = vec(channel == 0 ? 0 : 1, 0, 0, 0); take.rVector = vec(channel == 0 ? 1 : 0, 0, 0, 0)
        keep.gVector = vec(0, channel == 1 ? 0 : 1, 0, 0); take.gVector = vec(0, channel == 1 ? 1 : 0, 0, 0)
        keep.bVector = vec(0, 0, channel == 2 ? 0 : 1, 0); take.bVector = vec(0, 0, channel == 2 ? 1 : 0, 0)
        keep.aVector = vec(0, 0, 0, 1); take.aVector = vec(0, 0, 0, 1)
        let add = CIFilter.additionCompositing()
        add.inputImage = take.outputImage; add.backgroundImage = keep.outputImage
        guard let summed = add.outputImage else { return base }
        let fix = CIFilter.colorMatrix(); fix.inputImage = summed
        fix.rVector = vec(2, 0, 0, 0); fix.gVector = vec(0, 2, 0, 0); fix.bVector = vec(0, 0, 2, 0)
        fix.aVector = vec(0, 0, 0, 0.5)
        return (fix.outputImage ?? base).cropped(to: base.extent)
    }

    // MARK: - Local layer and subject (completed in Task 5)

    static func applyLocalLayer(_ image: CIImage, _ layer: EditLocalLayer?) -> CIImage { image }

    static func applySubject(_ image: CIImage, _ subject: EditSubject, mask: CIImage?,
                             geometry: EditGeometry, sourceExtent: CGRect) -> CIImage { image }
}
