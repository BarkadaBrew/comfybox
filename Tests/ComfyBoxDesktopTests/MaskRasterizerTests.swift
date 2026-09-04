// MaskRasterizerTests.swift
import Testing
import Foundation
import CoreGraphics
import ImageIO
import AppKit
@testable import ComfyBoxDesktop

@Suite("MaskRasterizer")
struct MaskRasterizerTests {
    @Test("single stroke paints white at its points and black elsewhere")
    func paintsStroke() {
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 0.2, erase: false))
        let img = MaskRasterizer.render(strokes, size: CGSize(width: 100, height: 100))!
        #expect(img.width == 100 && img.height == 100)
        #expect(EditTestSupport.gray(img, x: 50, y: 50) > 200)
        #expect(EditTestSupport.gray(img, x: 5, y: 5) < 20)
    }

    @Test("erase stroke clears painted area")
    func eraseClears() {
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 0.3, erase: false))
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 0.1, erase: true))
        let img = MaskRasterizer.render(strokes, size: CGSize(width: 100, height: 100))!
        #expect(EditTestSupport.gray(img, x: 50, y: 50) < 20)
        #expect(EditTestSupport.gray(img, x: 50, y: 40) > 200)   // ring outside the erase still painted
    }

    @Test("y axis is top-down: a stroke at y=0.1 lands near the top row")
    func orientation() {
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.1)], size: 0.1, erase: false))
        let img = MaskRasterizer.render(strokes, size: CGSize(width: 100, height: 100))!
        #expect(EditTestSupport.gray(img, x: 50, y: 10) > 200)
        #expect(EditTestSupport.gray(img, x: 50, y: 90) < 20)
    }

    @Test("empty strokes render all black; zero size returns nil")
    func emptyAndZero() {
        let img = MaskRasterizer.render(MaskStrokes(), size: CGSize(width: 10, height: 10))!
        #expect(EditTestSupport.gray(img, x: 5, y: 5) == 0)
        #expect(MaskRasterizer.render(MaskStrokes(), size: .zero) == nil)
    }

    // MARK: - Fix wave (M1, M16)

    @Test("a non-finite size returns nil instead of trapping")
    func nonFiniteSizeReturnsNil() {
        #expect(MaskRasterizer.render(MaskStrokes(), size: CGSize(width: CGFloat.infinity, height: 10)) == nil)
        #expect(MaskRasterizer.render(MaskStrokes(), size: CGSize(width: 10, height: CGFloat.infinity)) == nil)
        #expect(MaskRasterizer.render(MaskStrokes(), size: CGSize(width: CGFloat.nan, height: 10)) == nil)
    }

    @Test("pngData round-trips through PNG encode/decode to the same pixels as render")
    func pngDataRoundTrips() throws {
        var strokes = MaskStrokes()
        strokes.append(MaskStroke(points: [CGPoint(x: 0.5, y: 0.5)], size: 0.2, erase: false))
        let size = CGSize(width: 40, height: 40)
        let direct = MaskRasterizer.render(strokes, size: size)!
        let data = try #require(MaskRasterizer.pngData(strokes, size: size))
        let src = try #require(CGImageSourceCreateWithData(data as CFData, nil))
        let decoded = try #require(CGImageSourceCreateImageAtIndex(src, 0, nil))
        #expect(decoded.width == direct.width && decoded.height == direct.height)
        for (x, y) in [(20, 20), (2, 2), (38, 38)] {
            #expect(EditTestSupport.gray(decoded, x: x, y: y) == EditTestSupport.gray(direct, x: x, y: y))
        }
        #expect(EditTestSupport.gray(decoded, x: 20, y: 20) > 200)   // stroke centre
        #expect(EditTestSupport.gray(decoded, x: 2, y: 2) < 20)      // untouched corner
    }

    @Test("MaskStrokes round-trips through JSON and undoLast/clear work")
    func modelBasics() throws {
        var s = MaskStrokes()
        #expect(s.isEmpty)
        s.append(MaskStroke(points: [CGPoint(x: 0.1, y: 0.2)], size: 0.05, erase: false))
        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(MaskStrokes.self, from: data)
        #expect(back == s)
        s.undoLast(); #expect(s.isEmpty)
        s.append(MaskStroke(points: [], size: 0.05, erase: true)); s.clear(); #expect(s.isEmpty)
    }

    /// The pre-refactor Inpaint rasterizer, kept verbatim as the oracle.
    private func legacyMaskPNG(_ strokes: MaskStrokes, pixelSize: CGSize) -> Data? {
        let W = Int(pixelSize.width), H = Int(pixelSize.height)
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: W, pixelsHigh: H, bitsPerSample: 8,
            samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else { return nil }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.black.setFill(); NSBezierPath(rect: NSRect(x: 0, y: 0, width: W, height: H)).fill()
        for stroke in strokes.strokes {
            (stroke.erase ? NSColor.black : NSColor.white).setStroke()
            let path = NSBezierPath()
            path.lineWidth = CGFloat(stroke.size) * pixelSize.width
            path.lineCapStyle = .round; path.lineJoinStyle = .round
            for (i, pt) in stroke.points.enumerated() {
                let x = pt.x * pixelSize.width
                let y = pixelSize.height - pt.y * pixelSize.height
                if i == 0 { path.move(to: NSPoint(x: x, y: y)) } else { path.line(to: NSPoint(x: x, y: y)) }
            }
            path.stroke()
        }
        NSGraphicsContext.restoreGraphicsState()
        return rep.representation(using: .png, properties: [:])
    }

    @Test("matches the legacy Inpaint rasterizer on a sampled grid")
    func legacyParity() {
        var s = MaskStrokes()
        s.append(MaskStroke(points: [CGPoint(x: 0.2, y: 0.3), CGPoint(x: 0.7, y: 0.6)], size: 0.08, erase: false))
        // A short line (not a single point) so the legacy oracle — which only strokes
        // a lineto, not a lone moveto — actually draws the erase mark too.
        s.append(MaskStroke(points: [CGPoint(x: 0.45, y: 0.5), CGPoint(x: 0.55, y: 0.5)], size: 0.12, erase: true))
        let size = CGSize(width: 120, height: 80)
        let new = MaskRasterizer.render(s, size: size)!
        let legacy = NSBitmapImageRep(data: legacyMaskPNG(s, pixelSize: size)!)!.cgImage!
        var mismatches = 0
        for y in stride(from: 2, to: 80, by: 4) {
            for x in stride(from: 2, to: 120, by: 4) {
                let a = EditTestSupport.gray(new, x: x, y: y) > 127
                let b = EditTestSupport.gray(legacy, x: x, y: y) > 127
                if a != b { mismatches += 1 }
            }
        }
        #expect(mismatches <= 4)   // anti-aliasing at stroke edges only
        // Direct check at the erase centre: both rasterizers must agree the paint was erased.
        let cx = Int(0.5 * size.width), cy = Int(0.5 * size.height)
        #expect(EditTestSupport.gray(new, x: cx, y: cy) < 127)
        #expect(EditTestSupport.gray(legacy, x: cx, y: cy) < 127)
    }
}
