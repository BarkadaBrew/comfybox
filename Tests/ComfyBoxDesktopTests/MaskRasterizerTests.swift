// MaskRasterizerTests.swift
import Testing
import Foundation
import CoreGraphics
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
}
