// MaskStrokes.swift — brush mask model shared by Inpaint and the Edit tab
//
// Points are normalized (0…1) in the displayed image rect with a top-left
// origin; `size` is a fraction of the image width. The rasterizer flips Y
// because CGContext bitmaps are bottom-up.

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public struct MaskStroke: Codable, Equatable, Sendable, Identifiable {
    public var id: UUID
    public var points: [CGPoint]
    public var size: Double
    public var erase: Bool

    public init(points: [CGPoint], size: Double, erase: Bool) {
        self.id = UUID(); self.points = points; self.size = size; self.erase = erase
    }
}

public struct MaskStrokes: Codable, Equatable, Sendable {
    public var strokes: [MaskStroke]
    public init(strokes: [MaskStroke] = []) { self.strokes = strokes }
    public var isEmpty: Bool { strokes.isEmpty }
    public mutating func append(_ s: MaskStroke) { strokes.append(s) }
    public mutating func undoLast() { if !strokes.isEmpty { strokes.removeLast() } }
    public mutating func clear() { strokes.removeAll() }
}

public enum MaskRasterizer {
    public static func render(_ strokes: MaskStrokes, size: CGSize) -> CGImage? {
        let w = Int(size.width.rounded()), h = Int(size.height.rounded())
        guard w > 0, h > 0 else { return nil }
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.setFillColor(gray: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.setLineCap(.round); ctx.setLineJoin(.round)
        for stroke in strokes.strokes {
            guard let first = stroke.points.first else { continue }
            ctx.setStrokeColor(gray: stroke.erase ? 0 : 1, alpha: 1)
            ctx.setLineWidth(CGFloat(stroke.size) * CGFloat(w))
            ctx.beginPath()
            ctx.move(to: CGPoint(x: first.x * CGFloat(w), y: CGFloat(h) - first.y * CGFloat(h)))
            if stroke.points.count == 1 {
                // A dot: zero-length line still draws a round cap.
                ctx.addLine(to: CGPoint(x: first.x * CGFloat(w), y: CGFloat(h) - first.y * CGFloat(h)))
            }
            for p in stroke.points.dropFirst() {
                ctx.addLine(to: CGPoint(x: p.x * CGFloat(w), y: CGFloat(h) - p.y * CGFloat(h)))
            }
            ctx.strokePath()
        }
        return ctx.makeImage()
    }

    public static func pngData(_ strokes: MaskStrokes, size: CGSize) -> Data? {
        guard let img = render(strokes, size: size) else { return nil }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }
}
