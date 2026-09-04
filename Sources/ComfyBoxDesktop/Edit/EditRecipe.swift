// EditRecipe.swift — value model for a non-destructive edit
//
// Every field has a neutral default so `EditRecipe()` renders the source
// unchanged. Ranges are documented per field; the renderer (EditRenderer)
// owns the mapping from these unit ranges to Core Image parameters.

import Foundation
import CoreGraphics

public struct CurvePoint: Codable, Equatable, Sendable {
    public var x: Double
    public var y: Double
    public init(x: Double, y: Double) { self.x = x; self.y = y }
}

public struct ToneCurves: Codable, Equatable, Sendable {
    public var rgb: [CurvePoint] = []
    public var r: [CurvePoint] = []
    public var g: [CurvePoint] = []
    public var b: [CurvePoint] = []
    public init() {}

    public var isIdentity: Bool { rgb.isEmpty && r.isEmpty && g.isEmpty && b.isEmpty }

    public static func normalized(_ pts: [CurvePoint]) -> [CurvePoint] {
        var out = pts.map { CurvePoint(x: min(max($0.x, 0), 1), y: min(max($0.y, 0), 1)) }
            .sorted { $0.x < $1.x }
        if out.first?.x != 0 { out.insert(CurvePoint(x: 0, y: 0), at: 0) }
        if out.last?.x != 1 { out.append(CurvePoint(x: 1, y: 1)) }
        return out
    }

    /// Fritsch–Carlson monotone cubic interpolation through the normalized points.
    public static func sample(_ pts: [CurvePoint], at x: Double) -> Double {
        let p = normalized(pts)
        let n = p.count
        if n == 2 && p[0] == CurvePoint(x: 0, y: 0) && p[1] == CurvePoint(x: 1, y: 1) { return min(max(x, 0), 1) }
        let xs = p.map(\.x), ys = p.map(\.y)
        var d = [Double](repeating: 0, count: n - 1)
        for i in 0..<(n - 1) { let h = xs[i + 1] - xs[i]; d[i] = h > 0 ? (ys[i + 1] - ys[i]) / h : 0 }
        var m = [Double](repeating: 0, count: n)
        m[0] = d[0]; m[n - 1] = d[n - 2]
        for i in 1..<(n - 1) { m[i] = (d[i - 1] * d[i] <= 0) ? 0 : (d[i - 1] + d[i]) / 2 }
        for i in 0..<(n - 1) where d[i] != 0 {
            let a = m[i] / d[i], b = m[i + 1] / d[i]
            let s = a * a + b * b
            if s > 9 { let t = 3 / s.squareRoot(); m[i] = t * a * d[i]; m[i + 1] = t * b * d[i] }
        }
        let cx = min(max(x, 0), 1)
        var i = 0
        while i < n - 2 && cx > xs[i + 1] { i += 1 }
        let h = xs[i + 1] - xs[i]
        guard h > 0 else { return ys[i] }
        let t = (cx - xs[i]) / h
        let t2 = t * t, t3 = t2 * t
        let h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2
        let y = h00 * ys[i] + h10 * h * m[i] + h01 * ys[i + 1] + h11 * h * m[i + 1]
        return min(max(y, 0), 1)
    }
}

public struct EditAdjustments: Codable, Equatable, Sendable {
    public var exposure = 0.0        // EV, −5…5
    public var contrast = 0.0        // −1…1
    public var highlights = 0.0      // −1…1
    public var shadows = 0.0         // −1…1
    public var whites = 0.0          // −1…1
    public var blacks = 0.0          // −1…1
    public var temperature = 0.0     // −1…1 (cool…warm)
    public var tint = 0.0            // −1…1 (green…magenta)
    public var vibrance = 0.0        // −1…1
    public var saturation = 0.0      // −1…1
    public var sharpen = 0.0         // 0…1
    public var noiseReduction = 0.0  // 0…1
    public var vignette = 0.0        // 0…1
    public var curves = ToneCurves()
    public init() {}

    public var isIdentity: Bool { self == EditAdjustments() }

    public var restrictedToLocal: EditAdjustments {
        var r = EditAdjustments()
        r.exposure = exposure; r.contrast = contrast; r.highlights = highlights; r.shadows = shadows
        r.temperature = temperature; r.tint = tint; r.saturation = saturation; r.sharpen = sharpen
        return r
    }
}

public struct EditGeometry: Codable, Equatable, Sendable {
    /// Normalized crop in source coordinates, origin top-left. nil = full frame.
    public var crop: CGRect? = nil
    public var straightenDegrees = 0.0   // −45…45
    public var quarterTurns = 0          // 0…3 clockwise
    public var flipH = false
    public var flipV = false
    public init() {}
    public var isIdentity: Bool { self == EditGeometry() }
}

public struct EditLocalLayer: Codable, Equatable, Sendable {
    public var mask = MaskStrokes()
    public var feather = 0.0             // 0…1 → blur radius up to 5 % of the shorter side
    public var adjustments = EditAdjustments()
    public init() {}
}

public struct EditSubject: Codable, Equatable, Sendable {
    public var removeBackground = false
    public var invert = false
    public init(removeBackground: Bool = false, invert: Bool = false) {
        self.removeBackground = removeBackground; self.invert = invert
    }
}

public struct EditRecipe: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public var version = EditRecipe.currentVersion
    public var geometry = EditGeometry()
    public var adjustments = EditAdjustments()
    public var local: EditLocalLayer? = nil
    public var subject = EditSubject()
    public init() {}

    public var isIdentity: Bool {
        geometry.isIdentity && adjustments.isIdentity && local == nil && subject == EditSubject()
    }
}
