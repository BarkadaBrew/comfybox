// DirectorLayout.swift — pixel <-> frame mapping for the Director timeline
// canvas (WP3). Pure: the views do no timeline arithmetic of their own; every
// px<->frame conversion goes through here and every frame rule through
// DirectorMath (ZImage), so the ruler snaps to the same 8-frame latent grid
// the engine floors keyframes onto.

import Foundation
import ZImage

struct DirectorLayout: Equatable {

    /// Pixels per frame at zoom 1.
    static let basePixelsPerFrame: Double = 2.0
    /// Minor (8-frame grid) ticks are drawn only when they are at least this
    /// many pixels apart.
    static let minMinorTickSpacing: Double = 10

    var zoom: Double
    /// Horizontal offset of frame 0 inside the canvas.
    var leadingInset: Double = 0

    init(zoom: Double, leadingInset: Double = 0) {
        self.zoom = zoom
        self.leadingInset = leadingInset
    }

    static func pixelsPerFrame(zoom: Double) -> Double {
        basePixelsPerFrame * max(zoom, 0.01)
    }

    var pixelsPerFrame: Double { Self.pixelsPerFrame(zoom: zoom) }

    func x(forFrame frame: Int) -> Double {
        leadingInset + Double(frame) * pixelsPerFrame
    }

    /// Width of a span of frames.
    func width(forFrames frames: Int) -> Double {
        Double(frames) * pixelsPerFrame
    }

    /// Canvas width for the whole timeline.
    func contentWidth(lengthFrames: Int) -> Double {
        x(forFrame: lengthFrames) + leadingInset
    }

    /// Frame under `x`, clamped to [0, length-1]; `snap` rounds to the
    /// nearest multiple of 8 via DirectorMath.snapToGrid.
    func frame(forX x: Double, length: Int, snap: Bool) -> Int {
        let raw = Int(((x - leadingInset) / pixelsPerFrame).rounded())
        if snap { return DirectorMath.snapToGrid(frame: raw, length: length) }
        return min(max(0, raw), max(0, length - 1))
    }

    /// Frames under a horizontal drag distance (unclamped, unsnapped).
    func frames(forDeltaX dx: Double) -> Int {
        Int((dx / pixelsPerFrame).rounded())
    }

    struct Tick: Equatable, Hashable {
        let frame: Int
        let label: String
        let isMajor: Bool
    }

    /// Major tick every second ("m:ss"); minor ticks on the 8-frame grid
    /// ("96f") only when zoomed in far enough for them to be legible.
    func rulerTicks(lengthFrames: Int, fps: Int) -> [Tick] {
        let fps = max(1, fps)
        var ticks: [Tick] = []
        let showMinor = Double(DirectorMath.latentStride) * pixelsPerFrame >= Self.minMinorTickSpacing
        var frame = 0
        while frame < lengthFrames {
            if frame % fps == 0 {
                let seconds = frame / fps
                ticks.append(Tick(frame: frame, label: String(format: "%d:%02d", seconds / 60, seconds % 60), isMajor: true))
            } else if showMinor && frame % DirectorMath.latentStride == 0 {
                ticks.append(Tick(frame: frame, label: "\(frame)f", isMajor: false))
            }
            frame += 1
        }
        return ticks
    }

    /// Chunk boundaries (start_k for k > 0) from a plan.
    func boundaryTickXs(plan: DirectorPlan) -> [Double] {
        plan.boundaryFrames.map { x(forFrame: $0) }
    }

    /// Keyframe positions the plan compiled (end frames already pinned).
    func keyframeTickXs(plan: DirectorPlan) -> [Double] {
        plan.keyframeTicks.map { x(forFrame: $0.frame) }
    }
}
