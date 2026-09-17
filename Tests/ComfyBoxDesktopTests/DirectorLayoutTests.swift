// DirectorLayoutTests.swift — WP3: pixel <-> frame mapping for the timeline canvas.

import Foundation
import Testing
import ZImage
@testable import ComfyBoxDesktop

@Suite("DirectorLayout")
struct DirectorLayoutTests {

    @Test("x(forFrame:) and frame(forX:) round-trip")
    func xFrameRoundTrip() {
        for zoom in [0.25, 1.0, 3.0] {
            let layout = DirectorLayout(zoom: zoom, leadingInset: 12)
            for frame in [0, 7, 96, 288, 577] {
                let x = layout.x(forFrame: frame)
                #expect(layout.frame(forX: x, length: 1000, snap: false) == frame)
            }
        }
        #expect(DirectorLayout(zoom: 2).pixelsPerFrame == 2 * DirectorLayout.basePixelsPerFrame)
    }

    @Test("frame(forX:snap:) snaps to the 8-frame grid and clamps")
    func frameForXSnapsToGrid() {
        let layout = DirectorLayout(zoom: 1)
        #expect(layout.frame(forX: layout.x(forFrame: 100), length: 289, snap: true) == 96)
        #expect(layout.frame(forX: layout.x(forFrame: 101), length: 289, snap: true) == 104)
        #expect(layout.frame(forX: -50, length: 289, snap: true) == 0)
        #expect(layout.frame(forX: layout.x(forFrame: 5000), length: 289, snap: true) == 288)
        #expect(layout.frame(forX: layout.x(forFrame: 5000), length: 289, snap: false) == 288)
    }

    @Test("boundary tick x equals the plan's chunk-1 start x")
    func boundaryTickMatchesPlanChunkStart() throws {
        var t = DirectorTimeline(settings: .init(width: 576, height: 896, lengthFrames: 577), globalPrompt: "x")
        t.keyframes = [.init(id: "k1", imagePath: "/a.png", frame: 0), .init(id: "k2", imagePath: "/b.png", frame: 400)]
        let validation = DirectorValidator.validate(t, fileExists: { _ in true }, audioProbe: { _ in nil })
        let plan = try #require(validation.plan)
        let layout = DirectorLayout(zoom: 1.5)
        let xs = layout.boundaryTickXs(plan: plan)
        #expect(xs.count == 1)
        #expect(xs[0] == layout.x(forFrame: plan.chunks[1].startFrame))
        #expect(layout.keyframeTickXs(plan: plan) == [layout.x(forFrame: 0), layout.x(forFrame: 400)])
    }

    @Test("ruler has a major labelled tick every second")
    func rulerTicksMajorEverySecond() {
        let layout = DirectorLayout(zoom: 1)
        let ticks = layout.rulerTicks(lengthFrames: 289, fps: 24)
        let majors = ticks.filter(\.isMajor)
        #expect(majors.map(\.frame) == Array(stride(from: 0, through: 288, by: 24)))
        #expect(majors.first?.label == "0:00")
        #expect(majors[4].label == "0:04")
        // Zoomed in, minor ticks appear on the 8-frame grid.
        let zoomed = DirectorLayout(zoom: 4).rulerTicks(lengthFrames: 289, fps: 24)
        let minors = zoomed.filter { !$0.isMajor }
        #expect(minors.contains { $0.frame == 8 && $0.label == "8f" })
        #expect(minors.allSatisfy { $0.frame % 8 == 0 && $0.frame % 24 != 0 })
        // Zoomed far out, no minor ticks.
        #expect(DirectorLayout(zoom: 0.1).rulerTicks(lengthFrames: 289, fps: 24).allSatisfy { $0.isMajor })
    }
}
