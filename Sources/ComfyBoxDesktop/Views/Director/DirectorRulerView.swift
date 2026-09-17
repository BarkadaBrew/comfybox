// DirectorRulerView.swift — the timeline ruler (WP3): second/8-frame ticks
// from DirectorLayout, the plan overlay (chunk boundaries + keyframe ticks),
// and click/drag to move the playhead.

import SwiftUI
import ZImage

struct DirectorRulerView: View {
    let model: DirectorDocumentModel
    let layout: DirectorLayout

    static let height: CGFloat = 30

    var body: some View {
        let length = model.lengthFrames
        let ticks = layout.rulerTicks(lengthFrames: length, fps: model.fps)
        ZStack(alignment: .topLeading) {
            Rectangle().fill(Color(nsColor: .windowBackgroundColor))

            ForEach(ticks, id: \.frame) { tick in
                Rectangle()
                    .fill(Color.secondary.opacity(tick.isMajor ? 0.8 : 0.35))
                    .frame(width: 1, height: tick.isMajor ? 12 : 6)
                    .offset(x: layout.x(forFrame: tick.frame), y: Self.height - (tick.isMajor ? 12 : 6))
                if tick.isMajor || layout.pixelsPerFrame * 8 >= 40 {
                    Text(tick.label)
                        .font(.system(size: 9).monospacedDigit())
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: true, vertical: true)
                        .offset(x: layout.x(forFrame: tick.frame) + 2, y: 2)
                }
            }

            if let plan = model.plan {
                ForEach(Array(layout.keyframeTickXs(plan: plan).enumerated()), id: \.offset) { _, x in
                    Image(systemName: "triangle.fill")
                        .font(.system(size: 7))
                        .rotationEffect(.degrees(180))
                        .foregroundStyle(Color.accentColor)
                        .offset(x: x - 4, y: Self.height - 16)
                }
                ForEach(Array(layout.boundaryTickXs(plan: plan).enumerated()), id: \.offset) { _, x in
                    Rectangle().fill(Color.orange)
                        .frame(width: 2, height: Self.height)
                        .offset(x: x - 1)
                        .help("Chunk boundary: the next chunk starts from this rendered frame")
                }
            }
        }
        .frame(width: layout.contentWidth(lengthFrames: length), height: Self.height, alignment: .topLeading)
        .contentShape(Rectangle())
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    model.playheadFrame = layout.frame(forX: value.location.x, length: length, snap: model.snapToGrid)
                }
        )
    }
}
