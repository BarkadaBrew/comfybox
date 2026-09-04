// MaskCanvas.swift — brush overlay + gesture shared by Inpaint and Edit
//
// Draws the strokes tinted over the fitted image rect and records new
// strokes in normalized coordinates. The caller positions this view in
// the same rect as the image (see `ImageFit.rect`).

import SwiftUI

enum ImageFit {
    /// Aspect-fit rect for `imageSize` centered in `container`.
    static func rect(imageSize: CGSize, in container: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scale = min(container.width / imageSize.width, container.height / imageSize.height)
        let w = imageSize.width * scale, h = imageSize.height * scale
        return CGRect(x: (container.width - w) / 2, y: (container.height - h) / 2, width: w, height: h)
    }
}

struct MaskCanvas: View {
    let imageSize: CGSize
    @Binding var strokes: MaskStrokes
    var brushPoints: CGFloat
    var erase: Bool
    var tint: Color = .red
    var enabled: Bool = true

    @State private var current: MaskStroke?

    var body: some View {
        ZStack {
            Canvas { ctx, _ in
                for stroke in strokes.strokes + (current.map { [$0] } ?? []) {
                    var path = Path()
                    let pts = stroke.points.map { CGPoint(x: $0.x * imageSize.width, y: $0.y * imageSize.height) }
                    if pts.count == 1 { path.move(to: pts[0]); path.addLine(to: pts[0]) } else { path.addLines(pts) }
                    ctx.blendMode = stroke.erase ? .destinationOut : .normal
                    ctx.stroke(path, with: .color(stroke.erase ? .white : tint.opacity(0.45)),
                               style: StrokeStyle(lineWidth: CGFloat(stroke.size) * imageSize.width,
                                                  lineCap: .round, lineJoin: .round))
                }
            }
            .allowsHitTesting(false)
            if enabled {
                Color.clear.contentShape(Rectangle())
                    .gesture(DragGesture(minimumDistance: 0).onChanged { v in
                        let p = CGPoint(x: v.location.x / imageSize.width, y: v.location.y / imageSize.height)
                        if current == nil {
                            current = MaskStroke(points: [p], size: Double(brushPoints / imageSize.width), erase: erase)
                        } else { current?.points.append(p) }
                    }.onEnded { _ in
                        if let s = current { strokes.append(s); current = nil }
                    })
            }
        }
        .frame(width: imageSize.width, height: imageSize.height)
    }
}
