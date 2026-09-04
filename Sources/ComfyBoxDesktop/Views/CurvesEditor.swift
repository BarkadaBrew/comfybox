// CurvesEditor.swift — draggable tone curve for one channel at a time

import SwiftUI

struct CurvesEditor: View {
    @Binding var curves: ToneCurves
    var onCommit: () -> Void

    enum Channel: String, CaseIterable, Identifiable { case rgb = "RGB", r = "R", g = "G", b = "B"; var id: String { rawValue } }
    @State private var channel: Channel = .rgb
    @State private var dragIndex: Int?
    /// Set when a drag removes its point (dragged far outside vertically);
    /// ignored until the gesture ends so continued movement can't reacquire
    /// or insert a new point in the same drag.
    @State private var dragRemoved = false

    private var points: Binding<[CurvePoint]> {
        switch channel {
        case .rgb: return $curves.rgb
        case .r: return $curves.r
        case .g: return $curves.g
        case .b: return $curves.b
        }
    }
    private var color: Color {
        switch channel { case .rgb: return .primary; case .r: return .red; case .g: return .green; case .b: return .blue }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Picker("", selection: $channel) { ForEach(Channel.allCases) { Text($0.rawValue).tag($0) } }
                    .pickerStyle(.segmented).labelsHidden()
                Button("Reset") { points.wrappedValue = []; onCommit() }
                    .controlSize(.small).disabled(points.wrappedValue.isEmpty)
            }
            GeometryReader { geo in
                let size = min(geo.size.width, geo.size.height)
                ZStack {
                    RoundedRectangle(cornerRadius: 4).fill(Color(nsColor: .controlBackgroundColor))
                    Path { p in p.move(to: CGPoint(x: 0, y: size)); p.addLine(to: CGPoint(x: size, y: 0)) }
                        .stroke(Color.secondary.opacity(0.3), style: StrokeStyle(lineWidth: 1, dash: [4]))
                    Path { p in
                        for i in 0..<64 {
                            let x = Double(i) / 63
                            let y = ToneCurves.sample(points.wrappedValue, at: x)
                            let pt = CGPoint(x: x * size, y: (1 - y) * size)
                            if i == 0 { p.move(to: pt) } else { p.addLine(to: pt) }
                        }
                    }.stroke(color, lineWidth: 1.5)
                    ForEach(Array(points.wrappedValue.enumerated()), id: \.offset) { idx, pt in
                        Circle().fill(color).frame(width: 9, height: 9)
                            .position(x: pt.x * size, y: (1 - pt.y) * size)
                    }
                }
                .frame(width: size, height: size)
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0)
                    .onChanged { v in
                        // A drag only ever acquires an EXISTING point; it never inserts one.
                        // Insertion happens exclusively via double-click, below.
                        if dragRemoved { return }
                        if dragIndex == nil {
                            guard let idx = nearestIndex(to: v.startLocation, size: size) else { return }
                            dragIndex = idx
                        }
                        guard let i = dragIndex, points.wrappedValue.indices.contains(i) else { return }
                        let nx = min(max(v.location.x / size, 0), 1)
                        let ny = min(max(1 - v.location.y / size, 0), 1)
                        var pts = points.wrappedValue
                        let lo = i > 0 ? pts[i - 1].x + 0.01 : 0.0
                        let hi = i < pts.count - 1 ? pts[i + 1].x - 0.01 : 1.0
                        pts[i] = CurvePoint(x: min(max(nx, lo), hi), y: ny)
                        // Drag far outside vertically to remove. Once removed, further
                        // movement in this same gesture must not reacquire or insert.
                        if v.location.y < -30 || v.location.y > size + 30 {
                            pts.remove(at: i)
                            dragRemoved = true
                        }
                        points.wrappedValue = pts
                    }
                    .onEnded { _ in dragIndex = nil; dragRemoved = false; onCommit() })
                .onTapGesture(count: 2) { location in
                    // Insert only when no existing point is already near this location.
                    guard nearestIndex(to: location, size: size) == nil else { return }
                    _ = insertPoint(x: min(max(location.x / size, 0), 1), y: min(max(1 - location.y / size, 0), 1))
                    onCommit()
                }
                .frame(maxWidth: .infinity, alignment: .center)
            }
            .aspectRatio(1, contentMode: .fit)
        }
    }

    private func nearestIndex(to location: CGPoint, size: CGFloat) -> Int? {
        var best: (Int, CGFloat)?
        for (i, p) in points.wrappedValue.enumerated() {
            let d = hypot(p.x * size - location.x, (1 - p.y) * size - location.y)
            if d < 12, best == nil || d < best!.1 { best = (i, d) }
        }
        return best?.0
    }

    @discardableResult
    private func insertPoint(x: Double, y: Double) -> Int {
        var pts = points.wrappedValue
        let idx = pts.firstIndex { $0.x > x } ?? pts.count
        pts.insert(CurvePoint(x: x, y: y), at: idx)
        points.wrappedValue = pts
        return idx
    }
}
