// ActivityHeatmapView.swift — GitHub-style render-activity heatmap
//
// Columns are weeks, rows are weekdays; each cell is one calendar day
// colored by render count on a single-hue sequential ramp (zero days stay
// a neutral well so "no activity" never reads as "low activity"). Hovering
// a cell shows the exact date and count; a less→more legend anchors the ramp.

import SwiftUI

struct ActivityHeatmapView: View {
    /// Dense day cells, oldest first (ActivityStats.dayCounts output).
    let days: [DayActivity]

    private static let cellSize: CGFloat = 13
    private static let cellSpacing: CGFloat = 3

    /// Single-hue sequential ramp (violet), dark-surface tuned: light = more.
    private static let ramp: [Color] = [
        Color(red: 0.24, green: 0.19, blue: 0.36),  // level 1
        Color(red: 0.37, green: 0.29, blue: 0.60),  // level 2
        Color(red: 0.53, green: 0.43, blue: 0.82),  // level 3
        Color(red: 0.70, green: 0.60, blue: 0.98),  // level 4
    ]
    private static let zeroColor = Color.primary.opacity(0.07)

    private var maxCount: Int { days.map(\.count).max() ?? 0 }

    /// Weeks as columns; the first column is padded so rows align to weekday.
    private var weeks: [[DayActivity?]] {
        guard let first = days.first else { return [] }
        let calendar = Calendar.current
        // 0-based row index for the first day (respecting the locale's first weekday).
        let weekday = calendar.component(.weekday, from: first.date)
        let leading = (weekday - calendar.firstWeekday + 7) % 7

        var cells: [DayActivity?] = Array(repeating: nil, count: leading)
        cells.append(contentsOf: days.map { Optional($0) })
        while cells.count % 7 != 0 { cells.append(nil) }

        return stride(from: 0, to: cells.count, by: 7).map {
            Array(cells[$0..<min($0 + 7, cells.count)])
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: Self.cellSpacing) {
                    ForEach(Array(weeks.enumerated()), id: \.offset) { _, week in
                        VStack(spacing: Self.cellSpacing) {
                            ForEach(Array(week.enumerated()), id: \.offset) { _, day in
                                cell(day)
                            }
                        }
                    }
                }
                .padding(2)
            }
            legend
        }
    }

    @ViewBuilder
    private func cell(_ day: DayActivity?) -> some View {
        if let day {
            RoundedRectangle(cornerRadius: 3)
                .fill(color(for: day.count))
                .frame(width: Self.cellSize, height: Self.cellSize)
                .help("\(day.count) render\(day.count == 1 ? "" : "s") — \(day.date.formatted(date: .abbreviated, time: .omitted))")
        } else {
            // Alignment filler outside the data range.
            RoundedRectangle(cornerRadius: 3)
                .fill(Color.clear)
                .frame(width: Self.cellSize, height: Self.cellSize)
        }
    }

    private var legend: some View {
        HStack(spacing: 4) {
            Spacer()
            Text("Less")
                .font(.caption2)
                .foregroundStyle(.secondary)
            RoundedRectangle(cornerRadius: 2).fill(Self.zeroColor)
                .frame(width: 10, height: 10)
            ForEach(Array(Self.ramp.enumerated()), id: \.offset) { _, color in
                RoundedRectangle(cornerRadius: 2).fill(color)
                    .frame(width: 10, height: 10)
            }
            Text("More")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    /// Quantize a count onto the ramp; zero always gets the neutral well.
    private func color(for count: Int) -> Color {
        guard count > 0, maxCount > 0 else { return Self.zeroColor }
        let level = Int((Double(count) / Double(maxCount) * Double(Self.ramp.count)).rounded(.up))
        return Self.ramp[min(max(level, 1), Self.ramp.count) - 1]
    }
}
