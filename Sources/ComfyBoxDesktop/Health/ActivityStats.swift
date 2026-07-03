// ActivityStats.swift — Render-activity summary math for the Health board
//
// Pure date-bucketing logic behind the activity heatmap and stat tiles:
// day counts, active days, streaks, and peak hour, computed from asset
// creation timestamps. UI-free so it's testable without a view harness.

import Foundation

/// One heatmap cell: a calendar day and how many renders landed on it.
struct DayActivity: Identifiable, Equatable {
    let date: Date
    let count: Int
    var id: Date { date }
}

/// Headline numbers for the stat tiles.
struct ActivitySummary: Equatable {
    var totalCount: Int = 0
    var activeDays: Int = 0
    var currentStreak: Int = 0
    var longestStreak: Int = 0
    /// Hour of day (0-23) with the most renders, nil when empty.
    var peakHour: Int? = nil
}

enum ActivityStats {
    /// Summarize a set of creation timestamps.
    static func summarize(
        timestamps: [Date],
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> ActivitySummary {
        guard !timestamps.isEmpty else { return ActivitySummary() }

        var summary = ActivitySummary(totalCount: timestamps.count)

        var hourCounts: [Int: Int] = [:]
        var activeDayStarts = Set<Date>()
        for stamp in timestamps {
            hourCounts[calendar.component(.hour, from: stamp), default: 0] += 1
            activeDayStarts.insert(calendar.startOfDay(for: stamp))
        }
        summary.activeDays = activeDayStarts.count
        summary.peakHour = hourCounts.max { a, b in
            (a.value, -a.key) < (b.value, -b.key)
        }?.key

        // Streaks over the sorted distinct active days.
        let sortedDays = activeDayStarts.sorted()
        var longest = 1
        var run = 1
        for (previous, day) in zip(sortedDays, sortedDays.dropFirst()) {
            if calendar.dateComponents([.day], from: previous, to: day).day == 1 {
                run += 1
                longest = max(longest, run)
            } else {
                run = 1
            }
        }
        summary.longestStreak = longest

        // Current streak: consecutive run ending today, or still alive if the
        // last activity was yesterday.
        let today = calendar.startOfDay(for: now)
        var cursor = today
        if !activeDayStarts.contains(cursor) {
            cursor = calendar.date(byAdding: .day, value: -1, to: cursor)!
        }
        var current = 0
        while activeDayStarts.contains(cursor) {
            current += 1
            cursor = calendar.date(byAdding: .day, value: -1, to: cursor)!
        }
        summary.currentStreak = current

        return summary
    }

    /// Dense per-day counts for the last `days` days (oldest first, today last).
    static func dayCounts(
        timestamps: [Date],
        days: Int,
        now: Date = Date(),
        calendar: Calendar = .current
    ) -> [DayActivity] {
        let today = calendar.startOfDay(for: now)
        var counts: [Date: Int] = [:]
        for stamp in timestamps {
            counts[calendar.startOfDay(for: stamp), default: 0] += 1
        }
        return (0..<max(days, 0)).reversed().compactMap { offset in
            guard let day = calendar.date(byAdding: .day, value: -offset, to: today) else { return nil }
            return DayActivity(date: day, count: counts[day] ?? 0)
        }
    }
}
