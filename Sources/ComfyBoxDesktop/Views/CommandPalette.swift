// CommandPalette.swift — ⌘K fuzzy launcher across the hub
//
// Jump to any tab or run an action (connect, restart the daemon, reload Bree,
// new canvas…) without hunting through 15 tabs. Fuzzy scoring is pure and
// unit-tested; the view is a focused overlay with arrow/enter/escape keys.

import SwiftUI

struct PaletteCommand: Identifiable {
    let id = UUID()
    let title: String
    var subtitle: String?
    let systemImage: String
    var keywords: [String] = []
    let action: () -> Void

    /// The text fuzzy matching runs against.
    var haystack: String { ([title, subtitle ?? ""] + keywords).joined(separator: " ") }
}

enum FuzzyMatcher {
    /// Subsequence score: nil = no match. Higher is better. Rewards contiguous
    /// runs, word-boundary starts, and an early first match. Case-insensitive.
    static func score(_ query: String, _ text: String) -> Int? {
        let q = Array(query.lowercased())
        if q.isEmpty { return 0 }
        let t = Array(text.lowercased())
        var qi = 0, score = 0, runLength = 0, firstIndex: Int? = nil
        var ti = 0
        while ti < t.count && qi < q.count {
            if t[ti] == q[qi] {
                if firstIndex == nil { firstIndex = ti }
                runLength += 1
                score += 1 + runLength          // contiguous runs compound
                let prev = ti > 0 ? t[ti - 1] : " "
                if prev == " " || prev == "-" || prev == "/" { score += 3 }  // word start
                qi += 1
            } else {
                runLength = 0
            }
            ti += 1
        }
        guard qi == q.count else { return nil }  // all query chars matched
        if let f = firstIndex { score += max(0, 5 - f) }  // earlier is better
        return score
    }

    /// Filter + rank commands for a query (empty query keeps original order).
    static func rank(_ commands: [PaletteCommand], query: String) -> [PaletteCommand] {
        let trimmed = query.trimmingCharacters(in: .whitespaces)
        if trimmed.isEmpty { return commands }
        return commands
            .compactMap { cmd -> (PaletteCommand, Int)? in
                score(trimmed, cmd.haystack).map { (cmd, $0) }
            }
            .sorted { $0.1 > $1.1 }
            .map { $0.0 }
    }
}

struct CommandPaletteView: View {
    @Binding var isPresented: Bool
    let commands: [PaletteCommand]

    @State private var query = ""
    @State private var selection = 0
    @FocusState private var fieldFocused: Bool

    private var results: [PaletteCommand] { FuzzyMatcher.rank(commands, query: query) }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(.secondary)
                TextField("Jump to… or run a command", text: $query)
                    .textFieldStyle(.plain)
                    .font(.title3)
                    .focused($fieldFocused)
                    .onSubmitOrKeys(
                        onSubmit: { runSelected() },
                        onUp: { move(-1) }, onDown: { move(1) },
                        onEscape: { isPresented = false })
                    .onChange(of: query) { _, _ in selection = 0 }
            }
            .padding(14)
            Divider()
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(Array(results.prefix(40).enumerated()), id: \.element.id) { index, cmd in
                            row(cmd, index: index).id(index)
                                .onTapGesture { run(cmd) }
                        }
                    }
                }
                .frame(maxHeight: 340)
                .onChange(of: selection) { _, s in withAnimation { proxy.scrollTo(s, anchor: .center) } }
            }
        }
        .frame(width: 560)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .onAppear { fieldFocused = true }
    }

    private func row(_ cmd: PaletteCommand, index: Int) -> some View {
        HStack(spacing: 10) {
            Image(systemName: cmd.systemImage).frame(width: 20).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(cmd.title).font(.body)
                if let sub = cmd.subtitle { Text(sub).font(.caption2).foregroundStyle(.secondary) }
            }
            Spacer()
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
        .background(index == selection ? Color.accentColor.opacity(0.2) : .clear)
        .contentShape(Rectangle())
    }

    private func move(_ delta: Int) {
        let count = min(results.count, 40)
        guard count > 0 else { return }
        selection = (selection + delta + count) % count
    }

    private func runSelected() {
        guard results.indices.contains(selection) else { return }
        run(results[selection])
    }

    private func run(_ cmd: PaletteCommand) {
        isPresented = false
        cmd.action()
    }
}

/// Attaches submit + arrow/escape handling to a text field.
private extension View {
    func onSubmitOrKeys(onSubmit: @escaping () -> Void, onUp: @escaping () -> Void,
                        onDown: @escaping () -> Void, onEscape: @escaping () -> Void) -> some View {
        self.onSubmit(onSubmit)
            .onKeyPress(.upArrow) { onUp(); return .handled }
            .onKeyPress(.downArrow) { onDown(); return .handled }
            .onKeyPress(.escape) { onEscape(); return .handled }
    }
}
