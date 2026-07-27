// AppContentGate.swift — app-wide "Rated G unless NSFW is toggled" gate.
//
// Todd 2026-07-17: the desktop app must present as Rated G by default; every
// mature-theme surface (Gallery, Recents, Prompts, …) stays hidden until a
// single global NSFW toggle is turned on. This is the one source of truth for
// that toggle.
//
// SAFETY POSTURE: `revealed` starts FALSE on every launch and is deliberately
// NOT persisted — a fresh launch is always G-rated. Revealing requires the
// gallery password when one is configured (reuses NSFWGate), so the toggle
// isn't a one-click bypass of an intentional lock.

import SwiftUI

@Observable
@MainActor
public final class AppContentGate {
    /// Whether mature content is revealed this session. Session-only by design.
    public private(set) var revealed = false

    public init() {}

    /// True when a gallery password is set — revealing then requires it.
    public var requiresPassword: Bool { NSFWGate.isConfigured }

    /// Reveal without a password (only valid when none is configured).
    public func reveal() {
        guard !requiresPassword else { return }
        revealed = true
    }

    /// Reveal by verifying the gallery password. Returns success.
    @discardableResult
    public func reveal(withPassword password: String) -> Bool {
        guard NSFWGate.verify(password) else { return false }
        revealed = true
        return true
    }

    /// Re-hide (always allowed — tightening is never gated).
    public func hide() { revealed = false }
}

// MARK: - Gating modifiers (all read the gate from the environment)

public extension View {
    /// Blur + lock this content while the app is G-rated (hidden). Interaction
    /// is disabled until revealed, so gated imagery can't be opened/clicked.
    func contentGated(cornerRadius: CGFloat = 6) -> some View {
        modifier(ContentGateModifier(cornerRadius: cornerRadius))
    }
}

// Explicitly MainActor (Kimi review 2026-07-27): body already runs on the main
// actor via SwiftUI, but the annotation makes the @Environment(AppContentGate)
// read provably safe when the package moves to Swift 6 language mode.
@MainActor
struct ContentGateModifier: ViewModifier {
    @Environment(AppContentGate.self) private var gate
    let cornerRadius: CGFloat

    func body(content: Content) -> some View {
        content
            .blur(radius: gate.revealed ? 0 : 18)
            .overlay {
                if !gate.revealed {
                    RoundedRectangle(cornerRadius: cornerRadius)
                        .fill(.ultraThinMaterial)
                        .overlay {
                            Image(systemName: "eye.slash.fill")
                                .foregroundStyle(.secondary)
                                .font(.caption)
                        }
                }
            }
            .allowsHitTesting(gate.revealed)
            .accessibilityHidden(!gate.revealed)
    }
}

/// Redact prompt/caption text while the app is G-rated. Titles/filenames that
/// might carry explicit phrasing route through this so nothing mature shows in
/// a screenshot-safe glance.
public struct GatedText: View {
    @Environment(AppContentGate.self) private var gate
    private let text: String
    private let font: Font?

    public init(_ text: String, font: Font? = nil) {
        self.text = text
        self.font = font
    }

    public var body: some View {
        Text(gate.revealed ? text : String(repeating: "•", count: min(max(text.count, 4), 24)))
            .font(font)
            .redacted(reason: gate.revealed ? [] : .placeholder)
    }
}

/// A neutral "locked" placeholder for gated browsing views (Gallery, etc.).
///
/// By design it presents NO reveal affordance — a visible "Show NSFW" button
/// would advertise hidden content to anyone who opens the app (Todd
/// 2026-07-18). Revealing is a hidden keyboard shortcut only, handled at the
/// app level. This just shows a quiet lock so nothing mature is on screen.
@MainActor
public struct ContentHiddenWall: View {
    private let note: String

    public init(note: String = "") {
        self.note = note
    }

    public var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "lock.fill")
                .font(.system(size: 30))
                .foregroundStyle(.tertiary)
            Text("Locked")
                .font(.headline)
                .foregroundStyle(.secondary)
            // The note was accepted-but-dropped (Kimi review 2026-07-27).
            // Rendered quietly: callers pass neutral copy, never a content tease.
            if !note.isEmpty {
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
        .accessibilityHidden(true)
    }
}
