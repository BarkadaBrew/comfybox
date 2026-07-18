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

/// A full-surface "content hidden" wall for browsing views (Gallery, etc.)
/// where the requirement is that nothing shows until revealed. Owns its own
/// reveal flow (including the password prompt) so any view can drop it in.
public struct ContentHiddenWall: View {
    @Environment(AppContentGate.self) private var gate
    @State private var showPassword = false
    @State private var password = ""
    @State private var passwordError = false
    private let note: String

    public init(note: String = "This view may contain mature content.") {
        self.note = note
    }

    public var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "eye.slash.fill")
                .font(.system(size: 34))
                .foregroundStyle(.secondary)
            Text("Content hidden")
                .font(.headline)
            Text("\(note) Turn on NSFW to view.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button {
                if gate.requiresPassword {
                    password = ""
                    passwordError = false
                    showPassword = true
                } else {
                    gate.reveal()
                }
            } label: {
                Label(gate.requiresPassword ? "Unlock NSFW…" : "Show NSFW", systemImage: "eye")
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
        .sheet(isPresented: $showPassword) {
            VStack(alignment: .leading, spacing: 12) {
                Label("Reveal mature content", systemImage: "eye.trianglebadge.exclamationmark")
                    .font(.headline)
                SecureField("Gallery password", text: $password)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit(unlock)
                if passwordError {
                    Text("Incorrect password").font(.caption).foregroundStyle(.red)
                }
                HStack {
                    Spacer()
                    Button("Cancel") { showPassword = false }
                    Button("Reveal") { unlock() }.buttonStyle(.borderedProminent)
                }
            }
            .padding(20)
            .frame(width: 360)
        }
    }

    private func unlock() {
        if gate.reveal(withPassword: password) {
            showPassword = false
        } else {
            passwordError = true
        }
    }
}
