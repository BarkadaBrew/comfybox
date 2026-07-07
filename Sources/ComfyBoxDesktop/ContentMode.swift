// ContentMode.swift — "fruit mode" for the Generate tab. Steers prompt TEXT
// only (optimizer hint + negative additions); never guidance/numeric params.
// Raw values must match the server's ContentModeManager.Mode.

import Foundation

public enum ContentMode: String, CaseIterable, Identifiable, Sendable {
    case neutral
    case banana
    case avocado

    public var id: String { rawValue }

    /// Emoji + name for the segmented control.
    public var label: String {
        switch self {
        case .neutral: return "🍎 Neutral"
        case .banana:  return "🍌 Banana"
        case .avocado: return "🥑 Avocado"
        }
    }
}
