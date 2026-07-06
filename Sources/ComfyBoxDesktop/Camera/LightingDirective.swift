// LightingDirective.swift — Composable lighting directives
//
// A lighting control that emits a natural-language lighting phrase appended to
// the prompt: source direction, quality/softness, and a named mood/time.
// Pure value logic so the phrasing is testable; pairs with the camera panel.

import Foundation

public struct LightingDirective: Equatable, Sendable {
    /// Where the key light comes from relative to the subject.
    public enum Direction: String, CaseIterable, Sendable {
        case none, front, left, right, back, top, under, rim
        public var phrase: String {
            switch self {
            case .none: return ""
            case .front: return "front-lit"
            case .left: return "lit from the left"
            case .right: return "lit from the right"
            case .back: return "backlit"
            case .top: return "top-lit from above"
            case .under: return "underlit from below"
            case .rim: return "rim-lit"
            }
        }
        public var label: String {
            switch self {
            case .none: return "—"
            case .front: return "Front"
            case .left: return "Left"
            case .right: return "Right"
            case .back: return "Back"
            case .top: return "Top"
            case .under: return "Under"
            case .rim: return "Rim"
            }
        }
    }

    /// Hardness / softness of the light.
    public enum Quality: String, CaseIterable, Sendable {
        case none, soft, hard, diffused, dramatic
        public var phrase: String {
            switch self {
            case .none: return ""
            case .soft: return "soft light"
            case .hard: return "hard directional light"
            case .diffused: return "soft diffused light"
            case .dramatic: return "dramatic chiaroscuro lighting"
            }
        }
        public var label: String {
            switch self {
            case .none: return "—"
            case .soft: return "Soft"
            case .hard: return "Hard"
            case .diffused: return "Diffused"
            case .dramatic: return "Dramatic"
            }
        }
    }

    /// A named lighting mood / time-of-day / source.
    public enum Mood: String, CaseIterable, Sendable {
        case none, goldenHour, blueHour, studio, natural
        case neon, candlelight, moonlight, highKey, lowKey
        public var phrase: String {
            switch self {
            case .none: return ""
            case .goldenHour: return "warm golden hour light"
            case .blueHour: return "cool blue hour light"
            case .studio: return "studio lighting"
            case .natural: return "natural light"
            case .neon: return "neon lighting"
            case .candlelight: return "warm candlelight"
            case .moonlight: return "cool moonlight"
            case .highKey: return "high-key lighting"
            case .lowKey: return "low-key lighting"
            }
        }
        public var label: String {
            switch self {
            case .none: return "—"
            case .goldenHour: return "Golden hour"
            case .blueHour: return "Blue hour"
            case .studio: return "Studio"
            case .natural: return "Natural"
            case .neon: return "Neon"
            case .candlelight: return "Candlelight"
            case .moonlight: return "Moonlight"
            case .highKey: return "High-key"
            case .lowKey: return "Low-key"
            }
        }
    }

    public var direction: Direction = .none
    public var quality: Quality = .none
    public var mood: Mood = .none

    public init(direction: Direction = .none, quality: Quality = .none, mood: Mood = .none) {
        self.direction = direction
        self.quality = quality
        self.mood = mood
    }

    /// The composed lighting phrase (empty when nothing is selected).
    public var phrase: String {
        [direction.phrase, quality.phrase, mood.phrase]
            .filter { !$0.isEmpty }
            .joined(separator: ", ")
    }

    public var isEmpty: Bool { phrase.isEmpty }

    /// Append the lighting phrase to an existing prompt (comma-joined).
    public func appended(to prompt: String) -> String {
        guard !isEmpty else { return prompt }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return phrase }
        return "\(trimmed), \(phrase)"
    }
}
