// CameraDirective.swift — Composable camera / shot directives
//
// A camera-placement control (v1) that emits a natural-language camera phrase
// appended to the prompt — angle, height, distance/shot size, lens, and
// orientation. Pure value logic so the phrasing is testable. Pairs with a
// character/reference for identity; v2 will add img2img/Klein conditioning so
// the same subject is re-rendered from the chosen viewpoint.

import Foundation

public struct CameraDirective: Equatable, Sendable {
    public enum Angle: String, CaseIterable, Sendable {
        case eyeLevel, lowAngle, highAngle, overhead, dutch
        public var phrase: String {
            switch self {
            case .eyeLevel: return "eye-level angle"
            case .lowAngle: return "low-angle shot looking up"
            case .highAngle: return "high-angle shot looking down"
            case .overhead: return "overhead top-down shot"
            case .dutch: return "dutch tilt"
            }
        }
        public var label: String {
            switch self {
            case .eyeLevel: return "Eye level"
            case .lowAngle: return "Low angle"
            case .highAngle: return "High angle"
            case .overhead: return "Overhead"
            case .dutch: return "Dutch tilt"
            }
        }
    }

    /// Horizontal camera position around the subject.
    public enum Orientation: String, CaseIterable, Sendable {
        case front, threeQuarter, profile, threeQuarterBack, back
        public var phrase: String {
            switch self {
            case .front: return "front view"
            case .threeQuarter: return "three-quarter view"
            case .profile: return "profile side view"
            case .threeQuarterBack: return "three-quarter rear view"
            case .back: return "rear view from behind"
            }
        }
        public var label: String {
            switch self {
            case .front: return "Front"
            case .threeQuarter: return "3/4"
            case .profile: return "Profile"
            case .threeQuarterBack: return "3/4 back"
            case .back: return "Back"
            }
        }
    }

    /// Shot size / camera distance.
    public enum ShotSize: String, CaseIterable, Sendable {
        case extremeCloseUp, closeUp, medium, full, wide
        public var phrase: String {
            switch self {
            case .extremeCloseUp: return "extreme close-up"
            case .closeUp: return "close-up shot"
            case .medium: return "medium shot"
            case .full: return "full-body shot"
            case .wide: return "wide establishing shot"
            }
        }
        public var label: String {
            switch self {
            case .extremeCloseUp: return "ECU"
            case .closeUp: return "Close-up"
            case .medium: return "Medium"
            case .full: return "Full body"
            case .wide: return "Wide"
            }
        }
    }

    /// Focal length; also implies depth-of-field feel.
    public enum Lens: String, CaseIterable, Sendable {
        case none, mm24, mm35, mm50, mm85, mm135
        public var phrase: String {
            switch self {
            case .none: return ""
            case .mm24: return "24mm wide lens"
            case .mm35: return "35mm lens"
            case .mm50: return "50mm lens"
            case .mm85: return "85mm portrait lens, shallow depth of field"
            case .mm135: return "135mm telephoto, compressed background"
            }
        }
        public var label: String {
            switch self {
            case .none: return "—"
            case .mm24: return "24mm"
            case .mm35: return "35mm"
            case .mm50: return "50mm"
            case .mm85: return "85mm"
            case .mm135: return "135mm"
            }
        }
    }

    public var orientation: Orientation
    public var angle: Angle
    public var shotSize: ShotSize
    public var lens: Lens

    public init(
        orientation: Orientation = .front,
        angle: Angle = .eyeLevel,
        shotSize: ShotSize = .medium,
        lens: Lens = .none
    ) {
        self.orientation = orientation
        self.angle = angle
        self.shotSize = shotSize
        self.lens = lens
    }

    /// The composed camera phrase, e.g.
    /// "medium shot, three-quarter view, low-angle shot looking up, 85mm …".
    public var phrase: String {
        var parts = [shotSize.phrase, orientation.phrase]
        if angle != .eyeLevel { parts.append(angle.phrase) }
        if lens != .none { parts.append(lens.phrase) }
        return parts.filter { !$0.isEmpty }.joined(separator: ", ")
    }

    /// Append the directive to an existing prompt (comma-joined), de-duplicating
    /// a trailing camera phrase if the user re-inserts.
    public func appended(to prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return phrase }
        return "\(trimmed), \(phrase)"
    }
}
