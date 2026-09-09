// ImageGenerationEngine.swift — Image-engine selection shared by the
// Generate UI and its HTTP request builder.

import Foundation

/// The two local image paths exposed by ComfyBox Desktop.
///
/// `active` preserves the existing warm image-model behavior. `ltx2` uses
/// the server's configured LTX-2.3 transformer/VAE as a native one-frame
/// text-to-image pipeline; it never selects or loads another image model.
public enum ImageGenerationEngine: String, CaseIterable, Identifiable, Sendable {
    case active = "default"
    case ltx2 = "ltx2"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .active: return "Active Model"
        case .ltx2: return "LTX-2.3"
        }
    }

    public var summaryLabel: String {
        switch self {
        case .active: return "Active image model"
        case .ltx2: return "LTX-2.3 native image"
        }
    }

    /// Accept the server's additive aliases when restoring metadata/presets,
    /// while persisting one canonical UI value.
    public init(serverValue: String?) {
        switch serverValue?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "ltx2", "ltx-2", "ltx2-image": self = .ltx2
        default: self = .active
        }
    }

    /// Local validation for controls the LTX route can represent. The server
    /// remains authoritative and repeats these checks before enqueueing.
    public func validationError(width: Int, height: Int, steps: Int, guidance: Float) -> String? {
        guard self == .ltx2 else { return nil }
        guard width > 0, height > 0, width.isMultiple(of: 32), height.isMultiple(of: 32) else {
            return "LTX-2.3 image width and height must be positive multiples of 32."
        }
        guard (1...100).contains(steps) else {
            return "LTX-2.3 image steps must be between 1 and 100."
        }
        guard guidance.isFinite, guidance >= 1 else {
            return "LTX-2.3 image guidance must be at least 1."
        }
        return nil
    }
}
