// KiraPresetCatalog.swift — canonical preset choices for Kira Creation

import Foundation

/// Splits the engine's canonical preset inventory for the Kira Creation
/// controls. Invalid presets stay visible in Presets for repair, but must not
/// be offered for scheduled production renders.
enum KiraPresetCatalog {
    struct Choices: Equatable {
        var images: [ServerPreset]
        var videos: [ServerPreset]
    }

    static func choices(from presets: [ServerPreset]) -> Choices {
        let valid = presets.filter { $0.invalid != true }
        return Choices(
            images: valid
                .filter { preset in
                    preset.mediaKind == "image"
                        || (preset.mediaKind == nil
                            && !preset.id.localizedCaseInsensitiveContains("video"))
                }
                .sorted(by: displayOrder),
            videos: valid
                .filter { preset in
                    preset.mediaKind == "video"
                        || (preset.mediaKind == nil
                            && preset.id.localizedCaseInsensitiveContains("video"))
                }
                .sorted(by: displayOrder))
    }

    static func displayLabel(for preset: ServerPreset) -> String {
        let name = preset.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name != preset.id else { return preset.id }
        return "\(name) · \(preset.id)"
    }

    private static func displayOrder(_ lhs: ServerPreset, _ rhs: ServerPreset) -> Bool {
        let lhsName = lhs.name.lowercased()
        let rhsName = rhs.name.lowercased()
        return lhsName == rhsName ? lhs.id < rhs.id : lhsName < rhsName
    }
}
