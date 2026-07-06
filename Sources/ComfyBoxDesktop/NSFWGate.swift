// NSFWGate.swift — NSFW classification + a gallery password gate
//
// Distinct from per-asset "secured" (Touch-ID) assets: this is a content-based
// filter/blur over the whole gallery driven by an asset's content mode, unlocked
// for the session by a user-set gallery password. The password is never stored —
// only its salted SHA-256 hash, in the Keychain.

import Foundation
import CryptoKit

/// Content classification from an asset's content mode.
public enum ContentRating {
    /// Content modes considered NSFW. `apple`/nil are SFW; `banana`
    /// (suggestive) and `avocado` (explicit) are NSFW.
    public static let nsfwModes: Set<String> = ["banana", "avocado", "nsfw", "explicit", "suggestive"]

    public static func isNSFW(contentMode: String?) -> Bool {
        guard let m = contentMode?.lowercased(), !m.isEmpty else { return false }
        return nsfwModes.contains(m)
    }
}

public extension DAMAsset {
    var isNSFW: Bool { ContentRating.isNSFW(contentMode: contentMode) }
}

/// How the gallery treats NSFW assets.
public enum NSFWFilterMode: String, CaseIterable, Identifiable, Sendable {
    case show = "Show All"       // no gating
    case blur = "Blur NSFW"      // visible but blurred until unlocked
    case hide = "Hide NSFW"      // excluded entirely
    public var id: String { rawValue }
    public var symbol: String {
        switch self { case .show: return "eye"; case .blur: return "eye.trianglebadge.exclamationmark"
        case .hide: return "eye.slash" }
    }
}

/// A password gate whose hash lives in the Keychain. No password stored.
public enum NSFWGate {
    private static let account = "nsfw_gate_hash"
    private static let salt = "comfybox.nsfw.v1"

    public static func hash(_ password: String) -> String {
        let data = Data((salt + password).utf8)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    public static var isConfigured: Bool {
        (Keychain.get(account) ?? "").isEmpty == false
    }

    public static func setPassword(_ password: String?) {
        guard let password, !password.isEmpty else { Keychain.set(nil, account); return }
        Keychain.set(hash(password), account)
    }

    public static func verify(_ password: String) -> Bool {
        guard let stored = Keychain.get(account), !stored.isEmpty else {
            return true   // no password set → nothing to gate against
        }
        return stored == hash(password)
    }
}
