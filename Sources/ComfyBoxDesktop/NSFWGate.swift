// NSFWGate.swift — NSFW classification + a gallery password gate
//
// Distinct from per-asset "secured" (Touch-ID) assets: this is a content-based
// filter/blur over the whole gallery driven by an asset's content mode, unlocked
// for the session by a user-set gallery password. The password is never stored —
// only its salted SHA-256 hash, in the Keychain.

import Foundation
import CryptoKit
import ComfyBoxCatalog

/// Content classification from an asset's content mode.
///
/// The vocabulary is NOT restated here. It used to be — a literal set beside the
/// catalog's own tier tables — and the two copies failed in OPPOSITE directions
/// for the same unknown string: the catalog's `tierRank` fails closed (an
/// unrecognised tier outranks every ceiling and is withheld), while a
/// `Set.contains` membership test fails OPEN (an unrecognised mode is not in the
/// set, so it was shown unblurred). Two lists that disagree about the dangerous
/// case are worse than one list, so the gate now asks the catalog.
public enum ContentRating {
    /// The spellings this gate RECOGNISES as NSFW, derived from the catalog's
    /// ladder: every tier vocabulary word that sits above the strictest ceiling.
    ///
    /// For enumeration and labelling only. Membership is NOT the gate — ask
    /// `isNSFW`. A spelling absent from this set is not thereby SFW; unknown
    /// vocabulary fails CLOSED and `isNSFW` returns true for it.
    public static let nsfwModes: Set<String> = Set(
        CATALOG_TIER_SPELLINGS.filter {
            isWithheld(tier: $0, ceiling: CATALOG_STRICTEST_CEILING)
        })

    /// Whether an asset's content mode sits above SFW.
    ///
    /// One question, one answer, resolved by the same clamp the catalog and the
    /// gallery server use — so a mode the desktop blurs is a mode the server
    /// withholds. nil/empty stays SFW: an asset that was never tiered is
    /// untiered, not secretly explicit.
    public static func isNSFW(contentMode: String?) -> Bool {
        isWithheld(tier: contentMode, ceiling: CATALOG_STRICTEST_CEILING)
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
