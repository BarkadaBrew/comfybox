// CollectionRules.swift — derived filing: lane / source / family / mode → the
// collections an asset lands in when nobody names one.
//
// Pure and table-driven on purpose. This is the layer that lets the Studio bot,
// Krita and the desktop tool file into the same bodies of work as Kira WITHOUT
// adopting her scheduler's vocabulary — each producer keeps its own words and
// the table translates.
//
// Precedence lives in CatalogStore: manual > explicit > derived. This function
// is only the derived tier, and returns [] rather than guessing.

import Foundation

public enum CollectionRules {

    /// Kira's lane → her genre. `kira` lane is disambiguated by `family`.
    private static let kiraLaneToCollection: [String: String] = [
        "still": "col-kira-still-life",
        "shoot": "col-kira-autocord",
        "tile": "col-kira-decoupage",
        "film-erotic": "col-kira-erotic-portraiture",
        "kira": "col-kira-adult-scenes",
    ]

    /// Her genres that are also part of a shared, cross-producer body of work.
    private static let alsoShared: [String: String] = [
        "col-kira-decoupage": "col-decoupage",
    ]

    /// Non-Kira producers → the shared body they contribute to.
    private static let sourceToSharedCollection: [String: String] = [
        "desktop-decoupage": "col-decoupage",
        "tile-engine": "col-decoupage",
        "krita": "col-decoupage",
        "studio-tile": "col-decoupage",
    ]

    public static func defaultCollectionIDs(for asset: CatalogAsset) -> [String] {
        switch asset.realm {
        case .kira:
            return kiraCollectionIDs(for: asset)
        case .shared:
            // A shared asset can only ever land in a shared collection.
            guard let source = asset.source,
                  let id = sourceToSharedCollection[source] else { return [] }
            return [id]
        }
    }

    private static func kiraCollectionIDs(for asset: CatalogAsset) -> [String] {
        guard let lane = asset.lane else { return [] }

        // t2v on the video lane is Dreams & Memories — the one genre that is
        // only video. An i2v clip belongs to the genre of the scene it animates.
        if lane == "video" {
            return asset.mode == "t2v" ? ["col-kira-dreams-memories"] : []
        }

        if lane == "kira", asset.family == "nightlife" {
            return ["col-kira-nightlife"]
        }

        guard let own = kiraLaneToCollection[lane] else { return [] }
        if let shared = alsoShared[own] { return [own, shared] }
        return [own]
    }
}
