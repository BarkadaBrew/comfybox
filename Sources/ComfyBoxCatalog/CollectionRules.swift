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

    /// Last-resort filing by content tier, used ONLY when the lane rules above
    /// yield nothing.
    ///
    /// Sidecars in the field record lane `render` / `quality` / `sketch` — none
    /// of which the lane table maps — so lane-based filing tops out at whatever
    /// the render journal covers, leaving ~88% of the catalog in no collection
    /// at all. `content_mode` is populated on 2483 of 2882 rows and is the one
    /// facet broad enough to stand in. Coarser than a lane, and honest about it:
    /// a real lane always wins, and manual filing in the gallery survives
    /// re-ingest because `applyDerivedFiling` only ever deletes `manual = 0`.
    private static let kiraModeToCollection: [String: String] = [
        "avocado": "col-kira-adult-scenes",
        "banana": "col-kira-nightlife",
        "neutral": "col-kira-still-life",
        // `apple` is deliberately absent. There is no existing genre for her SFW
        // lifestyle work and inventing one is a decision for the owner, not for
        // this table. Those assets stay unfiled, and visibly so.
    ]

    /// The SHARED roots — never her genres.
    ///
    /// `applyDerivedFiling` forbids a `shared` asset from entering any `col-kira-*`
    /// collection, and `~/Pictures/ComfyBox` has no metadata root so every
    /// Mac-only asset is stamped `.shared`. Mapping this branch to her genres
    /// would therefore file nothing at all — the guard would drop every row on
    /// the way in. Hence the roots.
    private static let sharedModeToCollection: [String: String] = [
        "neutral": "col-photography",
        "apple": "col-photography",
        "banana": "col-adult",
        "avocado": "col-adult",
    ]

    public static func defaultCollectionIDs(for asset: CatalogAsset) -> [String] {
        switch asset.realm {
        case .kira:
            let byLane = kiraCollectionIDs(for: asset)
            if !byLane.isEmpty { return byLane }
            // Keyed on the REALM, never on character_name: a shared asset that
            // merely depicts her must not reach her genres.
            return [asset.contentMode.flatMap { kiraModeToCollection[$0] }].compactMap { $0 }
        case .shared:
            // A shared asset can only ever land in a shared collection.
            if let source = asset.source, let id = sourceToSharedCollection[source] {
                return [id]
            }
            return [asset.contentMode.flatMap { sharedModeToCollection[$0] }].compactMap { $0 }
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
