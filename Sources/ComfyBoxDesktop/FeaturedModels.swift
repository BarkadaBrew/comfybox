// FeaturedModels.swift — Curated "art models" ComfyBox knows about
//
// Todd's own Z-Image-based models (Zeta-Chroma, CoffeeShop) aren't CivitAI
// downloads and may live on attached storage, so they don't always appear in
// the live catalog. This curated list gives them a first-class representation
// in the Models tab: a friendly name, what they are, and where to find them —
// matched against the live catalog / nearline when present.

import Foundation

public struct FeaturedModel: Identifiable, Sendable {
    public let id: String
    public let name: String
    public let family: String
    public let blurb: String
    /// Substrings that identify this model in a catalog id/name or nearline item.
    public let matchers: [String]
    /// Where it lives when not staged (for the "mount to stage" hint).
    public let nearlineHint: String?

    public func matches(_ text: String) -> Bool {
        let lower = text.lowercased()
        return matchers.contains { lower.contains($0) }
    }
}

public enum FeaturedModels {
    public static let all: [FeaturedModel] = [
        FeaturedModel(
            id: "zeta-chroma",
            name: "Zeta-Chroma",
            family: "Z-Image",
            blurb: "Todd's Z-Image-based art model (distinct from the Flux-based Chroma). Used for the tile / mixed-media pipeline.",
            matchers: ["zeta-chroma", "zetachroma", "zeta chroma"],
            nearlineHint: "/Volumes/Seagate 22T/Models/zeta-chroma-editions"),
        FeaturedModel(
            id: "coffeeshop",
            name: "CoffeeShop",
            family: "Z-Image Turbo",
            blurb: "Merged Z-Image-Turbo checkpoint (8-bit) — the CoffeeShop house model.",
            matchers: ["coffeeshop", "coffee-shop"],
            nearlineHint: "~/Downloads/CoffeeShop-8bit"),
    ]
}
