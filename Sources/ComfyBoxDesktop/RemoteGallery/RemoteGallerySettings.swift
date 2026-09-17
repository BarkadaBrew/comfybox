// RemoteGallerySettings.swift — the decisions the Settings list makes.
//
// The repo has no SwiftUI test harness, so the rules live here as pure
// functions and the view is a thin shell over them (the pattern
// `GalleryView.folderMembers` established).

import Foundation

public enum RemoteGallerySettings {

    /// Add a remote, unless it is invalid or already configured.
    ///
    /// "Already configured" is by DESTINATION, not by id: adding the same
    /// drive twice would give one gallery two location hosts, and assets moved
    /// under the first id would stop resolving.
    public static func appending(_ config: RemoteGalleryConfig,
                                 to list: [RemoteGalleryConfig]) -> [RemoteGalleryConfig] {
        guard config.isValid else { return list }
        guard !list.contains(where: { sameDestination($0, config) }) else { return list }
        return list + [config]
    }

    public static func removing(id: String, from list: [RemoteGalleryConfig]) -> [RemoteGalleryConfig] {
        list.filter { $0.id != id }
    }

    /// Two configs point at the same place.
    public static func sameDestination(_ a: RemoteGalleryConfig, _ b: RemoteGalleryConfig) -> Bool {
        guard a.kind == b.kind else { return false }
        switch a.kind {
        case .folder:
            if let ua = a.volumeUUID, let ub = b.volumeUUID, !ua.isEmpty, !ub.isEmpty { return ua == ub }
            return a.galleryRoot == b.galleryRoot
        case .immich:
            return a.normalizedBaseURL == b.normalizedBaseURL && a.albumName == b.albumName
        case .unsupported:
            return a.id == b.id
        }
    }

    /// The second line of a row: where this remote actually is.
    public static func summary(for config: RemoteGalleryConfig) -> String {
        switch config.kind {
        case .folder:
            return config.galleryRoot ?? "No folder chosen"
        case .immich:
            let host = config.normalizedBaseURL ?? "No server"
            let album = config.albumName ?? "ComfyBox"
            return "\(host) · album \(album)"
        case .unsupported:
            return "Unsupported remote — written by a newer version"
        }
    }

    /// The remotes that may be offered as a send destination: configured
    /// completely, and switched on. Reachability is a separate question
    /// (`RemoteGalleryRegistry`), asked at send time.
    public static func destinations(in list: [RemoteGalleryConfig]) -> [RemoteGalleryConfig] {
        list.filter { $0.enabled && $0.isValid }
    }

    /// The volume UUID of the drive holding `path`, when macOS reports one.
    /// Stored so a remote survives a rename or a different mount point.
    public static func volumeUUID(forPath path: String) -> String? {
        let url = URL(fileURLWithPath: path)
        let values = try? url.resourceValues(forKeys: [.volumeUUIDStringKey])
        return values?.volumeUUIDString
    }
}
