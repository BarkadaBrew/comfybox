// RemoteGalleryConfig.swift — where a remote gallery lives.
//
// Todd 2026-09-17: "I would like to wire through configuration where the
// remote galleries are and I want to create a remote gallery on a USB SSD
// thumb drive … Sensitive content, private content or content I want to make
// portable would go there," and "we can use the immich store on the server as
// an additional remote gallery."
//
// One row per remote in ~/.comfybox/desktop-config.json, following the
// `archiveRoots` / `watchedServices` pattern. Credentials are NOT fields here:
// an Immich key lives in the login keychain under `keychainAccount`
// (FDD-remote-galleries §3.1).

import Foundation

public struct RemoteGalleryConfig: Codable, Identifiable, Hashable, Sendable {

    public enum Kind: String, Codable, Sendable {
        /// A directory tree, typically on a portable drive.
        case folder
        /// An Immich server.
        case immich
        /// Written by a newer build than this one. Never offered as a
        /// destination, never edited, and preserved on save.
        case unsupported

        public init(from decoder: Decoder) throws {
            let raw = try decoder.singleValueContainer().decode(String.self)
            self = Kind(rawValue: raw) ?? .unsupported
        }
    }

    /// Stable for the life of the remote: every moved asset's location host is
    /// derived from it, so regenerating it would orphan them.
    public let id: String
    public var name: String
    public var kind: Kind
    public var enabled: Bool

    // .folder
    public var rootPath: String?
    /// Identifies the drive across mount points and renames.
    public var volumeUUID: String?

    // .immich
    public var baseURL: String?
    public var albumName: String?
    public var albumID: String?

    public init(id: String = UUID().uuidString,
                name: String,
                kind: Kind,
                enabled: Bool = true,
                rootPath: String? = nil,
                volumeUUID: String? = nil,
                baseURL: String? = nil,
                albumName: String? = nil,
                albumID: String? = nil) {
        self.id = id
        self.name = name
        self.kind = kind
        self.enabled = enabled
        self.rootPath = rootPath
        self.volumeUUID = volumeUUID
        self.baseURL = Self.normalizedURL(baseURL)
        self.albumName = albumName
        self.albumID = albumID
    }

    public static func folder(name: String, rootPath: String, volumeUUID: String?) -> RemoteGalleryConfig {
        RemoteGalleryConfig(name: name, kind: .folder, rootPath: rootPath, volumeUUID: volumeUUID)
    }

    public static func immich(name: String, baseURL: String, albumName: String) -> RemoteGalleryConfig {
        RemoteGalleryConfig(name: name, kind: .immich, baseURL: baseURL, albumName: albumName)
    }

    /// The directory the gallery itself occupies inside `rootPath`.
    public static let galleryDirectoryName = "ComfyBoxGallery"

    public var galleryRoot: String? {
        guard kind == .folder, let rootPath, !rootPath.isEmpty else { return nil }
        let trimmed = rootPath.hasSuffix("/") ? String(rootPath.dropLast()) : rootPath
        return (trimmed as NSString).appendingPathComponent(Self.galleryDirectoryName)
    }

    /// The `asset_locations.host` value for assets that live on this remote.
    public var locationHost: String { "remote:\(id)" }

    /// The keychain account holding this remote's credential, if it needs one.
    public var keychainAccount: String { "remote-gallery-\(id)" }

    /// Configured well enough to be offered as a destination.
    public var isValid: Bool {
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        switch kind {
        case .folder: return !(rootPath ?? "").isEmpty
        case .immich: return !(baseURL ?? "").isEmpty
        case .unsupported: return false
        }
    }

    private static func normalizedURL(_ url: String?) -> String? {
        guard var url, !url.isEmpty else { return url }
        while url.hasSuffix("/") { url.removeLast() }
        return url
    }
}

extension RemoteGalleryConfig {
    /// Assign a new base URL, normalised the same way the initialiser does.
    public var normalizedBaseURL: String? { Self.normalizedURL(baseURL) }
}
