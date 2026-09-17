// RemoteGalleryRegistry.swift — which remotes are here right now.
//
// Todd 2026-09-17: when the drive is unplugged, the remote and its assets are
// hidden until it is back. Reachability gates every send destination and
// everything the gallery shows (FDD-remote-galleries §3.5).
//
// A drive is followed by its volume UUID first and its configured path second,
// so renaming a drive or mounting it somewhere else does not lose a gallery.

import Foundation
import Observation

public enum RemoteGalleryStatus: String, Sendable, Equatable {
    /// Here now: it can receive sends and its assets can be shown.
    case reachable
    /// Configured but not present — unplugged, or the server is down.
    case absent
    /// Switched off, or written by a newer version of the app.
    case disabled
}

@Observable
@MainActor
public final class RemoteGalleryRegistry {

    /// Status per remote id, refreshed by `refresh()`.
    public private(set) var statuses: [String: RemoteGalleryStatus] = [:]
    /// Where each reachable folder remote actually is right now.
    public private(set) var resolvedRoots: [String: String] = [:]

    private var remotes: [RemoteGalleryConfig]
    private let session: URLSession

    public init(remotes: [RemoteGalleryConfig] = [], session: URLSession = .shared) {
        self.remotes = remotes
        self.session = session
    }

    public func update(remotes: [RemoteGalleryConfig]) {
        self.remotes = remotes
    }

    public func status(of id: String) -> RemoteGalleryStatus {
        statuses[id] ?? .absent
    }

    /// Remotes that can receive a send right now.
    public var reachable: [RemoteGalleryConfig] {
        remotes.filter { statuses[$0.id] == .reachable }
    }

    /// Re-check every remote. Folder checks are a stat; Immich gets one ping
    /// inside a short budget so a dead server cannot stall the UI.
    public func refresh() async {
        let volumes = Self.mountedVolumesByUUID()
        var next: [String: RemoteGalleryStatus] = [:]
        var roots: [String: String] = [:]
        for remote in remotes {
            switch remote.kind {
            case .folder:
                next[remote.id] = Self.folderStatus(for: remote, mountedVolumes: volumes)
                if let root = Self.resolvedGalleryRoot(for: remote, mountedVolumes: volumes) {
                    roots[remote.id] = root
                }
            case .immich:
                next[remote.id] = await immichStatus(for: remote)
            case .unsupported:
                next[remote.id] = .disabled
            }
        }
        statuses = next
        resolvedRoots = roots
    }

    // MARK: - Folder remotes (pure, so they are testable)

    public nonisolated static func folderStatus(for remote: RemoteGalleryConfig,
                                    mountedVolumes: [String: String]) -> RemoteGalleryStatus {
        guard remote.enabled, remote.kind == .folder, remote.isValid else { return .disabled }
        guard let root = resolvedRoot(for: remote, mountedVolumes: mountedVolumes) else { return .absent }
        var isDirectory: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: root, isDirectory: &isDirectory)
        return exists && isDirectory.boolValue ? .reachable : .absent
    }

    /// Where the drive is now: by UUID when we know it, else the configured path.
    public nonisolated static func resolvedRoot(for remote: RemoteGalleryConfig,
                                    mountedVolumes: [String: String]) -> String? {
        if let uuid = remote.volumeUUID, !uuid.isEmpty, let mounted = mountedVolumes[uuid] { return mounted }
        guard let path = remote.rootPath, !path.isEmpty else { return nil }
        return path
    }

    /// The gallery directory inside the resolved root.
    public nonisolated static func resolvedGalleryRoot(for remote: RemoteGalleryConfig,
                                           mountedVolumes: [String: String]) -> String? {
        guard let root = resolvedRoot(for: remote, mountedVolumes: mountedVolumes) else { return nil }
        let trimmed = root.hasSuffix("/") ? String(root.dropLast()) : root
        return (trimmed as NSString).appendingPathComponent(RemoteGalleryConfig.galleryDirectoryName)
    }

    /// Mounted volumes keyed by UUID, so a renamed drive still resolves.
    public nonisolated static func mountedVolumesByUUID() -> [String: String] {
        var map: [String: String] = [:]
        let keys: [URLResourceKey] = [.volumeUUIDStringKey]
        let urls = FileManager.default.mountedVolumeURLs(includingResourceValuesForKeys: keys, options: []) ?? []
        for url in urls {
            if let uuid = (try? url.resourceValues(forKeys: Set(keys)))?.volumeUUIDString {
                map[uuid] = url.path
            }
        }
        return map
    }

    // MARK: - Immich

    private func immichStatus(for remote: RemoteGalleryConfig) async -> RemoteGalleryStatus {
        guard remote.enabled, remote.isValid, let base = remote.normalizedBaseURL,
              let url = URL(string: base + "/api/server/ping") else { return .disabled }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        if let key = Keychain.get(remote.keychainAccount) {
            request.setValue(key, forHTTPHeaderField: "x-api-key")
        }
        do {
            let (_, response) = try await session.data(for: request)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            return (200..<300).contains(code) ? .reachable : .absent
        } catch {
            return .absent
        }
    }
}
