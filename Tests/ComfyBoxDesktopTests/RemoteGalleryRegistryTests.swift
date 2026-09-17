// RemoteGalleryRegistryTests.swift — is this remote here right now?
// (FDD-remote-galleries §3.5).
//
// Todd 2026-09-17 chose: when the drive is unplugged, the remote and its
// assets are HIDDEN until it is back. So reachability is the gate on every
// send destination and on what the gallery shows.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("RemoteGalleryRegistry")
struct RemoteGalleryRegistryTests {

    private func tempDir() -> String {
        let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("reg-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: path, withIntermediateDirectories: true)
        return path
    }

    @Test("a folder remote is reachable when its drive is there")
    func folderPresent() {
        let root = tempDir()
        defer { try? FileManager.default.removeItem(atPath: root) }
        let remote = RemoteGalleryConfig.folder(name: "Test", rootPath: root, volumeUUID: nil)
        #expect(RemoteGalleryRegistry.folderStatus(for: remote, mountedVolumes: [:]) == .reachable)
    }

    @Test("a folder remote is absent when its drive is not")
    func folderAbsent() {
        let remote = RemoteGalleryConfig.folder(name: "Bolt", rootPath: "/Volumes/NoSuchDrive", volumeUUID: nil)
        #expect(RemoteGalleryRegistry.folderStatus(for: remote, mountedVolumes: [:]) == .absent)
    }

    @Test("a drive is followed by its UUID when it mounts somewhere else")
    func followsVolumeUUID() {
        let root = tempDir()
        defer { try? FileManager.default.removeItem(atPath: root) }
        // Configured at /Volumes/Vault, actually mounted at `root` this time.
        let remote = RemoteGalleryConfig.folder(name: "Vault", rootPath: "/Volumes/Vault", volumeUUID: "UUID-1")
        let status = RemoteGalleryRegistry.folderStatus(for: remote, mountedVolumes: ["UUID-1": root])
        #expect(status == .reachable)
        #expect(RemoteGalleryRegistry.resolvedRoot(for: remote, mountedVolumes: ["UUID-1": root]) == root)
    }

    @Test("a disabled remote is never reachable, plugged in or not")
    func disabled() {
        let root = tempDir()
        defer { try? FileManager.default.removeItem(atPath: root) }
        var remote = RemoteGalleryConfig.folder(name: "Test", rootPath: root, volumeUUID: nil)
        remote.enabled = false
        #expect(RemoteGalleryRegistry.folderStatus(for: remote, mountedVolumes: [:]) == .disabled)
    }

    @Test("an unsupported remote is never offered")
    func unsupported() {
        let remote = RemoteGalleryConfig(id: "x", name: "Future", kind: .unsupported)
        #expect(RemoteGalleryRegistry.folderStatus(for: remote, mountedVolumes: [:]) == .disabled)
    }

    @Test("the gallery root is what gets checked, not merely the drive")
    func galleryRootMatters() {
        let root = tempDir()
        defer { try? FileManager.default.removeItem(atPath: root) }
        let remote = RemoteGalleryConfig.folder(name: "Test", rootPath: root, volumeUUID: nil)
        // The drive is here but holds no gallery yet: still reachable, because
        // the first send creates it.
        #expect(RemoteGalleryRegistry.folderStatus(for: remote, mountedVolumes: [:]) == .reachable)
        #expect(RemoteGalleryRegistry.resolvedGalleryRoot(for: remote, mountedVolumes: [:])
                == (root as NSString).appendingPathComponent("ComfyBoxGallery"))
    }
}
