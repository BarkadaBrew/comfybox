// RemoteGalleryConfigTests.swift — where a remote gallery lives
// (FDD-remote-galleries §3.1).
//
// Todd 2026-09-17: "I would like to wire through configuration where the
// remote galleries are." A remote is a row in desktop-config.json; an Immich
// key is never one of its fields.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("RemoteGalleryConfig")
struct RemoteGalleryConfigTests {

    private func decode(_ json: String) throws -> RemoteGalleryConfig {
        try JSONDecoder().decode(RemoteGalleryConfig.self, from: Data(json.utf8))
    }

    @Test("a folder remote names its drive and its gallery root")
    func folderRemote() {
        let remote = RemoteGalleryConfig.folder(name: "Vault SSD", rootPath: "/Volumes/Vault", volumeUUID: "ABC-123")
        #expect(remote.kind == .folder)
        #expect(remote.galleryRoot == "/Volumes/Vault/ComfyBoxGallery")
        #expect(remote.locationHost == "remote:\(remote.id)")
        #expect(remote.isValid)
    }

    @Test("a trailing slash on the root does not double up")
    func trailingSlash() {
        let remote = RemoteGalleryConfig.folder(name: "Vault", rootPath: "/Volumes/Vault/", volumeUUID: nil)
        #expect(remote.galleryRoot == "/Volumes/Vault/ComfyBoxGallery")
    }

    @Test("an Immich remote names its server and album")
    func immichRemote() {
        let remote = RemoteGalleryConfig.immich(name: "Immich", baseURL: "http://10.0.100.232:2283/", albumName: "ComfyBox")
        #expect(remote.kind == .immich)
        #expect(remote.baseURL == "http://10.0.100.232:2283", "the trailing slash is normalised away")
        #expect(remote.albumName == "ComfyBox")
        #expect(remote.isValid)
    }

    @Test("a remote missing what its kind needs is invalid")
    func validation() {
        var folder = RemoteGalleryConfig.folder(name: "No path", rootPath: "", volumeUUID: nil)
        #expect(!folder.isValid)
        folder.rootPath = "/Volumes/Vault"
        #expect(folder.isValid)

        var immich = RemoteGalleryConfig.immich(name: "No host", baseURL: "", albumName: "ComfyBox")
        #expect(!immich.isValid)
        immich.baseURL = "http://10.0.100.232:2283"
        #expect(immich.isValid)

        var unnamed = RemoteGalleryConfig.folder(name: "  ", rootPath: "/Volumes/Vault", volumeUUID: nil)
        #expect(!unnamed.isValid, "a remote the user cannot recognise is not configured")
        unnamed.name = "Vault"
        #expect(unnamed.isValid)
    }

    @Test("ids are stable across a decode and never regenerate")
    func stableIdentity() throws {
        let remote = RemoteGalleryConfig.folder(name: "Vault", rootPath: "/Volumes/Vault", volumeUUID: nil)
        let data = try JSONEncoder().encode(remote)
        let back = try JSONDecoder().decode(RemoteGalleryConfig.self, from: data)
        #expect(back.id == remote.id, "the location host of every moved asset depends on this")
        #expect(back == remote)
    }

    @Test("an older config file with no remotes decodes as no remotes")
    func absentListIsFine() throws {
        struct Slice: Codable { var remoteGalleries: [RemoteGalleryConfig]? }
        let slice = try JSONDecoder().decode(Slice.self, from: Data("{}".utf8))
        #expect(slice.remoteGalleries == nil)
    }

    @Test("an unknown kind from a newer build decodes as unsupported, not a crash")
    func forwardCompatibleKind() throws {
        let remote = try decode("""
        {"id":"r1","name":"Future","kind":"s3","enabled":true}
        """)
        #expect(remote.kind == .unsupported)
        #expect(!remote.isValid, "an unsupported remote is never offered as a destination")
    }

    @Test("no secret ever reaches the config file")
    func noSecretsInJSON() throws {
        var remote = RemoteGalleryConfig.immich(name: "Immich", baseURL: "http://10.0.100.232:2283", albumName: "ComfyBox")
        remote.albumID = "album-1"
        let json = String(data: try JSONEncoder().encode(remote), encoding: .utf8) ?? ""
        #expect(!json.lowercased().contains("apikey"))
        #expect(!json.lowercased().contains("token"))
        #expect(!json.lowercased().contains("password"))
        // The key is addressed by account name, and lives in the keychain.
        #expect(remote.keychainAccount == "remote-gallery-\(remote.id)")
    }
}
