// RemoteGallerySettingsTests.swift — the list of remotes in Settings
// (FDD-remote-galleries §3.1). The view itself has no test harness in this
// repo, so every decision it makes lives in these pure helpers.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("RemoteGallerySettings")
struct RemoteGallerySettingsTests {

    private let vault = RemoteGalleryConfig.folder(name: "Vault SSD", rootPath: "/Volumes/Vault", volumeUUID: "UUID-1")
    private let immich = RemoteGalleryConfig.immich(name: "Immich", baseURL: "http://10.0.100.232:2283", albumName: "ComfyBox")

    @Test("adding a remote keeps the existing ones")
    func adding() {
        let list = RemoteGallerySettings.appending(immich, to: [vault])
        #expect(list.count == 2)
        #expect(list.first?.id == vault.id)
    }

    @Test("the same drive is never added twice")
    func duplicateFolder() {
        let again = RemoteGalleryConfig.folder(name: "Vault again", rootPath: "/Volumes/Vault/", volumeUUID: "UUID-1")
        let list = RemoteGallerySettings.appending(again, to: [vault])
        #expect(list.count == 1, "one drive, one remote")
        #expect(list.first?.id == vault.id, "the original id survives, so moved assets keep resolving")
    }

    @Test("the same Immich album is never added twice")
    func duplicateImmich() {
        let again = RemoteGalleryConfig.immich(name: "Immich 2", baseURL: "http://10.0.100.232:2283/", albumName: "ComfyBox")
        #expect(RemoteGallerySettings.appending(again, to: [immich]).count == 1)
        let other = RemoteGalleryConfig.immich(name: "Other album", baseURL: "http://10.0.100.232:2283", albumName: "Private")
        #expect(RemoteGallerySettings.appending(other, to: [immich]).count == 2, "a different album is a different remote")
    }

    @Test("an invalid remote is refused rather than stored half-configured")
    func refuseInvalid() {
        let broken = RemoteGalleryConfig.folder(name: "", rootPath: "", volumeUUID: nil)
        #expect(RemoteGallerySettings.appending(broken, to: [vault]) == [vault])
    }

    @Test("removing takes exactly one remote and leaves its contents alone")
    func removing() {
        let list = RemoteGallerySettings.removing(id: vault.id, from: [vault, immich])
        #expect(list.map(\.id) == [immich.id])
        #expect(RemoteGallerySettings.removing(id: "nope", from: [vault]) == [vault])
    }

    @Test("each row reads as where it actually is")
    func summaries() {
        #expect(RemoteGallerySettings.summary(for: vault) == "/Volumes/Vault/ComfyBoxGallery")
        #expect(RemoteGallerySettings.summary(for: immich) == "http://10.0.100.232:2283 · album ComfyBox")
        let unsupported = RemoteGalleryConfig(id: "x", name: "Future", kind: .unsupported)
        #expect(RemoteGallerySettings.summary(for: unsupported) == "Unsupported remote — written by a newer version")
    }

    @Test("only valid, enabled remotes are offered as send destinations")
    func destinations() {
        var disabled = immich
        disabled.enabled = false
        let broken = RemoteGalleryConfig.folder(name: "Broken", rootPath: "", volumeUUID: nil)
        let offered = RemoteGallerySettings.destinations(in: [vault, disabled, broken])
        #expect(offered.map(\.id) == [vault.id])
    }
}
