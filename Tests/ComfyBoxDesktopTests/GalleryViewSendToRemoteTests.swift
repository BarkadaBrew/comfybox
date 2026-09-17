// GalleryViewSendToRemoteTests.swift — what the gallery says before and after
// a send (FDD-remote-galleries §3.3, plan task R9).
//
// The repo has no SwiftUI harness, so the wording — which is the only warning
// a user gets before a file leaves this Mac — is pinned here.

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("GalleryView: send to remote")
@MainActor
struct GalleryViewSendToRemoteTests {

    @Test("the confirmation names the count and the destination")
    func title() {
        #expect(GalleryView.sendConfirmationTitle(count: 1, remoteName: "Vault SSD")
            == "Move 1 item to Vault SSD?")
        #expect(GalleryView.sendConfirmationTitle(count: 12, remoteName: "Vault SSD")
            == "Move 12 items to Vault SSD?")
    }

    @Test("the confirmation says plainly that the local copy is deleted")
    func message() {
        let one = GalleryView.sendConfirmationMessage(count: 1, remoteName: "Vault SSD")
        #expect(one.contains("only copy"))
        #expect(one.contains("deleted from this Mac"))
        let many = GalleryView.sendConfirmationMessage(count: 3, remoteName: "Vault SSD")
        #expect(many.contains("files") && many.contains("thumbnails"))
    }

    @Test("the result line distinguishes a clean send, a partial one and a total failure")
    func resultLines() {
        #expect(GalleryView.sendResultLine(sent: 5, failed: 0, remoteName: "Vault")
            == "Moved 5 to Vault")
        let partial = GalleryView.sendResultLine(sent: 4, failed: 1, remoteName: "Vault")
        #expect(partial.contains("4") && partial.contains("1 failed"))
        #expect(partial.contains("local copies"), "a failed asset keeps its local copy, and the user is told")
        let none = GalleryView.sendResultLine(sent: 0, failed: 3, remoteName: "Vault")
        #expect(none.contains("Nothing moved"))
        #expect(none.contains("local copies kept"))
    }
}
