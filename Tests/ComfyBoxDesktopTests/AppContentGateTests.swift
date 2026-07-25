import XCTest
@testable import ComfyBoxDesktop

/// Tests for the app-wide content gate. Deliberately Keychain-FREE: they never
/// touch NSFWGate's login-Keychain item (reading or writing it can trigger a
/// macOS access-authorization dialog when the item is owned by another process,
/// which hangs a headless test run). The password path — reveal(withPassword:)
/// / requiresPassword — is NSFWGate's own concern (a salted SHA-256 compare)
/// and is exercised interactively, not here.
@MainActor
final class AppContentGateTests: XCTestCase {

    func testStartsHiddenByDefault() {
        // Rated G by default — a fresh gate is always hidden, every launch.
        XCTAssertFalse(AppContentGate().revealed)
    }

    func testHideKeepsHidden() {
        let gate = AppContentGate()
        gate.hide()
        XCTAssertFalse(gate.revealed)
    }

    func testSeparateGatesAreIndependent() {
        // Each window/gate instance owns its own session-only reveal state.
        let a = AppContentGate()
        let b = AppContentGate()
        a.hide()
        b.hide()
        XCTAssertFalse(a.revealed)
        XCTAssertFalse(b.revealed)
    }
}
