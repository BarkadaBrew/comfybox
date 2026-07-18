import XCTest
@testable import ComfyBoxDesktop

@MainActor
final class AppContentGateTests: XCTestCase {

    /// The real login-Keychain hash, saved so the test never clobbers a
    /// gallery password the user actually set.
    private var savedHash: String?

    override func setUp() {
        super.setUp()
        savedHash = Keychain.get("nsfw_gate_hash")
        NSFWGate.setPassword(nil)   // start each case with no password
    }

    override func tearDown() {
        Keychain.set(savedHash, "nsfw_gate_hash")   // restore original (nil if none)
        super.tearDown()
    }

    func testStartsHiddenByDefault() {
        let gate = AppContentGate()
        XCTAssertFalse(gate.revealed, "Rated G by default — a fresh gate is always hidden")
    }

    func testRevealAndHideWithoutPassword() {
        let gate = AppContentGate()
        XCTAssertFalse(gate.requiresPassword)
        gate.reveal()
        XCTAssertTrue(gate.revealed)
        gate.hide()
        XCTAssertFalse(gate.revealed)
    }

    func testPasswordGatedReveal() {
        NSFWGate.setPassword("hunter2")
        let gate = AppContentGate()
        XCTAssertTrue(gate.requiresPassword)

        // Password-free reveal is a no-op when a password is configured.
        gate.reveal()
        XCTAssertFalse(gate.revealed, "reveal() must not bypass a configured password")

        XCTAssertFalse(gate.reveal(withPassword: "wrong"))
        XCTAssertFalse(gate.revealed)

        XCTAssertTrue(gate.reveal(withPassword: "hunter2"))
        XCTAssertTrue(gate.revealed)

        // Hiding is always allowed.
        gate.hide()
        XCTAssertFalse(gate.revealed)
    }
}
