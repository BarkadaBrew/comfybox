import XCTest
@testable import ComfyBoxDesktop

final class ServiceControllerTests: XCTestCase {

    func testLaunchctlArgs() {
        XCTAssertEqual(
            ServiceController.launchctlArgs(.restart, label: "com.barkadabrew.comfybox", uid: 501),
            ["kickstart", "-k", "gui/501/com.barkadabrew.comfybox"])
        XCTAssertEqual(
            ServiceController.launchctlArgs(.start, label: "x", uid: 501),
            ["kickstart", "gui/501/x"])
        XCTAssertEqual(
            ServiceController.launchctlArgs(.stop, label: "x", uid: 501),
            ["kill", "SIGTERM", "gui/501/x"])
    }

    func testCustomCommandLocal() {
        let ctl = ServiceControl(startCommand: "pm2 start bree", stopCommand: "pm2 stop bree")
        XCTAssertEqual(ServiceController.customCommand(.start, control: ctl), "pm2 start bree")
        XCTAssertEqual(ServiceController.customCommand(.stop, control: ctl), "pm2 stop bree")
        // Restart falls back to stop && start.
        XCTAssertEqual(ServiceController.customCommand(.restart, control: ctl), "pm2 stop bree && pm2 start bree")
    }

    func testCustomCommandExplicitRestart() {
        let ctl = ServiceControl(restartCommand: "pm2 restart bree")
        XCTAssertEqual(ServiceController.customCommand(.restart, control: ctl), "pm2 restart bree")
    }

    func testCustomCommandOverSSH() {
        let ctl = ServiceControl(sshHost: "todd@10.0.100.232", restartCommand: "systemctl --user restart bree")
        XCTAssertEqual(
            ServiceController.customCommand(.restart, control: ctl),
            "ssh todd@10.0.100.232 'systemctl --user restart bree'")
    }

    func testCustomCommandSSHQuotesInnerQuotes() {
        let ctl = ServiceControl(sshHost: "h", startCommand: "echo 'hi there'")
        let cmd = ServiceController.customCommand(.start, control: ctl)
        XCTAssertEqual(cmd, "ssh h 'echo '\\''hi there'\\'''")
    }

    func testCustomCommandNilWhenUndefined() {
        let ctl = ServiceControl(startCommand: "only start")
        XCTAssertNil(ServiceController.customCommand(.stop, control: ctl))
    }

    func testIsActionable() {
        XCTAssertFalse(ServiceControl().isActionable)
        XCTAssertTrue(ServiceControl(launchdLabel: "x").isActionable)
        XCTAssertTrue(ServiceControl(restartCommand: "y").isActionable)
    }

    func testWatchedServiceDecodesWithoutControl() throws {
        // Back-compat: older configs have no `control` field.
        let json = #"{"id":"a","name":"Svc","urlString":"http://x/health"}"#.data(using: .utf8)!
        let svc = try JSONDecoder().decode(WatchedService.self, from: json)
        XCTAssertNil(svc.control)
    }
}
