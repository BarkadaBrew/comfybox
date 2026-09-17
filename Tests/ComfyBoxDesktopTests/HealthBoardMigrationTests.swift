import Testing
@testable import ComfyBoxDesktop

@Suite("Health board: LM Studio → mlx-serve")
struct HealthBoardMigrationTests {
    @Test("defaults watch mlx-serve on :11234 with launchd control, and no LM Studio row")
    func defaults() {
        let names = DesktopSettings.defaultWatchedServices.map(\.name)
        #expect(names.contains("mlx-serve (Glimmer)"))
        #expect(!names.contains("LM Studio"))
        let mlx = DesktopSettings.mlxServeWatchedService
        #expect(mlx.urlString == "http://127.0.0.1:11234/health")
        #expect(mlx.control?.launchdLabel == "com.barkadabrew.mlx-serve")
    }

    @Test("a saved LM Studio row migrates in place (same id); other rows untouched; a clean list is returned equal")
    func migration() {
        let lm = WatchedService(id: "keep-me", name: "LM Studio", urlString: "http://127.0.0.1:1234/v1/models")
        let bree = WatchedService(id: "b", name: "Bree Server", urlString: "http://10.0.100.232:3000/health")
        let out = DesktopSettings.migratingRetiredServices([lm, bree])
        #expect(out[0].id == "keep-me")
        #expect(out[0].name == "mlx-serve (Glimmer)")
        #expect(out[0].urlString == "http://127.0.0.1:11234/health")
        #expect(out[0].control?.launchdLabel == "com.barkadabrew.mlx-serve")
        #expect(out[1] == bree)
        #expect(DesktopSettings.migratingRetiredServices([bree]) == [bree])
        let renamed = WatchedService(id: "r", name: "Dan's model", urlString: "http://localhost:1234/v1/models")
        #expect(DesktopSettings.migratingRetiredServices([renamed])[0].urlString == "http://127.0.0.1:11234/health", "matches the retired port even under a custom name")
    }
}
