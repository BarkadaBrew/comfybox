import XCTest
@testable import ZImage

/// WP-E10 — `loras[].role` (FDD §3.10 `Applied.role`). The engine labels each
/// adapter's configuration slot ONCE so the client can read `kroma_strength`
/// back as applied (AC-45) instead of matching filenames. On `/v1/generate`
/// the slot is declared on the entry; it travels into `LoRAConfiguration`
/// and is read back from the pipeline's loaded configs — an unknown label is
/// refused, never stored.
final class LoRAEntryRoleTests: XCTestCase {

  private func decode(_ json: String) throws -> GeneratePayload {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return try d.decode(GeneratePayload.self, from: Data(json.utf8))
  }

  func testRoleDecodesAndReachesTheConfiguration() throws {
    let p = try decode(#"{"prompt":"x","loras":[{"path":"/tmp/kroma.safetensors","scale":0.3,"role":"kroma"},{"path":"/tmp/turbo.safetensors","scale":0.6}]}"#)
    let entries = try XCTUnwrap(p.loras)
    XCTAssertEqual(entries[0].role, "kroma")
    XCTAssertNil(entries[1].role)
    let cfg = try entries[0].makeConfiguration()
    XCTAssertEqual(cfg.role, "kroma")
    XCTAssertEqual(cfg.scale, 0.3)
    XCTAssertNil(try entries[1].makeConfiguration().role)
  }

  func testR256DistillKeepsExplicitAcceleratorRole() throws {
    let p = try decode(
      #"{"prompt":"x","loras":[{"path":"/tmp/krea2_turbo_distill_r256.safetensors","scale":0.6,"role":"accel"}]}"#)
    let entry = try XCTUnwrap(p.loras?.first)
    let configuration = try entry.makeConfiguration()

    XCTAssertEqual(entry.role, "accel")
    XCTAssertEqual(configuration.role, "accel")
    XCTAssertEqual(configuration.scale, 0.6)
  }

  func testEveryDeclaredRoleIsAccepted() throws {
    for role in LoRAEntry.roles {
      let entry = try decode(#"{"prompt":"x","loras":[{"path":"/tmp/a.safetensors","role":"\#(role)"}]}"#).loras![0]
      XCTAssertEqual(try entry.makeConfiguration().role, role)
    }
    XCTAssertEqual(Set(LoRAEntry.roles), ["kroma", "accel", "bypass", "control"])
  }

  func testUnknownRoleIsRefusedNotStored() throws {
    let entry = try decode(#"{"prompt":"x","loras":[{"path":"/tmp/a.safetensors","role":"style"}]}"#).loras![0]
    XCTAssertThrowsError(try entry.makeConfiguration()) { error in
      guard case WarmServerError.invalidRequest(let message) = error else { return XCTFail("\(error)") }
      XCTAssertTrue(message.contains("style"), message)
      XCTAssertTrue(message.contains("kroma"), message)
      XCTAssertEqual(WarmServer.errorResponse(for: error).status, 400)
    }
  }

  func testRoleDoesNotDisturbConfigurationEquality() {
    var a = LoRAConfiguration.local("/tmp/a.safetensors", scale: 1.0)
    let b = LoRAConfiguration.local("/tmp/a.safetensors", scale: 1.0)
    XCTAssertEqual(a, b)
    a.role = "kroma"
    XCTAssertNotEqual(a, b, "the slot label is part of what was applied")
  }
}
