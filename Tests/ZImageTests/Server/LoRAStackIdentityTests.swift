import XCTest
@testable import ZImage

/// #286 review round 1, I2 — the same-stack shortcut for Krea 2 and Flux 2.
///
/// `ZImagePipeline.loadLoRAs` has always skipped an identical stack; the Krea 2
/// and Flux 2 pipelines unload and reload every adapter on every call. That was
/// tolerable while a per-request stack only arrived on the explicit `loras`
/// path — now that a named preset applies its stack on EVERY render, a 5-10
/// adapter preset would clear and re-bind the whole stack per render, on a
/// daemon that renders around the clock and shares unified memory with LM
/// Studio and the embeddings service.
final class LoRAStackIdentityTests: XCTestCase {

  private func config(_ path: String, _ scale: Float, role: String? = nil,
                      base: Krea2Variant? = nil) -> LoRAConfiguration {
    var c = LoRAConfiguration.local(path, scale: scale)
    c.role = role
    c.requiresBase = base
    return c
  }

  /// Kira's live stack, applied twice in a row — the second render must not
  /// reload it.
  func testIdenticalStackIsRecognised() {
    let stack = [
      config("/loras/kroma-v0.3-base-lora-rank-384-fro-0985.safetensors", 0.6, role: "kroma"),
      config("/loras/krea2_turbo_distill_r256.safetensors", 0.6, role: "accel"),
      config("/loras/RealisticSnapshotKrea2.safetensors", 0.4),
    ]
    XCTAssertTrue(LoRAStackIdentity.isSameStack(stack, stack))
  }

  func testEmptyStacksMatch() {
    XCTAssertTrue(LoRAStackIdentity.isSameStack([], []))
  }

  // Any real difference must still reload — a missed reload renders the wrong
  // look, which is the defect #286 exists to fix.

  func testDifferentScaleReloads() {
    XCTAssertFalse(LoRAStackIdentity.isSameStack(
      [config("/loras/a.safetensors", 0.6)], [config("/loras/a.safetensors", 0.8)]))
  }

  func testDifferentFileReloads() {
    XCTAssertFalse(LoRAStackIdentity.isSameStack(
      [config("/loras/a.safetensors", 0.6)], [config("/loras/b.safetensors", 0.6)]))
  }

  /// Same basename in two directories is NOT the same artifact.
  func testSameNameDifferentDirectoryReloads() {
    XCTAssertFalse(LoRAStackIdentity.isSameStack(
      [config("/loras/a.safetensors", 0.6)], [config("/nearline/a.safetensors", 0.6)]))
  }

  func testDifferentLengthReloads() {
    XCTAssertFalse(LoRAStackIdentity.isSameStack(
      [config("/loras/a.safetensors", 0.6)],
      [config("/loras/a.safetensors", 0.6), config("/loras/b.safetensors", 0.5)]))
  }

  /// Order is part of the stack — LoRA application is not commutative.
  func testReorderedStackReloads() {
    let a = config("/loras/a.safetensors", 0.6)
    let b = config("/loras/b.safetensors", 0.5)
    XCTAssertFalse(LoRAStackIdentity.isSameStack([a, b], [b, a]))
  }

  /// The declared slot is part of what is applied (`applied.loras[].role`), so
  /// a role change must re-bind.
  func testDifferentRoleReloads() {
    XCTAssertFalse(LoRAStackIdentity.isSameStack(
      [config("/loras/a.safetensors", 0.6, role: "accel")],
      [config("/loras/a.safetensors", 0.6, role: "kroma")]))
    XCTAssertFalse(LoRAStackIdentity.isSameStack(
      [config("/loras/a.safetensors", 0.6)],
      [config("/loras/a.safetensors", 0.6, role: "accel")]))
  }

  /// `requiresBase` is extraction metadata, folded in identically on both
  /// sides before the comparison, and never changes what binds.
  func testRelativityIsNotPartOfTheIdentity() {
    XCTAssertTrue(LoRAStackIdentity.isSameStack(
      [config("/loras/a.safetensors", 0.6, base: .raw)],
      [config("/loras/a.safetensors", 0.6, base: .raw)]))
  }

  /// A HuggingFace reference compares on its id, not a local path.
  func testHuggingFaceSourcesCompare() {
    XCTAssertTrue(LoRAStackIdentity.isSameStack(
      [.huggingFace("org/lora", scale: 0.5)], [.huggingFace("org/lora", scale: 0.5)]))
    XCTAssertFalse(LoRAStackIdentity.isSameStack(
      [.huggingFace("org/lora", scale: 0.5)], [.huggingFace("org/other", scale: 0.5)]))
  }
}
