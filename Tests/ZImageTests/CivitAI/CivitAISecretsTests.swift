import XCTest
@testable import ZImage

/// CivitAISecrets.resolve's precedence chain (#234): explicit > env >
/// Keychain. Uses the `keychainLookup` testing seam rather than the real
/// login Keychain, so behavior is deterministic regardless of whatever the
/// test machine actually has saved under com.barkadabrew.comfybox.desktop.
final class CivitAISecretsTests: XCTestCase {

  private var originalKeychainLookup: (() -> String?)!
  private var originalEnv: String?

  override func setUp() {
    super.setUp()
    originalKeychainLookup = CivitAISecrets.keychainLookup
    originalEnv = ProcessInfo.processInfo.environment["CIVITAI_API_KEY"]
  }

  override func tearDown() {
    CivitAISecrets.keychainLookup = originalKeychainLookup
    if let originalEnv {
      setenv("CIVITAI_API_KEY", originalEnv, 1)
    } else {
      unsetenv("CIVITAI_API_KEY")
    }
    super.tearDown()
  }

  func testExplicitValueWinsOverEverything() {
    setenv("CIVITAI_API_KEY", "env-key", 1)
    CivitAISecrets.keychainLookup = { "keychain-key" }
    XCTAssertEqual(CivitAISecrets.resolve(explicit: "explicit-key"), "explicit-key")
  }

  func testEmptyExplicitValueFallsThroughToEnv() {
    setenv("CIVITAI_API_KEY", "env-key", 1)
    CivitAISecrets.keychainLookup = { "keychain-key" }
    XCTAssertEqual(CivitAISecrets.resolve(explicit: ""), "env-key")
  }

  func testEnvVarWinsOverKeychainWhenNoExplicitValue() {
    setenv("CIVITAI_API_KEY", "env-key", 1)
    CivitAISecrets.keychainLookup = { "keychain-key" }
    XCTAssertEqual(CivitAISecrets.resolve(explicit: nil), "env-key")
  }

  func testKeychainIsUsedWhenNoExplicitOrEnvValue() {
    unsetenv("CIVITAI_API_KEY")
    CivitAISecrets.keychainLookup = { "keychain-key" }
    XCTAssertEqual(CivitAISecrets.resolve(explicit: nil), "keychain-key")
  }

  func testResolvesToNilWhenNothingIsConfiguredAnywhere() {
    unsetenv("CIVITAI_API_KEY")
    CivitAISecrets.keychainLookup = { nil }
    XCTAssertNil(CivitAISecrets.resolve(explicit: nil))
  }

  func testEmptyKeychainValueIsTreatedAsAbsent() {
    unsetenv("CIVITAI_API_KEY")
    CivitAISecrets.keychainLookup = { "" }
    XCTAssertNil(CivitAISecrets.resolve(explicit: nil))
  }

  /// Never throws / never crashes on a missing key — the whole point of the
  /// resolve() -> nil contract the WarmServer routes rely on for their 503s.
  func testNeverThrowsOnAnyCombinationOfMissingSources() {
    unsetenv("CIVITAI_API_KEY")
    CivitAISecrets.keychainLookup = { nil }
    XCTAssertNoThrow(_ = CivitAISecrets.resolve())
  }
}
