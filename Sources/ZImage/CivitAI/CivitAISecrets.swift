// CivitAISecrets.swift — CivitAI API key resolution for the headless server (#234)
//
// Desktop sources its CivitAI key from the macOS Keychain via AppSecrets
// (Sources/ComfyBoxDesktop/AppSecrets.swift, service
// "com.barkadabrew.comfybox.desktop", account "civitai"). The headless warm
// server has no Desktop UI, so it needs the same key through a resolution
// chain instead:
//
//   1. An explicit value — the `--civitai-key` CLI flag on `serve`, threaded
//      through WarmServerConfiguration.civitaiApiKey.
//   2. The CIVITAI_API_KEY environment variable.
//   3. The macOS Keychain, using the EXACT SAME service/account AppSecrets
//      uses — so a key already saved from the Desktop app's Settings screen
//      is picked up with no extra configuration.
//
// Never throws and never crashes on a missing key: callers get `nil` and are
// expected to respond 503, not trap (see WarmServer's /v1/civitai/* routes).

import Foundation
#if canImport(Security)
import Security
#endif

/// Resolves the CivitAI API key for server-side (headless) use.
public enum CivitAISecrets {
  /// Keychain service/account AppSecrets.swift (ComfyBoxDesktop) uses for its
  /// CivitAI credential. Kept as literals here (not a shared constant)
  /// because ZImage must not depend on the Desktop target — see the
  /// dependency-direction note in Package.swift (ZImage -> nothing,
  /// Desktop -> ZImage).
  static let keychainService = "com.barkadabrew.comfybox.desktop"
  static let keychainAccount = "civitai"

  /// Testing seam: production always resolves the real login Keychain.
  /// Tests substitute a stub so behavior doesn't depend on whatever is (or
  /// isn't) actually saved in the machine's login keychain.
  static var keychainLookup: () -> String? = { defaultKeychainLookup() }

  /// Resolve in order: `explicit` (e.g. --civitai-key), then
  /// `CIVITAI_API_KEY`, then the Keychain. Returns nil (never throws) when
  /// none of the three yield a non-empty value.
  public static func resolve(explicit: String? = nil) -> String? {
    if let explicit, !explicit.isEmpty { return explicit }
    if let env = ProcessInfo.processInfo.environment["CIVITAI_API_KEY"], !env.isEmpty {
      return env
    }
    if let fromKeychain = keychainLookup(), !fromKeychain.isEmpty {
      return fromKeychain
    }
    return nil
  }

  #if canImport(Security)
  /// Mirrors AppSecrets/Keychain's exact generic-password query — same
  /// service, same account, same attribute keys — so it reads the value the
  /// Desktop app already wrote to the login Keychain.
  private static func defaultKeychainLookup() -> String? {
    let query: [String: Any] = [
      kSecClass as String: kSecClassGenericPassword,
      kSecAttrService as String: keychainService,
      kSecAttrAccount as String: keychainAccount,
      kSecReturnData as String: true,
      kSecMatchLimit as String: kSecMatchLimitOne,
    ]
    var item: CFTypeRef?
    guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
          let data = item as? Data, let string = String(data: data, encoding: .utf8),
          !string.isEmpty else { return nil }
    return string
  }
  #else
  private static func defaultKeychainLookup() -> String? { nil }
  #endif
}
