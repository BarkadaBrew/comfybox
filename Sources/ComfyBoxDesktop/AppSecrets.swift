// AppSecrets.swift — API keys in the macOS Keychain, not plaintext JSON
//
// The desktop holds several third-party credentials (CivitAI, Replicate, Fal).
// These used to live in ~/.comfybox/desktop-config.json in cleartext; they now
// live in the login Keychain. A one-time migration moves any existing plaintext
// keys over and clears them from the JSON.

import Foundation
import Security

/// Minimal Keychain wrapper for generic-password items keyed by account.
///
/// Items live in the **data-protection keychain** (`kSecUseDataProtectionKeychain`),
/// whose access is governed by the app's `keychain-access-groups` entitlement rather
/// than an interactive per-signature ACL. That's what stops macOS from re-prompting
/// "…wants to access…" every time the app is re-signed. Legacy (file-based) items
/// written by older builds are read as a fallback and migrated forward on first
/// access, so existing keys are preserved (one Keychain prompt during migration,
/// then never again).
enum Keychain {
    static let service = "com.barkadabrew.comfybox.desktop"

    private static func baseQuery(_ account: String, dataProtection: Bool) -> [String: Any] {
        var q: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        // true  -> modern data-protection keychain (no interactive prompts)
        // false -> legacy file-based keychain (old items live here)
        q[kSecUseDataProtectionKeychain as String] = dataProtection
        return q
    }

    private static func read(_ account: String, dataProtection: Bool) -> String? {
        var q = baseQuery(account, dataProtection: dataProtection)
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data, let string = String(data: data, encoding: .utf8),
              !string.isEmpty else { return nil }
        return string
    }

    @discardableResult
    private static func write(_ value: String, _ account: String, dataProtection: Bool) -> Bool {
        let data = Data(value.utf8)
        var base = baseQuery(account, dataProtection: dataProtection)
        let update = SecItemUpdate(base as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return true }
        base[kSecValueData as String] = data
        base[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(base as CFDictionary, nil) == errSecSuccess
    }

    static func get(_ account: String) -> String? {
        // Prefer the prompt-free data-protection keychain.
        if let v = read(account, dataProtection: true) { return v }
        // Fall back to a legacy item (may prompt once) and migrate it forward so
        // the next read is prompt-free.
        if let v = read(account, dataProtection: false) {
            _ = write(v, account, dataProtection: true)
            return v
        }
        return nil
    }

    @discardableResult
    static func set(_ value: String?, _ account: String) -> Bool {
        // Clearing the value deletes the item (from both keychains).
        guard let value, !value.isEmpty else { return delete(account) }
        return write(value, account, dataProtection: true)
    }

    @discardableResult
    static func delete(_ account: String) -> Bool {
        let dp = SecItemDelete(baseQuery(account, dataProtection: true) as CFDictionary)
        let legacy = SecItemDelete(baseQuery(account, dataProtection: false) as CFDictionary)
        let ok: (OSStatus) -> Bool = { $0 == errSecSuccess || $0 == errSecItemNotFound }
        return ok(dp) && ok(legacy)
    }
}

/// Named application secrets, backed by the Keychain.
enum AppSecrets {
    enum Key: String { case civitai, replicate, fal }

    /// Standard environment-variable names, used as a fallback when the Keychain
    /// has no value (e.g. the app launched from a shell that exports them).
    private static let envVarName: [Key: String] = [
        .civitai: "CIVITAI_API_KEY",
        .replicate: "REPLICATE_API_TOKEN",
        .fal: "FAL_KEY",
    ]

    static func value(_ key: Key) -> String? {
        if let k = Keychain.get(key.rawValue), !k.isEmpty { return k }
        if let name = envVarName[key],
           let v = ProcessInfo.processInfo.environment[name], !v.isEmpty { return v }
        return nil
    }

    static func set(_ key: Key, _ value: String?) { Keychain.set(value, key.rawValue) }

    static var civitai: String? { value(.civitai) }
    static var replicate: String? { value(.replicate) }
    static var fal: String? { value(.fal) }

    /// Move any plaintext keys from the JSON config into the Keychain (once),
    /// then clear them from the JSON. Idempotent.
    static func migrateFromSettingsIfNeeded() {
        var settings = DesktopSettings.load()
        var changed = false
        if let k = settings.civitaiApiKey, !k.isEmpty {
            set(.civitai, k); settings.civitaiApiKey = nil; changed = true
        }
        if let k = settings.replicateApiKey, !k.isEmpty {
            set(.replicate, k); settings.replicateApiKey = nil; changed = true
        }
        if let k = settings.falApiKey, !k.isEmpty {
            set(.fal, k); settings.falApiKey = nil; changed = true
        }
        if changed { settings.save() }

        // Seed the Keychain from environment variables when it's empty, so a
        // key exported in the shell persists for later Finder launches too.
        for key in [Key.civitai, .replicate, .fal] {
            if Keychain.get(key.rawValue) == nil,
               let name = envVarName[key],
               let v = ProcessInfo.processInfo.environment[name], !v.isEmpty {
                set(key, v)
            }
        }
    }
}
