// AppSecrets.swift — API keys in the macOS Keychain, not plaintext JSON
//
// The desktop holds several third-party credentials (CivitAI, Replicate, Fal).
// These used to live in ~/.comfybox/desktop-config.json in cleartext; they now
// live in the login Keychain. A one-time migration moves any existing plaintext
// keys over and clears them from the JSON.

import Foundation
import Security

/// Minimal Keychain wrapper for generic-password items keyed by account.
enum Keychain {
    static let service = "com.barkadabrew.comfybox.desktop"

    static func get(_ account: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data, let string = String(data: data, encoding: .utf8),
              !string.isEmpty else { return nil }
        return string
    }

    @discardableResult
    static func set(_ value: String?, _ account: String) -> Bool {
        // Clearing the value deletes the item.
        guard let value, !value.isEmpty else { return delete(account) }
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let data = Data(value.utf8)
        // Try update first, then add.
        let update = SecItemUpdate(base as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return true }
        var add = base
        add[kSecValueData as String] = data
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }

    @discardableResult
    static func delete(_ account: String) -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
        ]
        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
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
