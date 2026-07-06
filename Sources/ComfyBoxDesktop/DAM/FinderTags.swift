// FinderTags.swift — Finder color labels as the source of truth
//
// The gallery's color coding is stored as genuine Finder tags in the file's
// `com.apple.metadata:_kMDItemUserTags` extended attribute (a binary plist
// of "Name\nColorIndex" strings — exactly what Finder writes). That means:
// the color shows in Finder and Get Info, survives moves/copies, and a tag
// set in Finder shows up in the gallery. Non-color user tags on the file
// are preserved untouched.

import Foundation
import SwiftUI

/// The seven Finder label colors, with Finder's fixed color indices.
public enum FinderColor: String, CaseIterable, Sendable, Codable {
    case gray = "Gray"
    case green = "Green"
    case purple = "Purple"
    case blue = "Blue"
    case yellow = "Yellow"
    case red = "Red"
    case orange = "Orange"

    /// Finder's color index for `_kMDItemUserTags` entries.
    public var finderIndex: Int {
        switch self {
        case .gray: return 1
        case .green: return 2
        case .purple: return 3
        case .blue: return 4
        case .yellow: return 5
        case .red: return 6
        case .orange: return 7
        }
    }

    public init?(finderIndex: Int) {
        guard let match = Self.allCases.first(where: { $0.finderIndex == finderIndex }) else {
            return nil
        }
        self = match
    }

    /// Photo Mechanic-style ordering for the 1–7 keyboard shortcuts.
    public static let keyboardOrder: [FinderColor] = [
        .red, .orange, .yellow, .green, .blue, .purple, .gray,
    ]

    public var displayColor: Color {
        switch self {
        case .gray: return .gray
        case .green: return .green
        case .purple: return .purple
        case .blue: return .blue
        case .yellow: return .yellow
        case .red: return .red
        case .orange: return .orange
        }
    }
}

public enum FinderTags {
    private static let xattrName = "com.apple.metadata:_kMDItemUserTags"

    /// All raw tag entries ("Name" or "Name\nIndex") on a file.
    private static func rawTags(atPath path: String) -> [String] {
        let length = getxattr(path, xattrName, nil, 0, 0, 0)
        guard length > 0 else { return [] }
        var data = Data(count: length)
        let read = data.withUnsafeMutableBytes { buffer in
            getxattr(path, xattrName, buffer.baseAddress, length, 0, 0)
        }
        guard read == length,
              let plist = try? PropertyListSerialization.propertyList(from: data, format: nil),
              let tags = plist as? [String]
        else { return [] }
        return tags
    }

    private static func writeRawTags(_ tags: [String], atPath path: String) throws {
        if tags.isEmpty {
            removexattr(path, xattrName, 0)
            return
        }
        let data = try PropertyListSerialization.data(
            fromPropertyList: tags, format: .binary, options: 0)
        let result = data.withUnsafeBytes { buffer in
            setxattr(path, xattrName, buffer.baseAddress, data.count, 0, 0)
        }
        guard result == 0 else {
            throw NSError(domain: NSPOSIXErrorDomain, code: Int(errno),
                          userInfo: [NSLocalizedDescriptionKey: "setxattr failed for \(path)"])
        }
    }

    /// The file's plain text tags (excludes color-label tags).
    public static func textTags(atPath path: String) -> [String] {
        rawTags(atPath: path).compactMap { tag in
            let name = String(tag.split(separator: "\n", maxSplits: 1)[0])
            return FinderColor(rawValue: name) == nil ? name : nil
        }
    }

    /// Merge new plain text tags into the file's Finder tags (dedup,
    /// case-insensitive), preserving any existing color-label tags.
    public static func addTextTags(_ newTags: [String], atPath path: String) throws {
        let existing = rawTags(atPath: path)
        let existingLower = Set(existing.map { $0.split(separator: "\n").first.map(String.init)?.lowercased() ?? "" })
        var merged = existing
        for t in newTags {
            let clean = t.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !clean.isEmpty, !existingLower.contains(clean.lowercased()) else { continue }
            merged.append(clean)
        }
        try writeRawTags(merged, atPath: path)
    }

    // MARK: - Caption (custom xattr, Finder-aligned)

    private static let captionXattr = "com.barkadabrew.comfybox.caption"

    public static func caption(atPath path: String) -> String? {
        let length = getxattr(path, captionXattr, nil, 0, 0, 0)
        guard length > 0 else { return nil }
        var data = Data(count: length)
        let read = data.withUnsafeMutableBytes { getxattr(path, captionXattr, $0.baseAddress, length, 0, 0) }
        guard read == length else { return nil }
        return String(data: data, encoding: .utf8)
    }

    public static func setCaption(_ caption: String?, atPath path: String) {
        guard let caption, !caption.isEmpty else { removexattr(path, captionXattr, 0); return }
        let data = Data(caption.utf8)
        _ = data.withUnsafeBytes { setxattr(path, captionXattr, $0.baseAddress, data.count, 0, 0) }
    }

    /// The file's Finder color label, if any (first color tag wins).
    public static func colorLabel(atPath path: String) -> FinderColor? {
        for tag in rawTags(atPath: path) {
            let parts = tag.split(separator: "\n", maxSplits: 1)
            let name = String(parts[0])
            if let color = FinderColor(rawValue: name) {
                return color
            }
            if parts.count > 1, let index = Int(parts[1]), let color = FinderColor(finderIndex: index) {
                return color
            }
        }
        return nil
    }

    /// Set (or clear, with nil) the file's Finder color label. Non-color
    /// user tags are preserved.
    public static func setColorLabel(_ color: FinderColor?, atPath path: String) throws {
        let colorNames = Set(FinderColor.allCases.map(\.rawValue))
        // Keep everything that isn't a color tag.
        var kept = rawTags(atPath: path).filter { tag in
            let name = String(tag.split(separator: "\n", maxSplits: 1)[0])
            return !colorNames.contains(name)
        }
        if let color {
            kept.append("\(color.rawValue)\n\(color.finderIndex)")
        }
        try writeRawTags(kept, atPath: path)
    }
}
