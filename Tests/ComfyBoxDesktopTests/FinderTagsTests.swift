// FinderTagsTests.swift — Finder color-label round-trips on real files

import Testing
import Foundation
@testable import ComfyBoxDesktop

@Suite("FinderTags")
struct FinderTagsTests {
    private func tempFile() throws -> String {
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("tag-test-\(UUID().uuidString).png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: URL(fileURLWithPath: path))
        return path
    }

    @Test("set, read, change, and clear a color label")
    func roundTrip() throws {
        let path = try tempFile()
        defer { try? FileManager.default.removeItem(atPath: path) }

        #expect(FinderTags.colorLabel(atPath: path) == nil)

        try FinderTags.setColorLabel(.red, atPath: path)
        #expect(FinderTags.colorLabel(atPath: path) == .red)

        try FinderTags.setColorLabel(.blue, atPath: path)
        #expect(FinderTags.colorLabel(atPath: path) == .blue)

        try FinderTags.setColorLabel(nil, atPath: path)
        #expect(FinderTags.colorLabel(atPath: path) == nil)
    }

    @Test("the xattr uses Finder's exact Name-newline-index format")
    func wireFormat() throws {
        let path = try tempFile()
        defer { try? FileManager.default.removeItem(atPath: path) }
        try FinderTags.setColorLabel(.orange, atPath: path)

        // Read the raw xattr and decode the binary plist Finder reads.
        let name = "com.apple.metadata:_kMDItemUserTags"
        let length = getxattr(path, name, nil, 0, 0, 0)
        #expect(length > 0)
        var data = Data(count: length)
        _ = data.withUnsafeMutableBytes { getxattr(path, name, $0.baseAddress, length, 0, 0) }
        let tags = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String]
        #expect(tags == ["Orange\n7"])
    }

    @Test("non-color user tags survive label changes")
    func preservesUserTags() throws {
        let path = try tempFile()
        defer { try? FileManager.default.removeItem(atPath: path) }

        // Simulate a user tag written by Finder alongside a color.
        let plist = try PropertyListSerialization.data(
            fromPropertyList: ["Vacation", "Green\n2"], format: .binary, options: 0)
        _ = plist.withUnsafeBytes {
            setxattr(path, "com.apple.metadata:_kMDItemUserTags", $0.baseAddress, plist.count, 0, 0)
        }
        #expect(FinderTags.colorLabel(atPath: path) == .green)

        try FinderTags.setColorLabel(.red, atPath: path)
        #expect(FinderTags.colorLabel(atPath: path) == .red)

        try FinderTags.setColorLabel(nil, atPath: path)
        #expect(FinderTags.colorLabel(atPath: path) == nil)

        // "Vacation" is still there.
        let length = getxattr(path, "com.apple.metadata:_kMDItemUserTags", nil, 0, 0, 0)
        var data = Data(count: length)
        _ = data.withUnsafeMutableBytes {
            getxattr(path, "com.apple.metadata:_kMDItemUserTags", $0.baseAddress, length, 0, 0)
        }
        let tags = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String]
        #expect(tags == ["Vacation"])
    }

    @Test("plain-name color tags (no index) are recognized")
    func plainNameTags() throws {
        let path = try tempFile()
        defer { try? FileManager.default.removeItem(atPath: path) }
        let plist = try PropertyListSerialization.data(
            fromPropertyList: ["Purple"], format: .binary, options: 0)
        _ = plist.withUnsafeBytes {
            setxattr(path, "com.apple.metadata:_kMDItemUserTags", $0.baseAddress, plist.count, 0, 0)
        }
        #expect(FinderTags.colorLabel(atPath: path) == .purple)
    }

    @Test("keyboard order covers all seven colors starting with red")
    func keyboardOrder() {
        #expect(FinderColor.keyboardOrder.count == 7)
        #expect(FinderColor.keyboardOrder.first == .red)
        #expect(Set(FinderColor.keyboardOrder) == Set(FinderColor.allCases))
    }
}
