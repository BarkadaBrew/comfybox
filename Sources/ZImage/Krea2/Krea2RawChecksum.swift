// Krea2RawChecksum.swift — SHA-256 pin support for the Krea 2 Raw DiT
// (WP-E10 "E9b", FDD §7.1 row "Krea 2 Raw DiT", Addendum A.2).
//
// `~/LocalModels/krea2-raw/raw.safetensors` is the Comfy-Org mirror's
// `krea2_raw_bf16.safetensors` (the official repo is gated; the direct fetch
// returned a 149-byte body). Its SHA-256 is pinned in
// `Tests/ZImageTests/Fixtures/krea2-raw.sha256` in `shasum -a 256` format and
// asserted by `ZImageIntegrationTests.Krea2RawChecksumTests`, so a silently
// replaced or truncated file never passes as "Raw".

import CryptoKit
import Foundation

public enum Krea2RawChecksum {

  public struct Pin: Equatable, Sendable {
    /// Lowercase 64-hex digest.
    public let sha256: String
    public let filename: String
  }

  public enum ParseError: Error, Equatable, CustomStringConvertible {
    case malformed(String)
    public var description: String {
      switch self {
      case .malformed(let line):
        return "not a `shasum -a 256` line (<64 hex>  <filename>): '\(line)'"
      }
    }
  }

  /// Parse one `shasum -a 256` output line: `<64 hex>  <filename>`.
  /// Whitespace-tolerant, hex normalised to lowercase; anything else throws.
  public static func parseShasumLine(_ text: String) throws -> Pin {
    let line = text.trimmingCharacters(in: .whitespacesAndNewlines)
    let parts = line.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
    guard parts.count == 2 else { throw ParseError.malformed(line) }
    let hex = parts[0].lowercased()
    guard hex.count == 64, hex.allSatisfy({ $0.isHexDigit }) else { throw ParseError.malformed(line) }
    // shasum prefixes a binary-mode filename with '*'; strip it.
    var name = parts[1]
    if name.hasPrefix("*") { name.removeFirst() }
    guard !name.isEmpty else { throw ParseError.malformed(line) }
    return Pin(sha256: hex, filename: name)
  }

  /// Streaming SHA-256 of a file (lowercase hex). Reads in 8 MiB chunks so a
  /// 26 GB checkpoint never lands in memory at once.
  public static func sha256Hex(of file: URL, chunkBytes: Int = 8 << 20) throws -> String {
    let handle = try FileHandle(forReadingFrom: file)
    defer { try? handle.close() }
    var hasher = SHA256()
    while true {
      let chunk = try handle.read(upToCount: chunkBytes) ?? Data()
      if chunk.isEmpty { break }
      hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}
