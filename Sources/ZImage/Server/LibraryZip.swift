// LibraryZip.swift — a minimal, in-process zip reader for library packs.
//
// The first cut shelled out to `/usr/bin/ditto`. On the command line that
// unpacks the 145 MB reference pack in 0.4 s; spawned from inside the running
// engine the same call never returned, while `/health` kept answering — so the
// server sat on a stuck child for as long as the caller waited. Rather than
// debug a subprocess we do not need, this reads the archive directly with
// Apple's compression library. It also matches the repo rule that ComfyBox is
// self-standing Swift (no external tools on a serving path).
//
// Scope: exactly what a `.soslibrary`/`.cblibrary` pack needs — stored (0) and
// deflate (8) entries, no encryption, no zip64, no symlinks. Anything else is
// reported as an unsupported entry rather than silently skipped.
//
// Path safety: an entry that escapes the destination (`../`, absolute) is
// refused. A pack is downloaded from the internet; it does not get to write
// outside the directory we chose.

import Compression
import Foundation

public enum LibraryZipError: Error, LocalizedError, Equatable {
  case notAZip
  case truncated
  case unsupportedEntry(name: String, method: UInt16)
  case unsafePath(String)
  case inflateFailed(String)

  public var errorDescription: String? {
    switch self {
    case .notAZip: return "not a zip archive (no end-of-central-directory record)"
    case .truncated: return "zip archive is truncated"
    case .unsupportedEntry(let name, let method):
      return "zip entry '\(name)' uses unsupported compression method \(method)"
    case .unsafePath(let name): return "zip entry '\(name)' escapes the destination directory"
    case .inflateFailed(let name): return "could not inflate zip entry '\(name)'"
    }
  }
}

public enum LibraryZip {

  /// Extract `archive` into `destination` (created if needed).
  /// Returns the number of files written.
  @discardableResult
  public static func extract(_ archive: URL, to destination: URL) throws -> Int {
    let data = try Data(contentsOf: archive, options: .mappedIfSafe)
    return try extract(data: data, to: destination)
  }

  @discardableResult
  public static func extract(data: Data, to destination: URL) throws -> Int {
    let fm = FileManager.default
    try fm.createDirectory(at: destination, withIntermediateDirectories: true)
    let canonicalRoot = destination.standardizedFileURL.path

    var written = 0
    for entry in try entries(in: data) {
      // Directories are implied by the files inside them.
      if entry.name.hasSuffix("/") { continue }
      // Skip the resource forks ditto writes into its own archives.
      if entry.name.hasPrefix("__MACOSX/") || entry.name.contains("/._") { continue }

      let target = destination.appendingPathComponent(entry.name).standardizedFileURL
      guard target.path == canonicalRoot || target.path.hasPrefix(canonicalRoot + "/") else {
        throw LibraryZipError.unsafePath(entry.name)
      }
      let bytes = try payload(of: entry, in: data)
      try fm.createDirectory(
        at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
      try bytes.write(to: target, options: .atomic)
      written += 1
    }
    return written
  }

  // MARK: - Central directory

  struct Entry {
    let name: String
    let method: UInt16
    let compressedSize: Int
    let uncompressedSize: Int
    let localHeaderOffset: Int
  }

  static let endOfCentralDirectorySignature: UInt32 = 0x0605_4b50
  static let centralFileHeaderSignature: UInt32 = 0x0201_4b50
  static let localFileHeaderSignature: UInt32 = 0x0403_4b50

  static func entries(in data: Data) throws -> [Entry] {
    guard data.count > 22 else { throw LibraryZipError.notAZip }
    // The end record is at the tail, after an optional comment (<= 64 KB).
    let searchStart = max(0, data.count - 22 - 65_536)
    var eocd: Int?
    var index = data.count - 22
    while index >= searchStart {
      if read32(data, index) == endOfCentralDirectorySignature { eocd = index; break }
      index -= 1
    }
    guard let eocd else { throw LibraryZipError.notAZip }

    let count = Int(read16(data, eocd + 10))
    var offset = Int(read32(data, eocd + 16))
    var entries: [Entry] = []
    entries.reserveCapacity(count)
    for _ in 0..<count {
      guard offset + 46 <= data.count, read32(data, offset) == centralFileHeaderSignature else {
        throw LibraryZipError.truncated
      }
      let method = read16(data, offset + 10)
      let compressed = Int(read32(data, offset + 20))
      let uncompressed = Int(read32(data, offset + 24))
      let nameLength = Int(read16(data, offset + 28))
      let extraLength = Int(read16(data, offset + 30))
      let commentLength = Int(read16(data, offset + 32))
      let localOffset = Int(read32(data, offset + 42))
      let nameStart = offset + 46
      guard nameStart + nameLength <= data.count else { throw LibraryZipError.truncated }
      let name = String(
        decoding: data[data.startIndex + nameStart ..< data.startIndex + nameStart + nameLength],
        as: UTF8.self)
      entries.append(Entry(
        name: name, method: method, compressedSize: compressed,
        uncompressedSize: uncompressed, localHeaderOffset: localOffset))
      offset = nameStart + nameLength + extraLength + commentLength
    }
    return entries
  }

  /// The decompressed bytes of one entry.
  static func payload(of entry: Entry, in data: Data) throws -> Data {
    let header = entry.localHeaderOffset
    guard header + 30 <= data.count, read32(data, header) == localFileHeaderSignature else {
      throw LibraryZipError.truncated
    }
    // The local header's own name/extra lengths are authoritative for the data
    // offset (the central directory's extra field is often a different size).
    let nameLength = Int(read16(data, header + 26))
    let extraLength = Int(read16(data, header + 28))
    let start = header + 30 + nameLength + extraLength
    let end = start + entry.compressedSize
    guard end <= data.count else { throw LibraryZipError.truncated }
    let slice = data.subdata(in: (data.startIndex + start) ..< (data.startIndex + end))

    switch entry.method {
    case 0:
      return slice
    case 8:
      guard entry.uncompressedSize > 0 else { return Data() }
      return try inflate(slice, expecting: entry.uncompressedSize, name: entry.name)
    default:
      throw LibraryZipError.unsupportedEntry(name: entry.name, method: entry.method)
    }
  }

  /// Raw deflate (no zlib header), which is what zip stores.
  static func inflate(_ input: Data, expecting size: Int, name: String) throws -> Data {
    var output = Data(count: size)
    let produced: Int = try output.withUnsafeMutableBytes { outBuffer in
      guard let outBase = outBuffer.bindMemory(to: UInt8.self).baseAddress else {
        throw LibraryZipError.inflateFailed(name)
      }
      return input.withUnsafeBytes { inBuffer -> Int in
        guard let inBase = inBuffer.bindMemory(to: UInt8.self).baseAddress else { return 0 }
        return compression_decode_buffer(
          outBase, size, inBase, input.count, nil, COMPRESSION_ZLIB)
      }
    }
    guard produced == size else { throw LibraryZipError.inflateFailed(name) }
    return output
  }

  // MARK: - Little-endian reads

  private static func read16(_ data: Data, _ offset: Int) -> UInt16 {
    let i = data.startIndex + offset
    guard i + 1 < data.endIndex else { return 0 }
    return UInt16(data[i]) | (UInt16(data[i + 1]) << 8)
  }

  private static func read32(_ data: Data, _ offset: Int) -> UInt32 {
    let i = data.startIndex + offset
    guard i + 3 < data.endIndex else { return 0 }
    return UInt32(data[i]) | (UInt32(data[i + 1]) << 8)
      | (UInt32(data[i + 2]) << 16) | (UInt32(data[i + 3]) << 24)
  }
}
