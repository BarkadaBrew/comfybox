// GalleryStore.swift — named, independently-addressable render/browse scopes
// ("remote galleries"). A caller (Kira, Desktop, any /v1/generate client) can
// target one by id via GeneratePayload.gallery instead of rendering into the
// default allowedOutputDirectory, and Desktop's gallery hub lists/creates/
// deletes them through the /v1/galleries routes.
//
// Mirrors PresetStore's shape: a plain lock-guarded class persisted to
// ~/.comfybox/galleries.json, not an actor — every route handler in
// WarmServer.swift that touches it stays synchronous, like the Presets ones.

import Foundation
import CryptoKit

/// One named gallery: an isolated output directory, optionally password-locked
/// and/or hidden from discovery listings (a hidden gallery is still directly
/// reachable by id + password — "hidden" means omitted from GET /v1/galleries,
/// not inaccessible).
public struct GalleryEntry: Codable, Sendable, Equatable {
  public let id: String
  public var name: String
  public var directoryPath: String
  public var hidden: Bool
  public var passwordHash: String?
  public let createdAt: Date

  public init(
    id: String, name: String, directoryPath: String, hidden: Bool,
    passwordHash: String?, createdAt: Date
  ) {
    self.id = id
    self.name = name
    self.directoryPath = directoryPath
    self.hidden = hidden
    self.passwordHash = passwordHash
    self.createdAt = createdAt
  }

  /// Public-facing view returned by every route — never exposes the password hash.
  public struct Summary: Codable, Sendable, Identifiable, Equatable {
    public let id: String
    public let name: String
    public let hidden: Bool
    public let locked: Bool
    public let createdAt: Date
  }

  public var summary: Summary {
    Summary(id: id, name: name, hidden: hidden, locked: passwordHash != nil, createdAt: createdAt)
  }
}

public enum GalleryStoreError: Error, LocalizedError, Equatable {
  case invalidName
  case duplicateName(String)
  case notFound(String)
  case unauthorized

  public var errorDescription: String? {
    switch self {
    case .invalidName: return "Gallery name cannot be empty"
    case .duplicateName(let name): return "A gallery named '\(name)' already exists"
    case .notFound(let id): return "Gallery not found: \(id)"
    case .unauthorized: return "Incorrect or missing gallery password"
    }
  }
}

/// Persists named galleries to `~/.comfybox/galleries.json`. Thread-safe: all
/// access is guarded by an internal lock (same convention as PresetStore) —
/// safe to hold and call synchronously from inside WarmServerCoordinator (the
/// actor), the same way it already holds LiveHealthState/RenderProgressTracker.
public final class GalleryStore: @unchecked Sendable {

  private let path: URL
  private let fileManager: FileManager
  private let lock = NSLock()
  private var galleries: [GalleryEntry]

  private struct GalleryFile: Codable {
    var galleries: [GalleryEntry]
  }

  /// `~/.comfybox/galleries.json`.
  public static func defaultPath() -> URL {
    ComfyBoxServerConfig.homeDirectory().appendingPathComponent(".comfybox/galleries.json")
  }

  /// Directory a gallery's files are rendered/browsed under: `~/.comfybox/galleries/<id>/`.
  private static func directory(for id: String) -> URL {
    ComfyBoxServerConfig.homeDirectory()
      .appendingPathComponent(".comfybox/galleries/\(id)", isDirectory: true)
  }

  public init(path: URL = GalleryStore.defaultPath(), fileManager: FileManager = .default) {
    self.path = path
    self.fileManager = fileManager
    if fileManager.fileExists(atPath: path.path), let data = try? Data(contentsOf: path) {
      self.galleries = GalleryStore.decode(data)
    } else {
      self.galleries = []
    }
  }

  // MARK: Reads

  /// Gallery summaries, insertion order. Hidden galleries are omitted unless
  /// `includeHidden` — distinguishes Desktop's own-server "manage everything"
  /// hub view from a plain discovery listing.
  public func list(includeHidden: Bool) -> [GalleryEntry.Summary] {
    lock.lock(); defer { lock.unlock() }
    return galleries.filter { includeHidden || !$0.hidden }.map { $0.summary }
  }

  /// Look up a gallery by id and verify its password, if any — the single
  /// gate every list/file/generate route targeting a named gallery goes
  /// through. Throws `.notFound`/`.unauthorized`; never returns a locked
  /// entry without a matching password.
  public func authorize(id: String, password: String?) throws -> GalleryEntry {
    lock.lock(); defer { lock.unlock() }
    guard let entry = galleries.first(where: { $0.id == id }) else {
      throw GalleryStoreError.notFound(id)
    }
    if let hash = entry.passwordHash {
      guard let password, GalleryStore.hash(password) == hash else {
        throw GalleryStoreError.unauthorized
      }
    }
    return entry
  }

  // MARK: Writes

  @discardableResult
  public func create(name: String, hidden: Bool, password: String?) throws -> GalleryEntry {
    let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { throw GalleryStoreError.invalidName }
    lock.lock(); defer { lock.unlock() }
    guard !galleries.contains(where: { $0.name.caseInsensitiveCompare(trimmed) == .orderedSame }) else {
      throw GalleryStoreError.duplicateName(trimmed)
    }
    let id = GalleryStore.slug(for: trimmed, existing: Set(galleries.map { $0.id }))
    let directory = GalleryStore.directory(for: id)
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
    let entry = GalleryEntry(
      id: id, name: trimmed, directoryPath: directory.path, hidden: hidden,
      passwordHash: (password?.isEmpty == false) ? GalleryStore.hash(password!) : nil,
      createdAt: Date()
    )
    galleries.append(entry)
    try GalleryStore.persist(galleries, to: path, fileManager: fileManager)
    return entry
  }

  /// Delete a gallery's registry entry. Its rendered files are left on disk —
  /// this unregisters the gallery, it doesn't destroy media. Requires the
  /// password if the gallery is locked.
  @discardableResult
  public func delete(id: String, password: String?) throws -> GalleryEntry {
    lock.lock(); defer { lock.unlock() }
    guard let entry = galleries.first(where: { $0.id == id }) else {
      throw GalleryStoreError.notFound(id)
    }
    if let hash = entry.passwordHash {
      guard let password, GalleryStore.hash(password) == hash else {
        throw GalleryStoreError.unauthorized
      }
    }
    galleries.removeAll { $0.id == id }
    try GalleryStore.persist(galleries, to: path, fileManager: fileManager)
    return entry
  }

  // MARK: Password hashing

  /// Salted SHA-256, same scheme as Desktop's NSFWGate (NSFWGate.swift) but
  /// server-side — no Keychain, the hash lives in galleries.json.
  private static let passwordSalt = "comfybox.gallery.v1"

  static func hash(_ password: String) -> String {
    let data = Data((passwordSalt + password).utf8)
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  // MARK: Slug generation

  private static func slug(for name: String, existing: Set<String>) -> String {
    let base = name.lowercased()
      .replacingOccurrences(of: "[^a-z0-9]+", with: "-", options: .regularExpression)
      .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
    let root = base.isEmpty ? "gallery" : base
    var candidate = root
    var suffix = 2
    while existing.contains(candidate) {
      candidate = "\(root)-\(suffix)"
      suffix += 1
    }
    return candidate
  }

  // MARK: Persistence helpers

  static func decode(_ data: Data) -> [GalleryEntry] {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .iso8601
    if let file = try? decoder.decode(GalleryFile.self, from: data) {
      return file.galleries
    }
    if let array = try? decoder.decode([GalleryEntry].self, from: data) {
      return array
    }
    return []
  }

  static func persist(_ galleries: [GalleryEntry], to path: URL, fileManager: FileManager) throws {
    let dir = path.deletingLastPathComponent()
    try fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    encoder.dateEncodingStrategy = .iso8601
    let data = try encoder.encode(GalleryFile(galleries: galleries))
    try data.write(to: path, options: .atomic)
  }
}
