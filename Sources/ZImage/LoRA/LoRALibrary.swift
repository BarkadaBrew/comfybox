// LoRALibrary.swift — Core LoRA library manager
//
// Part of the LoRA Library Manager (#73). Manages the library.json index
// file, provides query methods, filesystem scanning, and metadata updates.
//
// Thread-safe via NSLock. The library root defaults to ~/Models/loras/
// and can be overridden via the COMFYBOX_MODELS environment variable.

import Foundation
import Logging

// MARK: - Scan Result

/// Summary of a library scan operation.
public struct LoRALibraryScanResult: Sendable {
  /// Number of new LoRAs discovered and added.
  public let added: Int
  /// Number of existing entries updated (file changed on disk).
  public let updated: Int
  /// Number of entries removed (file no longer exists).
  public let removed: Int
  /// Number of entries unchanged.
  public let unchanged: Int
  /// Errors encountered during scanning (non-fatal).
  public let errors: [(String, String)]

  /// Total entries in the library after scan.
  public var total: Int { added + updated + unchanged }
}

// MARK: - Library Errors

public enum LoRALibraryError: Error, LocalizedError {
  case libraryRootNotFound(String)
  case entryNotFound(String)
  case saveFailed(Error)
  case loadFailed(Error)

  public var errorDescription: String? {
    switch self {
    case .libraryRootNotFound(let path):
      return "Library root not found: \(path)"
    case .entryNotFound(let id):
      return "LoRA entry not found: \(id)"
    case .saveFailed(let error):
      return "Failed to save library: \(error.localizedDescription)"
    case .loadFailed(let error):
      return "Failed to load library: \(error.localizedDescription)"
    }
  }
}

// MARK: - Library Manager

/// Manages the LoRA library index (library.json) and provides query, scan,
/// and update operations.
///
/// The library root directory contains LoRA files in subdirectories.
/// `library.json` at the root stores the index. Scanning walks the filesystem
/// recursively, analyzes each .safetensors file via `LoRAScanner`, and
/// preserves user-edited metadata for existing entries.
public final class LoRALibrary: @unchecked Sendable {

  private let libraryRoot: URL
  private let indexPath: URL
  private var entries: [String: LoRALibraryEntry]  // keyed by id
  private let lock = NSLock()
  private let logger: Logger

  /// File manager for filesystem operations.
  private let fm = FileManager.default

  // MARK: - Initialization

  /// Create a library manager for the given root directory.
  ///
  /// - Parameters:
  ///   - root: The library root directory (default: ~/Models/loras or COMFYBOX_MODELS env).
  ///   - logger: Logger instance for status messages.
  public init(root: URL? = nil, logger: Logger = Logger(label: "lora-library")) throws {
    let resolvedRoot: URL
    if let root {
      resolvedRoot = root
    } else if let envPath = ProcessInfo.processInfo.environment["COMFYBOX_MODELS"], !envPath.isEmpty {
      resolvedRoot = URL(fileURLWithPath: (envPath as NSString).expandingTildeInPath)
    } else {
      let home = fm.homeDirectoryForCurrentUser
      resolvedRoot = home.appendingPathComponent("Models/loras")
    }

    guard fm.fileExists(atPath: resolvedRoot.path) else {
      throw LoRALibraryError.libraryRootNotFound(resolvedRoot.path)
    }

    self.libraryRoot = resolvedRoot
    self.indexPath = resolvedRoot.appendingPathComponent("library.json")
    self.entries = [:]
    self.logger = logger

    try loadIndex()
  }

  // MARK: - Index Persistence

  /// Load the library index from disk.
  private func loadIndex() throws {
    lock.lock()
    defer { lock.unlock() }

    guard fm.fileExists(atPath: indexPath.path) else {
      entries = [:]
      return
    }

    do {
      let data = try Data(contentsOf: indexPath)
      let decoder = JSONDecoder()
      let index = try decoder.decode(LoRALibraryIndex.self, from: data)
      entries = [:]
      for entry in index.entries {
        entries[entry.id] = entry
      }
      logger.info("Loaded \(entries.count) entries from library.json")
    } catch {
      throw LoRALibraryError.loadFailed(error)
    }
  }

  /// Save the library index to disk.
  private func saveIndex() throws {
    // Caller must hold lock
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime]
    let now = formatter.string(from: Date())

    let sortedEntries = entries.values.sorted { $0.id < $1.id }
    let index = LoRALibraryIndex(
      version: 1,
      updatedAt: now,
      entries: sortedEntries
    )

    do {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let data = try encoder.encode(index)
      try data.write(to: indexPath, options: .atomic)
    } catch {
      throw LoRALibraryError.saveFailed(error)
    }
  }

  // MARK: - Query

  /// List all entries, optionally filtered by compatibility, tags, or quarantine status.
  ///
  /// - Parameters:
  ///   - compatibility: Filter to entries compatible with this model (e.g. "z-image").
  ///   - tags: Filter to entries with any of these tags.
  ///   - includeQuarantined: Whether to include quarantined entries (default: false).
  /// - Returns: Matching entries sorted by ID.
  public func list(
    compatibility: String? = nil,
    tags: [String]? = nil,
    includeQuarantined: Bool = false
  ) -> [LoRALibraryEntry] {
    lock.lock()
    defer { lock.unlock() }

    var result = Array(entries.values)

    if !includeQuarantined {
      result = result.filter { !$0.quarantined }
    }

    if let compat = compatibility {
      let lower = compat.lowercased()
      result = result.filter { entry in
        entry.modelCompatibility.contains { $0.lowercased() == lower }
      }
    }

    if let tags, !tags.isEmpty {
      let tagSet = Set(tags.map { $0.lowercased() })
      result = result.filter { entry in
        entry.tags.contains { tagSet.contains($0.lowercased()) }
      }
    }

    return result.sorted { $0.id < $1.id }
  }

  /// Get an entry by ID or filename.
  ///
  /// - Parameter identifier: The entry ID or bare filename.
  /// - Returns: The matching entry, or nil if not found.
  public func entry(for identifier: String) -> LoRALibraryEntry? {
    lock.lock()
    defer { lock.unlock() }

    // Try ID first
    if let entry = entries[identifier] {
      return entry
    }

    // Try ID case-insensitively
    let lower = identifier.lowercased()
    if let entry = entries.values.first(where: { $0.id.lowercased() == lower }) {
      return entry
    }

    // Try filename match
    if let entry = entries.values.first(where: { $0.filename == identifier }) {
      return entry
    }

    // Try filename without extension
    let withoutExt = (identifier as NSString).deletingPathExtension
    if let entry = entries.values.first(where: {
      ($0.filename as NSString).deletingPathExtension == withoutExt
    }) {
      return entry
    }

    return nil
  }

  /// Search entries by text across id, filename, tags, and notes.
  ///
  /// - Parameter query: Search text (case-insensitive).
  /// - Returns: Matching entries sorted by relevance (ID match first).
  public func search(_ query: String) -> [LoRALibraryEntry] {
    lock.lock()
    defer { lock.unlock() }

    let lower = query.lowercased()
    return entries.values.filter { entry in
      entry.id.lowercased().contains(lower) ||
      entry.filename.lowercased().contains(lower) ||
      entry.tags.contains { $0.lowercased().contains(lower) } ||
      entry.notes.lowercased().contains(lower) ||
      entry.category.lowercased().contains(lower)
    }.sorted { a, b in
      // Prioritize ID matches over other fields
      let aIdMatch = a.id.lowercased().contains(lower)
      let bIdMatch = b.id.lowercased().contains(lower)
      if aIdMatch && !bIdMatch { return true }
      if !aIdMatch && bIdMatch { return false }
      return a.id < b.id
    }
  }

  /// Get entries compatible with a given model family.
  ///
  /// - Parameter family: The model family to filter for.
  /// - Returns: Non-quarantined entries compatible with this family.
  public func compatible(with family: ComfyBoxModelFamily) -> [LoRALibraryEntry] {
    let compatStrings = LoRACompatibility.compatibilityStrings(for: family)
    guard !compatStrings.isEmpty else { return [] }

    lock.lock()
    defer { lock.unlock() }

    return entries.values.filter { entry in
      !entry.quarantined &&
      entry.modelCompatibility.contains { compat in
        compatStrings.contains(compat.lowercased())
      }
    }.sorted { $0.id < $1.id }
  }

  /// Resolve a LoRA identifier to its absolute file path.
  ///
  /// - Parameter identifier: Entry ID or filename.
  /// - Returns: The absolute file URL.
  /// - Throws: `LoRALibraryError.entryNotFound` if not in the library.
  public func resolve(_ identifier: String) throws -> URL {
    guard let entry = entry(for: identifier) else {
      throw LoRALibraryError.entryNotFound(identifier)
    }
    return libraryRoot.appendingPathComponent(entry.relativePath)
  }

  /// Get the library root URL.
  public var root: URL { libraryRoot }

  /// Get the total entry count.
  public var count: Int {
    lock.lock()
    defer { lock.unlock() }
    return entries.count
  }

  // MARK: - Scan

  /// Scan the filesystem and rebuild/update the library index.
  ///
  /// Walks the library root recursively, analyzing each .safetensors file.
  /// Existing entries are preserved (user metadata kept), new files are added,
  /// and missing files are removed.
  ///
  /// - Parameter force: If true, re-analyze all files even if size hasn't changed.
  /// - Returns: A summary of changes made during the scan.
  public func scan(force: Bool = false) throws -> LoRALibraryScanResult {
    logger.info("Scanning \(libraryRoot.path)...")

    // Collect all .safetensors files recursively
    let safetensorsFiles = collectSafetensorsFiles(in: libraryRoot)
    logger.info("Found \(safetensorsFiles.count) safetensors files")

    lock.lock()
    let existingEntries = entries
    lock.unlock()

    var newEntries: [String: LoRALibraryEntry] = [:]
    var addedCount = 0
    var updatedCount = 0
    var unchangedCount = 0
    var scanErrors: [(String, String)] = []

    let dateFormatter = ISO8601DateFormatter()
    dateFormatter.formatOptions = [.withFullDate]
    let today = dateFormatter.string(from: Date())

    for fileURL in safetensorsFiles {
      let relativePath = fileURL.path.replacingOccurrences(of: libraryRoot.path + "/", with: "")
      let filename = fileURL.lastPathComponent
      let slug = LoRAScanner.slugify(filename)

      // Ensure unique ID
      var id = slug
      var counter = 2
      while newEntries[id] != nil {
        id = "\(slug)-\(counter)"
        counter += 1
      }

      // Check if we have an existing entry for this file
      let existingEntry = existingEntries.values.first { $0.relativePath == relativePath }
        ?? existingEntries[id]

      // Get file size
      let fileSize: UInt64
      do {
        let attrs = try fm.attributesOfItem(atPath: fileURL.path)
        fileSize = attrs[.size] as? UInt64 ?? 0
      } catch {
        scanErrors.append((relativePath, "Cannot read file attributes: \(error.localizedDescription)"))
        continue
      }

      // Skip re-analysis if file size unchanged and not forced
      if let existing = existingEntry, existing.sizeBytes == fileSize && !force {
        // Preserve existing entry but update id/path if needed
        var preserved = existing
        preserved.id = id
        preserved.relativePath = relativePath
        preserved.filename = filename
        newEntries[id] = preserved
        unchangedCount += 1
        continue
      }

      // Analyze the file
      let scanResult: LoRAScanResult
      do {
        scanResult = try LoRAScanner.analyze(fileURL)
      } catch {
        scanErrors.append((relativePath, error.localizedDescription))
        continue
      }

      // Determine category from parent directory
      let parentDir = fileURL.deletingLastPathComponent().lastPathComponent
      let category = (parentDir == libraryRoot.lastPathComponent) ? "uncategorized" : parentDir

      // Determine quarantine status from path
      let isQuarantined = relativePath.lowercased().contains("quarantine")

      if let existing = existingEntry {
        // Update: preserve user-edited fields, update auto-detected fields
        var updated = existing
        updated.id = id
        updated.filename = filename
        updated.relativePath = relativePath
        updated.sizeBytes = fileSize
        updated.sha256 = nil  // Invalidate hash since file changed
        updated.modelCompatibility = scanResult.compatibility
        updated.format = scanResult.format
        updated.rank = scanResult.rank
        updated.alpha = scanResult.alpha
        updated.keyCount = scanResult.keyCount
        updated.layerTargets = scanResult.layerTargets
        updated.safetensorsMetadata = scanResult.safetensorsMetadata
        updated.category = category
        updated.quarantined = isQuarantined
        // Only backfill auto-extracted trigger words when the entry has none
        // yet — never clobber a user's manually-curated keywords on rescan.
        if updated.triggerwords.allSatisfy({ $0.trimmingCharacters(in: .whitespaces).isEmpty }) {
          updated.triggerwords = scanResult.triggerWords
        }
        newEntries[id] = updated
        updatedCount += 1
      } else {
        // New entry
        let entry = LoRALibraryEntry(
          id: id,
          filename: filename,
          relativePath: relativePath,
          sizeBytes: fileSize,
          sha256: nil,
          modelCompatibility: scanResult.compatibility,
          format: scanResult.format,
          rank: scanResult.rank,
          alpha: scanResult.alpha,
          keyCount: scanResult.keyCount,
          layerTargets: scanResult.layerTargets,
          triggerwords: scanResult.triggerWords,
          recommendedScale: 1.0,
          scaleRange: [0.0, 2.0],
          tags: [],
          category: category,
          notes: "",
          sourceURL: nil,
          civitaiModelId: nil,
          dateAdded: today,
          quarantined: isQuarantined,
          quarantineReason: isQuarantined ? "In quarantine directory" : nil,
          safetensorsMetadata: scanResult.safetensorsMetadata
        )
        newEntries[id] = entry
        addedCount += 1
      }

      logger.info("  \(existingEntry != nil ? "Updated" : "Added"): \(relativePath) [\(scanResult.compatibility.joined(separator: ", ")), \(scanResult.format.rawValue)]")
    }

    // Calculate removed count
    let removedCount = existingEntries.count - (updatedCount + unchangedCount)

    // Commit the new index
    lock.lock()
    entries = newEntries
    do {
      try saveIndex()
    } catch {
      lock.unlock()
      throw error
    }
    lock.unlock()

    let result = LoRALibraryScanResult(
      added: addedCount,
      updated: updatedCount,
      removed: removedCount,
      unchanged: unchangedCount,
      errors: scanErrors
    )

    logger.info("Scan complete: +\(addedCount) added, ~\(updatedCount) updated, -\(removedCount) removed, =\(unchangedCount) unchanged")

    return result
  }

  // MARK: - Mutate

  /// Update metadata for an existing entry.
  ///
  /// Only non-nil fields in the patch are applied. Auto-detected fields
  /// (compatibility, format, rank, etc.) are not affected.
  ///
  /// - Parameters:
  ///   - id: The entry ID.
  ///   - patch: Fields to update.
  /// - Throws: `LoRALibraryError.entryNotFound` if the ID doesn't exist.
  public func update(_ id: String, patch: LoRAEntryPatch) throws {
    lock.lock()
    defer { lock.unlock() }

    guard var entry = entries[id] else {
      throw LoRALibraryError.entryNotFound(id)
    }

    if let triggerwords = patch.triggerwords { entry.triggerwords = triggerwords }
    if let scale = patch.recommendedScale { entry.recommendedScale = scale }
    if let range = patch.scaleRange { entry.scaleRange = range }
    if let tags = patch.tags { entry.tags = tags }
    if let notes = patch.notes { entry.notes = notes }
    if let url = patch.sourceURL { entry.sourceURL = url }
    if let civitaiId = patch.civitaiModelId { entry.civitaiModelId = civitaiId }

    entries[id] = entry
    try saveIndex()
  }

  /// Quarantine a LoRA entry.
  ///
  /// Marks the entry as quarantined with an optional reason. Does not move
  /// the file on disk (flagging only per design decision).
  ///
  /// - Parameters:
  ///   - id: The entry ID.
  ///   - reason: Why this LoRA is quarantined.
  /// - Throws: `LoRALibraryError.entryNotFound` if the ID doesn't exist.
  public func quarantine(_ id: String, reason: String) throws {
    lock.lock()
    defer { lock.unlock() }

    guard var entry = entries[id] else {
      throw LoRALibraryError.entryNotFound(id)
    }

    entry.quarantined = true
    entry.quarantineReason = reason
    entries[id] = entry
    try saveIndex()

    logger.info("Quarantined: \(entry.filename) — \(reason)")
  }

  /// Un-quarantine a LoRA entry.
  ///
  /// Clears the quarantine flag and reason. Does not move the file on disk.
  ///
  /// - Parameter id: The entry ID.
  /// - Throws: `LoRALibraryError.entryNotFound` if the ID doesn't exist.
  public func unquarantine(_ id: String) throws {
    lock.lock()
    defer { lock.unlock() }

    guard var entry = entries[id] else {
      throw LoRALibraryError.entryNotFound(id)
    }

    entry.quarantined = false
    entry.quarantineReason = nil
    entries[id] = entry
    try saveIndex()

    logger.info("Un-quarantined: \(entry.filename)")
  }

  // MARK: - SHA-256

  /// Compute and cache the SHA-256 hash for an entry.
  ///
  /// - Parameter id: The entry ID.
  /// - Returns: The hex-encoded SHA-256 hash.
  public func computeHash(for id: String) throws -> String {
    lock.lock()
    guard var entry = entries[id] else {
      lock.unlock()
      throw LoRALibraryError.entryNotFound(id)
    }

    if let existing = entry.sha256 {
      lock.unlock()
      return existing
    }
    lock.unlock()

    let fileURL = libraryRoot.appendingPathComponent(entry.relativePath)
    let hash = try sha256Hash(of: fileURL)

    lock.lock()
    entry.sha256 = hash
    entries[id] = entry
    do { try saveIndex() } catch { /* non-fatal, hash will be recomputed */ }
    lock.unlock()

    return hash
  }

  // MARK: - Filesystem Helpers

  /// Recursively collect all .safetensors files under a directory.
  private func collectSafetensorsFiles(in directory: URL) -> [URL] {
    var results: [URL] = []

    guard let enumerator = fm.enumerator(
      at: directory,
      includingPropertiesForKeys: [.isRegularFileKey],
      options: [.skipsHiddenFiles]
    ) else {
      return results
    }

    for case let fileURL as URL in enumerator {
      if fileURL.pathExtension.lowercased() == "safetensors" {
        results.append(fileURL)
      }
    }

    return results.sorted { $0.path < $1.path }
  }

  /// Compute SHA-256 hash of a file using CommonCrypto.
  private func sha256Hash(of url: URL) throws -> String {
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    // Use CC_SHA256 via bridging
    var hash = [UInt8](repeating: 0, count: 32)
    data.withUnsafeBytes { buffer in
      _ = CC_SHA256(buffer.baseAddress, CC_LONG(buffer.count), &hash)
    }
    return hash.map { String(format: "%02x", $0) }.joined()
  }
}

// CommonCrypto bridging for SHA-256
import CommonCrypto
