// ComfyBridgeImageCache.swift — Temporary image storage for ComfyUI bridge
//
// Stores images uploaded via PUT /api/etn/image/<id> and serves them back
// via GET /api/etn/image/<id>. Images are keyed by their CRC32 hash ID
// (computed by the Krita plugin from the PNG data).
//
// Uses a temp directory. The directory is cleaned up in deinit and via cleanup().

import Foundation
import Logging

/// Thread-safe temporary image storage for the ComfyUI ETN image transfer protocol.
final class ComfyImageCache {
  private let logger: Logger
  private let cacheDirectory: URL
  private let lock = NSLock()
  /// In-memory index of stored image IDs for fast existence checks.
  private var storedIds: Set<String> = []

  /// Maximum number of cached images before oldest entries are evicted.
  private static let maxCacheEntries = 256

  init(logger: Logger) {
    self.logger = logger

    // Create a dedicated temp directory for bridge images.
    let tempBase = URL(fileURLWithPath: NSTemporaryDirectory())
      .appendingPathComponent("zimage-comfybridge", isDirectory: true)

    do {
      try FileManager.default.createDirectory(at: tempBase, withIntermediateDirectories: true)
    } catch {
      logger.error("ComfyImageCache: failed to create cache directory: \(error)")
    }

    self.cacheDirectory = tempBase
    logger.info("ComfyImageCache: using \(tempBase.path)")
  }

  deinit {
    cleanupDirectory()
  }

  /// Store image data with the given ID. Overwrites any existing image with the same ID.
  /// Returns true on success, false if the write failed.
  @discardableResult
  func store(id: String, data: Data) -> Bool {
    let url = fileURL(for: id)
    do {
      try data.write(to: url, options: .atomic)
      lock.lock()
      storedIds.insert(id)
      let count = storedIds.count
      lock.unlock()

      // Evict oldest files if cache is over the size limit.
      if count > Self.maxCacheEntries {
        evictOldest()
      }
      return true
    } catch {
      logger.error("ComfyImageCache: failed to write image \(id): \(error)")
      return false
    }
  }

  /// Retrieve image data by ID. Returns nil if not found.
  func retrieve(id: String) -> Data? {
    lock.lock()
    let exists = storedIds.contains(id)
    lock.unlock()

    guard exists else { return nil }

    let url = fileURL(for: id)
    return try? Data(contentsOf: url)
  }

  /// Check whether an image with the given ID exists in the cache.
  func contains(id: String) -> Bool {
    lock.lock()
    defer { lock.unlock() }
    return storedIds.contains(id)
  }

  /// Remove a specific image from the cache.
  func remove(id: String) {
    lock.lock()
    storedIds.remove(id)
    lock.unlock()

    let url = fileURL(for: id)
    try? FileManager.default.removeItem(at: url)
  }

  /// Remove all cached images.
  func clear() {
    lock.lock()
    let ids = storedIds
    storedIds.removeAll()
    lock.unlock()

    for id in ids {
      try? FileManager.default.removeItem(at: fileURL(for: id))
    }
    logger.info("ComfyImageCache: cleared \(ids.count) images")
  }

  /// Number of cached images.
  var count: Int {
    lock.lock()
    defer { lock.unlock() }
    return storedIds.count
  }

  /// Explicitly clean up all cached files and the temp directory.
  func cleanup() {
    clear()
    cleanupDirectory()
  }

  // MARK: - Helpers

  private func fileURL(for id: String) -> URL {
    // Sanitize the ID to prevent path traversal.
    let safeId = id.replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "..", with: "_")
    return cacheDirectory.appendingPathComponent(safeId)
  }

  /// Remove the oldest files when the cache exceeds maxCacheEntries.
  private func evictOldest() {
    let fm = FileManager.default
    guard let files = try? fm.contentsOfDirectory(
      at: cacheDirectory,
      includingPropertiesForKeys: [.contentModificationDateKey],
      options: .skipsHiddenFiles
    ) else { return }

    // Sort by modification date ascending (oldest first).
    let sorted = files.compactMap { url -> (URL, Date)? in
      guard let values = try? url.resourceValues(forKeys: [.contentModificationDateKey]),
            let date = values.contentModificationDate else { return nil }
      return (url, date)
    }.sorted { $0.1 < $1.1 }

    let toRemove = sorted.count - Self.maxCacheEntries
    guard toRemove > 0 else { return }

    lock.lock()
    for i in 0..<toRemove {
      let url = sorted[i].0
      let id = url.lastPathComponent
      storedIds.remove(id)
      try? fm.removeItem(at: url)
    }
    lock.unlock()

    logger.info("ComfyImageCache: evicted \(toRemove) oldest images")
  }

  /// Remove the temp directory and all contents.
  private func cleanupDirectory() {
    try? FileManager.default.removeItem(at: cacheDirectory)
  }
}
