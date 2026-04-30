// ComfyBridgeImageCache.swift — Temporary image storage for ComfyUI bridge
//
// Stores images uploaded via PUT /api/etn/image/<id> and serves them back
// via GET /api/etn/image/<id>. Images are keyed by their CRC32 hash ID
// (computed by the Krita plugin from the PNG data).
//
// Uses a temp directory with automatic cleanup on dealloc.

import Foundation
import Logging

/// Thread-safe temporary image storage for the ComfyUI ETN image transfer protocol.
final class ComfyImageCache {
  private let logger: Logger
  private let cacheDirectory: URL
  private let lock = NSLock()
  /// In-memory index of stored image IDs for fast existence checks.
  private var storedIds: Set<String> = []

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

  /// Store image data with the given ID. Overwrites any existing image with the same ID.
  func store(id: String, data: Data) {
    let url = fileURL(for: id)
    do {
      try data.write(to: url, options: .atomic)
      lock.lock()
      storedIds.insert(id)
      lock.unlock()
    } catch {
      logger.error("ComfyImageCache: failed to write image \(id): \(error)")
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

  // MARK: - Helpers

  private func fileURL(for id: String) -> URL {
    // Sanitize the ID to prevent path traversal.
    let safeId = id.replacingOccurrences(of: "/", with: "_")
      .replacingOccurrences(of: "..", with: "_")
    return cacheDirectory.appendingPathComponent(safeId)
  }
}
