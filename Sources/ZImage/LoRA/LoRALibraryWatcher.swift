// LoRALibraryWatcher.swift — auto-rescan the LoRA library on filesystem change
//
// Watches the library root recursively via FSEvents so a file dropped in by
// any means (CivitAI browser download, curl, cp, an MCP/Bree fetch) shows up
// without a manual `lora scan`. Previously only the Desktop CivitAI browser's
// explicit post-download call triggered a rescan — anything else landing in
// the directory (as happened with a manually-fetched LoRA) sat un-indexed
// until someone remembered to run `lora scan` by hand.

import Foundation
import CoreServices
import Logging

/// Debounced, recursive filesystem watcher over a `LoRALibrary`'s root
/// directory. On any change it re-runs `library.scan()` and logs the result.
///
/// `kFSEventStreamCreateFlagIgnoreSelf` is essential, not cosmetic: every
/// scan rewrites `library.json` (and the auto-import step can copy files
/// into `vault/`), which would otherwise retrigger this same watcher and
/// scan forever. Ignoring events from this process means only genuinely
/// external writes (a different process — Desktop app, curl, cp, Bree) fire
/// the callback.
final class LoRALibraryWatcher {
  private var stream: FSEventStreamRef?
  private let library: LoRALibrary
  private let logger: Logger

  init(library: LoRALibrary, queue: DispatchQueue, logger: Logger) {
    self.library = library
    self.logger = logger
    start(root: library.root, queue: queue)
  }

  private func start(root: URL, queue: DispatchQueue) {
    let pathsToWatch = [root.path] as CFArray
    var context = FSEventStreamContext(
      version: 0,
      info: Unmanaged.passUnretained(self).toOpaque(),
      retain: nil, release: nil, copyDescription: nil
    )

    let flags = UInt32(kFSEventStreamCreateFlagIgnoreSelf)
    guard let stream = FSEventStreamCreate(
      kCFAllocatorDefault,
      { (_, clientInfo, _, _, _, _) in
        guard let clientInfo else { return }
        Unmanaged<LoRALibraryWatcher>.fromOpaque(clientInfo).takeUnretainedValue().rescan()
      },
      &context,
      pathsToWatch,
      FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
      /* latency */ 3.0,  // coalesce a burst of writes (e.g. a download) into one callback
      flags
    ) else {
      logger.warning("LoRA Library: failed to create filesystem watcher for \(root.path)")
      return
    }

    self.stream = stream
    FSEventStreamSetDispatchQueue(stream, queue)
    FSEventStreamStart(stream)
    logger.info("LoRA Library: watching \(root.path) for changes (auto-rescan)")
  }

  private func rescan() {
    let library = self.library
    let logger = self.logger
    DispatchQueue.global(qos: .utility).async {
      do {
        let result = try library.scan()
        if result.added > 0 || result.updated > 0 || result.removed > 0 {
          logger.info("LoRA Library: auto-rescan — +\(result.added) added, ~\(result.updated) updated, -\(result.removed) removed")
        }
      } catch {
        logger.warning("LoRA Library: auto-rescan failed — \(error.localizedDescription)")
      }
    }
  }

  deinit {
    if let stream {
      FSEventStreamStop(stream)
      FSEventStreamInvalidate(stream)
      FSEventStreamRelease(stream)
    }
  }
}
