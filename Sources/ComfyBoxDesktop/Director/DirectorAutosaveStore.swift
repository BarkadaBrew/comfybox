// DirectorAutosaveStore.swift — debounced autosave of the Director tab's open
// timeline (WP3).
//
// Lives at ~/.comfybox/director-autosave.cbdirector: a desktop-owned file
// beside prompt-library.json, never the engine's config.json/presets.json.
// (FDD §4.4 said Application Support; recorded as a Phase 1 delta.) The file
// is an ordinary .cbdirector written through DirectorDocument, never with
// embedded assets.

import Foundation
import ZImage

@MainActor
final class DirectorAutosaveStore {

    nonisolated static func defaultPath() -> URL {
        URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
            .appendingPathComponent(".comfybox/director-autosave.cbdirector")
    }

    let path: URL
    let debounce: TimeInterval

    /// Completed writes (tests assert debouncing coalesces).
    private(set) var writeCount = 0
    private var pending: DirectorTimeline?
    private var debounceTask: Task<Void, Never>?

    init(path: URL = DirectorAutosaveStore.defaultPath(), debounce: TimeInterval = 1.0) {
        self.path = path
        self.debounce = debounce
    }

    /// Remember `timeline` and write it after `debounce` seconds of quiet
    /// (immediately when `debounce <= 0`). A newer schedule replaces an
    /// older pending one.
    func schedule(_ timeline: DirectorTimeline) {
        pending = timeline
        debounceTask?.cancel()
        guard debounce > 0 else {
            flushNow()
            return
        }
        let delay = debounce
        debounceTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            self?.flushNow()
        }
    }

    /// Write the pending timeline now, if any.
    func flushNow() {
        debounceTask?.cancel()
        debounceTask = nil
        guard let timeline = pending else { return }
        pending = nil
        do {
            try FileManager.default.createDirectory(
                at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
            try DirectorDocument.write(timeline, to: path, embedAssets: false)
            writeCount += 1
        } catch {
            // Autosave is best-effort; the explicit Save surfaces errors.
        }
    }

    func load() -> DirectorTimeline? {
        guard FileManager.default.fileExists(atPath: path.path) else { return nil }
        return try? DirectorDocument.read(from: path)
    }

    /// Drop any pending write and delete the file.
    func clear() {
        debounceTask?.cancel()
        debounceTask = nil
        pending = nil
        try? FileManager.default.removeItem(at: path)
    }
}
