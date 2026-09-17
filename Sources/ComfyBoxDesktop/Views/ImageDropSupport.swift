// ImageDropSupport.swift — shared drag-and-drop-a-PNG handling for Generate
// (img2img reference) and Motion (I2V reference).

import Foundation
import UniformTypeIdentifiers

/// Accepts a dropped image (from Finder, Gallery, another app) and hands its
/// resolved local file path to `apply`. Prefers a file URL (regular files,
/// drags from this app's own Gallery); falls back to writing raw image data
/// to a temp file for sources that only expose bytes (e.g. some browser
/// drags). Returns whether the drop was recognized as an image at all —
/// SwiftUI's `.onDrop` uses this to decide whether to accept the drop.
@MainActor
func handleImageDrop(_ providers: [NSItemProvider], apply: @escaping (String) -> Void) -> Bool {
    guard let provider = providers.first else { return false }

    if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
        _ = provider.loadObject(ofClass: URL.self) { url, _ in
            guard let url else { return }
            DispatchQueue.main.async { apply(url.path) }
        }
        return true
    }

    if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
        provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
            guard let data else { return }
            let tempPath = NSTemporaryDirectory() + "dropped-\(UUID().uuidString).png"
            guard (try? data.write(to: URL(fileURLWithPath: tempPath))) != nil else { return }
            DispatchQueue.main.async { apply(tempPath) }
        }
        return true
    }

    return false
}

/// Multi-image variant of `handleImageDrop` for drops that may carry several
/// files at once (the Director keyframe track). Resolves every provider —
/// file URLs first, raw image bytes written to temp PNGs — and calls `apply`
/// ONCE on the main queue with the resolved paths in provider order.
/// Returns whether any provider looked like an image.
@MainActor
func handleImageDrops(_ providers: [NSItemProvider], apply: @escaping ([String]) -> Void) -> Bool {
    let group = DispatchGroup()
    let lock = NSLock()
    var resolved: [Int: String] = [:]
    var accepted = false

    for (index, provider) in providers.enumerated() {
        if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            accepted = true
            group.enter()
            _ = provider.loadObject(ofClass: URL.self) { url, _ in
                if let url {
                    lock.lock(); resolved[index] = url.path; lock.unlock()
                }
                group.leave()
            }
        } else if provider.hasItemConformingToTypeIdentifier(UTType.image.identifier) {
            accepted = true
            group.enter()
            provider.loadDataRepresentation(forTypeIdentifier: UTType.image.identifier) { data, _ in
                if let data {
                    let tempPath = NSTemporaryDirectory() + "dropped-\(UUID().uuidString).png"
                    if (try? data.write(to: URL(fileURLWithPath: tempPath))) != nil {
                        lock.lock(); resolved[index] = tempPath; lock.unlock()
                    }
                }
                group.leave()
            }
        }
    }

    guard accepted else { return false }
    group.notify(queue: .main) {
        lock.lock()
        let paths = resolved.keys.sorted().compactMap { resolved[$0] }
        lock.unlock()
        if !paths.isEmpty { apply(paths) }
    }
    return true
}
