// LoRAImportPlanner.swift — pure planning behind the Models tab's
// "Import LoRA…" action (spec 2026-08-10): resolve a mixed file/folder
// selection into the exact batch the import sheet shows and submits, and
// derive the category choices for the filing dropdown.

import Foundation

enum LoRAImportPlanner {

  /// The resolved import batch: what will be sent, and how much of the
  /// selection was dropped on the floor (non-safetensors, duplicate names).
  struct Expansion: Equatable {
    var files: [URL] = []
    var skipped: Int = 0
  }

  /// Resolve a picker selection into importable files. Folders are walked
  /// recursively (hidden files skipped); anything that isn't a
  /// `.safetensors` counts as skipped. The library keys entries by filename,
  /// so duplicate names across the selection are dropped (first wins) rather
  /// than left to collide server-side. Output is ordered: direct selections
  /// in picker order, folder contents sorted by name.
  static func expand(urls: [URL], fileManager fm: FileManager = .default) -> Expansion {
    var result = Expansion()
    var seenNames = Set<String>()

    func admit(_ url: URL) {
      guard url.pathExtension.lowercased() == "safetensors" else {
        result.skipped += 1
        return
      }
      guard seenNames.insert(url.lastPathComponent).inserted else {
        result.skipped += 1
        return
      }
      result.files.append(url)
    }

    for url in urls {
      var isDirectory: ObjCBool = false
      guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else { continue }
      if isDirectory.boolValue {
        guard
          let walker = fm.enumerator(
            at: url, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles])
        else { continue }
        var found: [URL] = []
        for case let child as URL in walker {
          let isFile = (try? child.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile
          if isFile == true { found.append(child) }
        }
        for child in found.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
          admit(child)
        }
      } else {
        admit(url)
      }
    }
    return result
  }

  /// Category choices for the filing dropdown: `vault` (the default landing
  /// folder) first, then every distinct non-empty category already in the
  /// library, sorted.
  static func categories(from loras: [LoRAInfo]) -> [String] {
    let existing = Set(loras.map(\.category).filter { !$0.isEmpty && $0 != "vault" })
    return ["vault"] + existing.sorted()
  }
}
