// StudioPackLibrary.swift — Loads built-in + user Studio Packs.
//
// Built-in packs (BuiltInStudioPacks.all) are always present. User packs are
// JSON files under a pack directory (default ~/.comfybox/studio-packs); a
// user pack whose id matches a built-in pack overrides it entirely, letting
// Todd customize a shipped pack without forking source.

import Foundation

public enum StudioPackLibraryError: Error, LocalizedError {
  case invalidPackFile(URL, Error)

  public var errorDescription: String? {
    switch self {
    case .invalidPackFile(let url, let error):
      return "Invalid Studio Pack file \(url.lastPathComponent): \(error.localizedDescription)"
    }
  }
}

public enum StudioPackLibrary {
  /// Default location for user-authored/overridden packs.
  public static var defaultPackDirectory: URL {
    FileManager.default
      .homeDirectoryForCurrentUser
      .appendingPathComponent(".comfybox")
      .appendingPathComponent("studio-packs")
  }

  /// Load all available packs: built-ins first, then user packs merged in
  /// (a user pack with a matching id replaces the built-in entirely).
  ///
  /// - Parameter directory: Directory to scan for user pack JSON files.
  ///   Missing/absent directories are not an error — built-ins are returned.
  /// - Returns: Packs sorted by id, plus any per-file load errors encountered.
  public static func loadAll(
    from directory: URL = defaultPackDirectory
  ) -> (packs: [StudioPack], errors: [StudioPackLibraryError]) {
    var byId: [String: StudioPack] = [:]
    for pack in BuiltInStudioPacks.all {
      byId[pack.id] = pack
    }

    var errors: [StudioPackLibraryError] = []
    let fm = FileManager.default
    guard let files = try? fm.contentsOfDirectory(
      at: directory, includingPropertiesForKeys: nil
    ) else {
      // No user pack directory yet — built-ins only, not an error.
      return (byId.values.sorted { $0.id < $1.id }, [])
    }

    let decoder = JSONDecoder()
    for file in files.filter({ $0.pathExtension.lowercased() == "json" }).sorted(by: { $0.path < $1.path }) {
      do {
        let data = try Data(contentsOf: file)
        let pack = try decoder.decode(StudioPack.self, from: data)
        byId[pack.id] = pack
      } catch {
        errors.append(.invalidPackFile(file, error))
      }
    }

    return (byId.values.sorted { $0.id < $1.id }, errors)
  }

  /// Look up a single pack by id from the merged built-in + user set.
  public static func pack(
    id: String, from directory: URL = defaultPackDirectory
  ) -> StudioPack? {
    loadAll(from: directory).packs.first { $0.id == id }
  }
}
