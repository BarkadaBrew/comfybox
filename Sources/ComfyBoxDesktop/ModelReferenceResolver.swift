// ModelReferenceResolver.swift — Resolve a short, embedded-metadata model
// name (e.g. "krea-2-turbo", "cyberrealisticZImage_v50") against either the
// currently active model or the known catalog (coffeeshop-server#1180:
// Gallery → Generate must show the image's real model, not a stale default).
//
// Embedded PNG metadata only ever stores a display name — never the
// server's full path/spec — so a raw string comparison against
// engine.currentModel almost always mismatches even when the image's model
// IS already active. Pure logic, extracted for direct unit testing.

import Foundation

public enum ModelReferenceResolution: Equatable {
  case alreadyActive
  case resolved(String)
  case unresolved
}

public enum ModelReferenceResolver {
  /// - Parameters:
  ///   - raw: The short model name from embedded metadata / a saved preset.
  ///   - currentModel: The server's currently active model (a full path or spec).
  ///   - availableModels: The known model catalog, each with an id and display name.
  public static func resolve(
    _ raw: String, currentModel: String?, availableModels: [(id: String, displayName: String)]
  ) -> ModelReferenceResolution {
    func normalize(_ s: String) -> String { s.lowercased().filter { $0.isLetter || $0.isNumber } }
    let target = normalize(raw)
    guard !target.isEmpty else { return .unresolved }

    if let current = currentModel {
      let stem = ((current as NSString).lastPathComponent as NSString).deletingPathExtension
      if normalize(stem) == target || normalize(current) == target {
        return .alreadyActive
      }
    }
    if let match = availableModels.first(where: {
      normalize($0.id) == target || normalize($0.displayName) == target
    }) {
      return .resolved(match.id)
    }
    return .unresolved
  }
}
