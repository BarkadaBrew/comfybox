// Krea2ModelDetection.swift — Detects Krea-2-Turbo model specs and directories.

import Foundation

public enum Krea2ModelDetection {
  /// Known spec aliases that always mean Krea-2-Turbo.
  public static func isKnownKrea2Model(_ spec: String) -> Bool {
    let lower = spec.lowercased()
    if lower == "krea2" || lower == "krea-2" || lower == "krea-2-turbo" { return true }
    if lower.contains("krea-2-turbo") || lower == "krea/krea-2-turbo" { return true }
    return false
  }

  /// A directory is a Krea-2 model root if it holds the native transformer plus
  /// the Qwen text-encoder/VAE subdirectories.
  public static func detect(at url: URL) -> Krea2ModelPaths? {
    let fm = FileManager.default
    let paths = Krea2ModelPaths(root: url)
    guard fm.fileExists(atPath: paths.transformerFile.path),
          fm.fileExists(atPath: paths.textEncoderFile.path),
          fm.fileExists(atPath: paths.vaeFile.path)
    else { return nil }
    return paths
  }

  /// Resolve a spec ("krea2", explicit dir, …) to model paths.
  public static func resolve(spec: String) throws -> Krea2ModelPaths {
    let expanded = (spec as NSString).expandingTildeInPath
    var isDir: ObjCBool = false
    if FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir), isDir.boolValue,
       let detected = detect(at: URL(fileURLWithPath: expanded)) {
      return detected
    }
    // Alias → newest HF cache snapshot.
    return try Krea2ModelPaths.resolve(modelDir: nil)
  }
}
