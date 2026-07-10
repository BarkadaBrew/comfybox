// SVGExporter.swift — Shared vtracer wrapper for PNG → SVG conversion.
//
// Extracted from the CLI's original inline implementation (Sources/ComfyBox/
// main.swift) so both the CLI and the Studio Packs vector-first mode
// (Desktop, FR-3 / #196) share one preset table instead of duplicating it.

import Foundation

public enum SVGExportError: Error, LocalizedError {
  case vtracerNotFound
  case conversionFailed(status: Int32, message: String)

  public var errorDescription: String? {
    switch self {
    case .vtracerNotFound:
      return "vtracer not found. Install with: cargo install vtracer"
    case .conversionFailed(let status, let message):
      return "vtracer failed (\(status)): \(message)"
    }
  }
}

public enum SVGExporter {
  /// One of the recognized presets; anything else falls back to the same
  /// defaults as `.default`.
  public static let presets = ["default", "logo", "detailed", "simplified", "bw"]

  /// Default location vtracer installs to via `cargo install vtracer`.
  public static var defaultVTracerPath: String {
    FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".cargo/bin/vtracer").path
  }

  public static func isAvailable(vtracerPath: String = defaultVTracerPath) -> Bool {
    FileManager.default.fileExists(atPath: vtracerPath)
  }

  /// Pure argument builder — the preset table, exactly as used by the CLI.
  /// Separated from process execution so it's unit-testable without vtracer
  /// installed.
  public static func arguments(input: URL, output: URL, preset: String) -> [String] {
    var args = ["--input", input.path, "--output", output.path]
    switch preset {
    case "logo":
      args += ["--colormode", "color", "--hierarchical", "cutout", "--mode", "polygon",
               "-f", "10", "-p", "3", "-g", "48", "-c", "120", "-l", "8", "-s", "90", "--path_precision", "2"]
    case "detailed":
      args += ["--colormode", "color", "--hierarchical", "stacked", "--mode", "spline",
               "-f", "2", "-p", "8", "-g", "0", "-c", "45", "-l", "4", "-s", "60", "--path_precision", "8"]
    case "simplified":
      args += ["--colormode", "color", "--hierarchical", "stacked", "--mode", "polygon",
               "-f", "6", "-p", "5", "-g", "16", "-c", "90", "-l", "6", "-s", "75", "--path_precision", "3"]
    case "bw":
      args += ["--colormode", "binary", "--hierarchical", "stacked", "--mode", "spline",
               "-f", "4", "-p", "6", "-g", "0", "-c", "60", "-l", "4", "-s", "60", "--path_precision", "5"]
    default:
      args += ["--colormode", "color", "--hierarchical", "stacked", "--mode", "spline",
               "-f", "4", "-p", "6", "-g", "0", "-c", "60", "-l", "4", "-s", "60", "--path_precision", "5"]
    }
    return args
  }

  /// Convert a PNG to SVG via vtracer. Throws `.vtracerNotFound` if the
  /// binary isn't installed, `.conversionFailed` if it exits non-zero —
  /// callers must treat this as an independent failure that never hides an
  /// already-successful PNG render (see the PRD's FR-3 acceptance).
  public static func convert(
    input: URL, output: URL, preset: String, vtracerPath: String = defaultVTracerPath
  ) throws {
    guard isAvailable(vtracerPath: vtracerPath) else {
      throw SVGExportError.vtracerNotFound
    }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: vtracerPath)
    process.arguments = arguments(input: input, output: output, preset: preset)
    let pipe = Pipe()
    process.standardError = pipe
    try process.run()
    process.waitUntilExit()
    if process.terminationStatus != 0 {
      let errorData = pipe.fileHandleForReading.readDataToEndOfFile()
      let message = String(data: errorData, encoding: .utf8) ?? "Unknown error"
      throw SVGExportError.conversionFailed(status: process.terminationStatus, message: message)
    }
  }
}
