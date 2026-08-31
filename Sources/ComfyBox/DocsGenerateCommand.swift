// DocsGenerateCommand.swift — `comfybox docs generate` (FDD-ui-api-parity §3.4,
// §4.5; comfybox#300 Phase 4).
//
// Emits `docs/api-reference.md` from the compile-time `ControlRegistry` plus
// the route table parsed from the dispatch switches — the same sources the
// parity test reads, so the checked-in file byte-matches a fresh generation or
// CI fails (`ControlSurfaceParityTests.testAPIReferenceIsFresh`).

import Foundation
import ZImage

extension ZImageCLI {
  private static func docsUsageError(_ message: String) -> NSError {
    NSError(
      domain: "ZImageCLI", code: 64,
      userInfo: [
        NSLocalizedDescriptionKey:
          "\(message)\nUsage: comfybox docs generate [--repo-root <path>]"
          + "\nRegenerates docs/api-reference.md from the ControlRegistry and the dispatch"
          + " switches (run from the repo root, or pass --repo-root)."
      ])
  }

  static func runDocs(args: [String]) throws {
    guard args.first == "generate" else {
      throw docsUsageError("Unknown docs subcommand: \(args.first ?? "(none)")")
    }

    var repoRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
    var iterator = args.dropFirst().makeIterator()
    while let arg = iterator.next() {
      switch arg {
      case "--repo-root":
        guard let value = iterator.next() else {
          throw docsUsageError("--repo-root requires a path")
        }
        repoRoot = URL(fileURLWithPath: (value as NSString).expandingTildeInPath, isDirectory: true)
      default:
        throw docsUsageError("Unknown argument: \(arg)")
      }
    }

    let markdown = try APIReferenceDoc.markdown(repoRoot: repoRoot)
    let outputURL = repoRoot.appendingPathComponent(APIReferenceDoc.relativeOutputPath())
    try Data(markdown.utf8).write(to: outputURL, options: .atomic)
    print("Wrote \(outputURL.path) (\(markdown.utf8.count) bytes)")
  }
}
