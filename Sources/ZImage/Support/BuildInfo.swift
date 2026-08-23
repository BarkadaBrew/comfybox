// BuildInfo.swift — The git identity of this binary (WP-E10, FDD §3.10 sink 3,
// §7.3 smoke step e: "/health build_sha equals the sha we just deployed —
// without it a clobbered binary is undetectable from outside").
//
// HOW IT WORKS (no plugin, no -D define):
//   * This file is COMMITTED with the placeholder `"unknown"`.
//   * `scripts/gen-build-info.sh` rewrites the `gitSHA` line with the short
//     sha of HEAD (plus `-dirty` when the worktree has uncommitted changes)
//     immediately before a release build; `scripts/gen-build-info.sh --reset`
//     puts the placeholder back. `scripts/deploy-serve.sh` does both (the
//     reset runs from its EXIT trap), so the tree is never left dirty by a
//     deploy and a stale sha can never be committed by accident.
//   * A development build that did not run the script reports `"unknown"`,
//     which the deploy smoke treats as a FAILURE for a deployed binary.
//
// `swift build` does not know the sha, so this is the one honest seam: the
// value is written at build time, read at runtime, never guessed.

import Foundation

public enum BuildInfo {
  /// The committed placeholder. `gen-build-info.sh --reset` restores it.
  public static let placeholder = "unknown"

  /// Short git sha of the commit this binary was built from, `-dirty` when
  /// the worktree had uncommitted changes, or `placeholder` for a build that
  /// did not run `scripts/gen-build-info.sh`.
  public static let gitSHA = "unknown"  // gen-build-info: DO NOT EDIT BY HAND — scripts/gen-build-info.sh rewrites this line

  /// Whether the sha was stamped at build time.
  public static var isKnown: Bool { gitSHA != placeholder }
}
