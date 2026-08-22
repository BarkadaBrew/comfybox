// ComfyBoxStateDirectoryIsolation.swift — no test may read or write the LIVE
// engine state directory (K-FIX-1 follow-up).
//
// `~/.comfybox` is not scratch space: the engine running on :7870 persists its
// QUEUE there (`queue-state.json`) and its pause flag (`queue-paused`). And
// `QueueStateStore.save` of an empty state does not write an empty file — it
// DELETES the snapshot. So a unit test that exercised the store round-trip was
// destroying the live engine's persisted queue every time the suite ran on
// this machine, and a test that constructed a coordinator would read (and
// could clear) the live pause flag.
//
// `QueueStateStore.stateDirectory` honours `COMFYBOX_STATE_DIR` and is
// computed rather than a cached `static let`, precisely so a test can redirect
// it after process start. This is the one place that does it, so no future
// test has to remember: call `isolateComfyBoxStateDirectory()` in `setUp` and
// the redirection, its assertions and its teardown all come with it.
//
// Production behaviour with the variable unset is unchanged and is asserted in
// `ComfyBoxStateDirectoryIsolationTests` below.

import Foundation
import XCTest

@testable import ZImage

extension XCTestCase {

  /// Point every engine state path at a fresh per-test temp directory for the
  /// duration of this test, and prove it took effect.
  ///
  /// Registers its own teardown (unset + delete), so callers need only call it.
  /// Returns the directory, for a test that wants to inspect what was written.
  @discardableResult
  func isolateComfyBoxStateDirectory(
    file: StaticString = #filePath, line: UInt = #line
  ) throws -> URL {
    let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
      .appendingPathComponent("comfybox-state-\(UUID().uuidString)", isDirectory: true)
      .standardizedFileURL
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    setenv("COMFYBOX_STATE_DIR", directory.path, 1)

    addTeardownBlock {
      unsetenv("COMFYBOX_STATE_DIR")
      try? FileManager.default.removeItem(at: directory)
    }

    // The assertion that makes this helper worth having: every state path the
    // engine resolves must now be INSIDE the temp directory. If the override
    // ever stops being honoured (e.g. someone re-caches it in a `static let`),
    // every suite using this helper fails here rather than quietly going back
    // to eating the live queue.
    let resolved = QueueStateStore.stateDirectory.standardizedFileURL
    XCTAssertEqual(
      resolved.path, directory.path,
      "COMFYBOX_STATE_DIR was not honoured — the test is pointed at the LIVE state directory",
      file: file, line: line)
    XCTAssertTrue(
      QueueStateStore.path.standardizedFileURL.path.hasPrefix(directory.path + "/"),
      "queue-state.json resolved outside the temp directory: \(QueueStateStore.path.path)",
      file: file, line: line)
    XCTAssertTrue(
      WarmServerQueueProbe.pauseSentinelPath.hasPrefix(directory.path + "/"),
      "the pause sentinel resolved outside the temp directory: \(WarmServerQueueProbe.pauseSentinelPath)",
      file: file, line: line)
    XCTAssertFalse(
      QueueStateStore.path.path.contains("/.comfybox/"),
      "a test resolved the live ~/.comfybox state path", file: file, line: line)
    return directory
  }
}

/// The helper's own contract, including the half that matters most: with the
/// variable UNSET, production still resolves `~/.comfybox`.
final class ComfyBoxStateDirectoryIsolationTests: XCTestCase {

  func testOverrideRedirectsEveryStatePath() throws {
    let directory = try isolateComfyBoxStateDirectory()
    XCTAssertEqual(QueueStateStore.path.deletingLastPathComponent().standardizedFileURL.path,
                   directory.path)
    XCTAssertEqual(QueueStateStore.path.lastPathComponent, "queue-state.json")
    XCTAssertEqual(
      WarmServerQueueProbe.pauseSentinelPath,
      directory.appendingPathComponent("queue-paused").path)
  }

  /// Unset ⇒ `~/.comfybox`, exactly as before this override existed. Resolving
  /// the path is a `mkdir -p` of a directory that already exists on a real
  /// install; nothing is read, written or deleted here.
  func testProductionPathIsUnchangedWhenTheVariableIsUnset() throws {
    let directory = try isolateComfyBoxStateDirectory()
    unsetenv("COMFYBOX_STATE_DIR")
    defer { setenv("COMFYBOX_STATE_DIR", directory.path, 1) }

    let home = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL.path
    XCTAssertEqual(
      QueueStateStore.stateDirectory.standardizedFileURL.path, home + "/.comfybox")
    XCTAssertEqual(
      QueueStateStore.path.standardizedFileURL.path, home + "/.comfybox/queue-state.json")
    XCTAssertEqual(
      WarmServerQueueProbe.pauseSentinelPath, home + "/.comfybox/queue-paused")
  }

  /// An empty override is ignored — it must not redirect state to the process
  /// working directory.
  func testAnEmptyOverrideFallsBackToTheHomeDirectory() throws {
    let directory = try isolateComfyBoxStateDirectory()
    setenv("COMFYBOX_STATE_DIR", "", 1)
    defer { setenv("COMFYBOX_STATE_DIR", directory.path, 1) }
    let home = FileManager.default.homeDirectoryForCurrentUser.standardizedFileURL.path
    XCTAssertEqual(QueueStateStore.stateDirectory.standardizedFileURL.path, home + "/.comfybox")
  }

  /// A `~`-prefixed override is expanded, not taken literally.
  func testATildeOverrideIsExpanded() throws {
    let directory = try isolateComfyBoxStateDirectory()
    defer { setenv("COMFYBOX_STATE_DIR", directory.path, 1) }
    let relative = ".comfybox-isolation-\(UUID().uuidString)"
    setenv("COMFYBOX_STATE_DIR", "~/\(relative)", 1)
    let expected = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(relative).standardizedFileURL.path
    XCTAssertEqual(QueueStateStore.stateDirectory.standardizedFileURL.path, expected)
    try? FileManager.default.removeItem(atPath: expected)
  }
}
