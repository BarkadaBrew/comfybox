// MotionViewServerDefaultsOverlayTests.swift — comfybox#324 (Phase 3 config
// follow-ups from the ad8d01b adversarial review, F5).
//
// `MotionView.applyServerVideoDefaults()` overlays the server's
// `videoDefaults` onto the Motion tab from an async `Task` fired by
// `applyDefaults()` at view load. Between that fetch starting and resolving,
// the user is free to edit `resolution`/`seconds` themselves — the overlay
// used to apply unconditionally, silently discarding whatever they just
// typed. `resolvedServerVideoDefaultsOverlay` is the pure decision extracted
// from that function; this drives it directly.

import XCTest
@testable import ComfyBoxDesktop

final class MotionViewServerDefaultsOverlayTests: XCTestCase {

  func testAppliesServerValuesWhenTheUserHasNotEditedAnything() {
    let overlay = MotionView.resolvedServerVideoDefaultsOverlay(
      userDidEditVideoSettings: false,
      serverFrames: 121, serverWidth: 448, serverHeight: 704)

    XCTAssertEqual(overlay.frames, 121)
    XCTAssertEqual(overlay.resolution, .portrait)
  }

  /// The core F5 regression: a user edit made during the fetch window must
  /// win, not the server value that resolves after it.
  func testAppliesNothingOnceTheUserHasEditedEitherControl() {
    let overlay = MotionView.resolvedServerVideoDefaultsOverlay(
      userDidEditVideoSettings: true,
      serverFrames: 121, serverWidth: 448, serverHeight: 704)

    XCTAssertNil(overlay.frames, "a user edit must not be stomped by the server overlay")
    XCTAssertNil(overlay.resolution, "a user edit must not be stomped by the server overlay")
  }

  func testIgnoresAFrameCountThatIsNotOneOfTheLegalOptions() {
    // 100 is not in MotionView's frameOptions ladder (25/49/97/121/145/193/241/289).
    let overlay = MotionView.resolvedServerVideoDefaultsOverlay(
      userDidEditVideoSettings: false,
      serverFrames: 100, serverWidth: nil, serverHeight: nil)

    XCTAssertNil(overlay.frames, "an illegal frame count must not overlay")
  }

  func testIgnoresDimsThatDoNotMatchAKnownResolutionPreset() {
    let overlay = MotionView.resolvedServerVideoDefaultsOverlay(
      userDidEditVideoSettings: false,
      serverFrames: nil, serverWidth: 999, serverHeight: 999)

    XCTAssertNil(overlay.resolution, "dims with no matching VideoResolution case must not overlay")
  }

  func testResolvesEachKnownResolutionPresetByDims() {
    let landscape = MotionView.resolvedServerVideoDefaultsOverlay(
      userDidEditVideoSettings: false, serverFrames: nil, serverWidth: 704, serverHeight: 448)
    XCTAssertEqual(landscape.resolution, .landscape)

    let square = MotionView.resolvedServerVideoDefaultsOverlay(
      userDidEditVideoSettings: false, serverFrames: nil, serverWidth: 512, serverHeight: 512)
    XCTAssertEqual(square.resolution, .square)
  }

  func testNilServerValuesOverlayNothingRegardlessOfTheUserFlag() {
    let overlay = MotionView.resolvedServerVideoDefaultsOverlay(
      userDidEditVideoSettings: false,
      serverFrames: nil, serverWidth: nil, serverHeight: nil)

    XCTAssertNil(overlay.frames)
    XCTAssertNil(overlay.resolution)
  }
}
