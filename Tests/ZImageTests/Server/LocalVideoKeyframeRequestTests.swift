import XCTest
@testable import ZImage

/// WP2a (docs/FDD-ltx-director-tab.md): the generic `keyframes[]` field on the
/// local video wire body and the pure `WarmServer.resolveKeyframes` split
/// (frame 0 → init image, frame > 0 → `LTX2VideoRequest.keyframes` extras).
final class LocalVideoKeyframeRequestTests: XCTestCase {

  private func decode(_ json: String) throws -> WarmServer.LocalVideoRequest {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return try decoder.decode(WarmServer.LocalVideoRequest.self, from: Data(json.utf8))
  }

  private func assertConflict(_ json: String, file: StaticString = #filePath, line: UInt = #line) throws {
    let req = try decode(json)
    XCTAssertThrowsError(try WarmServer.resolveKeyframes(req), file: file, line: line) { error in
      guard case LTX2VideoError.invalidKeyframe(let why) = error else {
        return XCTFail("expected invalidKeyframe, got \(error)", file: file, line: line)
      }
      XCTAssertTrue(why.contains("keyframe_conflict"), why, file: file, line: line)
      XCTAssertTrue(error.localizedDescription.contains("keyframe_conflict"), error.localizedDescription, file: file, line: line)
    }
  }

  func testKeyframesDecodeSnakeCase() throws {
    let req = try decode(#"{"prompt":"x","keyframes":[{"image_path":"/a.png","frame":0},{"image_path":"/b.png","frame":96,"strength":0.8}]}"#)
    let kfs = try XCTUnwrap(req.keyframes)
    XCTAssertEqual(kfs.count, 2)
    XCTAssertEqual(kfs[0].imagePath, "/a.png")
    XCTAssertEqual(kfs[0].frame, 0)
    XCTAssertNil(kfs[0].strength, "absent strength stays nil on the wire struct; defaults to 1.0 in resolveKeyframes")
    XCTAssertEqual(kfs[1].imagePath, "/b.png")
    XCTAssertEqual(kfs[1].frame, 96)
    XCTAssertEqual(kfs[1].strength, 0.8)
  }

  func testFrameZeroKeyframeStrengthIsRangeChecked() throws {
    for bad in ["0", "-0.5", "7", "1.0001"] {
      let req = try decode(#"{"prompt":"x","keyframes":[{"image_path":"/a.png","frame":0,"strength":"# + bad + "}]}")
      XCTAssertThrowsError(try WarmServer.resolveKeyframes(req), "strength \(bad)") { error in
        guard case LTX2VideoError.invalidKeyframe(let why) = error else {
          return XCTFail("expected invalidKeyframe, got \(error)")
        }
        XCTAssertTrue(why.contains("frame 0"), why)
      }
    }
    let ok = try WarmServer.resolveKeyframes(
      try decode(#"{"prompt":"x","keyframes":[{"image_path":"/a.png","frame":0,"strength":1.0}]}"#))
    XCTAssertEqual(ok.strength, 1.0)
    let partial = try WarmServer.resolveKeyframes(
      try decode(#"{"prompt":"x","keyframes":[{"image_path":"/a.png","frame":0,"strength":0.4}]}"#))
    XCTAssertEqual(partial.strength, 0.4)
  }

  func testAbsentKeyframesPassesLegacyFieldsThrough() throws {
    let req = try decode(#"{"prompt":"x","image_path":"/init.png","strength":0.6}"#)
    let resolved = try WarmServer.resolveKeyframes(req)
    XCTAssertEqual(resolved.initImagePath, "/init.png")
    XCTAssertEqual(resolved.strength, 0.6)
    XCTAssertEqual(resolved.extras, [])

    let t2v = try WarmServer.resolveKeyframes(try decode(#"{"prompt":"x"}"#))
    XCTAssertNil(t2v.initImagePath)
    XCTAssertNil(t2v.strength)
    XCTAssertEqual(t2v.extras, [])
  }

  func testEmptyKeyframesArrayBehavesLikeAbsent() throws {
    let req = try decode(#"{"prompt":"x","image_path":"/init.png","keyframes":[]}"#)
    let resolved = try WarmServer.resolveKeyframes(req)
    XCTAssertEqual(resolved.initImagePath, "/init.png")
    XCTAssertEqual(resolved.extras, [])
  }

  func testResolveKeyframesSplitsFrameZero() throws {
    let req = try decode(#"{"prompt":"x","keyframes":[{"image_path":"/a.png","frame":0},{"image_path":"/b.png","frame":96,"strength":0.8}]}"#)
    let resolved = try WarmServer.resolveKeyframes(req)
    XCTAssertEqual(resolved.initImagePath, "/a.png")
    XCTAssertEqual(resolved.strength, 1.0)
    XCTAssertEqual(resolved.extras, [LTX2KeyframeRef(imagePath: "/b.png", frame: 96, strength: 0.8)])
  }

  func testResolveKeyframesFrameZeroStrengthCarries() throws {
    let req = try decode(#"{"prompt":"x","keyframes":[{"image_path":"/b.png","frame":288},{"image_path":"/a.png","frame":0,"strength":0.7}]}"#)
    let resolved = try WarmServer.resolveKeyframes(req)
    XCTAssertEqual(resolved.initImagePath, "/a.png")
    XCTAssertEqual(resolved.strength, 0.7)
    XCTAssertEqual(resolved.extras, [LTX2KeyframeRef(imagePath: "/b.png", frame: 288, strength: 1.0)])
  }

  func testResolveKeyframesConflictWithImagePath() throws {
    try assertConflict(#"{"prompt":"x","image_path":"/init.png","keyframes":[{"image_path":"/a.png","frame":0}]}"#)
  }

  func testResolveKeyframesConflictWithImageBase64() throws {
    try assertConflict(#"{"prompt":"x","image_base64":"aGVsbG8=","keyframes":[{"image_path":"/a.png","frame":0}]}"#)
  }

  func testResolveKeyframesConflictWithStrength() throws {
    try assertConflict(#"{"prompt":"x","strength":0.5,"keyframes":[{"image_path":"/a.png","frame":0}]}"#)
  }

  func testResolveKeyframesWithoutFrameZeroIsT2VPlusExtras() throws {
    let req = try decode(#"{"prompt":"x","keyframes":[{"image_path":"/end.png","frame":288}]}"#)
    let resolved = try WarmServer.resolveKeyframes(req)
    XCTAssertNil(resolved.initImagePath)
    XCTAssertNil(resolved.strength)
    XCTAssertEqual(resolved.extras, [LTX2KeyframeRef(imagePath: "/end.png", frame: 288, strength: 1.0)])
  }

  func testTwoFrameZeroEntriesRejected() throws {
    let req = try decode(#"{"prompt":"x","keyframes":[{"image_path":"/a.png","frame":0},{"image_path":"/b.png","frame":0}]}"#)
    XCTAssertThrowsError(try WarmServer.resolveKeyframes(req)) { error in
      guard case LTX2VideoError.invalidKeyframe = error else { return XCTFail("expected invalidKeyframe, got \(error)") }
    }
  }

  /// The extras ride `LTX2VideoRequest.keyframes`; the frame-0 entry rides the
  /// legacy init-image fields, so the untouched i2v arm still serves it.
  func testBuildLocalVideoRequestThreadsResolvedKeyframes() throws {
    let req = try decode(#"{"prompt":"x","keyframes":[{"image_path":"/a.png","frame":0,"strength":0.9},{"image_path":"/b.png","frame":96}]}"#)
    let resolved = try WarmServer.resolveKeyframes(req)
    let request = WarmServer.buildLocalVideoRequest(
      req: req, videoPreset: nil,
      effectivePrompt: "x", effectiveInitImage: resolved.initImagePath,
      renderWidth: 576, renderHeight: 896,
      foldedFramesPerChunk: 289, foldedExtendSeconds: 0,
      resolvedLoRAs: [], effectiveBeatSchedule: nil,
      resolvedOutput: "/tmp/o.mp4",
      initStrength: resolved.strength, keyframes: resolved.extras)
    XCTAssertEqual(request.initImagePath, "/a.png")
    XCTAssertEqual(request.strength, 0.9)
    XCTAssertEqual(request.keyframes, [LTX2KeyframeRef(imagePath: "/b.png", frame: 96, strength: 1.0)])
  }

  /// The recipe fingerprint must move when the keyframe table moves (a
  /// different end frame is a different render), and stay put for the
  /// keyframe-less shape so existing hashes are unchanged.
  func testRecipeFingerprintCoversKeyframes() throws {
    let snapshot = LTX2ConfigResolver.resolveTyped(request: nil, preset: nil)
    func request(_ kfs: [LTX2KeyframeRef]) -> LTX2VideoRequest {
      LTX2VideoRequest(
        prompt: "x", initImagePath: "/a.png", width: 576, height: 896, framesPerChunk: 289,
        outputPath: "/tmp/o.mp4", resolvedConfigSnapshot: snapshot, keyframes: kfs)
    }
    let none = try ResolvedVideoRecipe.build(request: request([]), transformerFile: "t.safetensors").fingerprint()
    let end = try ResolvedVideoRecipe.build(
      request: request([LTX2KeyframeRef(imagePath: "/b.png", frame: 288)]), transformerFile: "t.safetensors").fingerprint()
    let mid = try ResolvedVideoRecipe.build(
      request: request([LTX2KeyframeRef(imagePath: "/b.png", frame: 96)]), transformerFile: "t.safetensors").fingerprint()
    XCTAssertNotEqual(none, end)
    XCTAssertNotEqual(end, mid)
    let recipe = try ResolvedVideoRecipe.build(request: request([]), transformerFile: "t.safetensors")
    XCTAssertNil(recipe.keyframes, "keyframe-less recipes omit the key so pre-WP2a fingerprints are unchanged")
  }
}
