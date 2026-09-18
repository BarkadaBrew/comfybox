import Foundation
import XCTest

@testable import ZImage

/// A field sent in the wrong place must be REFUSED, not dropped.
///
/// `Codable` ignores unknown keys, so `negative_prompt` at the timeline root
/// used to decode cleanly and render with the preset's negative instead. That
/// is not hypothetical: two 34-second monologues were rendered without the
/// negative prompt they were written with, and nothing anywhere said so. The
/// clips looked plausible, which is precisely why it went unnoticed.
final class DirectorMisplacedFieldTests: XCTestCase {

  private var decoder: JSONDecoder {
    let d = JSONDecoder()
    d.keyDecodingStrategy = .convertFromSnakeCase
    return d
  }

  private func timeline(extraRootKeys: String = "") -> Data {
    Data("""
      {"version":1,
       "settings":{"width":576,"height":896,"length_frames":289,"fps":24},
       "global_prompt":"she talks to camera"\(extraRootKeys),
       "keyframes":[],"prompt_segments":[],"audio_clips":[],
       "audio":{"mode":"imported"}}
      """.utf8)
  }

  func testAWellFormedTimelineStillDecodes() throws {
    let t = try decoder.decode(DirectorTimeline.self, from: timeline())
    XCTAssertEqual(t.globalPrompt, "she talks to camera")
    XCTAssertEqual(t.settings.lengthFrames, 289)
  }

  func testTheNEGATIVEPromptAtTheRootIsRefusedByName() {
    // The exact mistake: the render would otherwise use a different negative
    // than the one the author wrote.
    XCTAssertThrowsError(
      try decoder.decode(
        DirectorTimeline.self, from: timeline(extraRootKeys: #","negative_prompt":"blurry""#))
    ) { error in
      guard case DirectorError.misplacedField(let key, let belongsIn) = error else {
        return XCTFail("expected misplacedField, got \(error)")
      }
      XCTAssertEqual(key, "negative_prompt")
      XCTAssertEqual(belongsIn, "settings")
      XCTAssertTrue(
        "\(error)".contains("silently"),
        "the message must say WHY it is refused rather than just that it is")
    }
  }

  func testEverySettingsOnlyKeyIsRefusedAtTheRoot() {
    // Each of these would change the render if honoured and change it
    // differently if dropped, so each must be caught.
    for key in DirectorTimeline.settingsOnlyKeys {
      let value = ["loras"].contains(key) ? "[]" : (
        ["seed", "fps", "width", "height", "length_frames", "steps"].contains(key) ? "7" : "\"x\"")
      XCTAssertThrowsError(
        try decoder.decode(
          DirectorTimeline.self, from: timeline(extraRootKeys: ",\"\(key)\":\(value)")),
        "\(key) at the root must be refused")
    }
  }

  func testAGenuinelyUNKNOWNFieldIsStillAccepted() throws {
    // Forward compatibility: a client sending something this build has never
    // heard of keeps working. The list is named on purpose — blanket
    // unknown-key rejection would break that.
    let t = try decoder.decode(
      DirectorTimeline.self,
      from: timeline(extraRootKeys: #","some_future_field":{"a":1}"#))
    XCTAssertEqual(t.globalPrompt, "she talks to camera")
  }

  func testTheSameKeyInSETTINGSIsFine() throws {
    let data = Data("""
      {"version":1,
       "settings":{"width":576,"height":896,"length_frames":289,
                   "negative_prompt":"blurry","seed":7},
       "global_prompt":"x","keyframes":[],"prompt_segments":[],
       "audio_clips":[],"audio":{"mode":"imported"}}
      """.utf8)
    let t = try decoder.decode(DirectorTimeline.self, from: data)
    XCTAssertEqual(t.settings.negativePrompt, "blurry")
    XCTAssertEqual(t.settings.seed, 7)
  }
}
