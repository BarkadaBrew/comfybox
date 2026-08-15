import XCTest
@testable import ZImage

final class TuningDecodeProbeTests: XCTestCase {
  func testStage1SigmasDecodesFromSnakeCase() throws {
    let json = #"{"stage1_sigmas":[1.0,0.955,0.121,0.0],"img_compression":30}"#
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let t = try decoder.decode(LTX2VideoTuning.self, from: Data(json.utf8))
    XCTAssertEqual(t.stage1Sigmas, [1.0, 0.955, 0.121, 0.0])
    XCTAssertEqual(t.imgCompression, 30)
  }
}
