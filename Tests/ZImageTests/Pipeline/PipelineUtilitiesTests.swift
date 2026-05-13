import XCTest
@testable import ZImage

final class PipelineUtilitiesTests: XCTestCase {
  func testZImagePackedImageSeqLenUsesLatentPatchTokens() {
    XCTAssertEqual(
      PipelineUtilities.zImagePackedImageSeqLen(latentHeight: 128, latentWidth: 128),
      4096
    )
    XCTAssertEqual(
      PipelineUtilities.zImagePackedImageSeqLen(latentHeight: 64, latentWidth: 64),
      1024
    )
  }

  func testZImagePackedImageSeqLenClampsTinyDimensions() {
    XCTAssertEqual(
      PipelineUtilities.zImagePackedImageSeqLen(latentHeight: 1, latentWidth: 1),
      1
    )
  }
}
