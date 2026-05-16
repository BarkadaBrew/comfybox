import XCTest
@testable import ZImage

final class PipelineSnapshotTests: XCTestCase {

  func testCivitAIFilePatternsIncludeTextEncoderAndVAEButExcludeTransformerWeights() {
    let patterns = PipelineSnapshot.configTokenizerTextEncoderAndVAEFilePatterns

    XCTAssertTrue(patterns.contains("text_encoder/*.safetensors"))
    XCTAssertTrue(patterns.contains("text_encoder/*.safetensors.index.json"))
    XCTAssertTrue(patterns.contains("vae/*.safetensors"))
    XCTAssertTrue(patterns.contains("vae/*.safetensors.index.json"))
    XCTAssertFalse(patterns.contains("transformer/*.safetensors"))
    XCTAssertFalse(patterns.contains("transformer/*.safetensors.index.json"))
  }
}
