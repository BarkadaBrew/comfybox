import XCTest
@testable import ZImage

/// Pure-logic tests for the LTX-2 video generator (frame/chunk math, request
/// validation). The heavy model load/generate is exercised only in a live run.
final class LTX2VideoGeneratorTests: XCTestCase {

    func testFrameCountValidation() {
        // Valid: 1 + 8k, at least 9.
        for n in [9, 17, 25, 33, 97, 121] {
            XCTAssertTrue(LTX2VideoGenerator.isValidFrameCount(n), "\(n) should be valid")
        }
        // Invalid.
        for n in [1, 8, 10, 16, 96, 100, 0, -1] {
            XCTAssertFalse(LTX2VideoGenerator.isValidFrameCount(n), "\(n) should be invalid")
        }
    }

    func testDimensionValidation() {
        XCTAssertTrue(LTX2VideoGenerator.areValidDimensions(width: 704, height: 448))
        XCTAssertTrue(LTX2VideoGenerator.areValidDimensions(width: 1024, height: 1024))
        XCTAssertFalse(LTX2VideoGenerator.areValidDimensions(width: 700, height: 448))
        XCTAssertFalse(LTX2VideoGenerator.areValidDimensions(width: 704, height: 0))
    }

    func testSingleChunkPlan() {
        let plan = LTX2VideoGenerator.chunkPlan(framesPerChunk: 97, extendToSeconds: 0, fps: 24)
        XCTAssertEqual(plan.totalChunks, 1)
        XCTAssertEqual(plan.totalFrames, 97)
        XCTAssertEqual(plan.durationSeconds, 97.0 / 24.0, accuracy: 0.001)
    }

    func testExtendedChunkPlan() {
        // 97-frame chunks (each continuation adds 96 new frames), target 8s @24 =
        // 192 frames. Need ceil((192-97)/96)=ceil(0.99)=1 continuation -> 2 chunks,
        // 97 + 96 = 193 frames.
        let plan = LTX2VideoGenerator.chunkPlan(framesPerChunk: 97, extendToSeconds: 8, fps: 24)
        XCTAssertEqual(plan.totalChunks, 2)
        XCTAssertEqual(plan.totalFrames, 193)
    }

    func testExtendShorterThanOneChunkStaysSingle() {
        // 2s @24 = 48 target frames < 97 -> no continuations.
        let plan = LTX2VideoGenerator.chunkPlan(framesPerChunk: 97, extendToSeconds: 2, fps: 24)
        XCTAssertEqual(plan.totalChunks, 1)
        XCTAssertEqual(plan.totalFrames, 97)
    }

    func testValidateRejectsBadFramesAndDims() throws {
        let gen = LTX2VideoGenerator(config: .init(weightsDir: "/nope", gemmaPath: "/nope"))
        XCTAssertThrowsError(try gen.validate(
            LTX2VideoRequest(prompt: "x", width: 704, height: 448, framesPerChunk: 10, outputPath: "/tmp/o.mp4")))
        XCTAssertThrowsError(try gen.validate(
            LTX2VideoRequest(prompt: "x", width: 700, height: 448, framesPerChunk: 97, outputPath: "/tmp/o.mp4")))
    }

    func testValidateReportsMissingWeights() {
        let gen = LTX2VideoGenerator(config: .init(weightsDir: "/definitely/not/here", gemmaPath: "/nope"))
        XCTAssertThrowsError(try gen.validate(
            LTX2VideoRequest(prompt: "x", width: 704, height: 448, framesPerChunk: 97, outputPath: "/tmp/o.mp4"))) { error in
            guard case LTX2VideoError.weightsMissing = error else {
                return XCTFail("expected weightsMissing, got \(error)")
            }
        }
    }
}
