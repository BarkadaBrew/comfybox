import XCTest
@testable import ZImage

/// Pure-logic tests for the LTX-2 video generator (frame/chunk math, request
/// validation). The heavy model load/generate is exercised only in a live run.
final class LTX2VideoGeneratorTests: XCTestCase {

    private func wordTokens(_ text: String) -> [Int] {
        text.split(whereSeparator: { $0.isWhitespace }).indices.map { $0 }
    }

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

    func testAudioPromptGuardMovesTrailingAudioSectionAheadOfLongVisualPrompt() throws {
        let visual = Array(repeating: "neutral-detail", count: 132).joined(separator: " ")
        let audio = #"audio: quiet room tone, an adult speaker says "the kettle is ready"."#
        let original = visual + " " + audio

        let guarded = try LTX2AudioPromptGuard.prepare(
            prompt: original, audio: true, maxLength: 128, tokenize: wordTokens)

        XCTAssertTrue(guarded.reordered)
        XCTAssertTrue(guarded.effectivePrompt.hasPrefix(audio + " Visual: "))
        XCTAssertTrue(guarded.effectivePrompt.hasSuffix(visual))
        XCTAssertEqual(guarded.audioMarkerTokenIndex, 0)
        XCTAssertTrue(guarded.quotedLineSurvived)
        XCTAssertEqual(guarded.preTruncationTokenCount, wordTokens(guarded.effectivePrompt).count)
        XCTAssertEqual(guarded.effectivePromptHash.count, 12)
    }

    func testAudioPromptGuardMakesMatchedInsideAndBeyondPromptsByteIdentical() throws {
        let visual = Array(repeating: "neutral-detail", count: 132).joined(separator: " ")
        let audio = #"Audio: quiet room tone, an adult speaker says "the kettle is ready"."#
        let inside = audio + " Visual: " + visual
        let beyond = "Visual: " + visual + " " + audio

        let guardedInside = try LTX2AudioPromptGuard.prepare(
            prompt: inside, audio: true, maxLength: 128, tokenize: wordTokens)
        let guardedBeyond = try LTX2AudioPromptGuard.prepare(
            prompt: beyond, audio: true, maxLength: 128, tokenize: wordTokens)

        XCTAssertFalse(guardedInside.reordered)
        XCTAssertTrue(guardedBeyond.reordered)
        XCTAssertEqual(guardedBeyond.effectivePrompt, guardedInside.effectivePrompt)
        XCTAssertEqual(guardedBeyond.effectivePromptHash, guardedInside.effectivePromptHash)
    }

    func testAudioPromptGuardLeavesBoundedLeadingAudioSectionInPlace() throws {
        let visual = Array(repeating: "neutral-detail", count: 140).joined(separator: " ")
        let prompt = #"Audio: soft rain, an adult speaker says "welcome home". Visual: "# + visual

        let guarded = try LTX2AudioPromptGuard.prepare(
            prompt: prompt, audio: true, maxLength: 128, tokenize: wordTokens)

        XCTAssertFalse(guarded.reordered)
        XCTAssertEqual(guarded.effectivePrompt, prompt)
        XCTAssertEqual(guarded.audioMarkerTokenIndex, 0)
        XCTAssertTrue(guarded.quotedLineSurvived)
        XCTAssertGreaterThan(guarded.preTruncationTokenCount, 128)
    }

    func testAudioPromptGuardLeavesSafeTrailingAudioSectionInPlace() throws {
        let prompt = #"An adult potter finishes a cup at a quiet studio table. audio: soft rain, the potter says "the glaze is ready"."#

        let guarded = try LTX2AudioPromptGuard.prepare(
            prompt: prompt, audio: true, maxLength: 128, tokenize: wordTokens)

        XCTAssertFalse(guarded.reordered)
        XCTAssertEqual(guarded.effectivePrompt, prompt)
        XCTAssertLessThan(guarded.audioMarkerTokenIndex ?? 128, 128)
        XCTAssertTrue(guarded.quotedLineSurvived)
    }

    func testAudioPromptGuardRejectsAudioSectionThatCannotFit() {
        let overlongAudio = "audio: "
            + Array(repeating: "sound-detail", count: 130).joined(separator: " ")
            + #" an adult speaker says "this line cannot survive"."#
        let prompt = overlongAudio + " Visual: a neutral studio scene."

        XCTAssertThrowsError(try LTX2AudioPromptGuard.prepare(
            prompt: prompt, audio: true, maxLength: 128, tokenize: wordTokens)) { error in
            guard case LTX2VideoError.audioUnsupported(let reason) = error else {
                return XCTFail("expected audioUnsupported, got \(error)")
            }
            XCTAssertTrue(reason.contains("128-token"), reason)
        }
    }

    func testAudioPromptGuardRejectsUnmarkedDialogueBeyondLimit() {
        let visual = Array(repeating: "neutral-detail", count: 130).joined(separator: " ")
        let prompt = visual + #" An adult speaker says "this dialogue is beyond the boundary"."#

        XCTAssertThrowsError(try LTX2AudioPromptGuard.prepare(
            prompt: prompt, audio: true, maxLength: 128, tokenize: wordTokens)) { error in
            guard case LTX2VideoError.audioUnsupported(let reason) = error else {
                return XCTFail("expected audioUnsupported, got \(error)")
            }
            XCTAssertTrue(reason.contains("audio:"), reason)
        }
    }

    func testAudioPromptGuardDoesNotRewriteVideoOnlyPrompt() throws {
        let visual = Array(repeating: "neutral-detail", count: 132).joined(separator: " ")
        let prompt = visual + " audio: quiet room tone."

        let guarded = try LTX2AudioPromptGuard.prepare(
            prompt: prompt, audio: false, maxLength: 128, tokenize: wordTokens)

        XCTAssertFalse(guarded.reordered)
        XCTAssertEqual(guarded.effectivePrompt, prompt)
        XCTAssertFalse(guarded.quotedLineSurvived)
        XCTAssertGreaterThan(guarded.audioMarkerTokenIndex ?? -1, 127)
    }
}

  // MARK: - Gemma max length (2026-08-07)

  /// 128 was a port artifact, not the trained recipe: the official Lightricks
  /// pipeline tokenizes at max_length 1024 (LTX2_GEMMA_MAX_LENGTH default), the
  /// ComfyUI Gemma loader defaults to 1024, and Todd's reference PinkCherry
  /// workflow feeds enhancer output up to 256 tokens through this encoder. The
  /// connector tiles 128 learnable registers via integer division, so the value
  /// must stay a positive multiple of 128.
  func testGemmaMaxLengthDefaultsToUpstream1024() {
    XCTAssertEqual(LTX2VideoGenerator.resolveGemmaMaxLength(env: nil), 1024)
    XCTAssertEqual(LTX2VideoGenerator.resolveGemmaMaxLength(env: ""), 1024)
  }

  func testGemmaMaxLengthHonorsValidOverride() {
    XCTAssertEqual(LTX2VideoGenerator.resolveGemmaMaxLength(env: "128"), 128)
    XCTAssertEqual(LTX2VideoGenerator.resolveGemmaMaxLength(env: "256"), 256)
  }

  func testGemmaMaxLengthRejectsNonMultipleOf128() {
    // Register tiling is seqLen / 128 by integer division — a non-multiple
    // silently drops register coverage, so fall back to the default instead.
    XCTAssertEqual(LTX2VideoGenerator.resolveGemmaMaxLength(env: "200"), 1024)
    XCTAssertEqual(LTX2VideoGenerator.resolveGemmaMaxLength(env: "0"), 1024)
    XCTAssertEqual(LTX2VideoGenerator.resolveGemmaMaxLength(env: "-128"), 1024)
    XCTAssertEqual(LTX2VideoGenerator.resolveGemmaMaxLength(env: "abc"), 1024)
  }
