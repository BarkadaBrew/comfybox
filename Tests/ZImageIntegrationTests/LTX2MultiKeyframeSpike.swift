// LTX2MultiKeyframeSpike.swift — empirical feasibility spike for
// LTX2Pipeline.generateMultiKeyframe: does the existing dense-splice I2V
// conditioning (built for a single image at frame 0) also produce a
// coherent result when a SECOND image is placed later in the sequence,
// letting the transformer's own temporal attention "tween" between them?
//
// Not a pass/fail correctness test — this writes frames to /tmp for visual
// inspection. See docs/ltx2-multi-keyframe-fdd.md for the resulting design.

import XCTest
import MLX
import MLXRandom
import Logging
import CoreGraphics
import ImageIO
@testable import ZImage

final class LTX2MultiKeyframeSpike: XCTestCase {

  func testTwoKeyframesAcrossTheSequence() throws {
    let logger = Logger(label: "ltx2.spike")

    let weightsDir = "/Volumes/Bolt/Models/ltx2-distilled"
    let gemmaPath = "/Users/toddwalderman/.cache/huggingface/hub/models/unsloth/gemma-3-12b-it"
    guard FileManager.default.fileExists(atPath: weightsDir),
          FileManager.default.fileExists(atPath: gemmaPath) else {
      throw XCTSkip("LTX-2 weights not available locally")
    }

    let generator = LTX2VideoGenerator(
      config: .init(weightsDir: weightsDir, gemmaPath: gemmaPath),
      logger: logger
    )
    logger.info("Loading LTX-2 pipeline (transformer + VAE + Gemma text encoder)...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    try generator.load()
    logger.info("Loaded in \(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - loadStart))s")

    guard let pipeline = generator.loadedPipeline, let tokenizer = generator.loadedTokenizer else {
      XCTFail("Generator reported loaded but exposed no pipeline/tokenizer")
      return
    }

    let width = 512
    let height = 320
    let numFrames = 33  // 1 + 8*4 -> latF = 5

    func loadKeyframeImage(_ path: String) throws -> MLXArray {
      let url = URL(fileURLWithPath: path)
      guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw XCTSkip("Could not load test image at \(path)")
      }
      let pixels = try QwenImageIO.resizedPixelArray(
        from: cgImage, width: width, height: height, addBatchDimension: true, dtype: .float32
      )
      return QwenImageIO.normalizeForEncoder(pixels)
    }

    let imageA = try loadKeyframeImage(
      "/private/tmp/claude-501/-Users-toddwalderman-Projects-zimage-swift/2bc157ab-de29-4db1-b8ae-d9643e633773/scratchpad/crash-test-out/crash-recovery-test.png"
    )
    let imageB = try loadKeyframeImage(
      "/Users/toddwalderman/Pictures/ComfyBox/zimage-krea2-781369D9-E6F8-43CC-8E30-6B1BB342727A.png"
    )

    let batch = tokenizer.encode(prompt: "smooth cinematic transition", maxLength: 128)
    MLX.eval(batch.inputIds, batch.attentionMask)

    logger.info("Generating \(numFrames)-frame clip with keyframes at frame 0 and frame 32...")
    let genStart = CFAbsoluteTimeGetCurrent()
    let output = pipeline.generateMultiKeyframe(
      inputIds: batch.inputIds,
      attentionMask: batch.attentionMask,
      keyframes: [
        .init(image: imageA, videoFrameIndex: 0, strength: 1.0),
        .init(image: imageB, videoFrameIndex: numFrames - 1, strength: 1.0),
      ],
      width: width,
      height: height,
      numFrames: numFrames,
      steps: 8,
      seed: 7
    ) { step, total in
      logger.info("  step \(step)/\(total)")
    }
    logger.info("Generated in \(String(format: "%.0f", CFAbsoluteTimeGetCurrent() - genStart))s")

    XCTAssertFalse(MLX.any(MLX.isNaN(output.decoded)).item(Bool.self), "Output contains NaN")

    let frames = LTX2PostProcess.framesToImages(from: output.decoded)
    XCTAssertEqual(frames.count, numFrames)

    let outDir = "/tmp/ltx2-multi-keyframe-spike"
    try? FileManager.default.createDirectory(atPath: outDir, withIntermediateDirectories: true)
    for i in [0, 8, 16, 24, numFrames - 1] where i < frames.count {
      let path = "\(outDir)/frame-\(String(format: "%03d", i)).png"
      let destURL = URL(fileURLWithPath: path) as CFURL
      guard let destination = CGImageDestinationCreateWithURL(destURL, "public.png" as CFString, 1, nil) else {
        continue
      }
      CGImageDestinationAddImage(destination, frames[i], nil)
      CGImageDestinationFinalize(destination)
      logger.info("Wrote \(path)")
    }

    let mp4Path = "\(outDir)/spike.mp4"
    try LTX2PostProcess.writeMP4(frames: frames, outputPath: mp4Path, fps: 24, width: width, height: height)
    logger.info("Wrote \(mp4Path)")
  }
}
