import Foundation
import Logging
import MLX

/// Multi-chunk video generation (extend) for Wan 2.2 I2V.
///
/// Generates arbitrary-duration video by chaining multiple I2V chunks:
///
/// ```
/// Chunk 1: i2v from source image → 81 frames (~5s)
/// Chunk 2: i2v from last frame of chunk 1 → 81 frames
/// ...
/// Stitch: drop first frame of each continuation chunk, concatenate
/// ```
///
/// This is the simple last-frame extraction approach (same as LTX2).
/// Advanced context-aware extend (latent-space continuity) is future scope.
public enum WanExtend {

  // MARK: - Constants

  /// Default FPS for Wan I2V (from shared_config.py).
  public static let defaultFPS = 16

  // MARK: - Chunk Calculation

  /// Computes the number of chunks needed for a target duration.
  ///
  /// - Parameters:
  ///   - targetSeconds: Desired video duration in seconds.
  ///   - framesPerChunk: Frames per generation chunk (must be 4n+1).
  ///   - fps: Output frames per second.
  /// - Returns: Number of chunks needed.
  public static func chunksNeeded(
    targetSeconds: Float,
    framesPerChunk: Int = 81,
    fps: Int = defaultFPS
  ) -> Int {
    let totalFrames = Int(ceil(targetSeconds * Float(fps)))

    // First chunk produces framesPerChunk frames.
    // Each subsequent chunk produces (framesPerChunk - 1) frames
    // (dropping the duplicate first frame).
    if totalFrames <= framesPerChunk {
      return 1
    }

    let remainingFrames = totalFrames - framesPerChunk
    let framesPerContinuation = framesPerChunk - 1
    return 1 + Int(ceil(Float(remainingFrames) / Float(framesPerContinuation)))
  }

  /// Computes the total frame count for a given number of chunks.
  ///
  /// - Parameters:
  ///   - chunks: Number of generation chunks.
  ///   - framesPerChunk: Frames per chunk (must be 4n+1).
  /// - Returns: Total output frames after stitching.
  public static func totalFrames(
    chunks: Int,
    framesPerChunk: Int = 81
  ) -> Int {
    guard chunks > 0 else { return 0 }
    // First chunk: framesPerChunk
    // Each continuation: framesPerChunk - 1
    return framesPerChunk + (chunks - 1) * (framesPerChunk - 1)
  }

  // MARK: - Extended Generation

  /// Generates a multi-chunk video by chaining I2V generations.
  ///
  /// - Parameters:
  ///   - pipeline: The I2V pipeline.
  ///   - prompt: Text prompt.
  ///   - negativePrompt: Negative prompt (nil uses config default).
  ///   - image: Init image [C, H, W] in [0, 1] range.
  ///   - targetSeconds: Desired video duration in seconds.
  ///   - seed: Random seed for first chunk (incremented for each subsequent chunk).
  ///   - width: Explicit width override.
  ///   - height: Explicit height override.
  ///   - logger: Logger.
  ///   - progressCallback: Called with (currentChunk, totalChunks, stepInChunk, totalSteps).
  /// - Returns: Concatenated video frames [C, totalFrames, H, W] in [0, 1] range.
  public static func generateExtended(
    pipeline: WanI2VPipeline,
    prompt: String,
    negativePrompt: String? = nil,
    image: MLXArray,
    targetSeconds: Float,
    seed: UInt64? = nil,
    width: Int? = nil,
    height: Int? = nil,
    logger: Logger,
    progressCallback: ((Int, Int, Int, Int) -> Void)? = nil
  ) throws -> MLXArray {
    let numChunks = chunksNeeded(
      targetSeconds: targetSeconds,
      framesPerChunk: pipeline.config.frameNum,
      fps: pipeline.config.fps
    )

    let total = totalFrames(chunks: numChunks, framesPerChunk: pipeline.config.frameNum)
    let duration = Float(total) / Float(pipeline.config.fps)

    logger.info("Extend: \(numChunks) chunks, \(total) frames, \(String(format: "%.1f", duration))s at \(pipeline.config.fps)fps")

    var allFrames: [MLXArray] = []
    var currentImage = image
    var currentSeed = seed

    for chunk in 0..<numChunks {
      logger.info("Chunk \(chunk + 1)/\(numChunks)")

      let chunkVideo = try pipeline.generate(
        prompt: prompt,
        negativePrompt: negativePrompt,
        image: currentImage,
        seed: currentSeed,
        width: width,
        height: height,
        progressCallback: { step, total in
          progressCallback?(chunk + 1, numChunks, step, total)
        }
      )

      // Extract frames for stitching
      if chunk == 0 {
        // First chunk: keep all frames
        allFrames.append(chunkVideo)
      } else {
        // Continuation chunks: drop the first frame (duplicate of previous chunk's last frame)
        let numFrames = chunkVideo.dim(1)
        if numFrames > 1 {
          allFrames.append(chunkVideo[0..., 1..., 0..., 0...])
        }
      }

      // Prepare next chunk: use last frame as init image
      currentImage = WanI2VPipeline.extractLastFrame(chunkVideo)
      eval(currentImage)

      // Increment seed for variation
      if let s = currentSeed {
        currentSeed = s &+ 1
      }

      logger.info("Chunk \(chunk + 1) complete: \(chunkVideo.dim(1)) frames generated")
    }

    // Concatenate all frame segments along the temporal axis
    let stitched: MLXArray
    if allFrames.count == 1 {
      stitched = allFrames[0]
    } else {
      stitched = MLX.concatenated(allFrames, axis: 1)
    }

    logger.info("Extend complete: \(stitched.dim(1)) frames total")
    return stitched
  }
}
