import Foundation

/// Configuration for the Wan 2.2 I2V generation pipeline.
///
/// Default values match `wan_i2v_A14B.py` and `shared_config.py`.
public struct WanI2VConfig: Sendable {

  /// Path to the model directory containing high_noise_model/, low_noise_model/, VAE, and T5.
  public let modelDir: String

  /// Noise schedule shift parameter (5.0 for 720p, 3.0 for 480p).
  public let shift: Float

  /// Number of denoising steps.
  public let steps: Int

  /// MoE boundary threshold (0.0-1.0, multiplied by numTrainTimesteps internally).
  public let boundary: Float

  /// CFG guide scale: (lowNoise, highNoise).
  public let guideScale: (Float, Float)

  /// Output video FPS.
  public let fps: Int

  /// Maximum pixel area for resolution computation.
  public let maxArea: Int

  /// Number of frames per chunk (must be 4n+1).
  public let frameNum: Int

  /// Negative prompt (Chinese default from shared_config.py).
  public let negativePrompt: String

  /// Number of training timesteps.
  public let numTrainTimesteps: Int

  /// Whether to use lazy MoE loading (one expert at a time).
  public let lazyMoE: Bool

  /// Default negative prompt from Wan shared_config.py.
  ///
  /// Chinese text that suppresses common artifacts:
  /// overexposure, static, blurry details, subtitles, style artifacts,
  /// paintings, gray overall, worst quality, low quality, JPEG artifacts,
  /// ugly, extra fingers, poorly drawn hands/faces, deformed, mutated limbs,
  /// fused fingers, static frames, cluttered backgrounds, three legs, etc.
  public static let defaultNegativePrompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

  /// Default configuration for Wan I2V-A14B.
  public static let `default` = WanI2VConfig(
    modelDir: "/Volumes/Bolt/Models/wan22-i2v",
    shift: 5.0,
    steps: 40,
    boundary: 0.9,
    guideScale: (3.5, 3.5),
    fps: 16,
    maxArea: 720 * 1280,
    frameNum: 81,
    negativePrompt: defaultNegativePrompt,
    numTrainTimesteps: 1000,
    lazyMoE: true
  )

  public init(
    modelDir: String,
    shift: Float = 5.0,
    steps: Int = 40,
    boundary: Float = 0.9,
    guideScale: (Float, Float) = (3.5, 3.5),
    fps: Int = 16,
    maxArea: Int = 720 * 1280,
    frameNum: Int = 81,
    negativePrompt: String = defaultNegativePrompt,
    numTrainTimesteps: Int = 1000,
    lazyMoE: Bool = true
  ) {
    precondition((frameNum - 1) % 4 == 0, "frameNum must be 4n+1, got \(frameNum)")
    self.modelDir = modelDir
    self.shift = shift
    self.steps = steps
    self.boundary = boundary
    self.guideScale = guideScale
    self.fps = fps
    self.maxArea = maxArea
    self.frameNum = frameNum
    self.negativePrompt = negativePrompt
    self.numTrainTimesteps = numTrainTimesteps
    self.lazyMoE = lazyMoE
  }
}
