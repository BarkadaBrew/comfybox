import Foundation

/// Configuration for SeedVR2 model variants (3B and 7B).
///
/// The 3B and 7B variants have different architectures:
///
/// | Parameter | 3B | 7B |
/// |-----------|----|----|
/// | vidDim | 2560 | 3072 |
/// | heads | 20 | 24 |
/// | numLayers | 32 | 36 |
/// | mmLayers | 10 | 36 (all separate) |
/// | mlpType | SwiGLU | GELU |
/// | hasOutputNorm | true | false |
///
/// The model variant is auto-detected from the weights directory by checking
/// which transformer weight file is present.
public struct SeedVR2ModelConfig: Equatable {

  /// Video/text feature dimension.
  public let vidDim: Int

  /// Text encoder input dimension (always 5120).
  public let txtInDim: Int

  /// Number of attention heads.
  public let heads: Int

  /// Dimension per attention head (always 128).
  public let headDim: Int

  /// RoPE dimension. 3B uses 128 (full head_dim), 7B uses 64 (head_dim/2).
  public let ropeDim: Int

  /// MLP expansion ratio (always 4).
  public let expandRatio: Int

  /// Total number of transformer blocks.
  public let numLayers: Int

  /// Number of multi-modal (separate weight) layers. Blocks >= mmLayers share weights.
  public let mmLayers: Int

  /// MLP type used in transformer blocks.
  public let mlpType: SeedVR2MLPType

  /// Whether the transformer has output normalization and adaptive shift/scale.
  /// True for 3B (uses vid_out_norm + out_shift + out_scale), false for 7B.
  public let hasOutputNorm: Bool

  /// Whether the last block freezes text (vid-only). True for 3B, false for 7B.
  public let hasLastLayerFreeze: Bool

  /// Transformer weight file name (without path).
  public let transformerWeightFile: String

  /// FP8 transformer weight file name (without path).
  public let transformerWeightFileFP8: String

  /// 3B model configuration preset.
  public static let preset3B = SeedVR2ModelConfig(
    vidDim: 2560,
    txtInDim: 5120,
    heads: 20,
    headDim: 128,
    ropeDim: 128,
    expandRatio: 4,
    numLayers: 32,
    mmLayers: 10,
    mlpType: .swiglu,
    hasOutputNorm: true,
    hasLastLayerFreeze: true,
    transformerWeightFile: "seedvr2_ema_3b_fp16.safetensors",
    transformerWeightFileFP8: "seedvr2_ema_3b_fp8.safetensors"
  )

  /// 7B model configuration preset.
  public static let preset7B = SeedVR2ModelConfig(
    vidDim: 3072,
    txtInDim: 5120,
    heads: 24,
    headDim: 128,
    ropeDim: 64,
    expandRatio: 4,
    numLayers: 36,
    mmLayers: 36,
    mlpType: .gelu,
    hasOutputNorm: false,
    hasLastLayerFreeze: false,
    transformerWeightFile: "seedvr2_ema_7b_fp16.safetensors",
    transformerWeightFileFP8: "seedvr2_ema_7b_fp8.safetensors"
  )

  /// Auto-detects the model variant from the weights directory.
  ///
  /// Checks for the presence of 7B or 3B weight files and returns the corresponding config.
  /// Prefers 7B if both are present.
  ///
  /// - Parameter directory: The weights directory URL.
  /// - Returns: The detected model config, or nil if no recognized weight file is found.
  public static func detect(from directory: URL) -> SeedVR2ModelConfig? {
    let fm = FileManager.default

    // Check for 7B files first
    if fm.fileExists(atPath: directory.appendingPathComponent(preset7B.transformerWeightFile).path)
      || fm.fileExists(atPath: directory.appendingPathComponent(preset7B.transformerWeightFileFP8).path)
    {
      return .preset7B
    }

    // Check for 3B files
    if fm.fileExists(atPath: directory.appendingPathComponent(preset3B.transformerWeightFile).path)
      || fm.fileExists(atPath: directory.appendingPathComponent(preset3B.transformerWeightFileFP8).path)
    {
      return .preset3B
    }

    return nil
  }
}

/// MLP type variants used in SeedVR2 transformer blocks.
public enum SeedVR2MLPType: Equatable {
  /// SwiGLU MLP (3B): gate = SiLU(proj_in_gate(x)) * proj_in(x), out = proj_out(gate). No bias.
  case swiglu
  /// Standard GELU MLP (7B): out = proj_out(GELU(proj_in(x))). Has bias.
  case gelu
}
