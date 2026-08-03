// LTX2AudioVAE.swift — 2D mel-spectrogram VAE for JoyAI-Echo (Phase 2)
//
// JoyAI-Echo's audio half runs on a compact 2D convolutional VAE (`audio_vae.*`,
// 102 tensors) that compresses a stereo mel spectrogram to an 8-channel latent
// and back. It is a standard AutoencoderKL topology (down / mid / up stages,
// resnet blocks with 1x1 `nin_shortcut`s, strided-conv downsample, NN-upsample)
// but — like the LTX-2 video VAE — it uses **PixelNorm** (weight-free L2 channel
// norm) and SiLU rather than GroupNorm, which is why the checkpoint carries no
// norm weights.
//
// Shape flow (channels-first logical view):
//   encode:  mel (B, 2, F, T)  → (B, 16, F/4, T/4) → split → mean (B, 8, F/4, T/4)
//   decode:  z   (B, 8, F/4, T/4) → (B, 2, F, T)
//
// Internally everything runs channels-last (NHWC = [B, F, T, C]) to match MLX's
// Conv2d weight layout `[O, kH, kW, I]`. Encode/decode accept and return the
// channels-first convention the rest of the LTX-2 stack uses.
//
// Config (from echo-keymap.json + tensor manifest):
//   in_channels 2 (stereo), mel_bins 64, z_channels 8 (double_z → 16 encoder
//   out), ch 128, ch_mult [1,2,4] → [128,256,512], 2 down / 2 up (4× compress),
//   16 kHz. per_channel_statistics carries 128-dim stats for the *patchified*
//   latent (8ch → 128ch via a 4×4 space-to-depth patch the pipeline applies).
//
// NOTE (fidelity, deferred): the reference uses causal conv padding on the time
// axis; this port uses symmetric same-padding, which is shape-exact and decodes
// non-silent audio but is not numerically golden. Causal-padding parity is a
// Phase-4 item (golden tensors unavailable on this box).

import Foundation
import Logging
import MLX
import MLXNN

// MARK: - Config

public struct LTX2AudioVAEConfig {
  public var inChannels: Int = 2
  public var baseChannels: Int = 128
  public var channelMultipliers: [Int] = [1, 2, 4]
  public var numResBlocks: Int = 2      // encoder blocks per level (decoder = +1)
  public var zChannels: Int = 8
  public var latentChannels: Int = 128  // patchified latent width (stats dim)
  public var eps: Float = 1e-6

  public init() {}

  /// Resolved per-level channel counts, e.g. [128, 256, 512].
  public var levelChannels: [Int] { channelMultipliers.map { $0 * baseChannels } }
}

// MARK: - Shared conv wrapper (`<name>.conv.weight`)

/// A single conv wrapped in a `conv` child so the checkpoint's `foo.conv.weight`
/// key path resolves. Mirrors `LTX2ConvWrapper` (video VAE) but 2D.
public final class LTX2AudioConvWrapper: Module {
  @ModuleInfo(key: "conv") var conv: Conv2d

  let kernelSize: Int

  public init(_ inChannels: Int, _ outChannels: Int, kernelSize: Int = 3) {
    self.kernelSize = kernelSize
    // Padding is applied MANUALLY: causal (front-only) on TIME, symmetric on
    // FREQ — reference CausalConv2d with causality_axis=HEIGHT, where height
    // is time. Internal layout: (B, T, F, C).
    self._conv.wrappedValue = Conv2d(
      inputChannels: inChannels, outputChannels: outChannels,
      kernelSize: IntOrPair(kernelSize), padding: IntOrPair(0))
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    guard kernelSize > 1 else { return conv(x) }
    let k = kernelSize - 1
    let padded = MLX.padded(x, widths: [
      IntOrPair((0, 0)),          // B
      IntOrPair((k, 0)),          // T: full pad FRONT (causal)
      IntOrPair((k / 2, k - k / 2)),  // F: symmetric
      IntOrPair((0, 0)),          // C
    ])
    return conv(padded)
  }
}

// MARK: - Resnet block (PixelNorm + SiLU)

public final class LTX2AudioResnetBlock: Module {
  @ModuleInfo(key: "conv1") var conv1: LTX2AudioConvWrapper
  @ModuleInfo(key: "conv2") var conv2: LTX2AudioConvWrapper
  @ModuleInfo(key: "nin_shortcut") var ninShortcut: LTX2AudioConvWrapper?

  let eps: Float
  let hasShortcut: Bool

  public init(_ inChannels: Int, _ outChannels: Int, eps: Float = 1e-6) {
    self.eps = eps
    self.hasShortcut = inChannels != outChannels
    self._conv1.wrappedValue = LTX2AudioConvWrapper(inChannels, outChannels)
    self._conv2.wrappedValue = LTX2AudioConvWrapper(outChannels, outChannels)
    self._ninShortcut.wrappedValue = hasShortcut
      ? LTX2AudioConvWrapper(inChannels, outChannels, kernelSize: 1) : nil
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let residual = hasShortcut ? ninShortcut!(x) : x
    var h = conv1(silu(ltx2AudioPixelNorm(x, eps: eps)))
    h = conv2(silu(ltx2AudioPixelNorm(h, eps: eps)))
    return h + residual
  }
}

// MARK: - Down / Up / Mid stages

/// Encoder downsample: asymmetric pad + stride-2 3x3 conv (`downsample.conv.weight`).
public final class LTX2AudioDownsample: Module {
  @ModuleInfo(key: "conv") var conv: Conv2d

  public init(_ channels: Int) {
    self._conv.wrappedValue = Conv2d(
      inputChannels: channels, outputChannels: channels,
      kernelSize: 3, stride: 2, padding: 0)
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // NHWC: pad bottom/right by 1 on F(H) and T(W), then stride-2 3x3.
    let padded = MLX.padded(
      x, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair((0, 0))])
    return conv(padded)
  }
}

/// Decoder upsample: nearest-neighbor 2× then 3x3 conv (`upsample.conv.conv.weight`).
public final class LTX2AudioUpsample: Module {
  @ModuleInfo(key: "conv") var conv: LTX2AudioConvWrapper

  public init(_ channels: Int) {
    self._conv.wrappedValue = LTX2AudioConvWrapper(channels, channels)
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = MLX.repeated(x, count: 2, axis: 1)   // T
    h = MLX.repeated(h, count: 2, axis: 2)       // F
    h = conv(h)
    // Causal: drop the FIRST time element to undo the encoder-side padding
    // (reference Upsample: x[:, :, 1:, :] on the causal axis). Two decoder
    // levels turn T latents into 4T-3 mel frames exactly.
    return h[0..., 1..., 0..., 0...]
  }
}

public final class LTX2AudioEncoderLevel: Module {
  @ModuleInfo(key: "block") var block: [LTX2AudioResnetBlock]
  @ModuleInfo(key: "downsample") var downsample: LTX2AudioDownsample?

  public init(inChannels: Int, outChannels: Int, numBlocks: Int, downsample: Bool) {
    var blocks: [LTX2AudioResnetBlock] = []
    var dim = inChannels
    for _ in 0..<numBlocks {
      blocks.append(LTX2AudioResnetBlock(dim, outChannels))
      dim = outChannels
    }
    self._block.wrappedValue = blocks
    self._downsample.wrappedValue = downsample ? LTX2AudioDownsample(outChannels) : nil
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = x
    for b in block { h = b(h) }
    if let d = downsample { h = d(h) }
    return h
  }
}

public final class LTX2AudioDecoderLevel: Module {
  @ModuleInfo(key: "block") var block: [LTX2AudioResnetBlock]
  @ModuleInfo(key: "upsample") var upsample: LTX2AudioUpsample?

  public init(inChannels: Int, outChannels: Int, numBlocks: Int, upsample: Bool) {
    var blocks: [LTX2AudioResnetBlock] = []
    var dim = inChannels
    for _ in 0..<numBlocks {
      blocks.append(LTX2AudioResnetBlock(dim, outChannels))
      dim = outChannels
    }
    self._block.wrappedValue = blocks
    self._upsample.wrappedValue = upsample ? LTX2AudioUpsample(outChannels) : nil
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = x
    for b in block { h = b(h) }
    if let u = upsample { h = u(h) }
    return h
  }
}

/// Mid block: two resnets, no attention (checkpoint carries no mid-attn keys).
public final class LTX2AudioMidBlock: Module {
  @ModuleInfo(key: "block_1") var block1: LTX2AudioResnetBlock
  @ModuleInfo(key: "block_2") var block2: LTX2AudioResnetBlock

  public init(_ channels: Int) {
    self._block1.wrappedValue = LTX2AudioResnetBlock(channels, channels)
    self._block2.wrappedValue = LTX2AudioResnetBlock(channels, channels)
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray { block2(block1(x)) }
}

// MARK: - Encoder / Decoder

public final class LTX2AudioEncoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: LTX2AudioConvWrapper
  @ModuleInfo(key: "down") var down: [LTX2AudioEncoderLevel]
  @ModuleInfo(key: "mid") var mid: LTX2AudioMidBlock
  @ModuleInfo(key: "conv_out") var convOut: LTX2AudioConvWrapper
  let eps: Float

  public init(config: LTX2AudioVAEConfig) {
    self.eps = config.eps
    let chans = config.levelChannels
    let nLevels = chans.count

    self._convIn.wrappedValue = LTX2AudioConvWrapper(config.inChannels, chans[0])

    var levels: [LTX2AudioEncoderLevel] = []
    for i in 0..<nLevels {
      let inC = i == 0 ? chans[0] : chans[i - 1]
      let outC = chans[i]
      // Downsample on every level except the last (2 downsamples for 3 levels).
      levels.append(LTX2AudioEncoderLevel(
        inChannels: inC, outChannels: outC,
        numBlocks: config.numResBlocks, downsample: i < nLevels - 1))
    }
    self._down.wrappedValue = levels

    self._mid.wrappedValue = LTX2AudioMidBlock(chans[nLevels - 1])
    // double_z: emit 2 * zChannels.
    self._convOut.wrappedValue = LTX2AudioConvWrapper(chans[nLevels - 1], 2 * config.zChannels)
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = convIn(x)
    for level in down { h = level(h) }
    h = mid(h)
    h = convOut(silu(ltx2AudioPixelNorm(h, eps: eps)))
    return h
  }
}

public final class LTX2AudioDecoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: LTX2AudioConvWrapper
  @ModuleInfo(key: "mid") var mid: LTX2AudioMidBlock
  @ModuleInfo(key: "up") var up: [LTX2AudioDecoderLevel]
  @ModuleInfo(key: "conv_out") var convOut: LTX2AudioConvWrapper
  let eps: Float

  public init(config: LTX2AudioVAEConfig) {
    self.eps = config.eps
    let chans = config.levelChannels          // [128, 256, 512]
    let nLevels = chans.count
    let topCh = chans[nLevels - 1]

    self._convIn.wrappedValue = LTX2AudioConvWrapper(config.zChannels, topCh)
    self._mid.wrappedValue = LTX2AudioMidBlock(topCh)

    // Levels are stored 0..nLevels-1 but the checkpoint's `up.i` are indexed so
    // that up.(nLevels-1) is the highest-channel (deepest) stage; decode walks
    // them in descending order. up.i output channel = chans[i]; input channel =
    // chans[i+1] for i < top (channel halving), else topCh. Upsample on every
    // level except up.0 (2 upsamples for 3 levels).
    var levels: [LTX2AudioDecoderLevel] = []
    for i in 0..<nLevels {
      let outC = chans[i]
      let inC = i == nLevels - 1 ? topCh : chans[i + 1]
      levels.append(LTX2AudioDecoderLevel(
        inChannels: inC, outChannels: outC,
        numBlocks: config.numResBlocks + 1, upsample: i > 0))
    }
    self._up.wrappedValue = levels

    self._convOut.wrappedValue = LTX2AudioConvWrapper(chans[0], config.inChannels)
    super.init()
  }

  public func callAsFunction(_ z: MLXArray) -> MLXArray {
    var h = convIn(z)
    h = mid(h)
    for i in stride(from: up.count - 1, through: 0, by: -1) { h = up[i](h) }
    h = convOut(silu(ltx2AudioPixelNorm(h, eps: eps)))
    return h
  }
}

// MARK: - Top-level VAE

public final class LTX2AudioVAE: Module {
  @ModuleInfo(key: "encoder") var encoder: LTX2AudioEncoder
  @ModuleInfo(key: "decoder") var decoder: LTX2AudioDecoder
  @ModuleInfo(key: "per_channel_statistics") var perChannelStatistics: LTX2PerChannelStatistics

  public let config: LTX2AudioVAEConfig

  /// Bound by `load(path:)` for full-chain decode; nil for VAE-only use.
  public var vocoder: LTX2Vocoder?

  public init(config: LTX2AudioVAEConfig = LTX2AudioVAEConfig()) {
    self.config = config
    self._encoder.wrappedValue = LTX2AudioEncoder(config: config)
    self._decoder.wrappedValue = LTX2AudioDecoder(config: config)
    self._perChannelStatistics.wrappedValue =
      LTX2PerChannelStatistics(latentChannels: config.latentChannels)
    super.init()
  }

  /// Encode a mel spectrogram to the posterior-mean latent.
  ///
  /// - Parameter mel: `(B, 2, F, T)` channels-first.
  /// - Returns: `(B, zChannels, F/4, T/4)` channels-first (distribution mean).
  public func encode(_ mel: MLXArray) -> MLXArray {
    let nhwc = mel.transposed(0, 2, 3, 1)         // (B, F, T, 2)
    let moments = encoder(nhwc)                    // (B, F/4, T/4, 16)
    let z = config.zChannels
    let mean = moments[0..., 0..., 0..., 0..<z]    // first zChannels = mean
    return mean.transposed(0, 3, 1, 2)             // (B, z, F/4, T/4)
  }

  /// Decode a latent back to a mel spectrogram.
  ///
  /// - Parameter z: `(B, zChannels, F/4, T/4)` channels-first.
  /// - Returns: `(B, 2, F, T)` channels-first.
  public func decode(_ z: MLXArray) -> MLXArray {
    let nhwc = z.transposed(0, 2, 3, 1)            // (B, F/4, T/4, z)
    let mel = decoder(nhwc)                        // (B, F, T, 2)
    return mel.transposed(0, 3, 1, 2)              // (B, 2, F, T)
  }
}

// MARK: - PixelNorm (channels-last)

/// L2 channel-normalization over the last (channel) axis: `x / sqrt(mean(x²)+eps)`.
@inline(__always)
func ltx2AudioPixelNorm(_ x: MLXArray, eps: Float) -> MLXArray {
  x / MLX.sqrt(MLX.mean(x * x, axis: -1, keepDims: true) + eps)
}

// MARK: - Weight loading

extension LTX2AudioVAE {
  /// Remap monolith `audio_vae.*` keys to this module's parameter namespace and
  /// transpose PyTorch conv weights `[O,I,H,W]` → MLX `[O,H,W,I]`.
  public static func remapKeys(_ tensors: [String: MLXArray]) -> [(String, MLXArray)] {
    var out: [(String, MLXArray)] = []
    for (key, value) in tensors {
      guard key.hasPrefix("audio_vae.") else { continue }

      var newKey: String
      if key == "audio_vae.per_channel_statistics.mean-of-means" {
        newKey = "per_channel_statistics.mean"
      } else if key == "audio_vae.per_channel_statistics.std-of-means" {
        newKey = "per_channel_statistics.std"
      } else if key.hasPrefix("audio_vae.per_channel_statistics.") {
        continue  // drop any other statistics (already-mirrored decoder copies etc.)
      } else if key.hasPrefix("audio_vae.encoder.") {
        newKey = "encoder." + String(key.dropFirst("audio_vae.encoder.".count))
      } else if key.hasPrefix("audio_vae.decoder.") {
        newKey = "decoder." + String(key.dropFirst("audio_vae.decoder.".count))
      } else {
        continue
      }

      var newValue = value
      // 4D conv weights are PyTorch OIHW; MLX Conv2d wants OHWI.
      if newKey.hasSuffix(".weight") && newValue.ndim == 4 {
        newValue = newValue.transposed(0, 2, 3, 1)
      }
      out.append((newKey, newValue))
    }
    return out
  }

  /// Apply monolith audio-VAE weights with an anti-noise guard (mirrors
  /// `LTX2VideoGenerator.load`): throws if the remap covers <half the module's
  /// parameters, which would otherwise decode silence while reporting success.
  ///
  /// - Returns: `(matched, moduleParamCount)`.
  @discardableResult
  public func loadWeightsFromTensors(
    tensors: [String: MLXArray], logger: Logger
  ) throws -> (matched: Int, total: Int) {
    let remapped = LTX2AudioVAE.remapKeys(tensors)
    let moduleKeys = Set(parameters().flattened().map { $0.0 })
    let matched = remapped.filter { moduleKeys.contains($0.0) }.count
    logger.info(
      "LTX-2 audio VAE: remap matched \(matched)/\(moduleKeys.count) module params (\(remapped.count) remapped keys).")
    if matched * 2 < moduleKeys.count {
      throw LTX2AudioVAEError.weightCoverageTooLow(matched: matched, total: moduleKeys.count)
    }
    let params = ModuleParameters.unflattened(remapped.map { ($0.0, $0.1) })
    try update(parameters: params, verify: [.shapeMismatch])
    return (matched, moduleKeys.count)
  }
}

// MARK: - Reference-layout codec APIs (task #21 parity)

extension LTX2AudioVAE {
  /// Load the standalone reference checkpoint (audio_vae.* + vocoder.* keys):
  /// binds the VAE half AND the vocoder chain, both coverage-guarded.
  public static func load(path: String, logger: Logger = Logger(label: "ltx2.audio-vae")) throws -> LTX2AudioVAE {
    let tensors = try MLX.loadArrays(url: URL(fileURLWithPath: path)).mapValues { $0.asType(.float32) }
    let vae = LTX2AudioVAE()
    try vae.loadWeightsFromTensors(tensors: tensors, logger: logger)
    let voc = LTX2Vocoder()
    try voc.loadWeightsFromTensors(tensors: tensors, logger: logger)
    vae.vocoder = voc
    eval(vae.parameters())
    eval(voc.parameters())
    return vae
  }

  /// Full reference decode: normalized latent (B,C,T,F) → mel → base vocoder
  /// → BWE → 48 kHz stereo waveform (B, 2, samples). Mirrors
  /// AudioVAE.decode + run_vocoder: mel (B,2,T,F) transposes to (B,2,F,T),
  /// stereo-folds to [B,128,T], then the full BWE chain.
  public func decodeToWaveform(_ zNormalized: MLXArray) -> MLXArray {
    guard let vocoder else {
      fatalError("decodeToWaveform requires load(path:) — vocoder not bound")
    }
    let mel = decodeToMel(zNormalized)                 // (B, 2, T, F)
    let bft = mel.transposed(0, 1, 3, 2)               // (B, 2, F, T)
    let folded = MLX.concatenated([bft[0..., 0], bft[0..., 1]], axis: 1)  // [B, 128, T]
    return vocoder.synthesizeFull(folded)
  }

  /// Per-channel latent denormalization: `z * std + mean` where the 128-long
  /// statistics vectors are the FLATTENED (c f) token layout (reference
  /// AudioPatchifier: b c t f -> b t (c f)). Input/output: (B, C, T, F).
  public func denormalize(_ z: MLXArray) -> MLXArray {
    let c = z.dim(1), f = z.dim(3)
    let std = perChannelStatistics.std.reshaped([1, c, 1, f])
    let mean = perChannelStatistics.mean.reshaped([1, c, 1, f])
    return z * std + mean
  }

  /// Full reference decode contract: normalized latent (B, C, T, F) ->
  /// denorm -> causal decode -> mel (B, 2, 4T-3, melBins).
  public func decodeToMel(_ zNormalized: MLXArray) -> MLXArray {
    let t = zNormalized.dim(2)
    let zDenorm = denormalize(zNormalized)
    // Internal layout: (B, T, F, C) — time as the causal height axis.
    let nhwc = zDenorm.transposed(0, 2, 3, 1)
    var melNHWC = decoder(nhwc)                       // (B, T', F', 2)
    // Reference _adjust_output_shape: crop to (2, 4T-3, melBins). The two
    // causal upsample drops land T' = 4T-3 exactly; freq crops to mel bins.
    let targetT = 4 * t - 3
    let melBins = 64
    melNHWC = melNHWC[0..., 0..<min(melNHWC.dim(1), targetT), 0..<min(melNHWC.dim(2), melBins), 0..<2]
    return melNHWC.transposed(0, 3, 1, 2)             // (B, 2, T, F)
  }
}

public enum LTX2AudioVAEError: Error, CustomStringConvertible {
  case weightCoverageTooLow(matched: Int, total: Int)

  public var description: String {
    switch self {
    case .weightCoverageTooLow(let matched, let total):
      return "audio VAE weight remap matched only \(matched)/\(total) module params — unrecognized key format; decode would be silence"
    }
  }
}
