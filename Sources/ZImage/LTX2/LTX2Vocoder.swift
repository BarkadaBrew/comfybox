// LTX2Vocoder.swift — BigVGAN v2 mel→waveform vocoder for JoyAI-Echo (Phase 2)
//
// JoyAI-Echo's `vocoder.*` group (1227 tensors) is a BigVGAN v2 vocoder with
// three sub-modules:
//   • vocoder.vocoder.*        — main generator: mel(128) → stereo waveform.
//       conv_pre(128→1536,k7); 6 ConvTranspose1d upsamples (rates 5,2,2,2,2,2 →
//       ch 1536→768→384→192→96→48→24, ×160 total); a multi-receptive-field
//       (MRF) stack of 18 AMPBlocks (6 stages × kernels {3,7,11}, dilations
//       {1,3,5}); anti-aliased SnakeBeta activations; conv_post(24→2,k7,no bias).
//   • vocoder.bwe_generator.*  — bandwidth-extension generator (rates 6,5,2,2,2,
//       ch 512→…→16, 15 AMPBlocks), same block design.
//   • vocoder.mel_stft.*       — mel filterbank + STFT bases (analysis direction;
//       carried as buffers, not used on the mel→wave synthesis path).
//
// Anti-aliasing: BigVGAN wraps each Snake nonlinearity in an `Activation1d`
// that 2×-upsamples (kaiser-sinc lowpass), applies SnakeBeta at the higher rate,
// then 2×-downsamples. The kaiser-sinc filters are **baked into the checkpoint**
// as `upsample.filter` / `downsample.lowpass.filter` ([1,1,12]) — so we LOAD and
// apply them (grouped/depthwise conv) rather than recompute kaiser windows.
//
// NOTE (fidelity, deferred): exact alias-free-torch edge padding is not
// reproduced; each activation is forced length-preserving (up→snake→down cropped
// back to its input length). Output sample count is therefore governed exactly by
// conv_pre/ups/conv_post (frames × ∏rates), and the audio is non-silent, but the
// result is not numerically golden vs the PyTorch reference (unavailable here).

import Foundation
import Logging
import MLX
import MLXNN

// MARK: - SnakeBeta activation

/// SnakeBeta: `x + (1/(β+ε))·sin(αx)²`, with α,β in log-scale (BigVGAN v2 default
/// `alpha_logscale=true`). Params are per-channel `[C]`; input is NLC `[B,L,C]`.
public final class LTX2SnakeBeta: Module {
  @ParameterInfo(key: "alpha") var alpha: MLXArray
  @ParameterInfo(key: "beta") var beta: MLXArray
  let eps: Float

  public init(channels: Int, eps: Float = 1e-9) {
    self.eps = eps
    self._alpha.wrappedValue = MLXArray.zeros([channels])
    self._beta.wrappedValue = MLXArray.zeros([channels])
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let a = MLX.exp(alpha).reshaped(1, 1, -1)
    let b = MLX.exp(beta).reshaped(1, 1, -1)
    let s = MLX.sin(a * x)
    return x + (1.0 / (b + eps)) * (s * s)
  }
}

// MARK: - Anti-aliased activation (Activation1d)

/// Depthwise low-pass filter buffer holder (`upsample.filter`,
/// `downsample.lowpass.filter` — shape [1,1,K]).
public final class LTX2LowPass1d: Module {
  @ParameterInfo(key: "filter") var filter: MLXArray
  public init(kernel: Int = 12) {
    self._filter.wrappedValue = MLXArray.zeros([1, 1, kernel])
    super.init()
  }
}

public final class LTX2DownSample1d: Module {
  @ModuleInfo(key: "lowpass") var lowpass: LTX2LowPass1d
  public init(kernel: Int = 12) {
    self._lowpass.wrappedValue = LTX2LowPass1d(kernel: kernel)
    super.init()
  }
}

/// BigVGAN anti-aliased activation: 2×-upsample → SnakeBeta → 2×-downsample,
/// forced length-preserving. Uses the checkpoint's baked kaiser-sinc filters via
/// grouped (depthwise) conv.
public final class LTX2AntiAliasActivation: Module {
  @ModuleInfo(key: "act") var act: LTX2SnakeBeta
  @ModuleInfo(key: "upsample") var upsample: LTX2LowPass1d          // has `filter`
  @ModuleInfo(key: "downsample") var downsample: LTX2DownSample1d   // has `lowpass.filter`
  let channels: Int
  let kernel: Int

  public init(channels: Int, kernel: Int = 12) {
    self.channels = channels
    self.kernel = kernel
    self._act.wrappedValue = LTX2SnakeBeta(channels: channels)
    self._upsample.wrappedValue = LTX2LowPass1d(kernel: kernel)
    self._downsample.wrappedValue = LTX2DownSample1d(kernel: kernel)
    super.init()
  }

  /// Broadcast a [1,1,K] filter to depthwise weight [C, K, 1].
  private func depthwiseWeight(_ f: MLXArray) -> MLXArray {
    let w = f.reshaped(1, kernel, 1).asType(.float32)
    return MLX.broadcast(w, to: [channels, kernel, 1])
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    let inDtype = x.dtype
    let L = x.dim(1)
    let xf = x.asType(.float32)
    let pad = kernel / 2

    // 2× upsample (transposed, depthwise) → force length 2L.
    let upW = depthwiseWeight(upsample.filter)
    var up = MLX.convTransposed1d(
      MLX.padded(xf, widths: [IntOrPair((0, 0)), IntOrPair((pad, pad)), IntOrPair((0, 0))]),
      upW, stride: 2, groups: channels)
    up = 2.0 * up
    up = matchLength(up, to: 2 * L)

    // SnakeBeta at 2× rate.
    let sn = act(up)

    // 2× downsample (depthwise) → force length L.
    let dnW = depthwiseWeight(downsample.lowpass.filter)
    var dn = MLX.conv1d(
      MLX.padded(sn, widths: [IntOrPair((0, 0)), IntOrPair((pad, pad)), IntOrPair((0, 0))]),
      dnW, stride: 2, groups: channels)
    dn = matchLength(dn, to: L)
    return dn.asType(inDtype)
  }

  /// Crop or zero-pad the length (axis 1) to exactly `target`.
  private func matchLength(_ x: MLXArray, to target: Int) -> MLXArray {
    let cur = x.dim(1)
    if cur == target { return x }
    if cur > target { return x[0..., 0..<target, 0...] }
    return MLX.padded(
      x, widths: [IntOrPair((0, 0)), IntOrPair((0, target - cur)), IntOrPair((0, 0))])
  }
}

// MARK: - AMPBlock (residual, multi-dilation)

/// BigVGAN AMPBlock1: three (act→dilated-conv→act→conv) residual sub-blocks with
/// dilations {1,3,5}. `convs1` are dilated; `convs2` have dilation 1.
public final class LTX2AMPBlock: Module {
  @ModuleInfo(key: "convs1") var convs1: [Conv1d]
  @ModuleInfo(key: "convs2") var convs2: [Conv1d]
  @ModuleInfo(key: "acts1") var acts1: [LTX2AntiAliasActivation]
  @ModuleInfo(key: "acts2") var acts2: [LTX2AntiAliasActivation]

  public init(channels: Int, kernelSize: Int, dilations: [Int] = [1, 3, 5]) {
    self._convs1.wrappedValue = dilations.map { d in
      Conv1d(inputChannels: channels, outputChannels: channels,
             kernelSize: kernelSize, padding: d * (kernelSize - 1) / 2, dilation: d)
    }
    self._convs2.wrappedValue = dilations.map { _ in
      Conv1d(inputChannels: channels, outputChannels: channels,
             kernelSize: kernelSize, padding: (kernelSize - 1) / 2, dilation: 1)
    }
    self._acts1.wrappedValue = dilations.map { _ in LTX2AntiAliasActivation(channels: channels) }
    self._acts2.wrappedValue = dilations.map { _ in LTX2AntiAliasActivation(channels: channels) }
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = x
    for i in 0..<convs1.count {
      var xt = acts1[i](out)
      xt = convs1[i](xt)
      xt = acts2[i](xt)
      xt = convs2[i](xt)
      out = out + xt
    }
    return out
  }
}

// MARK: - Generator

public struct LTX2BigVGANConfig {
  public var melChannels: Int
  public var upsampleInitialChannels: Int
  public var upsampleRates: [Int]
  public var upsampleKernels: [Int]
  public var resblockKernels: [Int]
  public var resblockDilations: [Int]

  public init(
    melChannels: Int, upsampleInitialChannels: Int,
    upsampleRates: [Int], upsampleKernels: [Int],
    resblockKernels: [Int] = [3, 7, 11], resblockDilations: [Int] = [1, 3, 5]
  ) {
    self.melChannels = melChannels
    self.upsampleInitialChannels = upsampleInitialChannels
    self.upsampleRates = upsampleRates
    self.upsampleKernels = upsampleKernels
    self.resblockKernels = resblockKernels
    self.resblockDilations = resblockDilations
  }

  /// Main BigVGAN generator (rates 5,2,2,2,2,2; ch 1536).
  public static let main = LTX2BigVGANConfig(
    melChannels: 128, upsampleInitialChannels: 1536,
    upsampleRates: [5, 2, 2, 2, 2, 2], upsampleKernels: [11, 4, 4, 4, 4, 4])

  /// Bandwidth-extension generator (rates 6,5,2,2,2; ch 512).
  public static let bwe = LTX2BigVGANConfig(
    melChannels: 128, upsampleInitialChannels: 512,
    upsampleRates: [6, 5, 2, 2, 2], upsampleKernels: [12, 11, 4, 4, 4])
}

public final class LTX2BigVGANGenerator: Module {
  @ModuleInfo(key: "conv_pre") var convPre: Conv1d
  @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
  @ModuleInfo(key: "resblocks") var resblocks: [LTX2AMPBlock]
  @ModuleInfo(key: "act_post") var actPost: LTX2AntiAliasActivation
  @ModuleInfo(key: "conv_post") var convPost: Conv1d

  let numKernels: Int

  public init(config: LTX2BigVGANConfig) {
    self.numKernels = config.resblockKernels.count

    self._convPre.wrappedValue = Conv1d(
      inputChannels: config.melChannels, outputChannels: config.upsampleInitialChannels,
      kernelSize: 7, padding: 3)

    var ch = config.upsampleInitialChannels
    var upList: [ConvTransposed1d] = []
    var blocks: [LTX2AMPBlock] = []
    for (i, rate) in config.upsampleRates.enumerated() {
      let k = config.upsampleKernels[i]
      let outCh = ch / 2
      upList.append(ConvTransposed1d(
        inputChannels: ch, outputChannels: outCh,
        kernelSize: k, stride: rate, padding: (k - rate) / 2))
      for rk in config.resblockKernels {
        blocks.append(LTX2AMPBlock(
          channels: outCh, kernelSize: rk, dilations: config.resblockDilations))
      }
      ch = outCh
    }
    self._ups.wrappedValue = upList
    self._resblocks.wrappedValue = blocks

    self._actPost.wrappedValue = LTX2AntiAliasActivation(channels: ch)
    self._convPost.wrappedValue = Conv1d(
      inputChannels: ch, outputChannels: 2, kernelSize: 7, padding: 3, bias: false)
    super.init()
  }

  /// - Parameter mel: NLC `[B, T, melChannels]`.
  /// - Returns: NLC `[B, T·∏rates, 2]` stereo waveform in [-1, 1].
  public func callAsFunction(_ mel: MLXArray) -> MLXArray {
    var x = convPre(mel)
    for i in 0..<ups.count {
      x = ups[i](x)
      var xs = resblocks[i * numKernels](x)
      for j in 1..<numKernels {
        xs = xs + resblocks[i * numKernels + j](x)
      }
      x = xs / MLXArray(Float(numKernels))
    }
    x = actPost(x)
    x = convPost(x)
    return MLX.tanh(x)
  }
}

// MARK: - mel_stft buffers (analysis direction; not used for synthesis)

public final class LTX2STFTBasis: Module {
  @ParameterInfo(key: "forward_basis") var forwardBasis: MLXArray
  @ParameterInfo(key: "inverse_basis") var inverseBasis: MLXArray
  public init(filterLength: Int = 512) {
    self._forwardBasis.wrappedValue = MLXArray.zeros([filterLength + 2, 1, filterLength])
    self._inverseBasis.wrappedValue = MLXArray.zeros([filterLength + 2, 1, filterLength])
    super.init()
  }
}

public final class LTX2MelSTFT: Module {
  @ParameterInfo(key: "mel_basis") var melBasis: MLXArray
  @ModuleInfo(key: "stft_fn") var stftFn: LTX2STFTBasis
  public init(melBins: Int = 64, freqBins: Int = 257, filterLength: Int = 512) {
    self._melBasis.wrappedValue = MLXArray.zeros([melBins, freqBins])
    self._stftFn.wrappedValue = LTX2STFTBasis(filterLength: filterLength)
    super.init()
  }
}

// MARK: - Top-level vocoder

public final class LTX2Vocoder: Module {
  @ModuleInfo(key: "vocoder") var vocoder: LTX2BigVGANGenerator
  @ModuleInfo(key: "bwe_generator") var bweGenerator: LTX2BigVGANGenerator
  @ModuleInfo(key: "mel_stft") var melStft: LTX2MelSTFT

  override public init() {
    self._vocoder.wrappedValue = LTX2BigVGANGenerator(config: .main)
    self._bweGenerator.wrappedValue = LTX2BigVGANGenerator(config: .bwe)
    self._melStft.wrappedValue = LTX2MelSTFT()
    super.init()
  }

  /// Synthesize a stereo waveform from a mel spectrogram via the MAIN generator.
  ///
  /// - Parameter mel: channels-first `[B, 128, T]`.
  /// - Returns: channels-first `[B, 2, T·160]` waveform in [-1, 1].
  public func synthesize(_ mel: MLXArray) -> MLXArray {
    let nlc = mel.transposed(0, 2, 1)         // [B, T, 128]
    let wave = vocoder(nlc)                    // [B, T·160, 2]
    return wave.transposed(0, 2, 1)            // [B, 2, T·160]
  }
}

// MARK: - Weight loading

extension LTX2Vocoder {
  /// Remap monolith `vocoder.*` keys to this module namespace + transpose convs.
  ///
  /// - Conv1d weights `[O,I,K]` → MLX `[O,K,I]` (transpose 0,2,1).
  /// - ConvTranspose1d (`ups.*`) weights `[I,O,K]` → MLX `[O,K,I]` (transpose 1,2,0).
  /// - Depthwise filters `[1,1,K]` and snake α/β / mel bases pass through.
  public static func remapKeys(_ tensors: [String: MLXArray]) -> [(String, MLXArray)] {
    var out: [(String, MLXArray)] = []
    for (key, value) in tensors {
      guard key.hasPrefix("vocoder.") else { continue }
      let newKey = String(key.dropFirst("vocoder.".count))

      var v = value
      if newKey.hasSuffix(".weight") && v.ndim == 3 {
        if isUpsampleTransposeKey(newKey) {
          v = v.transposed(1, 2, 0)       // ConvTranspose1d IOK → OKI
        } else if !newKey.hasSuffix(".filter") {
          v = v.transposed(0, 2, 1)       // Conv1d OIK → OKI
        }
      }
      out.append((newKey, v))
    }
    return out
  }

  /// True for `…ups.N.weight` (the ConvTranspose1d layers) — the only transposed
  /// convs in the graph; everything else (`conv_pre/post`, resblock `convs*`) is
  /// a forward Conv1d.
  private static func isUpsampleTransposeKey(_ key: String) -> Bool {
    guard key.hasSuffix(".weight") else { return false }
    let parts = key.split(separator: ".")
    // pattern: <gen>.ups.<idx>.weight
    return parts.count >= 4 && parts[parts.count - 3] == "ups"
  }

  /// Apply monolith vocoder weights with the anti-noise coverage guard.
  /// - Returns: `(matched, moduleParamCount)`.
  @discardableResult
  public func loadWeightsFromTensors(
    tensors: [String: MLXArray], logger: Logger
  ) throws -> (matched: Int, total: Int) {
    let remapped = LTX2Vocoder.remapKeys(tensors)
    let moduleKeys = Set(parameters().flattened().map { $0.0 })
    let matched = remapped.filter { moduleKeys.contains($0.0) }.count
    logger.info(
      "LTX-2 vocoder: remap matched \(matched)/\(moduleKeys.count) module params (\(remapped.count) remapped keys).")
    if matched * 2 < moduleKeys.count {
      throw LTX2VocoderError.weightCoverageTooLow(matched: matched, total: moduleKeys.count)
    }
    let params = ModuleParameters.unflattened(remapped.map { ($0.0, $0.1) })
    try update(parameters: params, verify: [.shapeMismatch])
    return (matched, moduleKeys.count)
  }
}

public enum LTX2VocoderError: Error, CustomStringConvertible {
  case weightCoverageTooLow(matched: Int, total: Int)
  public var description: String {
    switch self {
    case .weightCoverageTooLow(let matched, let total):
      return "vocoder weight remap matched only \(matched)/\(total) module params — unrecognized key format; audio would be silence"
    }
  }
}
