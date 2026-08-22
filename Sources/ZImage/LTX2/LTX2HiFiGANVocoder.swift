// LTX2HiFiGANVocoder.swift — the OFFICIAL Lightricks LTX-2.3 vocoder (`LTX2Vocoder`
// class, HiFi-GAN generator). Todd 2026-08-17.
//
// WHY: the JoyAI/pinkcherry checkpoint bundled a FOREIGN BigVGAN-v2 vocoder
// (LTX2Vocoder.swift, 1227 tensors) paired with the official audio_vae — the
// mismatch is what makes the audio METALLIC. The official audio_vae was trained
// matched to THIS HiFi-GAN vocoder (194 tensors, bf16). Loading it and routing
// audio decode through it is the real fix (validated: our audio_vae shapes match
// the official; only the vocoder differs).
//
// Architecture (official vocoder/config.json, `_class_name: LTX2Vocoder`):
//   in_channels 128 (mel) → conv_in(128→1024,k7) → 5 upsample stages
//   rates [6,5,2,2,2], kernels [16,15,8,4,4], channels 1024→512→256→128→64→32,
//   each followed by an MRF = mean of 3 ResBlock1 (kernels [3,7,11], dil [1,3,5])
//   → LeakyReLU(0.1) → conv_out(32→2,k7) → tanh. Output 24 kHz stereo.
// Standard HiFi-GAN — no SnakeBeta, no anti-aliasing, no BWE. weight_norm is
// pre-fused in the checkpoint (keys are `.weight`, not `.weight_g/_v`).

import Foundation
import Logging
import MLX
import MLXNN

private func leakyReLU(_ x: MLXArray, _ slope: Float = 0.1) -> MLXArray {
  MLX.maximum(x, x * slope)
}

// MARK: - HiFi-GAN ResBlock1 (`resnets.N`)

/// Three (LeakyReLU → dilated conv → LeakyReLU → conv) residual sub-blocks with
/// dilations {1,3,5}. `convs1` are dilated; `convs2` have dilation 1.
public final class LTX2HiFiResBlock: Module {
  @ModuleInfo(key: "convs1") var convs1: [Conv1d]
  @ModuleInfo(key: "convs2") var convs2: [Conv1d]
  let slope: Float

  public init(channels: Int, kernelSize: Int, dilations: [Int] = [1, 3, 5], slope: Float = 0.1) {
    self.slope = slope
    self._convs1.wrappedValue = dilations.map { d in
      Conv1d(inputChannels: channels, outputChannels: channels,
             kernelSize: kernelSize, padding: d * (kernelSize - 1) / 2, dilation: d)
    }
    self._convs2.wrappedValue = dilations.map { _ in
      Conv1d(inputChannels: channels, outputChannels: channels,
             kernelSize: kernelSize, padding: (kernelSize - 1) / 2, dilation: 1)
    }
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = x
    for i in 0..<convs1.count {
      var xt = leakyReLU(out, slope)
      xt = convs1[i](xt)
      xt = leakyReLU(xt, slope)
      xt = convs2[i](xt)
      out = out + xt
    }
    return out
  }
}

// MARK: - HiFi-GAN generator (the official LTX2Vocoder)

public final class LTX2HiFiGANVocoder: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv1d
  @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
  @ModuleInfo(key: "resnets") var resnets: [LTX2HiFiResBlock]
  @ModuleInfo(key: "conv_out") var convOut: Conv1d

  let numKernels: Int
  let slope: Float

  // Official config.
  static let melChannels = 128
  static let hiddenChannels = 1024
  static let upsampleRates = [6, 5, 2, 2, 2]
  static let upsampleKernels = [16, 15, 8, 4, 4]
  static let resblockKernels = [3, 7, 11]
  static let resblockDilations = [1, 3, 5]
  /// ∏ upsampleRates — mel frames → samples.
  public static let hopUpsample = 6 * 5 * 2 * 2 * 2   // 120

  public override init() {
    self.slope = 0.1
    self.numKernels = LTX2HiFiGANVocoder.resblockKernels.count

    self._convIn.wrappedValue = Conv1d(
      inputChannels: LTX2HiFiGANVocoder.melChannels,
      outputChannels: LTX2HiFiGANVocoder.hiddenChannels, kernelSize: 7, padding: 3)

    var ch = LTX2HiFiGANVocoder.hiddenChannels
    var upList: [ConvTransposed1d] = []
    var blocks: [LTX2HiFiResBlock] = []
    for (i, rate) in LTX2HiFiGANVocoder.upsampleRates.enumerated() {
      let k = LTX2HiFiGANVocoder.upsampleKernels[i]
      let outCh = ch / 2
      upList.append(ConvTransposed1d(
        inputChannels: ch, outputChannels: outCh,
        kernelSize: k, stride: rate, padding: (k - rate) / 2))
      for rk in LTX2HiFiGANVocoder.resblockKernels {
        blocks.append(LTX2HiFiResBlock(
          channels: outCh, kernelSize: rk, dilations: LTX2HiFiGANVocoder.resblockDilations, slope: slope))
      }
      ch = outCh
    }
    self._ups.wrappedValue = upList
    self._resnets.wrappedValue = blocks
    self._convOut.wrappedValue = Conv1d(
      inputChannels: ch, outputChannels: 2, kernelSize: 7, padding: 3)
    super.init()
  }

  /// mel NLC `[B, T, 128]` → stereo NLC `[B, T·120, 2]` in [-1, 1].
  public func callAsFunction(_ mel: MLXArray) -> MLXArray {
    var x = convIn(mel)
    for i in 0..<ups.count {
      x = leakyReLU(x, slope)
      x = ups[i](x)
      var xs = resnets[i * numKernels](x)
      for j in 1..<numKernels {
        xs = xs + resnets[i * numKernels + j](x)
      }
      x = xs / MLXArray(Float(numKernels))
    }
    x = leakyReLU(x, slope)
    x = convOut(x)
    return MLX.tanh(x)
  }

  /// mel channels-first `[B, 128, T]` → stereo `[B, 2, T·120]` @24 kHz.
  public func synthesize(_ mel: MLXArray) -> MLXArray {
    let nlc = mel.transposed(0, 2, 1)          // [B, T, 128]
    let wave = callAsFunction(nlc)             // [B, T·120, 2]
    return wave.transposed(0, 2, 1)            // [B, 2, T·120]
  }
}

// MARK: - Weight loading (standalone official vocoder .safetensors)

extension LTX2HiFiGANVocoder {
  /// Load from the official `vocoder/diffusion_pytorch_model.safetensors`.
  /// Keys are top-level (`conv_in.*`, `ups.N.*`, `resnets.N.convs{1,2}.M.*`,
  /// `conv_out.*`). Conv weights transpose OIK→OKI; ConvTranspose (`ups.*`)
  /// IOK→OKI. weight_norm is pre-fused (`.weight`, not `.weight_g/_v`).
  public static func load(path: String, logger: Logger = Logger(label: "ltx2.vocoder-official")) throws -> LTX2HiFiGANVocoder {
    let tensors = try MLX.loadArrays(url: URL(fileURLWithPath: path)).mapValues { $0.asType(.float32) }
    let voc = LTX2HiFiGANVocoder()
    let remapped = remapKeys(tensors)
    let moduleKeys = Set(voc.parameters().flattened().map { $0.0 })
    let matched = remapped.filter { moduleKeys.contains($0.0) }.count
    logger.info("LTX-2 OFFICIAL vocoder: matched \(matched)/\(moduleKeys.count) module params (\(remapped.count) remapped).")
    if matched * 2 < moduleKeys.count {
      throw LTX2HiFiGANError.weightCoverageTooLow(matched: matched, total: moduleKeys.count)
    }
    let params = ModuleParameters.unflattened(remapped.map { ($0.0, $0.1) })
    try voc.update(parameters: params, verify: [.shapeMismatch])
    eval(voc.parameters())
    return voc
  }

  static func remapKeys(_ tensors: [String: MLXArray]) -> [(String, MLXArray)] {
    var out: [(String, MLXArray)] = []
    for (key, value) in tensors {
      if key == "__metadata__" { continue }
      var v = value
      if key.hasSuffix(".weight") && v.ndim == 3 {
        // Only the `ups.N` layers are ConvTranspose1d (IOK); everything else
        // (conv_in, conv_out, resnets.*.convs*) is a forward Conv1d (OIK).
        let isUp = key.hasPrefix("ups.")
        if isUp {
          v = v.transposed(1, 2, 0)   // ConvTranspose1d IOK → OKI
        } else {
          v = v.transposed(0, 2, 1)   // Conv1d OIK → OKI
        }
      }
      out.append((key, v))
    }
    return out
  }
}

public enum LTX2HiFiGANError: Error, CustomStringConvertible {
  case weightCoverageTooLow(matched: Int, total: Int)
  public var description: String {
    switch self {
    case .weightCoverageTooLow(let m, let t):
      return "official vocoder matched only \(m)/\(t) module params — key/shape mismatch"
    }
  }
}
