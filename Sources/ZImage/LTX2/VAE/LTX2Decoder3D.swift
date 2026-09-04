import Foundation
import MLX
import MLXNN
import MLXRandom

/// 3D decoder for the LTX-2 Video VAE with timestep conditioning.
///
/// Decodes 128-channel latents back to RGB video. Uses DepthToSpace upsampling
/// with residual connections and optional timestep conditioning via PixArtAlpha-
/// style embeddings.
///
/// ## Architecture (Default)
///
/// ```
/// Input (B, 128, F', H', W')
///   ├─ [optional] add noise (decode_noise_scale=0.025)
///   ├─ per_channel_statistics.un_normalize
///   ├─ conv_in.conv: CausalConv3d(128 → 1024, k=3)
///   ├─ up_blocks.0: res_x (5 layers, 1024ch, timestep-conditioned)
///   ├─ up_blocks.1: depth_to_space(1024 → 512, stride=2,2,2)
///   ├─ up_blocks.2: res_x (5 layers, 512ch, timestep-conditioned)
///   ├─ up_blocks.3: depth_to_space(512 → 256, stride=2,2,2)
///   ├─ up_blocks.4: res_x (5 layers, 256ch, timestep-conditioned)
///   ├─ up_blocks.5: depth_to_space(256 → 128, stride=2,2,2)
///   ├─ up_blocks.6: res_x (5 layers, 128ch, timestep-conditioned)
///   ├─ PixelNorm → timestep modulation (last_scale_shift_table)
///   ├─ SiLU
///   ├─ conv_out.conv: CausalConv3d(128 → 48, k=3)
///   ├─ unpatchify(48 → 3, patch_size=4)
///   └─ Output (B, 3, F, H, W)
/// ```
///
/// The decoder block structure is inferred from weights during loading, with
/// the default being 4 res_x blocks interleaved with 3 depth_to_space upsamplers.
public final class LTX2Decoder3D: Module {

  /// Initial convolution wrapper. Has a nested `conv` to match PyTorch weight naming.
  @ModuleInfo(key: "conv_in") var convIn: LTX2ConvWrapper

  /// Decoder blocks: res_x groups and depth-to-space upsamplers.
  @ModuleInfo(key: "up_blocks") var upBlocks: [String: Module]

  /// Output convolution wrapper. Has a nested `conv` to match PyTorch weight naming.
  @ModuleInfo(key: "conv_out") var convOut: LTX2ConvWrapper

  /// Per-channel statistics for denormalization.
  @ModuleInfo(key: "per_channel_statistics") var perChannelStatistics: LTX2PerChannelStatistics

  /// Timestep scale multiplier for conditioning.
  public var timestepScaleMultiplier: MLXArray

  /// Last timestep embedder for final modulation.
  @ModuleInfo(key: "last_time_embedder") var lastTimeEmbedder: LTX2PixArtTimestepEmbedder

  /// Last scale/shift table for final timestep modulation.
  public var lastScaleShiftTable: MLXArray

  /// Spatial patch size.
  public let patchSize: Int

  /// Number of input latent channels.
  public let inChannels: Int

  /// Whether timestep conditioning is enabled.
  public let timestepConditioning: Bool

  /// Noise scale for conditioning.
  public let decodeNoiseScale: Float

  /// Default timestep value.
  public let decodeTimestep: Float

  /// Last block channel count (needed for final modulation).
  public let lastChannels: Int

  /// Whether convolutions use causal temporal padding.
  public let causalDecoder: Bool

  /// Creates an LTX-2 3D decoder with the default architecture.
  ///
  /// - Parameter config: Video VAE configuration. Defaults to `.default`.
  public init(config: LTX2VideoVAEConfig = .default) {
    self.patchSize = config.patchSize
    self.inChannels = config.latentChannels
    self.timestepConditioning = config.timestepConditioning
    self.decodeNoiseScale = config.decodeNoiseScale
    self.decodeTimestep = config.decodeTimestep

    // Compute channel progression from decoder block definitions
    // The decoder reverses the encoder's compression, starting from the
    // highest channel count and working down.
    var featureChannels = config.latentChannels
    // Walk the decoder blocks in reverse to compute initial (highest) channels
    for blockDef in config.decoderBlocks.reversed() {
      switch blockDef {
      case .compressAll(let multiplier, _):
        featureChannels = featureChannels * multiplier
      case .compressSpace(let multiplier):
        featureChannels = featureChannels * multiplier
      case .compressTime(let multiplier):
        featureChannels = featureChannels * multiplier
      default:
        break
      }
    }

    let firstChannels = featureChannels
    let isCausal = config.causalDecoder

    // Build decoder blocks (reversed order matches Python decoder logic)
    var blocks: [String: Module] = [:]
    var idx = 0
    for blockDef in config.decoderBlocks.reversed() {
      switch blockDef {
      case .resX(let numLayers):
        blocks[String(idx)] = LTX2DecoderResBlockGroup(
          channels: featureChannels,
          numLayers: numLayers,
          timestepConditioning: config.timestepConditioning,
          causalTemporal: isCausal
        )

      case .compressSpace(let multiplier):
        let outChannels = featureChannels / multiplier
        blocks[String(idx)] = LTX2DepthToSpaceUpsample(
          inChannels: featureChannels,
          stride: (1, 2, 2),
          residual: false,
          outChannelsReductionFactor: multiplier,
          causalTemporal: isCausal
        )
        featureChannels = outChannels

      case .compressTime(let multiplier):
        let outChannels = featureChannels / multiplier
        blocks[String(idx)] = LTX2DepthToSpaceUpsample(
          inChannels: featureChannels,
          stride: (2, 1, 1),
          residual: false,
          outChannelsReductionFactor: multiplier,
          causalTemporal: isCausal
        )
        featureChannels = outChannels

      case .compressAll(let multiplier, let residual):
        let outChannels = featureChannels / multiplier
        blocks[String(idx)] = LTX2DepthToSpaceUpsample(
          inChannels: featureChannels,
          stride: (2, 2, 2),
          residual: residual,
          outChannelsReductionFactor: multiplier,
          causalTemporal: isCausal
        )
        featureChannels = outChannels
      }
      idx += 1
    }
    self._upBlocks.wrappedValue = blocks
    self.lastChannels = featureChannels
    self.causalDecoder = config.causalDecoder

    // Conv in: latent_channels → first_channels
    self._convIn.wrappedValue = LTX2ConvWrapper(
      inChannels: config.latentChannels, outChannels: firstChannels,
      causalTemporal: isCausal
    )

    // Conv out: last_channels → output_channels * patch_size^2
    let finalOutChannels = config.inChannels * config.patchSize * config.patchSize
    self._convOut.wrappedValue = LTX2ConvWrapper(
      inChannels: featureChannels, outChannels: finalOutChannels,
      causalTemporal: isCausal
    )

    // Per-channel statistics
    self._perChannelStatistics.wrappedValue = LTX2PerChannelStatistics(
      latentChannels: config.latentChannels
    )

    // Timestep conditioning
    self.timestepScaleMultiplier = MLXArray(config.timestepScaleMultiplier)
    self._lastTimeEmbedder.wrappedValue = LTX2PixArtTimestepEmbedder(
      embeddingDim: featureChannels * 2
    )
    self.lastScaleShiftTable = MLXArray.zeros([2, featureChannels])

    super.init()
  }

  /// Decodes latent representation to video.
  ///
  /// - Parameters:
  ///   - sample: Latent tensor of shape `(B, 128, F', H', W')`.
  ///   - timestep: Optional timestep for conditioning.
  /// - Returns: Decoded video tensor of shape `(B, 3, F, H, W)`.
  public func callAsFunction(_ sample: MLXArray, timestep: MLXArray? = nil) -> MLXArray {
    let batchSize = sample.dim(0)
    var x = sample

    // Add noise if timestep conditioning is enabled
    if timestepConditioning {
      let noise = MLXRandom.normal(x.shape) * decodeNoiseScale
      x = noise + (1.0 - decodeNoiseScale) * x
    }

    // Denormalize latents
    x = perChannelStatistics.unNormalize(x)

    // Get timestep
    let ts: MLXArray?
    if timestepConditioning {
      let t = timestep ?? MLX.full([batchSize], values: MLXArray(decodeTimestep), dtype: .float32)
      ts = t * timestepScaleMultiplier
    } else {
      ts = nil
    }

    // Initial convolution
    x = convIn(x)

    // Process through decoder blocks
    let sortedKeys = upBlocks.keys.sorted { Int($0)! < Int($1)! }
    for key in sortedKeys {
      let block = upBlocks[key]!
      if let resGroup = block as? LTX2DecoderResBlockGroup {
        x = resGroup(x, timestep: ts)
      } else if let upsample = block as? LTX2DepthToSpaceUpsample {
        x = upsample(x)
      }
    }

    // Final PixelNorm
    x = pixelNorm(x)

    // Timestep modulation at the end
    if timestepConditioning, let scaledTimestep = ts {
      let embeddedTimestep = lastTimeEmbedder(
        scaledTimestep.flattened(),
        hiddenDtype: x.dtype
      )
      let embedded = embeddedTimestep.reshaped(batchSize, -1, 1, 1, 1)

      // ada_values = table[None, :, :, None, None, None] + ts_reshaped
      let adaBase = lastScaleShiftTable.reshaped(1, 2, lastChannels, 1, 1, 1)
      let tsReshaped = embedded.reshaped(batchSize, 2, lastChannels, 1, 1, 1)
      let adaValues = adaBase + tsReshaped

      let shift = adaValues[0..., 0, 0..., 0..., 0..., 0...]
      let scale = adaValues[0..., 1, 0..., 0..., 0..., 0...]

      x = x * (1 + scale) + shift
    }

    // SiLU + conv_out
    x = silu(x)
    x = convOut(x)

    // Unpatchify: (B, 48, F', H', W') -> (B, 3, F, H*4, W*4)
    x = LTX2Patchify.unpatchify(x, patchSizeHW: patchSize, patchSizeT: 1)

    return x
  }

  /// PixelNorm: L2 normalize over channel dimension.
  private func pixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
    x / MLX.sqrt(MLX.mean(x * x, axis: 1, keepDims: true) + eps)
  }

  // MARK: - Streamed decode (#36)

  /// Exact chunked-io decode — faithful port of ComfyUI's recursive `run_up`
  /// (causal_video_autoencoder.py).
  ///
  /// Each up-block processes its input at the CURRENT level; if the block's
  /// output is too large for one downstream call it is split along the frame
  /// axis and each piece recurses into the remaining blocks. Per-conv temporal
  /// caches make chunk boundaries exact, DepthToSpace applies its causal trim
  /// once per stream, res-block skips consume frames at the conv path's pace,
  /// and `ended` propagates only along the LAST piece of every split so each
  /// conv appends its trailing padding exactly once. Numerically identical to
  /// a single full-tensor `callAsFunction` (parity + real-data tests).
  ///
  /// Chunk-at-input topologies DON'T work here: v2.3's convs are non-causal,
  /// so every conv lags its input and a ~40-conv stack starves until a final
  /// flush pushes all frames through the last convs in one call — recreating
  /// the int32-offset conv corruption this decode exists to avoid (MLX Metal,
  /// ml-explore/mlx #3836/#3609/#3524; fixed only in mlx core >= 0.32.0,
  /// which no mlx-swift release bundles yet).
  ///
  /// - Parameters:
  ///   - sample: Latent tensor `(B, 128, F', H', W')`.
  ///   - timestep: Optional timestep for conditioning.
  ///   - maxRowsPerConv: Ceiling on frames x spatial positions per conv call
  ///     at any level (int32 offset budget: rows x 3456 must stay < 2^31).
  /// - Returns: Decoded video tensor `(B, 3, F, H, W)`.
  ///
  /// comfybox#322: `throws`. This is the production decode path for every clip
  /// above the plain-decode volume gate — minutes of work on a long render —
  /// so it checks for cancellation as each output volume is emitted and before
  /// each piece is pushed down the block stack.
  public func decodeStreamed(
    _ sample: MLXArray,
    timestep: MLXArray? = nil,
    maxRowsPerConv: Int = 350_000
  ) throws -> MLXArray {
    // Collect every stateful module, keyed per block for ended-marking.
    var allConvs: [CausalConv3d] = []
    var allUps: [LTX2DepthToSpaceUpsample] = []
    var allRes: [LTX2DecoderResBlock] = []
    for m in self.modules() {
      if let c = m as? CausalConv3d { allConvs.append(c) }
      if let u = m as? LTX2DepthToSpaceUpsample { allUps.append(u) }
      if let r = m as? LTX2DecoderResBlock { allRes.append(r) }
    }
    for c in allConvs { c.resetStream(active: true) }
    for u in allUps { u.resetStream(active: true) }
    for r in allRes { r.resetStream(active: true) }
    defer {
      for c in allConvs { c.resetStream(active: false) }
      for u in allUps { u.resetStream(active: false) }
      for r in allRes { r.resetStream(active: false) }
    }

    func convsOf(_ module: Module) -> [CausalConv3d] {
      module.modules().compactMap { $0 as? CausalConv3d }
    }

    let batchSize = sample.dim(0)
    let ts: MLXArray?
    if timestepConditioning {
      let t = timestep ?? MLX.full([batchSize], values: MLXArray(decodeTimestep), dtype: .float32)
      ts = t * timestepScaleMultiplier
    } else {
      ts = nil
    }

    let sortedKeys = upBlocks.keys.sorted { Int($0)! < Int($1)! }
    var outputs: [MLXArray] = []

    // Split `x` along frames so each piece keeps frames x H x W under the
    // conv row budget at THIS level.
    func splitByBudget(_ x: MLXArray) -> [MLXArray] {
      let f = x.dim(2)
      guard f > 0 else { return [] }
      let rowsPerFrame = max(x.dim(3) * x.dim(4), 1)
      // Budget the CONV-call extent, not the piece extent: each downstream
      // conv adds up to 2 cached + 1 end-pad frames to the piece, and the
      // empirical Metal corruption onset at 128ch/61440-spatial sits between
      // 9-frame bodies (clean) and ~15-frame bodies (corrupt). Keep piece
      // frames + 3 within budget.
      let maxFrames = max(1, maxRowsPerConv / rowsPerFrame - 3)
      guard f > maxFrames else { return [x] }
      var pieces: [MLXArray] = []
      var s = 0
      while s < f {
        let e = min(s + maxFrames, f)
        pieces.append(x[0..., 0..., s..<e, 0..., 0...])
        s = e
      }
      return pieces
    }

    func runUp(_ idx: Int, _ input: MLXArray, _ ended: Bool) throws {
      // comfybox#322: one check per piece at every level of the block stack.
      // The recursion's leaves are the emitted output volumes, so a cancel is
      // observed within one volume rather than at the end of the decode.
      try Task.checkCancellation()
      if idx >= sortedKeys.count {
        // Final head: PixelNorm -> (mod) -> SiLU -> conv_out -> unpatchify.
        var x = pixelNorm(input)
        if timestepConditioning, let scaledTimestep = ts {
          let embeddedTimestep = lastTimeEmbedder(
            scaledTimestep.flattened(), hiddenDtype: x.dtype)
          let embedded = embeddedTimestep.reshaped(batchSize, -1, 1, 1, 1)
          let adaBase = lastScaleShiftTable.reshaped(1, 2, lastChannels, 1, 1, 1)
          let tsReshaped = embedded.reshaped(batchSize, 2, lastChannels, 1, 1, 1)
          let adaValues = adaBase + tsReshaped
          let shift = adaValues[0..., 0, 0..., 0..., 0..., 0...]
          let scale = adaValues[0..., 1, 0..., 0..., 0..., 0...]
          x = x * (1 + scale) + shift
        }
        x = silu(x)
        for c in convsOf(convOut) { c.streamEnded = ended }
        x = convOut(x)
        if x.dim(2) > 0 {
          x = LTX2Patchify.unpatchify(x, patchSizeHW: patchSize, patchSizeT: 1)
          eval(x)
          outputs.append(x)
        }
        MLX.GPU.clearCache()
        return
      }

      let block = upBlocks[sortedKeys[idx]]!
      for c in convsOf(block) { c.streamEnded = ended }
      var out: MLXArray
      if let resGroup = block as? LTX2DecoderResBlockGroup {
        out = resGroup(input, timestep: ts)
      } else if let upsample = block as? LTX2DepthToSpaceUpsample {
        out = upsample(input)
      } else {
        out = input
      }
      guard out.dim(2) > 0 else { return }
      eval(out)

      let pieces = splitByBudget(out)
      for (i, piece) in pieces.enumerated() {
        try runUp(idx + 1, piece, ended && i == pieces.count - 1)
      }
    }

    // Entry: noise + denorm + conv_in on the full latent (small at this
    // level), then recurse with ended = true (splits scope it downward).
    var x = sample
    if timestepConditioning {
      let noise = MLXRandom.normal(x.shape) * decodeNoiseScale
      x = noise + (1.0 - decodeNoiseScale) * x
    }
    x = perChannelStatistics.unNormalize(x)
    for c in convsOf(convIn) { c.streamEnded = true }
    x = convIn(x)
    eval(x)
    let entryPieces = splitByBudget(x)
    for (i, piece) in entryPieces.enumerated() {
      try runUp(0, piece, i == entryPieces.count - 1)
    }

    precondition(!outputs.isEmpty, "streamed decode emitted no frames")
    if outputs.count == 1 { return outputs[0] }
    // Materialize via incremental slice-assignment, NOT a single concatenate:
    // one-shot materialization of the ~1G-element output walks into the same
    // int32-offset kernel bug (corrupt-frame pattern identical to plain
    // decode). The tiled path's slice-assign accumulation is proven safe.
    let totalF = outputs.reduce(0) { $0 + $1.dim(2) }
    let proto = outputs[0]
    var full = MLXArray.zeros(
      [proto.dim(0), proto.dim(1), totalF, proto.dim(3), proto.dim(4)],
      dtype: proto.dtype)
    var offset = 0
    for chunk in outputs {
      // comfybox#322: the materialization walk is itself a per-volume loop.
      try Task.checkCancellation()
      let n = chunk.dim(2)
      full[0..., 0..., offset..<(offset + n), 0..., 0...] = chunk
      eval(full)
      offset += n
    }
    return full
  }
}

// MARK: - Decoder-specific helper modules

/// Wrapper for conv_in and conv_out to match PyTorch weight key naming.
///
/// PyTorch has `conv_in.conv.weight` and `conv_out.conv.weight`, so we need
/// a wrapper module with a `conv` child.
public final class LTX2ConvWrapper: Module {

  @ModuleInfo(key: "conv") var conv: CausalConv3d

  public init(inChannels: Int, outChannels: Int, causalTemporal: Bool = true) {
    self._conv.wrappedValue = CausalConv3d(
      inChannels: inChannels, outChannels: outChannels,
      kernelSize: (3, 3, 3), stride: (1, 1, 1), padding: (1, 1, 1),
      causalTemporal: causalTemporal
    )
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    conv(x)
  }
}

/// Sinusoidal timestep embedding matching PixArtAlpha.
///
/// Produces sinusoidal embeddings with flip-sin-to-cos ordering (cos then sin),
/// matching the PyTorch `get_timestep_embedding` implementation.
public func ltx2TimestepEmbedding(
  _ timesteps: MLXArray,
  embeddingDim: Int,
  flipSinToCos: Bool = true,
  downscaleFreqShift: Float = 0,
  scale: Float = 1,
  maxPeriod: Int = 10000
) -> MLXArray {
  let halfDim = embeddingDim / 2
  let indices = MLXArray(Int32(0) ..< Int32(halfDim)).asType(.float32)
  let exponent = -Foundation.log(Float(maxPeriod))
    * indices
    / (Float(halfDim) - downscaleFreqShift)

  var emb = exponent.exp()
  emb = timesteps.expandedDimensions(axis: 1).asType(.float32) * emb.expandedDimensions(axis: 0)
  emb = scale * emb

  emb = MLX.concatenated([emb.sin(), emb.cos()], axis: -1)

  if flipSinToCos {
    let left = emb[0..., halfDim...]
    let right = emb[0..., ..<halfDim]
    emb = MLX.concatenated([left, right], axis: -1)
  }

  return emb
}

/// MLP for timestep embedding.
public final class LTX2TimestepMLP: Module {

  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear

  public init(inChannels: Int, timeEmbedDim: Int) {
    self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
    self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim)
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = linear1(x)
    h = silu(h)
    h = linear2(h)
    return h
  }
}

/// Combined PixArtAlpha timestep embedder (sinusoidal + MLP).
public final class LTX2PixArtTimestepEmbedder: Module {

  @ModuleInfo(key: "timestep_embedder") var timestepEmbedder: LTX2TimestepMLP

  public init(embeddingDim: Int) {
    self._timestepEmbedder.wrappedValue = LTX2TimestepMLP(
      inChannels: 256, timeEmbedDim: embeddingDim
    )
    super.init()
  }

  public func callAsFunction(_ timestep: MLXArray, hiddenDtype: DType = .float32) -> MLXArray {
    let proj = ltx2TimestepEmbedding(
      timestep, embeddingDim: 256,
      flipSinToCos: true, downscaleFreqShift: 0
    )
    return timestepEmbedder(proj.asType(hiddenDtype))
  }
}

/// Decoder ResNet block with optional timestep conditioning.
///
/// Uses PixelNorm (not GroupNorm) and scale/shift modulation from timestep
/// embeddings. Each block has a `scale_shift_table` parameter for the adaptive
/// modulation.
///
/// Weight key: `conv1.conv`, `conv2.conv`, `scale_shift_table`
public final class LTX2DecoderResBlock: Module {

  /// Nested conv wrapper for conv1 (matches PyTorch `conv1.conv.weight`).
  @ModuleInfo(key: "conv1") var conv1: LTX2ConvWrapper

  /// Nested conv wrapper for conv2 (matches PyTorch `conv2.conv.weight`).
  @ModuleInfo(key: "conv2") var conv2: LTX2ConvWrapper

  /// Scale-shift table for timestep conditioning: `[shift1, scale1, shift2, scale2]`.
  public var scaleShiftTable: MLXArray

  /// Whether timestep conditioning is enabled.
  public let timestepConditioning: Bool

  /// Number of channels.
  public let channels: Int

  // MARK: Streaming decode state (#36)
  /// Deferred skip-connection frames. Under streaming the conv path lags the
  /// raw input (each conv buffers temporal context), so the residual add must
  /// consume skip frames at the conv path's emission pace and carry the
  /// overhang to the next chunk (ComfyUI's add_exchange_cache).
  public var streamActive = false
  public var streamSkipQueue: MLXArray? = nil

  public func resetStream(active: Bool) {
    streamActive = active
    streamSkipQueue = nil
  }

  public init(channels: Int, timestepConditioning: Bool = false, causalTemporal: Bool = true) {
    self.channels = channels
    self.timestepConditioning = timestepConditioning

    self._conv1.wrappedValue = LTX2ConvWrapper(inChannels: channels, outChannels: channels, causalTemporal: causalTemporal)
    self._conv2.wrappedValue = LTX2ConvWrapper(inChannels: channels, outChannels: channels, causalTemporal: causalTemporal)

    self.scaleShiftTable = timestepConditioning
      ? MLXArray.zeros([4, channels])
      : MLXArray.zeros([0])

    super.init()
  }

  public func callAsFunction(
    _ x: MLXArray,
    timestepEmbed: MLXArray? = nil
  ) -> MLXArray {
    let residual = x
    let batchSize = x.dim(0)

    var h = pixelNorm(x)

    // First block with optional timestep conditioning
    if timestepConditioning, let tsEmbed = timestepEmbed {
      // Combine table with timestep embedding
      let adaBase = scaleShiftTable.reshaped(1, 4, channels, 1, 1, 1)
      let tsReshaped = tsEmbed.reshaped(batchSize, 4, channels, 1, 1, 1)
      let adaValues = adaBase + tsReshaped

      let shift1 = adaValues[0..., 0, 0..., 0..., 0..., 0...]
      let scale1 = adaValues[0..., 1, 0..., 0..., 0..., 0...]
      let shift2 = adaValues[0..., 2, 0..., 0..., 0..., 0...]
      let scale2 = adaValues[0..., 3, 0..., 0..., 0..., 0...]

      h = h * (1 + scale1) + shift1
      h = silu(h)
      h = conv1(h)

      h = pixelNorm(h)
      h = h * (1 + scale2) + shift2
      h = silu(h)
      h = conv2(h)
    } else {
      h = silu(h)
      h = conv1(h)

      h = pixelNorm(h)
      h = silu(h)
      h = conv2(h)
    }

    if streamActive {
      var queue = residual
      if let q = streamSkipQueue {
        queue = MLX.concatenated([q, residual], axis: 2)
      }
      let n = h.dim(2)
      let out = n > 0 ? h + queue[0..., 0..., ..<n, 0..., 0...] : h
      streamSkipQueue = queue[0..., 0..., n..., 0..., 0...]
      return out
    }
    return h + residual
  }

  private func pixelNorm(_ x: MLXArray, eps: Float = 1e-8) -> MLXArray {
    x / MLX.sqrt(MLX.mean(x * x, axis: 1, keepDims: true) + eps)
  }
}

/// Group of decoder ResNet blocks with shared timestep embedding.
///
/// Weight key: `res_blocks.{i}`, `time_embedder`.
public final class LTX2DecoderResBlockGroup: Module {

  /// Timestep embedder for this block group.
  @ModuleInfo(key: "time_embedder") var timeEmbedder: LTX2PixArtTimestepEmbedder?

  /// Residual blocks.
  @ModuleInfo(key: "res_blocks") var resBlocks: [String: LTX2DecoderResBlock]

  /// Whether timestep conditioning is active.
  public let timestepConditioning: Bool

  public init(
    channels: Int,
    numLayers: Int,
    timestepConditioning: Bool,
    causalTemporal: Bool = true
  ) {
    self.timestepConditioning = timestepConditioning

    if timestepConditioning {
      self._timeEmbedder.wrappedValue = LTX2PixArtTimestepEmbedder(
        embeddingDim: channels * 4
      )
    }

    var blocks: [String: LTX2DecoderResBlock] = [:]
    for i in 0..<numLayers {
      blocks[String(i)] = LTX2DecoderResBlock(
        channels: channels,
        timestepConditioning: timestepConditioning,
        causalTemporal: causalTemporal
      )
    }
    self._resBlocks.wrappedValue = blocks

    super.init()
  }

  public func callAsFunction(_ x: MLXArray, timestep: MLXArray? = nil) -> MLXArray {
    var timestepEmbed: MLXArray? = nil
    if timestepConditioning, let ts = timestep {
      let batchSize = x.dim(0)
      let emb = timeEmbedder!(ts.reshaped(-1), hiddenDtype: x.dtype)
      timestepEmbed = emb.reshaped(batchSize, -1, 1, 1, 1)
    }

    var hidden = x
    let sortedKeys = resBlocks.keys.sorted { Int($0)! < Int($1)! }
    for key in sortedKeys {
      hidden = resBlocks[key]!(hidden, timestepEmbed: timestepEmbed)
    }
    return hidden
  }
}
