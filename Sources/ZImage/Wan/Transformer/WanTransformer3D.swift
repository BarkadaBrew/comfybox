import Foundation
import MLX
import MLXNN

/// Wan 2.2 I2V-A14B video diffusion transformer.
///
/// Full model wiring: embeddings -> 40 transformer blocks -> head -> unpatchify.
///
/// ## Forward Flow
/// 1. Concatenate noise + conditioning along channel dim (I2V)
/// 2. Patch embed via Conv3d → flatten → transpose to `[B, seqLen, dim]`
/// 3. Pad sequences to max_seq_len
/// 4. Time embedding: sinusoidal → MLP → projection to 6*dim (per-position)
/// 5. Text embedding: MLP maps `[B, textLen, textDim]` → `[B, textLen, dim]`
/// 6. Run through 40 transformer blocks
/// 7. Head: modulated output projection
/// 8. Unpatchify: reshape patches back to `[C, F, H, W]`
///
/// ## Weight Structure
/// - `patch_embedding.weight/bias` — Conv3d
/// - `text_embedding.{0,2}.weight/bias` — MLP with GELU
/// - `time_embedding.{0,2}.weight/bias` — MLP with SiLU
/// - `time_projection.1.weight/bias` — Linear (index 0 = SiLU, no params)
/// - `blocks.{0..39}.*` — 40 transformer blocks
/// - `head.*` — output head
public final class WanTransformer3D: Module {

  // MARK: - Embeddings

  @ModuleInfo(key: "patch_embedding") var patchEmbedding: WanPatchEmbedding
  @ModuleInfo(key: "text_embedding") var textEmbedding: WanTextEmbeddingMLP
  @ModuleInfo(key: "time_embedding") var timeEmbedding: WanTimeEmbeddingMLP
  @ModuleInfo(key: "time_projection") var timeProjection: WanTimeProjection

  // MARK: - Blocks

  @ModuleInfo(key: "blocks") var blocks: [WanTransformerBlock]

  // MARK: - Head

  @ModuleInfo(key: "head") var headModule: WanHead

  // MARK: - Config & Buffers

  public let config: WanTransformerConfig

  /// Precomputed RoPE frequencies (buffer, not parameter).
  public let freqs: MLXArray

  // MARK: - Init

  public init(config: WanTransformerConfig = .i2vA14B) {
    self.config = config

    self._patchEmbedding.wrappedValue = WanPatchEmbedding(
      inDim: config.inDim, dim: config.dim, patchSize: config.patchSize
    )

    self._textEmbedding.wrappedValue = WanTextEmbeddingMLP(
      textDim: config.textDim, dim: config.dim
    )

    self._timeEmbedding.wrappedValue = WanTimeEmbeddingMLP(
      freqDim: config.freqDim, dim: config.dim
    )

    self._timeProjection.wrappedValue = WanTimeProjection(dim: config.dim)

    self._blocks.wrappedValue = (0..<config.numLayers).map { _ in
      WanTransformerBlock(
        dim: config.dim,
        ffnDim: config.ffnDim,
        numHeads: config.numHeads,
        qkNorm: config.qkNorm,
        crossAttnNorm: config.crossAttnNorm,
        eps: config.eps
      )
    }

    self._headModule.wrappedValue = WanHead(
      dim: config.dim, outDim: config.outDim,
      patchSize: config.patchSize, eps: config.eps
    )

    // Build RoPE frequencies (buffer)
    self.freqs = WanRoPE.buildFrequencies(
      maxSeqLen: 1024, headDim: config.headDim
    )

    super.init()
  }

  // MARK: - Forward

  /// Forward pass through the diffusion model.
  ///
  /// - Parameters:
  ///   - x: List of noise tensors, each `[C_in, F, H, W]`.
  ///   - t: Timestep tensor `[B]`.
  ///   - context: List of text embeddings, each `[textLen, textDim]`.
  ///   - seqLen: Maximum sequence length for positional encoding.
  ///   - y: Conditioning tensors for I2V, each `[C_cond, F, H, W]`, or nil.
  /// - Returns: List of denoised tensors, each `[outDim, F, H, W]`.
  public func forward(
    x: [MLXArray],
    t: MLXArray,
    context: [MLXArray],
    seqLen: Int,
    y: [MLXArray]?
  ) -> [MLXArray] {
    let b = x.count

    // For I2V, concatenate noise and conditioning along channel dim
    var inputs = x
    if let y = y {
      inputs = zip(x, y).map { noise, cond in
        MLX.concatenated([noise, cond], axis: 0)
      }
    }

    // Patch embedding: each [C, F, H, W] -> [1, seqLen_i, dim]
    var embedded: [MLXArray] = []
    var gridSizesList: [[Int]] = []
    for inp in inputs {
      let unsqueezed = inp.expandedDimensions(axis: 0)  // [1, C, F, H, W]
      let patched = patchEmbedding(unsqueezed)
      // patched is [1, dim, F_p, H_p, W_p]
      let fp = patched.dim(2)
      let hp = patched.dim(3)
      let wp = patched.dim(4)
      gridSizesList.append([fp, hp, wp])

      // Flatten spatial dims and transpose: [1, seqLen_i, dim]
      let flat = patched.reshaped(1, config.dim, fp * hp * wp).transposed(0, 2, 1)
      embedded.append(flat)
    }

    // Compute sequence lengths and pad to max
    let seqLens = embedded.map { $0.dim(1) }
    let maxSeq = seqLen

    var padded: [MLXArray] = []
    for emb in embedded {
      let curLen = emb.dim(1)
      if curLen < maxSeq {
        let pad = MLXArray.zeros([1, maxSeq - curLen, config.dim], type: Float.self)
        padded.append(MLX.concatenated([emb, pad], axis: 1))
      } else {
        padded.append(emb)
      }
    }
    var h = MLX.concatenated(padded, axis: 0)  // [B, maxSeq, dim]

    // Time embeddings (per-position)
    var tExpanded = t
    if t.ndim == 1 {
      // Expand timestep to all positions: [B] -> [B, maxSeq]
      tExpanded = MLX.broadcast(t.reshaped(-1, 1), to: [b, maxSeq])
    }

    // Sinusoidal embedding -> time MLP -> projection
    let tFlat = tExpanded.reshaped(-1)  // [B * maxSeq]
    let sinEmb = sinusoidalEmbedding1D(dim: config.freqDim, position: tFlat)
      .reshaped(b, maxSeq, config.freqDim).asType(.float32)
    let timeEmb = timeEmbedding(sinEmb)  // [B, maxSeq, dim]
    let timeProj = timeProjection(timeEmb)  // [B, maxSeq, 6*dim]
    let e0 = timeProj.reshaped(b, maxSeq, 6, config.dim)  // [B, maxSeq, 6, dim]

    // Trace embedding pipeline values
    debugLog("[EMBED] sinEmb[0,0,0:8] = [\((0..<min(8, config.freqDim)).map { sinEmb[0, 0, $0].item(Float.self).description }.joined(separator: ", "))]")
    debugLog("[EMBED] timeEmb mean=\(timeEmb.mean().item(Float.self)), std=\(MLX.sqrt(timeEmb.variance()).item(Float.self))")
    debugLog("[EMBED] e0 mean=\(e0.mean().item(Float.self)), std=\(MLX.sqrt(e0.variance()).item(Float.self))")
    debugLog("[EMBED] h (after patch) mean=\(h.mean().item(Float.self)), std=\(MLX.sqrt(h.variance()).item(Float.self))")

    // Text embedding
    var contextPadded: [MLXArray] = []
    for ctx in context {
      let curLen = ctx.dim(0)
      if curLen < config.textLen {
        let pad = MLXArray.zeros([config.textLen - curLen, ctx.dim(1)], type: Float.self)
        contextPadded.append(MLX.concatenated([ctx, pad], axis: 0))
      } else {
        contextPadded.append(ctx)
      }
    }
    let contextStacked = MLX.stacked(contextPadded, axis: 0)  // [B, textLen, textDim]
    let contextEmb = textEmbedding(contextStacked)  // [B, textLen, dim]

    // Run through blocks
    for (blockIdx, block) in blocks.enumerated() {
      h = block(
        h, e: e0, seqLens: seqLens, gridSizes: gridSizesList,
        freqs: freqs, context: contextEmb, contextLens: nil
      )
      // Trace first and last few blocks to find divergence point
      if blockIdx < 3 || blockIdx >= blocks.count - 2 {
        let hf = h.asType(.float32)
        debugLog("[BLOCK] block \(blockIdx): mean=\(hf.mean().item(Float.self)), std=\(MLX.sqrt(hf.variance()).item(Float.self)), min=\(hf.min().item(Float.self)), max=\(hf.max().item(Float.self))")
      }
    }

    // Head
    let headOut = headModule(h, e: timeEmb)  // [B, maxSeq, outDim * prod(patchSize)]

    // Unpatchify
    let results = Self.unpatchify(
      headOut, gridSizes: gridSizesList,
      patchSize: config.patchSize, outDim: config.outDim
    )
    return results.map { $0.asType(.float32) }
  }

  // MARK: - Unpatchify

  /// Reconstructs video tensors from patch embeddings.
  ///
  /// For each sample, given grid_sizes `[F, H, W]` and patch_size `(pT, pH, pW)`:
  /// ```
  /// u = u[:F*H*W].reshape(F, H, W, pT, pH, pW, outDim)
  /// u = einsum('fhwpqrc->cfphqwr', u)
  /// u = u.reshape(outDim, F*pT, H*pH, W*pW)
  /// ```
  ///
  /// - Parameters:
  ///   - x: Batched patch features `[B, maxSeq, outDim * prod(patchSize)]`.
  ///   - gridSizes: Patch grid dimensions per sample.
  ///   - patchSize: 3D patch dimensions.
  ///   - outDim: Output channels.
  /// - Returns: List of video tensors, each `[outDim, F*pT, H*pH, W*pW]`.
  public static func unpatchify(
    _ x: MLXArray,
    gridSizes: [[Int]],
    patchSize: (Int, Int, Int),
    outDim: Int
  ) -> [MLXArray] {
    let (pT, pH, pW) = patchSize
    var results: [MLXArray] = []

    for (i, grid) in gridSizes.enumerated() {
      let f = grid[0]
      let h = grid[1]
      let w = grid[2]
      let numPatches = f * h * w

      // Extract patches for this sample: [numPatches, outDim*pT*pH*pW]
      let patches = x[i, 0..<numPatches]

      // Reshape to [F, H, W, pT, pH, pW, outDim]
      let reshaped = patches.reshaped(f, h, w, pT, pH, pW, outDim)

      // Permute: fhwpqrc -> cfphqwr (0123456 -> 6031425)
      let permuted = reshaped.transposed(6, 0, 3, 1, 4, 2, 5)

      // Flatten to [outDim, F*pT, H*pH, W*pW]
      let output = permuted.reshaped(outDim, f * pT, h * pH, w * pW)
      results.append(output)
    }

    return results
  }
}

// MARK: - Embedding Modules

/// Patch embedding via Conv3d for the Wan transformer.
///
/// Weight key: `patch_embedding.weight` `[dim, inDim, pT, pH, pW]`,
/// `patch_embedding.bias` `[dim]`.
public final class WanPatchEmbedding: Module {

  public var weight: MLXArray
  public var bias: MLXArray

  public let inDim: Int
  public let dim: Int
  public let patchSize: (Int, Int, Int)

  public init(inDim: Int, dim: Int, patchSize: (Int, Int, Int)) {
    self.inDim = inDim
    self.dim = dim
    self.patchSize = patchSize

    // Conv3d weight: [outCh, kT, kH, kW, inCh] in MLX channels-last
    self.weight = MLXArray.zeros([dim, patchSize.0, patchSize.1, patchSize.2, inDim])
    self.bias = MLXArray.zeros([dim])

    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Transpose BCTHW -> BTHWC for MLX conv3d
    var input = x.transposed(0, 2, 3, 4, 1)

    let output = MLX.conv3d(
      input, weight,
      stride: IntOrTriple((patchSize.0, patchSize.1, patchSize.2)),
      padding: IntOrTriple(0)
    )

    let biased = output + bias

    // Transpose BTHWC -> BCTHW
    return biased.transposed(0, 4, 1, 2, 3)
  }
}

/// Text embedding MLP: Linear(textDim, dim) -> GELU(tanh) -> Linear(dim, dim).
///
/// Weight keys:
/// ```
/// text_embedding.0.weight  [dim, textDim]
/// text_embedding.0.bias    [dim]
/// text_embedding.2.weight  [dim, dim]
/// text_embedding.2.bias    [dim]
/// ```
public final class WanTextEmbeddingMLP: Module {

  public let layers: [Module]

  public init(textDim: Int, dim: Int) {
    self.layers = [
      Linear(textDim, dim),    // index 0
      GELUPlaceholder(),       // index 1 (GELU, no params)
      Linear(dim, dim),        // index 2
    ]
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = (layers[0] as! Linear)(x)
    h = geluApprox(h)
    h = (layers[2] as! Linear)(h)
    return h
  }
}

/// Time embedding MLP: Linear(freqDim, dim) -> SiLU -> Linear(dim, dim).
///
/// Weight keys:
/// ```
/// time_embedding.0.weight  [dim, freqDim]
/// time_embedding.0.bias    [dim]
/// time_embedding.2.weight  [dim, dim]
/// time_embedding.2.bias    [dim]
/// ```
public final class WanTimeEmbeddingMLP: Module {

  public let layers: [Module]

  public init(freqDim: Int, dim: Int) {
    self.layers = [
      Linear(freqDim, dim),    // index 0
      SiLUPlaceholder(),       // index 1 (SiLU, no params -- reuse from VAE)
      Linear(dim, dim),        // index 2
    ]
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = (layers[0] as! Linear)(x)
    h = silu(h)
    h = (layers[2] as! Linear)(h)
    return h
  }
}

/// Time projection: SiLU -> Linear(dim, dim*6).
///
/// Weight keys:
/// ```
/// time_projection.1.weight  [dim*6, dim]
/// time_projection.1.bias    [dim*6]
/// ```
/// Index 0 is SiLU (no parameters).
public final class WanTimeProjection: Module {

  public let layers: [Module]

  public init(dim: Int) {
    self.layers = [
      SiLUPlaceholder(),       // index 0 (SiLU, no params)
      Linear(dim, dim * 6),    // index 1
    ]
    super.init()
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = silu(x)
    h = (layers[1] as! Linear)(h)
    return h
  }
}
