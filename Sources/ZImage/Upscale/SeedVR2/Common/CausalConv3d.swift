import Foundation
import MLX
import MLXNN

/// A 3D convolution with causal temporal padding for the SeedVR2 video VAE.
///
/// In video autoencoders, temporal causality ensures that the encoding of frame `t`
/// depends only on frames `<= t`. This is achieved by replicating the first frame
/// to fill the temporal receptive field, rather than using symmetric zero-padding.
///
/// ## Weight Layout
///
/// Weights are stored as `(C_out, K_t, K_h, K_w, C_in)` — the MLX channels-last
/// layout for 3D convolution kernels.
///
/// ## Input Layout
///
/// The Python reference uses BCTHW (batch, channels, time, height, width) as the
/// external interface. Internally, the convolution operates on BTHWC (channels-last)
/// because MLX `convGeneral` expects `(N, ..., C_in)` inputs. This module handles
/// the necessary transposes.
///
/// ## Padding Modes
///
/// - `causalTemporal = true` (default): Replicates the first temporal frame
///   `(kernel_t - 1)` times and prepends them, with zero temporal padding in the
///   convolution itself. Spatial padding is applied normally.
/// - `causalTemporal = true, usePaddingCausal = true`: Uses `2 * padding_t` replicated
///   frames instead of `kernel_t - 1`.
/// - `causalTemporal = false`: Standard symmetric temporal padding (no replication).
public final class CausalConv3d: Module {

  /// Convolution kernel weights: shape `(C_out, K_t, K_h, K_w, C_in)`.
  public var weight: MLXArray

  /// Bias vector: shape `(C_out,)`.
  public var bias: MLXArray

  /// Kernel size as `(temporal, height, width)`.
  public let kernelSize: (Int, Int, Int)

  /// Stride as `(temporal, height, width)`.
  public let stride: (Int, Int, Int)

  /// Nominal padding as `(temporal, height, width)`.
  /// When ``causalTemporal`` is true, the temporal component drives the causal
  /// pad size but is not passed to the convolution.
  public let padding: (Int, Int, Int)

  /// Whether to apply causal temporal padding via frame replication.
  public let causalTemporal: Bool

  /// When true and ``causalTemporal`` is true, uses `2 * padding.temporal`
  /// replicated frames instead of `kernel_t - 1`.
  public let usePaddingCausal: Bool

  // MARK: Streaming decode state (#36)
  //
  // Port of ComfyUI's CausalConv3d temporal streaming (causal_conv3d.py):
  // when active, temporal context is carried between calls so a long clip can
  // be decoded in frame chunks with results IDENTICAL to a single full-tensor
  // call — no seams, no blending. Required because MLX Metal kernels silently
  // corrupt outputs via int32 offset overflow on very large tensors
  // (ml-explore/mlx #3836/#3609/#3524; fixed only in mlx core >= 0.32.0,
  // which no mlx-swift release bundles yet).
  //
  // Protocol per stream: resetStream(active: true) on every conv; feed chunks
  // in order; set streamEnded = true on all convs BEFORE the final chunk;
  // resetStream(active: false) when done. Renders are serialized on the GPU
  // FIFO queue, so plain instance state is safe (no concurrent streams).

  /// Whether streaming mode is active (temporal context carried across calls).
  public var streamActive = false
  /// Cached trailing input frames from the previous chunk (pre-conv, BCTHW).
  public var streamCache: MLXArray? = nil
  /// Set before the final chunk: appends trailing replicate-padding
  /// (non-causal mode) and stops caching.
  public var streamEnded = false

  /// Reset (and enable/disable) streaming state.
  public func resetStream(active: Bool) {
    streamActive = active
    streamCache = nil
    streamEnded = false
  }

  /// Creates a causal 3D convolution layer.
  ///
  /// - Parameters:
  ///   - inChannels: Number of input channels.
  ///   - outChannels: Number of output channels.
  ///   - kernelSize: Convolution kernel size as `(t, h, w)`.
  ///   - stride: Convolution stride as `(t, h, w)`. Default `(1, 1, 1)`.
  ///   - padding: Nominal padding as `(t, h, w)`. Default `(1, 1, 1)`.
  ///   - causalTemporal: Whether to replicate the first frame for causal padding.
  ///     Default `true`.
  ///   - usePaddingCausal: If true, the causal pad count is `2 * padding.t` instead
  ///     of `kernelSize.t - 1`. Default `false`.
  public init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: (Int, Int, Int) = (3, 3, 3),
    stride: (Int, Int, Int) = (1, 1, 1),
    padding: (Int, Int, Int) = (1, 1, 1),
    causalTemporal: Bool = true,
    usePaddingCausal: Bool = false
  ) {
    self.kernelSize = kernelSize
    self.stride = stride
    self.padding = padding
    self.causalTemporal = causalTemporal
    self.usePaddingCausal = usePaddingCausal

    let (kt, kh, kw) = kernelSize
    self.weight = MLXArray.zeros([outChannels, kt, kh, kw, inChannels])
    self.bias = MLXArray.zeros([outChannels])

    super.init()
  }

  /// Applies the causal 3D convolution.
  ///
  /// - Parameter x: Input tensor of shape `(B, C, T, H, W)` (channels-first,
  ///   matching the Python SeedVR2 VAE convention).
  /// - Returns: Output tensor of shape `(B, C_out, T_out, H_out, W_out)`.
  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var input = x
    let (kt, _, _) = kernelSize
    let (pt, ph, pw) = padding

    // --- Temporal padding ---
    let temporalPadding: Int
    if streamActive && kt > 1 && stride.0 == 1 {
      // Streaming: assemble [cache | chunk | end-pad?]; the cache replaces
      // ordinary temporal padding for every chunk after the first.
      // An empty chunk before the stream has started must NOT touch the
      // cache — caching a 0-frame tensor would suppress the first-chunk
      // replicate padding when real frames arrive (ComfyUI guards this the
      // same way: `if x.shape[2] == 0: return x` before cache init).
      if input.dim(2) == 0 && streamCache == nil {
        return MLXArray.zeros(
          [x.dim(0), weight.dim(0), 0, x.dim(3), x.dim(4)], dtype: x.dtype)
      }
      var pieces: [MLXArray] = []
      if let cached = streamCache {
        pieces.append(cached)
      } else if input.dim(2) > 0 {
        // First chunk: left-pad by replicating frame 0 (matches the
        // non-streaming replicate padding on both causal and non-causal paths).
        let padLen = causalTemporal ? (usePaddingCausal ? 2 * pt : kt - 1) : (kt - 1) / 2
        if padLen > 0 {
          let firstFrame = input[0..., 0..., ..<1, 0..., 0...]
          pieces.append(MLX.repeated(firstFrame, count: padLen, axis: 2))
        }
      }
      pieces.append(input)
      let body = pieces.count == 1 ? pieces[0] : MLX.concatenated(pieces, axis: 2)
      if streamEnded {
        streamCache = nil
        if !causalTemporal {
          let halfPad = (kt - 1) / 2
          if halfPad > 0, body.dim(2) > 0 {
            let lastFrame = body[0..., 0..., (body.dim(2) - 1)..., 0..., 0...]
            input = MLX.concatenated(
              [body, MLX.repeated(lastFrame, count: halfPad, axis: 2)], axis: 2)
          } else {
            input = body
          }
        } else {
          input = body
        }
      } else {
        // Cache the trailing (kt - 1) pre-conv frames for the next chunk.
        if body.dim(2) > 0 {
          let cacheLen = min(kt - 1, body.dim(2))
          streamCache = body[0..., 0..., (body.dim(2) - cacheLen)..., 0..., 0...]
        }
        input = body
      }
      if input.dim(2) < kt {
        // Not enough frames yet — everything is in the cache; emit nothing.
        return MLXArray.zeros(
          [x.dim(0), weight.dim(0), 0, x.dim(3), x.dim(4)], dtype: x.dtype)
      }
      temporalPadding = 0
    } else if causalTemporal && kt > 1 {
      let causalPad = usePaddingCausal ? (2 * pt) : (kt - 1)
      if causalPad > 0 {
        // Replicate the first frame along the temporal axis.
        // Input shape: (B, C, T, H, W) — temporal is axis 2.
        let firstFrame = input[0..., 0..., ..<1, 0..., 0...]
        let padFrames = MLX.repeated(firstFrame, count: causalPad, axis: 2)
        input = MLX.concatenated([padFrames, input], axis: 2)
      }
      temporalPadding = 0
    } else if kt > 1 {
      // Non-causal: symmetric temporal padding by replicating boundary frames.
      let halfPad = (kt - 1) / 2
      if halfPad > 0 {
        let firstFrame = input[0..., 0..., ..<1, 0..., 0...]
        let lastFrame = input[0..., 0..., (input.dim(2) - 1)..., 0..., 0...]
        let padLeft = MLX.repeated(firstFrame, count: halfPad, axis: 2)
        let padRight = MLX.repeated(lastFrame, count: halfPad, axis: 2)
        input = MLX.concatenated([padLeft, input, padRight], axis: 2)
      }
      temporalPadding = 0
    } else {
      temporalPadding = pt
    }

    // --- Transpose BCTHW -> BTHWC for convGeneral ---
    input = input.transposed(0, 2, 3, 4, 1)

    // --- 3D convolution with temporal chunking ---
    // MLX convGeneral produces incorrect results for 5D tensors when the
    // temporal dimension (after padding) exceeds ~64 frames at large spatial
    // resolution. Work around by processing overlapping temporal chunks.
    //
    // After temporal padding and BTHWC transpose, input is [B, T_padded, H, W, C].
    // With kernel_t=kt and stride_t=1, each output frame i depends on input
    // frames [i, i+1, ..., i+kt-1].  So chunks need (kt-1) frames of overlap
    // and each chunk of size S produces S-(kt-1) valid output frames.
    let temporalFrames = input.dim(1)
    let maxTemporalChunk = 64  // Safe limit for MLX Metal conv kernels

    var out: MLXArray
    if temporalFrames > maxTemporalChunk && stride.0 == 1 && kt > 1 {
      let overlap = kt - 1
      let validPerChunk = maxTemporalChunk - overlap  // Net new output frames per chunk
      var chunks: [MLXArray] = []
      var tStart = 0

      while tStart < temporalFrames {
        var tEnd = min(tStart + maxTemporalChunk, temporalFrames)

        // If the leftover after this chunk is too small for a valid conv,
        // extend this chunk to consume the rest.
        let remaining = temporalFrames - tEnd
        if remaining > 0 && remaining < kt {
          tEnd = temporalFrames
        }

        let chunkSize = tEnd - tStart
        // Skip if chunk is smaller than the kernel (shouldn't happen with the guard above)
        guard chunkSize >= kt else { break }

        let chunk = input[0..., tStart..<tEnd, 0..., 0..., 0...]

        var chunkOut = convGeneral(
          chunk, weight,
          strides: IntOrArray([stride.0, stride.1, stride.2]),
          padding: IntOrArray([temporalPadding, ph, pw])
        )
        eval(chunkOut)

        // No output trimming needed: the input overlap provides convolution
        // context but doesn't produce duplicate output frames. Chunk 1's
        // last output frame covers input position (chunkEnd - kt), and
        // chunk 2's first output frame starts at input position chunkStart,
        // which is exactly (chunkEnd - overlap) = next valid position.

        if chunkOut.dim(1) > 0 {
          chunks.append(chunkOut)
        }

        // Advance by validPerChunk so the next chunk starts with `overlap`
        // frames of context from this chunk.
        tStart += validPerChunk
      }

      out = MLX.concatenated(chunks, axis: 1)
    } else {
      out = convGeneral(
        input, weight,
        strides: IntOrArray([stride.0, stride.1, stride.2]),
        padding: IntOrArray([temporalPadding, ph, pw])
      )
    }

    // --- Add bias ---
    out = out + bias

    // --- Transpose BTHWC -> BCTHW ---
    out = out.transposed(0, 4, 1, 2, 3)

    return out
  }
}
