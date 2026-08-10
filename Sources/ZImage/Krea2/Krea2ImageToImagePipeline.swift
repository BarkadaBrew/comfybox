// Krea2ImageToImagePipeline.swift — img2img for Krea-2-Turbo.
//
// Encodes a source image via Krea2VAE.encode, mixes it with noise at the
// point in the flow-matching schedule that `strength` selects, and runs the
// Euler loop only from that point onward — the same "start partway through
// the schedule" approach Z-Image's img2img path uses (ImageToImagePipeline.swift),
// just without going through the inpainting/white-mask indirection Z-Image
// uses to get there (Krea2 has no inpainting path to reuse, so this talks to
// Krea2Sampling directly).

import Foundation
import MLX
import MLXRandom

extension Krea2Pipeline {

  public struct Img2ImgRequest {
    public var prompt: String
    /// Source image, NHWC (1, H, W, 3), RGB in [-1, 1], already resized to
    /// the request's width/height. NOTE: `QwenImageIO.resizedPixelArray` +
    /// `normalizeForEncoder` (the usual way to load+normalize an image in
    /// this codebase) produce NCHW (1, 3, H, W) — transpose with
    /// `.transposed(0, 2, 3, 1)` before constructing this request. NHWC here
    /// matches `Krea2VAE.encode`'s own convention (and its `decode()`
    /// counterpart), not the NCHW convention Z-Image's pipeline uses.
    public var sourceImage: MLXArray
    public var width: Int
    public var height: Int
    public var steps: Int
    public var seed: UInt64
    /// Same convention as Z-Image's img2img (ImageToImagePipeline.swift):
    /// 0.3 = heavy rework (default), 0.7 = light touch. Range (0, 1].
    public var strength: Float
    /// High-resolution position handling. `.disabled` keeps vanilla RoPE.
    public var dyPE: DyPEConfig = .disabled

    public init(
      prompt: String, sourceImage: MLXArray, width: Int = 1024, height: Int = 1024,
      steps: Int = 9, seed: UInt64 = 0, strength: Float = 0.3,
      dyPE: DyPEConfig = .disabled
    ) {
      self.prompt = prompt
      self.sourceImage = sourceImage
      self.width = width
      self.height = height
      self.steps = steps
      self.seed = seed
      self.strength = strength
      self.dyPE = dyPE
    }
  }

  /// Generate one image conditioned on a source image. Returns RGB float
  /// array (H, W, 3) in [0,1] — same output shape as `generate(_:progress:)`.
  public func generateImg2Img(
    _ request: Img2ImgRequest,
    progress: ((Int, Int) -> Void)? = nil
  ) -> MLXArray {
    let dtype = DType.bfloat16
    let patch = config.patch
    let comp = Krea2VAE.spatialScale
    let align = comp * patch
    let width = Krea2Sampling.roundUp(request.width, multiple: align)
    let height = Krea2Sampling.roundUp(request.height, multiple: align)

    let latH = height / comp, latW = width / comp
    let hTok = latH / patch, wTok = latW / patch

    MLXRandom.seed(request.seed)
    let noise = MLXRandom.normal([1, Krea2VAE.latentChannels, latH, latW]).asType(dtype)

    // Encode the source image (VAE works NHWC; the Euler loop below, like
    // generate(_:progress:), works NCHW).
    let sourceLatentNHWC = vae.encode(request.sourceImage)
    let sourceLatent = sourceLatentNHWC.transposed(0, 3, 1, 2).asType(dtype)

    let (ctxRaw, mask) = conditioner.encode([request.prompt])
    let ctx = ctxRaw.asType(dtype)
    let txtLen = ctx.dim(1)

    let pos = Krea2Sampling.buildPositions(txtLen: txtLen, h: hTok, w: wTok)
    let ropeScales = Krea2Sampling.ropeScales(
      hTok: hTok, wTok: wTok, patch: patch, dyPE: request.dyPE)
    let fullMask = MLX.concatenated([mask, MLX.ones([1, hTok * wTok])], axis: 1)

    let x1 = Float((256 / align) * (256 / align))
    let x2 = Float((1280 / align) * (1280 / align))
    let seqLen = hTok * wTok
    let ts = Krea2Sampling.timesteps(seqLen: seqLen, steps: request.steps, x1: x1, x2: x2)
    let total = ts.count - 1

    // strength -> denoise -> startIndex, matching Z-Image's img2img convention
    // exactly (Img2ImgRequest.denoise in ImageToImagePipeline.swift):
    //   denoise = 1 - strength; startStep = max(0, steps - ceil(steps * denoise))
    let denoise = 1.0 - max(0.01, min(0.99, request.strength))
    let startIndex = max(0, total - Int((Double(total) * Double(denoise)).rounded(.up)))

    // Mix noise and the source latent at ts[startIndex]. Krea2Sampling.timesteps
    // runs 1 (pure noise) -> 0 (clean data), so this is the standard
    // rectified-flow "noise a real sample to time t" interpolation — the
    // same formula the pure-noise path implicitly uses at t=ts[0]≈1 (all
    // noise, no data).
    let tStart = MLXArray(ts[startIndex])
    let mixedNCHW = (noise * tStart + sourceLatent * (1.0 - tStart)).asType(dtype)
    var img = Krea2Sampling.patchify(mixedNCHW, patch: patch)

    for i in startIndex..<total {
      let tc = ts[i], tp = ts[i + 1]
      let t = MLX.full([1], values: MLXArray(tc)).asType(dtype)
      let v = transformer(img: img, context: ctx, t: t, pos: pos, mask: fullMask,
                          ropeScales: ropeScales)
      img = img + (tp - tc) * v
      MLX.eval(img)
      progress?(i + 1, total)
    }

    let latentNCHW = Krea2Sampling.unpatchify(
      img, patch: patch, h: hTok, w: wTok, c: Krea2VAE.latentChannels)
    let latentNHWC = latentNCHW.transposed(0, 2, 3, 1).asType(.float32)
    let decoded = vae.decode(latentNHWC)
    MLX.eval(decoded)
    return decoded[0]
  }
}
