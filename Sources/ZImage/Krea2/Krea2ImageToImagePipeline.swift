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
    /// Negative prompt for the CFG branch — only consulted when guidance > 1.
    public var negativePrompt: String?
    /// CFG scale; 1.0 (default) = distilled single-pass, no negative. See
    /// Krea2Pipeline.Request.guidance.
    public var guidance: Float = 1.0
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
    /// Explicit schedule shift (FDD-krea2-raw-recipe D3); see `Krea2Pipeline.Request.shift`.
    public var shift: Float? = nil
    /// The sampler the denoise loop runs; see `Krea2Pipeline.Request.sampler`.
    public var sampler: SchedulerKind = .euler
    /// The sigma grid the sampler walks; see `Krea2Pipeline.Request.sigmaSchedule`.
    public var sigmaSchedule: SigmaScheduleKind = .krea2
    /// The raw schedule name the caller sent (D22); see `Krea2Pipeline.Request.sigmaScheduleRequested`.
    public var sigmaScheduleRequested: String? = nil
    /// RES4LYF SDE eta (T2, WP-E15); see `Krea2Pipeline.Request.eta`.
    public var eta: Float = 0.0
    /// RES4LYF `bongmath` (T3, WP-E16); see `Krea2Pipeline.Request.bongmath`.
    public var bongmath: Bool = false
    /// `res_2s` / `res_3s` substep; absent on the wire defaults to 0.5.
    public var c2: Float = 0.5
    /// Text-conditioning gain on the fusion projector; see `Krea2Pipeline.Request.projectorScale`.
    public var projectorScale: Float = 1.0
    /// RES4LYF spatial noise generator; see `Krea2Pipeline.Request.noiseType`.
    public var noiseType: RES4LYFNoiseType = .gaussian
    /// Fractal `alpha`; see `Krea2Pipeline.Request.noiseAlpha`.
    public var noiseAlpha: Float = 0.0
    /// RES4LYF implicit-RK refinement; see `Krea2Pipeline.Request.implicitStepsFull`.
    /// 0 (default) is byte-identical to today.
    public var implicitStepsFull: Int = 0

    public init(
      prompt: String, negativePrompt: String? = nil, guidance: Float = 1.0,
      sourceImage: MLXArray, width: Int = 1024, height: Int = 1024,
      steps: Int = 9, seed: UInt64 = 0, strength: Float = 0.3,
      dyPE: DyPEConfig = .disabled, shift: Float? = nil,
      sampler: SchedulerKind = .euler, sigmaSchedule: SigmaScheduleKind = .krea2,
      sigmaScheduleRequested: String? = nil,
      eta: Float = 0.0, bongmath: Bool = false, c2: Float = 0.5,
      projectorScale: Float = 1.0,
      noiseType: RES4LYFNoiseType = .gaussian, noiseAlpha: Float = 0.0,
      implicitStepsFull: Int = 0
    ) {
      self.prompt = prompt
      self.negativePrompt = negativePrompt
      self.guidance = guidance
      self.sourceImage = sourceImage
      self.width = width
      self.height = height
      self.steps = steps
      self.seed = seed
      self.strength = strength
      self.dyPE = dyPE
      self.shift = shift
      self.sampler = sampler
      self.sigmaSchedule = sigmaSchedule
      self.sigmaScheduleRequested = sigmaScheduleRequested
      self.eta = eta
      self.bongmath = bongmath
      self.c2 = c2
      self.projectorScale = projectorScale
      self.noiseType = noiseType
      self.noiseAlpha = noiseAlpha
      self.implicitStepsFull = implicitStepsFull
    }
  }

  /// Generate one image conditioned on a source image. Returns RGB float
  /// array (H, W, 3) in [0,1] — same output shape as `generate(_:progress:)`.
  ///
  /// A one-line wrapper over ``generateImg2ImgWithRecipe(_:progress:)``.
  ///
  /// - Throws: ``Krea2ScheduleError`` when the request's `shift` is invalid or
  ///   a field belongs to an unimplemented tier (checked before any model work).
  public func generateImg2Img(
    _ request: Img2ImgRequest,
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> MLXArray {
    try generateImg2ImgWithRecipe(request, progress: progress).image
  }

  /// Generate one img2img render and the record of how (WP-E3 + WP-E10).
  public func generateImg2ImgWithRecipe(
    _ request: Img2ImgRequest,
    progress: ((Int, Int) -> Void)? = nil
  ) throws -> (image: MLXArray, trace: Krea2RunTrace) {
    let dtype = DType.bfloat16
    let patch = config.patch
    let comp = Krea2VAE.spatialScale
    let align = comp * patch
    let width = Krea2Sampling.roundUp(request.width, multiple: align)
    let height = Krea2Sampling.roundUp(request.height, multiple: align)

    let latH = height / comp, latW = width / comp
    let hTok = latH / patch, wTok = latW / patch
    // D3/A.1: nil → resolution-dependent mu (unchanged); explicit → mu = shift. Fails before any model work.
    let scheduleShift = try Krea2Sampling.resolveShift(
      explicit: request.shift, seqLen: hTok * wTok, align: align)
    // D18: unimplemented tiers are refused before any model work.
    try Krea2Pipeline.validateTiers(eta: request.eta, bongmath: request.bongmath)
    // WP-E15: the T2 SDE, on the same terms as the t2i path — `nil` at eta 0,
    // a refusal (never a silent drop) on a sampler it is not defined against.
    let sdeNoise = try Krea2Pipeline.makeSDEInjector(
      eta: request.eta, sampler: request.sampler, stageSeed: request.seed,
      layout: Krea2Pipeline.sdeNoiseLayout,
      noiseType: request.noiseType, noiseGrid: (hTok: hTok, wTok: wTok),
      noiseAlpha: Double(request.noiseAlpha))
    // WP-E16: the T3 fixed point. `nil` at bongmath false — the loop is then
    // handed no hook and is bit-identical to the run without one — and a
    // refusal naming the sampler when it is asked for with one RES4LYF's
    // tableau inversion is not defined against. Before any model work, for the
    // same reason the SDE injector is.
    let bongMath = try Krea2Pipeline.makeBongMath(
      bongmath: request.bongmath, sampler: request.sampler,
      sigmaSchedule: request.sigmaSchedule, shift: scheduleShift)

    // Projector-scale trick: fusion gain on the warm transformer (always set,
    // default 1.0, so no value leaks between renders).
    transformer.txtfusion.projectorScale = request.projectorScale
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

    // CFG branch — mirrors generate() (opt-in via guidance > 1, sequential).
    let useCFG = request.guidance > 1.0
    // K-FIX-1 / Codex I4: what the CFG branch ACTUALLY conditions on,
    // resolved here — beside the encode, from the same `useCFG` predicate —
    // and carried in the trace so every provenance sink records the render
    // rather than the request. An omitted negative under CFG is `""`, not
    // "no negative prompt": the second model pass ran either way.
    let negativePromptApplied = Krea2RunTrace.negativePromptApplied(
      cfgActive: useCFG, requested: request.negativePrompt)
    var negCtx: MLXArray? = nil
    var negPos: MLXArray? = nil
    var negFullMask: MLXArray? = nil
    if useCFG {
      let (nRaw, nMask) = conditioner.encode([request.negativePrompt ?? ""])
      negCtx = nRaw.asType(dtype)
      negPos = Krea2Sampling.buildPositions(txtLen: negCtx!.dim(1), h: hTok, w: wTok)
      negFullMask = MLX.concatenated([nMask, MLX.ones([1, hTok * wTok])], axis: 1)
    }

    var scheduler = try Krea2Pipeline.makeScheduler(
      sampler: request.sampler, sigmaSchedule: request.sigmaSchedule,
      steps: request.steps, shift: scheduleShift, seed: request.seed, c2: request.c2)
    let total = scheduler.numInferenceSteps

    // strength -> denoise -> startIndex, matching Z-Image's img2img convention
    // exactly (Img2ImgRequest.denoise in ImageToImagePipeline.swift):
    //   denoise = 1 - strength; startStep = max(0, steps - ceil(steps * denoise))
    let denoise = 1.0 - max(0.01, min(0.99, request.strength))
    let startIndex = Krea2Sampling.img2imgStartIndex(total: total, strength: request.strength)

    // Mix noise and the source latent at sigmas[startIndex]. The grid runs
    // 1 (pure noise) -> 0 (clean data), so this is the standard rectified-flow
    // "noise a real sample to time σ" interpolation — the same formula the
    // pure-noise path implicitly uses at σ=1 (all noise, no data). The sigma
    // stays a float32 MLXArray: see `Krea2Sampling.mixSourceLatent` for why
    // `.item(Float.self)` would silently move every img2img render (§3.3).
    let mixedNCHW = Krea2Sampling.mixSourceLatent(
      noise: noise, source: sourceLatent, sigma: scheduler.sigmas[startIndex], dtype: dtype)
    let img = Krea2Sampling.patchify(mixedNCHW, patch: patch)

    let (denoised, stats) = try Krea2DenoiseLoop.run(
      scheduler: &scheduler,
      initialSample: img,
      startIndex: startIndex,
      modelEvalsPerEvaluate: useCFG ? 2 : 1,
      implicitStepsFull: request.implicitStepsFull,
      evaluate: { [transformer] latent, sigma in
        let t = MLX.full([1], values: MLXArray(sigma)).asType(dtype)
        let vCond = transformer(img: latent, context: ctx, t: t, pos: pos, mask: fullMask,
                                ropeScales: ropeScales)
        guard useCFG, let negCtx, let negPos, let negFullMask else { return vCond }
        let vUncond = transformer(img: latent, context: negCtx, t: t, pos: negPos,
                                  mask: negFullMask, ropeScales: ropeScales)
        return Krea2Sampling.applyCFG(cond: vCond, uncond: vUncond, scale: request.guidance)
      },
      noise: sdeNoise,
      bongmath: bongMath,
      progress: progress)

    let latentNCHW = Krea2Sampling.unpatchify(
      denoised, patch: patch, h: hTok, w: wTok, c: Krea2VAE.latentChannels)
    let latentNHWC = latentNCHW.transposed(0, 2, 3, 1).asType(.float32)
    let decoded = vae.decode(latentNHWC)
    MLX.eval(decoded)

    let trace = Krea2RunTrace(
      request: request, shift: scheduleShift, scheduler: scheduler, stats: stats,
      startIndex: startIndex, denoise: denoise, width: width, height: height,
      negativePromptApplied: negativePromptApplied,
      // WP-E16: what the fixed point DID, counted by the hook while it ran —
      // `nil` when the render had none.
      bong: Krea2RunTrace.BongMathParameters.forRun(bongMath))
    return (decoded[0], trace)
  }
}
