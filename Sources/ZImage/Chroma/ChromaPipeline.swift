import Foundation
import MLX
import MLXRandom
import MLXNN

/// Chroma image generation pipeline.
///
/// Combines T5-XXL text encoder, Chroma transformer (with Approximator),
/// FLUX VAE decoder, and Chroma-specific flow matching sampler.
public final class ChromaPipeline {
  public let transformer: ChromaTransformer
  public let t5: T5Encoder
  public let vae: AutoencoderKL  // Same VAE as Flux
  public let sampler: ChromaSampler
  public let config: ChromaConfig

  public init(
    transformer: ChromaTransformer,
    t5: T5Encoder,
    vae: AutoencoderKL,
    config: ChromaConfig = .standard
  ) {
    self.transformer = transformer
    self.t5 = t5
    self.vae = vae
    self.sampler = ChromaSampler()
    self.config = config
  }

  // MARK: - Latent Preparation

  /// Pack image latents into 2x2 patches and create position IDs.
  func prepareLatentImages(_ latents: MLXArray) -> (MLXArray, MLXArray) {
    let b = latents.dim(0)
    let h = latents.dim(1)
    let w = latents.dim(2)
    let c = latents.dim(3)

    let packed = latents
      .reshaped(b, h / 2, 2, w / 2, 2, c)
      .transposed(0, 1, 3, 5, 2, 4)
      .reshaped(b, h * w / 4, c * 4)

    let hTokens = h / 2
    let wTokens = w / 2
    var posIds: [Int32] = []
    for row in 0..<hTokens {
      for col in 0..<wTokens {
        posIds.append(contentsOf: [0, Int32(row), Int32(col)])
      }
    }
    let ids = MLXArray(posIds).reshaped(1, hTokens * wTokens, 3)
    let batchIds = MLX.broadcast(ids, to: [b, hTokens * wTokens, 3])

    return (packed, batchIds)
  }

  /// Prepare text conditioning from T5 embeddings.
  func prepareConditioning(txt: MLXArray, nImages: Int) -> (MLXArray, MLXArray) {
    var txtEmb = txt
    if txt.dim(0) == 1 && nImages > 1 {
      txtEmb = MLX.broadcast(txt, to: [nImages, txt.dim(1), txt.dim(2)])
    }
    let txtIds = MLX.zeros([nImages, txtEmb.dim(1), 3], dtype: .int32)
    return (txtEmb, txtIds)
  }

  // MARK: - Denoising

  /// Run the denoising loop with Euler stepping and optional CFG.
  public func denoise(
    xT: MLXArray,
    xIds: MLXArray,
    txt: MLXArray,
    txtIds: MLXArray,
    negTxt: MLXArray? = nil,
    negTxtIds: MLXArray? = nil,
    numSteps: Int = 28,
    guidance: Float = 0.0,
    cfg: Float = 4.0,
    firstNStepsWithoutCFG: Int = 0,
    schedulerType: ChromaSchedulerType = .euler,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) throws -> MLXArray {
    let batch = xT.dim(0)
    let imageSeqLen = xT.dim(1)

    let guidanceArr = MLX.full([batch], values: MLXArray(guidance), dtype: .bfloat16)

    // Select timestep schedule
    let timesteps: [Float]
    switch schedulerType {
    case .euler, .heun:
      timesteps = sampler.timesteps(numSteps: numSteps, imageSequenceLength: imageSeqLen)
    case .beta:
      timesteps = sampler.betaTimesteps(numSteps: numSteps, imageSequenceLength: imageSeqLen)
    }

    let useCFG = negTxt != nil && firstNStepsWithoutCFG != -1
    let useHeun = (schedulerType == .heun || schedulerType == .beta)

    var x = xT
    for i in 0..<numSteps {
      // comfybox#304: step-boundary cancellation check, matching the
      // Flux2/Fibo idiom (Flux2Pipeline.swift, FiboPipeline.swift) — one
      // check per sampler step, CancellationError propagates unmodified.
      try Task.checkCancellation()
      let t = timesteps[i]
      let tPrev = timesteps[i + 1]

      let tArr = MLX.full([batch], values: MLXArray(t), dtype: .bfloat16)

      // First model evaluation
      var pred = transformer(
        img: x,
        imgIds: xIds,
        txt: txt,
        txtIds: txtIds,
        timesteps: tArr,
        guidance: guidanceArr
      )

      // Apply CFG if active
      if useCFG && i >= firstNStepsWithoutCFG {
        let predNeg = transformer(
          img: x,
          imgIds: xIds,
          txt: negTxt!,
          txtIds: negTxtIds!,
          timesteps: tArr,
          guidance: guidanceArr
        )
        pred = predNeg + (pred - predNeg) * MLXArray(cfg)
      }

      if useHeun && i < numSteps - 1 {
        // Heun: second-order step
        // 1. Euler step to intermediate
        let xMid = sampler.eulerIntermediate(pred: pred, xT: x, t: t, tPrev: tPrev)
        eval(xMid)

        // 2. Evaluate at intermediate point
        let tPrevArr = MLX.full([batch], values: MLXArray(tPrev), dtype: .bfloat16)
        var pred2 = transformer(
          img: xMid,
          imgIds: xIds,
          txt: txt,
          txtIds: txtIds,
          timesteps: tPrevArr,
          guidance: guidanceArr
        )

        // Apply CFG to second prediction too
        if useCFG && i >= firstNStepsWithoutCFG {
          let predNeg2 = transformer(
            img: xMid,
            imgIds: xIds,
            txt: negTxt!,
            txtIds: negTxtIds!,
            timesteps: tPrevArr,
            guidance: guidanceArr
          )
          pred2 = predNeg2 + (pred2 - predNeg2) * MLXArray(cfg)
        }

        // 3. Trapezoidal combination
        x = sampler.heunStep(pred1: pred, pred2: pred2, xT: x, t: t, tPrev: tPrev)
      } else {
        // Euler step (first order) — also used for last Heun step
        x = sampler.step(pred: pred, xT: x, t: t, tPrev: tPrev)
      }

      eval(x)
      progressCallback?(i + 1, numSteps)
    }

    return x
  }

  // MARK: - VAE Decoding

  /// Unpack and decode latents to pixel space.
  ///
  /// Returns NHWC `[B, H, W, 3]` with values in `[0, 1]`.
  /// VAE decoding runs in float32 for precision (Python reference requirement).
  public func decode(_ x: MLXArray, latentSize: (Int, Int)) -> MLXArray {
    let (h, w) = latentSize
    let b = x.dim(0)

    // Unpack from 2x2 patches → NHWC [B, H, W, C]
    let unpacked = x
      .reshaped(b, h / 2, w / 2, -1, 2, 2)
      .transposed(0, 1, 4, 2, 5, 3)
      .reshaped(b, h, w, -1)

    // Cast to float32 for VAE precision
    let fp32 = unpacked.asType(.float32)

    // Transpose NHWC → NCHW for AutoencoderKL.decode()
    let nchw = fp32.transposed(0, 3, 1, 2)
    let (decoded, _) = vae.decode(nchw)

    // decoded is NCHW [B, C, H, W] — transpose back to NHWC [B, H, W, C]
    let nhwc = decoded.transposed(0, 2, 3, 1)

    // Map from [-1, 1] to [0, 1]
    return MLX.clip(nhwc + 1, min: 0, max: 2) * 0.5
  }

  // MARK: - Full Generation

  /// Generate an image from pre-tokenized T5 input.
  ///
  /// - Parameters:
  ///   - schedulerType: Denoising schedule (`.euler`, `.heun`, `.beta`). Default `.euler`.
  ///   - negativeTokenIds: Optional negative prompt token IDs for CFG.
  ///   - cfg: Classifier-free guidance scale (default 4.0).
  ///   - firstNStepsWithoutCFG: Steps to run without CFG before enabling it (default 0).
  public func generate(
    tokenIds: MLXArray,
    negativeTokenIds: MLXArray? = nil,
    width: Int = 512,
    height: Int = 512,
    numSteps: Int = 28,
    guidance: Float = 0.0,
    cfg: Float = 4.0,
    firstNStepsWithoutCFG: Int = 0,
    schedulerType: ChromaSchedulerType = .euler,
    seed: UInt64? = nil,
    progressCallback: ((Int, Int) -> Void)? = nil
  ) throws -> MLXArray {
    if let seed { MLXRandom.seed(seed) }

    let latentH = height / 8
    let latentW = width / 8

    // T5 encode positive prompt
    let txt = t5(tokenIds)
    eval(txt)

    // Prepare positive conditioning
    let (txtEmb, txtIds) = prepareConditioning(txt: txt, nImages: 1)

    // T5 encode negative prompt for CFG
    var negTxt: MLXArray? = nil
    var negTxtIds: MLXArray? = nil
    if let negativeTokenIds, firstNStepsWithoutCFG != -1 {
      let negTxtRaw = t5(negativeTokenIds)
      eval(negTxtRaw)
      let (negTxtEmb, negTxtIdsPrepared) = prepareConditioning(txt: negTxtRaw, nImages: 1)
      negTxt = negTxtEmb
      negTxtIds = negTxtIdsPrepared
    }

    // Sample noise
    let noise = sampler.samplePrior(shape: [1, latentH, latentW, 16], seed: seed)
    let (packed, imgIds) = prepareLatentImages(noise)

    // Denoise
    let denoised = try denoise(
      xT: packed,
      xIds: imgIds,
      txt: txtEmb,
      txtIds: txtIds,
      negTxt: negTxt,
      negTxtIds: negTxtIds,
      numSteps: numSteps,
      guidance: guidance,
      cfg: cfg,
      firstNStepsWithoutCFG: firstNStepsWithoutCFG,
      schedulerType: schedulerType,
      progressCallback: progressCallback
    )

    // Decode
    return decode(denoised, latentSize: (latentH, latentW))
  }
}
