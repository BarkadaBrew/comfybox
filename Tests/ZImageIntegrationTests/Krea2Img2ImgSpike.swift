// Krea2Img2ImgSpike.swift — empirical verification for Krea2Pipeline's new
// img2img path (Krea2ImageToImagePipeline.swift): does mixing noise with a
// real encoded source latent at a partial schedule point actually produce a
// coherent image resembling the source, or does it corrupt like the LoKr
// alpha bug did? Writes output to /tmp for visual inspection.

import XCTest
import MLX
import CoreGraphics
import ImageIO
@testable import ZImage

final class Krea2Img2ImgSpike: XCTestCase {

  func testImg2ImgProducesCoherentOutput() throws {
    let paths: Krea2ModelPaths
    do {
      paths = try Krea2ModelPaths.resolve()
    } catch {
      throw XCTSkip("Krea-2 weights not available locally: \(error)")
    }

    let pipeline = try Krea2Pipeline(paths: paths, quantizeTransformer: 8)

    let width = 768
    let height = 1024

    let sourcePath = "/Users/toddwalderman/Pictures/ComfyBox/zimage-krea2-781369D9-E6F8-43CC-8E30-6B1BB342727A.png"
    guard FileManager.default.fileExists(atPath: sourcePath) else {
      throw XCTSkip("Test source image not available at \(sourcePath)")
    }
    let url = URL(fileURLWithPath: sourcePath)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
      throw XCTSkip("Could not load test source image")
    }
    // QwenImageIO.resizedPixelArray is NCHW (1,3,H,W); Krea2VAE.encode (and
    // Img2ImgRequest.sourceImage) is NHWC (1,H,W,3), matching Krea2VAE's own
    // decode() convention — transpose before handing it to the pipeline.
    let pixelsNCHW = try QwenImageIO.resizedPixelArray(
      from: cgImage, width: width, height: height, addBatchDimension: true, dtype: .float32
    )
    let normalizedNCHW = QwenImageIO.normalizeForEncoder(pixelsNCHW)
    let normalizedNHWC = normalizedNCHW.transposed(0, 2, 3, 1)

    let request = Krea2Pipeline.Img2ImgRequest(
      prompt: "a woman in a lace dress sitting on a bed, warm sunset light",
      sourceImage: normalizedNHWC,
      width: width, height: height,
      steps: 9, seed: 7, strength: 0.5
    )

    let decoded = try pipeline.generateImg2Img(request) { step, total in
      print("  img2img step \(step)/\(total)")
    }

    XCTAssertFalse(MLX.any(MLX.isNaN(decoded)).item(Bool.self), "img2img output contains NaN")

    // generateImg2Img returns (H, W, 3); QwenImageIO.imageData expects (3, H, W).
    let chw = decoded.transposed(2, 0, 1)
    let outPath = "/tmp/krea2-img2img-spike.png"
    let data = try QwenImageIO.imageData(from: chw)
    try data.write(to: URL(fileURLWithPath: outPath))
    print("Wrote \(outPath)")
  }
}
