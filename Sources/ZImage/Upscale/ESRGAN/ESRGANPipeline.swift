import Foundation
import Logging
import MLX
import MLXNN

#if canImport(CoreGraphics)
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

/// End-to-end ESRGAN/Real-ESRGAN image upscaling pipeline.
public final class ESRGANPipeline {
  public let model: RRDBNet
  public let config: ESRGANConfig
  public let weightsDirectory: URL
  public let logger: Logger

  public enum PipelineError: Error, CustomStringConvertible {
    case weightsDirectoryNotFound(String)
    case imageLoadFailed(String)
    case imagePreprocessFailed(Error)
    case imagePostprocessFailed(Error)
    case saveFailed(String)

    public var description: String {
      switch self {
      case .weightsDirectoryNotFound(let path):
        return "ESRGAN weights directory not found: \(path)"
      case .imageLoadFailed(let path):
        return "Failed to load input image: \(path)"
      case .imagePreprocessFailed(let error):
        return "ESRGAN image preprocessing failed: \(error)"
      case .imagePostprocessFailed(let error):
        return "ESRGAN image postprocessing failed: \(error)"
      case .saveFailed(let path):
        return "Failed to save ESRGAN output image to: \(path)"
      }
    }
  }

  public init(
    weightsDirectory: URL,
    config: ESRGANConfig? = nil,
    logger: Logger? = nil
  ) throws {
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: weightsDirectory.path, isDirectory: &isDirectory) else {
      throw PipelineError.weightsDirectoryNotFound(weightsDirectory.path)
    }

    self.weightsDirectory = weightsDirectory
    self.logger = logger ?? Logger(label: "esrgan.pipeline")
    self.config = config ?? ESRGANConfig.detect(from: weightsDirectory)

    self.logger.info("Initializing ESRGAN RRDBNet: blocks=\(self.config.numBlock), scale=\(self.config.scale), features=\(self.config.numFeat), growth=\(self.config.numGrowCh)")
    self.model = RRDBNet(config: self.config)

    self.logger.info("Loading ESRGAN weights...")
    try ESRGANWeightLoader.loadWeights(
      into: model,
      from: weightsDirectory,
      dtype: .float32,
      logger: self.logger
    )
    self.logger.info("ESRGAN pipeline ready")
  }

  public func upscale(
    imagePath: String,
    tileSize: Int = 512,
    tilePad: Int = 32
  ) throws -> CGImage {
    logger.info("ESRGAN upscaling \(imagePath) with tileSize=\(tileSize), tilePad=\(tilePad)")

    let inputImage: CGImage
    do {
      inputImage = try SeedVR2Util.loadImage(from: imagePath)
    } catch {
      throw PipelineError.imageLoadFailed(imagePath)
    }

    let input: MLXArray
    do {
      input = try Self.preprocess(inputImage)
    } catch {
      throw PipelineError.imagePreprocessFailed(error)
    }

    logger.info("Input tensor shape: \(input.shape)")
    let output: MLXArray
    if shouldTile(input, tileSize: tileSize) {
      output = upscaleTiled(input, tileSize: tileSize, tilePad: tilePad)
    } else {
      output = MLX.clip(model(input), min: 0, max: 1)
      MLX.eval(output)
    }

    do {
      let image = try Self.postprocess(output)
      logger.info("ESRGAN upscale complete: \(image.width)x\(image.height)")
      return image
    } catch {
      throw PipelineError.imagePostprocessFailed(error)
    }
  }

  @discardableResult
  public func upscaleAndSave(
    imagePath: String,
    outputPath: String? = nil,
    tileSize: Int = 512,
    tilePad: Int = 32
  ) throws -> String {
    let result = try upscale(
      imagePath: imagePath,
      tileSize: tileSize,
      tilePad: tilePad
    )

    let outPath: String
    if let outputPath {
      outPath = outputPath
    } else {
      let inputURL = URL(fileURLWithPath: imagePath)
      let baseName = inputURL.deletingPathExtension().lastPathComponent
      outPath = inputURL.deletingLastPathComponent()
        .appendingPathComponent("\(baseName)-upscaled.png").path
    }

    let outURL = URL(fileURLWithPath: outPath)
    guard let destination = CGImageDestinationCreateWithURL(
      outURL as CFURL,
      UTType.png.identifier as CFString,
      1,
      nil
    ) else {
      throw PipelineError.saveFailed(outPath)
    }

    CGImageDestinationAddImage(destination, result, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw PipelineError.saveFailed(outPath)
    }

    logger.info("Saved ESRGAN upscaled image to \(outPath)")
    return outPath
  }

  private static func preprocess(_ image: CGImage) throws -> MLXArray {
    let chw = try QwenImageIO.array(from: image, addBatchDimension: true, dtype: .float32)
    let nhwc = chw.transposed(0, 2, 3, 1)
    return MLX.clip(nhwc, min: 0, max: 1)
  }

  private static func postprocess(_ tensor: MLXArray) throws -> CGImage {
    var clipped = MLX.clip(tensor, min: 0, max: 1).asType(.float32)
    MLX.eval(clipped)
    if clipped.ndim == 3 {
      clipped = clipped.reshaped(1, clipped.dim(0), clipped.dim(1), clipped.dim(2))
    }
    precondition(clipped.ndim == 4 && clipped.dim(0) == 1 && clipped.dim(3) == 3)
    let bchw = clipped.transposed(0, 3, 1, 2)
    return try QwenImageIO.image(from: bchw)
  }

  private func shouldTile(_ input: MLXArray, tileSize: Int) -> Bool {
    guard tileSize > 0 else { return false }
    return input.dim(1) > tileSize || input.dim(2) > tileSize
  }

  private func upscaleTiled(_ input: MLXArray, tileSize: Int, tilePad: Int) -> MLXArray {
    let height = input.dim(1)
    let width = input.dim(2)
    let channels = config.numOutCh
    let scale = config.scale
    let outHeight = height * scale
    let outWidth = width * scale
    let clampedTileSize = max(1, tileSize)
    let overlap = max(0, min(tilePad, clampedTileSize - 1))
    let stride = max(1, clampedTileSize - overlap)
    let yStarts = Self.tileStarts(length: height, tileSize: clampedTileSize, stride: stride)
    let xStarts = Self.tileStarts(length: width, tileSize: clampedTileSize, stride: stride)

    logger.info("Processing ESRGAN tiles: \(xStarts.count * yStarts.count) tiles for \(width)x\(height)")

    var output = [Float](repeating: 0, count: outHeight * outWidth * channels)
    var weights = [Float](repeating: 0, count: outHeight * outWidth)

    for y0 in yStarts {
      let y1 = min(y0 + clampedTileSize, height)
      for x0 in xStarts {
        let x1 = min(x0 + clampedTileSize, width)
        let tile = input[0..., y0..<y1, x0..<x1, 0...]
        let tileOut = MLX.clip(model(tile), min: 0, max: 1).asType(.float32)
        MLX.eval(tileOut)
        accumulate(
          tileOut,
          into: &output,
          weights: &weights,
          tileX: x0,
          tileY: y0,
          outWidth: outWidth,
          channels: channels,
          scale: scale
        )
      }
    }

    for pixel in 0..<weights.count where weights[pixel] > 0 {
      let invWeight = 1.0 / weights[pixel]
      let base = pixel * channels
      for channel in 0..<channels {
        output[base + channel] *= invWeight
      }
    }

    return MLXArray(output, [1, outHeight, outWidth, channels]).asType(.float32)
  }

  private func accumulate(
    _ tileOut: MLXArray,
    into output: inout [Float],
    weights: inout [Float],
    tileX: Int,
    tileY: Int,
    outWidth: Int,
    channels: Int,
    scale: Int
  ) {
    let tileOutHeight = tileOut.dim(1)
    let tileOutWidth = tileOut.dim(2)
    let values = tileOut.asArray(Float.self)
    let outX0 = tileX * scale
    let outY0 = tileY * scale

    for y in 0..<tileOutHeight {
      let outY = outY0 + y
      for x in 0..<tileOutWidth {
        let outX = outX0 + x
        let outputPixel = outY * outWidth + outX
        let tilePixel = y * tileOutWidth + x
        weights[outputPixel] += 1.0

        let outputBase = outputPixel * channels
        let tileBase = tilePixel * channels
        for channel in 0..<channels {
          output[outputBase + channel] += values[tileBase + channel]
        }
      }
    }
  }

  private static func tileStarts(length: Int, tileSize: Int, stride: Int) -> [Int] {
    guard length > tileSize else { return [0] }
    var starts: [Int] = []
    var position = 0
    while position + tileSize < length {
      starts.append(position)
      position += stride
    }

    let finalStart = max(0, length - tileSize)
    if starts.last != finalStart {
      starts.append(finalStart)
    }
    return starts
  }
}
#endif
