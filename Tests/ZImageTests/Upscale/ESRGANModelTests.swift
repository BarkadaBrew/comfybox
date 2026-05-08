import XCTest
import MLX
@testable import ZImage

final class ESRGANModelTests: XCTestCase {
  func testResidualDenseBlockForwardPassShape() {
    let block = ResidualDenseBlock(numFeat: 64, numGrowCh: 32)
    let input = MLX.ones([1, 64, 64, 64], dtype: .float32)

    let output = block(input)
    MLX.eval(output)

    XCTAssertEqual(output.shape, [1, 64, 64, 64])
  }

  func testRRDBForwardPassShape() {
    let block = RRDB(numFeat: 64, numGrowCh: 32)
    let input = MLX.ones([1, 64, 64, 64], dtype: .float32)

    let output = block(input)
    MLX.eval(output)

    XCTAssertEqual(output.shape, [1, 64, 64, 64])
  }

  func testRRDBNetForwardPassShape() {
    let config = ESRGANConfig(
      numInCh: 3,
      numOutCh: 3,
      scale: 4,
      numFeat: 8,
      numBlock: 1,
      numGrowCh: 4
    )
    let model = RRDBNet(config: config)
    let input = MLX.ones([1, 64, 64, 3], dtype: .float32)

    let output = model(input)
    MLX.eval(output)

    XCTAssertEqual(output.shape, [1, 256, 256, 3])
  }

  func testNearestUpsample2xShapeAndValues() {
    let values = (0..<48).map(Float.init)
    let input = MLXArray(values, [1, 4, 4, 3])

    let output = RRDBNet.nearestUpsample2x(input)
    MLX.eval(output)

    XCTAssertEqual(output.shape, [1, 8, 8, 3])
    let result = output.asArray(Float.self)
    XCTAssertEqual(result[Self.nhwcIndex(y: 0, x: 0, c: 0, width: 8)], 0)
    XCTAssertEqual(result[Self.nhwcIndex(y: 1, x: 1, c: 2, width: 8)], 2)
    XCTAssertEqual(result[Self.nhwcIndex(y: 2, x: 4, c: 0, width: 8)], 18)
    XCTAssertEqual(result[Self.nhwcIndex(y: 3, x: 5, c: 2, width: 8)], 20)
  }

  func testResidualScalingIsNotIdentity() {
    let block = ResidualDenseBlock(numFeat: 4, numGrowCh: 2)
    let input = MLX.ones([1, 8, 8, 4], dtype: .float32)

    let output = block(input)
    let diff = MLX.sum(MLX.abs((output - input).asType(.float32)))
    MLX.eval(diff)

    XCTAssertGreaterThan(diff.item(Float.self), 0)
  }

  func testDetectConfigCountsBodyBlocksFromSafeTensors() throws {
    let directory = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let file = directory.appendingPathComponent("tiny.safetensors")
    let arrays: [String: MLXArray] = [
      "body.0.rdb1.conv1.weight": MLXArray.zeros([2, 4, 3, 3]),
      "body.1.rdb1.conv1.weight": MLXArray.zeros([2, 4, 3, 3])
    ]
    try MLX.save(arrays: arrays, metadata: [:], url: file)

    let config = ESRGANConfig.detect(from: directory)

    XCTAssertEqual(config.numBlock, 2)
    XCTAssertEqual(config.scale, 4)
  }

  func testWeightLoadingFromMockSafeTensors() throws {
    let directory = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }

    let config = ESRGANConfig(
      numInCh: 3,
      numOutCh: 3,
      scale: 4,
      numFeat: 4,
      numBlock: 1,
      numGrowCh: 2
    )
    let model = RRDBNet(config: config)
    let arrays = makePyTorchLayoutWeights(for: model)
    try MLX.save(arrays: arrays, metadata: [:], url: directory.appendingPathComponent("mock.safetensors"))

    try ESRGANWeightLoader.loadWeights(into: model, from: directory, dtype: .float32)

    let shapes = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
    XCTAssertEqual(shapes["conv_first.weight"], [4, 3, 3, 3])
    XCTAssertEqual(shapes["body.0.rdb1.conv1.weight"], [2, 3, 3, 4])
    XCTAssertEqual(shapes["body.0.rdb1.conv5.weight"], [4, 3, 3, 12])
    XCTAssertEqual(ESRGANConfig.detect(from: directory).numBlock, 1)
  }

  private static func nhwcIndex(y: Int, x: Int, c: Int, width: Int, channels: Int = 3) -> Int {
    (y * width + x) * channels + c
  }

  private func makeTempDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
      "esrgan-tests-\(UUID().uuidString)",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }

  private func makePyTorchLayoutWeights(for model: RRDBNet) -> [String: MLXArray] {
    var arrays: [String: MLXArray] = [:]
    for (key, value) in model.parameters().flattened() {
      if key.hasSuffix(".weight"), value.ndim == 4 {
        let shape = value.shape
        let pytorchShape = [shape[0], shape[3], shape[1], shape[2]]
        arrays[key] = MLXArray.zeros(pytorchShape).asType(.float32)
      } else {
        arrays[key] = MLXArray.zeros(value.shape).asType(.float32)
      }
    }
    return arrays
  }
}
