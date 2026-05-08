import XCTest
import MLX
@testable import ZImage

final class ImageLevelsTests: XCTestCase {
  func testIdentityPassDoesNotChangeInput() {
    let values: [Float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    let image = MLXArray(values, [1, 1, 1, values.count])

    let adjusted = ImageLevels.apply(image: image, min: 0.0, max: 1.0)
    MLX.eval(adjusted)

    XCTAssertEqual(adjusted.asArray(Float.self), values)
  }

  func testContrastStretch() {
    let image = MLXArray([Float(0.1), 0.5, 0.9], [1, 1, 1, 3])

    let adjusted = ImageLevels.apply(image: image, min: 0.1, max: 0.9)
    MLX.eval(adjusted)

    let result = adjusted.asArray(Float.self)
    XCTAssertEqual(result[0], 0.0, accuracy: 1e-5)
    XCTAssertEqual(result[1], 0.5, accuracy: 1e-5)
    XCTAssertEqual(result[2], 1.0, accuracy: 1e-5)
  }

  func testClampingBehavior() {
    let image = MLXArray([Float(0.0), 0.1, 0.5, 0.9, 1.0], [1, 1, 1, 5])

    let adjusted = ImageLevels.apply(image: image, min: 0.25, max: 0.75)
    MLX.eval(adjusted)

    let result = adjusted.asArray(Float.self)
    XCTAssertEqual(result[0], 0.0, accuracy: 1e-5)
    XCTAssertEqual(result[1], 0.0, accuracy: 1e-5)
    XCTAssertEqual(result[2], 0.5, accuracy: 1e-5)
    XCTAssertEqual(result[3], 1.0, accuracy: 1e-5)
    XCTAssertEqual(result[4], 1.0, accuracy: 1e-5)
  }

  func testMinEqualMaxReturnsZeros() {
    let image = MLXArray([Float(0.2), 0.4, 0.6], [1, 1, 1, 3])

    let adjusted = ImageLevels.apply(image: image, min: 0.5, max: 0.5)
    MLX.eval(adjusted)

    XCTAssertEqual(adjusted.asArray(Float.self), [0.0, 0.0, 0.0])
  }
}
