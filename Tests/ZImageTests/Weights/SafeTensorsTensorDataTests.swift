import XCTest
import MLX
@testable import ZImage

/// Regression tests for SafeTensorsReader.tensorData(named:): the returned
/// Data must own an independent copy of the bytes rather than pointing into
/// the reader's memory-mapped buffer.
final class SafeTensorsTensorDataTests: XCTestCase {

  // 5 floats = 20 bytes, above Data's inline-storage threshold so the
  // buffer-identity check below observes real heap allocations.
  private let tensorValues: [Float] = [1.5, 2.5, 3.5, 4.5, 5.5]

  private func writeTensorFile() throws -> URL {
    let tempDir = FileManager.default.temporaryDirectory
    let fileURL = tempDir.appendingPathComponent("tensor_data_\(UUID().uuidString).safetensors")
    let arrays: [String: MLXArray] = [
      "tensor": MLXArray(tensorValues, [tensorValues.count])
    ]
    try MLX.save(arrays: arrays, metadata: [:], url: fileURL)
    return fileURL
  }

  private func floats(from data: Data) -> [Float] {
    data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  }

  func testTensorDataReturnsExpectedBytes() throws {
    let fileURL = try writeTensorFile()
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let reader = try SafeTensorsReader(fileURL: fileURL)
    let data = try reader.tensorData(named: "tensor")

    XCTAssertEqual(data.count, tensorValues.count * MemoryLayout<Float>.size)
    XCTAssertEqual(floats(from: data), tensorValues)
  }

  func testTensorDataReturnsIndependentBuffers() throws {
    let fileURL = try writeTensorFile()
    defer { try? FileManager.default.removeItem(at: fileURL) }

    let reader = try SafeTensorsReader(fileURL: fileURL)
    let first = try reader.tensorData(named: "tensor")
    let second = try reader.tensorData(named: "tensor")

    // Two simultaneously-alive copies must not share storage with each
    // other (or with the mmapped file). The old no-copy implementation
    // returned the same interior pointer for both calls.
    let firstAddress = first.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) }
    let secondAddress = second.withUnsafeBytes { UInt(bitPattern: $0.baseAddress!) }
    XCTAssertNotEqual(firstAddress, secondAddress)
    XCTAssertEqual(first, second)
  }

  func testTensorDataOutlivesReaderAndBackingFile() throws {
    let fileURL = try writeTensorFile()

    var data: Data?
    try autoreleasepool {
      let reader = try SafeTensorsReader(fileURL: fileURL)
      data = try reader.tensorData(named: "tensor")
    }
    // Reader (and its mapped buffer) are gone; the file is deleted too.
    try FileManager.default.removeItem(at: fileURL)

    let bytes = try XCTUnwrap(data)
    XCTAssertEqual(floats(from: bytes), tensorValues)
  }
}
