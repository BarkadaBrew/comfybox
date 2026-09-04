import XCTest
import MLX
@testable import ZImage

/// #298 review finding 1: readiness was judged by filename suffix alone —
/// exactly the truncated-safetensors trap intent.md warns about ("a
/// truncated safetensors file loads silently in MLX"). These pin
/// `SafetensorsIntegrity.check` against a synthetic header + short file, a
/// genuinely truncated real file, a valid file, and a missing one.
final class SafetensorsIntegrityTests: XCTestCase {

  private func tempURL(_ name: String) -> URL {
    FileManager.default.temporaryDirectory.appendingPathComponent("\(name)-\(UUID().uuidString).safetensors")
  }

  private func writeValidSafetensors(at url: URL) throws {
    let values: [Float] = [1, 2, 3, 4]
    try MLX.save(arrays: ["weight": MLXArray(values, [4]).asType(.bfloat16)], metadata: [:], url: url)
  }

  // MARK: - synthetic header + short file (the ruling's literal ask)

  func testSyntheticHeaderWithOffsetsPastFileSizeIsTruncated() throws {
    let url = tempURL("synthetic")
    defer { try? FileManager.default.removeItem(at: url) }

    // Header claims 16 bytes of tensor data; the file only carries 2.
    let header: [String: Any] = [
      "weight": ["dtype": "F32", "shape": [4], "data_offsets": [0, 16]]
    ]
    let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
    var data = Data()
    var headerLength = UInt64(headerData.count).littleEndian
    withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
    data.append(headerData)
    data.append(Data([0, 0]))
    try data.write(to: url)

    XCTAssertEqual(SafetensorsIntegrity.check(url: url), .invalid(reason: "truncated:\(url.lastPathComponent)"))
  }

  func testEmptyFileIsTruncated() throws {
    let url = tempURL("empty")
    defer { try? FileManager.default.removeItem(at: url) }
    FileManager.default.createFile(atPath: url.path, contents: Data(), attributes: nil)

    XCTAssertEqual(SafetensorsIntegrity.check(url: url), .invalid(reason: "truncated:\(url.lastPathComponent)"))
  }

  // MARK: - a real file cut short mid-payload

  func testARealFileTruncatedMidPayloadIsReportedByItsOwnName() throws {
    let url = tempURL("cut-mid-payload")
    defer { try? FileManager.default.removeItem(at: url) }
    try writeValidSafetensors(at: url)

    let full = try Data(contentsOf: url)
    try full.dropLast(4).write(to: url)  // header still claims the full tensor range

    XCTAssertEqual(SafetensorsIntegrity.check(url: url), .invalid(reason: "truncated:\(url.lastPathComponent)"))
  }

  // MARK: - passing cases

  func testAFullyWrittenFilePassesIntegrityCheck() throws {
    let url = tempURL("valid")
    defer { try? FileManager.default.removeItem(at: url) }
    try writeValidSafetensors(at: url)

    XCTAssertEqual(SafetensorsIntegrity.check(url: url), .ok)
  }

  func testAMissingFileIsInvalidNotOk() {
    let url = URL(fileURLWithPath: "/nonexistent/\(UUID().uuidString)/model.safetensors")
    guard case .invalid = SafetensorsIntegrity.check(url: url) else {
      XCTFail("a nonexistent file must not report .ok")
      return
    }
  }
}
