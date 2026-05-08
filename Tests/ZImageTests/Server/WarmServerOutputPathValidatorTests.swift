import XCTest
@testable import ZImage

final class WarmServerOutputPathValidatorTests: XCTestCase {

  func testAllowsOutputPathInsideAllowedDirectory() throws {
    let allowedDirectory = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: allowedDirectory) }

    let outputPath = allowedDirectory.appendingPathComponent("render.png")
    let resolved = try WarmServerOutputPathValidator.resolveOutputPath(
      outputPath.path,
      allowedOutputDirectory: allowedDirectory.path
    )

    XCTAssertEqual(resolved.lastPathComponent, "render.png")
    XCTAssertEqual(
      resolved.deletingLastPathComponent().path,
      allowedDirectory.resolvingSymlinksInPath().standardizedFileURL.path
    )
  }

  func testRejectsDotDotTraversalOutsideAllowedDirectory() throws {
    let rootDirectory = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: rootDirectory) }
    let allowedDirectory = rootDirectory.appendingPathComponent("allowed")
    try FileManager.default.createDirectory(at: allowedDirectory, withIntermediateDirectories: true)

    let escapingPath = allowedDirectory.path + "/../outside.png"

    assertInvalidOutputPath {
      try WarmServerOutputPathValidator.resolveOutputPath(
        escapingPath,
        allowedOutputDirectory: allowedDirectory.path
      )
    }
  }

  func testRejectsSiblingDirectoryWithSamePrefix() throws {
    let rootDirectory = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: rootDirectory) }
    let allowedDirectory = rootDirectory.appendingPathComponent("output")
    let siblingDirectory = rootDirectory.appendingPathComponent("output-sibling")
    try FileManager.default.createDirectory(at: allowedDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: siblingDirectory, withIntermediateDirectories: true)

    let outputPath = siblingDirectory.appendingPathComponent("render.png")

    assertInvalidOutputPath {
      try WarmServerOutputPathValidator.resolveOutputPath(
        outputPath.path,
        allowedOutputDirectory: allowedDirectory.path
      )
    }
  }

  func testRejectsSymlinkEscapeFromAllowedDirectory() throws {
    let rootDirectory = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: rootDirectory) }
    let allowedDirectory = rootDirectory.appendingPathComponent("allowed")
    let outsideDirectory = rootDirectory.appendingPathComponent("outside")
    try FileManager.default.createDirectory(at: allowedDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: outsideDirectory, withIntermediateDirectories: true)

    let linkURL = allowedDirectory.appendingPathComponent("link")
    try FileManager.default.createSymbolicLink(at: linkURL, withDestinationURL: outsideDirectory)
    let outputPath = linkURL.appendingPathComponent("render.png")

    assertInvalidOutputPath {
      try WarmServerOutputPathValidator.resolveOutputPath(
        outputPath.path,
        allowedOutputDirectory: allowedDirectory.path
      )
    }
  }

  func testRejectsSymlinkDotDotEscapeFromAllowedDirectory() throws {
    let rootDirectory = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: rootDirectory) }
    let allowedDirectory = rootDirectory.appendingPathComponent("allowed")
    let outsideDirectory = rootDirectory.appendingPathComponent("outside")
    try FileManager.default.createDirectory(at: allowedDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: outsideDirectory, withIntermediateDirectories: true)

    let linkURL = allowedDirectory.appendingPathComponent("link")
    try FileManager.default.createSymbolicLink(at: linkURL, withDestinationURL: outsideDirectory)
    let outputPath = linkURL.path + "/../render.png"

    assertInvalidOutputPath {
      try WarmServerOutputPathValidator.resolveOutputPath(
        outputPath,
        allowedOutputDirectory: allowedDirectory.path
      )
    }
  }

  private func makeTempDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
      .appendingPathComponent("WarmServerOutputPathValidatorTests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
  }

  private func assertInvalidOutputPath(_ expression: () throws -> URL) {
    XCTAssertThrowsError(try expression()) { error in
      guard case WarmServerError.invalidOutputPath = error else {
        XCTFail("Expected invalidOutputPath, got \(error)")
        return
      }
    }
  }
}
