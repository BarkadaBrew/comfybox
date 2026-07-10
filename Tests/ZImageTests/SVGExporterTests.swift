import XCTest
@testable import ZImage

final class SVGExporterTests: XCTestCase {
  private let input = URL(fileURLWithPath: "/tmp/in.png")
  private let output = URL(fileURLWithPath: "/tmp/out.svg")

  func testArgumentsAlwaysIncludeInputAndOutput() {
    for preset in SVGExporter.presets {
      let args = SVGExporter.arguments(input: input, output: output, preset: preset)
      XCTAssertEqual(args[0], "--input")
      XCTAssertEqual(args[1], input.path)
      XCTAssertEqual(args[2], "--output")
      XCTAssertEqual(args[3], output.path)
    }
  }

  func testLogoPresetUsesCutoutHierarchical() {
    let args = SVGExporter.arguments(input: input, output: output, preset: "logo")
    XCTAssertTrue(args.contains("cutout"))
    XCTAssertTrue(args.contains("polygon"))
  }

  func testUnknownPresetFallsBackToDefaultArgs() {
    let unknown = SVGExporter.arguments(input: input, output: output, preset: "not-a-real-preset")
    let defaultArgs = SVGExporter.arguments(input: input, output: output, preset: "default")
    XCTAssertEqual(unknown, defaultArgs)
  }

  func testEachPresetProducesDistinctArguments() {
    let all = SVGExporter.presets.map { SVGExporter.arguments(input: input, output: output, preset: $0) }
    // "default" and "bw" both use distinct colormodes; every preset besides
    // the unrecognized-fallback case should differ from at least one other.
    XCTAssertEqual(Set(all.map { $0.joined(separator: " ") }).count, all.count)
  }

  func testConvertThrowsWhenVTracerMissing() {
    let missingPath = "/tmp/definitely-not-vtracer-\(UUID().uuidString)"
    XCTAssertFalse(SVGExporter.isAvailable(vtracerPath: missingPath))
    XCTAssertThrowsError(
      try SVGExporter.convert(input: input, output: output, preset: "default", vtracerPath: missingPath)
    ) { error in
      guard case SVGExportError.vtracerNotFound = error else {
        XCTFail("Expected .vtracerNotFound, got \(error)")
        return
      }
    }
  }
}
