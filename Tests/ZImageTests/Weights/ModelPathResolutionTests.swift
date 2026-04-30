import XCTest
@testable import ZImage

final class ModelPathResolutionTests: XCTestCase {

  func testRecognizesModelDirectoryWithoutConfigFiles() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    try makeDirectory(modelDir.appendingPathComponent("tokenizer"))
    try makeDirectory(modelDir.appendingPathComponent("transformer"))
    try makeDirectory(modelDir.appendingPathComponent("text_encoder"))
    try makeDirectory(modelDir.appendingPathComponent("vae"))

    FileManager.default.createFile(
      atPath: modelDir.appendingPathComponent("transformer/0.safetensors").path,
      contents: Data(),
      attributes: nil
    )
    FileManager.default.createFile(
      atPath: modelDir.appendingPathComponent("text_encoder/model.safetensors").path,
      contents: Data(),
      attributes: nil
    )
    FileManager.default.createFile(
      atPath: modelDir.appendingPathComponent("vae/0.safetensors").path,
      contents: Data(),
      attributes: nil
    )

    XCTAssertTrue(ZImageFiles.hasRecognizableModelDirectory(at: modelDir))
  }

  func testTextEncoderSelectionPriorityIsOverrideThenEnvThenAutoThenDefault() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let standardDir = modelDir.appendingPathComponent("text_encoder")
    let preferredDir = modelDir.appendingPathComponent("text_encoder QWen Large")
    let explicitDir = modelDir.appendingPathComponent("encoder-override")

    try makeDirectory(standardDir)
    try makeDirectory(preferredDir)
    try makeDirectory(explicitDir)

    FileManager.default.createFile(atPath: standardDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)
    FileManager.default.createFile(atPath: preferredDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)
    FileManager.default.createFile(atPath: explicitDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)

    let overrideSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: explicitDir.path,
      environment: ["ZIMAGE_ENCODER_PATH": standardDir.path]
    )
    XCTAssertEqual(overrideSelection.directory.standardizedFileURL.path, explicitDir.standardizedFileURL.path)
    XCTAssertEqual(overrideSelection.source, .overridePath)

    let envSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: ["ZIMAGE_ENCODER_PATH": standardDir.path]
    )
    XCTAssertEqual(envSelection.directory.standardizedFileURL.path, standardDir.standardizedFileURL.path)
    XCTAssertEqual(envSelection.source, .environment)

    let autoSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: [:]
    )
    XCTAssertEqual(autoSelection.directory.standardizedFileURL.path, preferredDir.standardizedFileURL.path)
    XCTAssertEqual(autoSelection.source, .autoDetectedPreferred)

    try FileManager.default.removeItem(at: preferredDir)

    let defaultSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: [:]
    )
    XCTAssertEqual(defaultSelection.directory.standardizedFileURL.path, standardDir.standardizedFileURL.path)
    XCTAssertEqual(defaultSelection.source, .defaultDirectory)
  }

  func testPromptEncodingModeDefaultsToChatTemplateForAllSelections() throws {
    // New contract (post-mosaic fix): every encoder directory routes through
    // the chat template unless explicitly opted out. Previously, any
    // non-default directory (including the auto-preferred
    // "text_encoder QWen Large") silently flipped to `.plain`, which
    // corrupted conditioning embeddings and produced mosaic artifacts.
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let standardDir = modelDir.appendingPathComponent("text_encoder")
    let preferredDir = modelDir.appendingPathComponent("text_encoder QWen Large")
    let customDir = modelDir.appendingPathComponent("encoder-override")
    try makeDirectory(standardDir)
    try makeDirectory(preferredDir)
    try makeDirectory(customDir)
    FileManager.default.createFile(atPath: standardDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)
    FileManager.default.createFile(atPath: preferredDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)
    FileManager.default.createFile(atPath: customDir.appendingPathComponent("model.safetensors").path, contents: Data(), attributes: nil)

    let defaultSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: [:]
    )
    XCTAssertEqual(defaultSelection.source, .autoDetectedPreferred,
      "with both dirs present, the preferred dir is auto-selected; that's the hot path that used to trigger the bug")
    XCTAssertEqual(
      ZImageFiles.resolvePromptEncodingMode(at: modelDir, selection: defaultSelection, environment: [:]),
      .chatTemplate,
      "auto-preferred QWen Large directory MUST now resolve to chatTemplate, not plain"
    )

    let overrideSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: customDir.path,
      environment: [:]
    )
    XCTAssertEqual(
      ZImageFiles.resolvePromptEncodingMode(at: modelDir, selection: overrideSelection, environment: [:]),
      .chatTemplate,
      "override path MUST route through chatTemplate by default"
    )

    let environmentSelection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: ["ZIMAGE_ENCODER_PATH": standardDir.path]
    )
    XCTAssertEqual(
      ZImageFiles.resolvePromptEncodingMode(at: modelDir, selection: environmentSelection, environment: [:]),
      .chatTemplate,
      "env-selected path MUST route through chatTemplate by default"
    )
  }

  func testPromptEncodingModeOptsOutToPlainWhenEnvRequests() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let standardDir = modelDir.appendingPathComponent("text_encoder")
    try makeDirectory(standardDir)
    FileManager.default.createFile(
      atPath: standardDir.appendingPathComponent("model.safetensors").path,
      contents: Data(),
      attributes: nil
    )

    let selection = ZImageFiles.resolveTextEncoderSelection(
      at: modelDir,
      overridePath: nil,
      environment: [:]
    )
    XCTAssertEqual(
      ZImageFiles.resolvePromptEncodingMode(
        at: modelDir,
        selection: selection,
        environment: ["ZIMAGE_PROMPT_MODE": "plain"]
      ),
      .plain,
      "explicit opt-out via ZIMAGE_PROMPT_MODE=plain must be honored"
    )

    XCTAssertEqual(
      ZImageFiles.resolvePromptEncodingMode(
        at: modelDir,
        selection: selection,
        environment: ["ZIMAGE_PROMPT_MODE": "PLAIN"]
      ),
      .plain,
      "opt-out is case-insensitive"
    )

    XCTAssertEqual(
      ZImageFiles.resolvePromptEncodingMode(
        at: modelDir,
        selection: selection,
        environment: ["ZIMAGE_PROMPT_MODE": "chat"]
      ),
      .chatTemplate,
      "any value other than 'plain' preserves the default chatTemplate"
    )

    XCTAssertEqual(
      ZImageFiles.resolvePromptEncodingMode(
        at: modelDir,
        selection: selection,
        environment: [:]
      ),
      .chatTemplate,
      "unset env variable preserves the default chatTemplate"
    )
  }

  func testResolveVAEWeightsSupportsGenericShardNamesAndIndex() throws {
    let modelDir = try makeTempDirectory()
    defer { try? FileManager.default.removeItem(at: modelDir) }

    let vaeDir = modelDir.appendingPathComponent("vae")
    try makeDirectory(vaeDir)

    let indexJSON = """
    {
      "weight_map": {
        "decoder.conv_in.conv.weight": "0.safetensors"
      }
    }
    """

    try indexJSON.write(to: vaeDir.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
    FileManager.default.createFile(atPath: vaeDir.appendingPathComponent("0.safetensors").path, contents: Data(), attributes: nil)

    XCTAssertEqual(ZImageFiles.resolveVAEWeights(at: modelDir), ["vae/0.safetensors"])
  }

  private func makeTempDirectory() throws -> URL {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    try makeDirectory(directory)
    return directory
  }

  private func makeDirectory(_ url: URL) throws {
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
  }
}
