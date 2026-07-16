import XCTest
@testable import ZImage

final class WorkflowStoreTests: XCTestCase {

  private var dir: URL!
  private var store: WorkflowStore!

  override func setUpWithError() throws {
    dir = FileManager.default.temporaryDirectory
      .appendingPathComponent("workflow-store-test-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    store = WorkflowStore(directory: dir)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: dir)
  }

  private func minimalGraph() -> [String: Any] {
    [
      "1": ["class_type": "CLIPTextEncode", "inputs": ["text": "a cat"]] as [String: Any],
      "2": ["class_type": "EmptySD3LatentImage",
            "inputs": ["width": 1024, "height": 1024, "batch_size": 1]] as [String: Any],
      "3": ["class_type": "PreviewImage", "inputs": ["images": ["2", 0] as [Any]]] as [String: Any],
    ]
  }

  func testApiGraphAcceptsBareAndWrappedRejectsUIFormat() throws {
    let bare = try JSONSerialization.data(withJSONObject: minimalGraph())
    XCTAssertNoThrow(try WorkflowStore.apiGraph(fromJSON: bare))

    let wrapped = try JSONSerialization.data(withJSONObject: ["prompt": minimalGraph()])
    XCTAssertEqual(try WorkflowStore.apiGraph(fromJSON: wrapped).count, 3)

    // UI format (nodes + links arrays) gets the pointed error.
    let ui = try JSONSerialization.data(withJSONObject: ["nodes": [], "links": []] as [String: Any])
    XCTAssertThrowsError(try WorkflowStore.apiGraph(fromJSON: ui)) { error in
      guard case WorkflowError.uiFormatNotSupported = error else { return XCTFail("\(error)") }
    }

    XCTAssertThrowsError(try WorkflowStore.apiGraph(fromJSON: Data("not json".utf8)))
    let notGraph = try JSONSerialization.data(withJSONObject: ["a": 1])
    XCTAssertThrowsError(try WorkflowStore.apiGraph(fromJSON: notGraph)) { error in
      guard case WorkflowError.notAGraph = error else { return XCTFail("\(error)") }
    }
  }

  func testNormalizeStagesLoadImageAndRewritesSaveImage() throws {
    let asset = dir.appendingPathComponent("input.png")
    try Data([0x89, 0x50, 0x4E, 0x47]).write(to: asset)
    var graph = minimalGraph()
    graph["10"] = ["class_type": "LoadImage", "inputs": ["image": asset.path]] as [String: Any]
    graph["11"] = ["class_type": "SaveImage",
                   "inputs": ["images": ["10", 0] as [Any], "filename_prefix": "out"]] as [String: Any]

    var staged: [Data] = []
    let normalized = try WorkflowStore.normalizeGenericNodes(graph) { data in
      staged.append(data)
      return "cache-\(staged.count)"
    }
    XCTAssertEqual(staged.count, 1)
    let load = normalized["10"] as? [String: Any]
    XCTAssertEqual(load?["class_type"] as? String, "ETN_LoadImageCache")
    XCTAssertEqual((load?["inputs"] as? [String: Any])?["id"] as? String, "cache-1")
    let save = normalized["11"] as? [String: Any]
    XCTAssertEqual(save?["class_type"] as? String, "PreviewImage")
    XCTAssertNil((save?["inputs"] as? [String: Any])?["filename_prefix"])

    // Dry-run (nil stage): no file reads, placeholder ids.
    let dry = try WorkflowStore.normalizeGenericNodes(graph, stageImage: nil)
    let dryLoad = dry["10"] as? [String: Any]
    XCTAssertEqual((dryLoad?["inputs"] as? [String: Any])?["id"] as? String, "dryrun-10")
  }

  func testNormalizeErrors() throws {
    // Missing file with real staging.
    var graph = minimalGraph()
    graph["10"] = ["class_type": "LoadImage", "inputs": ["image": "/nonexistent-asset.png"]] as [String: Any]
    XCTAssertThrowsError(try WorkflowStore.normalizeGenericNodes(graph) { _ in "x" }) { error in
      guard case WorkflowError.inputFileMissing = error else { return XCTFail("\(error)") }
    }
    // MASK socket consumer flagged.
    var maskGraph = minimalGraph()
    maskGraph["10"] = ["class_type": "LoadImage", "inputs": ["image": "/x.png"]] as [String: Any]
    maskGraph["12"] = ["class_type": "INPAINT_ExpandMask",
                       "inputs": ["mask": ["10", 1] as [Any]]] as [String: Any]
    XCTAssertThrowsError(try WorkflowStore.normalizeGenericNodes(maskGraph, stageImage: nil)) { error in
      guard case WorkflowError.maskOutputUnsupported = error else { return XCTFail("\(error)") }
    }
  }

  func testCompatReportClassifiesNodes() {
    var graph = minimalGraph()
    graph["20"] = ["class_type": "VAEDecode", "inputs": [:] as [String: Any]] as [String: Any]
    graph["21"] = ["class_type": "SomeExoticNode", "inputs": [:] as [String: Any]] as [String: Any]
    let report = WorkflowStore.compatReport(for: graph, parses: true, parseError: nil)
    XCTAssertEqual(report.nodeCount, 5)
    XCTAssertTrue(report.mappedNodes.contains("CLIPTextEncode"))
    XCTAssertTrue(report.glueNodes.contains("VAEDecode"))
    XCTAssertEqual(report.unknownNodes, ["SomeExoticNode"])
  }

  func testStoreRoundtripListDelete() throws {
    let compat = WorkflowStore.compatReport(for: minimalGraph(), parses: true, parseError: nil)
    let workflow = StoredWorkflow(
      id: "test-id-1", name: "My Flow", importedAt: Date(), graph: minimalGraph(), compat: compat)
    try store.save(workflow)

    let loaded = try XCTUnwrap(store.get("test-id-1"))
    XCTAssertEqual(loaded.name, "My Flow")
    XCTAssertEqual(loaded.graph.count, 3)
    XCTAssertTrue(loaded.compat.parses)

    XCTAssertEqual(store.list().count, 1)
    XCTAssertTrue(store.delete("test-id-1"))
    XCTAssertNil(store.get("test-id-1"))
    XCTAssertFalse(store.delete("test-id-1"))
    // Path traversal in an id can't escape the store dir.
    XCTAssertNil(store.get("../../etc/passwd"))
  }

  func testHistoryImageExtraction() {
    let entry: [String: Any] = [
      "outputs": [
        "9": ["images": [["filename": "img-abc", "subfolder": "", "type": "output"]]] as [String: Any]
      ] as [String: Any]
    ]
    XCTAssertEqual(WarmServer.firstImageId(inHistoryEntry: entry), "img-abc")
    XCTAssertNil(WarmServer.firstImageId(inHistoryEntry: ["outputs": [:] as [String: Any]]))
  }
}
