// BridgeModelSamplingShiftTests.swift — comfybox#154
//
// The ComfyUI-protocol bridge half: a workflow carrying a
// `ModelSamplingAuraFlow` node must have its `shift` input reach the
// `GeneratePayload`, and a workflow without one must be byte-identical to
// today (no `shift`, so the model's own resolution-dependent schedule runs).

import XCTest

@testable import ZImage

final class BridgeModelSamplingShiftTests: XCTestCase {

  /// The minimum Krita/ComfyUI txt2img graph the parser accepts, plus whatever
  /// extra nodes a test wants to splice in.
  private func workflow(extraNodes: [String: Any] = [:]) -> [String: Any] {
    var nodes: [String: Any] = [
      "1": ["class_type": "CLIPTextEncode", "inputs": ["text": "a rain-lit alley"]],
      "2": ["class_type": "CLIPTextEncode", "inputs": ["text": "blurry"]],
      "3": ["class_type": "CFGGuider",
            "inputs": ["positive": ["1", 0], "negative": ["2", 0], "cfg": 5.0]],
      "4": ["class_type": "EmptySD3LatentImage",
            "inputs": ["width": 1024, "height": 1024, "batch_size": 1]],
      "5": ["class_type": "BasicScheduler",
            "inputs": ["steps": 20, "denoise": 1.0, "scheduler": "normal"]],
      "6": ["class_type": "KSamplerSelect", "inputs": ["sampler_name": "euler"]],
      "7": ["class_type": "RandomNoise", "inputs": ["noise_seed": 44821]],
      "9": ["class_type": "PreviewImage", "inputs": [:]],
    ]
    for (id, node) in extraNodes { nodes[id] = node }
    return ["prompt_id": "p-1", "client_id": "c-1", "prompt": nodes]
  }

  // MARK: - Parsing

  func testWorkflowWithoutAuraFlowNodeCarriesNoShift() throws {
    let parsed = try ComfyBridgeWorkflowParser.parse(workflow())
    XCTAssertNil(parsed.shift)
    XCTAssertNil(parsed.makeGeneratePayload(
      width: 1024, height: 1024, steps: 20, guidance: 5.0,
      negativePrompt: nil, sampler: "euler").shift)
  }

  func testModelSamplingAuraFlowShiftIsExtracted() throws {
    let parsed = try ComfyBridgeWorkflowParser.parse(workflow(extraNodes: [
      "10": ["class_type": "ModelSamplingAuraFlow",
             "inputs": ["model": ["11", 0], "shift": 3.0]],
    ]))
    XCTAssertEqual(parsed.shift, 3.0)
  }

  /// Krita/ComfyUI serialise numbers as whatever JSON gave them — an int 3
  /// must parse the same as 3.0.
  func testIntegerShiftValueParses() throws {
    let parsed = try ComfyBridgeWorkflowParser.parse(workflow(extraNodes: [
      "10": ["class_type": "ModelSamplingAuraFlow", "inputs": ["shift": 3]],
    ]))
    XCTAssertEqual(parsed.shift, 3.0)
  }

  /// A shift that is not a positive finite number is DROPPED at the parser
  /// rather than carried to a 400 the Krita user cannot see — the graph then
  /// renders on the model's own schedule, and the engine says so in its log.
  func testNonPositiveShiftIsIgnored() throws {
    for bad in [0, -1] {
      let parsed = try ComfyBridgeWorkflowParser.parse(workflow(extraNodes: [
        "10": ["class_type": "ModelSamplingAuraFlow", "inputs": ["shift": bad]],
      ]))
      XCTAssertNil(parsed.shift, "shift \(bad) should not survive parsing")
    }
  }

  /// The node with no `shift` input at all contributes nothing.
  func testAuraFlowNodeWithoutShiftInputContributesNothing() throws {
    let parsed = try ComfyBridgeWorkflowParser.parse(workflow(extraNodes: [
      "10": ["class_type": "ModelSamplingAuraFlow", "inputs": ["model": ["11", 0]]],
    ]))
    XCTAssertNil(parsed.shift)
  }

  /// Several AuraFlow nodes (a multi-pass graph): the lowest node id wins, the
  /// same deterministic rule the parser uses for schedulers and controlnets.
  func testLowestNodeIdWinsWhenSeveralAuraFlowNodes() throws {
    let parsed = try ComfyBridgeWorkflowParser.parse(workflow(extraNodes: [
      "10": ["class_type": "ModelSamplingAuraFlow", "inputs": ["shift": 3.0]],
      "11": ["class_type": "ModelSamplingAuraFlow", "inputs": ["shift": 6.0]],
    ]))
    XCTAssertEqual(parsed.shift, 3.0)
  }

  /// `ModelSamplingSD3` is read the SAME way: `ModelSamplingAuraFlow` is an
  /// SD3 subclass upstream and they differ only in a timestep `multiplier`
  /// that cancels out of the sigma grid
  /// (`ModelSamplingShiftTests.testMultiplierCancelsOutOfTheSigmaGrid`), so the
  /// same `shift` is the same schedule. Dropping it silently would have been
  /// the substitution to avoid, not reading it.
  func testModelSamplingSD3IsMappedToo() throws {
    let parsed = try ComfyBridgeWorkflowParser.parse(workflow(extraNodes: [
      "10": ["class_type": "ModelSamplingSD3", "inputs": ["shift": 3.0]],
    ]))
    XCTAssertEqual(parsed.shift, 3.0)
  }

  /// `ModelSamplingFlux` stays unread: its shift is a LOG-shift feeding
  /// `e^shift / (e^shift + 1/t − 1)`, a genuinely different curve — not the
  /// same warp under another name.
  func testModelSamplingFluxIsNotMapped() throws {
    let parsed = try ComfyBridgeWorkflowParser.parse(workflow(extraNodes: [
      "10": ["class_type": "ModelSamplingFlux", "inputs": ["shift": 3.0]],
    ]))
    XCTAssertNil(parsed.shift)
  }

  // MARK: - Payload construction

  /// The one bridge payload constructor every family arm goes through
  /// (`BridgeKrea2Arm.swift`) forwards the shift verbatim — no clamp, no
  /// family default.
  func testMakeGeneratePayloadForwardsShift() throws {
    let parsed = try ComfyBridgeWorkflowParser.parse(workflow(extraNodes: [
      "10": ["class_type": "ModelSamplingAuraFlow", "inputs": ["shift": 3.0]],
    ]))
    let payload = parsed.makeGeneratePayload(
      width: 1024, height: 1024, steps: 20, guidance: 5.0,
      negativePrompt: "blurry", sampler: "euler")
    XCTAssertEqual(payload.shift, 3.0)
    // Nothing else moved.
    XCTAssertEqual(payload.steps, 20)
    XCTAssertEqual(payload.scheduler, "euler")
    XCTAssertEqual(payload.sigmaSchedule, "normal")
  }

  // MARK: - Discovery

  /// The node is advertised so a Krita/ComfyUI client can place it, under
  /// upstream's own category and default.
  func testObjectInfoAdvertisesTheNode() throws {
    // Asserted on the SERIALIZED response — the `required` block is an
    // insertion-ordered container in memory, and what a client parses is the
    // JSON, not the Swift value.
    let data = try XCTUnwrap(orderedJSONData(ComfyBridgeObjectInfo.build()))
    let info = try XCTUnwrap(
      try JSONSerialization.jsonObject(with: data) as? [String: Any])
    let node = try XCTUnwrap(info["ModelSamplingAuraFlow"] as? [String: Any],
                             "ModelSamplingAuraFlow must be advertised")
    XCTAssertEqual(node["category"] as? String, "model/patch")
    let input = try XCTUnwrap(node["input"] as? [String: Any])
    let required = try XCTUnwrap(input["required"] as? [String: Any])
    let shift = try XCTUnwrap(required["shift"] as? [Any])
    XCTAssertEqual(shift.first as? String, "FLOAT")
    XCTAssertEqual((shift.last as? [String: Any])?["default"] as? Double, 1.73)
    XCTAssertNotNil(required["model"])
    XCTAssertEqual(node["output"] as? [String], ["MODEL"])

    // SD3 is advertised too, at ITS upstream default (3.0).
    let sd3 = try XCTUnwrap(info["ModelSamplingSD3"] as? [String: Any])
    XCTAssertEqual(sd3["category"] as? String, "model/patch")
    let sd3Shift = try XCTUnwrap(
      ((sd3["input"] as? [String: Any])?["required"] as? [String: Any])?["shift"] as? [Any])
    XCTAssertEqual((sd3Shift.last as? [String: Any])?["default"] as? Double, 3.0)
  }

  /// The imported-workflow compat report must call the node MAPPED, not glue —
  /// it stopped being "ignored safely" the moment its `shift` was read.
  func testWorkflowStoreCountsTheNodeAsMapped() {
    for node in ["ModelSamplingAuraFlow", "ModelSamplingSD3"] {
      XCTAssertTrue(WorkflowStore.mappedNodeTypes.contains(node), node)
      XCTAssertFalse(WorkflowStore.glueNodeTypes.contains(node), node)
    }
    // The one the engine deliberately does NOT read stays glue.
    XCTAssertTrue(WorkflowStore.glueNodeTypes.contains("ModelSamplingFlux"))
    XCTAssertFalse(WorkflowStore.mappedNodeTypes.contains("ModelSamplingFlux"))
  }

  // MARK: - The AuraFlow-node family gate (review r1, CRITICAL 2)

  /// The node's `shift` is a LINEAR warp. The engine's `shift` field means that
  /// on `.flux1` and something else entirely on Krea 2 — `mu`, a LOG-shift —
  /// so forwarding the node's number into a Krea 2 render would apply
  /// `e^3 ≈ 20`, not 3, under the caller's number. It is REFUSED, and the
  /// refusal names the node and the family.
  func testAuraFlowNodeIsRefusedOnEveryFamilyButZImage() throws {
    XCTAssertNil(GeneratePayload.auraFlowNodeGate(3.0, family: .flux1))
    XCTAssertNil(GeneratePayload.auraFlowNodeGate(nil, family: .krea2),
                 "a workflow with no such node is untouched on every family")

    for family in [WarmModelFamily.krea2, .flux2, .fibo, .chroma] {
      let error = try XCTUnwrap(GeneratePayload.auraFlowNodeGate(3.0, family: family),
                                "\(family.rawValue) must refuse the node, not honour it")
      let text = "\(error)"
      XCTAssertTrue(text.contains("ModelSamplingAuraFlow"), "names the node: \(text)")
      XCTAssertTrue(text.contains(family.rawValue), "names the family: \(text)")
    }
    // The krea2 refusal must explain the mu/linear mismatch, not just say no.
    let krea2 = try XCTUnwrap(GeneratePayload.auraFlowNodeGate(3.0, family: .krea2))
    XCTAssertTrue("\(krea2)".lowercased().contains("log-shift"), "\(krea2)")
  }

  // MARK: - The schedule gate (review r1, IMPORTANT 3)

  /// `applied_shift` may only report a number that reached the sigma grid, so
  /// a shift asked for alongside a schedule that ignores it is refused.
  func testShiftIsRefusedOnASchedulerThatWouldIgnoreIt() throws {
    for schedule in [SigmaScheduleKind.krea2, .bongTangent] {
      let message = try XCTUnwrap(
        GeneratePayload.validateShiftSchedule(3.0, sigmaSchedule: schedule, family: .flux1),
        "\(schedule.rawValue) ignores shift and must refuse it")
      XCTAssertTrue(message.contains("shift"), message)
      XCTAssertTrue(message.contains(schedule.rawValue), message)
    }
    for schedule in [SigmaScheduleKind.flow, .simple, .beta, .beta57, .karras, .exponential] {
      XCTAssertNil(
        GeneratePayload.validateShiftSchedule(3.0, sigmaSchedule: schedule, family: .flux1),
        "\(schedule.rawValue) reads the shift and must accept it")
    }
    // An absent schedule is `.flow`, which honours it; an absent shift is
    // never gated; and krea2 keeps its own semantics untouched.
    XCTAssertNil(GeneratePayload.validateShiftSchedule(3.0, sigmaSchedule: nil, family: .flux1))
    XCTAssertNil(GeneratePayload.validateShiftSchedule(nil, sigmaSchedule: .krea2, family: .flux1))
    XCTAssertNil(GeneratePayload.validateShiftSchedule(3.0, sigmaSchedule: .krea2, family: .krea2))
  }

  // MARK: - The family gate this shift must pass

  func testShiftIsAcceptedOnTheZImageFamilyAndRefusedElsewhere() {
    XCTAssertNil(GeneratePayload.validateShift(3.0, family: .flux1))
    XCTAssertNil(GeneratePayload.validateShift(1.15, family: .krea2))
    XCTAssertNil(GeneratePayload.validateShift(nil, family: .chroma))
    for family in [WarmModelFamily.flux2, .fibo, .chroma] {
      XCTAssertNotNil(GeneratePayload.validateShift(3.0, family: family),
                      "\(family.rawValue) must refuse shift, not ignore it")
    }
    XCTAssertNotNil(GeneratePayload.validateShift(0, family: .flux1))
    XCTAssertNotNil(GeneratePayload.validateShift(-1, family: .flux1))
    XCTAssertNotNil(GeneratePayload.validateShift(.nan, family: .flux1))
    XCTAssertNotNil(GeneratePayload.validateShift(.infinity, family: .flux1))
  }
}
