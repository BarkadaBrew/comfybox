import XCTest
@testable import ZImage

/// WP-E10 — `RenderRecipe` (FDD §3.10, D4, D8, D12, D15, D22; AC-60/61/64).
/// The record is built from READ-BACKS (the pipeline's loaded configs +
/// bind reports, the resident VAE selection, the run trace the loop counted)
/// through one pure builder, so "read back, not echo" is asserted here with
/// no weights: the inputs are the pipeline-side facts, never the request.
final class RenderRecipeTests: XCTestCase {

  // MARK: - Fixtures

  private func counter(steps: Int, cfg: Bool) -> Krea2RunCounter {
    let c = Krea2RunCounter()
    for _ in 0..<steps { c.eval(); if cfg { c.eval() }; c.step() }
    return c
  }

  private func trace(steps: Int = 30, guidance: Float = 1.0, startIndex: Int = 0) -> Krea2RunTrace {
    let sigmas = SigmaSchedule.krea2(numSteps: steps, mu: 0.9062)
    return Krea2RunTrace(
      width: 1024, height: 1024, seed: 44821,
      mu: 0.9062, shift: 2.475, shiftSource: "dynamic",
      sigmas: sigmas, stepsRequested: steps, startIndex: startIndex,
      guidance: guidance, denoise: 1.0,
      sampler: "euler", sigmaSchedule: "krea2",
      counter: counter(steps: steps - startIndex, cfg: guidance > 1))
  }

  private func inputs(
    guidance: Float = 1.0, negativePrompt: String? = nil, requestedSigmaSchedule: String? = nil,
    loras: [RenderRecipe.LoRAReadBack] = [], control: RenderRecipe.ControlReadBack? = nil
  ) -> RenderRecipe.Krea2Inputs {
    RenderRecipe.Krea2Inputs(
      baseModel: "krea2-raw",
      variant: .raw,
      transformerFile: URL(fileURLWithPath: "/Users/me/LocalModels/krea2-raw/raw.safetensors"),
      quantization: "q8",
      vae: Krea2VAESelection(
        file: URL(fileURLWithPath: "/Users/me/LocalModels/vae/Wan2_1_VAE_fp32.safetensors"),
        layout: .wanNative, source: .payload),
      textEncoderFile: URL(fileURLWithPath: "/Users/me/LocalModels/krea2-raw/text_encoder/model.safetensors"),
      loras: loras,
      control: control,
      trace: trace(guidance: guidance),
      requestedSigmaSchedule: requestedSigmaSchedule,
      negativePrompt: negativePrompt)
  }

  private func kromaReadBack(scale: Float = 0.3) -> RenderRecipe.LoRAReadBack {
    var cfg = LoRAConfiguration.local("/vault/kroma-v0.2-base-lora-rank-384-fro-0985.safetensors", scale: scale, requiresBase: .raw)
    cfg.role = "kroma"
    return .init(configuration: cfg, report: .init(offered: 256, bound: 256, quantizedBound: 200, deltasApplied: 0, shapeRejected: 0, unbound: []))
  }

  private func accelReadBack() -> RenderRecipe.LoRAReadBack {
    var cfg = LoRAConfiguration.local("/vault/krea2_turbo_lora_rank_64_bf16.safetensors", scale: 0.6, requiresBase: .raw)
    cfg.role = "accel"
    return .init(configuration: cfg, report: .init(offered: 264, bound: 264, quantizedBound: 264, deltasApplied: 7, shapeRejected: 0, unbound: []))
  }

  // MARK: - Physics is read back

  func testPhysicsFieldsComeFromThePipelineNotTheRequest() {
    let r = RenderRecipe.krea2(inputs())
    XCTAssertEqual(r.baseModel, "krea2-raw")
    XCTAssertEqual(r.baseVariant, "raw")
    XCTAssertTrue(r.baseModelFile.hasSuffix("/raw.safetensors"))
    XCTAssertEqual(r.quantization, "q8")
    XCTAssertTrue(r.vae.hasSuffix("/Wan2_1_VAE_fp32.safetensors"))
    XCTAssertEqual(r.vaeLayout, "wanNative")
    XCTAssertEqual(r.vaeSource, "payload")
    XCTAssertEqual(r.textEncoder, "qwen3-vl-4b/bf16")
    XCTAssertTrue(r.textEncoderFile.hasSuffix("/text_encoder/model.safetensors"))
    XCTAssertEqual(r.width, 1024)
    XCTAssertEqual(r.height, 1024)
    XCTAssertEqual(r.seed, 44821)
    XCTAssertEqual(r.mu, 0.9062)
    XCTAssertEqual(r.shift, 2.475)
    XCTAssertEqual(r.shiftSource, "dynamic")
    XCTAssertNil(r.controlLora)
  }

  /// D4: one `stages[]` entry for a single-stage render; every number in it
  /// is the loop's own count.
  func testSingleStageIsReadBackFromTheTrace() throws {
    let r = RenderRecipe.krea2(inputs())
    XCTAssertEqual(r.stages.count, 1)
    let s = try XCTUnwrap(r.stages.first)
    XCTAssertEqual(s.index, 0)
    XCTAssertEqual(s.sampler, "euler")
    XCTAssertEqual(s.sigmaSchedule, "krea2")
    XCTAssertNil(s.sigmaScheduleRequested, "nothing requested → nothing to contrast")
    XCTAssertTrue(s.shiftApplied)
    XCTAssertEqual(s.stepsRequested, 30)
    XCTAssertEqual(s.stepsEffective, 30)
    XCTAssertEqual(s.stepsRun, 30)
    XCTAssertEqual(s.modelEvals, 30)
    XCTAssertEqual(s.denoise, 1.0)
    XCTAssertEqual(s.guidance, 1.0)
    XCTAssertEqual(s.eta, 0)
    XCTAssertFalse(s.bongmath)
    XCTAssertNil(s.warmupSampler)
    XCTAssertEqual(s.warmupSteps, 0)
    XCTAssertEqual(s.sigmaHead.count, 3)
    XCTAssertEqual(s.sigmaHead.first, 1.0)
    XCTAssertEqual(s.sigmaTail.last, 0.0)
    XCTAssertEqual(s.seed, 44821)
    XCTAssertEqual(r.modelEvalsTotal, 30)
  }

  /// D22: an aliased / different schedule name the caller sent is recorded
  /// beside what ran — never silently folded.
  func testRequestedScheduleNameIsRecordedWhenItDiffers() {
    let r = RenderRecipe.krea2(inputs(requestedSigmaSchedule: "normal"))
    XCTAssertEqual(r.stages[0].sigmaSchedule, "krea2")
    XCTAssertEqual(r.stages[0].sigmaScheduleRequested, "normal")
    let same = RenderRecipe.krea2(inputs(requestedSigmaSchedule: "krea2"))
    XCTAssertNil(same.stages[0].sigmaScheduleRequested)
  }

  /// AC-61: `negative_prompt` is present only when guidance > 1 — at
  /// guidance ≤ 1 the CFG branch never ran and the prompt did not apply.
  func testNegativePromptOnlyWhenItApplied() {
    let off = RenderRecipe.krea2(inputs(guidance: 1.0, negativePrompt: "blurry"))
    XCTAssertNil(off.stages[0].negativePrompt)
    XCTAssertEqual(off.stages[0].modelEvals, 30)
    let on = RenderRecipe.krea2(inputs(guidance: 2.0, negativePrompt: "blurry"))
    XCTAssertEqual(on.stages[0].negativePrompt, "blurry")
    XCTAssertEqual(on.stages[0].guidance, 2.0)
    XCTAssertEqual(on.stages[0].modelEvals, 60, "CFG cost is visible")
    XCTAssertEqual(on.modelEvalsTotal, 60)
  }

  // MARK: - LoRAs are the loaded configs joined with their bind reports

  func testLoRAsAreJoinedFromLoadedConfigsAndReports() throws {
    let r = RenderRecipe.krea2(inputs(loras: [accelReadBack(), kromaReadBack(scale: 0.3)]))
    XCTAssertEqual(r.loras.count, 2)
    let accel = r.loras[0]
    XCTAssertEqual(accel.file, "/vault/krea2_turbo_lora_rank_64_bf16.safetensors")
    XCTAssertEqual(accel.scaleApplied, 0.6)
    XCTAssertEqual(accel.relativeTo, "raw")
    XCTAssertEqual(accel.pairsOffered, 264)
    XCTAssertEqual(accel.pairsBound, 264)
    XCTAssertEqual(accel.deltasApplied, 7)
    XCTAssertEqual(accel.shapeRejected, 0)
    XCTAssertEqual(accel.role, "accel")
    // D15: kroma-on-Raw records deltas_applied: 0 on its face, and the
    // AS-APPLIED scale (0.3 here, whatever the preset said).
    let kroma = r.loras[1]
    XCTAssertEqual(kroma.role, "kroma")
    XCTAssertEqual(kroma.scaleApplied, 0.3)
    XCTAssertEqual(kroma.deltasApplied, 0)
    XCTAssertEqual(kroma.pairsOffered, 256)
    XCTAssertEqual(kroma.pairsBound, 256)
  }

  func testUndeclaredRoleAndRelativityAreNull() {
    let cfg = LoRAConfiguration.local("/vault/some-style.safetensors", scale: 0.8)
    let r = RenderRecipe.krea2(inputs(loras: [.init(configuration: cfg, report: .init(offered: 10, bound: 10, quantizedBound: 0, deltasApplied: 0, shapeRejected: 0, unbound: []))]))
    XCTAssertNil(r.loras[0].role)
    XCTAssertNil(r.loras[0].relativeTo)
  }

  func testControlLoRAIsItsOwnEntry() throws {
    let control = RenderRecipe.ControlReadBack(
      file: URL(fileURLWithPath: "/Volumes/Bolt/Models/krea2-controlnet/depth-control-lora.safetensors"),
      scale: 0.9,
      report: .init(offered: 224, bound: 224, quantizedBound: 224, deltasApplied: 0, shapeRejected: 0, unbound: []))
    let r = RenderRecipe.krea2(inputs(control: control))
    let c = try XCTUnwrap(r.controlLora)
    XCTAssertEqual(c.role, "control")
    XCTAssertEqual(c.scaleApplied, 0.9)
    XCTAssertEqual(c.pairsBound, 224)
    XCTAssertTrue(r.loras.isEmpty, "the control adapter is not in loras[]")
  }

  // MARK: - Wire shape

  /// D8: Swift `RenderRecipe`, wire `applied`, snake_case keys — through the
  /// `.convertToSnakeCase` encoder every sink uses. Nil optionals are ABSENT
  /// (AC-61's "negative_prompt is absent" is literal).
  func testWireKeysAreSnakeCase() throws {
    let snake = JSONEncoder(); snake.keyEncodingStrategy = .convertToSnakeCase
    let r = RenderRecipe.krea2(inputs(guidance: 2.0, negativePrompt: "n", requestedSigmaSchedule: "normal", loras: [kromaReadBack()]))
    let top = try XCTUnwrap(JSONSerialization.jsonObject(with: snake.encode(r)) as? [String: Any])
    for key in ["base_model", "base_variant", "base_model_file", "quantization", "vae", "vae_layout", "vae_source",
                "text_encoder", "text_encoder_file", "loras", "width", "height", "seed",
                "mu", "shift", "shift_source", "stages", "model_evals_total"] {
      XCTAssertNotNil(top[key], "missing \(key): \(top.keys.sorted())")
    }
    XCTAssertNil(top["control_lora"], "no control LoRA → absent")
    XCTAssertNil(top["baseModel"], "no camelCase leaks onto the wire")
    let stage = try XCTUnwrap((top["stages"] as? [[String: Any]])?.first)
    for key in ["index", "sampler", "sigma_schedule", "sigma_schedule_requested", "shift_applied", "steps_requested",
                "steps_effective", "steps_run", "model_evals", "denoise", "guidance", "negative_prompt", "eta",
                "bongmath", "warmup_steps", "sigma_head", "sigma_tail", "seed"] {
      XCTAssertNotNil(stage[key], "missing stage key \(key): \(stage.keys.sorted())")
    }
    XCTAssertNil(stage["warmup_sampler"], "nil → absent")
    let lora = try XCTUnwrap((top["loras"] as? [[String: Any]])?.first)
    for key in ["file", "scale_applied", "relative_to", "pairs_offered", "pairs_bound", "shape_rejected", "deltas_applied", "role"] {
      XCTAssertNotNil(lora[key], "missing lora key \(key): \(lora.keys.sorted())")
    }
    // guidance ≤ 1 → negative_prompt absent from the stage (AC-61).
    let off = RenderRecipe.krea2(inputs(guidance: 1.0, negativePrompt: "n"))
    let offTop = try XCTUnwrap(JSONSerialization.jsonObject(with: snake.encode(off)) as? [String: Any])
    let offStage = try XCTUnwrap((offTop["stages"] as? [[String: Any]])?.first)
    XCTAssertNil(offStage["negative_prompt"])
  }

  /// The wire bytes decode back through this codebase's `.convertFromSnakeCase`
  /// decoder convention (the same way `GeneratePayload` is read).
  func testRoundTripsThroughTheSnakeCaseCoders() throws {
    let snake = JSONEncoder(); snake.keyEncodingStrategy = .convertToSnakeCase
    let fromSnake = JSONDecoder(); fromSnake.keyDecodingStrategy = .convertFromSnakeCase
    let r = RenderRecipe.krea2(inputs(guidance: 2.0, negativePrompt: "n", requestedSigmaSchedule: "normal",
                                      loras: [accelReadBack(), kromaReadBack()],
                                      control: .init(file: URL(fileURLWithPath: "/c.safetensors"), scale: 1,
                                                     report: .init(offered: 224, bound: 224, quantizedBound: 0, deltasApplied: 0, shapeRejected: 0, unbound: []))))
    let back = try fromSnake.decode(RenderRecipe.self, from: snake.encode(r))
    XCTAssertEqual(back, r)
  }

  // MARK: - Sinks

  /// AC-64: persisted pre-upgrade JSON lacking `applied` still decodes into
  /// `ImageJobStatus`; a body carrying it round-trips.
  func testImageJobStatusBackwardCompatibleDecode() throws {
    let d = JSONDecoder(); d.keyDecodingStrategy = .convertFromSnakeCase
    let old = #"{"job_id":"j1","status":"succeeded","source":"api","output_path":"/x.png","duration_ms":10,"error":null,"elapsed_ms":12}"#
    let s = try d.decode(ImageJobStatus.self, from: Data(old.utf8))
    XCTAssertEqual(s.jobId, "j1")
    XCTAssertNil(s.applied)

    let r = RenderRecipe.krea2(inputs())
    let status = ImageJobStatus(
      jobId: "j2", status: .succeeded, source: "api", outputPath: "/y.png", durationMs: 5, error: nil,
      elapsedMs: 6, preemptRefused: nil, etaSec: nil, applied: r)
    let e = JSONEncoder(); e.keyEncodingStrategy = .convertToSnakeCase
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: e.encode(status)) as? [String: Any])
    let applied = try XCTUnwrap(json["applied"] as? [String: Any])
    XCTAssertEqual(applied["base_variant"] as? String, "raw")
    let back = try d.decode(ImageJobStatus.self, from: e.encode(status))
    XCTAssertEqual(back.applied, r)
  }

  /// Sink 2: the PNG's EXIF `UserComment` JSON carries the same record under
  /// `applied`, with `negative_prompt` present only when it applied (AC-61).
  func testPNGMetadataCarriesTheRecord() throws {
    let r = RenderRecipe.krea2(inputs(guidance: 2.0, negativePrompt: "blurry", loras: [kromaReadBack()]))
    let meta = QwenImageIO.ImageMetadata.generation(
      prompt: "p", negativePrompt: "blurry", seed: 44821, steps: 30, guidance: 2.0,
      width: 1024, height: 1024, model: "krea2-raw", applied: r)
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(meta.parametersJSON).utf8)) as? [String: Any])
    XCTAssertEqual(json["negative_prompt"] as? String, "blurry")
    let applied = try XCTUnwrap(json["applied"] as? [String: Any])
    XCTAssertEqual(applied["base_variant"] as? String, "raw")
    XCTAssertEqual(applied["quantization"] as? String, "q8")
    XCTAssertTrue((applied["base_model_file"] as? String ?? "").hasSuffix("raw.safetensors"))
    XCTAssertTrue((applied["vae"] as? String ?? "").hasSuffix("Wan2_1_VAE_fp32.safetensors"))
    let stage = try XCTUnwrap((applied["stages"] as? [[String: Any]])?.first)
    for key in ["sampler", "sigma_schedule", "steps_effective", "steps_run", "model_evals", "guidance"] {
      XCTAssertNotNil(stage[key], "PNG applied.stages[0] missing \(key)")
    }
    XCTAssertEqual(stage["negative_prompt"] as? String, "blurry")
    XCTAssertNotNil(applied["shift"])
    let lora = try XCTUnwrap((applied["loras"] as? [[String: Any]])?.first)
    XCTAssertEqual(lora["deltas_applied"] as? Int, 0)

    // guidance ≤ 1: the negative prompt did not apply — absent from both the
    // top-level params and the stage record.
    let off = QwenImageIO.ImageMetadata.generation(
      prompt: "p", negativePrompt: nil, guidance: 1.0, applied: RenderRecipe.krea2(inputs(guidance: 1.0, negativePrompt: "blurry")))
    let offJSON = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(off.parametersJSON).utf8)) as? [String: Any])
    XCTAssertNil(offJSON["negative_prompt"])
    let offStage = try XCTUnwrap(((offJSON["applied"] as? [String: Any])?["stages"] as? [[String: Any]])?.first)
    XCTAssertTrue(offStage["negative_prompt"] == nil || offStage["negative_prompt"] is NSNull)
  }

  /// D12: no record for another family — `applied` is absent, never a
  /// half-filled block.
  func testOtherFamiliesEmitNoRecordInMetadata() throws {
    let meta = QwenImageIO.ImageMetadata.generation(prompt: "p", steps: 9)
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(meta.parametersJSON).utf8)) as? [String: Any])
    XCTAssertNil(json["applied"])
  }
}
