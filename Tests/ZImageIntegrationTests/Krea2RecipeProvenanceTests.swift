// Krea2RecipeProvenanceTests.swift — WP-E10 at PRODUCTION config (FDD §5.3
// Raw batch; AC-34b, AC-60, AC-61, AC-63).
//
// Loads the real Raw checkpoint (q8, the deployed quantisation), renders at
// 1024×1024 with the variant default of 30 steps, and asserts that the
// record is READ BACK from the pipeline: the variant and file it loaded, the
// quantisation it applied, the resident VAE, the LoRA stack as bound, and
// the steps / model evaluations the loop counted. The PNG sink is read back
// off disk through ImageIO. Skipped with a named message unless the weights
// are present; never run at 256×256 or 2 steps (the two regressions this
// project already paid for came from convenient settings).
//
// AC-62 (the four sinks carry equal values for one render) needs the live
// server and is exercised by scripts/deploy-serve.sh's smoke on deploy; the
// sink plumbing itself is unit-tested in RenderRecipeTests / HealthSinkTests.

import ImageIO
import MLX
import XCTest
@testable import ZImage

final class Krea2RecipeProvenanceTests: XCTestCase {

  private static let vault = ("~/comfybox-models/loras/vault" as NSString).expandingTildeInPath
  private static let turboLoRA = vault + "/krea2_turbo_lora_rank_64_bf16.safetensors"
  private static let kromaRaw = vault + "/kroma-v0.2-base-lora-rank-384-fro-0985.safetensors"

  private func loadRaw() throws -> Krea2Pipeline {
    if ProcessInfo.processInfo.environment["CI"] != nil { throw XCTSkip("GPU test skipped in CI") }
    let paths: Krea2ModelPaths
    do {
      paths = try Krea2ModelDetection.resolve(spec: "krea2-raw")
    } catch {
      throw XCTSkip("krea2-raw not installed (\(error)) — Raw provenance batch not runnable here")
    }
    XCTAssertEqual(paths.variant, .raw)
    XCTAssertEqual(paths.transformerFile.lastPathComponent, "raw.safetensors")
    // Production quantisation (ModelPool / prepare both pass 8).
    return try Krea2Pipeline(paths: paths, quantizeTransformer: 8)
  }

  private func readUserComment(_ url: URL) throws -> [String: Any] {
    let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
    let props = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
    let exif = try XCTUnwrap(props[kCGImagePropertyExifDictionary] as? [CFString: Any])
    let comment = try XCTUnwrap(exif[kCGImagePropertyExifUserComment] as? String)
    return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(comment.utf8)) as? [String: Any])
  }

  /// AC-34b (engine half) + AC-60 + AC-61 + AC-63 on the real Raw checkpoint.
  func testRawRenderIsReadBackIntoTheRecordAndThePNG() throws {
    let k2 = try loadRaw()
    XCTAssertEqual(k2.variant, .raw)
    XCTAssertEqual(k2.quantization, "q8")
    XCTAssertEqual(k2.currentVAE.source, .modelDir)
    XCTAssertEqual(k2.currentVAE.layout, .qwenDiffusers)
    XCTAssertNil(k2.lastRunTrace, "no record before a run")

    // Two LoRAs when the vault has them (AC-60's "preset with two LoRAs");
    // each declared with its slot so the record labels it.
    var stack: [LoRAConfiguration] = []
    if FileManager.default.fileExists(atPath: Self.turboLoRA) {
      stack.append(.init(source: .local(URL(fileURLWithPath: Self.turboLoRA)), scale: 0.6, requiresBase: .raw, role: "accel"))
    }
    if FileManager.default.fileExists(atPath: Self.kromaRaw) {
      stack.append(.init(source: .local(URL(fileURLWithPath: Self.kromaRaw)), scale: 0.3, requiresBase: .raw, role: "kroma"))
    }
    let loadedStack = stack
    let loadExpectation = expectation(description: "loras")
    Task {
      do { try await k2.loadLoRAs(loadedStack) } catch { XCTFail("LoRA stack failed to load: \(error)") }
      loadExpectation.fulfill()
    }
    wait(for: [loadExpectation], timeout: 600)
    XCTAssertEqual(k2.loadedLoRAConfigs.count, stack.count)
    XCTAssertEqual(k2.loadedLoRAReports.count, stack.count)

    let steps = Krea2Variant.raw.defaultSteps  // 30
    let request = Krea2Pipeline.Request(
      prompt: "a wooden table by a window, soft morning light, a glass of water",
      negativePrompt: "blurry", guidance: 1.0, width: 1024, height: 1024, steps: steps, seed: 44821)
    let image = try k2.generate(request)
    MLX.eval(image)

    // The loop's own accounting.
    let trace = try XCTUnwrap(k2.lastRunTrace)
    XCTAssertEqual(trace.width, 1024)
    XCTAssertEqual(trace.height, 1024)
    XCTAssertEqual(trace.seed, 44821)
    XCTAssertEqual(trace.stepsRequested, steps)
    XCTAssertEqual(trace.stepsEffective, steps)
    XCTAssertEqual(trace.stepsRun, steps)
    XCTAssertEqual(trace.modelEvals, steps, "guidance 1 → one eval per step")
    XCTAssertEqual(trace.sigmas.count, steps + 1)
    XCTAssertEqual(trace.sigmas.first, 1.0)
    XCTAssertEqual(trace.sigmas.last, 0.0)
    XCTAssertEqual(trace.shiftSource, "dynamic")
    XCTAssertEqual(trace.sampler, "euler")
    XCTAssertEqual(trace.sigmaSchedule, "krea2")

    // The record, built exactly as runKrea2Generate builds it.
    let recipe = RenderRecipe.krea2(.init(
      baseModel: "krea2-raw", variant: k2.variant, transformerFile: k2.paths.transformerFile,
      quantization: k2.quantization, vae: k2.currentVAE, textEncoderFile: k2.paths.textEncoderFile,
      loras: zip(k2.loadedLoRAConfigs, k2.loadedLoRAReports).map { .init(configuration: $0, report: $1) },
      control: k2.controlLoRAActive ? k2.controlLoRAApplied : nil,
      trace: trace, requestedSigmaSchedule: nil, negativePrompt: request.negativePrompt))
    XCTAssertEqual(recipe.baseVariant, "raw")
    XCTAssertTrue(recipe.baseModelFile.hasSuffix("/raw.safetensors"), recipe.baseModelFile)
    XCTAssertEqual(recipe.quantization, "q8")
    XCTAssertEqual(recipe.vae, k2.currentVAE.file.path)
    XCTAssertEqual(recipe.vaeLayout, "qwenDiffusers")
    XCTAssertEqual(recipe.stages[0].stepsRequested, steps)
    XCTAssertEqual(recipe.stages[0].stepsRun, steps)
    XCTAssertNil(recipe.stages[0].negativePrompt, "guidance 1 → the negative did not apply")
    XCTAssertEqual(recipe.loras.count, stack.count)
    for (applied, cfg) in zip(recipe.loras, stack) {
      XCTAssertEqual(applied.scaleApplied, cfg.scale)
      XCTAssertEqual(applied.role, cfg.role)
      XCTAssertEqual(applied.relativeTo, "raw")
      XCTAssertEqual(applied.pairsBound, applied.pairsOffered, "strict apply — complete bind")
      XCTAssertEqual(applied.shapeRejected, 0)
      if cfg.role == "kroma" { XCTAssertEqual(applied.deltasApplied, 0, "D15: kroma-on-Raw carries no deltas") }
      if cfg.role == "accel" { XCTAssertEqual(applied.deltasApplied, 7, "the turbo LoRA's 7 bare deltas") }
    }

    // Sink 2: the PNG, read back off disk (AC-61).
    let out = FileManager.default.temporaryDirectory.appending(path: "krea2-provenance-\(UUID()).png")
    defer { try? FileManager.default.removeItem(at: out) }
    let metadata = QwenImageIO.ImageMetadata.generation(
      prompt: request.prompt, negativePrompt: trace.cfgActive ? request.negativePrompt : nil,
      seed: trace.seed, steps: trace.stepsRequested, guidance: trace.guidance,
      width: trace.width, height: trace.height, model: "krea2-raw", loras: k2.loadedLoRAConfigs, applied: recipe)
    try QwenImageIO.saveImage(array: image.transposed(2, 0, 1), to: out, metadata: metadata)
    let json = try readUserComment(out)
    XCTAssertNil(json["negative_prompt"])
    let applied = try XCTUnwrap(json["applied"] as? [String: Any])
    XCTAssertEqual(applied["base_variant"] as? String, "raw")
    XCTAssertTrue((applied["base_model_file"] as? String ?? "").hasSuffix("raw.safetensors"))
    XCTAssertEqual(applied["quantization"] as? String, "q8")
    XCTAssertNotNil(applied["vae"])
    XCTAssertNotNil(applied["shift"])
    let stage = try XCTUnwrap((applied["stages"] as? [[String: Any]])?.first)
    for key in ["sampler", "sigma_schedule", "steps_effective", "steps_run", "model_evals", "guidance"] {
      XCTAssertNotNil(stage[key], "PNG applied.stages[0] missing \(key)")
    }
    XCTAssertNil(stage["negative_prompt"])
    XCTAssertEqual(stage["steps_run"] as? Int, steps)
    if let first = (applied["loras"] as? [[String: Any]])?.first {
      XCTAssertNotNil(first["deltas_applied"])
    }

    // AC-60 second half: a LoRA that fails to load leaves NOTHING applied —
    // the server throws before an image or a record is written.
    let bad = LoRAConfiguration.local("/nonexistent/\(UUID()).safetensors", scale: 1.0, requiresBase: .raw)
    let failExpectation = expectation(description: "bad lora")
    Task {
      do {
        try await k2.loadLoRAs([bad])
        XCTFail("a missing LoRA must throw")
      } catch {}
      failExpectation.fulfill()
    }
    wait(for: [failExpectation], timeout: 120)
    XCTAssertTrue(k2.loadedLoRAConfigs.isEmpty, "rollback held — no partial record possible")
    XCTAssertTrue(k2.loadedLoRAReports.isEmpty)
  }
}
