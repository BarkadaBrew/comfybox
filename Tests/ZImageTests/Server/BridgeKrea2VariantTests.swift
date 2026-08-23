import XCTest

@testable import ZImage

/// WP-E19 (FDD-krea2-raw-recipe §3.5 bridge split, D13, AC-5a, AC-16a unit
/// half) — the Krita bridge's `.krea2` arm is variant-aware.
///
/// `.turbo` is byte-identical to what `296735d` built (steps clamped to 12,
/// guidance pinned 0, negative dropped). `.raw` takes what Krita actually
/// sends — steps unclamped, guidance honoured, negative prompt live — and
/// falls back to `Krea2Variant.raw` defaults (30 / 1.0) only when the
/// request carries nothing. A resident krea2 family with no known variant is
/// a fault, never "turbo". An unknown sampler name throws rather than
/// becoming euler.
///
/// The arm is a pure function (`BridgeKrea2Arm.resolve`) and the payload
/// constructor is the one `bridgeGenerate` uses for every family
/// (`ComfyBridgeGenerateRequest.makeGeneratePayload`), so AC-5a's
/// field-for-field comparison exercises the real construction with no
/// weights.
final class BridgeKrea2VariantTests: XCTestCase {

  // MARK: - Fixture: one fixed request, the shape Krita sends on every render

  private func request(
    steps: Int = 20, guidance: Float = 3.5, negativePrompt: String? = "blurry, low quality",
    sampler: String? = "euler", sigmaSchedule: String? = "normal"
  ) -> ComfyBridgeGenerateRequest {
    var r = ComfyBridgeGenerateRequest(
      promptId: "p-1", clientId: "c-1",
      prompt: "a portrait in a rain-lit alley",
      negativePrompt: negativePrompt,
      width: 1024, height: 1024,
      steps: steps, guidance: guidance,
      seed: 44821, batchSize: 1,
      outputNodeId: "9",
      sampler: sampler, sigmaSchedule: sigmaSchedule,
      levelsMin: 0.02, levelsMax: 0.98,
      inpaintImageId: nil, maskImageId: nil,
      denoise: 1.0, maskGrow: 4, maskFeather: 8, maskCropX: 16, maskCropY: 32,
      loras: [],
      controlnetModel: nil, controlnetStrength: 0.5, controlnetStart: 0.0, controlnetEnd: 1.0,
      controlImageId: nil,
      detectedModel: nil, optimizer: nil)
    r.inpaintImageData = Data([0x89, 0x50, 0x4E, 0x47])
    r.maskImageData = Data([0x01, 0x02])
    return r
  }

  /// Exactly the `GeneratePayload` `296735d`'s `bridgeGenerate` constructed
  /// for the `.krea2` arm (WarmServer.swift:3223-3228 + :3277-3297 at that
  /// commit): steps `request.steps > 0 ? min(request.steps, 12) : 9`,
  /// guidance `0.0`, negative `nil`, sampler/schedule forwarded verbatim,
  /// and every field the bridge never set left nil.
  private func expectedTurboPayload(for r: ComfyBridgeGenerateRequest) -> GeneratePayload {
    GeneratePayload(
      prompt: r.prompt,
      negativePrompt: nil,
      width: r.width, height: r.height,
      steps: r.steps > 0 ? min(r.steps, 12) : 9,
      guidance: 0.0,
      seed: r.seed,
      outputPath: nil,
      levelsMin: r.levelsMin, levelsMax: r.levelsMax,
      scheduler: r.sampler, sigmaSchedule: r.sigmaSchedule,
      inpaintImageData: r.inpaintImageData, maskData: r.maskImageData,
      denoise: r.denoise, maskGrow: r.maskGrow, maskFeather: r.maskFeather,
      maskCropX: r.maskCropX, maskCropY: r.maskCropY)
  }

  /// Field-for-field: every stored property of `GeneratePayload`, so a field
  /// the bridge starts setting (or stops setting) fails here by name.
  private func assertFieldForField(_ actual: GeneratePayload, _ expected: GeneratePayload,
                                   file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(actual.prompt, expected.prompt, "prompt", file: file, line: line)
    XCTAssertEqual(actual.negativePrompt, expected.negativePrompt, "negativePrompt", file: file, line: line)
    XCTAssertEqual(actual.width, expected.width, "width", file: file, line: line)
    XCTAssertEqual(actual.height, expected.height, "height", file: file, line: line)
    XCTAssertEqual(actual.steps, expected.steps, "steps", file: file, line: line)
    XCTAssertEqual(actual.guidance, expected.guidance, "guidance", file: file, line: line)
    XCTAssertEqual(actual.seed, expected.seed, "seed", file: file, line: line)
    XCTAssertEqual(actual.outputPath, expected.outputPath, "outputPath", file: file, line: line)
    XCTAssertEqual(actual.levelsMin, expected.levelsMin, "levelsMin", file: file, line: line)
    XCTAssertEqual(actual.levelsMax, expected.levelsMax, "levelsMax", file: file, line: line)
    XCTAssertEqual(actual.scheduler, expected.scheduler, "scheduler", file: file, line: line)
    XCTAssertEqual(actual.sigmaSchedule, expected.sigmaSchedule, "sigmaSchedule", file: file, line: line)
    XCTAssertEqual(actual.eta, expected.eta, "eta", file: file, line: line)
    XCTAssertEqual(actual.dype, expected.dype, "dype", file: file, line: line)
    XCTAssertEqual(actual.inpaintImageData, expected.inpaintImageData, "inpaintImageData", file: file, line: line)
    XCTAssertEqual(actual.maskData, expected.maskData, "maskData", file: file, line: line)
    XCTAssertEqual(actual.denoise, expected.denoise, "denoise", file: file, line: line)
    XCTAssertEqual(actual.maskGrow, expected.maskGrow, "maskGrow", file: file, line: line)
    XCTAssertEqual(actual.maskFeather, expected.maskFeather, "maskFeather", file: file, line: line)
    XCTAssertEqual(actual.maskCropX, expected.maskCropX, "maskCropX", file: file, line: line)
    XCTAssertEqual(actual.maskCropY, expected.maskCropY, "maskCropY", file: file, line: line)
    XCTAssertEqual(actual.cfg, expected.cfg, "cfg", file: file, line: line)
    XCTAssertEqual(actual.firstNStepsWithoutCFG, expected.firstNStepsWithoutCFG, "firstNStepsWithoutCFG", file: file, line: line)
    XCTAssertEqual(actual.imagePath, expected.imagePath, "imagePath", file: file, line: line)
    XCTAssertEqual(actual.initImageData, expected.initImageData, "initImageData", file: file, line: line)
    XCTAssertEqual(actual.imageStrength, expected.imageStrength, "imageStrength", file: file, line: line)
    XCTAssertEqual(actual.creativity, expected.creativity, "creativity", file: file, line: line)
    XCTAssertEqual(actual.maskPath, expected.maskPath, "maskPath", file: file, line: line)
    XCTAssertEqual(actual.maskRegion, expected.maskRegion, "maskRegion", file: file, line: line)
    XCTAssertEqual(actual.maskInvert, expected.maskInvert, "maskInvert", file: file, line: line)
    XCTAssertEqual(actual.source, expected.source, "source", file: file, line: line)
    XCTAssertEqual(actual.preset, expected.preset, "preset", file: file, line: line)
    XCTAssertEqual(actual.contentMode, expected.contentMode, "contentMode", file: file, line: line)
    XCTAssertEqual(actual.model, expected.model, "model", file: file, line: line)
    XCTAssertEqual(actual.loras?.count, expected.loras?.count, "loras", file: file, line: line)
    XCTAssertEqual(actual.controlImageData, expected.controlImageData, "controlImageData", file: file, line: line)
    XCTAssertEqual(actual.controlnetStrength, expected.controlnetStrength, "controlnetStrength", file: file, line: line)
    XCTAssertEqual(actual.controlImage, expected.controlImage, "controlImage", file: file, line: line)
    XCTAssertEqual(actual.preempt, expected.preempt, "preempt", file: file, line: line)
    XCTAssertEqual(actual.vae, expected.vae, "vae", file: file, line: line)
  }

  /// What `bridgeGenerate` does on the krea2 arm, minus the coordinator.
  private func payload(_ r: ComfyBridgeGenerateRequest, variant: Krea2Variant?) throws -> GeneratePayload {
    let resolved = try BridgeKrea2Arm.resolve(r, variant: variant)
    return r.makeGeneratePayload(
      width: r.width, height: r.height,
      steps: resolved.steps, guidance: resolved.guidance,
      negativePrompt: resolved.negativePrompt, sampler: resolved.sampler)
  }

  // MARK: - AC-5a: .turbo is byte-identical to 296735d

  func testTurboArmIsByteIdenticalToToday() throws {
    // Over the clamp: 20 → 12, guidance 3.5 → 0, negative dropped.
    let over = request(steps: 20, guidance: 3.5)
    let resolvedOver = try BridgeKrea2Arm.resolve(over, variant: .turbo)
    XCTAssertEqual(resolvedOver.variant, .turbo)
    XCTAssertEqual(resolvedOver.steps, 12)
    XCTAssertEqual(resolvedOver.guidance, 0.0)
    XCTAssertNil(resolvedOver.negativePrompt)
    XCTAssertEqual(resolvedOver.sampler, "euler")
    assertFieldForField(try payload(over, variant: .turbo), expectedTurboPayload(for: over))

    // Under the clamp: 6 → 6. Krita's default 9 → 9 (not 12).
    for steps in [6, 9, 12] {
      let r = request(steps: steps)
      XCTAssertEqual(try BridgeKrea2Arm.resolve(r, variant: .turbo).steps, steps, "steps \(steps)")
      assertFieldForField(try payload(r, variant: .turbo), expectedTurboPayload(for: r))
    }

    // Absent (0): the 9-step default, as today.
    let absent = request(steps: 0, guidance: 0)
    XCTAssertEqual(try BridgeKrea2Arm.resolve(absent, variant: .turbo).steps, 9)
    assertFieldForField(try payload(absent, variant: .turbo), expectedTurboPayload(for: absent))
  }

  // MARK: - AC-5a: .raw honours steps / CFG / negative — no clamp

  func testRawArmHonoursStepsGuidanceAndNegative() throws {
    let r = request(steps: 20, guidance: 3.5, negativePrompt: "blurry, low quality")
    let resolved = try BridgeKrea2Arm.resolve(r, variant: .raw)
    XCTAssertEqual(resolved.variant, .raw)
    XCTAssertEqual(resolved.steps, 20, "no clamp on Raw")
    XCTAssertEqual(resolved.guidance, 3.5, "CFG honoured on Raw")
    XCTAssertEqual(resolved.negativePrompt, "blurry, low quality", "negative prompt live on Raw")
    XCTAssertEqual(resolved.sampler, "euler")

    // The same request above the turbo clamp is untouched on Raw: 52 → 52.
    XCTAssertEqual(try BridgeKrea2Arm.resolve(request(steps: 52), variant: .raw).steps, 52)

    // And the payload carries exactly those values; everything else is the
    // same construction the turbo arm uses.
    let p = try payload(r, variant: .raw)
    XCTAssertEqual(p.steps, 20)
    XCTAssertEqual(p.guidance, 3.5)
    XCTAssertEqual(p.negativePrompt, "blurry, low quality")
    XCTAssertEqual(p.scheduler, "euler")
    XCTAssertEqual(p.sigmaSchedule, "normal")
    var turboShaped = expectedTurboPayload(for: r)
    turboShaped = GeneratePayload(
      prompt: turboShaped.prompt, negativePrompt: "blurry, low quality",
      width: turboShaped.width, height: turboShaped.height, steps: 20, guidance: 3.5,
      seed: turboShaped.seed, outputPath: nil,
      levelsMin: turboShaped.levelsMin, levelsMax: turboShaped.levelsMax,
      scheduler: turboShaped.scheduler, sigmaSchedule: turboShaped.sigmaSchedule,
      inpaintImageData: turboShaped.inpaintImageData, maskData: turboShaped.maskData,
      denoise: turboShaped.denoise, maskGrow: turboShaped.maskGrow, maskFeather: turboShaped.maskFeather,
      maskCropX: turboShaped.maskCropX, maskCropY: turboShaped.maskCropY)
    assertFieldForField(p, turboShaped)
  }

  func testRawArmFallsBackToVariantDefaultsOnlyWhenAbsent() throws {
    // Steps / guidance absent (Krita sends both on every render, so this is
    // the unreachable-in-practice fallback — D13): 30 / 1.0, NEVER 3.5.
    let absent = request(steps: 0, guidance: 0, negativePrompt: nil)
    let resolved = try BridgeKrea2Arm.resolve(absent, variant: .raw)
    XCTAssertEqual(resolved.steps, Krea2Variant.raw.defaultSteps)
    XCTAssertEqual(resolved.steps, 30)
    XCTAssertEqual(resolved.guidance, Krea2Variant.raw.defaultGuidance)
    XCTAssertEqual(resolved.guidance, 1.0)
    XCTAssertNil(resolved.negativePrompt, "nil in → nil out; the arm invents no negative")

    // Guidance 1.0 explicit is a value, not an absence: honoured as 1.0.
    XCTAssertEqual(try BridgeKrea2Arm.resolve(request(guidance: 1.0), variant: .raw).guidance, 1.0)
    // Guidance 2.0 with an empty negative: still 2.0 — the arm does not
    // second-guess a CFG the caller asked for.
    let cfgNoNeg = try BridgeKrea2Arm.resolve(request(guidance: 2.0, negativePrompt: nil), variant: .raw)
    XCTAssertEqual(cfgNoNeg.guidance, 2.0)
    XCTAssertNil(cfgNoNeg.negativePrompt)
  }

  // MARK: - AC-5a: an unknown sampler throws, never becomes euler

  func testUnknownSamplerThrowsOnBothVariants() throws {
    for variant in Krea2Variant.allCases {
      let r = request(sampler: "uni_pc")
      let p = try payload(r, variant: variant)
      // The arm forwards the string verbatim — it does NOT coerce it…
      XCTAssertEqual(p.scheduler, "uni_pc", "\(variant)")
      // …and the bridge's validation (the same call bridgeGenerate makes)
      // refuses it by name.
      XCTAssertThrowsError(try p.validateRecipeNames(), "\(variant)") { error in
        guard case WarmServerError.unknownSampler(let name, _) = error else {
          return XCTFail("expected unknownSampler on \(variant), got \(error)")
        }
        XCTAssertEqual(name, "uni_pc")
      }
    }
  }

  // MARK: - E5 discrepancy: nil variant on the krea2 family is a fault

  func testNilVariantIsAFaultNotTurbo() {
    XCTAssertThrowsError(try BridgeKrea2Arm.resolve(request(), variant: nil)) { error in
      guard case WarmServerError.krea2VariantUnknown = error else {
        return XCTFail("expected krea2VariantUnknown, got \(error)")
      }
      XCTAssertTrue(error.localizedDescription.lowercased().contains("variant"), error.localizedDescription)
    }
  }

  // MARK: - AC-16a (unit half): every Krita default resolves or is refused by name

  /// Krita AI Diffusion `style.py` `_scheduler_map` / `_sampler_map` values
  /// (read 2026-08-22). Each either resolves or throws a 400-class error that
  /// names the value — never a silent euler/flow. The default style
  /// (`Euler` → `euler` / `normal`) passes the bridge's full validation on
  /// both variants. Rendering end to end is the integration half.
  func testKritaStyleMatrix() throws {
    let kritaSchedulers = ["normal", "karras", "ddim_uniform", "sgm_uniform"]
    let kritaSamplers = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde_gpu", "dpmpp_sde_gpu", "uni_pc_bh2", "lcm"]

    for name in kritaSchedulers {
      do {
        let kind = try RecipeNameResolver.resolveSigmaScheduleKind(name)
        XCTAssertNotNil(kind, name)
      } catch {
        guard case WarmServerError.unknownSigmaSchedule(let offending, _) = error else {
          return XCTFail("\(name): expected unknownSigmaSchedule, got \(error)")
        }
        XCTAssertEqual(offending, name)
        XCTAssertTrue(error.localizedDescription.contains("'\(name)'"), error.localizedDescription)
      }
    }
    var refusedSamplers: [String] = []
    for name in kritaSamplers {
      do {
        let kind = try RecipeNameResolver.resolveSchedulerKind(name)
        XCTAssertNotNil(kind, name)
      } catch {
        refusedSamplers.append(name)
        guard case WarmServerError.unknownSampler(let offending, _) = error else {
          return XCTFail("\(name): expected unknownSampler, got \(error)")
        }
        XCTAssertEqual(offending, name)
        XCTAssertTrue(error.localizedDescription.contains("'\(name)'"), error.localizedDescription)
      }
    }
    // The ones we implement resolve; the rest are refused BY NAME (above),
    // never rendered as euler. Pinned so a silent coercion cannot creep back.
    XCTAssertEqual(refusedSamplers, ["euler_ancestral", "dpmpp_2m_sde_gpu", "dpmpp_sde_gpu", "uni_pc_bh2", "lcm"])
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("euler"), .euler)
    XCTAssertEqual(try RecipeNameResolver.resolveSchedulerKind("dpmpp_2m"), .dpmplusplus2m)
    // Every Krita scheduler name is one we advertise and resolve (D22).
    for name in kritaSchedulers {
      XCTAssertEqual(try RecipeNameResolver.resolveSigmaScheduleKind(name), name == "karras" ? .karras : .flow, name)
    }

    // The default style renders on both variants: the arm resolves, the
    // names validate, and the krea2 tier gates accept euler / normal.
    for variant in Krea2Variant.allCases {
      let p = try payload(request(sampler: "euler", sigmaSchedule: "normal"), variant: variant)
      let names = try p.validateRecipeNames()
      XCTAssertEqual(names.scheduler, .euler, "\(variant)")
      XCTAssertEqual(names.sigmaSchedule, .flow, "\(variant)")
      XCTAssertEqual(names.sigmaScheduleRequested, "normal", "\(variant)")
      XCTAssertNoThrow(try p.validateKrea2TierGates(names), "\(variant)")
    }
  }
}
