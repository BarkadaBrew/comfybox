import Foundation

/// #286 — what a named `preset` contributes to a `/v1/generate` request.
///
/// `GeneratePayload.preset` used to be a pure provenance LABEL on the image
/// path: it reached the gallery filename, the PNG metadata and the render
/// trace, and nothing else. The only place a request could set the resident
/// stack was the explicit `loras` array (`WarmServer.applyActiveLoRAs`, called
/// at dequeue when `payload.loras != nil`). So
/// `POST /v1/generate {"preset":"krea-kira-avocado"}` — the shape Kira's
/// daemon sends — rendered on whatever stack happened to be left in the warm
/// pipeline: stale adapters from an earlier `/v1/lora/swap` or an earlier
/// job's per-job override, or NOTHING at all right after a restart. Both
/// reported `success: true`.
///
/// This is the decision that closes it, kept pure so it can be tested without
/// a pipeline. Three rules, in order:
///
/// 1. **A preset is expanded as a whole or not at all.** Its `model` travels
///    with its `loras` — round 1 of review found that expanding only the
///    adapters lets a preset's LoRAs be applied to whatever base happens to be
///    active, which is worse than not expanding at all because the render then
///    *looks* successful. `steps`/`guidance` come too, but only as DECLARED on
///    the preset and only when the request omitted them.
/// 2. **The request always wins, and a contradiction is never resolved
///    silently.** Explicit `loras` stand (with ``PresetExpansion/stackMismatch``
///    raised when they disagree with the preset), and an explicit `model` that
///    contradicts the preset's is a 409 — never a preset's adapters on another
///    family.
/// 3. **A preset the engine cannot expand is a LABEL, exactly as before, and
///    says so.** No 400: an unknown id was harmless provenance for the daemon's
///    whole life, and the daemon contract is production. The render behaves as
///    it did pre-#286 and the response carries
///    ``PresetExpansion/unresolved`` naming the preset, with a warning log —
///    visible instead of silent.
public enum PresetLoRAStack: Sendable, Equatable {

  /// No `preset` on the request — nothing to decide. Byte-identical to
  /// pre-#286 behaviour, which is what swap-first clients (`/v1/lora/swap`
  /// then generate) rely on.
  case unchanged

  /// What the preset contributes. Fields left nil contribute nothing.
  case apply(PresetExpansion)

  /// 409: the request's own `model` contradicts the preset's. Applying the
  /// preset's adapters to the requested base, or the request's base under the
  /// preset's name, would both be wrong — so neither happens.
  case modelConflict(preset: String, presetModel: String, requestModel: String)

  /// What the preset store had to say about the requested name.
  public enum Lookup: Sendable, Equatable {
    /// The preset as `/v1/presets/resolve` returns it, plus the preset AS
    /// DECLARED.
    ///
    /// Both, because they differ where it matters: `ResolvedPreset` fills
    /// `steps`/`guidance` from ``PresetDefaults`` (whose `steps` default is
    /// **4**), so adopting the resolved value would drop a 52-step raw-stock
    /// render to 4 steps under the preset's name. Only a DECLARED
    /// `steps`/`guidance` is ever adopted. The LoRA stack, `kroma`, `bypass`,
    /// `model` and `media_kind` are the same in both.
    case resolved(ResolvedPreset, declared: ImagePreset)
    /// No preset with that id (`PresetStoreError.notFound`).
    case notFound
    /// Flagged invalid at load (`PresetStoreError.invalid`, WP-E20 AC-44c).
    case invalid(reason: String)
  }

  /// Decide.
  ///
  /// - Parameters:
  ///   - presetId: `payload.preset`.
  ///   - lookup: the store's answer, or nil when no preset was named.
  ///   - requestLoras: the request's own `loras`, as (filename, scale) pairs.
  ///     nil = the key was absent. An explicitly EMPTY array is a statement
  ///     ("no adapters"), not an absence, and still wins.
  ///   - requestModel: the request's own `model`.
  ///   - requestSteps / requestGuidance: present ⇒ the preset's are not adopted.
  ///   - requestVAE: the request's own `vae` (#285). Present ⇒ the preset's
  ///     declared `vae` is not adopted — same "declared, and only when the
  ///     request said nothing" rule as steps/guidance, so an undeclared VAE
  ///     is never manufactured from some other default.
  ///   - requestShift: the request's own `shift` (#154). Same rule again: a
  ///     preset's DECLARED `shift` is adopted only when the request named
  ///     none — and, additionally, only when the preset's declared family is
  ///     Z-Image, because `shift` means a different quantity on Krea 2 and
  ///     four live krea2 presets already declare one (see the expansion site
  ///     below). `ResolvedPreset.shift` has no `PresetDefaults` fallback, so
  ///     there is no manufactured-default hazard here either.
  ///   - requestRecipe: the request's own sampler-recipe fields (#419) —
  ///     `scheduler`/`sampler`, `sigma_schedule`, `eta`, `bongmath`, `stage2`,
  ///     `noise_type`, `noise_alpha`, `implicit_steps`, `c2`,
  ///     `projector_scale`. Same rule as every field above, PER FIELD: the
  ///     preset's DECLARED value is adopted only where the request carried
  ///     none. `ResolvedPreset` has no `PresetDefaults` fallback for any of
  ///     them, and they are read off `declared` anyway.
  ///   - normalizeModelSpec: how two model strings are compared — production
  ///     passes `WarmServer.parseModelSpec`, so an alias and the directory it
  ///     names are the same model, not a conflict.
  public static func decide(
    presetId: String?,
    lookup: Lookup?,
    requestLoras: [LoraReference]?,
    requestModel: String? = nil,
    requestSteps: Int? = nil,
    requestGuidance: Double? = nil,
    requestVAE: String? = nil,
    requestShift: Double? = nil,
    requestRecipe: RequestRecipe = RequestRecipe(),
    normalizeModelSpec: (String) -> String = { $0 }
  ) -> PresetLoRAStack {
    let id = presetId?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard !id.isEmpty else { return .unchanged }

    guard let lookup else {
      // Named but never looked up — a wiring mistake here would silently
      // reopen #286, so it is reported rather than shrugged off.
      return unresolved(id, "not_resolved", "preset '\(id)' was not resolved before dispatch")
    }

    let resolved: ResolvedPreset
    let declared: ImagePreset
    switch lookup {
    case .notFound:
      return unresolved(id, "unknown_preset",
        "unknown preset '\(id)' — not in this engine's store (GET /v1/presets)")
    case .invalid(let reason):
      return unresolved(id, "invalid_preset", "preset '\(id)' is flagged invalid: \(reason)")
    case .resolved(let r, let d):
      resolved = r
      declared = d
    }

    // --- Cases the engine cannot reproduce: label-only, and say why. --------

    // A video preset on the image path would push LTX adapters at a Krea 2
    // pipeline.
    if resolved.mediaKind.lowercased() == "video" {
      return unresolved(id, "media_kind:video",
        "preset '\(id)' is a video preset (media_kind \"\(resolved.mediaKind)\") "
          + "— /v1/generate is the image path")
    }

    // Round 2, finding 1: the ENGINE/PROVIDER gate. The seeded default
    // `schnell-hq` declares `engine: "mflux"` / `model: "schnell"`, and a
    // Replicate-routed preset declares a remote provider. Expanding either
    // turns `model` into a `poolLoad` of something this engine cannot load —
    // a render that fails where it used to be a harmless label.
    //
    // Read from the DECLARED preset, never from `ResolvedPreset`: the resolved
    // view fills `engine` from `PresetDefaults`, whose default is literally
    // "mflux", so every preset that simply omits the field would be refused.
    // An omitted engine/provider is not a declaration and gates nothing.
    if let engine = declared.engine?.trimmingCharacters(in: .whitespacesAndNewlines),
       !engine.isEmpty, !localEngines.contains(engine.lowercased()) {
      return unresolved(id, "engine:\(engine)",
        "preset '\(id)' declares engine '\(engine)', which is not this engine — "
          + "expanding it would ask ComfyBox to load a model it does not serve")
    }
    if let provider = declared.provider?.trimmingCharacters(in: .whitespacesAndNewlines),
       !provider.isEmpty, !localProviders.contains(provider.lowercased()) {
      return unresolved(id, "provider:\(provider)",
        "preset '\(id)' declares provider '\(provider)', which is not local — "
          + "ComfyBox renders locally or not at all")
    }

    // The bypass `.diff` adapter is a preset-schema dial the engine has no
    // application path for (the expanding sender compiles it into `loras[]`).
    if let bypass = resolved.bypass, bypass.isActive {
      return unresolved(id, "bypass_declared",
        "preset '\(id)' declares bypass.strength \(bypass.strength), which the engine "
          + "cannot expand — send the resolved stack in `loras`")
    }
    var expansion = PresetExpansion(presetId: id)

    // --- Model ------------------------------------------------------------

    let presetModel = (resolved.model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    let asked = (requestModel ?? "").trimmingCharacters(in: .whitespacesAndNewlines)

    if presetModel.isEmpty {
      // Round 2, finding 2 (C1 residual): a preset that names no model must NOT
      // hand its adapters to whatever base is resident — that is the original
      // #286 defect wearing a different hat. `custom_model_path` does not
      // count: it is stored and echoed, and the engine has never read it.
      let hint = (resolved.customModelPath ?? "").isEmpty
        ? "" : " (custom_model_path is stored for the desktop app; the engine never loads from it)"
      guard !asked.isEmpty else {
        return unresolved(id, "no_model",
          "preset '\(id)' names no model\(hint), so the engine cannot know which base its LoRA "
            + "stack belongs to — name a model on the preset, or send one on the request")
      }
      // The request named a base, but that alone is not permission: round 3
      // found that with neither `model` nor `checkpoint_family` the family
      // check was SKIPPED, so any request model at all took the preset's stack
      // — a krea2 stack onto Z-Image binds zero layers and only warns, which
      // is #286's silent-wrong-look defect again.
      //
      // Unknowable is not a match. The stack expands ONLY when the preset's
      // family is known AND the requested base's family is known AND they
      // agree.
      guard let presetFamily = declaredFamily(declared) else {
        return unresolved(id, "no_model",
          "preset '\(id)' names neither a model nor a checkpoint_family\(hint), so the engine "
            + "cannot tell whether its LoRA stack belongs on the requested '\(asked)' — "
            + "declare `model` or `checkpoint_family` on the preset")
      }
      guard let requestFamily = modelFamily(asked) else {
        return unresolved(id, "no_model",
          "preset '\(id)' names no model and the engine does not classify the requested base "
            + "'\(asked)', so its \(presetFamily) LoRA stack cannot be shown to belong there")
      }
      guard presetFamily == requestFamily else {
        return unresolved(id, "no_model",
          "preset '\(id)' names no model and its declared \(presetFamily) LoRA stack does not "
            + "belong on the requested '\(asked)' (\(requestFamily)) base")
      }
      // Both known and agreeing: expand the stack only — the request's own
      // model stands.
    } else if asked.isEmpty {
      // Flows through the request's existing model-switch semantics exactly as
      // if the client had sent it.
      expansion.model = presetModel
    } else if normalizedModel(asked, normalizeModelSpec) != normalizedModel(presetModel, normalizeModelSpec) {
      return .modelConflict(preset: id, presetModel: presetModel, requestModel: asked)
    }

    // --- The stack ---------------------------------------------------------

    // Todd 2026-09-04: kroma has no special semantics here (or anywhere else
    // in the engine) — it is a regular LoRA. `resolved.loras` is applied
    // exactly as declared, in order; no prepend, no strip, no reordering.
    // (This reverses the #350/#276-era structured-kroma expansion. See
    // `ImagePreset.migratingKromaDeprecation` in PresetStore.swift for the
    // one-release compatibility shim that keeps an already-declared
    // structured `kroma` field working as a derived, read-only view.)
    let stack = resolved.loras

    if let requestLoras {
      // Explicit `loras` keep their precedence — but a disagreement is
      // reported rather than absorbed. The production async client sends BOTH
      // `preset` and a FLAT `loras` list that has already dropped `bypass`/
      // `role`, so this is the flag that makes that visible from the
      // response.
      expansion.stackMismatch = !isSameStack(requestLoras, stack)
    } else {
      expansion.loras = stack
    }

    // --- Declared steps/guidance, only where the request said nothing. -----

    if requestSteps == nil, let steps = declared.steps { expansion.steps = steps }
    if requestGuidance == nil, let guidance = declared.guidance { expansion.guidance = guidance }

    // --- Declared VAE (#285), same rule: declared, and only when the request
    // said nothing. `declared.vae` and `resolved.vae` are always identical
    // (`ResolvedPreset.vae = preset.vae`, no `PresetDefaults` fallback for
    // it) so there is no "undeclared default" hazard the steps/guidance
    // comment above warns about — but the request still wins outright. -----

    if requestVAE == nil, let vae = declared.vae { expansion.vae = vae }

    // --- Declared schedule shift (#154) — DECLARED, only when the request
    // said nothing, and ONLY for a preset whose declared family is Z-Image.
    //
    // The family gate is not cosmetic and it is not a "safety margin": `shift`
    // means two different things on the two families that read it (the linear
    // `ModelSamplingAuraFlow` warp on Z-Image, the `ModelSamplingFlux`
    // LOG-shift `mu` on Krea 2), and four live krea2 presets in
    // `~/.comfybox/presets.json` — krea-kira, krea-kira-sfw, krea-kira-avocado,
    // krea2-base — declare `shift: 1.15` today. Before #154 nothing on the
    // image path read that value; expanding it for every family would turn it
    // into a real `mu` on Kira's production renders the moment this deploys.
    // Krea 2 keeps taking its shift from the REQUEST only, exactly as it did.
    //
    // Fails closed: `declaredFamily` answers from `checkpoint_family` first and
    // the `model` spec second, and returns nil when the preset says neither —
    // in which case nothing is adopted. A preset that wants its shift applied
    // must SAY it is Z-Image (`checkpoint_family: zimage-base` /
    // `zimage-turbo`, or a z-image model spec).
    //
    // `PresetStore.validate` already refuses a non-positive or non-finite
    // `shift`, so an adopted value is always one `validateShift` can accept.

    if requestShift == nil, let shift = declared.shift, declaredFamily(declared) == "z-image" {
      expansion.shift = shift
    }

    // --- Declared sampler recipe (#419) — DECLARED, per field, only where
    // the request said nothing. No family gate for the NAMES: sampler /
    // schedule are family-agnostic and the capability matrix, the bongmath
    // sampler gate, the `stage2` family gate and the T3 stage2.bongmath stub
    // all run at dispatch on the EXPANDED payload, so a preset naming a
    // combination its family refuses gets the same 400 an explicit request
    // does. `eta` has its own rule below (B2). `noise_type` / `noise_alpha` /
    // `implicit_steps` / `c2` / `projector_scale` are read by the Krea 2 loop
    // only; the other families do not gate them for a request today and do
    // not for a preset either. Before #419 every one of these was silently
    // ignored and the render used the engine default under the preset's
    // name. -----------------------------------------------------------------

    if requestRecipe.sampler == nil,
       let sampler = nonEmpty(declared.sampler) ?? nonEmpty(declared.scheduler) {
      expansion.sampler = sampler
    }
    if requestRecipe.sigmaSchedule == nil, let schedule = nonEmpty(declared.sigmaSchedule) {
      expansion.sigmaSchedule = schedule
    }
    // PR #420 review B2 — the daemon's #1797 rule, engine-side: on the Krea 2
    // family a PRESET-sourced non-zero eta is adopted only when the EFFECTIVE
    // stage-1 sampler (the request's, else the preset's, else the family
    // default euler) is RES4LYF; otherwise it is left off and recorded. Not
    // a 400: the preset is the only layer that asked for it, and refusing
    // the render would make every `{preset}` caller pay for a request-side
    // sampler override. A request-sourced eta is not touched here and still
    // hits `validateKrea2TierGates`. Z-Image `eta` is a different, shipped
    // parameter (DDIM η) and is adopted as declared.
    let family = declaredFamily(declared) ?? (asked.isEmpty ? nil : modelFamily(asked))
    let effectiveSamplerName = requestRecipe.sampler ?? expansion.sampler
    let effectiveSampler = resolvedSampler(effectiveSamplerName)
    if requestRecipe.eta == nil, let eta = declared.eta {
      if family == "krea2", eta != 0, !effectiveSampler.isRES4LYFFamily {
        expansion.skipped.append("eta (non-RES4LYF sampler '\(effectiveSampler.rawValue)')")
      } else {
        expansion.eta = eta
      }
    }
    if requestRecipe.bongmath == nil, let bongmath = declared.bongmath { expansion.bongmath = bongmath }
    // `stage2` is adopted as ONE object: a request that sent its own stage
    // keeps it whole (no field-wise merge across the two sources — that would
    // build a stage nobody declared). An all-nil `{}` is not a declaration.
    // PR #420 review B1: a request that switched stage 2 OFF explicitly
    // (`detail_pass: false` / `stage2: null`) is honoured and recorded.
    if let stage2 = declared.stage2, !isEmptyStage(stage2) {
      if let off = requestRecipe.stage2Off {
        expansion.skipped.append("stage2 (\(off))")
      } else if !requestRecipe.stage2Declared {
        var stage = stage2
        // B2 again, for the stage: the effective stage-2 sampler is the
        // stage's own, else the render's (`stage2Gate`'s fallback).
        let stageSampler = nonEmpty(stage.sampler).map(resolvedSampler) ?? effectiveSampler
        if family == "krea2", let eta = stage.eta, eta != 0, !stageSampler.isRES4LYFFamily {
          stage.eta = nil
          expansion.skipped.append("stage2.eta (non-RES4LYF sampler '\(stageSampler.rawValue)')")
        }
        expansion.stage2 = stage
      }
    }
    if requestRecipe.noiseType == nil, let noiseType = nonEmpty(declared.noiseType) {
      expansion.noiseType = noiseType
    }
    if requestRecipe.noiseAlpha == nil, let noiseAlpha = declared.noiseAlpha {
      expansion.noiseAlpha = noiseAlpha
    }
    if requestRecipe.implicitSteps == nil, let implicitSteps = declared.implicitSteps {
      expansion.implicitSteps = implicitSteps
    }
    if requestRecipe.c2 == nil, let c2 = declared.c2 { expansion.c2 = c2 }
    if requestRecipe.projectorScale == nil, let projectorScale = declared.projectorScale {
      expansion.projectorScale = projectorScale
    }

    return .apply(expansion)
  }

  /// #419: the request's own sampler-recipe fields, as PRESENCE — the only
  /// thing `decide` needs to know about them is whether the caller said
  /// anything, so the values are carried as the wire's own types and never
  /// interpreted here.
  public struct RequestRecipe: Sendable, Equatable {
    public var sampler: String?
    public var sigmaSchedule: String?
    public var eta: Double?
    public var bongmath: Bool?
    /// The request carried a `stage2` object (whatever it said).
    public var stage2Declared: Bool
    /// PR #420 review B1: the request switched stage 2 OFF explicitly —
    /// `"detail_pass=false"` or `"stage2=null"` (the label the skipped record
    /// carries), nil when it said nothing. A preset's stage is then not
    /// adopted, and the refusal is recorded rather than silent.
    public var stage2Off: String?
    public var noiseType: String?
    public var noiseAlpha: Double?
    public var implicitSteps: Int?
    public var c2: Double?
    public var projectorScale: Double?

    public init(
      sampler: String? = nil, sigmaSchedule: String? = nil,
      eta: Double? = nil, bongmath: Bool? = nil, stage2Declared: Bool = false,
      stage2Off: String? = nil,
      noiseType: String? = nil, noiseAlpha: Double? = nil,
      implicitSteps: Int? = nil, c2: Double? = nil, projectorScale: Double? = nil
    ) {
      self.sampler = sampler
      self.sigmaSchedule = sigmaSchedule
      self.eta = eta
      self.bongmath = bongmath
      self.stage2Declared = stage2Declared
      self.stage2Off = stage2Off
      self.noiseType = noiseType
      self.noiseAlpha = noiseAlpha
      self.implicitSteps = implicitSteps
      self.c2 = c2
      self.projectorScale = projectorScale
    }
  }

  /// A trimmed, non-empty string, or nil — a blank is not a declaration.
  static func nonEmpty(_ value: String?) -> String? {
    guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
          !trimmed.isEmpty else { return nil }
    return trimmed
  }

  /// The sampler a name resolves to for the B2 rule, with the family default
  /// (euler) for an absent name. A name the resolver does not know also reads
  /// as euler HERE only so the eta decision is total; the name itself is
  /// refused by name at the `/v1/generate` seam before anything renders.
  static func resolvedSampler(_ name: String?) -> SchedulerKind {
    guard let name else { return .euler }
    return (try? RecipeNameResolver.resolveSchedulerKind(name)) ?? .euler
  }

  /// A `stage2: {}` with nothing in it declares nothing.
  static func isEmptyStage(_ stage: PresetStage) -> Bool {
    nonEmpty(stage.sampler) == nil && nonEmpty(stage.sigmaSchedule) == nil
      && stage.steps == nil && stage.denoise == nil && stage.eta == nil && stage.bongmath == nil
  }

  // MARK: Gates

  /// Engine labels that mean "this ComfyBox process". `zimage` is what the
  /// engine's own presets and the FDD's image-preset discriminator use.
  static let localEngines: Set<String> = ["zimage", "comfybox"]
  /// Provider labels that mean "rendered here". Anything else (replicate, …)
  /// is somebody else's renderer.
  static let localProviders: Set<String> = ["local"]

  /// The checkpoint family a preset's LoRA stack belongs to, when the preset
  /// says enough to know. Declared `checkpoint_family` first (D14/O4a policy
  /// label), then the `model` spec. Never guessed from a filename.
  static func declaredFamily(_ preset: ImagePreset) -> String? {
    if let family = preset.checkpointFamily?.trimmingCharacters(in: .whitespacesAndNewlines),
       !family.isEmpty {
      if PresetStore.krea2CheckpointFamilies.contains(family) { return "krea2" }
      if PresetStore.zimageCheckpointFamilies.contains(family) { return "z-image" }
      return nil
    }
    guard let model = preset.model, !model.isEmpty else { return nil }
    return modelFamily(model)
  }

  /// The family a model spec belongs to, when it is one the engine classifies.
  static func modelFamily(_ spec: String) -> String? {
    if Krea2ModelDetection.isKnownKrea2Model(spec) { return "krea2" }
    if Krea2ModelDetection.specDirectory(spec) != nil { return "krea2" }
    if spec.lowercased().contains("z-image") || spec.lowercased().contains("zimage") {
      return "z-image"
    }
    return nil
  }

  /// Round 3 (minor 3): compare model specs with `~` expanded on both sides —
  /// `~/LocalModels/krea2-raw` and `/Users/x/LocalModels/krea2-raw` are the
  /// same base, and a 409 between them would refuse a valid request. Tilde
  /// expansion happens BEFORE `normalizeModelSpec` (an alias has no tilde) and
  /// again after it (the spec→directory table returns tilde-free paths, but a
  /// spec that is already a path passes straight through).
  static func normalizedModel(_ spec: String, _ normalize: (String) -> String) -> String {
    let expanded = (spec as NSString).expandingTildeInPath
    return (normalize(expanded) as NSString).expandingTildeInPath
  }

  private static func unresolved(_ id: String, _ code: String, _ message: String) -> PresetLoRAStack {
    .apply(PresetExpansion(
      presetId: id, unresolved: PresetExpansion.Unresolved(code: code, message: message)))
  }

  /// Are these the same stack? Compared as an ordered list of (file NAME,
  /// scale) — the request may name a LoRA by absolute path where the preset
  /// names it bare, and that is the same adapter. Roles are compared only when
  /// both sides declare one, since a flat client list carries none.
  public static func isSameStack(_ lhs: [LoraReference], _ rhs: [LoraReference]) -> Bool {
    guard lhs.count == rhs.count else { return false }
    for (a, b) in zip(lhs, rhs) {
      guard (a.filename as NSString).lastPathComponent == (b.filename as NSString).lastPathComponent
      else { return false }
      guard (a.scale * 10_000).rounded() == (b.scale * 10_000).rounded() else { return false }
      if let ra = a.role, let rb = b.role, ra != rb { return false }
    }
    return true
  }

  /// One-line summary for the render log, so a wrong stack is visible in the
  /// engine's own output and not only in the response.
  public static func describe(_ loras: [LoraReference]) -> String {
    loras.isEmpty
      ? "(none)"
      : loras.map { "\($0.filename)@\(String(format: "%.4g", $0.scale))" }.joined(separator: ", ")
  }
}

/// What a named preset contributes to one request. Every field is "contribute
/// nothing" when nil — a preset never removes what the request already said.
public struct PresetExpansion: Sendable, Equatable {
  public let presetId: String
  /// The stack to apply. nil = the request brought its own, or the preset
  /// could not be expanded.
  public var loras: [LoraReference]?
  /// The base to render on. nil = the request named one, or the preset did not.
  public var model: String?
  /// DECLARED steps/guidance, adopted only where the request omitted them.
  public var steps: Int?
  public var guidance: Double?
  /// #285: the preset's declared `vae`, adopted only where the request
  /// omitted its own. nil = the request named one, the preset declared none,
  /// or the preset could not be expanded.
  public var vae: String?
  /// #154: the preset's declared schedule `shift`, adopted only where the
  /// request omitted its own. On the Z-Image family this is ComfyUI's
  /// `ModelSamplingAuraFlow` linear shift; on Krea 2 it is `mu`
  /// (`ModelSamplingFlux`). Same field, family-dependent meaning — the
  /// per-family gate is `GeneratePayload.validateShift(_:family:)`.
  public var shift: Double?
  /// #419: the preset's declared stage-1 SAMPLER RECIPE, each field adopted
  /// only where the request omitted its own — the `shift` rule, applied to
  /// every other dial a preset can declare. Unlike `shift` there is no
  /// family gate at expansion for the NAMES: sampler / schedule are
  /// family-agnostic and the family capability matrix, the bongmath sampler
  /// gate and the `stage2` family gate all run at DISPATCH on the expanded
  /// payload, so a preset declaring a combination its family refuses gets the
  /// same 400 an explicit request does. `eta` is the one field with an
  /// expansion-time rule (``skipped``, PR #420 review B2). `noise_type` /
  /// `noise_alpha` / `implicit_steps` / `c2` / `projector_scale` are Krea 2
  /// dials that the other families' loops do not read — for a request OR a
  /// preset, today, unchanged here.
  ///
  /// `sampler` is `declared.sampler ?? declared.scheduler` — the legacy
  /// `scheduler` key is the same field under its older name (the daemon
  /// still reads it), and a preset that carries only that spelling still
  /// names a sampler.
  public var sampler: String?
  public var sigmaSchedule: String?
  public var eta: Double?
  public var bongmath: Bool?
  /// The preset's declared second stage, AS DECLARED (`PresetStage`, every
  /// field optional). The `/v1/generate` seam turns it into the wire's
  /// `Stage2Payload`, which REQUIRES `steps` and `denoise` — a preset that
  /// declares a stage without both is refused there (400), never completed
  /// with a guess. An all-nil `stage2: {}` is treated as undeclared.
  public var stage2: PresetStage?
  public var noiseType: String?
  public var noiseAlpha: Double?
  public var implicitSteps: Int?
  public var c2: Double?
  public var projectorScale: Double?
  /// PR #420 review (B1/B2): declared recipe fields the expansion decided
  /// NOT to adopt, each with its reason — never a silent drop. Reaches the
  /// response as `preset_recipe_skipped`. Two sources today: the request
  /// switched stage 2 off explicitly (`detail_pass: false` / `stage2: null`)
  /// while the preset declares one, and the daemon's #1797 rule mirrored
  /// engine-side — a PRESET-sourced non-zero `eta` (stage 1 or 2) whose
  /// EFFECTIVE sampler (request's, else preset's; stage 2 falls back to
  /// stage 1) is not RES4LYF on the Krea 2 family is left off rather than
  /// turned into a 400, because the preset is the only layer that asked for
  /// it. A REQUEST-sourced eta is untouched and still hits the existing gate.
  public var skipped: [String] = []
  /// C2: the engine could not expand this preset. It behaves as the label it
  /// always was, and this reaches the response as `preset_unresolved` (the
  /// preset's name) plus `preset_unresolved_reason` (the machine-readable
  /// code).
  public var unresolved: Unresolved?

  /// Why a preset stayed a label.
  ///
  /// `code` is short and machine-readable so a daemon can branch on it:
  /// `unknown_preset`, `invalid_preset`, `media_kind:video`, `engine:<x>`,
  /// `provider:<x>`, `no_model`, `bypass_declared`,
  /// `missing_lora:<name>`, `not_resolved`. `message` is the full sentence
  /// that goes in the engine log.
  public struct Unresolved: Sendable, Equatable {
    public let code: String
    public let message: String
    public init(code: String, message: String) {
      self.code = code
      self.message = message
    }
  }
  /// I1: the request's explicit `loras` differ from the preset's resolved
  /// stack. Explicit still wins; this reaches the response as
  /// `preset_stack_mismatch`.
  public var stackMismatch: Bool

  public init(
    presetId: String, loras: [LoraReference]? = nil, model: String? = nil,
    steps: Int? = nil, guidance: Double? = nil, vae: String? = nil,
    shift: Double? = nil,
    sampler: String? = nil, sigmaSchedule: String? = nil,
    eta: Double? = nil, bongmath: Bool? = nil, stage2: PresetStage? = nil,
    noiseType: String? = nil, noiseAlpha: Double? = nil,
    implicitSteps: Int? = nil, c2: Double? = nil, projectorScale: Double? = nil,
    unresolved: Unresolved? = nil,
    stackMismatch: Bool = false
  ) {
    self.presetId = presetId
    self.loras = loras
    self.model = model
    self.steps = steps
    self.guidance = guidance
    self.vae = vae
    self.shift = shift
    self.sampler = sampler
    self.sigmaSchedule = sigmaSchedule
    self.eta = eta
    self.bongmath = bongmath
    self.stage2 = stage2
    self.noiseType = noiseType
    self.noiseAlpha = noiseAlpha
    self.implicitSteps = implicitSteps
    self.c2 = c2
    self.projectorScale = projectorScale
    self.unresolved = unresolved
    self.stackMismatch = stackMismatch
  }

  /// #419: the wire keys `sampler`… above adopt, in the order the response's
  /// `preset_recipe_applied` lists them. One place, so the expansion seam,
  /// the replay rewrite and the tests agree on the spelling.
  public static let recipeWireKeys: [String] = [
    "scheduler", "sigma_schedule", "eta", "bongmath", "stage2",
    "noise_type", "noise_alpha", "implicit_steps", "c2", "projector_scale",
  ]
}

// MARK: - The `/v1/generate` seam

extension GeneratePayload {

  /// #286 — return `payload` with its named `preset` expanded, or throw a 409.
  ///
  /// This is the ONE place a preset becomes a stack on the image path. It runs
  /// in `WarmServer.decodedGeneratePayload`, so `/v1/generate`,
  /// `/v1/generate/async` and persisted-queue replay all go through it, and the
  /// existing per-job model/LoRA application at dequeue does the actual work —
  /// no second application path to drift out of sync.
  ///
  /// - Parameters:
  ///   - resolve: the preset store lookup. Injected so the seam is testable
  ///     without a warm pipeline; production passes `PresetStore.lookup`, which
  ///     reads the preset and its validity flag under one lock — the same read
  ///     `POST /v1/presets/resolve` makes.
  ///   - normalizeModelSpec: `WarmServer.parseModelSpec` in production.
  ///   - log: warnings and the expanded stack, so this is visible in the
  ///     engine's own output and not only in the response.
  static func expandingPreset(
    _ payload: GeneratePayload,
    resolve: (String) -> PresetLoRAStack.Lookup,
    normalizeModelSpec: (String) -> String = { $0 },
    log: (String) -> Void = { _ in }
  ) throws -> GeneratePayload {
    var out = payload
    let id = payload.preset?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    guard !id.isEmpty else { return out }

    let decision = PresetLoRAStack.decide(
      presetId: id,
      lookup: resolve(id),
      requestLoras: payload.loras?.map {
        LoraReference(filename: $0.path, scale: Double($0.scale ?? 1.0), role: $0.role)
      },
      requestModel: payload.model,
      requestSteps: payload.steps,
      requestGuidance: payload.guidance.map(Double.init),
      requestVAE: payload.vae,
      requestShift: payload.shift.map(Double.init),
      requestRecipe: PresetLoRAStack.RequestRecipe(
        sampler: payload.scheduler, sigmaSchedule: payload.sigmaSchedule,
        eta: payload.eta.map(Double.init), bongmath: payload.bongmath,
        stage2Declared: payload.stage2 != nil,
        stage2Off: payload.stage2ExplicitlyOff,
        noiseType: payload.noiseType, noiseAlpha: payload.noiseAlpha.map(Double.init),
        implicitSteps: payload.implicitSteps, c2: payload.c2.map(Double.init),
        projectorScale: payload.projectorScale.map(Double.init)),
      normalizeModelSpec: normalizeModelSpec)

    switch decision {
    case .unchanged:
      return out

    case .modelConflict(let preset, let presetModel, let requestModel):
      throw WarmServerError.presetModelConflict(
        preset: preset, presetModel: presetModel, requestModel: requestModel)

    case .apply(let expansion):
      if let reason = expansion.unresolved {
        return out.asUnresolvedPreset(expansion.presetId, reason, log: log)
      }
      if let loras = expansion.loras {
        out.loras = loras.map { LoRAEntry(path: $0.filename, scale: Float($0.scale), role: $0.role) }
        // #282: mark the stack PRESET-owned. `loras` is one field with two
        // possible owners, and the dequeue resolver must be able to tell them
        // apart to report an honest `lora_stack_origin` — and, more to the
        // point, so the warm default can never displace a preset's stack.
        out.presetStackApplied = true
        log("Preset '\(expansion.presetId)': applying its resolved LoRA stack — "
          + PresetLoRAStack.describe(loras))
      }
      if expansion.stackMismatch {
        out.presetStackMismatch = true
        log("WARNING: preset '\(expansion.presetId)' resolves to a different LoRA stack than the "
          + "explicit `loras` on this request — the explicit list wins (response carries "
          + "preset_stack_mismatch)")
      }
      if let model = expansion.model {
        out.model = model
        log("Preset '\(expansion.presetId)': rendering on its declared model '\(model)'")
      }
      if let steps = expansion.steps { out.steps = steps }
      if let guidance = expansion.guidance { out.guidance = Float(guidance) }
      if let shift = expansion.shift {
        // #154: no `presetShiftApplied` twin of `presetVAEApplied` — nothing
        // downstream needs to tell a preset-owned shift from a request-owned
        // one (the VAE flag exists only so `Krea2VAESelector` can record
        // `vae_source`). The response's `applied_shift` reports the value that
        // reached the sigma grid whichever source it came from, which is the
        // question a caller actually asks.
        out.shift = Float(shift)
        log("Preset '\(expansion.presetId)': applying its declared schedule shift \(shift)")
      }
      if let vae = expansion.vae {
        out.vae = vae
        // #285: the ONLY way `Krea2VAESelector.resolve` (called later, at
        // dispatch, once `payload.vae` and `payload.preset` have long since
        // collapsed onto the same field) can record `vae_source: "preset"`
        // instead of `"payload"` — mirrors `presetStackApplied` for LoRAs.
        out.presetVAEApplied = true
        log("Preset '\(expansion.presetId)': applying its declared VAE '\(vae)'")
      }
      try out.applyPresetRecipe(expansion, log: log)
      return out
    }
  }

  /// #419: put the preset's declared sampler recipe on the payload — the
  /// fields `decide` adopted (request > preset, per field) — and RECORD which
  /// ones, so the response's `preset_recipe_applied` can say where each dial
  /// came from the way `vae_source: "preset"` and `lora_stack_origin` do.
  ///
  /// Two things are refused HERE, with a 400 naming the preset, rather than
  /// left for dispatch: a sampler / schedule name the engine does not resolve
  /// (the request's own names were validated BEFORE expansion, so a preset's
  /// would otherwise slip past that check until the dequeue re-check — and
  /// the daemon should learn which preset is broken, not just which name),
  /// and a declared `stage2` missing `steps` or `denoise` (the wire's
  /// `Stage2Payload` requires both because they decide the stretched grid;
  /// a default for either would be an engine-invented recipe — exactly what
  /// `Stage2Payload.init(from:)` refuses on the wire).
  private mutating func applyPresetRecipe(
    _ expansion: PresetExpansion, log: (String) -> Void
  ) throws {
    let id = expansion.presetId
    var applied: [String] = []

    if let sampler = expansion.sampler {
      _ = try Self.presetName(id, field: "sampler") {
        try RecipeNameResolver.resolveSchedulerKind(sampler)
      }
      scheduler = sampler
      applied.append("scheduler")
    }
    if let schedule = expansion.sigmaSchedule {
      _ = try Self.presetName(id, field: "sigma_schedule") {
        try RecipeNameResolver.resolveSigmaScheduleKind(schedule)
      }
      sigmaSchedule = schedule
      applied.append("sigma_schedule")
    }
    if let eta = expansion.eta { self.eta = Float(eta); applied.append("eta") }
    if let bongmath = expansion.bongmath { self.bongmath = bongmath; applied.append("bongmath") }
    if let stage = expansion.stage2 {
      var missing: [String] = []
      if stage.steps == nil { missing.append("steps") }
      if stage.denoise == nil { missing.append("denoise") }
      guard let steps = stage.steps, let denoise = stage.denoise else {
        throw WarmServerError.presetRecipeInvalid(
          preset: id, field: "stage2",
          reason: "the preset declares a second stage without " + missing.joined(separator: " and ")
            + " — both decide the stretched grid and have no default, so the engine will not "
            + "guess them. Declare stage2.steps and stage2.denoise on the preset, or send "
            + "`stage2` on the request")
      }
      let stageSampler = PresetLoRAStack.nonEmpty(stage.sampler)
      let stageSchedule = PresetLoRAStack.nonEmpty(stage.sigmaSchedule)
      _ = try Self.presetName(id, field: "stage2.sampler") {
        try RecipeNameResolver.resolveSchedulerKind(stageSampler)
      }
      _ = try Self.presetName(id, field: "stage2.sigma_schedule") {
        try RecipeNameResolver.resolveSigmaScheduleKind(stageSchedule)
      }
      stage2 = Stage2Payload(
        steps: steps, denoise: denoise, scheduler: stageSampler, sigmaSchedule: stageSchedule,
        eta: stage.eta.map(Float.init), bongmath: stage.bongmath)
      applied.append("stage2")
    }
    if let noiseType = expansion.noiseType { self.noiseType = noiseType; applied.append("noise_type") }
    if let noiseAlpha = expansion.noiseAlpha {
      self.noiseAlpha = Float(noiseAlpha); applied.append("noise_alpha")
    }
    if let implicitSteps = expansion.implicitSteps {
      self.implicitSteps = implicitSteps; applied.append("implicit_steps")
    }
    if let c2 = expansion.c2 { self.c2 = Float(c2); applied.append("c2") }
    if let projectorScale = expansion.projectorScale {
      self.projectorScale = Float(projectorScale); applied.append("projector_scale")
    }

    // PR #420 review B1: `detail_pass: true` with no stage of its own asks
    // for THE PRESET's second stage. The preset declares none (or the request
    // also said `stage2: null`, a contradiction `detailPassGate` names) — a
    // 400 naming the preset, never a silently single-stage render.
    if detailPass == true, stage2 == nil, stage2ExplicitlyOff == nil {
      throw WarmServerError.presetRecipeInvalid(
        preset: id, field: "stage2",
        reason: "`detail_pass: true` asks for the preset's second stage, and preset '\(id)' "
          + "declares no stage2. Declare stage2 (steps + denoise) on the preset, send `stage2` "
          + "on the request, or drop detail_pass")
    }

    if !expansion.skipped.isEmpty {
      presetRecipeSkipped = expansion.skipped
      for entry in expansion.skipped {
        log("Preset '\(id)': NOT applying its declared \(entry) — recorded in preset_recipe_skipped")
      }
    }

    guard !applied.isEmpty else { return }
    presetRecipeApplied = applied
    let summary = [
      expansion.sampler.map { "sampler=\($0)" },
      expansion.sigmaSchedule.map { "sigma_schedule=\($0)" },
      expansion.eta.map { "eta=\($0)" },
      expansion.bongmath.map { "bongmath=\($0)" },
      expansion.stage2.map { "stage2={steps: \($0.steps ?? 0), denoise: \($0.denoise ?? 0)}" },
      expansion.noiseType.map { "noise_type=\($0)" },
      expansion.noiseAlpha.map { "noise_alpha=\($0)" },
      expansion.implicitSteps.map { "implicit_steps=\($0)" },
      expansion.c2.map { "c2=\($0)" },
      expansion.projectorScale.map { "projector_scale=\($0)" },
    ].compactMap { $0 }.joined(separator: " ")
    log("Preset '\(id)': applying its declared sampler recipe — \(summary) "
      + "(request fields, where sent, took precedence)")
  }

  /// Resolve one preset-declared recipe name, rewrapping the resolver's
  /// `unknownSampler` / `unknownSigmaSchedule` so the 400 names the PRESET
  /// and the field, not just the string.
  private static func presetName<T>(
    _ presetId: String, field: String, _ resolve: () throws -> T
  ) throws -> T {
    do {
      return try resolve()
    } catch let error as WarmServerError {
      throw WarmServerError.presetRecipeInvalid(
        preset: presetId, field: field,
        reason: error.errorDescription ?? "\(error)")
    }
  }

  /// Pre-#286 behaviour, announced: the preset stays the provenance label it
  /// always was, nothing it declared is applied, and the response carries
  /// `preset_unresolved` + `preset_unresolved_reason`. Never a 400 — an
  /// unexpandable preset was harmless for the daemon's whole life.
  func asUnresolvedPreset(
    _ presetId: String, _ reason: PresetExpansion.Unresolved, log: (String) -> Void
  ) -> GeneratePayload {
    var out = self
    out.presetUnresolved = presetId
    out.presetUnresolvedReason = reason.code
    log("WARNING: \(reason.message) [\(reason.code)] — rendering with the request's own settings "
      + "and the resident LoRA stack, exactly as before #286; response carries preset_unresolved")
    return out
  }
}
