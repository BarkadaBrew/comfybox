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

    return .apply(expansion)
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
    steps: Int? = nil, guidance: Double? = nil, unresolved: Unresolved? = nil,
    stackMismatch: Bool = false
  ) {
    self.presetId = presetId
    self.loras = loras
    self.model = model
    self.steps = steps
    self.guidance = guidance
    self.unresolved = unresolved
    self.stackMismatch = stackMismatch
  }
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
      return out
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
