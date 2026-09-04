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
      return .apply(PresetExpansion(
        presetId: id, unresolved: "preset '\(id)' was not resolved before dispatch"))
    }

    let resolved: ResolvedPreset
    let declared: ImagePreset
    switch lookup {
    case .notFound:
      return .apply(PresetExpansion(
        presetId: id,
        unresolved: "unknown preset '\(id)' — not in this engine's store (GET /v1/presets)"))
    case .invalid(let reason):
      return .apply(PresetExpansion(
        presetId: id, unresolved: "preset '\(id)' is flagged invalid: \(reason)"))
    case .resolved(let r, let d):
      resolved = r
      declared = d
    }

    // --- Cases the engine cannot reproduce: label-only, and say why. --------

    // A video preset on the image path would push LTX adapters at a Krea 2
    // pipeline.
    if resolved.mediaKind.lowercased() == "video" {
      return .apply(PresetExpansion(
        presetId: id,
        unresolved: "preset '\(id)' is a video preset (media_kind \"\(resolved.mediaKind)\") "
          + "— /v1/generate is the image path"))
    }
    // The bypass `.diff` adapter is a preset-schema dial the engine has no
    // application path for (the expanding sender compiles it into `loras[]`).
    if let bypass = resolved.bypass, bypass.isActive {
      return .apply(PresetExpansion(
        presetId: id,
        unresolved: "preset '\(id)' declares bypass.strength \(bypass.strength), which the engine "
          + "cannot expand — send the resolved stack in `loras`"))
    }
    // D14: kroma is a first-class field, and the engine has no
    // family→default-file table (that policy is client-side, FDD §3.17), so an
    // unnamed file is not guessed.
    if let kroma = resolved.kroma, kroma.strength > 0,
       (kroma.file ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
      return .apply(PresetExpansion(
        presetId: id,
        unresolved: "preset '\(id)' declares kroma.strength \(kroma.strength) with no kroma.file, "
          + "and the engine has no family-default kroma table"))
    }

    var expansion = PresetExpansion(presetId: id)

    // --- Model: the half round 1 dropped. ----------------------------------

    let presetModel = (resolved.model ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
    if !presetModel.isEmpty {
      let asked = (requestModel ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
      if asked.isEmpty {
        // Flows through the request's existing model-switch semantics exactly
        // as if the client had sent it.
        expansion.model = presetModel
      } else if normalizeModelSpec(asked) != normalizeModelSpec(presetModel) {
        return .modelConflict(preset: id, presetModel: presetModel, requestModel: asked)
      }
    }

    // --- The stack ---------------------------------------------------------

    var stack: [LoraReference] = []
    if let kroma = resolved.kroma, kroma.strength > 0, let file = kroma.file {
      // D14: the expanding sender PREPENDS the structured kroma.
      stack.append(LoraReference(filename: file, scale: kroma.strength, role: "kroma"))
    }
    stack.append(contentsOf: resolved.loras)

    if let requestLoras {
      // Explicit `loras` keep their precedence — but a disagreement is
      // reported rather than absorbed. The production async client sends BOTH
      // `preset` and a FLAT `loras` list that has already dropped the
      // structured kroma/bypass/role, so this is the flag that makes that
      // visible from the response.
      expansion.stackMismatch = !isSameStack(requestLoras, stack)
    } else {
      expansion.loras = stack
    }

    // --- Declared steps/guidance, only where the request said nothing. -----

    if requestSteps == nil, let steps = declared.steps { expansion.steps = steps }
    if requestGuidance == nil, let guidance = declared.guidance { expansion.guidance = guidance }

    return .apply(expansion)
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
  /// always was, and this reaches the response as `preset_unresolved`.
  public var unresolved: String?
  /// I1: the request's explicit `loras` differ from the preset's resolved
  /// stack. Explicit still wins; this reaches the response as
  /// `preset_stack_mismatch`.
  public var stackMismatch: Bool

  public init(
    presetId: String, loras: [LoraReference]? = nil, model: String? = nil,
    steps: Int? = nil, guidance: Double? = nil, unresolved: String? = nil,
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
        // C2: NOT a 400. Pre-#286 behaviour (label only) plus a warning and an
        // additive response field, so nobody has to guess again.
        out.presetUnresolved = expansion.presetId
        log("WARNING: \(reason) — rendering with the request's own settings and the resident "
          + "LoRA stack, exactly as before #286; response carries preset_unresolved")
        return out
      }
      if let loras = expansion.loras {
        out.loras = loras.map { LoRAEntry(path: $0.filename, scale: Float($0.scale), role: $0.role) }
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
}
