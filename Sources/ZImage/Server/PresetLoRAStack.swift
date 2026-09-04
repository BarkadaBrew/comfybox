import Foundation

/// #286 — "which LoRA stack must be resident for THIS `/v1/generate` request?"
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
/// This is the decision that closes that hole, kept pure so it can be tested
/// without a pipeline: given the request's own `loras`, the named preset, and
/// what the preset store resolves that name to (the SAME `PresetStore.resolve`
/// that backs `POST /v1/presets/resolve` — one code path, not a parallel one),
/// it answers with the stack to apply, or a refusal.
///
/// The rule, in one line: **a render either runs with exactly the stack
/// `/v1/presets/resolve` reports for its preset, or it fails loud.** It never
/// runs on residency it did not ask for.
public enum PresetLoRAStack: Sendable, Equatable {

  /// No `preset` and no `loras` — the caller said nothing about adapters, so
  /// nothing changes. Byte-identical to pre-#286 behaviour (a bare
  /// `/v1/generate` still renders on the resident stack, which is what
  /// `/v1/lora/swap` + generate clients rely on).
  case unchanged

  /// The request carried its own `loras` array. It wins outright — `preset`
  /// stays the provenance label it always was, and an unknown/foreign preset
  /// id is harmless because nothing is being read from it. This preserves the
  /// existing "explicit loras always work" behaviour the issue confirmed.
  case requestExplicit

  /// Apply exactly these, in this order, before the render.
  case apply(presetId: String, loras: [LoraReference])

  /// The request named a preset whose resolved stack the engine cannot
  /// reproduce. A 400 — never a render on the wrong stack.
  case refuse(String)

  /// What the preset store had to say about the requested name. Mirrors the
  /// three outcomes of ``PresetStore/resolve(_:)``.
  public enum Lookup: Sendable, Equatable {
    case resolved(ResolvedPreset)
    /// No preset with that id (`PresetStoreError.notFound`).
    case notFound
    /// Flagged invalid at load/save (`PresetStoreError.invalid`, WP-E20 AC-44c).
    case invalid(reason: String)
  }

  /// Decide the stack.
  ///
  /// - Parameters:
  ///   - requestHasLoras: whether the request body carried a `loras` array at
  ///     all. An explicitly EMPTY array is still "explicit" — it means "no
  ///     adapters", which is a statement, not an absence.
  ///   - presetId: `payload.preset`, trimmed by the caller or not.
  ///   - lookup: nil when `presetId` is nil/empty; otherwise the store's answer.
  public static func decide(
    requestHasLoras: Bool,
    presetId: String?,
    lookup: Lookup?
  ) -> PresetLoRAStack {
    let id = presetId?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

    // Request `loras` win, exactly as before — the preset is only a label then.
    if requestHasLoras { return .requestExplicit }
    if id.isEmpty { return .unchanged }

    guard let lookup else {
      // Caller named a preset but did not look it up — a programming error
      // here would silently reopen the bug, so it refuses rather than shrugs.
      return .refuse(
        "Preset '\(id)' was not resolved before dispatch — refusing to render on the resident LoRA stack")
    }

    switch lookup {
    case .notFound:
      return .refuse(
        "Unknown preset '\(id)' — /v1/generate cannot resolve its LoRA stack, and rendering on "
          + "whatever stack is resident would silently produce the wrong look. "
          + "Send the LoRAs explicitly in `loras`, or use a preset from GET /v1/presets.")

    case .invalid(let reason):
      return .refuse(
        "Preset '\(id)' is flagged invalid (\(reason)) — /v1/generate refuses it for the same "
          + "reason /v1/presets/resolve does.")

    case .resolved(let resolved):
      // A video preset on the image path would push LTX adapters at a Krea 2
      // pipeline. Refuse by name rather than fail obscurely at bind time.
      if resolved.mediaKind.lowercased() == "video" {
        return .refuse(
          "Preset '\(id)' is a video preset (media_kind \"\(resolved.mediaKind)\") — "
            + "/v1/generate is the image path; use /v1/video/generate.")
      }

      // The bypass `.diff` adapter is a preset-schema dial the engine has no
      // application path for (it is compiled into `loras[]` by the sender that
      // expands the preset). Declared-on and unappliable ⇒ refuse; silently
      // dropping it is exactly the failure mode #286 is about.
      if let bypass = resolved.bypass, bypass.isActive {
        return .refuse(
          "Preset '\(id)' declares bypass.strength \(bypass.strength), which /v1/generate cannot "
            + "expand engine-side — send the resolved stack explicitly in `loras`.")
      }

      var stack: [LoraReference] = []
      if let kroma = resolved.kroma, kroma.strength > 0 {
        // D14: kroma is a first-class field, never a `loras[]` entry, and the
        // expanding sender PREPENDS it. The engine has no family→default-file
        // table (that policy lives client-side, FDD §3.17), so an unnamed file
        // is a refusal, not a guess.
        guard let file = kroma.file, !file.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
          return .refuse(
            "Preset '\(id)' declares kroma.strength \(kroma.strength) with no kroma.file — the "
              + "engine has no family-default kroma table, so it cannot reproduce the resolved "
              + "stack. Name the file on the preset, or send `loras` explicitly.")
        }
        stack.append(LoraReference(filename: file, scale: kroma.strength, role: "kroma"))
      }
      stack.append(contentsOf: resolved.loras)
      return .apply(presetId: id, loras: stack)
    }
  }

  /// One-line summary for the render log, so a wrong stack is visible in the
  /// engine's own output and not only in the response.
  public static func describe(_ loras: [LoraReference]) -> String {
    loras.isEmpty
      ? "(none)"
      : loras.map { "\($0.filename)@\(String(format: "%.4g", $0.scale))" }.joined(separator: ", ")
  }
}

// MARK: - The `/v1/generate` seam

extension GeneratePayload {

  /// #286 — return `payload` with its named `preset` expanded into `loras`, or
  /// throw.
  ///
  /// This is the ONE place a preset becomes a stack on the image path. It runs
  /// in `WarmServer.decodedGeneratePayload`, so `/v1/generate`,
  /// `/v1/generate/async` and persisted-queue replay all go through it, and the
  /// existing per-job application at dequeue (`applyActiveLoRAs`) does the
  /// actual loading — no second application path to drift out of sync.
  ///
  /// - Parameters:
  ///   - resolve: the preset store lookup. Injected so the seam is testable
  ///     without a warm pipeline; production passes `PresetStore.resolve`, the
  ///     same call `POST /v1/presets/resolve` makes.
  ///   - log: one line naming the preset and the stack it expanded to, so a
  ///     wrong stack is visible in the engine's own output.
  static func expandingPresetLoRAs(
    _ payload: GeneratePayload,
    resolve: (String) -> PresetLoRAStack.Lookup,
    log: (String) -> Void = { _ in }
  ) throws -> GeneratePayload {
    var out = payload
    let id = payload.preset?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    let lookup: PresetLoRAStack.Lookup? =
      (id.isEmpty || payload.loras != nil) ? nil : resolve(id)

    switch PresetLoRAStack.decide(
      requestHasLoras: payload.loras != nil, presetId: payload.preset, lookup: lookup
    ) {
    case .unchanged, .requestExplicit:
      return out
    case .refuse(let message):
      throw WarmServerError.invalidRequest(message: message)
    case .apply(let presetId, let loras):
      out.loras = loras.map {
        LoRAEntry(path: $0.filename, scale: Float($0.scale), role: $0.role)
      }
      log("Preset '\(presetId)': applying its resolved LoRA stack — \(PresetLoRAStack.describe(loras))")
      return out
    }
  }
}
