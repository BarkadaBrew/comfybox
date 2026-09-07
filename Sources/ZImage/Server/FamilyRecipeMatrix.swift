// FamilyRecipeMatrix.swift — which family can actually RUN which sampler and
// which sigma schedule (K-FIX-1, Codex engine review I5).
//
// WP-E4 made NAME resolution fail-loud, and correctly did it at the decoder,
// family-agnostically: a name that is not a `SchedulerKind` / `SigmaScheduleKind`
// raw value or a declared alias is wrong for every family (D18's table).
//
// The other half was missing. A KNOWN name reaching a family whose loop does
// not implement it was silently ignored, which is the same lost-week failure
// WP-E4 exists to prevent — only harder to see, because the request is
// well-formed and the render succeeds:
//
//   * Chroma has native `.heun` and `.beta` (`ChromaSchedulerType`, driven by
//     `ChromaPipeline.generate(schedulerType:)`) and `runChromaGenerate`
//     forwarded NEITHER, so `scheduler: "heun"` rendered Euler pixels under
//     the name "heun" and `sigma_schedule: "beta"` was inert.
//   * Flux 2 and FIBO run FIXED flow-match Euler loops (`Flux2Pipeline`,
//     `FiboPipeline`) with no sampler seam at all, and accepted `heun`,
//     `res_2s`, `karras`, `beta`… all rendering Euler.
//   * `GeneratePayload.validateTableauSampler` refused N-row tableaus outside
//     Krea 2 — one true row of this table, maintained as a sibling gate.
//
// So: ONE table at the dispatch point. An explicitly named sampler/schedule
// either maps to the family's real implementation or is HTTP 400 naming the
// value and the family. The advertised `/object_info` option lists and the
// table-driven rejection test are generated from this same table, so a new
// `SchedulerKind` or a new family has to declare itself rather than
// inheriting a silent pass.
//
// ABSENT stays absent: a request naming neither field is accepted everywhere
// and each family applies its own default — that is the path every existing
// caller takes and it is untouched.

import Foundation

/// What one family's denoise loop can actually honour.
struct FamilyRecipeCapability: Sendable {
  /// Samplers the family's loop really dispatches.
  let samplers: Set<SchedulerKind>
  /// Sigma schedules the family really builds its grid from.
  let sigmaSchedules: Set<SigmaScheduleKind>
  /// Pairs the family cannot express even though both halves are supported
  /// (Chroma's `beta` implies heun stepping — there is no euler+beta).
  /// Returns `false` for a combination that must be refused.
  let pairIsSupported: @Sendable (SchedulerKind, SigmaScheduleKind) -> Bool
  /// Why a rejected pair is rejected — appended to the 400.
  let pairRejectionReason: String
  /// #399: does this family's save path run the ``StylePack`` post-process?
  ///
  /// The pass lives in `runKrea2Generate` and nowhere else, so on every other
  /// family a `style` (or `phone_look`) would be accepted, reported, and then
  /// silently skipped — the exact class of silence this whole table exists to
  /// end. A family that grows the pass flips this flag in the same commit.
  let appliesStylePack: Bool

  init(
    samplers: Set<SchedulerKind>,
    sigmaSchedules: Set<SigmaScheduleKind>,
    pairIsSupported: @escaping @Sendable (SchedulerKind, SigmaScheduleKind) -> Bool = { _, _ in true },
    pairRejectionReason: String = "",
    appliesStylePack: Bool = false
  ) {
    self.samplers = samplers
    self.sigmaSchedules = sigmaSchedules
    self.pairIsSupported = pairIsSupported
    self.pairRejectionReason = pairRejectionReason
    self.appliesStylePack = appliesStylePack
  }

  func isPairSupported(sampler: SchedulerKind, schedule: SigmaScheduleKind) -> Bool {
    pairIsSupported(sampler, schedule)
  }
}

enum FamilyRecipeMatrix {

  // MARK: - The table

  /// Z-Image drives `SchedulerFactory` directly, so it honours every
  /// non-tableau sampler and every model-independent schedule. `krea2` and
  /// `bong_tangent` are Krea 2's own (the first needs Krea 2's `mu` and
  /// throws `missingMu` here today; the second is D6's model-free warp), so
  /// they are refused by name rather than by a factory throw.
  private static let flux1 = FamilyRecipeCapability(
    samplers: Set(SchedulerKind.allCases.filter { !$0.isNRowTableau }),
    sigmaSchedules: [.flow, .karras, .exponential, .beta, .beta57])

  /// Krea 2 is the only loop that dispatches N-row tableaus (WP-E13), and the
  /// only one that builds the `krea2` and `bong_tangent` grids.
  private static let krea2 = FamilyRecipeCapability(
    samplers: Set(SchedulerKind.allCases),
    sigmaSchedules: Set(SigmaScheduleKind.allCases),
    // #399: the StylePack post-process pass is in `runKrea2Generate` only.
    appliesStylePack: true)

  /// Chroma's three native modes collapse a sampler and a schedule into one
  /// `ChromaSchedulerType`: euler = flow grid + euler step, heun = flow grid +
  /// heun step, beta = beta grid + heun step. There is no euler+beta.
  private static let chroma = FamilyRecipeCapability(
    samplers: [.euler, .heun],
    sigmaSchedules: [.flow, .beta],
    pairIsSupported: { sampler, schedule in
      chromaSchedulerType(sampler: sampler, schedule: schedule) != nil
    },
    pairRejectionReason:
      "Chroma's 'beta' schedule is beta-distributed timesteps stepped with heun "
      + "(ChromaSchedulerType.beta) — ask for sampler 'heun' with it, or drop the schedule")

  /// Flux 2 and FIBO run a fixed flow-match Euler loop with no sampler seam.
  /// `euler` + `flow` name exactly what runs (and are what the ComfyUI bridge
  /// forwards on every render), so they pass; anything else would have been
  /// Euler under another name.
  private static let fixedEulerLoop = FamilyRecipeCapability(
    samplers: [.euler], sigmaSchedules: [.flow])

  static func capability(for family: WarmModelFamily) -> FamilyRecipeCapability {
    // Exhaustive on purpose (no `default`): a new family must declare itself.
    switch family {
    case .flux1: return flux1
    case .krea2: return krea2
    case .chroma: return chroma
    case .flux2, .fibo: return fixedEulerLoop
    }
  }

  // MARK: - The gate

  /// `nil` when the family can run what was explicitly asked for; otherwise
  /// the 400 to return, naming the value and the family.
  ///
  /// Only EXPLICIT fields are gated — `names.scheduler`/`names.sigmaSchedule`
  /// are `nil` when the caller sent nothing, and each family's own default
  /// then applies untouched.
  /// #399 — `nil` when this family's save path runs the ``StylePack`` pass or
  /// the request asked for no look; otherwise the 400 to return.
  ///
  /// `style` here is the RESOLVED name (`WarmServer.decodedGeneratePayload`
  /// has already collapsed `phone_look` and the preset's declaration onto it
  /// and refused an unknown one), so this gate answers exactly one question:
  /// can the family that is about to render actually apply it?
  static func validateStyle(_ style: String?, family: WarmModelFamily) -> WarmServerError? {
    guard let style, !style.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
          !capability(for: family).appliesStylePack
    else { return nil }
    return .unsupportedRecipeField(
      field: "style", value: style, family: family.rawValue,
      reason: "engine-applied style packs are a Krea 2 post-process (#399) — this family's save "
        + "path has no such pass, so the look would have been accepted and then silently skipped. "
        + "Supported here: " + stylePackFamilyNames().joined(separator: ", ")
        + ". Remove `style` / `phone_look`, or render on one of those")
  }

  /// Families whose save path runs the pass — named in the 400 above and in
  /// `docs/PARITY.md`, from this one table rather than a hand-kept list.
  static func stylePackFamilyNames() -> [String] {
    WarmModelFamily.allCases
      .filter { capability(for: $0).appliesStylePack }
      .map(\.rawValue)
      .sorted()
  }

  static func validate(
    _ names: ResolvedRecipeNames, family: WarmModelFamily
  ) -> WarmServerError? {
    let capability = capability(for: family)

    if let sampler = names.scheduler, !capability.samplers.contains(sampler) {
      return .unsupportedSampler(
        name: names.schedulerRequested ?? sampler.rawValue,
        family: family.rawValue,
        reason: reason(forSampler: sampler, family: family, capability: capability))
    }

    if let schedule = names.sigmaSchedule, !capability.sigmaSchedules.contains(schedule) {
      return .unsupportedRecipeField(
        field: "sigma_schedule",
        value: names.sigmaScheduleRequested ?? schedule.rawValue,
        family: family.rawValue,
        reason: "this family does not build that grid — it would have been ignored. "
          + "Supported here: " + supportedSigmaScheduleNames(for: family).joined(separator: ", "))
    }

    // Both halves are individually supported: is the COMBINATION?
    let sampler = names.scheduler ?? defaultSampler(for: family)
    let schedule = names.sigmaSchedule ?? defaultSchedule(for: family)
    if (names.scheduler != nil || names.sigmaSchedule != nil),
       !capability.isPairSupported(sampler: sampler, schedule: schedule) {
      return .unsupportedRecipeField(
        field: "sigma_schedule",
        value: names.sigmaScheduleRequested ?? schedule.rawValue,
        family: family.rawValue,
        reason: "not supported with sampler '\(sampler.rawValue)'. "
          + capability.pairRejectionReason)
    }

    return nil
  }

  /// The samplers a family runs, as ENUM raw values in enum order — what a
  /// refusal message offers as the alternative. (`advertisedSamplerNames` is
  /// the ComfyUI spelling for `/object_info`; an error message names the
  /// engine's own values, as WP-E13's pinned wording does.)
  static func supportedSamplerNames(for family: WarmModelFamily) -> [String] {
    let allowed = capability(for: family).samplers
    return SchedulerKind.allCases.filter { allowed.contains($0) }.map(\.rawValue)
  }

  /// The same for sigma schedules.
  static func supportedSigmaScheduleNames(for family: WarmModelFamily) -> [String] {
    let allowed = capability(for: family).sigmaSchedules
    return SigmaScheduleKind.allCases.filter { allowed.contains($0) }.map(\.rawValue)
  }

  private static func reason(
    forSampler sampler: SchedulerKind, family: WarmModelFamily,
    capability: FamilyRecipeCapability
  ) -> String {
    let supported = supportedSamplerNames(for: family).joined(separator: ", ")
    if sampler.isNRowTableau {
      // WP-E13's rule, now a row of this table rather than a sibling gate.
      return "it is an N-row tableau sampler (\(sampler.rawValue)) dispatched only by the "
        + "Krea 2 denoise loop; this family takes one model evaluation per step and would "
        + "render first-order Euler under that name. Load a krea2 model, or use one of: "
        + supported
    }
    return "this family's denoise loop does not implement it — it would have rendered "
      + "under another sampler's behaviour. Supported here: " + supported
  }

  /// What the family runs when the caller names no sampler. Used only to
  /// evaluate a pair constraint; it never becomes the request's value.
  private static func defaultSampler(for family: WarmModelFamily) -> SchedulerKind { .euler }

  /// Likewise for the schedule. Krea 2's own default is its native warp.
  private static func defaultSchedule(for family: WarmModelFamily) -> SigmaScheduleKind {
    family == .krea2 ? .krea2 : .flow
  }

  // MARK: - Chroma's mapping

  /// Chroma's sampler+schedule pair → the native `ChromaSchedulerType` that
  /// implements it, or `nil` when the pair has no honest mapping (the caller
  /// turns that into the 400 `validate` produces for the same pair).
  ///
  /// `nil` inputs mean the caller named nothing and gets Chroma's default.
  static func chromaSchedulerType(
    sampler: SchedulerKind?, schedule: SigmaScheduleKind?
  ) -> ChromaSchedulerType? {
    switch (sampler ?? .euler, schedule ?? .flow) {
    case (.euler, .flow): return .euler
    case (.heun, .flow): return .heun
    case (.heun, .beta): return .beta
    default: return nil
    }
  }

  // MARK: - Advertised options (generated from the table)

  /// The ComfyUI-spelled sampler names a family implements, in enum order.
  /// `family: nil` is the UNION — what `/object_info` advertises, because it
  /// is served before a model is chosen.
  static func advertisedSamplerNames(for family: WarmModelFamily?) -> [String] {
    let allowed: Set<SchedulerKind> = family.map { capability(for: $0).samplers }
      ?? Set(SchedulerKind.allCases)
    return RecipeNameResolver.advertisedSamplerNames.filter { name in
      guard let kind = try? RecipeNameResolver.resolveSchedulerKind(name) else { return false }
      return allowed.contains(kind)
    }
  }

  /// The same for sigma schedules, aliases included (a family that runs
  /// `.flow` advertises Krita's `normal`/`simple`/… spellings for it).
  static func advertisedSigmaScheduleNames(for family: WarmModelFamily?) -> [String] {
    let allowed: Set<SigmaScheduleKind> = family.map { capability(for: $0).sigmaSchedules }
      ?? Set(SigmaScheduleKind.allCases)
    return RecipeNameResolver.advertisedSigmaScheduleNames.filter { name in
      guard let kind = try? RecipeNameResolver.resolveSigmaScheduleKind(name) else { return false }
      return allowed.contains(kind)
    }
  }
}

// MARK: - Public sampling-option surface

/// Public, UI-safe access to the same sampler/sigma-schedule capability table
/// the warm server validates at render time.
///
/// Keeping this facade beside ``FamilyRecipeMatrix`` prevents desktop clients
/// from maintaining a second option list that can advertise a recipe the
/// active model family will later reject. Strings are used at the boundary so
/// callers do not need access to the server's internal ``WarmModelFamily``.
public enum SamplingRecipeCatalog {
  /// Canonical warm-server family name inferred from either `/health`'s
  /// `model_family` value or a model id/path shown in the desktop UI.
  public static func canonicalFamily(_ raw: String?) -> String? {
    guard let raw else { return nil }
    let value = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard !value.isEmpty else { return nil }

    if value == "krea2" || value.contains("krea-2") || value.contains("krea2") {
      return WarmModelFamily.krea2.rawValue
    }
    if value == "flux2" || value.contains("flux-2") || value.contains("flux.2") {
      return WarmModelFamily.flux2.rawValue
    }
    if value == "fibo" || value.contains("fibo") {
      return WarmModelFamily.fibo.rawValue
    }
    if value == "chroma" || value.contains("chroma") {
      return WarmModelFamily.chroma.rawValue
    }
    if value == "flux1" || value.contains("z-image") || value.contains("zimage")
      || value.contains("flux")
    {
      return WarmModelFamily.flux1.rawValue
    }
    return nil
  }

  /// Supported sampler wire names, in the engine enum's stable display order.
  /// Unknown/absent family returns the union so a disconnected preset remains
  /// editable; the server still performs authoritative validation on save.
  public static func samplerNames(forModelFamily raw: String?) -> [String] {
    guard let family = family(raw) else { return SchedulerKind.allCases.map(\.rawValue) }
    return FamilyRecipeMatrix.supportedSamplerNames(for: family)
  }

  /// Supported sigma-schedule wire names. When a sampler is supplied, pair
  /// constraints are applied too (notably Chroma's heun+beta requirement).
  public static func sigmaScheduleNames(
    forModelFamily raw: String?, sampler: String? = nil
  ) -> [String] {
    guard let family = family(raw) else { return SigmaScheduleKind.allCases.map(\.rawValue) }
    let capability = FamilyRecipeMatrix.capability(for: family)
    let samplerName = normalized(sampler)
    let samplerKind = samplerName.flatMap { try? RecipeNameResolver.resolveSchedulerKind($0) }
      ?? (samplerName == nil ? SchedulerKind.euler : nil)
    return SigmaScheduleKind.allCases.filter { schedule in
      guard capability.sigmaSchedules.contains(schedule) else { return false }
      guard let samplerKind else { return true }
      return capability.isPairSupported(sampler: samplerKind, schedule: schedule)
    }.map(\.rawValue)
  }

  /// Whether the named values are known and runnable together on the family.
  /// Empty/nil values mean "model default" and are intentionally accepted.
  public static func supports(
    sampler: String?, sigmaSchedule: String?, forModelFamily raw: String?
  ) -> Bool {
    let sampler = normalized(sampler)
    let schedule = normalized(sigmaSchedule)
    let resolvedSampler = sampler.flatMap { try? RecipeNameResolver.resolveSchedulerKind($0) }
    let resolvedSchedule = schedule.flatMap { try? RecipeNameResolver.resolveSigmaScheduleKind($0) }
    if sampler != nil, resolvedSampler == nil { return false }
    if schedule != nil, resolvedSchedule == nil { return false }

    guard let family = family(raw) else { return true }
    let names = ResolvedRecipeNames(
      scheduler: resolvedSampler, schedulerRequested: sampler,
      sigmaSchedule: resolvedSchedule, sigmaScheduleRequested: schedule)
    return FamilyRecipeMatrix.validate(names, family: family) == nil
  }

  public static func defaultSamplerName(forModelFamily _: String?) -> String {
    SchedulerKind.euler.rawValue
  }

  public static func defaultSigmaScheduleName(forModelFamily raw: String?) -> String {
    family(raw) == .krea2 ? SigmaScheduleKind.krea2.rawValue : SigmaScheduleKind.flow.rawValue
  }

  // MARK: - Family-aware field gates (#419)
  //
  // The desktop preset editor and the Generate panel enable/disable/label
  // the RES4LYF and stage-2 knobs from THESE answers, so the gating logic
  // lives beside the capability matrix the warm server validates against,
  // not in a second table inside the UI. Each answer names the engine gate
  // it mirrors; the tests pin every one of them to that gate's actual
  // behaviour. An UNKNOWN family (nil / unrecognised / the engine has not
  // answered yet) is `.honoured` everywhere — the same "editable while
  // disconnected" rule `samplerNames(forModelFamily:)` follows — and the
  // engine's own validation remains the authority on save.

  /// What a family does with a recipe field: reads it, accepts-and-ignores
  /// it (no gate, no effect — the UI labels it), or 400s on it (the UI
  /// greys the control, offers Clear, and blocks Save).
  public enum RecipeFieldStatus: Equatable, Sendable {
    case honoured
    case inert(String)
    case refused(String)

    public var isRefused: Bool { if case .refused = self { return true }; return false }
    public var isHonoured: Bool { self == .honoured }
    /// The label / message text, nil when honoured.
    public var note: String? {
      switch self {
      case .honoured: return nil
      case .inert(let why), .refused(let why): return why
      }
    }
  }

  /// The RES4LYF ports, as the engine's own refusal lists them.
  public static let res4lyfSamplerList =
    "res_2s / res_3s / ralston_2s / ralston_3s / ralston_4s / heun_2s / heun_3s / deis_2m / deis_3m / deis_4m"

  /// Whether the named sampler (raw value, declared alias, or RES4LYF
  /// `exponential/`-style prefixed spelling) is one of the RES4LYF ports.
  /// `nil`/empty (model default = euler) and unknown names answer false.
  public static func isRES4LYFSampler(_ name: String?) -> Bool {
    resolvedSampler(name)?.isRES4LYFFamily ?? false
  }

  /// `eta` — `WarmServer.validateKrea2TierGates` (run under `family ==
  /// .krea2` only): on Krea 2 it is RES4LYF's SDE, refused on any other
  /// sampler. On Z-Image there is no gate: the value is forwarded and read
  /// by DDIM / DPM++ 2S-A (`SchedulerKind.readsAncestralEta`) and ignored by
  /// every other sampler. The remaining families forward nothing that reads it.
  public static func etaStatus(sampler: String?, forModelFamily raw: String?) -> RecipeFieldStatus {
    guard let family = family(raw) else { return .honoured }
    let kind = resolvedSampler(sampler) ?? .euler
    switch family {
    case .krea2:
      return kind.isRES4LYFFamily
        ? .honoured
        : .refused("eta is RES4LYF's SDE (parity tier T2) and applies to the RES4LYF samplers only; "
          + "'\(kind.rawValue)' is not one of them. Send eta 0, or a sampler from " + res4lyfSamplerList)
    case .flux1:
      return kind.readsAncestralEta
        ? .honoured
        : .inert("eta is not read by '\(kind.rawValue)' on Z-Image (only DDIM and DPM++ 2S Ancestral use it); it is accepted and ignored")
    case .chroma, .flux2, .fibo:
      return .inert("eta is not read by \(family.rawValue); it is accepted and ignored")
    }
  }

  /// `bongmath` — the eta arm's twin in `validateKrea2TierGates`: Krea 2 +
  /// RES4LYF sampler only. No other family has a gate or a reader for it.
  public static func bongmathStatus(sampler: String?, forModelFamily raw: String?) -> RecipeFieldStatus {
    guard let family = family(raw) else { return .honoured }
    let kind = resolvedSampler(sampler) ?? .euler
    switch family {
    case .krea2:
      return kind.isRES4LYFFamily
        ? .honoured
        : .refused("bongmath is RES4LYF's fixed point (parity tier T3) over its own tableau rows "
          + "and applies to the RES4LYF samplers only; '\(kind.rawValue)' is not one of them. "
          + "Send bongmath false, or a sampler from " + res4lyfSamplerList)
    case .flux1, .chroma, .flux2, .fibo:
      return .inert("bongmath is a Krea 2 RES4LYF setting; \(family.rawValue) accepts and ignores it")
    }
  }

  /// The RES4LYF spatial-noise / implicit-RK / substep recipe (`noise_type`,
  /// `noise_alpha`, `implicit_steps`, `c2`) and the projector-scale gain:
  /// decoded and range-checked on every family, read by the Krea 2 loop only
  /// — no family gate exists, so elsewhere they are inert, never refused.
  public static func noiseRecipeStatus(forModelFamily raw: String?) -> RecipeFieldStatus {
    guard let family = family(raw) else { return .honoured }
    return family == .krea2
      ? .honoured
      : .inert("Noise, implicit steps, C2 and projector scale are Krea 2 settings; \(family.rawValue) accepts and ignores them")
  }

  /// `stage2` — `GeneratePayload.stage2Gate`: a Krea 2 mechanism (WP-E17),
  /// refused by name on every other family.
  public static func stage2Status(forModelFamily raw: String?) -> RecipeFieldStatus {
    guard let family = family(raw) else { return .honoured }
    return family == .krea2
      ? .honoured
      : .refused("a second stage inside one render is a Krea 2 mechanism (WP-E17); "
        + "\(family.rawValue) has no such seam — load a krea2 model, or remove stage2")
  }

  /// `vae` — `GeneratePayload.vaeGate` (WP-E9): a Krea 2 request field,
  /// refused by name on every other family.
  public static func vaeStatus(forModelFamily raw: String?) -> RecipeFieldStatus {
    guard let family = family(raw) else { return .honoured }
    return family == .krea2
      ? .honoured
      : .refused("VAE selection is a Krea 2 request field (WP-E9); \(family.rawValue) decodes through its own VAE and does not honour it — remove it")
  }

  /// `shift` — `GeneratePayload.validateShift` (krea2 reads it as `mu`,
  /// flux1 as a linear warp, the fixed-schedule families refuse it) and, on
  /// flux1, `validateShiftSchedule` (comfybox#154): the `krea2` and
  /// `bong_tangent` grids are defined by mu / by construction and would drop
  /// it, so the pairing is a 400. `nil`/empty schedule = the family default.
  public static func shiftStatus(sigmaSchedule: String?, forModelFamily raw: String?) -> RecipeFieldStatus {
    guard let family = family(raw) else { return .honoured }
    switch family {
    case .krea2:
      return .honoured
    case .flux1:
      guard let name = normalized(sigmaSchedule),
        let schedule = try? RecipeNameResolver.resolveSigmaScheduleKind(name)
      else { return .honoured }
      return SchedulerFactory.honoursExplicitShift(schedule: schedule, modelSampling: .discreteFlow)
        ? .honoured
        : .refused("shift is not read by sigma schedule '\(schedule.rawValue)' (it is defined by mu / by index "
          + "arithmetic, not by the model's shift) — drop shift, or ask for a schedule that honours it "
          + "(flow/normal, simple, beta, beta57, karras, exponential)")
    case .chroma, .flux2, .fibo:
      return .refused("shift is a schedule field for the krea2 and flux1 (Z-Image) families and is not "
        + "honoured by model family '\(family.rawValue)'; remove it")
    }
  }

  /// The label the UI shows for `shift`, naming what the number MEANS on
  /// this family: Krea 2 reads it as `mu` (a log-shift feeding
  /// `ModelSamplingFlux`), Z-Image as the linear `σ' = shift·σ / (1 + (shift−1)·σ)`.
  public static func shiftLabel(forModelFamily raw: String?) -> String {
    switch family(raw) {
    case .krea2: return "Shift (mu)"
    case .flux1: return "Shift (linear)"
    default: return "Shift"
    }
  }

  private static func resolvedSampler(_ name: String?) -> SchedulerKind? {
    guard let name = normalized(name) else { return nil }
    return try? RecipeNameResolver.resolveSchedulerKind(name)
  }

  private static func normalized(_ value: String?) -> String? {
    guard let value else { return nil }
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
  }

  private static func family(_ raw: String?) -> WarmModelFamily? {
    canonicalFamily(raw).flatMap(WarmModelFamily.init(rawValue:))
  }
}
