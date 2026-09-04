import Foundation

/// #282 — which LoRA stack does THIS job render with?
///
/// ## The model this replaces
///
/// Before #282 the engine had one *resident* stack (`WarmServerCoordinator
/// .activeLoRAs`) and two ways to move it: `POST /v1/lora/swap`, and a job's
/// own explicit `loras` array applied at dequeue. A request that carried
/// neither rendered on **whatever was left there** — the previous job's
/// per-job override, an hours-old swap, or nothing at all right after a
/// restart. #286 (PR #350) closed the preset half of that by expanding a named
/// `preset` into `payload.loras` at decode, so a preset render stopped
/// inheriting residency. What remained was the swap-first model itself: a
/// shared mutable stack that later jobs silently inherit.
///
/// ## The model this establishes
///
/// **The unit of truth is the per-job resolved stack.** Every render resolves
/// its own stack, once, and applies exactly that at dequeue. There are three
/// sources, in strict precedence:
///
/// 1. ``Origin/request`` — the request's own `loras`. An explicitly EMPTY
///    array is a statement ("no adapters"), not an absence, and still wins.
/// 2. ``Origin/preset`` — the stack the named `preset` expanded to
///    (``PresetLoRAStack``, at decode). Also a statement when empty: the
///    seeded `zimage-chat` preset declares `loras: []` and means it.
/// 3. ``Origin/warmDefault`` — the WARM DEFAULT stack, which is the ONLY thing
///    `/v1/lora/swap` now sets, and which applies ONLY to a request that
///    carried neither `preset` nor `loras`.
///
/// `/v1/lora/swap` keeps its route, its payload and its response JSON — it is
/// production and the daemon contract is not this ticket's to break. What
/// changes is what it *means*: it publishes a default for bare requests
/// instead of mutating a stack that later jobs inherit. A job never inherits
/// it implicitly except through that default.
///
/// Pure, so the precedence can be tested without a pipeline, weights or a GPU.
public enum RequestStackResolver {

  /// Where this job's stack came from. The raw values are the wire spellings
  /// (`lora_stack_origin` on the generate response).
  public enum Origin: String, Sendable, Equatable, CaseIterable, Codable {
    /// The request's own `loras` array.
    case request = "request"
    /// The named `preset`, expanded at decode by ``PresetLoRAStack``.
    case preset = "preset"
    /// The warm default published by `POST /v1/lora/swap` (or the engine's
    /// launch-time `--lora` arguments before any swap).
    case warmDefault = "warm_default"
  }

  /// One job's stack, and where it came from.
  public struct Resolved<Element> {
    public let stack: [Element]
    public let origin: Origin
    public init(stack: [Element], origin: Origin) {
      self.stack = stack
      self.origin = origin
    }
  }

  /// The precedence decision on its own, from presence alone.
  ///
  /// Presence, not emptiness: `loras: []` and a preset expanding to `[]` are
  /// both statements that this job wants no adapters, and both must beat the
  /// warm default — otherwise "render bare" would be unexpressible and a
  /// caller asking for it would silently get the last swap.
  public static func origin(hasRequestLoras: Bool, hasPresetStack: Bool) -> Origin {
    if hasRequestLoras { return .request }
    if hasPresetStack { return .preset }
    return .warmDefault
  }

  /// Resolve one job's stack.
  ///
  /// - Parameters:
  ///   - requestLoras: the request's own `loras`. nil = the key was absent.
  ///   - presetStack: the stack the named `preset` expanded to. nil = no
  ///     preset, or one that stayed a label (``PresetExpansion/unresolved``).
  ///   - warmDefault: the warm default stack. Never nil — an engine with no
  ///     `--lora` arguments and no swap yet simply has an empty one, and an
  ///     empty warm default is a real answer ("render bare"), not a fallthrough.
  public static func resolve<Element>(
    requestLoras: [Element]?,
    presetStack: [Element]?,
    warmDefault: [Element]
  ) -> Resolved<Element> {
    switch origin(hasRequestLoras: requestLoras != nil, hasPresetStack: presetStack != nil) {
    case .request: return Resolved(stack: requestLoras ?? [], origin: .request)
    case .preset: return Resolved(stack: presetStack ?? [], origin: .preset)
    case .warmDefault: return Resolved(stack: warmDefault, origin: .warmDefault)
    }
  }

  /// Is the resolved stack APPLIED at dequeue for this family?
  ///
  /// **Review r1, M1 — the two answers are now the same one.** FIBO and Chroma
  /// have no LoRA application path of their own: `ChromaPipeline.generate`
  /// takes no adapters, `ModelPool` never forwards `initialLoRAs` to either
  /// family (only the krea2 and flux1 branches do), and `/v1/lora/swap` refuses
  /// them outright (``WarmServerError/loraSwapNotSupported``). They render
  /// with **no adapters, always**.
  ///
  /// The first cut of this function skipped only a warm default there, which
  /// left the inconsistency the review named: a request-NAMED stack still went
  /// to `applyActiveLoRAs`'s default arm and was loaded into the Flux-1
  /// `ZImagePipeline` — an instance those families are not rendering through.
  /// It changed nothing about the pixels, and it corrupted `activeLoRAs`,
  /// `/health.loras` and the PNG's `loras` list into naming adapters that had
  /// no part in the render.
  ///
  /// So: on a family with no LoRA path, **nothing** is applied, whatever the
  /// origin, and the render's own record says so (`applied_loras` stays absent
  /// for them, and the Chroma PNG now records the empty stack it actually
  /// rendered with). Nothing 4xx's that did not before — a caller sending
  /// `loras` to Chroma gets the same bare render it always got, now honestly
  /// reported and logged.
  public static func appliesAtDequeue(origin: Origin, familyHasLoRAPath: Bool) -> Bool {
    familyHasLoRAPath
  }

  // MARK: - The warm default is only valid for the base it was published under

  /// #282 review r1 (C1) — what a warm default was published against.
  ///
  /// `POST /v1/lora/swap` applies its stack to whatever base is resident and
  /// publishes it as the default. A bare request that arrives later may
  /// activate a DIFFERENT checkpoint at dequeue (its own `model`, or a preset's
  /// — the per-job model switch runs before the stack is applied). Force-loading
  /// a krea2-raw stack into, say, a Flux-2 pipeline is at best a stack of
  /// adapters that bind zero layers, and at worst a throw — turning a request
  /// that always rendered into a 500. So the default carries its provenance.
  public struct WarmDefaultTag: Sendable, Equatable {
    /// `WarmModelFamily.rawValue` at the moment of publication. nil = untagged.
    public let family: String?
    /// The active model spec at publication, already normalised (alias →
    /// directory, `~` expanded) so a spelling difference is not a mismatch.
    public let modelSpec: String?

    public init(family: String? = nil, modelSpec: String? = nil) {
      self.family = family
      self.modelSpec = modelSpec
    }

    /// The engine's launch-time `--lora` stack, which has no swap behind it.
    /// Admitted everywhere: it is what the operator declared for this process,
    /// and refusing it would change boot behaviour rather than protect anything.
    public static let untagged = WarmDefaultTag()
  }

  /// May this job take the warm default?
  public enum WarmDefaultAdmission: Sendable, Equatable {
    case admit
    /// Render with NO adapters, and say so. Never an error — the reason code
    /// reaches the response as `warm_default_skipped`.
    case skip(reason: String)

    /// The published reason codes.
    public static let familyMismatch = "family_mismatch"
    public static let modelMismatch = "model_mismatch"
  }

  /// Decide whether a warm default published under `tag` may be applied to a
  /// job that is rendering on `requestFamily` / `requestModelSpec`.
  ///
  /// - An EMPTY default is always admitted: "clear the adapters" is
  ///   base-agnostic and cannot throw.
  /// - An UNTAGGED default (the launch-time `--lora` stack) is always admitted.
  /// - A different family ⇒ ``WarmDefaultAdmission/familyMismatch``.
  /// - Same family, both model specs known and different ⇒
  ///   ``WarmDefaultAdmission/modelMismatch``. Two krea2 checkpoints are the
  ///   same family and still the wrong base for each other's adapters — that is
  ///   the silent-wrong-look defect #286 spent three review rounds on, and a
  ///   default nobody asked for is the last place to reintroduce it.
  /// - An unknown spec on either side is not a mismatch: the family agreed, and
  ///   refusing on ignorance would strand the ordinary case where the engine
  ///   never recorded a spec.
  public static func admitWarmDefault(
    isEmpty: Bool,
    tag: WarmDefaultTag,
    requestFamily: String,
    requestModelSpec: String?
  ) -> WarmDefaultAdmission {
    if isEmpty { return .admit }
    guard let tagFamily = tag.family else { return .admit }
    guard tagFamily == requestFamily else {
      return .skip(reason: WarmDefaultAdmission.familyMismatch)
    }
    guard let tagSpec = tag.modelSpec, let requestSpec = requestModelSpec,
          !tagSpec.isEmpty, !requestSpec.isEmpty
    else { return .admit }
    return tagSpec == requestSpec ? .admit : .skip(reason: WarmDefaultAdmission.modelMismatch)
  }
}

extension RequestStackResolver.Resolved: Equatable where Element: Equatable {}
extension RequestStackResolver.Resolved: Sendable where Element: Sendable {}
