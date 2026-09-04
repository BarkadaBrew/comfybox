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
  /// FIBO and Chroma have no LoRA application path of their own — they sit on
  /// top of the Flux-1 `ZImagePipeline` instance, `POST /v1/lora/swap` refuses
  /// them outright (``WarmServerError/loraSwapNotSupported``) and
  /// `applied_loras` is deliberately ABSENT for them so it can never read as
  /// "rendered bare". Pushing a WARM DEFAULT at them would reach a pipeline
  /// they are not rendering through, which is a new behaviour, not a fix —
  /// so for them a warm default is skipped and the pre-#282 contract stands.
  /// A stack the job named explicitly is still applied exactly as it was
  /// before this ticket.
  public static func appliesAtDequeue(origin: Origin, familyHasLoRAPath: Bool) -> Bool {
    familyHasLoRAPath || origin != .warmDefault
  }
}

extension RequestStackResolver.Resolved: Equatable where Element: Equatable {}
extension RequestStackResolver.Resolved: Sendable where Element: Sendable {}
